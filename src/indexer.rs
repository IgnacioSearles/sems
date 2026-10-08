//! Brings the index in line with the files under a root, embedding only what changed.
//!
//! Change detection is layered so unchanged trees cost almost nothing: size and modification time
//! first (no file reads), then a content hash (no embedding when, say, a checkout only touched
//! timestamps). Changed chunks from many files are embedded together to keep batches full.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::chunking::{Chunk, ChunkingConfig, chunk_text};
use crate::discovery::{DiscoveredFile, decode_text, discover_files};
use crate::encoder::{Document, Encoder};
use crate::store::{FileRecord, IndexStore, PathScope};

/// Chunks gathered before calling the encoder; the encoder splits them into token-budget batches.
const CHUNKS_PER_ENCODE_CALL: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct IndexOptions {
    pub max_file_size: u64,
    pub chunking: ChunkingConfig,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self { max_file_size: 1024 * 1024, chunking: ChunkingConfig::default() }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IndexSummary {
    pub files_seen: usize,
    pub files_unchanged: usize,
    pub files_embedded: usize,
    pub files_removed: usize,
    pub files_skipped_binary: usize,
    pub files_skipped_too_large: usize,
    pub chunks_embedded: usize,
    pub elapsed: Duration,
}

/// Receives progress while embedding; lets the CLI render without the indexer knowing how.
pub trait IndexProgress {
    fn embedding_started(&mut self, files_to_embed: usize);
    fn files_embedded(&mut self, files_done: usize, chunks_done: usize);
}

/// Progress sink that ignores everything.
pub struct SilentProgress;

impl IndexProgress for SilentProgress {
    fn embedding_started(&mut self, _files_to_embed: usize) {}
    fn files_embedded(&mut self, _files_done: usize, _chunks_done: usize) {}
}

struct PendingFile {
    path: PathBuf,
    record: FileRecord,
    chunks: Vec<Chunk>,
}

pub fn index_directory(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    root: &Path,
    options: &IndexOptions,
    progress: &mut dyn IndexProgress,
) -> Result<IndexSummary> {
    let started = Instant::now();
    let scope = PathScope::new(root);
    let mut indexed = store.files_in_scope(&scope)?;
    let discovery = discover_files(root, options.max_file_size)?;
    let mut summary = IndexSummary {
        files_seen: discovery.files.len(),
        files_skipped_too_large: discovery.skipped_too_large,
        ..IndexSummary::default()
    };

    let mut pending = Vec::new();
    for file in &discovery.files {
        let previous = indexed.remove(&file.path);
        if previous.as_ref().is_some_and(|record| metadata_matches(record, file)) {
            summary.files_unchanged += 1;
            continue;
        }
        let bytes = std::fs::read(&file.path).with_context(|| format!("failed to read {}", file.path.display()))?;
        let record = FileRecord {
            size: file.size,
            modified_nanoseconds: file.modified_nanoseconds,
            content_hash: *blake3::hash(&bytes).as_bytes(),
        };
        if previous.as_ref().is_some_and(|previous| previous.content_hash == record.content_hash) {
            store.update_file_metadata(&file.path, &record)?;
            summary.files_unchanged += 1;
            continue;
        }
        let Some(text) = decode_text(&bytes) else {
            if previous.is_some() {
                store.remove_file(&file.path)?; // became binary since the last run
            }
            summary.files_skipped_binary += 1;
            continue;
        };
        pending.push(PendingFile { path: file.path.clone(), record, chunks: chunk_text(&text, &options.chunking) });
    }

    // Anything indexed under the root that discovery no longer reports was deleted or is now ignored.
    for path in indexed.into_keys() {
        store.remove_file(&path)?;
        summary.files_removed += 1;
    }

    progress.embedding_started(pending.len());
    embed_and_store(store, encoder, pending, progress, &mut summary)?;
    summary.elapsed = started.elapsed();
    Ok(summary)
}

fn metadata_matches(record: &FileRecord, file: &DiscoveredFile) -> bool {
    record.size == file.size && record.modified_nanoseconds == file.modified_nanoseconds
}

fn embed_and_store(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    pending: Vec<PendingFile>,
    progress: &mut dyn IndexProgress,
    summary: &mut IndexSummary,
) -> Result<()> {
    let mut group: Vec<PendingFile> = Vec::new();
    let mut group_chunks = 0;
    for file in pending {
        group_chunks += file.chunks.len();
        group.push(file);
        if group_chunks >= CHUNKS_PER_ENCODE_CALL {
            store_group(store, encoder, std::mem::take(&mut group), summary)?;
            group_chunks = 0;
            progress.files_embedded(summary.files_embedded, summary.chunks_embedded);
        }
    }
    if !group.is_empty() {
        store_group(store, encoder, group, summary)?;
        progress.files_embedded(summary.files_embedded, summary.chunks_embedded);
    }
    Ok(())
}

fn store_group(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    group: Vec<PendingFile>,
    summary: &mut IndexSummary,
) -> Result<()> {
    let titles: Vec<String> = group.iter().map(|file| file_title(&file.path)).collect();
    let documents: Vec<Document<'_>> = group
        .iter()
        .zip(&titles)
        .flat_map(|(file, title)| file.chunks.iter().map(move |chunk| Document { title, text: &chunk.text }))
        .collect();
    let mut embeddings = encoder.encode_documents(&documents)?.into_iter();

    for file in group {
        let chunks: Vec<(Chunk, Vec<f32>)> = file
            .chunks
            .into_iter()
            .map(|chunk| (chunk, embeddings.next().expect("encoder returns one embedding per document")))
            .collect();
        summary.chunks_embedded += chunks.len();
        summary.files_embedded += 1;
        store.replace_file(&file.path, &file.record, &chunks)?;
    }
    Ok(())
}

/// The model card recommends the file name as the document title for code retrieval.
fn file_title(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
}
