//! Brings the index in line with the files under a root, embedding only what changed.
//!
//! Change detection is layered so unchanged trees cost almost nothing: size and modification time
//! first (no file reads), then a content hash (no embedding when, say, a checkout only touched
//! timestamps). Changed text chunks from many files are embedded together to keep batches full;
//! images are decoded and embedded one at a time so a folder of large photos never sits in memory.
//!
//! Embedding runs in phases by encoder: text, then everything visual (photos, then video frames),
//! then everything audible (audio files and video soundtracks). A 4 GB GPU cannot hold the vision
//! and audio encoders at once, so this order loads each of them once per run.

mod recordings;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::av::Ffmpeg;
use crate::chunking::{Chunk, ChunkingConfig, chunk_text};
use crate::discovery::{DiscoveredFile, DiscoveryOptions, FileKind, decode_text, discover_files};
use crate::documents::extract_pdf_text;
use crate::encoder::{Document, Encoder};
use crate::media::{LoadedImage, load_image};
use crate::segments::MediaSegmentation;
use crate::store::{FileRecord, IndexStore, PathScope};
use recordings::{PendingRecording, embed_recordings};

/// Chunks gathered before calling the encoder; the encoder splits them into token-budget batches.
const CHUNKS_PER_ENCODE_CALL: usize = 64;

#[derive(Debug, Clone)]
pub struct IndexOptions {
    pub discovery: DiscoveryOptions,
    pub chunking: ChunkingConfig,
    pub segmentation: MediaSegmentation,
    /// Decodes audio and video. Without it (the default), recordings are counted but left out of
    /// the index, and a later run with ffmpeg indexes them.
    pub ffmpeg: Option<Ffmpeg>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            discovery: DiscoveryOptions {
                max_text_size: 1024 * 1024,
                max_media_size: 64 * 1024 * 1024,
                include_images: true,
                include_audio: true,
                include_video: true,
            },
            chunking: ChunkingConfig::default(),
            segmentation: MediaSegmentation::default(),
            ffmpeg: None,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IndexSummary {
    pub files_seen: usize,
    pub files_unchanged: usize,
    pub files_embedded: usize,
    pub images_embedded: usize,
    pub files_removed: usize,
    pub files_skipped_binary: usize,
    pub files_skipped_too_large: usize,
    pub images_skipped_too_small: usize,
    pub images_unreadable: usize,
    pub pdfs_without_text: usize,
    pub pdfs_unreadable: usize,
    /// Audio and video files embedded (also counted in `files_embedded`).
    pub recordings_embedded: usize,
    /// Recordings found while ffmpeg is unavailable; not tracked, so they are indexed once it is.
    pub recordings_without_ffmpeg: usize,
    pub recordings_unreadable: usize,
    /// Recordings with nothing to embed: silent, or too short.
    pub recordings_without_content: usize,
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

/// A text file or a PDF, chunked and waiting for embeddings. PDF chunks carry their page.
struct PendingText {
    path: PathBuf,
    record: FileRecord,
    kind: FileKind,
    chunks: Vec<PendingChunk>,
}

struct PendingChunk {
    page: Option<usize>,
    chunk: Chunk,
}

/// Images are only hashed while scanning; pixels are decoded when it is their turn to embed.
struct PendingImage {
    path: PathBuf,
    record: FileRecord,
}

#[derive(Default)]
struct Pending {
    texts: Vec<PendingText>,
    images: Vec<PendingImage>,
    recordings: Vec<PendingRecording>,
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
    let discovery = discover_files(root, &options.discovery)?;
    let mut summary = IndexSummary {
        files_seen: discovery.files.len(),
        files_skipped_too_large: discovery.skipped_too_large,
        ..IndexSummary::default()
    };

    let mut pending = Pending::default();
    for file in &discovery.files {
        let previous = indexed.remove(&file.path);
        scan_file(store, file, previous, options, &mut pending, &mut summary)?;
    }

    // Anything indexed under the root that discovery no longer reports was deleted or is now ignored.
    for path in indexed.into_keys() {
        store.remove_file(&path)?;
        summary.files_removed += 1;
    }

    progress.embedding_started(pending.texts.len() + pending.images.len() + pending.recordings.len());
    embed_texts(store, encoder, pending.texts, progress, &mut summary)?;
    embed_images(store, encoder, pending.images, progress, &mut summary)?;
    // Recordings are only queued when ffmpeg is available.
    if let Some(ffmpeg) = &options.ffmpeg {
        embed_recordings(store, encoder, ffmpeg, pending.recordings, &options.segmentation, progress, &mut summary)?;
    }
    summary.elapsed = started.elapsed();
    Ok(summary)
}

/// Queues `file` for embedding if it is new or its content changed since `previous`.
fn scan_file(
    store: &mut IndexStore,
    file: &DiscoveredFile,
    previous: Option<FileRecord>,
    options: &IndexOptions,
    pending: &mut Pending,
    summary: &mut IndexSummary,
) -> Result<()> {
    if previous.as_ref().is_some_and(|record| metadata_matches(record, file)) {
        summary.files_unchanged += 1;
        return Ok(());
    }
    if file.kind.is_recording() && options.ffmpeg.is_none() {
        summary.recordings_without_ffmpeg += 1;
        return Ok(());
    }
    let record = FileRecord {
        size: file.size,
        modified_nanoseconds: file.modified_nanoseconds,
        content_hash: content_hash(&file.path)?,
    };
    if previous.is_some_and(|previous| previous.content_hash == record.content_hash) {
        store.update_file_metadata(&file.path, &record)?;
        summary.files_unchanged += 1;
        return Ok(());
    }
    queue_changed_file(store, file, record, &options.chunking, pending, summary)
}

/// Streams the file through the hasher rather than reading it whole: recordings can be gigabytes.
fn content_hash(path: &Path) -> Result<[u8; 32]> {
    let file = std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(file).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(*hasher.finalize().as_bytes())
}

/// Text and PDFs are chunked right away (chunks are small); images and recordings wait for their
/// encoder's phase. Files with nothing to index are tracked as skipped, so unchanged runs do not
/// read them again.
fn queue_changed_file(
    store: &mut IndexStore,
    file: &DiscoveredFile,
    record: FileRecord,
    chunking: &ChunkingConfig,
    pending: &mut Pending,
    summary: &mut IndexSummary,
) -> Result<()> {
    let path = file.path.clone();
    match file.kind {
        FileKind::Image => pending.images.push(PendingImage { path, record }),
        FileKind::Audio | FileKind::Video => {
            pending.recordings.push(PendingRecording { path, record, kind: file.kind })
        }
        FileKind::Text => match decode_text(&read_file(&path)?) {
            Some(text) => {
                let chunks = chunk_text(&text, chunking).into_iter().map(|chunk| PendingChunk { page: None, chunk });
                pending.texts.push(PendingText { path, record, kind: FileKind::Text, chunks: chunks.collect() });
            }
            None => {
                store.record_skipped_file(&path, &record)?;
                summary.files_skipped_binary += 1;
            }
        },
        FileKind::Pdf => match extract_pdf_text(&read_file(&path)?) {
            Ok(text) if text.has_text() => {
                let chunks = chunk_pdf_pages(&text.pages, chunking);
                pending.texts.push(PendingText { path, record, kind: FileKind::Pdf, chunks });
            }
            Ok(_) => {
                store.record_skipped_file(&path, &record)?;
                summary.pdfs_without_text += 1;
            }
            Err(_) => {
                store.record_skipped_file(&path, &record)?;
                summary.pdfs_unreadable += 1;
            }
        },
    }
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

/// Chunks each page separately, so no chunk spans a page break and every result has one page.
fn chunk_pdf_pages(pages: &[String], chunking: &ChunkingConfig) -> Vec<PendingChunk> {
    pages
        .iter()
        .enumerate()
        .flat_map(|(index, page)| {
            chunk_text(page, chunking).into_iter().map(move |chunk| PendingChunk { page: Some(index + 1), chunk })
        })
        .collect()
}

fn metadata_matches(record: &FileRecord, file: &DiscoveredFile) -> bool {
    record.size == file.size && record.modified_nanoseconds == file.modified_nanoseconds
}

fn embed_texts(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    pending: Vec<PendingText>,
    progress: &mut dyn IndexProgress,
    summary: &mut IndexSummary,
) -> Result<()> {
    let mut group: Vec<PendingText> = Vec::new();
    let mut group_chunks = 0;
    for file in pending {
        group_chunks += file.chunks.len();
        group.push(file);
        if group_chunks >= CHUNKS_PER_ENCODE_CALL {
            store_text_group(store, encoder, std::mem::take(&mut group), summary)?;
            group_chunks = 0;
            progress.files_embedded(summary.files_embedded, summary.chunks_embedded);
        }
    }
    if !group.is_empty() {
        store_text_group(store, encoder, group, summary)?;
        progress.files_embedded(summary.files_embedded, summary.chunks_embedded);
    }
    Ok(())
}

fn store_text_group(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    group: Vec<PendingText>,
    summary: &mut IndexSummary,
) -> Result<()> {
    let titles: Vec<String> = group.iter().map(|file| file_title(&file.path)).collect();
    let documents: Vec<Document<'_>> = group
        .iter()
        .zip(&titles)
        .flat_map(|(file, title)| file.chunks.iter().map(move |pending| Document { title, text: &pending.chunk.text }))
        .collect();
    let mut embeddings = encoder.encode_documents(&documents)?.into_iter();

    for file in group {
        summary.chunks_embedded += file.chunks.len();
        summary.files_embedded += 1;
        let embedded = file.chunks.into_iter().map(|pending| {
            let embedding = embeddings.next().expect("encoder returns one embedding per document");
            (pending, embedding)
        });
        if file.kind == FileKind::Pdf {
            let chunks: Vec<(usize, Chunk, Vec<f32>)> = embedded
                .map(|(pending, embedding)| (pending.page.expect("PDF chunks carry a page"), pending.chunk, embedding))
                .collect();
            store.replace_pdf_file(&file.path, &file.record, &chunks)?;
        } else {
            let chunks: Vec<(Chunk, Vec<f32>)> =
                embedded.map(|(pending, embedding)| (pending.chunk, embedding)).collect();
            store.replace_text_file(&file.path, &file.record, &chunks)?;
        }
    }
    Ok(())
}

/// Embeds images one at a time while a background thread decodes the next ones.
///
/// Decoding a 12 MP JPEG takes ~50 ms, embedding ~270 ms on CUDA, so overlapping them hides the
/// decode; the bounded queue caps how many full-resolution photos are in memory at once. Images
/// are not batched: on a 4 GB GPU a batch of 2 was 3x slower per image (each vision layer already
/// materializes a large attention matrix) and a batch of 8 ran out of memory.
///
/// An undecodable image (corrupt, truncated, unsupported variant) is skipped and counted rather
/// than aborting the run: one bad download should not block indexing a photo library.
fn embed_images(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    pending: Vec<PendingImage>,
    progress: &mut dyn IndexProgress,
    summary: &mut IndexSummary,
) -> Result<()> {
    const DECODED_IMAGES_AHEAD: usize = 2;
    std::thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::sync_channel(DECODED_IMAGES_AHEAD);
        scope.spawn(move || {
            for file in pending {
                let loaded = load_image(&file.path);
                // A send error means embedding stopped with an error; stop decoding too.
                if sender.send((file, loaded)).is_err() {
                    break;
                }
            }
        });
        for (file, loaded) in receiver {
            match loaded {
                Ok(LoadedImage::Image(image)) => {
                    let embedding = encoder
                        .encode_image(&image)
                        .with_context(|| format!("failed to embed {}", file.path.display()))?;
                    store.replace_image_file(&file.path, &file.record, &embedding)?;
                    summary.files_embedded += 1;
                    summary.images_embedded += 1;
                    summary.chunks_embedded += 1;
                    progress.files_embedded(summary.files_embedded, summary.chunks_embedded);
                }
                Ok(LoadedImage::TooSmall) => {
                    store.record_skipped_file(&file.path, &file.record)?;
                    summary.images_skipped_too_small += 1;
                }
                Err(_) => {
                    store.record_skipped_file(&file.path, &file.record)?;
                    summary.images_unreadable += 1;
                }
            }
        }
        Ok(())
    })
}

/// The model card recommends the file name as the document title for code retrieval.
fn file_title(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
}
