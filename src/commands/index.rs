//! `sems index`.

use std::path::Path;

use anyhow::{Result, bail};

use sems::av::Ffmpeg;
use sems::discovery::DiscoveryOptions;
use sems::embedding::ExecutionDevice;
use sems::encoder::Encoder;
use sems::indexer::{IndexOptions, IndexSummary, index_directory};
use sems::store::IndexStore;

use super::progress::TerminalProgress;
use super::{device_name, index_identity, load_encoder, now_seconds, resolve_existing_path};
use crate::{IndexArguments, Locations};

pub fn run(arguments: &IndexArguments, locations: &Locations) -> Result<()> {
    let root = resolve_existing_path(arguments.path.as_deref())?;
    if !root.is_dir() {
        bail!("{} is not a directory", root.display());
    }
    // Load the model first: a missing runtime should fail before the index is created or cleared.
    let (mut encoder, device) = load_encoder(locations, arguments.device)?;
    let index_path = locations.index_path()?;
    let mut store = if arguments.rebuild {
        IndexStore::recreate(&index_path, index_identity())?
    } else {
        IndexStore::open(&index_path, index_identity())?
    };
    let options = IndexOptions {
        discovery: DiscoveryOptions {
            max_text_size: arguments.max_file_size,
            max_media_size: arguments.max_media_size,
            include_images: !arguments.skip_images,
            include_audio: !arguments.skip_audio,
            include_video: !arguments.skip_video,
        },
        ffmpeg: Ffmpeg::locate(),
        ..IndexOptions::default()
    };
    let mut progress = TerminalProgress::new();
    let summary = index_directory(&mut store, &mut encoder, &root, &options, &mut progress)?;
    progress.finish();
    // Remembered so searches can tell what changed since, and `sems status` can list the folder.
    store.record_indexed_root(&root, &options.discovery, now_seconds())?;
    eprintln!("{}", summary_line(&root, &summary, encoder.dimensions(), device));
    Ok(())
}

fn summary_line(root: &Path, summary: &IndexSummary, dimensions: usize, device: ExecutionDevice) -> String {
    let skipped: Vec<String> = [
        (summary.files_skipped_binary, "binary"),
        (summary.files_skipped_too_large, "too large"),
        (summary.images_skipped_too_small, "tiny images"),
        (summary.images_unreadable, "unreadable images"),
        (summary.pdfs_without_text, "PDFs without text (scans need OCR)"),
        (summary.pdfs_unreadable, "unreadable PDFs"),
        (summary.recordings_without_ffmpeg, "audio/video files (install ffmpeg to index them)"),
        (summary.recordings_unreadable, "unreadable audio/video files"),
        (summary.recordings_without_content, "silent audio/video files"),
    ]
    .into_iter()
    .filter(|(count, _)| *count > 0)
    .map(|(count, reason)| format!("{count} {reason}"))
    .collect();
    let skipped = if skipped.is_empty() { String::new() } else { format!(", skipped {}", skipped.join(", ")) };
    let media: Vec<String> = [(summary.images_embedded, "images"), (summary.recordings_embedded, "audio/video")]
        .into_iter()
        .filter(|(count, _)| *count > 0)
        .map(|(count, kind)| format!("{count} {kind}"))
        .collect();
    let media = if media.is_empty() { String::new() } else { format!(" ({})", media.join(", ")) };
    format!(
        "indexed {}: {} files ({} unchanged, {} embedded{media} into {} chunks, {} removed{skipped}) in {:.1}s [{}, {dimensions}d]",
        root.display(),
        summary.files_seen,
        summary.files_unchanged,
        summary.files_embedded,
        summary.chunks_embedded,
        summary.files_removed,
        summary.elapsed.as_secs_f64(),
        device_name(device),
    )
}
