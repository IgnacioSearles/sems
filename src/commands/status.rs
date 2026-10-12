//! `sems status` and `sems forget`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use sems::discovery::DiscoveryOptions;
use sems::freshness::{FolderScan, scan_folder};
use sems::render::display_path;
use sems::store::{IndexStore, IndexedRoot, PathScope, ScopeStatistics, SkipReason};

use super::progress::{files, readable_age, readable_size};
use super::{now_seconds, open_store, resolve_existing_path, write_stdout};
use crate::Locations;

/// Files shown per category of a folder report, unless `--list` asks for all of them.
const SAMPLE_FILES: usize = 5;

/// Every indexed folder, or, given a path, what is indexed under it and which files are not.
pub fn run_status(path: Option<&Path>, list_all: bool, locations: &Locations) -> Result<()> {
    let store = open_store(locations)?;
    // A file list belongs to one folder, so `--list` alone means the current directory.
    let mut report = match (path, list_all) {
        (None, false) => indexed_folders_report(&store)?,
        (path, _) => folder_report(&store, &resolve_existing_path(path)?, list_all)?,
    };
    let index_path = locations.index_path()?;
    let size = std::fs::metadata(&index_path).map(|metadata| readable_size(metadata.len())).unwrap_or_default();
    writeln!(report, "index: {} ({size})", index_path.display())?;
    write_stdout(&report)
}

/// One line per indexed folder: what is indexed, when, and what is not.
fn indexed_folders_report(store: &IndexStore) -> Result<String> {
    let roots = store.indexed_roots()?;
    if roots.is_empty() {
        return Ok("no indexed folders yet; run `sems index FOLDER` (folders indexed by older versions of sems \
                   are listed after their next `sems index`)\n"
            .to_string());
    }
    let working_directory = std::env::current_dir()?;
    let mut report = String::new();
    for root in roots {
        let statistics = store.statistics(&PathScope::new(&root.path))?;
        let scan = scan_folder(store, &root.path, &root.options)?;
        writeln!(
            report,
            "{}  {}, indexed {}; {}",
            display_path(&root.path, &working_directory),
            describe(&statistics),
            age(&root),
            summarize(&scan)
        )?;
    }
    Ok(report)
}

/// `up to date`, or `12 new or changed, 4 skipped, 3 deleted`.
fn summarize(scan: &FolderScan) -> String {
    let parts: Vec<String> = [
        (scan.new_files.len() + scan.modified_files.len(), "new or changed"),
        (scan.skipped_files.len() + scan.too_large.len(), "skipped"),
        (scan.left_out, "left out"),
        (scan.deleted_files.len(), "deleted"),
    ]
    .into_iter()
    .filter(|(count, _)| *count > 0)
    .map(|(count, label)| format!("{count} {label}"))
    .collect();
    if parts.is_empty() { "up to date".to_string() } else { parts.join(", ") }
}

/// What is indexed under `path`, then each kind of file that is not, with sample files.
fn folder_report(store: &IndexStore, path: &Path, list_all: bool) -> Result<String> {
    let working_directory = std::env::current_dir()?;
    let statistics = store.statistics(&PathScope::new(path))?;
    let mut report = format!("{}: {}\n", display_path(path, &working_directory), describe(&statistics));
    let Some(root) = store.indexed_root_containing(path)? else {
        writeln!(report, "not part of an indexed folder sems recorded; run `sems index` on it to see what is missing")?;
        return Ok(report);
    };
    writeln!(report, "indexed {} as part of {}", age(&root), display_path(&root.path, &working_directory))?;
    let scan = scan_folder(store, path, &root.options)?;
    if scan.not_indexed() == 0 && scan.deleted_files.is_empty() {
        writeln!(report, "every file is indexed and up to date")?;
        return Ok(report);
    }
    let categories = Categories { report: &mut report, base: path, list_all };
    categories.write(&scan, &root.options)?;
    if !scan.pending_changes().is_empty() {
        writeln!(report, "`sems index {}` brings the index up to date", root.path.display())?;
    }
    Ok(report)
}

/// Writes a folder's not-indexed files by category: a count, then sample paths (relative to the
/// folder), or all of them with `--list`.
struct Categories<'a> {
    report: &'a mut String,
    base: &'a Path,
    list_all: bool,
}

impl Categories<'_> {
    fn write(mut self, scan: &FolderScan, options: &DiscoveryOptions) -> Result<()> {
        if scan.not_indexed() > 0 {
            writeln!(self.report, "not indexed: {}", files(scan.not_indexed()))?;
        }
        self.category("new since the last index", &scan.new_files)?;
        self.category("changed since the last index", &scan.modified_files)?;
        let mut skipped: BTreeMap<Option<SkipReason>, Vec<PathBuf>> = BTreeMap::new();
        for file in &scan.skipped_files {
            skipped.entry(file.reason).or_default().push(file.path.clone());
        }
        for (reason, paths) in &skipped {
            self.category(describe_skip(*reason), paths)?;
        }
        self.category("too large", &scan.too_large)?;
        if scan.left_out > 0 {
            writeln!(self.report, "  {:>5}  left out ({})", scan.left_out, left_out_flags(options))?;
        }
        self.category("deleted since the last index (searches still find them until then)", &scan.deleted_files)?;
        Ok(())
    }

    fn category(&mut self, label: &str, paths: &[PathBuf]) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        writeln!(self.report, "  {:>5}  {label}", paths.len())?;
        let shown = if self.list_all { paths.len() } else { paths.len().min(SAMPLE_FILES) };
        for path in &paths[..shown] {
            writeln!(self.report, "         {}", display_path(path, self.base))?;
        }
        if shown < paths.len() {
            writeln!(self.report, "         ... and {} more (--list shows all)", paths.len() - shown)?;
        }
        Ok(())
    }
}

fn describe_skip(reason: Option<SkipReason>) -> &'static str {
    match reason {
        Some(SkipReason::Binary) => "skipped: binary",
        Some(SkipReason::ImageTooSmall) => "skipped: images too small to be photos (icons)",
        Some(SkipReason::ImageUnreadable) => "skipped: unreadable images",
        Some(SkipReason::PdfWithoutText) => "skipped: PDFs without text (scans need OCR)",
        Some(SkipReason::PdfUnreadable) => "skipped: unreadable PDFs",
        Some(SkipReason::RecordingUnreadable) => "skipped: unreadable audio and video",
        Some(SkipReason::RecordingWithoutContent) => "skipped: silent audio and video",
        None => "skipped (indexed before sems recorded why)",
    }
}

/// The `sems index` flags that left files out, as the folder was indexed with them.
fn left_out_flags(options: &DiscoveryOptions) -> String {
    let flags: Vec<&str> = [
        (!options.include_images, "--skip-images"),
        (!options.include_audio, "--skip-audio"),
        (!options.include_video, "--skip-video"),
    ]
    .into_iter()
    .filter(|(set, _)| *set)
    .map(|(_, flag)| flag)
    .collect();
    format!("indexed with {}", flags.join(" "))
}

fn describe(statistics: &ScopeStatistics) -> String {
    format!(
        "{} files ({} images, {} audio and video), {} chunks",
        statistics.files, statistics.images, statistics.recordings, statistics.chunks
    )
}

fn age(root: &IndexedRoot) -> String {
    readable_age(now_seconds() - root.indexed_at_seconds)
}

/// Removes everything under `path` from the index; the folder may no longer exist.
pub fn run_forget(path: &Path, locations: &Locations) -> Result<()> {
    let path = resolve_maybe_missing_path(path)?;
    let mut store = open_store(locations)?;
    let removed = store.forget(&PathScope::new(&path))?;
    if removed == 0 {
        eprintln!("nothing was indexed under {}", path.display());
        return Ok(());
    }
    store.compact()?;
    eprintln!("removed {} under {} from the index", files(removed), path.display());
    if let Some(root) = store.indexed_root_containing(&path)?.filter(|_| path.exists()) {
        eprintln!(
            "it is still inside {}, so the next `sems index` there adds it back; list it in a .semsignore \
             file to keep it out",
            root.path.display()
        );
    }
    Ok(())
}

/// The canonical path if it exists, otherwise the absolute one: a deleted folder cannot be
/// canonicalized, and forgetting it is the point.
fn resolve_maybe_missing_path(path: &Path) -> Result<PathBuf> {
    match dunce::canonicalize(path) {
        Ok(canonical) => Ok(canonical),
        Err(_) => std::path::absolute(path).with_context(|| format!("invalid path {}", path.display())),
    }
}
