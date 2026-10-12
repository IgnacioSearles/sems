//! Keeps searches current without a background service: before searching, the files in scope are
//! compared with the index by size and modification time (no file is read), and small text changes
//! are embedded on the spot. Larger changes, and photos or recordings (seconds each on a CPU), are
//! reported instead, so a search never turns into a long indexing run.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::discovery::{DiscoveryOptions, FileKind, discover_files};
use crate::encoder::Encoder;
use crate::indexer::{IndexOptions, IndexProgress, index_directory, metadata_matches};
use crate::store::{IndexStore, PathScope, SkippedFile};

/// Most changed text files a search re-embeds itself; about a second each on a CPU at worst.
pub const AUTO_UPDATE_LIMIT: usize = 25;

/// How the files under a folder differ from the index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PendingChanges {
    pub new_files: usize,
    pub modified_files: usize,
    pub removed_files: usize,
    /// New or modified images and recordings (also counted above), which a search does not embed.
    pub new_or_modified_media: usize,
}

impl PendingChanges {
    pub fn is_empty(&self) -> bool {
        self.new_files + self.modified_files + self.removed_files == 0
    }

    pub fn changed_files(&self) -> usize {
        self.new_files + self.modified_files + self.removed_files
    }

    /// Small enough, and cheap enough, to bring up to date before a search.
    fn can_update_before_search(&self) -> bool {
        self.new_or_modified_media == 0 && self.new_files + self.modified_files <= AUTO_UPDATE_LIMIT
    }
}

/// Everything under a folder that the index does not cover, or covers out of date. Each list is
/// sorted by path.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FolderScan {
    /// Not in the index yet.
    pub new_files: Vec<PathBuf>,
    /// In the index, but their size or timestamp changed since.
    pub modified_files: Vec<PathBuf>,
    /// In the index, but no longer on disk (or no longer included).
    pub deleted_files: Vec<PathBuf>,
    /// Tracked by the index without being embedded, and unchanged since.
    pub skipped_files: Vec<SkippedFile>,
    /// Over the folder's size limit for their kind.
    pub too_large: Vec<PathBuf>,
    /// Of a kind the folder was indexed without (for example images with --skip-images).
    pub left_out: usize,
    new_or_modified_media: usize,
}

impl FolderScan {
    pub fn pending_changes(&self) -> PendingChanges {
        PendingChanges {
            new_files: self.new_files.len(),
            modified_files: self.modified_files.len(),
            removed_files: self.deleted_files.len(),
            new_or_modified_media: self.new_or_modified_media,
        }
    }

    /// Files on disk that searches cannot find, or find as they were: all but the deleted.
    pub fn not_indexed(&self) -> usize {
        self.new_files.len()
            + self.modified_files.len()
            + self.skipped_files.len()
            + self.too_large.len()
            + self.left_out
    }
}

/// Compares the files under `root` (discovered with the options the folder was indexed with)
/// with the index; no file is read. A file whose timestamp changed but whose content did not
/// counts as modified: indexing it again only refreshes its metadata.
pub fn scan_folder(store: &IndexStore, root: &Path, options: &DiscoveryOptions) -> Result<FolderScan> {
    let scope = PathScope::new(root);
    let mut indexed = store.files_in_scope(&scope)?;
    let discovery = discover_files(root, options)?;
    let mut scan = FolderScan { too_large: discovery.too_large, left_out: discovery.left_out, ..FolderScan::default() };
    for file in discovery.files {
        match indexed.remove(&file.path) {
            Some(record) if metadata_matches(&record, &file) => continue,
            Some(_) => scan.modified_files.push(file.path),
            None => scan.new_files.push(file.path),
        }
        if matches!(file.kind, FileKind::Image | FileKind::Audio | FileKind::Video) {
            scan.new_or_modified_media += 1;
        }
    }
    scan.deleted_files = indexed.into_keys().collect();
    scan.deleted_files.sort();
    // A skipped file that changed is reconsidered by the next index, so it counts once, as modified.
    scan.skipped_files = store
        .skipped_files(&scope)?
        .into_iter()
        .filter(|skipped| {
            scan.modified_files.binary_search(&skipped.path).is_err()
                && scan.deleted_files.binary_search(&skipped.path).is_err()
        })
        .collect();
    Ok(scan)
}

/// How the files under `root` differ from the index (see [`scan_folder`]).
pub fn pending_changes(store: &IndexStore, root: &Path, options: &DiscoveryOptions) -> Result<PendingChanges> {
    Ok(scan_folder(store, root, options)?.pending_changes())
}

/// What [`refresh_before_search`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    UpToDate,
    /// The changes were applied to the index.
    Updated(PendingChanges),
    /// Too many or too costly to apply during a search; `sems index` will.
    Stale(PendingChanges),
}

/// Brings the index for `scope_root` up to date if that is cheap, using `encoder` (already loaded
/// for the query). `options` are the indexed folder's own, so files the folder was indexed
/// without (for example with --skip-images) do not count as new.
pub fn refresh_before_search(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    scope_root: &Path,
    options: &IndexOptions,
    progress: &mut dyn IndexProgress,
) -> Result<Freshness> {
    let changes = pending_changes(store, scope_root, &options.discovery)?;
    if changes.is_empty() {
        return Ok(Freshness::UpToDate);
    }
    if !changes.can_update_before_search() {
        return Ok(Freshness::Stale(changes));
    }
    index_directory(store, encoder, scope_root, options, progress)?;
    Ok(Freshness::Updated(changes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_small_text_changes_are_applied_during_a_search() {
        let text_only = PendingChanges { new_files: 2, modified_files: 3, removed_files: 40, new_or_modified_media: 0 };
        assert!(text_only.can_update_before_search(), "removals are cheap whatever their number");
        let one_photo = PendingChanges { new_files: 1, new_or_modified_media: 1, ..PendingChanges::default() };
        assert!(!one_photo.can_update_before_search());
        let many = PendingChanges { modified_files: AUTO_UPDATE_LIMIT + 1, ..PendingChanges::default() };
        assert!(!many.can_update_before_search());
    }
}
