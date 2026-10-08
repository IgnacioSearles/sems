//! Finds candidate files under a root, honoring .gitignore, hidden-file rules and `.semsignore`.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use ignore::WalkBuilder;

use crate::media::is_image_path;

/// Per-directory ignore file for paths that should stay out of the index but not out of git.
pub const IGNORE_FILE_NAME: &str = ".semsignore";
/// How much of a file to inspect when deciding whether it is binary (same heuristic as git).
const BINARY_SNIFF_BYTES: usize = 8 * 1024;

/// How a file is indexed: text is chunked and embedded as text, images are embedded from pixels,
/// and PDFs have their text extracted and chunked page by page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Text,
    Image,
    Pdf,
}

impl FileKind {
    fn of(path: &Path) -> Self {
        if is_image_path(path) {
            return Self::Image;
        }
        let is_pdf = path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"));
        if is_pdf { Self::Pdf } else { Self::Text }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub kind: FileKind,
    pub size: u64,
    pub modified_nanoseconds: i64,
}

/// Per-kind size limits: a large text file is usually generated data, while large photos and
/// PDFs are normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveryOptions {
    pub max_text_size: u64,
    /// Limit for images and PDFs.
    pub max_media_size: u64,
    pub include_images: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Discovery {
    pub files: Vec<DiscoveredFile>,
    pub skipped_too_large: usize,
}

/// Lists regular files under `root` without reading their contents.
pub fn discover_files(root: &Path, options: &DiscoveryOptions) -> Result<Discovery> {
    let mut discovery = Discovery::default();
    let walker = WalkBuilder::new(root).add_custom_ignore_filename(IGNORE_FILE_NAME).build();
    for entry in walker {
        let entry = entry.with_context(|| format!("failed to walk {}", root.display()))?;
        if !entry.file_type().is_some_and(|file_type| file_type.is_file()) {
            continue;
        }
        let metadata =
            entry.metadata().with_context(|| format!("failed to read metadata of {}", entry.path().display()))?;
        let kind = FileKind::of(entry.path());
        if kind == FileKind::Image && !options.include_images {
            continue;
        }
        let max_size = match kind {
            FileKind::Text => options.max_text_size,
            FileKind::Image | FileKind::Pdf => options.max_media_size,
        };
        if metadata.len() > max_size {
            discovery.skipped_too_large += 1;
            continue;
        }
        discovery.files.push(DiscoveredFile {
            path: entry.into_path(),
            kind,
            size: metadata.len(),
            modified_nanoseconds: modified_nanoseconds(&metadata),
        });
    }
    discovery.files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(discovery)
}

fn modified_nanoseconds(metadata: &std::fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX))
}

/// Decodes file bytes as text, or returns `None` for binary content.
pub fn decode_text(bytes: &[u8]) -> Option<String> {
    let sniffed = &bytes[..bytes.len().min(BINARY_SNIFF_BYTES)];
    if sniffed.contains(&0) {
        return None;
    }
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    Some(String::from_utf8_lossy(bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, relative: &str, contents: &[u8]) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn relative_paths(root: &Path, discovery: &Discovery) -> Vec<String> {
        discovery
            .files
            .iter()
            .map(|file| file.path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"))
            .collect()
    }

    #[test]
    fn honors_gitignore_semsignore_and_hidden_files() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::create_dir(root.join(".git")).unwrap(); // gitignore rules apply inside a repository
        write(root, ".gitignore", b"target/\n");
        write(root, ".semsignore", b"*.lock\n");
        write(root, "src/main.rs", b"fn main() {}");
        write(root, "target/debug/output.txt", b"build output");
        write(root, "Cargo.lock", b"lock");
        write(root, ".hidden/secret.txt", b"hidden");

        let discovery = discover_files(root, &unlimited()).unwrap();
        assert_eq!(relative_paths(root, &discovery), ["src/main.rs"]);
    }

    fn unlimited() -> DiscoveryOptions {
        DiscoveryOptions { max_text_size: u64::MAX, max_media_size: u64::MAX, include_images: true }
    }

    #[test]
    fn classifies_images_and_applies_per_kind_limits() {
        let directory = tempfile::tempdir().unwrap();
        write(directory.path(), "photo.JPG", &[b'x'; 100]);
        write(directory.path(), "notes.txt", &[b'x'; 100]);
        let options = DiscoveryOptions { max_text_size: 50, max_media_size: 1_000, include_images: true };

        let discovery = discover_files(directory.path(), &options).unwrap();
        assert_eq!(relative_paths(directory.path(), &discovery), ["photo.JPG"]);
        assert_eq!(discovery.files[0].kind, FileKind::Image);
        assert_eq!(discovery.skipped_too_large, 1);
    }

    #[test]
    fn recognizes_pdfs_case_insensitively() {
        assert_eq!(FileKind::of(Path::new("Lease.PDF")), FileKind::Pdf);
        assert_eq!(FileKind::of(Path::new("notes.md")), FileKind::Text);
        assert_eq!(FileKind::of(Path::new("photo.jpeg")), FileKind::Image);
    }

    #[test]
    fn images_can_be_excluded() {
        let directory = tempfile::tempdir().unwrap();
        write(directory.path(), "photo.png", b"x");
        write(directory.path(), "notes.txt", b"x");
        let options = DiscoveryOptions { include_images: false, ..unlimited() };
        let discovery = discover_files(directory.path(), &options).unwrap();
        assert_eq!(relative_paths(directory.path(), &discovery), ["notes.txt"]);
    }

    #[test]
    fn skips_files_above_the_size_limit() {
        let directory = tempfile::tempdir().unwrap();
        write(directory.path(), "small.txt", b"tiny");
        write(directory.path(), "large.txt", &[b'x'; 100]);

        let options = DiscoveryOptions { max_text_size: 50, ..unlimited() };
        let discovery = discover_files(directory.path(), &options).unwrap();
        assert_eq!(relative_paths(directory.path(), &discovery), ["small.txt"]);
        assert_eq!(discovery.skipped_too_large, 1);
    }

    #[test]
    fn decode_text_rejects_binary_and_strips_bom() {
        assert_eq!(decode_text(b"\x89PNG\r\n\x1a\n\x00\x00"), None);
        assert_eq!(decode_text(b"\xEF\xBB\xBFhello").as_deref(), Some("hello"));
        assert_eq!(decode_text(b"caf\xE9").as_deref(), Some("caf\u{FFFD}"));
    }
}
