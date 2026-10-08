//! End-to-end indexing and search over real files, with a deterministic fake encoder so the tests
//! exercise change detection, storage and retrieval without the model.

use std::path::{Path, PathBuf};

use anyhow::Result;
use image::DynamicImage;
use sems::chunking::ChunkingConfig;
use sems::discovery::FileKind;
use sems::encoder::{Document, Encoder};
use sems::indexer::{IndexOptions, IndexSummary, SilentProgress, index_directory};
use sems::search::{ResultContent, SearchOptions, search};
use sems::store::{IndexIdentity, IndexStore, PathScope};

const DIMENSIONS: usize = 64;

/// Hashes words into buckets: texts sharing words get similar vectors. Counts documents encoded so
/// tests can assert that unchanged files are never re-embedded.
#[derive(Default)]
struct BagOfWordsEncoder {
    documents_encoded: usize,
    images_encoded: usize,
}

impl BagOfWordsEncoder {
    fn vector(text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; DIMENSIONS];
        for word in text.split(|character: char| !character.is_alphanumeric()).filter(|word| !word.is_empty()) {
            let bucket = blake3::hash(word.to_lowercase().as_bytes()).as_bytes()[0] as usize % DIMENSIONS;
            vector[bucket] += 1.0;
        }
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt().max(f32::EPSILON);
        vector.iter().map(|value| value / norm).collect()
    }
}

impl Encoder for BagOfWordsEncoder {
    fn identity(&self) -> String {
        "bag-of-words".into()
    }

    fn dimensions(&self) -> usize {
        DIMENSIONS
    }

    fn encode_query(&mut self, query: &str) -> Result<Vec<f32>> {
        Ok(Self::vector(query))
    }

    fn encode_documents(&mut self, documents: &[Document<'_>]) -> Result<Vec<Vec<f32>>> {
        self.documents_encoded += documents.len();
        Ok(documents.iter().map(|document| Self::vector(document.text)).collect())
    }

    /// Names the image's dominant channel, so a red picture lands near the query "red".
    fn encode_image(&mut self, image: &DynamicImage) -> Result<Vec<f32>> {
        self.images_encoded += 1;
        let rgb = image.to_rgb8();
        let mut totals = [0u64; 3];
        for pixel in rgb.pixels() {
            for (total, value) in totals.iter_mut().zip(pixel.0) {
                *total += u64::from(value);
            }
        }
        let dominant = ["red", "green", "blue"][(0..3).max_by_key(|&channel| totals[channel]).unwrap()];
        Ok(Self::vector(dominant))
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    store: IndexStore,
    encoder: BagOfWordsEncoder,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(directory.path()).unwrap();
        let identity = IndexIdentity {
            encoder: "bag-of-words".into(),
            dimensions: DIMENSIONS,
            chunker_version: ChunkingConfig::VERSION,
        };
        let store = IndexStore::open_in_memory(identity).unwrap();
        Self { _directory: directory, root, store, encoder: BagOfWordsEncoder::default() }
    }

    fn write(&self, relative: &str, contents: &str) {
        let path = self.root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn index(&mut self) -> IndexSummary {
        let options = IndexOptions::default();
        index_directory(&mut self.store, &mut self.encoder, &self.root, &options, &mut SilentProgress).unwrap()
    }

    fn write_image(&self, relative: &str, color: [u8; 3], size: u32) {
        let path = self.root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        image::RgbImage::from_pixel(size, size, image::Rgb(color)).save(path).unwrap();
    }

    fn top_result(&mut self, query: &str) -> Option<PathBuf> {
        let scope = PathScope::new(&self.root);
        let results = search(&self.store, &mut self.encoder, &scope, query, SearchOptions::default()).unwrap();
        results.first().map(|result| relative(&self.root, &result.path))
    }
}

fn relative(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap().to_path_buf()
}

#[test]
fn indexes_and_finds_files_by_content() {
    let mut fixture = Fixture::new();
    fixture.write("recipes/pancakes.md", "Whisk flour, eggs and milk. Fry pancakes in butter.");
    fixture.write("src/network.rs", "fn retry_with_backoff() { sleep(delay); reconnect(); }");

    let summary = fixture.index();
    assert_eq!((summary.files_seen, summary.files_embedded), (2, 2));
    assert_eq!(fixture.top_result("pancakes with butter"), Some(PathBuf::from("recipes/pancakes.md")));
    assert_eq!(fixture.top_result("retry_with_backoff"), Some(PathBuf::from("src/network.rs")));
}

#[test]
fn second_run_without_changes_embeds_nothing() {
    let mut fixture = Fixture::new();
    fixture.write("a.txt", "alpha");
    fixture.write("b.txt", "beta");
    fixture.index();
    let encoded_before = fixture.encoder.documents_encoded;

    let summary = fixture.index();
    assert_eq!(summary.files_unchanged, 2);
    assert_eq!(fixture.encoder.documents_encoded, encoded_before);
}

#[test]
fn only_modified_files_are_re_embedded() {
    let mut fixture = Fixture::new();
    fixture.write("stable.txt", "unchanging content");
    fixture.write("edited.txt", "first draft about apples");
    fixture.index();

    fixture.write("edited.txt", "second draft about oranges and more text");
    let summary = fixture.index();
    assert_eq!((summary.files_embedded, summary.files_unchanged), (1, 1));
    assert_eq!(fixture.top_result("oranges"), Some(PathBuf::from("edited.txt")));
}

#[test]
fn touched_but_identical_files_are_not_re_embedded() {
    let mut fixture = Fixture::new();
    fixture.write("same.txt", "identical bytes");
    fixture.index();
    let encoded_before = fixture.encoder.documents_encoded;

    let file = std::fs::File::options().write(true).open(fixture.root.join("same.txt")).unwrap();
    file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(60)).unwrap();
    let summary = fixture.index();
    assert_eq!(summary.files_unchanged, 1);
    assert_eq!(fixture.encoder.documents_encoded, encoded_before);
}

#[test]
fn deleted_and_newly_ignored_files_leave_the_index() {
    let mut fixture = Fixture::new();
    fixture.write("keep.txt", "keep me");
    fixture.write("delete.txt", "delete me");
    fixture.write("generated.txt", "generated output");
    fixture.index();

    std::fs::remove_file(fixture.root.join("delete.txt")).unwrap();
    fixture.write(".semsignore", "generated.txt\n");
    let summary = fixture.index();
    assert_eq!(summary.files_removed, 2);
    let statistics = fixture.store.statistics(&PathScope::new(&fixture.root)).unwrap();
    assert_eq!(statistics.files, 1);
}

#[test]
fn binary_files_are_skipped() {
    let mut fixture = Fixture::new();
    std::fs::write(fixture.root.join("program.bin"), b"MZ\x90\x00\x03\x00\x00\x00").unwrap();
    fixture.write("readme.txt", "hello");
    let summary = fixture.index();
    assert_eq!((summary.files_embedded, summary.files_skipped_binary), (1, 1));
}

#[test]
fn search_is_limited_to_the_requested_directory() {
    let mut fixture = Fixture::new();
    fixture.write("work/report.txt", "quarterly revenue report");
    fixture.write("personal/diary.txt", "quarterly revenue report");
    fixture.index();

    let scope = PathScope::new(&fixture.root.join("work"));
    let results = search(&fixture.store, &mut fixture.encoder, &scope, "revenue", SearchOptions::default()).unwrap();
    let paths: Vec<PathBuf> = results.iter().map(|result| relative(&fixture.root, &result.path)).collect();
    assert_eq!(paths, [PathBuf::from("work/report.txt")]);
}

#[test]
fn one_result_per_file_makes_the_limit_count_files() {
    let mut fixture = Fixture::new();
    let long_file: String = (0..200)
        .map(|line| {
            format!(
                "revenue line {line}
"
            )
        })
        .collect();
    fixture.write("long.txt", &long_file);
    fixture.write("short.txt", "revenue summary");
    fixture.index();

    let scope = PathScope::new(&fixture.root);
    let options = SearchOptions { limit: 2, one_result_per_file: true, ..SearchOptions::default() };
    let results = search(&fixture.store, &mut fixture.encoder, &scope, "revenue", options).unwrap();
    let mut paths: Vec<PathBuf> = results.iter().map(|result| relative(&fixture.root, &result.path)).collect();
    paths.sort();
    assert_eq!(paths, [PathBuf::from("long.txt"), PathBuf::from("short.txt")]);
}

#[test]
fn images_are_indexed_and_found_by_text_queries() {
    let mut fixture = Fixture::new();
    fixture.write_image("photos/sunset.png", [220, 40, 30], 100);
    fixture.write_image("photos/forest.png", [20, 180, 40], 100);
    fixture.write("notes.txt", "meeting notes about the budget");

    let summary = fixture.index();
    assert_eq!((summary.files_embedded, summary.images_embedded), (3, 2));
    assert_eq!(fixture.top_result("red"), Some(PathBuf::from("photos/sunset.png")));

    let scope = PathScope::new(&fixture.root);
    let options = SearchOptions { kind: Some(FileKind::Image), ..SearchOptions::default() };
    let results = search(&fixture.store, &mut fixture.encoder, &scope, "budget", options).unwrap();
    assert!(results.iter().all(|result| result.content == ResultContent::Image), "kind filter leaked text");
    assert_eq!(fixture.store.statistics(&scope).unwrap().images, 2);
}

#[test]
fn tiny_and_corrupt_images_are_skipped_once_and_not_reread() {
    let mut fixture = Fixture::new();
    fixture.write_image("icon.png", [0, 0, 255], 16);
    std::fs::write(fixture.root.join("broken.jpg"), b"\xFF\xD8\xFF\xE0 truncated").unwrap();
    std::fs::write(fixture.root.join("blob.bin"), b"\x00\x01\x02").unwrap();

    let summary = fixture.index();
    assert_eq!((summary.images_skipped_too_small, summary.images_unreadable, summary.files_skipped_binary), (1, 1, 1));
    assert_eq!(fixture.encoder.images_encoded, 0);

    let second = fixture.index();
    assert_eq!(second.files_unchanged, 3, "skipped files are tracked, so unchanged ones are not re-read");
}

fn copy_pdf_fixture(fixture: &Fixture, name: &str) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval_corpus/docs").join(name);
    std::fs::copy(source, fixture.root.join(name)).unwrap();
}

#[test]
fn pdfs_are_indexed_page_by_page() {
    let mut fixture = Fixture::new();
    copy_pdf_fixture(&fixture, "lease_agreement.pdf");
    let summary = fixture.index();
    assert_eq!(summary.files_embedded, 1);

    let scope = PathScope::new(&fixture.root);
    let options = SearchOptions { kind: Some(FileKind::Pdf), relevance_cutoff: false, ..SearchOptions::default() };
    let results = search(&fixture.store, &mut fixture.encoder, &scope, "termination notice", options).unwrap();
    let pages: Vec<usize> = results
        .iter()
        .map(|result| match result.content {
            ResultContent::Pdf { page, .. } => page,
            ref other => panic!("expected a PDF result, got {other:?}"),
        })
        .collect();
    assert_eq!(pages.first(), Some(&3), "the termination clause is on page 3");
}

#[test]
fn scanned_and_corrupt_pdfs_are_skipped() {
    let mut fixture = Fixture::new();
    copy_pdf_fixture(&fixture, "scanned_receipt.pdf");
    std::fs::write(fixture.root.join("broken.pdf"), b"%PDF-1.4 truncated").unwrap();
    let summary = fixture.index();
    assert_eq!((summary.pdfs_without_text, summary.pdfs_unreadable, summary.files_embedded), (1, 1, 0));
    assert_eq!(fixture.index().files_unchanged, 2, "skipped PDFs are tracked and not re-read");
}

#[test]
fn skipping_images_leaves_them_out_of_the_index() {
    let mut fixture = Fixture::new();
    fixture.write_image("photo.png", [220, 40, 30], 100);
    fixture.write("readme.txt", "hello");
    let mut options = IndexOptions::default();
    options.discovery.include_images = false;

    let summary =
        index_directory(&mut fixture.store, &mut fixture.encoder, &fixture.root, &options, &mut SilentProgress)
            .unwrap();
    assert_eq!((summary.files_seen, summary.images_embedded), (1, 0));
}
