//! Retrieval quality on a small labelled corpus, with the real model. Guards against regressions
//! when changing chunking, prompts, fusion, or image handling, and shows which queries fail. Needs
//! the exported model and ONNX Runtime, so it is ignored by default:
//!
//!     SEMS_ONNXRUNTIME=path/to/onnxruntime.dll cargo test --release --test search_quality -- --ignored --nocapture
//!
//! Measured over one mixed index of text files and photos:
//! - retrieval: is the right file ranked first? (`eval_queries.json`)
//! - localization: does the top result contain the right line? (`eval_localization.json`)
//! - images: does a text description rank the right photo first, among text and other photos?
//!   (`eval_images.json`; photo file names are generic, so only pixels can answer)

use std::path::{Path, PathBuf};

use sems::chunking::ChunkingConfig;
use sems::discovery::FileKind;
use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::encoder::{Encoder, GemmaEncoder, GemmaEncoderConfig};
use sems::indexer::{IndexOptions, SilentProgress, index_directory};
use sems::search::{ResultContent, SearchOptions, SearchResult, search};
use sems::store::{IndexIdentity, IndexStore, PathScope};
use serde::Deserialize;

/// Floors set just below measured quality; raise them when quality improves.
const MINIMUM_RECALL_AT_1: f32 = 0.9;
const MINIMUM_LOCATED_AT_1: f32 = 0.9;
const MINIMUM_IMAGE_RECALL_AT_1: f32 = 0.9;

#[derive(Deserialize)]
struct LabelledQuery {
    query: String,
    expected: String,
    /// For localization queries: a line the top result must contain.
    expected_line: Option<usize>,
}

struct Evaluation {
    store: IndexStore,
    encoder: GemmaEncoder,
    corpus: PathBuf,
}

impl Evaluation {
    fn top_result(&mut self, query: &str, kind: Option<FileKind>) -> Option<SearchResult> {
        let scope = PathScope::new(&self.corpus);
        let options = SearchOptions { limit: 1, kind, ..SearchOptions::default() };
        search(&self.store, &mut self.encoder, &scope, query, options).unwrap().into_iter().next()
    }

    fn relative_path(&self, result: &SearchResult) -> String {
        result.path.strip_prefix(&self.corpus).unwrap().to_string_lossy().replace('\\', "/")
    }

    fn describe(&self, result: &SearchResult) -> String {
        match &result.content {
            ResultContent::Text { start_line, end_line, .. } => {
                format!("{}:{start_line}-{end_line}", self.relative_path(result))
            }
            ResultContent::Image => self.relative_path(result),
        }
    }

    fn is_hit(&self, result: &SearchResult, labelled: &LabelledQuery) -> bool {
        let line_matches = match (&result.content, labelled.expected_line) {
            (_, None) => true,
            (ResultContent::Text { start_line, end_line, .. }, Some(line)) => (*start_line..=*end_line).contains(&line),
            (ResultContent::Image, Some(_)) => false,
        };
        self.relative_path(result) == labelled.expected && line_matches
    }

    /// Fraction of queries whose top result is the expected file (and line, if labelled).
    fn score(&mut self, label: &str, queries: &[LabelledQuery], kind: Option<FileKind>) -> f32 {
        let mut misses = Vec::new();
        for labelled in queries {
            let top = self.top_result(&labelled.query, kind);
            if !top.as_ref().is_some_and(|result| self.is_hit(result, labelled)) {
                let got = top.map(|result| self.describe(&result));
                let wanted = labelled.expected_line.map_or(String::new(), |line| format!(":{line}"));
                misses.push(format!("{:?}: wanted {}{wanted}, got {got:?}", labelled.query, labelled.expected));
            }
        }
        let score = 1.0 - misses.len() as f32 / queries.len() as f32;
        println!("{label:<34} {score:.2}");
        for miss in misses {
            println!("    {miss}");
        }
        score
    }
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn load_queries(file_name: &str) -> Vec<LabelledQuery> {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(file_name)).unwrap()).unwrap()
}

fn build_evaluation() -> Evaluation {
    let library = std::env::var_os("SEMS_ONNXRUNTIME").expect("set SEMS_ONNXRUNTIME to the onnxruntime library path");
    let runtime = OnnxRuntime::load(Path::new(&library)).expect("failed to load ONNX Runtime");
    let model_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models").join("onnx");
    let model = EmbeddingModel::load(runtime, &model_directory, ExecutionDevice::Cpu).expect("failed to load model");
    let mut encoder = GemmaEncoder::new(model, GemmaEncoderConfig::default()).unwrap();

    let identity = IndexIdentity {
        encoder: encoder.identity(),
        dimensions: encoder.dimensions(),
        chunker_version: ChunkingConfig::VERSION,
    };
    let mut store = IndexStore::open_in_memory(identity).unwrap();
    let corpus = dunce::canonicalize(fixtures().join("eval_corpus")).unwrap();
    let summary =
        index_directory(&mut store, &mut encoder, &corpus, &IndexOptions::default(), &mut SilentProgress).unwrap();
    assert!(summary.images_embedded > 0, "the corpus photos were not indexed");
    Evaluation { store, encoder, corpus }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn retrieval_localization_and_images_meet_floor() {
    let mut evaluation = build_evaluation();
    let recall = evaluation.score("text retrieval recall@1", &load_queries("eval_queries.json"), None);
    let located = evaluation.score("localization located@1", &load_queries("eval_localization.json"), None);
    let images = load_queries("eval_images.json");
    let image_recall = evaluation.score("image recall@1 (mixed with text)", &images, None);
    let image_only_recall = evaluation.score("image recall@1 (--kind image)", &images, Some(FileKind::Image));

    assert!(recall >= MINIMUM_RECALL_AT_1, "text retrieval regressed: {recall:.2}");
    assert!(located >= MINIMUM_LOCATED_AT_1, "localization regressed: {located:.2}");
    assert!(image_recall >= MINIMUM_IMAGE_RECALL_AT_1, "image retrieval regressed: {image_recall:.2}");
    assert!(image_only_recall >= image_recall, "filtering to images should never hurt: {image_only_recall:.2}");
}
