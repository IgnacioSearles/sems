//! Retrieval quality on a small labelled corpus, with the real model. Guards against regressions
//! when changing chunking, prompts, or fusion, and shows which queries fail. Needs the exported
//! model and ONNX Runtime, so it is ignored by default:
//!
//!     SEMS_ONNXRUNTIME=path/to/onnxruntime.dll cargo test --release --test search_quality -- --ignored --nocapture
//!
//! Two questions are measured:
//! - retrieval: is the right file ranked first? (`eval_queries.json`)
//! - localization: does the top result point at the right lines? (`eval_localization.json`, where
//!   long multi-section files make chunk granularity matter)

use std::path::{Path, PathBuf};

use sems::chunking::ChunkingConfig;
use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::encoder::{Encoder, GemmaEncoder, GemmaEncoderConfig};
use sems::indexer::{IndexOptions, SilentProgress, index_directory};
use sems::search::{SearchOptions, SearchResult, search};
use sems::store::{IndexIdentity, IndexStore, PathScope};
use serde::Deserialize;

/// Floors set just below the measured quality of the defaults; raise them when quality improves.
const MINIMUM_RECALL_AT_1: f32 = 0.9;
const MINIMUM_LOCATED_AT_1: f32 = 0.9;

/// (min_lines, max_lines) alternatives reported alongside the default, to inform future tuning.
const COMPARED_CHUNKINGS: [(usize, usize); 4] = [(4, 30), (4, 20), (4, 45), (8, 30)];

#[derive(Deserialize)]
struct LabelledQuery {
    query: String,
    expected: String,
    /// For localization queries: a line the top result must contain.
    expected_line: Option<usize>,
}

#[derive(Debug, Default)]
struct Metrics {
    recall_at_1: f32,
    located_at_1: f32,
    mean_top_span_lines: f32,
}

struct Evaluation<'a> {
    store: &'a IndexStore,
    encoder: &'a mut GemmaEncoder,
    corpus: &'a Path,
}

impl Evaluation<'_> {
    fn top_result(&mut self, query: &str) -> Option<SearchResult> {
        let scope = PathScope::new(self.corpus);
        let options = SearchOptions { limit: 1, ..SearchOptions::default() };
        search(self.store, self.encoder, &scope, query, options).unwrap().into_iter().next()
    }

    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(self.corpus).unwrap().to_string_lossy().replace('\\', "/")
    }

    /// Fraction of queries whose top result is the expected file (and, if labelled, line).
    fn score(&mut self, queries: &[LabelledQuery], misses: &mut Vec<String>) -> (f32, f32) {
        let mut hits = 0.0;
        let mut span_lines = 0.0;
        for labelled in queries {
            let top = self.top_result(&labelled.query);
            let hit = top.as_ref().is_some_and(|result| {
                self.relative(&result.path) == labelled.expected
                    && labelled.expected_line.is_none_or(|line| (result.start_line..=result.end_line).contains(&line))
            });
            if let Some(result) = &top {
                span_lines += (result.end_line - result.start_line + 1) as f32;
            }
            if hit {
                hits += 1.0;
            } else {
                let top = top
                    .map(|result| format!("{}:{}-{}", self.relative(&result.path), result.start_line, result.end_line));
                let wanted = labelled.expected_line.map_or(String::new(), |line| format!(":{line}"));
                misses.push(format!("{:?}: wanted {}{wanted}, got {top:?}", labelled.query, labelled.expected));
            }
        }
        let count = queries.len() as f32;
        (hits / count, span_lines / count)
    }
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn load_queries(file_name: &str) -> Vec<LabelledQuery> {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(file_name)).unwrap()).unwrap()
}

fn load_encoder() -> GemmaEncoder {
    let library = std::env::var_os("SEMS_ONNXRUNTIME").expect("set SEMS_ONNXRUNTIME to the onnxruntime library path");
    let runtime = OnnxRuntime::load(Path::new(&library)).expect("failed to load ONNX Runtime");
    let model_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models").join("onnx");
    let model = EmbeddingModel::load(runtime, &model_directory, ExecutionDevice::Cpu).expect("failed to load model");
    GemmaEncoder::new(model, GemmaEncoderConfig::default()).unwrap()
}

fn evaluate_chunking(encoder: &mut GemmaEncoder, corpus: &Path, chunking: ChunkingConfig) -> (Metrics, Vec<String>) {
    let identity = IndexIdentity {
        encoder: encoder.identity(),
        dimensions: encoder.dimensions(),
        chunker_version: ChunkingConfig::VERSION,
    };
    let mut store = IndexStore::open_in_memory(identity).unwrap();
    let options = IndexOptions { chunking, ..IndexOptions::default() };
    index_directory(&mut store, encoder, corpus, &options, &mut SilentProgress).unwrap();

    let mut evaluation = Evaluation { store: &store, encoder, corpus };
    let mut misses = Vec::new();
    let (recall_at_1, _) = evaluation.score(&load_queries("eval_queries.json"), &mut misses);
    let (located_at_1, mean_top_span_lines) = evaluation.score(&load_queries("eval_localization.json"), &mut misses);
    (Metrics { recall_at_1, located_at_1, mean_top_span_lines }, misses)
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn retrieval_and_localization_meet_floor() {
    let corpus = dunce::canonicalize(fixtures().join("eval_corpus")).unwrap();
    let mut encoder = load_encoder();
    let default_chunking = ChunkingConfig::default();

    let mut default_metrics = None;
    for (min_lines, max_lines) in COMPARED_CHUNKINGS {
        let chunking = ChunkingConfig { min_lines, max_lines, ..default_chunking };
        let (metrics, misses) = evaluate_chunking(&mut encoder, &corpus, chunking);
        let is_default = min_lines == default_chunking.min_lines && max_lines == default_chunking.max_lines;
        println!(
            "chunks of {min_lines}-{max_lines:>2} lines{}: recall@1 {:.2}  located@1 {:.2}  mean span {:.1} lines",
            if is_default { " [default]" } else { "" },
            metrics.recall_at_1,
            metrics.located_at_1,
            metrics.mean_top_span_lines
        );
        for miss in &misses {
            println!("    {miss}");
        }
        if is_default {
            default_metrics = Some(metrics);
        }
    }

    let metrics = default_metrics.expect("the default chunking is among those compared");
    assert!(metrics.recall_at_1 >= MINIMUM_RECALL_AT_1, "retrieval regressed: {metrics:?}");
    assert!(metrics.located_at_1 >= MINIMUM_LOCATED_AT_1, "localization regressed: {metrics:?}");
}
