//! Retrieval quality on a small labelled corpus, with the real model. Guards against regressions
//! when changing chunking, prompts, or fusion, and shows which queries fail. Needs the exported
//! model and ONNX Runtime, so it is ignored by default:
//!
//!     SEMS_ONNXRUNTIME=path/to/onnxruntime.dll cargo test --release --test search_quality -- --ignored --nocapture

use std::path::{Path, PathBuf};

use sems::chunking::ChunkingConfig;
use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::encoder::{Encoder, GemmaEncoder, GemmaEncoderConfig};
use sems::indexer::{IndexOptions, SilentProgress, index_directory};
use sems::search::{SearchOptions, search};
use sems::store::{IndexIdentity, IndexStore, PathScope};
use serde::Deserialize;

/// Floors set just below the measured quality of the defaults; raise them when quality improves.
const MINIMUM_RECALL_AT_3: f32 = 0.9;
const MINIMUM_MEAN_RECIPROCAL_RANK: f32 = 0.85;
const KEYWORD_WEIGHTS_COMPARED: [f32; 3] = [0.0, 0.5, 1.0];

#[derive(Deserialize)]
struct LabelledQuery {
    query: String,
    expected: String,
}

#[derive(Debug, Default)]
struct Metrics {
    recall_at_1: f32,
    recall_at_3: f32,
    mean_reciprocal_rank: f32,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn load_encoder() -> GemmaEncoder {
    let library = std::env::var_os("SEMS_ONNXRUNTIME").expect("set SEMS_ONNXRUNTIME to the onnxruntime library path");
    let runtime = OnnxRuntime::load(Path::new(&library)).expect("failed to load ONNX Runtime");
    let model_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models").join("onnx");
    let model = EmbeddingModel::load(runtime, &model_directory, ExecutionDevice::Cpu).expect("failed to load model");
    GemmaEncoder::new(model, GemmaEncoderConfig::default()).unwrap()
}

/// 1-based rank of the first result from `expected`, if it appears at all.
fn rank_of(results: &[sems::search::SearchResult], corpus: &Path, expected: &str) -> Option<usize> {
    let expected = corpus.join(expected);
    results.iter().position(|result| result.path == expected).map(|index| index + 1)
}

fn evaluate(
    store: &IndexStore,
    encoder: &mut GemmaEncoder,
    corpus: &Path,
    queries: &[LabelledQuery],
    keyword_weight: f32,
) -> (Metrics, Vec<String>) {
    let scope = PathScope::new(corpus);
    let options = SearchOptions { limit: 10, keyword_weight, ..SearchOptions::default() };
    let mut metrics = Metrics::default();
    let mut misses = Vec::new();
    for labelled in queries {
        let results = search(store, encoder, &scope, &labelled.query, options).unwrap();
        let rank = rank_of(&results, corpus, &labelled.expected);
        metrics.recall_at_1 += f32::from(rank == Some(1));
        metrics.recall_at_3 += f32::from(rank.is_some_and(|rank| rank <= 3));
        metrics.mean_reciprocal_rank += rank.map_or(0.0, |rank| 1.0 / rank as f32);
        if rank != Some(1) {
            let top = results.first().map(|result| result.path.strip_prefix(corpus).unwrap().display().to_string());
            misses.push(format!("{:?}: expected rank {rank:?}, top was {top:?}", labelled.query));
        }
    }
    let count = queries.len() as f32;
    metrics.recall_at_1 /= count;
    metrics.recall_at_3 /= count;
    metrics.mean_reciprocal_rank /= count;
    (metrics, misses)
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn retrieval_quality_meets_floor() {
    let corpus = dunce::canonicalize(fixtures().join("eval_corpus")).unwrap();
    let queries: Vec<LabelledQuery> =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("eval_queries.json")).unwrap()).unwrap();

    let mut encoder = load_encoder();
    let identity = IndexIdentity {
        encoder: encoder.identity(),
        dimensions: encoder.dimensions(),
        chunker_version: ChunkingConfig::VERSION,
    };
    let mut store = IndexStore::open_in_memory(identity).unwrap();
    index_directory(&mut store, &mut encoder, &corpus, &IndexOptions::default(), &mut SilentProgress).unwrap();

    let default_weight = SearchOptions::default().keyword_weight;
    let mut default_metrics = None;
    for keyword_weight in KEYWORD_WEIGHTS_COMPARED {
        let (metrics, misses) = evaluate(&store, &mut encoder, &corpus, &queries, keyword_weight);
        println!(
            "keyword weight {keyword_weight:.1}: recall@1 {:.2}  recall@3 {:.2}  MRR {:.3}",
            metrics.recall_at_1, metrics.recall_at_3, metrics.mean_reciprocal_rank
        );
        for miss in &misses {
            println!("    {miss}");
        }
        if keyword_weight == default_weight {
            default_metrics = Some(metrics);
        }
    }

    let metrics = default_metrics.expect("the default keyword weight is among those compared");
    assert!(metrics.recall_at_3 >= MINIMUM_RECALL_AT_3, "recall@3 regressed: {metrics:?}");
    assert!(metrics.mean_reciprocal_rank >= MINIMUM_MEAN_RECIPROCAL_RANK, "MRR regressed: {metrics:?}");
}
