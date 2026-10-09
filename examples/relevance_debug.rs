//! Prints the similarity curve for a query against an index, to study relevance cutoffs.
//!
//!     cargo run --release --example relevance_debug -- <onnxruntime library> <model dir> <index> <scope> <query>

use std::path::Path;

use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::encoder::{Encoder, GemmaEncoder, GemmaEncoderConfig};
use sems::store::{IndexIdentity, IndexStore, PathScope};

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, library, model_directory, index, scope, query] = arguments.as_slice() else {
        anyhow::bail!("usage: relevance_debug <onnxruntime library> <model dir> <index> <scope> <query>");
    };
    let config = GemmaEncoderConfig::default();
    let identity = IndexIdentity::current(config.identity(), config.dimensions);
    let store = IndexStore::open(Path::new(index), identity)?;
    let runtime = OnnxRuntime::load(Path::new(library))?;
    let model = EmbeddingModel::load(runtime, Path::new(model_directory), ExecutionDevice::Cpu)?;
    let mut encoder = GemmaEncoder::new(model, config)?;

    let scope_root = dunce::canonicalize(scope)?;
    let vector = encoder.encode_query(query)?;
    let (ranked, background) =
        store.nearest_chunks_with_background(&PathScope::new(&scope_root), &vector, 1_000_000, None)?;
    let similarities: Vec<f32> = ranked.iter().map(|(_, similarity)| *similarity).collect();
    let percentile = |fraction: f64| similarities[((similarities.len() - 1) as f64 * (1.0 - fraction)) as usize];
    println!(
        "{} chunks: mean {:.3}, std {:.3}, p50 {:.3}, p90 {:.3}, p99 {:.3}",
        background.count(),
        background.mean(),
        background.standard_deviation(),
        percentile(0.5),
        percentile(0.9),
        percentile(0.99)
    );
    for (rank, (chunk_id, similarity)) in ranked.iter().take(15).enumerate() {
        let chunk = store.chunk(*chunk_id)?;
        let path = chunk.path.strip_prefix(&scope_root).unwrap_or(&chunk.path).display().to_string();
        println!(
            "{:>2}. {similarity:.3}  z {:>4.1}  {path}:{}-{}",
            rank + 1,
            background.standard_score(*similarity),
            chunk.start_line,
            chunk.end_line
        );
    }
    Ok(())
}
