//! Measures what a `sems "query"` invocation would pay: runtime + model load, then query embedding.
//!
//!     cargo run --release --example latency -- <onnxruntime.dll> <model dir> <cpu|cuda|directml> [image]

use std::path::Path;
use std::time::Instant;

use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, library, model_directory, device, rest @ ..] = arguments.as_slice() else {
        anyhow::bail!("usage: latency <onnxruntime library> <model directory> <cpu|cuda|directml> [image]");
    };
    let device: ExecutionDevice = device.parse().map_err(anyhow::Error::msg)?;
    let process_start = Instant::now();

    let started = Instant::now();
    let runtime = OnnxRuntime::load(Path::new(library))?;
    println!("load onnx runtime     {:>7.0} ms", started.elapsed().as_secs_f64() * 1000.0);

    let started = Instant::now();
    let mut model = EmbeddingModel::load(runtime, Path::new(model_directory), device)?;
    println!("load text model       {:>7.0} ms", started.elapsed().as_secs_f64() * 1000.0);

    for label in ["first query", "second query"] {
        let started = Instant::now();
        model.embed_texts(&["task: search result | query: photos of the beach at sunset"])?;
        println!("{label:<21} {:>7.0} ms", started.elapsed().as_secs_f64() * 1000.0);
    }
    println!(
        "total to first result {:>7.0} ms (excluding second query)",
        process_start.elapsed().as_secs_f64() * 1000.0
    );

    let documents: Vec<String> = (0..8)
        .map(|index| format!("title: none | text: {}", "fn handle_request() { retry(); } ".repeat(8 + index)))
        .collect();
    let document_references: Vec<&str> = documents.iter().map(String::as_str).collect();
    let started = Instant::now();
    model.embed_texts(&document_references)?;
    let elapsed = started.elapsed().as_secs_f64();
    println!("index batch of 8 chunks  {:>7.0} ms ({:.1} chunks/s)", elapsed * 1000.0, 8.0 / elapsed);

    if let Some(image_path) = rest.first() {
        let image = image::open(image_path)?;
        for label in ["image (loads vision)", "image (warm)"] {
            let started = Instant::now();
            model.embed_image(&image)?;
            println!("{label:<21} {:>7.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    Ok(())
}
