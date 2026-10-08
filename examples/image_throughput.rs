//! Where image indexing time goes: decoding vs embedding, per vision token budget.
//!
//!     cargo run --release --example image_throughput -- <onnxruntime library> <model dir> <device> <image dir> [max images]
//!
//! Measured on an RTX 3050 Ti (4 GB): batching images does not help. A 280-token image already
//! saturates the GPU (each vision layer materializes a 12 x 2520 x 2520 attention matrix, 305 MB),
//! so batches of 2 and 4 were 3-8x slower per image and batches of 8 ran out of memory.

use std::path::{Path, PathBuf};
use std::time::Instant;

use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::media::{LoadedImage, load_image};

const VISION_TOKEN_BUDGETS: [usize; 3] = [280, 140, 70];

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, library, model_directory, device, image_directory, rest @ ..] = arguments.as_slice() else {
        anyhow::bail!("usage: image_throughput <onnxruntime library> <model dir> <device> <image dir> [max images]");
    };
    let device: ExecutionDevice = device.parse().map_err(anyhow::Error::msg)?;
    let max_images: usize = rest.first().map_or(Ok(usize::MAX), |value| value.parse())?;
    let mut paths: Vec<PathBuf> =
        std::fs::read_dir(image_directory)?.map(|entry| entry.map(|entry| entry.path())).collect::<Result<_, _>>()?;
    paths.sort();
    paths.truncate(max_images);

    let started = Instant::now();
    let images: Vec<_> = paths
        .iter()
        .map(|path| match load_image(path)? {
            LoadedImage::Image(image) => Ok(image),
            LoadedImage::TooSmall => anyhow::bail!("{} is too small", path.display()),
        })
        .collect::<anyhow::Result<_>>()?;
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0 / images.len() as f64;
    println!("decode (one thread)        {decode_ms:7.0} ms/image");

    let runtime = OnnxRuntime::load(Path::new(library))?;
    let mut model = EmbeddingModel::load(runtime, Path::new(model_directory), device)?;
    for budget in VISION_TOKEN_BUDGETS {
        model.set_vision_token_budget(budget)?;
        model.embed_image(&images[0])?; // warm-up: loads the vision graph and selects kernels for this shape
        let started = Instant::now();
        for image in &images {
            model.embed_image(image)?;
        }
        let per_image = started.elapsed().as_secs_f64() * 1000.0 / images.len() as f64;
        println!("embed, {budget:>4} tokens/image  {per_image:7.0} ms/image");
    }
    Ok(())
}
