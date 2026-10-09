//! Cost of embedding audio windows and video clips, to size indexing segments.
//!
//!     cargo run --release --example media_throughput -- <onnxruntime library> <model dir> <device>

use std::path::Path;
use std::time::Instant;

use sems::av::{Ffmpeg, SAMPLE_RATE};
use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};

const AUDIO: &str = "tests/fixtures/eval_corpus/audio/memo_0003.m4a";
const VIDEO: &str = "tests/fixtures/eval_corpus/videos/clip_0001.mp4";
const REPETITIONS: usize = 3;

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, library, model_directory, device] = arguments.as_slice() else {
        anyhow::bail!("usage: media_throughput <onnxruntime library> <model dir> <cpu|cuda|directml>");
    };
    let device: ExecutionDevice = device.parse().map_err(anyhow::Error::msg)?;
    let ffmpeg = Ffmpeg::locate().ok_or_else(|| anyhow::anyhow!("ffmpeg not found"))?;
    let runtime = OnnxRuntime::load(Path::new(library))?;
    let mut model = EmbeddingModel::load(runtime, Path::new(model_directory), device)?;

    let mut frames = Vec::new();
    ffmpeg.for_each_video_frame(Path::new(VIDEO), 2, 480, 320, |_, frame| {
        frames.push(image::DynamicImage::ImageRgb8(frame));
        Ok(())
    })?;
    for budget in [140, 70] {
        model.set_vision_token_budget(budget)?;
        for frame_count in [3, 5] {
            let clip = &frames[..frame_count];
            model.embed_video(clip)?; // warm-up
            let started = Instant::now();
            for _ in 0..REPETITIONS {
                model.embed_video(clip)?;
            }
            println!("video, {frame_count} frames x {budget:>3} tokens  {:7.0} ms", milliseconds(started, REPETITIONS));
        }
    }
    for seconds in [10, 30] {
        let mut window = None;
        ffmpeg.for_each_audio_window(
            Path::new(AUDIO),
            seconds * SAMPLE_RATE,
            seconds * SAMPLE_RATE,
            |_, samples| {
                window.get_or_insert_with(|| samples.to_vec());
                Ok(())
            },
        )?;
        let window = window.expect("fixture has audio");
        model.embed_audio(&window)?; // warm-up: loads the audio graph
        let started = Instant::now();
        for _ in 0..REPETITIONS {
            model.embed_audio(&window)?;
        }
        println!("audio, {seconds:>2} s window          {:7.0} ms", milliseconds(started, REPETITIONS));
    }

    Ok(())
}

fn milliseconds(started: Instant, repetitions: usize) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0 / repetitions as f64
}
