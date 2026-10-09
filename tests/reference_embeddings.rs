//! Parity tests against embeddings produced by the official sentence-transformers pipeline
//! (`tools/export/make_reference.py`). They need the exported model and an ONNX Runtime library,
//! so they are ignored by default:
//!
//!     SEMS_ONNXRUNTIME=path/to/onnxruntime.dll cargo test --release -- --ignored
//!
//! Set SEMS_DEVICE=cuda or SEMS_DEVICE=directml (with the matching ONNX Runtime build) to verify
//! GPU execution against the same references.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sems::av::{Ffmpeg, SAMPLE_RATE};
use sems::embedding::{
    AudioFeatureExtractor, AudioFeatures, EmbeddingModel, ExecutionDevice, OnnxRuntime, PreprocessedImage,
};
use serde::Deserialize;

/// Same bar as the Python export check: anything lower indicates a real numerical divergence.
const MINIMUM_COSINE_SIMILARITY: f32 = 0.9999;

#[derive(Deserialize)]
struct Reference {
    kind: String,
    input: String,
    input_ids: Vec<i64>,
    embedding: Vec<f32>,
    pixel_values: Option<TensorFile>,
    position_ids: Option<TensorFile>,
    input_features: Option<TensorFile>,
    input_features_mask: Option<TensorFile>,
    seconds_per_frame: Option<u32>,
    frame_size: Option<(u32, u32)>,
}

#[derive(Deserialize)]
struct TensorFile {
    file: String,
    shape: Vec<usize>,
}

fn models_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn runtime() -> OnnxRuntime {
    static RUNTIME: OnceLock<OnnxRuntime> = OnceLock::new();
    *RUNTIME.get_or_init(|| {
        let library =
            std::env::var_os("SEMS_ONNXRUNTIME").expect("set SEMS_ONNXRUNTIME to the onnxruntime library path");
        OnnxRuntime::load(Path::new(&library)).expect("failed to load ONNX Runtime")
    })
}

fn device() -> ExecutionDevice {
    std::env::var("SEMS_DEVICE").map_or(ExecutionDevice::Cpu, |value| value.parse().expect("invalid SEMS_DEVICE"))
}

fn load_model() -> EmbeddingModel {
    EmbeddingModel::load(runtime(), &models_directory().join("onnx"), device()).expect("failed to load model")
}

fn load_references(kind: &str) -> Vec<Reference> {
    let contents = std::fs::read_to_string(models_directory().join("reference.json")).expect("missing reference.json");
    let references: Vec<Reference> = serde_json::from_str(&contents).expect("invalid reference.json");
    references.into_iter().filter(|reference| reference.kind == kind).collect()
}

fn read_f32_tensor(tensor: &TensorFile) -> Vec<f32> {
    let bytes = std::fs::read(models_directory().join(&tensor.file)).expect("missing tensor file");
    let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&chunk| f32::from_le_bytes(chunk)).collect();
    assert_eq!(values.len(), tensor.shape.iter().product::<usize>());
    values
}

fn read_i64_tensor(tensor: &TensorFile) -> Vec<i64> {
    let bytes = std::fs::read(models_directory().join(&tensor.file)).expect("missing tensor file");
    let values: Vec<i64> = bytes.as_chunks::<8>().0.iter().map(|&chunk| i64::from_le_bytes(chunk)).collect();
    assert_eq!(values.len(), tensor.shape.iter().product::<usize>());
    values
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn assert_matches_reference(label: &str, actual: &[f32], expected: &[f32]) {
    let similarity = cosine_similarity(actual, expected);
    println!("{label}: cosine similarity {similarity:.6}");
    assert!(similarity >= MINIMUM_COSINE_SIMILARITY, "{label} diverges from reference: {similarity:.6}");
}

/// Reference preprocessing as one `PreprocessedImage` per frame (images have a single frame).
fn reference_frames(reference: &Reference) -> Vec<PreprocessedImage> {
    let pixel_tensor = reference.pixel_values.as_ref().expect("reference without pixel values");
    let position_tensor = reference.position_ids.as_ref().expect("reference without position ids");
    let (frames, max_patches, patch_pixels) = (pixel_tensor.shape[0], pixel_tensor.shape[1], pixel_tensor.shape[2]);
    let pixels = read_f32_tensor(pixel_tensor);
    let positions = read_i64_tensor(position_tensor);
    (0..frames)
        .map(|frame| {
            let position_ids = positions[frame * max_patches * 2..(frame + 1) * max_patches * 2].to_vec();
            let real_patches = position_ids.as_chunks::<2>().0.iter().filter(|[x, _]| *x != -1).count();
            PreprocessedImage {
                pixel_values: pixels[frame * max_patches * patch_pixels..(frame + 1) * max_patches * patch_pixels]
                    .to_vec(),
                position_ids,
                max_patches,
                patch_pixels,
                soft_token_count: real_patches / 9,
            }
        })
        .collect()
}

fn reference_image(reference: &Reference) -> PreprocessedImage {
    reference_frames(reference).remove(0)
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn tokenizer_matches_reference_token_ids() {
    let model = load_model();
    for reference in load_references("text") {
        let token_ids = model.tokenize(&[reference.input.as_str()]).unwrap().remove(0);
        assert_eq!(token_ids, reference.input_ids, "token ids differ for {:?}", reference.input);
    }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn text_embeddings_match_reference_in_one_padded_batch() {
    let mut model = load_model();
    let references = load_references("text");
    let texts: Vec<&str> = references.iter().map(|reference| reference.input.as_str()).collect();
    let embeddings = model.embed_texts(&texts).unwrap();
    for (reference, embedding) in references.iter().zip(&embeddings) {
        let label: String = reference.input.chars().take(40).collect();
        assert_matches_reference(&label, embedding, &reference.embedding);
    }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn image_embedding_matches_reference_given_reference_preprocessing() {
    let mut model = load_model();
    for reference in load_references("image") {
        let image = reference_image(&reference);
        assert_eq!(model.image_token_ids(image.soft_token_count), reference.input_ids);
        let embedding = model.embed_preprocessed_image(&image).unwrap();
        assert_matches_reference(
            &format!("{} (reference preprocessing)", reference.input),
            &embedding,
            &reference.embedding,
        );
    }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn image_embedding_matches_reference_end_to_end() {
    let mut model = load_model();
    for reference in load_references("image") {
        let expected_pixels = reference_image(&reference);
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images").join(&reference.input);
        let image = image::open(image_path).unwrap();
        let ours = sems::embedding::ImagePreprocessor::new(&model.config().vision_config).preprocess(&image).unwrap();

        assert_eq!(ours.position_ids, expected_pixels.position_ids, "patch layout differs");
        let max_pixel_difference = ours
            .pixel_values
            .iter()
            .zip(&expected_pixels.pixel_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("max pixel difference after resize: {:.1}/255", max_pixel_difference * 255.0);
        assert!(max_pixel_difference * 255.0 <= 1.0 + 1e-3, "resize diverges from upstream bicubic");

        let embedding = model.embed_image(&image).unwrap();
        assert_matches_reference(
            &format!("{} (rust preprocessing)", reference.input),
            &embedding,
            &reference.embedding,
        );
    }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn batched_images_match_individual_embeddings() {
    let mut model = load_model();
    let reference = load_references("image").remove(0);
    let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images").join(&reference.input);
    let beach = image::open(image_path).unwrap();
    // Different aspect ratios give different soft-token counts, so the batch needs padding.
    let portrait = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(300, 800, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
    }));
    let images = [beach.clone(), portrait.clone(), beach];

    let batched = model.embed_images(&images).unwrap();
    for (index, image) in images.iter().enumerate() {
        let single = model.embed_image(image).unwrap();
        assert_matches_reference(&format!("batch item {index} vs single"), &batched[index], &single);
    }
    assert_matches_reference("batched beach vs reference", &batched[0], &reference.embedding);
}

fn ffmpeg() -> Ffmpeg {
    Ffmpeg::locate().expect("audio and video parity tests need ffmpeg on PATH or SEMS_FFMPEG")
}

fn repository_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// The first 30 s of a file, decoded like the reference (the model truncates longer audio).
fn first_audio_window(path: &Path) -> Vec<f32> {
    let window = 30 * SAMPLE_RATE;
    let mut samples = None;
    ffmpeg()
        .for_each_audio_window(path, window, window, |_, window_samples| {
            samples.get_or_insert_with(|| window_samples.to_vec());
            Ok(())
        })
        .unwrap();
    samples.expect("file has audio")
}

fn reference_audio_features(reference: &Reference) -> AudioFeatures {
    let features = reference.input_features.as_ref().expect("audio reference without features");
    let mask = reference.input_features_mask.as_ref().expect("audio reference without mask");
    AudioFeatures { features: read_f32_tensor(features), mask: read_i64_tensor(mask), frames: features.shape[1] }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn audio_features_match_reference() {
    let extractor = AudioFeatureExtractor::new();
    for reference in load_references("audio") {
        let expected = reference_audio_features(&reference);
        let ours = extractor.extract(&first_audio_window(&repository_path(&reference.input)));
        assert_eq!(ours.frames, expected.frames, "{}: frame count", reference.input);
        assert_eq!(ours.mask, expected.mask, "{}: mask", reference.input);
        let largest_difference = ours
            .features
            .iter()
            .zip(&expected.features)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        println!("{}: largest log-mel difference {largest_difference:.2e}", reference.input);
        assert!(largest_difference < 1e-4, "{}: log-mel features diverge by {largest_difference}", reference.input);
    }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn audio_embeddings_match_reference() {
    let mut model = load_model();
    for reference in load_references("audio") {
        let from_reference_features = model.embed_audio_features(&reference_audio_features(&reference)).unwrap();
        assert_matches_reference(
            &format!("{} (reference features)", reference.input),
            &from_reference_features,
            &reference.embedding,
        );
        let end_to_end = model.embed_audio(&first_audio_window(&repository_path(&reference.input))).unwrap();
        assert_matches_reference(&format!("{} (rust features)", reference.input), &end_to_end, &reference.embedding);
    }
}

/// The reference used the video processor's default of 140 tokens per frame (sems' budget too).
const VIDEO_TOKEN_BUDGET: usize = 140;

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn video_embedding_matches_reference_given_reference_preprocessing() {
    let mut model = load_model();
    for reference in load_references("video") {
        let embedding = model.embed_preprocessed_video(&reference_frames(&reference)).unwrap();
        assert_matches_reference(
            &format!("{} (reference preprocessing)", reference.input),
            &embedding,
            &reference.embedding,
        );
    }
}

/// End to end the only difference is resizing: within 1/255 of upstream per value, but photographic
/// frames have far more edge pixels than the synthetic test image, so 5-14% of values differ by
/// that 1/255 and three frames measured 0.99984. The model itself is held to the strict bar above.
const MINIMUM_VIDEO_END_TO_END_SIMILARITY: f32 = 0.9995;

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn video_embedding_matches_reference_end_to_end() {
    let mut model = load_model();
    model.set_vision_token_budget(VIDEO_TOKEN_BUDGET).unwrap();
    let preprocessor =
        sems::embedding::ImagePreprocessor::with_token_budget(&model.config().vision_config, VIDEO_TOKEN_BUDGET)
            .unwrap();
    for reference in load_references("video") {
        let (width, height) = reference.frame_size.expect("video reference without frame size");
        let mut frames = Vec::new();
        ffmpeg()
            .for_each_video_frame(
                &repository_path(&reference.input),
                reference.seconds_per_frame.expect("video reference without frame interval"),
                width,
                height,
                |_, frame| {
                    frames.push(image::DynamicImage::ImageRgb8(frame));
                    Ok(())
                },
            )
            .unwrap();

        for (ours, theirs) in frames.iter().zip(reference_frames(&reference)) {
            let ours = preprocessor.preprocess(ours).unwrap();
            assert_eq!(ours.position_ids, theirs.position_ids, "patch layout differs");
            let largest =
                ours.pixel_values.iter().zip(&theirs.pixel_values).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            assert!(largest * 255.0 <= 1.0 + 1e-3, "frame resize diverges from upstream bicubic: {largest}");
        }

        let embedding = model.embed_video(&frames).unwrap();
        let similarity = cosine_similarity(&embedding, &reference.embedding);
        println!(
            "{} ({} frames, rust preprocessing): cosine similarity {similarity:.6}",
            reference.input,
            frames.len()
        );
        assert!(similarity >= MINIMUM_VIDEO_END_TO_END_SIMILARITY, "video diverges from reference: {similarity:.6}");
    }
}
