//! Parity tests against embeddings produced by the official sentence-transformers pipeline
//! (`tools/export/make_reference.py`). They need the exported model and an ONNX Runtime library,
//! so they are ignored by default:
//!
//!     SEMS_ONNXRUNTIME=path/to/onnxruntime.dll cargo test --release -- --ignored

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sems::embedding::{EmbeddingModel, OnnxRuntime, PreprocessedImage};
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
        let library = std::env::var_os("SEMS_ONNXRUNTIME").expect("set SEMS_ONNXRUNTIME to the onnxruntime library path");
        OnnxRuntime::load(Path::new(&library)).expect("failed to load ONNX Runtime")
    })
}

fn load_model() -> EmbeddingModel {
    EmbeddingModel::load(runtime(), &models_directory().join("onnx")).expect("failed to load model")
}

fn load_references(kind: &str) -> Vec<Reference> {
    let contents = std::fs::read_to_string(models_directory().join("reference.json")).expect("missing reference.json");
    let references: Vec<Reference> = serde_json::from_str(&contents).expect("invalid reference.json");
    references.into_iter().filter(|reference| reference.kind == kind).collect()
}

fn read_f32_tensor(tensor: &TensorFile) -> Vec<f32> {
    let bytes = std::fs::read(models_directory().join(&tensor.file)).expect("missing tensor file");
    let values: Vec<f32> = bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect();
    assert_eq!(values.len(), tensor.shape.iter().product::<usize>());
    values
}

fn read_i64_tensor(tensor: &TensorFile) -> Vec<i64> {
    let bytes = std::fs::read(models_directory().join(&tensor.file)).expect("missing tensor file");
    let values: Vec<i64> = bytes.chunks_exact(8).map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap())).collect();
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

fn reference_image(reference: &Reference) -> PreprocessedImage {
    let pixel_tensor = reference.pixel_values.as_ref().expect("image reference without pixel values");
    let position_tensor = reference.position_ids.as_ref().expect("image reference without position ids");
    let max_patches = pixel_tensor.shape[1];
    let position_ids = read_i64_tensor(position_tensor);
    let real_patches = position_ids.chunks_exact(2).filter(|position| position[0] != -1).count();
    PreprocessedImage {
        pixel_values: read_f32_tensor(pixel_tensor),
        position_ids,
        max_patches,
        patch_pixels: pixel_tensor.shape[2],
        soft_token_count: real_patches / 9,
    }
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
        assert_matches_reference(&format!("{} (reference preprocessing)", reference.input), &embedding, &reference.embedding);
    }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn image_embedding_matches_reference_end_to_end() {
    let mut model = load_model();
    for reference in load_references("image") {
        let expected_pixels = reference_image(&reference);
        let image = image::open(models_directory().join("fixtures").join(&reference.input)).unwrap();
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
        assert_matches_reference(&format!("{} (rust preprocessing)", reference.input), &embedding, &reference.embedding);
    }
}
