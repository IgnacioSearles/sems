//! EmbeddingGemma 2 inference on ONNX Runtime.
//!
//! The model is split into three graphs (see `tools/export/export_onnx.py`):
//! token ids -> token embeddings, image patches -> soft tokens, and embeddings -> pooled vector.
//! This module tokenizes, splices image soft tokens into placeholder positions, and runs them.

mod config;
mod image_preprocessing;
mod onnx;

use std::path::{Path, PathBuf};

use image::DynamicImage;
use tokenizers::Tokenizer;

pub use config::ModelConfig;
pub use image_preprocessing::{ImagePreprocessor, PreprocessedImage, SUPPORTED_VISION_TOKEN_BUDGETS};
pub use onnx::{ExecutionDevice, OnnxRuntime};
use onnx::{GraphSession, TensorF32};

/// A unit-length embedding vector.
pub type Embedding = Vec<f32>;

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("failed to read {path}")]
    ReadFile { path: PathBuf, source: std::io::Error },
    #[error("invalid model config {path}")]
    InvalidConfig { path: PathBuf, source: serde_json::Error },
    #[error("failed to load tokenizer {path}: {message}")]
    Tokenizer { path: PathBuf, message: String },
    #[error("tokenization failed: {0}")]
    Tokenization(String),
    #[error("invalid image: {0}")]
    InvalidImage(String),
    #[error("failed to load ONNX Runtime from {path}: {message}")]
    RuntimeLibrary { path: PathBuf, message: String },
    #[error("onnx runtime error in {graph}: {source}")]
    Onnx { graph: &'static str, source: ort::Error },
    #[error("unexpected output from {graph}: {message}")]
    UnexpectedOutput { graph: &'static str, message: String },
}

/// Locations of the exported runtime artifacts inside a model directory.
struct ModelFiles {
    directory: PathBuf,
}

impl ModelFiles {
    fn path(&self, file_name: &str) -> PathBuf {
        self.directory.join(file_name)
    }
}

pub struct EmbeddingModel {
    runtime: OnnxRuntime,
    device: ExecutionDevice,
    files: ModelFiles,
    config: ModelConfig,
    tokenizer: Tokenizer,
    token_embedder: GraphSession,
    text_encoder: GraphSession,
    /// Loaded on first image request so text-only use (every search query) never pays for it.
    vision_encoder: Option<GraphSession>,
    image_preprocessor: ImagePreprocessor,
}

impl EmbeddingModel {
    /// Loads the tokenizer and text graphs concurrently: they are independent, and parsing the
    /// 32 MB tokenizer.json takes about as long as creating the text encoder session.
    ///
    /// The token embedder always runs on CPU: it is a table lookup, and placing its 512 MB table
    /// on a 4 GB GPU would only take memory from the encoders.
    pub fn load(runtime: OnnxRuntime, model_directory: &Path, device: ExecutionDevice) -> Result<Self, EmbeddingError> {
        let files = ModelFiles { directory: model_directory.to_path_buf() };
        let config = ModelConfig::load(&files.path("config.json"))?;
        let (tokenizer, token_embedder, text_encoder) = std::thread::scope(|scope| {
            let tokenizer = scope.spawn(|| load_tokenizer(&files.path("tokenizer.json")));
            let text_encoder =
                scope.spawn(|| GraphSession::load(runtime, "text_encoder", &files.path("text_encoder.onnx"), device));
            let token_embedder =
                GraphSession::load(runtime, "token_embedder", &files.path("token_embedder.onnx"), ExecutionDevice::Cpu);
            (join_propagating_panic(tokenizer), token_embedder, join_propagating_panic(text_encoder))
        });
        Ok(Self {
            tokenizer: tokenizer?,
            token_embedder: token_embedder?,
            text_encoder: text_encoder?,
            vision_encoder: None,
            image_preprocessor: ImagePreprocessor::new(&config.vision_config),
            runtime,
            device,
            files,
            config,
        })
    }

    /// Sets how many soft tokens each image is reduced to (see [`SUPPORTED_VISION_TOKEN_BUDGETS`]).
    pub fn set_vision_token_budget(&mut self, max_soft_tokens: usize) -> Result<(), EmbeddingError> {
        self.image_preprocessor = ImagePreprocessor::with_token_budget(&self.config.vision_config, max_soft_tokens)?;
        Ok(())
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn dimensions(&self) -> usize {
        self.config.text_config.embedding_dim
    }

    /// Embeds texts as-is; callers are responsible for task prefixes such as `task: search result | query: `.
    pub fn embed_texts(&mut self, texts: &[&str]) -> Result<Vec<Embedding>, EmbeddingError> {
        let token_ids = self.tokenize(texts)?;
        self.embed_token_sequences(&token_ids)
    }

    /// Embeds already-tokenized sequences (from [`Self::tokenize`]) as one padded batch. Lets
    /// callers tokenize once to plan batch sizes before running inference.
    pub fn embed_token_sequences(&mut self, sequences: &[Vec<i64>]) -> Result<Vec<Embedding>, EmbeddingError> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let batch = PaddedBatch::new(sequences, self.config.text_config.pad_token_id);
        let inputs_embeds = self.embed_tokens(&batch)?;
        self.encode(inputs_embeds, &batch)
    }

    pub fn embed_image(&mut self, image: &DynamicImage) -> Result<Embedding, EmbeddingError> {
        Ok(self.embed_images(std::slice::from_ref(image))?.remove(0))
    }

    /// Embeds images in one inference call. Preprocessing (resize, patchify) runs on one thread
    /// per image; batching amortizes per-call overhead and keeps a GPU busy.
    pub fn embed_images(&mut self, images: &[DynamicImage]) -> Result<Vec<Embedding>, EmbeddingError> {
        let preprocessor = &self.image_preprocessor;
        let preprocessed: Vec<PreprocessedImage> = std::thread::scope(|scope| {
            let handles: Vec<_> =
                images.iter().map(|image| scope.spawn(move || preprocessor.preprocess(image))).collect();
            handles.into_iter().map(join_propagating_panic).collect::<Result<_, _>>()
        })?;
        self.embed_preprocessed_images(&preprocessed)
    }

    /// Exposed separately so tests can feed reference preprocessing and isolate the model stages.
    pub fn embed_preprocessed_image(&mut self, image: &PreprocessedImage) -> Result<Embedding, EmbeddingError> {
        Ok(self.embed_preprocessed_images(std::slice::from_ref(image))?.remove(0))
    }

    /// Batched form of [`Self::embed_preprocessed_image`]. Rows of the padded text batch hold one
    /// image each; row-major, their placeholders appear in image order, which is also the order of
    /// the vision encoder's flattened soft tokens, so one splice fills the whole batch.
    pub fn embed_preprocessed_images(
        &mut self,
        images: &[PreprocessedImage],
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let sequences: Vec<Vec<i64>> =
            images.iter().map(|image| self.image_token_ids(image.soft_token_count)).collect();
        let batch = PaddedBatch::new(&sequences, self.config.text_config.pad_token_id);
        let mut inputs_embeds = self.embed_tokens(&batch)?;
        let soft_tokens = self.encode_vision(images)?;
        splice_soft_tokens(&mut inputs_embeds, &batch.token_ids, self.config.image_token_id, &soft_tokens)?;
        self.encode(inputs_embeds, &batch)
    }

    /// `[BOS] [BOI] <image>×n [EOI] [EOS]`, exactly as the upstream processor builds it.
    pub fn image_token_ids(&self, soft_token_count: usize) -> Vec<i64> {
        let text = &self.config.text_config;
        let mut ids = Vec::with_capacity(soft_token_count + 4);
        ids.extend([text.bos_token_id, self.config.boi_token_id]);
        ids.extend(std::iter::repeat_n(self.config.image_token_id, soft_token_count));
        ids.extend([self.config.eoi_token_id, text.eos_token_id]);
        ids
    }

    pub fn tokenize(&self, texts: &[&str]) -> Result<Vec<Vec<i64>>, EmbeddingError> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| EmbeddingError::Tokenization(error.to_string()))?;
        Ok(encodings.iter().map(|encoding| encoding.get_ids().iter().map(|&id| i64::from(id)).collect()).collect())
    }

    fn embed_tokens(&mut self, batch: &PaddedBatch) -> Result<TensorF32, EmbeddingError> {
        let input_ids = onnx::tensor_i64(vec![batch.rows, batch.columns], batch.token_ids.clone())?;
        self.token_embedder.run_single(vec![("input_ids", input_ids)], "inputs_embeds")
    }

    /// Runs the vision encoder over a batch; returns every image's soft tokens, flattened in order.
    fn encode_vision(&mut self, images: &[PreprocessedImage]) -> Result<TensorF32, EmbeddingError> {
        let (max_patches, patch_pixels) = (images[0].max_patches, images[0].patch_pixels);
        if images.iter().any(|image| image.max_patches != max_patches || image.patch_pixels != patch_pixels) {
            return Err(EmbeddingError::InvalidImage("images in one batch must share a patch budget".into()));
        }
        if self.vision_encoder.is_none() {
            let path = self.files.path("vision_encoder.onnx");
            self.vision_encoder = Some(GraphSession::load(self.runtime, "vision_encoder", &path, self.device)?);
        }
        let vision_encoder = self.vision_encoder.as_mut().expect("vision encoder was loaded above");
        let pixel_values: Vec<f32> = images.iter().flat_map(|image| image.pixel_values.iter().copied()).collect();
        let position_ids: Vec<i64> = images.iter().flat_map(|image| image.position_ids.iter().copied()).collect();
        let pixel_values = onnx::tensor_f32(vec![images.len(), max_patches, patch_pixels], pixel_values)?;
        let position_ids = onnx::tensor_i64(vec![images.len(), max_patches, 2], position_ids)?;
        let soft_tokens = vision_encoder
            .run_single(vec![("pixel_values", pixel_values), ("position_ids", position_ids)], "soft_tokens")?;
        let expected: usize = images.iter().map(|image| image.soft_token_count).sum();
        if soft_tokens.shape.first() != Some(&expected) {
            return Err(EmbeddingError::UnexpectedOutput {
                graph: "vision_encoder",
                message: format!("expected {expected} soft tokens, got shape {:?}", soft_tokens.shape),
            });
        }
        Ok(soft_tokens)
    }

    fn encode(&mut self, inputs_embeds: TensorF32, batch: &PaddedBatch) -> Result<Vec<Embedding>, EmbeddingError> {
        let embeds = onnx::tensor_f32(inputs_embeds.shape, inputs_embeds.data)?;
        let attention_mask = onnx::tensor_i64(vec![batch.rows, batch.columns], batch.attention_mask.clone())?;
        let pooled = self
            .text_encoder
            .run_single(vec![("inputs_embeds", embeds), ("attention_mask", attention_mask)], "embedding")?;
        let dimensions = self.dimensions();
        if pooled.shape != [batch.rows, dimensions] {
            return Err(EmbeddingError::UnexpectedOutput {
                graph: "text_encoder",
                message: format!("expected [{}, {dimensions}], got {:?}", batch.rows, pooled.shape),
            });
        }
        Ok(pooled.data.chunks_exact(dimensions).map(<[f32]>::to_vec).collect())
    }
}

fn join_propagating_panic<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, EmbeddingError> {
    Tokenizer::from_file(path)
        .map_err(|error| EmbeddingError::Tokenizer { path: path.to_path_buf(), message: error.to_string() })
}

/// Right-padded token ids with the matching attention mask, flattened row-major.
struct PaddedBatch {
    rows: usize,
    columns: usize,
    token_ids: Vec<i64>,
    attention_mask: Vec<i64>,
}

impl PaddedBatch {
    fn new(sequences: &[Vec<i64>], pad_token_id: i64) -> Self {
        let columns = sequences.iter().map(Vec::len).max().unwrap_or(0);
        let mut token_ids = Vec::with_capacity(sequences.len() * columns);
        let mut attention_mask = Vec::with_capacity(sequences.len() * columns);
        for sequence in sequences {
            let padding = columns - sequence.len();
            token_ids.extend(sequence.iter().copied().chain(std::iter::repeat_n(pad_token_id, padding)));
            attention_mask.extend(std::iter::repeat_n(1, sequence.len()).chain(std::iter::repeat_n(0, padding)));
        }
        Self { rows: sequences.len(), columns, token_ids, attention_mask }
    }
}

/// Overwrites the embedding of each image placeholder token, in order, with a vision soft token.
fn splice_soft_tokens(
    inputs_embeds: &mut TensorF32,
    token_ids: &[i64],
    image_token_id: i64,
    soft_tokens: &TensorF32,
) -> Result<(), EmbeddingError> {
    let hidden_size = *inputs_embeds.shape.last().unwrap_or(&0);
    let placeholder_positions: Vec<usize> =
        token_ids.iter().enumerate().filter(|(_, id)| **id == image_token_id).map(|(position, _)| position).collect();
    if placeholder_positions.len() * hidden_size != soft_tokens.data.len() {
        return Err(EmbeddingError::UnexpectedOutput {
            graph: "vision_encoder",
            message: format!(
                "{} image placeholders but {} soft token values",
                placeholder_positions.len(),
                soft_tokens.data.len()
            ),
        });
    }
    for (soft_token, position) in soft_tokens.data.chunks_exact(hidden_size).zip(placeholder_positions) {
        inputs_embeds.data[position * hidden_size..(position + 1) * hidden_size].copy_from_slice(soft_token);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_batch_right_pads_and_masks() {
        let batch = PaddedBatch::new(&[vec![5, 6, 7], vec![8]], 0);
        assert_eq!((batch.rows, batch.columns), (2, 3));
        assert_eq!(batch.token_ids, [5, 6, 7, 8, 0, 0]);
        assert_eq!(batch.attention_mask, [1, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn splice_replaces_only_placeholders_in_order() {
        let mut embeds = TensorF32 { shape: vec![1, 4, 2], data: vec![0.0; 8] };
        let soft_tokens = TensorF32 { shape: vec![2, 2], data: vec![1.0, 1.0, 2.0, 2.0] };
        splice_soft_tokens(&mut embeds, &[10, 99, 99, 11], 99, &soft_tokens).unwrap();
        assert_eq!(embeds.data, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn splice_rejects_count_mismatch() {
        let mut embeds = TensorF32 { shape: vec![1, 2, 2], data: vec![0.0; 4] };
        let soft_tokens = TensorF32 { shape: vec![2, 2], data: vec![1.0; 4] };
        assert!(splice_soft_tokens(&mut embeds, &[99, 10], 99, &soft_tokens).is_err());
    }
}
