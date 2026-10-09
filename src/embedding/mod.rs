//! EmbeddingGemma 2 inference on ONNX Runtime.
//!
//! The model is split into three graphs (see `tools/export/export_onnx.py`):
//! token ids -> token embeddings, image patches -> soft tokens, and embeddings -> pooled vector.
//! This module tokenizes, splices image soft tokens into placeholder positions, and runs them.

mod audio_features;
mod config;
mod image_preprocessing;
mod onnx;

use std::path::{Path, PathBuf};

use image::DynamicImage;
use tokenizers::Tokenizer;

pub use audio_features::{AudioFeatureExtractor, AudioFeatures};
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
    #[error("invalid audio: {0}")]
    InvalidAudio(String),
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
    /// Loaded on first image or video request so text-only use (every search query) never pays for it.
    vision_encoder: Option<GraphSession>,
    /// Loaded on first audio request, for the same reason. At most one of the vision and audio
    /// encoders is resident: with both (1.9 GB of weights plus the text model) a 4 GB GPU spills to
    /// shared memory and everything slows down (video measured 10x slower, audio 5x). Callers that
    /// mix modalities should group work by encoder, as the indexer does, so each loads once.
    audio_encoder: Option<GraphSession>,
    image_preprocessor: ImagePreprocessor,
    audio_feature_extractor: AudioFeatureExtractor,
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
            audio_encoder: None,
            image_preprocessor: ImagePreprocessor::new(&config.vision_config),
            audio_feature_extractor: AudioFeatureExtractor::new(),
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

    /// Embeds video frames (sampled by the caller) as one clip, in the same space as text.
    /// Frames are preprocessed like images, with the same token budget.
    pub fn embed_video(&mut self, frames: &[DynamicImage]) -> Result<Embedding, EmbeddingError> {
        if frames.is_empty() {
            return Err(EmbeddingError::InvalidImage("a video clip needs at least one frame".into()));
        }
        let preprocessor = &self.image_preprocessor;
        let preprocessed: Vec<PreprocessedImage> = std::thread::scope(|scope| {
            let handles: Vec<_> =
                frames.iter().map(|frame| scope.spawn(move || preprocessor.preprocess(frame))).collect();
            handles.into_iter().map(join_propagating_panic).collect::<Result<_, _>>()
        })?;
        self.embed_preprocessed_video(&preprocessed)
    }

    /// Exposed separately so tests can feed reference preprocessing and isolate the model stages.
    pub fn embed_preprocessed_video(&mut self, frames: &[PreprocessedImage]) -> Result<Embedding, EmbeddingError> {
        if frames.is_empty() {
            return Err(EmbeddingError::InvalidImage("a video clip needs at least one frame".into()));
        }
        let token_ids = self.video_token_ids(frames);
        let batch = PaddedBatch::new(std::slice::from_ref(&token_ids), self.config.text_config.pad_token_id);
        let mut inputs_embeds = self.embed_tokens(&batch)?;
        // One frame per vision call: batching frames is slower on small GPUs (each frame's attention
        // matrices compete for memory, measured 3 frames at 10.9 s batched on a 4 GB GPU) and no
        // faster on a CPU.
        let mut soft_tokens = TensorF32 { shape: vec![0, self.config.text_config.hidden_size], data: Vec::new() };
        for frame in frames {
            let frame_tokens = self.encode_vision(std::slice::from_ref(frame))?;
            soft_tokens.shape[0] += frame_tokens.shape[0];
            soft_tokens.data.extend(frame_tokens.data);
        }
        splice_soft_tokens(&mut inputs_embeds, &token_ids, self.config.video_token_id, &soft_tokens)?;
        Ok(self.encode(inputs_embeds, &batch)?.remove(0))
    }

    /// `[BOS]`, then `[BOI] <video>×n [EOI]` per frame, then `[EOS]`, as the upstream processor builds
    /// it without timestamps.
    fn video_token_ids(&self, frames: &[PreprocessedImage]) -> Vec<i64> {
        let text = &self.config.text_config;
        let mut ids = vec![text.bos_token_id];
        for frame in frames {
            ids.push(self.config.boi_token_id);
            ids.extend(std::iter::repeat_n(self.config.video_token_id, frame.soft_token_count));
            ids.push(self.config.eoi_token_id);
        }
        ids.push(text.eos_token_id);
        ids
    }

    /// Embeds up to 30 seconds of mono 16 kHz audio (longer input is truncated, as upstream does).
    pub fn embed_audio(&mut self, samples: &[f32]) -> Result<Embedding, EmbeddingError> {
        let features = self.audio_feature_extractor.extract(samples);
        self.embed_audio_features(&features)
    }

    /// Exposed separately so tests can feed reference features and isolate the model stages.
    pub fn embed_audio_features(&mut self, features: &AudioFeatures) -> Result<Embedding, EmbeddingError> {
        if !features.mask.contains(&1) {
            return Err(EmbeddingError::InvalidAudio("clip is too short to produce any features".into()));
        }
        let soft_tokens = self.encode_audio(features)?;
        let text = &self.config.text_config;
        let mut token_ids = vec![text.bos_token_id, self.config.boa_token_id];
        // One placeholder per soft token the encoder actually produced (25 per second of audio).
        token_ids.extend(std::iter::repeat_n(self.config.audio_token_id, soft_tokens.shape[0]));
        token_ids.extend([self.config.eoa_token_id, text.eos_token_id]);
        let batch = PaddedBatch::new(std::slice::from_ref(&token_ids), text.pad_token_id);
        let mut inputs_embeds = self.embed_tokens(&batch)?;
        splice_soft_tokens(&mut inputs_embeds, &token_ids, self.config.audio_token_id, &soft_tokens)?;
        Ok(self.encode(inputs_embeds, &batch)?.remove(0))
    }

    fn encode_audio(&mut self, features: &AudioFeatures) -> Result<TensorF32, EmbeddingError> {
        if self.audio_encoder.is_none() {
            // Release the vision encoder first: see the `audio_encoder` field.
            self.vision_encoder = None;
            let path = self.files.path("audio_encoder.onnx");
            self.audio_encoder = Some(GraphSession::load(self.runtime, "audio_encoder", &path, self.device)?);
        }
        let audio_encoder = self.audio_encoder.as_mut().expect("audio encoder was loaded above");
        // The graph takes a fixed 30 s window (see tools/export/export_onnx.py, AUDIO_FRAMES); shorter
        // clips are zero-padded with the padding masked out, which leaves their soft tokens unchanged.
        let frames = AudioFeatures::WINDOW_FRAMES;
        if features.frames > frames {
            return Err(EmbeddingError::InvalidAudio(format!("{} mel frames exceed the 30 s window", features.frames)));
        }
        let mut padded_features = features.features.clone();
        padded_features.resize(frames * AudioFeatures::MEL_BINS, 0.0);
        let mut padded_mask = features.mask.clone();
        padded_mask.resize(frames, 0);
        let input_features = onnx::tensor_f32(vec![1, frames, AudioFeatures::MEL_BINS], padded_features)?;
        let mask = onnx::tensor_i64(vec![1, frames], padded_mask)?;
        audio_encoder.run_single(vec![("input_features", input_features), ("input_features_mask", mask)], "soft_tokens")
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
            // Release the audio encoder first: see the `audio_encoder` field.
            self.audio_encoder = None;
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
