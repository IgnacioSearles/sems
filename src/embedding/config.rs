use std::path::Path;

use serde::Deserialize;

use super::EmbeddingError;

/// The subset of the Hugging Face `config.json` the runtime depends on.
///
/// Reading token ids from the shipped config, rather than hard-coding them, keeps the runtime
/// correct if a future model revision renumbers its special tokens.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub image_token_id: i64,
    pub boi_token_id: i64,
    pub eoi_token_id: i64,
    pub video_token_id: i64,
    pub audio_token_id: i64,
    pub boa_token_id: i64,
    /// Named `eoa_token_index` in the shipped config, unlike its siblings.
    #[serde(rename = "eoa_token_index")]
    pub eoa_token_id: i64,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub pad_token_id: i64,
    pub hidden_size: usize,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    #[serde(rename = "default_output_length")]
    pub max_soft_tokens: usize,
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self, EmbeddingError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|source| EmbeddingError::ReadFile { path: path.to_path_buf(), source })?;
        serde_json::from_str(&contents)
            .map_err(|source| EmbeddingError::InvalidConfig { path: path.to_path_buf(), source })
    }
}
