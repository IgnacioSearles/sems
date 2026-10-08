//! Turns queries and documents into comparable vectors.
//!
//! The indexer and search depend on the [`Encoder`] trait only, so they can be tested with a fake
//! encoder and stay independent of ONNX Runtime.

use anyhow::{Context, Result, ensure};
use image::DynamicImage;

use crate::embedding::EmbeddingModel;

/// A piece of content to embed. `title` is typically the file name.
#[derive(Debug, Clone, Copy)]
pub struct Document<'a> {
    pub title: &'a str,
    pub text: &'a str,
}

pub trait Encoder {
    /// Names the vector space. Stored in the index so vectors from different encoders never mix.
    fn identity(&self) -> String;
    fn dimensions(&self) -> usize;
    /// Unit-length vector for a search query.
    fn encode_query(&mut self, query: &str) -> Result<Vec<f32>>;
    /// Unit-length vectors for documents, in input order.
    fn encode_documents(&mut self, documents: &[Document<'_>]) -> Result<Vec<Vec<f32>>>;
    /// Unit-length vector for an image, in the same space as text, so text queries find images.
    fn encode_image(&mut self, image: &DynamicImage) -> Result<Vec<f32>>;
}

/// Prompt formats from the EmbeddingGemma 2 model card (asymmetric retrieval).
const QUERY_PREFIX: &str = "task: search result | query: ";
const MODEL_NAME: &str = "google/embeddinggemma-2";
/// Matryoshka sizes the model was trained for.
const SUPPORTED_DIMENSIONS: [usize; 4] = [768, 512, 256, 128];

pub struct GemmaEncoderConfig {
    /// Leading dimensions kept; 256 is near-lossless on the model card and a third of the storage.
    pub dimensions: usize,
    /// Upper bound on `rows * padded_length` per inference call; sized for a 4 GB GPU.
    pub padded_token_budget: usize,
    /// Longest document sequence; chunking keeps real chunks far below this.
    pub max_sequence_tokens: usize,
}

impl GemmaEncoderConfig {
    /// Available without loading the model, so commands like `status` can open the index cheaply.
    pub fn identity(&self) -> String {
        format!("{MODEL_NAME}@{}", self.dimensions)
    }
}

impl Default for GemmaEncoderConfig {
    fn default() -> Self {
        Self { dimensions: 256, padded_token_budget: 2_400, max_sequence_tokens: 2_048 }
    }
}

pub struct GemmaEncoder {
    model: EmbeddingModel,
    config: GemmaEncoderConfig,
}

impl GemmaEncoder {
    pub fn new(model: EmbeddingModel, config: GemmaEncoderConfig) -> Result<Self> {
        ensure!(
            SUPPORTED_DIMENSIONS.contains(&config.dimensions),
            "unsupported embedding dimensions {} (expected one of {SUPPORTED_DIMENSIONS:?})",
            config.dimensions
        );
        ensure!(config.max_sequence_tokens >= 2, "max_sequence_tokens must leave room for BOS and EOS");
        Ok(Self { model, config })
    }
}

impl Encoder for GemmaEncoder {
    fn identity(&self) -> String {
        self.config.identity()
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn encode_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let embedding = self.model.embed_texts(&[&format!("{QUERY_PREFIX}{query}")])?.remove(0);
        Ok(truncate_and_normalize(&embedding, self.config.dimensions))
    }

    fn encode_image(&mut self, image: &DynamicImage) -> Result<Vec<f32>> {
        // The model card: images take no task prefix.
        let embedding = self.model.embed_image(image)?;
        Ok(truncate_and_normalize(&embedding, self.config.dimensions))
    }

    fn encode_documents(&mut self, documents: &[Document<'_>]) -> Result<Vec<Vec<f32>>> {
        let prompts: Vec<String> = documents.iter().map(document_prompt).collect();
        let prompt_references: Vec<&str> = prompts.iter().map(String::as_str).collect();
        let mut sequences = self.model.tokenize(&prompt_references)?;
        for sequence in &mut sequences {
            truncate_sequence(sequence, self.config.max_sequence_tokens);
        }

        let lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
        let mut embeddings = vec![Vec::new(); documents.len()];
        for batch in plan_batches(&lengths, self.config.padded_token_budget) {
            let batch_sequences: Vec<Vec<i64>> = batch.iter().map(|&index| sequences[index].clone()).collect();
            let batch_embeddings = self
                .model
                .embed_token_sequences(&batch_sequences)
                .with_context(|| format!("failed to embed a batch of {} documents", batch.len()))?;
            for (index, embedding) in batch.into_iter().zip(batch_embeddings) {
                embeddings[index] = truncate_and_normalize(&embedding, self.config.dimensions);
            }
        }
        Ok(embeddings)
    }
}

fn document_prompt(document: &Document<'_>) -> String {
    let title = if document.title.trim().is_empty() { "none" } else { document.title };
    format!("title: {title} | text: {}", document.text)
}

/// Keeps the first `max_tokens - 1` tokens and the final EOS, so the model still sees a terminator.
fn truncate_sequence(sequence: &mut Vec<i64>, max_tokens: usize) {
    if sequence.len() > max_tokens {
        let end_of_sequence = *sequence.last().expect("non-empty after length check");
        sequence.truncate(max_tokens - 1);
        sequence.push(end_of_sequence);
    }
}

/// Groups sequence indices into batches whose padded size (`rows * longest`) fits the budget.
/// Sorting by length first keeps padding waste low. A sequence longer than the budget gets its
/// own batch rather than failing.
pub fn plan_batches(lengths: &[usize], padded_token_budget: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&index| lengths[index]);

    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for index in order {
        // Sorted ascending, so the newest member is always the longest.
        let padded_size = (current.len() + 1) * lengths[index];
        if !current.is_empty() && padded_size > padded_token_budget {
            batches.push(std::mem::take(&mut current));
        }
        current.push(index);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Matryoshka truncation. Re-normalizing is required: a slice of a unit vector is not unit length.
pub fn truncate_and_normalize(embedding: &[f32], dimensions: usize) -> Vec<f32> {
    let truncated = &embedding[..dimensions.min(embedding.len())];
    let norm = truncated.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        return truncated.to_vec();
    }
    truncated.iter().map(|value| value / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_renormalizes_to_unit_length() {
        let vector = truncate_and_normalize(&[3.0, 4.0, 12.0], 2);
        assert_eq!(vector, [0.6, 0.8]);
    }

    #[test]
    fn batches_respect_padded_budget_and_cover_all_indices() {
        let lengths = [10, 300, 20, 300, 15, 40];
        let batches = plan_batches(&lengths, 600);
        for batch in &batches {
            let longest = batch.iter().map(|&index| lengths[index]).max().unwrap();
            assert!(batch.len() == 1 || batch.len() * longest <= 600, "batch {batch:?} exceeds budget");
        }
        let mut covered: Vec<usize> = batches.concat();
        covered.sort_unstable();
        assert_eq!(covered, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn oversized_sequence_gets_its_own_batch() {
        assert_eq!(plan_batches(&[5_000, 10], 1_000), [vec![1], vec![0]]);
    }

    #[test]
    fn truncated_sequences_keep_their_terminator() {
        let mut sequence = vec![2, 10, 11, 12, 13, 1];
        truncate_sequence(&mut sequence, 4);
        assert_eq!(sequence, [2, 10, 11, 1]);
    }

    #[test]
    fn empty_titles_use_the_model_card_placeholder() {
        let prompt = document_prompt(&Document { title: " ", text: "body" });
        assert_eq!(prompt, "title: none | text: body");
    }
}
