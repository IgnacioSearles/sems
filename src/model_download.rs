//! Downloads the exported model from Hugging Face the first time each file is needed.
//!
//! The binary pins the repository commit and every file's size and SHA-256, so what runs is
//! exactly what the export verified against the reference embeddings (see [`crate::download`]).
//! Searches need only the text model (~550 MB); the vision and audio encoders download the first
//! time an image or recording is indexed.

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::download::{DownloadProgress, PinnedFile, download, is_in_place};
use crate::embedding::{EmbeddingError, ModelSource};

pub const REPOSITORY: &str = "neich-cereales/sems-embeddinggemma-2-onnx";
/// Commit of [`REPOSITORY`] holding the files in [`FILES`]; bump both together after an export.
pub const REVISION: &str = "e5d3e4fcc8679d8f036b0c32b9bdacf92cd79c86";

const FILES: [PinnedFile; 10] = [
    PinnedFile {
        name: "config.json",
        size: 4_455,
        sha256: "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
    },
    PinnedFile {
        name: "tokenizer.json",
        size: 32_170_510,
        sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    },
    PinnedFile {
        name: "token_embedder.onnx",
        size: 10_862,
        sha256: "8aeba37ccea0927fe9cf315e29d128816aa49ec37498dbad15468e2cd4f04d70",
    },
    PinnedFile {
        name: "token_embedder.onnx.data",
        size: 268_435_456,
        sha256: "5667d3471a2eaf6c4d1f9b78adf976b5e3ed9255a5ee20f2ac8619849fb48701",
    },
    PinnedFile {
        name: "text_encoder.onnx",
        size: 6_198_077,
        sha256: "bfb7540de83bdc4bb4416143e1a2b7d92d82590ce66dd84d48f841c67de51f6e",
    },
    PinnedFile {
        name: "text_encoder.onnx.data",
        size: 273_705_984,
        sha256: "02c8b646e1c79280565df295f99762288b9dde2ac5c47a849085684b3622753b",
    },
    PinnedFile {
        name: "vision_encoder.onnx",
        size: 4_377_050,
        sha256: "f6bea4af405b8399fa67c56aa51dda6c0c606d62133b482d404ce7d9f674bca2",
    },
    PinnedFile {
        name: "vision_encoder.onnx.data",
        size: 335_609_856,
        sha256: "0daceb088c53dbf2387fbb8dc0506ebf53f2290a7875ff439bc820b00a272f53",
    },
    PinnedFile {
        name: "audio_encoder.onnx",
        size: 10_651_204,
        sha256: "e027868ecaafe24919c10fe6bbbf76ab53405d3e94e03864c17d0637fd9d7958",
    },
    PinnedFile {
        name: "audio_encoder.onnx.data",
        size: 611_447_296,
        sha256: "7b9e1cb7afe152c7b90f71f5b82b2ca2960fb8044a090ab24d07a58451527f3b",
    },
];

/// The pinned model, kept in `directory` and downloaded from [`REPOSITORY`] as files are needed.
pub struct DownloadedModel {
    directory: PathBuf,
    base_url: String,
    progress: Box<dyn DownloadProgress>,
}

impl DownloadedModel {
    pub fn new(directory: PathBuf, progress: Box<dyn DownloadProgress>) -> Self {
        let base_url = format!("https://huggingface.co/{REPOSITORY}/resolve/{REVISION}");
        Self { directory, base_url, progress }
    }

    /// Makes `file_name` and, for a graph, its external weights available locally.
    fn ensure(&self, file_name: &str) -> Result<PathBuf> {
        let requested = pinned_file(file_name)?;
        let weights_name = format!("{file_name}.data");
        for file in std::iter::once(requested).chain(pinned_file(&weights_name).ok()) {
            let destination = self.directory.join(file.name);
            if !is_in_place(&destination, file.size) {
                download(&format!("{}/{}", self.base_url, file.name), file, &destination, &*self.progress)?;
            }
        }
        Ok(self.directory.join(file_name))
    }
}

impl ModelSource for DownloadedModel {
    fn file(&self, file_name: &str) -> Result<PathBuf, EmbeddingError> {
        self.ensure(file_name)
            .map_err(|error| EmbeddingError::ModelFile { file: file_name.to_string(), message: format!("{error:#}") })
    }
}

fn pinned_file(file_name: &str) -> Result<PinnedFile> {
    FILES
        .iter()
        .copied()
        .find(|file| file.name == file_name)
        .with_context(|| format!("{file_name} is not part of the model"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::tests::NoProgress;

    #[test]
    fn files_already_in_place_are_not_downloaded_again() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), vec![b' '; FILES[0].size as usize]).unwrap();
        let model = DownloadedModel {
            directory: directory.path().to_path_buf(),
            base_url: "http://127.0.0.1:9".into(), // nothing listens here; a download would fail
            progress: Box::new(NoProgress),
        };
        assert_eq!(model.file("config.json").unwrap(), directory.path().join("config.json"));
        assert!(model.file("tokenizer.json").is_err(), "a missing file is fetched, and fetching fails here");
    }

    #[test]
    fn every_graph_has_its_weights_and_unknown_files_are_rejected() {
        for graph in FILES.iter().filter(|file| file.name.ends_with(".onnx")) {
            assert!(pinned_file(&format!("{}.data", graph.name)).is_ok(), "{} has no weights", graph.name);
        }
        assert!(pinned_file("model.safetensors").is_err());
    }

    /// Downloads the smallest pinned file from Hugging Face, proving the URL and pins are right.
    #[test]
    #[ignore = "needs network access"]
    fn downloads_a_pinned_file_from_hugging_face() {
        let directory = tempfile::tempdir().unwrap();
        let model = DownloadedModel::new(directory.path().to_path_buf(), Box::new(NoProgress));
        let path = model.file("config.json").unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().len(), FILES[0].size);
    }
}
