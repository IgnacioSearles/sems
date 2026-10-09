//! Downloads the exported model from Hugging Face the first time each file is needed.
//!
//! The binary pins the repository commit and every file's size and SHA-256, so what runs is
//! exactly what the export verified against the reference embeddings. A download is written to a
//! temporary file, checked while it streams, and renamed into place only if it matches; a file
//! already in place is trusted by its size (hashing 600 MB on every search would cost seconds).
//! Searches need only the text model (~550 MB); the vision and audio encoders download the first
//! time an image or recording is indexed.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};

use crate::embedding::{EmbeddingError, ModelSource};

pub const REPOSITORY: &str = "neich-cereales/sems-embeddinggemma-2-onnx";
/// Commit of [`REPOSITORY`] holding the files in [`FILES`]; bump both together after an export.
pub const REVISION: &str = "e5d3e4fcc8679d8f036b0c32b9bdacf92cd79c86";

#[derive(Debug, Clone, Copy)]
struct RemoteFile {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

const FILES: [RemoteFile; 10] = [
    RemoteFile {
        name: "config.json",
        size: 4_455,
        sha256: "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
    },
    RemoteFile {
        name: "tokenizer.json",
        size: 32_170_510,
        sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    },
    RemoteFile {
        name: "token_embedder.onnx",
        size: 10_862,
        sha256: "8aeba37ccea0927fe9cf315e29d128816aa49ec37498dbad15468e2cd4f04d70",
    },
    RemoteFile {
        name: "token_embedder.onnx.data",
        size: 268_435_456,
        sha256: "5667d3471a2eaf6c4d1f9b78adf976b5e3ed9255a5ee20f2ac8619849fb48701",
    },
    RemoteFile {
        name: "text_encoder.onnx",
        size: 6_198_077,
        sha256: "bfb7540de83bdc4bb4416143e1a2b7d92d82590ce66dd84d48f841c67de51f6e",
    },
    RemoteFile {
        name: "text_encoder.onnx.data",
        size: 273_705_984,
        sha256: "02c8b646e1c79280565df295f99762288b9dde2ac5c47a849085684b3622753b",
    },
    RemoteFile {
        name: "vision_encoder.onnx",
        size: 4_377_050,
        sha256: "f6bea4af405b8399fa67c56aa51dda6c0c606d62133b482d404ce7d9f674bca2",
    },
    RemoteFile {
        name: "vision_encoder.onnx.data",
        size: 335_609_856,
        sha256: "0daceb088c53dbf2387fbb8dc0506ebf53f2290a7875ff439bc820b00a272f53",
    },
    RemoteFile {
        name: "audio_encoder.onnx",
        size: 10_651_204,
        sha256: "e027868ecaafe24919c10fe6bbbf76ab53405d3e94e03864c17d0637fd9d7958",
    },
    RemoteFile {
        name: "audio_encoder.onnx.data",
        size: 611_447_296,
        sha256: "7b9e1cb7afe152c7b90f71f5b82b2ca2960fb8044a090ab24d07a58451527f3b",
    },
];

/// Hears about downloads, so the CLI can show progress without the downloader knowing how.
pub trait DownloadProgress: Send + Sync {
    fn started(&self, file_name: &str, size: u64);
    /// Called at most once per percent.
    fn advanced(&self, file_name: &str, downloaded: u64, size: u64);
    fn finished(&self, file_name: &str);
}

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
        let requested = remote_file(file_name)?;
        let weights_name = format!("{file_name}.data");
        for file in std::iter::once(requested).chain(remote_file(&weights_name).ok()) {
            self.ensure_one(file)?;
        }
        Ok(self.directory.join(file_name))
    }

    fn ensure_one(&self, file: RemoteFile) -> Result<()> {
        let destination = self.directory.join(file.name);
        if is_in_place(&destination, file) {
            return Ok(());
        }
        std::fs::create_dir_all(&self.directory)
            .with_context(|| format!("failed to create {}", self.directory.display()))?;
        let url = format!("{}/{}", self.base_url, file.name);
        let response = ureq::get(&url).call().with_context(|| format!("failed to download {url}"))?;
        self.progress.started(file.name, file.size);
        download_verified(response.into_body().into_reader(), file, &destination, &*self.progress)?;
        self.progress.finished(file.name);
        Ok(())
    }
}

impl ModelSource for DownloadedModel {
    fn file(&self, file_name: &str) -> Result<PathBuf, EmbeddingError> {
        self.ensure(file_name)
            .map_err(|error| EmbeddingError::ModelFile { file: file_name.to_string(), message: format!("{error:#}") })
    }
}

fn remote_file(file_name: &str) -> Result<RemoteFile> {
    FILES
        .iter()
        .copied()
        .find(|file| file.name == file_name)
        .with_context(|| format!("{file_name} is not part of the model"))
}

fn is_in_place(path: &Path, file: RemoteFile) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.len() == file.size)
}

/// Streams `reader` into `destination`, which appears only if the size and SHA-256 match `file`.
/// The temporary name includes the process id, so concurrent sems processes never share one.
fn download_verified(
    reader: impl Read,
    file: RemoteFile,
    destination: &Path,
    progress: &dyn DownloadProgress,
) -> Result<()> {
    let partial = destination.with_file_name(format!("{}.{}.partial", file.name, std::process::id()));
    let written = write_verified(reader, file, &partial, progress);
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
        return written;
    }
    std::fs::rename(&partial, destination).with_context(|| format!("failed to move {} into place", file.name))
}

fn write_verified(mut reader: impl Read, file: RemoteFile, path: &Path, progress: &dyn DownloadProgress) -> Result<()> {
    let mut output = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 256 * 1024];
    let mut downloaded: u64 = 0;
    let mut reported_percent = 0;
    loop {
        let read = reader.read(&mut buffer).with_context(|| format!("download of {} was interrupted", file.name))?;
        if read == 0 {
            break;
        }
        downloaded += read as u64;
        ensure!(downloaded <= file.size, "{} is larger than the expected {} bytes", file.name, file.size);
        hasher.update(&buffer[..read]);
        output.write_all(&buffer[..read]).with_context(|| format!("failed to write {}", path.display()))?;
        let percent = downloaded * 100 / file.size.max(1);
        if percent > reported_percent {
            reported_percent = percent;
            progress.advanced(file.name, downloaded, file.size);
        }
    }
    output.flush().with_context(|| format!("failed to write {}", path.display()))?;
    ensure!(downloaded == file.size, "{} ended after {downloaded} of {} bytes", file.name, file.size);
    let digest = hex(&hasher.finalize());
    if digest != file.sha256 {
        bail!("{} does not match its pinned SHA-256 (got {digest})", file.name);
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoProgress;

    impl DownloadProgress for NoProgress {
        fn started(&self, _: &str, _: u64) {}
        fn advanced(&self, _: &str, _: u64, _: u64) {}
        fn finished(&self, _: &str) {}
    }

    /// "hello world" and its SHA-256.
    const HELLO: RemoteFile = RemoteFile {
        name: "hello.txt",
        size: 11,
        sha256: "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
    };

    fn entries(directory: &Path) -> Vec<String> {
        std::fs::read_dir(directory).unwrap().map(|entry| entry.unwrap().file_name().into_string().unwrap()).collect()
    }

    #[test]
    fn a_verified_download_is_moved_into_place() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join(HELLO.name);
        download_verified(&b"hello world"[..], HELLO, &destination, &NoProgress).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"hello world");
        assert_eq!(entries(directory.path()), ["hello.txt"], "no temporary file is left behind");
    }

    #[test]
    fn corrupt_or_truncated_downloads_leave_nothing_behind() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join(HELLO.name);
        for body in [&b"hello wOrld"[..], &b"hello"[..], &b"hello world!"[..]] {
            assert!(download_verified(body, HELLO, &destination, &NoProgress).is_err(), "{body:?} was accepted");
        }
        assert!(entries(directory.path()).is_empty());
    }

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
            assert!(remote_file(&format!("{}.data", graph.name)).is_ok(), "{} has no weights", graph.name);
        }
        assert!(remote_file("model.safetensors").is_err());
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
