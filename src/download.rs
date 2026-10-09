//! Downloads of pinned files: written to a temporary file, checked while they stream, and moved
//! into place only if their size and SHA-256 match. Used for the model and for ONNX Runtime.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};

/// A file whose exact content is known in advance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// Hears about downloads, so the CLI can show progress without the downloader knowing how.
pub trait DownloadProgress: Send + Sync {
    fn started(&self, file_name: &str, size: u64);
    /// Called at most once per percent.
    fn advanced(&self, file_name: &str, downloaded: u64, size: u64);
    fn finished(&self, file_name: &str);
}

/// A file already downloaded is trusted by its size: hashing hundreds of megabytes every time sems
/// starts would cost seconds, and downloads are only moved into place after their hash matched.
pub fn is_in_place(path: &Path, size: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.len() == size)
}

/// Downloads `url` to `destination`, which appears only if it matches `file`.
pub fn download(url: &str, file: PinnedFile, destination: &Path, progress: &dyn DownloadProgress) -> Result<()> {
    if let Some(directory) = destination.parent() {
        std::fs::create_dir_all(directory).with_context(|| format!("failed to create {}", directory.display()))?;
    }
    let response = ureq::get(url).call().with_context(|| format!("failed to download {url}"))?;
    progress.started(file.name, file.size);
    download_verified(response.into_body().into_reader(), file, destination, progress)?;
    progress.finished(file.name);
    Ok(())
}

/// Streams `reader` into `destination`, which appears only if the size and SHA-256 match `file`.
/// The temporary name includes the process id, so concurrent sems processes never share one.
fn download_verified(
    reader: impl Read,
    file: PinnedFile,
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

fn write_verified(mut reader: impl Read, file: PinnedFile, path: &Path, progress: &dyn DownloadProgress) -> Result<()> {
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
pub(crate) mod tests {
    use super::*;

    pub(crate) struct NoProgress;

    impl DownloadProgress for NoProgress {
        fn started(&self, _: &str, _: u64) {}
        fn advanced(&self, _: &str, _: u64, _: u64) {}
        fn finished(&self, _: &str) {}
    }

    /// "hello world" and its SHA-256.
    const HELLO: PinnedFile = PinnedFile {
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
}
