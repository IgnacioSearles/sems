//! Downloads ONNX Runtime builds, so the sems binary alone is a complete install whichever way it
//! was installed.
//!
//! - The default runtime is fetched the first time sems needs one: Microsoft's DirectML build on
//!   Windows (it also runs on the CPU, and needs no driver beyond the GPU's own) and its CPU build
//!   elsewhere.
//! - The CUDA pack (`sems gpu install`) is ONNX Runtime's CUDA build with the CUDA and cuDNN
//!   libraries it loads, for NVIDIA GPUs on Windows and Linux.
//!
//! Every archive is pinned by URL and SHA-256 (see [`packages`]). Only the libraries sems loads
//! are extracted, into a staging directory swapped in at the end, so an interrupted install never
//! leaves a half-populated runtime; archives are deleted once extracted.

mod packages;

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::device::{DEFAULT_RUNTIME_FLAVOR, ONNXRUNTIME_LIBRARY, RuntimeLayout};
use crate::download::{DownloadProgress, PinnedFile, download, is_in_place};

/// Directory name of the CUDA pack under the runtimes directory.
pub const CUDA_FLAVOR: &str = "cuda";

/// Each platform's packages use some formats only, so the others are unused outside tests.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchiveFormat {
    /// A Python wheel or plain zip.
    Zip,
    TarGz,
}

/// A library to take out of an archive.
#[derive(Debug, Clone, Copy)]
struct Member {
    /// Path inside the archive, without any leading `./`.
    archive_path: &'static str,
    /// Name in the runtime directory.
    file_name: &'static str,
    size: u64,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeArchive {
    url: &'static str,
    file: PinnedFile,
    format: ArchiveFormat,
    members: &'static [Member],
}

/// A runtime assembled from one or more archives into one directory.
#[derive(Debug, Clone, Copy)]
struct RuntimePackage {
    archives: &'static [RuntimeArchive],
}

impl RuntimePackage {
    fn members(&self) -> impl Iterator<Item = &Member> {
        self.archives.iter().flat_map(|archive| archive.members)
    }

    fn is_installed_in(&self, directory: &Path) -> bool {
        self.members().all(|member| is_in_place(&directory.join(member.file_name), member.size))
    }

    /// Bytes downloaded by an install.
    fn download_size(&self) -> u64 {
        self.archives.iter().map(|archive| archive.file.size).sum()
    }
}

/// The default ONNX Runtime library: one placed next to the sems executable if there is one,
/// otherwise the downloaded one, fetched now if it is not installed yet.
pub fn default_runtime(layout: &RuntimeLayout, progress: &dyn DownloadProgress) -> Result<PathBuf> {
    if let Some(bundled) = layout.bundled_library() {
        return Ok(bundled);
    }
    let package = packages::DEFAULT
        .context("sems cannot download ONNX Runtime for this platform; pass --onnxruntime or set SEMS_ONNXRUNTIME")?;
    install(&package, &layout.runtimes_directory.join(DEFAULT_RUNTIME_FLAVOR), progress)
}

/// What `sems gpu install` would download, or why it cannot.
pub fn cuda_pack_download_size() -> Result<u64> {
    Ok(cuda_package()?.download_size())
}

pub fn is_cuda_pack_installed(layout: &RuntimeLayout) -> bool {
    packages::CUDA.is_some_and(|package| package.is_installed_in(&layout.runtimes_directory.join(CUDA_FLAVOR)))
}

/// Installs the CUDA pack, returning its ONNX Runtime library.
pub fn install_cuda_pack(layout: &RuntimeLayout, progress: &dyn DownloadProgress) -> Result<PathBuf> {
    install(&cuda_package()?, &layout.runtimes_directory.join(CUDA_FLAVOR), progress)
}

fn cuda_package() -> Result<RuntimePackage> {
    packages::CUDA.context("CUDA acceleration is available on Windows and Linux (x64) with an NVIDIA GPU")
}

/// Makes `package`'s libraries available in `directory`, returning the main library's path.
fn install(package: &RuntimePackage, directory: &Path, progress: &dyn DownloadProgress) -> Result<PathBuf> {
    let library = directory.join(ONNXRUNTIME_LIBRARY);
    if package.is_installed_in(directory) {
        return Ok(library);
    }
    let parent = directory.parent().context("runtime directory has no parent")?;
    let staging = parent.join(format!("{}.{}.partial", file_name(directory), std::process::id()));
    let assembled = assemble(package, parent, &staging, progress);
    if let Err(error) = assembled {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    if directory.exists() {
        std::fs::remove_dir_all(directory).with_context(|| format!("failed to replace {}", directory.display()))?;
    }
    std::fs::rename(&staging, directory).with_context(|| format!("failed to install {}", directory.display()))?;
    Ok(library)
}

/// Downloads each archive into `download_directory`, extracts its members into `staging`, and
/// deletes it, so at most one archive is on disk at a time.
fn assemble(
    package: &RuntimePackage,
    download_directory: &Path,
    staging: &Path,
    progress: &dyn DownloadProgress,
) -> Result<()> {
    std::fs::create_dir_all(staging).with_context(|| format!("failed to create {}", staging.display()))?;
    for archive in package.archives {
        let path = download_directory.join(archive.file.name);
        download(archive.url, archive.file, &path, progress)?;
        let extracted = extract(archive, &path, staging);
        let _ = std::fs::remove_file(&path);
        extracted?;
    }
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Writes the archive's members into `destination`, checking each one's size.
fn extract(archive: &RuntimeArchive, path: &Path, destination: &Path) -> Result<()> {
    let file = std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    match archive.format {
        ArchiveFormat::Zip => extract_zip(file, archive.members, destination)?,
        ArchiveFormat::TarGz => extract_tar_gz(file, archive.members, destination)?,
    }
    for member in archive.members {
        ensure!(
            is_in_place(&destination.join(member.file_name), member.size),
            "{} is missing from {} or has an unexpected size",
            member.archive_path,
            archive.file.name
        );
    }
    Ok(())
}

fn extract_zip(file: std::fs::File, members: &[Member], destination: &Path) -> Result<()> {
    let mut archive = zip::ZipArchive::new(file).context("ONNX Runtime archive is not a valid zip")?;
    for member in members {
        let mut entry = archive
            .by_name(member.archive_path)
            .with_context(|| format!("{} is missing from the ONNX Runtime archive", member.archive_path))?;
        write_member(&mut entry, &destination.join(member.file_name))?;
    }
    Ok(())
}

fn extract_tar_gz(file: std::fs::File, members: &[Member], destination: &Path) -> Result<()> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in archive.entries().context("ONNX Runtime archive is not a valid tar.gz")? {
        let mut entry = entry.context("failed to read the ONNX Runtime archive")?;
        let path = entry.path().context("invalid path in the ONNX Runtime archive")?.to_string_lossy().into_owned();
        let path = path.strip_prefix("./").unwrap_or(&path).to_string();
        if let Some(member) = members.iter().find(|member| member.archive_path == path) {
            if !entry.header().entry_type().is_file() {
                bail!("{path} in the ONNX Runtime archive is not a regular file");
            }
            write_member(&mut entry, &destination.join(member.file_name))?;
        }
    }
    Ok(())
}

fn write_member(reader: &mut impl Read, path: &Path) -> Result<()> {
    let mut output = std::fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    std::io::copy(reader, &mut output).with_context(|| format!("failed to extract {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::tests::NoProgress;

    const LIBRARY: &[u8] = b"not really a library";

    fn tar_gz(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast()));
        for (path, contents) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            builder.append_data(&mut header, path, *contents).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    fn archive(format: ArchiveFormat, archive_path: &'static str) -> RuntimeArchive {
        RuntimeArchive {
            url: "http://127.0.0.1:9/unreachable", // tests never download
            file: PinnedFile { name: "runtime.archive", size: 0, sha256: "" },
            format,
            members: Box::leak(Box::new([Member {
                archive_path,
                file_name: ONNXRUNTIME_LIBRARY,
                size: LIBRARY.len() as u64,
            }])),
        }
    }

    #[test]
    fn extracts_only_the_listed_libraries_from_a_tar_gz() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("runtime.tgz");
        std::fs::write(&path, tar_gz(&[("./ort/include/api.h", b"header"), ("./ort/lib/library.so.1", LIBRARY)]))
            .unwrap();
        let destination = directory.path().join("cpu");
        std::fs::create_dir_all(&destination).unwrap();
        extract(&archive(ArchiveFormat::TarGz, "ort/lib/library.so.1"), &path, &destination).unwrap();
        assert_eq!(std::fs::read(destination.join(ONNXRUNTIME_LIBRARY)).unwrap(), LIBRARY);
        assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 1, "only the library is extracted");
    }

    #[test]
    fn extracts_from_a_zip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("runtime.whl");
        let mut writer = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        writer.start_file("pkg/capi/library.dll", zip::write::SimpleFileOptions::default()).unwrap();
        std::io::Write::write_all(&mut writer, LIBRARY).unwrap();
        writer.finish().unwrap();
        let destination = directory.path().join("directml");
        std::fs::create_dir_all(&destination).unwrap();
        extract(&archive(ArchiveFormat::Zip, "pkg/capi/library.dll"), &path, &destination).unwrap();
        assert_eq!(std::fs::read(destination.join(ONNXRUNTIME_LIBRARY)).unwrap(), LIBRARY);
    }

    #[test]
    fn a_missing_library_fails_the_extraction() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("runtime.tgz");
        std::fs::write(&path, tar_gz(&[("ort/lib/other.so", LIBRARY)])).unwrap();
        let destination = directory.path().join("cpu");
        std::fs::create_dir_all(&destination).unwrap();
        assert!(extract(&archive(ArchiveFormat::TarGz, "ort/lib/library.so.1"), &path, &destination).is_err());
    }

    #[test]
    fn an_installed_runtime_is_not_downloaded_again() {
        let directory = tempfile::tempdir().unwrap();
        let runtime = directory.path().join("cpu");
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::write(runtime.join(ONNXRUNTIME_LIBRARY), LIBRARY).unwrap();
        let package = RuntimePackage { archives: Box::leak(Box::new([archive(ArchiveFormat::TarGz, "lib.so")])) };
        assert_eq!(install(&package, &runtime, &NoProgress).unwrap(), runtime.join(ONNXRUNTIME_LIBRARY));
    }

    #[test]
    fn a_failed_install_keeps_the_previous_runtime_and_leaves_no_staging() {
        let directory = tempfile::tempdir().unwrap();
        let runtime = directory.path().join("cpu");
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::write(runtime.join("old.so"), b"old").unwrap();
        let package = RuntimePackage { archives: Box::leak(Box::new([archive(ArchiveFormat::TarGz, "lib.so")])) };
        assert!(install(&package, &runtime, &NoProgress).is_err(), "the download cannot succeed");
        let entries: Vec<_> =
            std::fs::read_dir(directory.path()).unwrap().map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(entries, ["cpu"]);
        assert!(runtime.join("old.so").is_file());
    }

    #[test]
    fn every_package_names_the_onnx_runtime_library() {
        for package in [packages::DEFAULT, packages::CUDA].into_iter().flatten() {
            assert!(package.members().any(|member| member.file_name == ONNXRUNTIME_LIBRARY));
        }
    }

    /// Downloads and extracts this platform's real default runtime, proving its pins and paths.
    #[test]
    #[ignore = "needs network access"]
    fn installs_the_pinned_default_runtime() {
        let Some(package) = packages::DEFAULT else { return };
        let directory = tempfile::tempdir().unwrap();
        let library = install(&package, &directory.path().join(DEFAULT_RUNTIME_FLAVOR), &NoProgress).unwrap();
        assert!(library.is_file());
    }

    /// Same for the CUDA pack: about 1 GB, so run it deliberately.
    #[test]
    #[ignore = "downloads about 1 GB"]
    fn installs_the_pinned_cuda_pack() {
        let Some(package) = packages::CUDA else { return };
        let directory = tempfile::tempdir().unwrap();
        let library = install(&package, &directory.path().join(CUDA_FLAVOR), &NoProgress).unwrap();
        assert!(library.is_file());
    }
}
