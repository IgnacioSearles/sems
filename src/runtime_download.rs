//! Downloads this platform's default ONNX Runtime the first time sems needs it, so the sems binary
//! alone is a complete install whichever way it was installed.
//!
//! Each runtime comes from an official Microsoft build pinned by URL and SHA-256: the DirectML build
//! on Windows (it also runs on the CPU, and needs no driver beyond the GPU's own) and the CPU build
//! elsewhere. Only the libraries sems loads are extracted; the archive is deleted afterwards.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

use crate::device::{DEFAULT_RUNTIME_FLAVOR, ONNXRUNTIME_LIBRARY, RuntimeLayout};
use crate::download::{DownloadProgress, PinnedFile, download, is_in_place};

/// Each platform's package uses one format, so the others are unused outside tests.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchiveFormat {
    /// A Python wheel or plain zip.
    Zip,
    TarGz,
}

/// A library to take out of the archive.
#[derive(Debug, Clone, Copy)]
struct Member {
    /// Path inside the archive, without any leading `./`.
    archive_path: &'static str,
    /// Name in the runtime directory.
    file_name: &'static str,
    size: u64,
}

#[derive(Debug, Clone, Copy)]
struct RuntimePackage {
    url: &'static str,
    archive: PinnedFile,
    format: ArchiveFormat,
    members: &'static [Member],
}

/// ONNX Runtime 1.24.4 with DirectML, the last DirectML release (see Cargo.toml), from its PyPI wheel.
#[cfg(all(windows, target_arch = "x86_64"))]
const DEFAULT_PACKAGE: Option<RuntimePackage> = Some(RuntimePackage {
    url: "https://files.pythonhosted.org/packages/60/53/2bd2696fac19cf8ca55496a0bcfe431f3aff9579eabbb0e231dc238acf6f/onnxruntime_directml-1.24.4-cp313-cp313-win_amd64.whl",
    archive: PinnedFile {
        name: "onnxruntime_directml-1.24.4-cp313-cp313-win_amd64.whl",
        size: 25_112_253,
        sha256: "2f1031cb2281e5b27cca9efe0b9399317c7286e4d226f7a79d4ab79bbd94d19e",
    },
    format: ArchiveFormat::Zip,
    members: &[
        Member { archive_path: "onnxruntime/capi/onnxruntime.dll", file_name: "onnxruntime.dll", size: 21_111_840 },
        Member {
            archive_path: "onnxruntime/capi/onnxruntime_providers_shared.dll",
            file_name: "onnxruntime_providers_shared.dll",
            size: 21_536,
        },
        Member { archive_path: "onnxruntime/capi/DirectML.dll", file_name: "DirectML.dll", size: 18_527_768 },
    ],
});

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const DEFAULT_PACKAGE: Option<RuntimePackage> = Some(RuntimePackage {
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz",
    archive: PinnedFile {
        name: "onnxruntime-linux-x64-1.30.0.tgz",
        size: 11_306_877,
        sha256: "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd",
    },
    format: ArchiveFormat::TarGz,
    members: &[Member {
        archive_path: "onnxruntime-linux-x64-1.30.0/lib/libonnxruntime.so.1.30.0",
        file_name: ONNXRUNTIME_LIBRARY,
        size: 28_985_152,
    }],
});

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const DEFAULT_PACKAGE: Option<RuntimePackage> = Some(RuntimePackage {
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-osx-arm64-1.30.0.tgz",
    archive: PinnedFile {
        name: "onnxruntime-osx-arm64-1.30.0.tgz",
        size: 42_373_116,
        sha256: "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012",
    },
    format: ArchiveFormat::TarGz,
    members: &[Member {
        archive_path: "onnxruntime-osx-arm64-1.30.0/lib/libonnxruntime.1.30.0.dylib",
        file_name: ONNXRUNTIME_LIBRARY,
        size: 43_879_424,
    }],
});

#[cfg(not(any(
    all(windows, target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
const DEFAULT_PACKAGE: Option<RuntimePackage> = None;

/// The default ONNX Runtime library: one placed next to the sems executable if there is one,
/// otherwise the downloaded one, fetched now if it is not installed yet.
pub fn default_runtime(layout: &RuntimeLayout, progress: &dyn DownloadProgress) -> Result<PathBuf> {
    if let Some(bundled) = layout.bundled_library() {
        return Ok(bundled);
    }
    let package = DEFAULT_PACKAGE
        .context("sems cannot download ONNX Runtime for this platform; pass --onnxruntime or set SEMS_ONNXRUNTIME")?;
    install(&package, &layout.runtimes_directory.join(DEFAULT_RUNTIME_FLAVOR), progress)
}

/// Makes `package`'s libraries available in `directory`, returning the main library's path.
fn install(package: &RuntimePackage, directory: &Path, progress: &dyn DownloadProgress) -> Result<PathBuf> {
    let library = directory.join(ONNXRUNTIME_LIBRARY);
    if package.members.iter().all(|member| is_in_place(&directory.join(member.file_name), member.size)) {
        return Ok(library);
    }
    let parent = directory.parent().context("runtime directory has no parent")?;
    let archive = parent.join(package.archive.name);
    download(package.url, package.archive, &archive, progress)?;
    // Extract beside the destination and swap it in, so an interrupted install never leaves a
    // half-populated runtime that sems would try to load.
    let staging = parent.join(format!("{}.{}.partial", file_name(directory), std::process::id()));
    let extracted = extract(package, &archive, &staging);
    let _ = std::fs::remove_file(&archive);
    if let Err(error) = extracted {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    if directory.exists() {
        std::fs::remove_dir_all(directory).with_context(|| format!("failed to replace {}", directory.display()))?;
    }
    std::fs::rename(&staging, directory).with_context(|| format!("failed to install {}", directory.display()))?;
    Ok(library)
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Writes the package's members into `destination`, checking each one's size.
fn extract(package: &RuntimePackage, archive: &Path, destination: &Path) -> Result<()> {
    std::fs::create_dir_all(destination).with_context(|| format!("failed to create {}", destination.display()))?;
    let file = std::fs::File::open(archive).with_context(|| format!("failed to open {}", archive.display()))?;
    match package.format {
        ArchiveFormat::Zip => extract_zip(file, package.members, destination)?,
        ArchiveFormat::TarGz => extract_tar_gz(file, package.members, destination)?,
    }
    for member in package.members {
        let path = destination.join(member.file_name);
        ensure!(
            is_in_place(&path, member.size),
            "{} is missing from {} or has an unexpected size",
            member.archive_path,
            package.archive.name
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

    fn package(format: ArchiveFormat, archive_path: &'static str) -> RuntimePackage {
        RuntimePackage {
            url: "http://127.0.0.1:9/unreachable", // tests never download
            archive: PinnedFile { name: "runtime.archive", size: 0, sha256: "" },
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
        let archive = directory.path().join("runtime.tgz");
        std::fs::write(&archive, tar_gz(&[("./ort/include/api.h", b"header"), ("./ort/lib/library.so.1", LIBRARY)]))
            .unwrap();
        let destination = directory.path().join("cpu");
        extract(&package(ArchiveFormat::TarGz, "ort/lib/library.so.1"), &archive, &destination).unwrap();
        assert_eq!(std::fs::read(destination.join(ONNXRUNTIME_LIBRARY)).unwrap(), LIBRARY);
        assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 1, "only the library is extracted");
    }

    #[test]
    fn extracts_from_a_zip() {
        let directory = tempfile::tempdir().unwrap();
        let archive = directory.path().join("runtime.whl");
        let mut writer = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        writer.start_file("pkg/capi/library.dll", zip::write::SimpleFileOptions::default()).unwrap();
        std::io::Write::write_all(&mut writer, LIBRARY).unwrap();
        writer.finish().unwrap();
        let destination = directory.path().join("directml");
        extract(&package(ArchiveFormat::Zip, "pkg/capi/library.dll"), &archive, &destination).unwrap();
        assert_eq!(std::fs::read(destination.join(ONNXRUNTIME_LIBRARY)).unwrap(), LIBRARY);
    }

    #[test]
    fn a_missing_library_fails_the_extraction() {
        let directory = tempfile::tempdir().unwrap();
        let archive = directory.path().join("runtime.tgz");
        std::fs::write(&archive, tar_gz(&[("ort/lib/other.so", LIBRARY)])).unwrap();
        let error =
            extract(&package(ArchiveFormat::TarGz, "ort/lib/library.so.1"), &archive, &directory.path().join("cpu"));
        assert!(error.is_err());
    }

    #[test]
    fn an_installed_runtime_is_not_downloaded_again() {
        let directory = tempfile::tempdir().unwrap();
        let runtime = directory.path().join("cpu");
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::write(runtime.join(ONNXRUNTIME_LIBRARY), LIBRARY).unwrap();
        let installed = install(&package(ArchiveFormat::TarGz, "ort/lib/library.so.1"), &runtime, &NoProgress).unwrap();
        assert_eq!(installed, runtime.join(ONNXRUNTIME_LIBRARY));
    }

    /// Downloads and extracts this platform's real runtime, proving its pins and member paths.
    #[test]
    #[ignore = "needs network access"]
    fn installs_the_pinned_runtime_for_this_platform() {
        let Some(package) = DEFAULT_PACKAGE else { return };
        let directory = tempfile::tempdir().unwrap();
        let library = install(&package, &directory.path().join(DEFAULT_RUNTIME_FLAVOR), &NoProgress).unwrap();
        assert!(library.is_file());
    }
}
