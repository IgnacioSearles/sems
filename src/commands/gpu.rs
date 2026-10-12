//! `sems gpu install` and `sems gpu remove`.

use std::path::Path;

use anyhow::{Context, Result, bail};

use sems::device::nvidia_driver_supports_cuda;
use sems::runtime_download::{self, CUDA_FLAVOR};

use super::progress::{TerminalDownloadProgress, readable_size};
use crate::Locations;

/// Checks the driver before downloading: the pack is about 1 GB and useless without it.
pub fn install(locations: &Locations) -> Result<()> {
    let layout = locations.runtime_layout()?;
    let download_size = runtime_download::cuda_pack_download_size()?;
    if runtime_download::is_cuda_pack_installed(&layout) {
        eprintln!("the CUDA pack is already installed in {}", layout.runtimes_directory.join(CUDA_FLAVOR).display());
        return Ok(());
    }
    if !nvidia_driver_supports_cuda() {
        bail!(
            "no NVIDIA GPU with a driver that supports CUDA 13 was found; the CUDA pack needs an NVIDIA GPU and \
             driver 580 or newer (other GPUs are used through DirectML on Windows)"
        );
    }
    eprintln!("downloading the CUDA pack ({})", readable_size(download_size));
    let library = runtime_download::install_cuda_pack(&layout, &TerminalDownloadProgress::new())?;
    let directory = library.parent().unwrap_or(Path::new(""));
    eprintln!("installed the CUDA pack in {}; `sems index` now uses your NVIDIA GPU", directory.display());
    Ok(())
}

pub fn remove(locations: &Locations) -> Result<()> {
    let directory = locations.runtime_layout()?.runtimes_directory.join(CUDA_FLAVOR);
    if !directory.exists() {
        eprintln!("the CUDA pack is not installed");
        return Ok(());
    }
    std::fs::remove_dir_all(&directory).with_context(|| format!("failed to delete {}", directory.display()))?;
    eprintln!("removed the CUDA pack from {}", directory.display());
    Ok(())
}
