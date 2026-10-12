//! What each `sems` command does; `main.rs` defines the arguments and dispatches here.

pub mod gpu;
pub mod index;
pub mod progress;
pub mod search;
pub mod status;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use sems::device::{DeviceChoice, nvidia_driver_supports_cuda, plan_runtime};
use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::encoder::{GemmaEncoder, GemmaEncoderConfig};
use sems::runtime_download;
use sems::store::{IndexIdentity, IndexStore};

use crate::Locations;
use progress::TerminalDownloadProgress;

fn index_identity() -> IndexIdentity {
    let config = GemmaEncoderConfig::default();
    IndexIdentity::current(config.identity(), config.dimensions)
}

fn open_store(locations: &Locations) -> Result<IndexStore> {
    IndexStore::open(&locations.index_path()?, index_identity())
}

/// Loads the model on the first device in the plan that works, and reports which one it was.
fn load_encoder(locations: &Locations, choice: DeviceChoice) -> Result<(GemmaEncoder, ExecutionDevice)> {
    let layout = locations.runtime_layout()?;
    let default_runtime = || runtime_download::default_runtime(&layout, &TerminalDownloadProgress::new());
    let plan = plan_runtime(
        choice,
        locations.onnxruntime.as_deref(),
        &layout,
        &nvidia_driver_supports_cuda,
        &default_runtime,
    )?;
    let runtime = OnnxRuntime::load(&plan.library)?;
    let mut failures = Vec::new();
    for &device in &plan.devices {
        match EmbeddingModel::load_from(runtime, locations.model_source()?, device) {
            Ok(model) => return Ok((GemmaEncoder::new(model, GemmaEncoderConfig::default())?, device)),
            Err(error) => failures.push(format!("{}: {error}", device_name(device))),
        }
    }
    bail!("could not load the model on any device ({})", failures.join("; "))
}

fn device_name(device: ExecutionDevice) -> &'static str {
    match device {
        ExecutionDevice::Cpu => "cpu",
        ExecutionDevice::Cuda => "cuda",
        ExecutionDevice::DirectMl => "directml",
    }
}

/// Canonical form of a user-supplied path (defaulting to the working directory).
fn resolve_existing_path(path: Option<&Path>) -> Result<PathBuf> {
    let path = path.map_or_else(std::env::current_dir, |path| Ok(path.to_path_buf()))?;
    dunce::canonicalize(&path).with_context(|| format!("{} does not exist", path.display()))
}

/// Writes to stdout, treating a closed pipe (`sems ... | head`) as success rather than a panic.
fn write_stdout(text: &str) -> Result<()> {
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => other.context("failed to write output"),
    }
}

/// Seconds since the Unix epoch, as recorded for indexed folders.
fn now_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX))
}
