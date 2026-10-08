//! Thin wrapper that confines the `ort` API (still a release candidate) to one file.

use std::path::Path;

use ort::ep::directml::{DeviceFilter, PerformancePreference};
use ort::ep::{CUDA, DirectML};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, Tensor};

use super::EmbeddingError;

/// An owned, row-major f32 tensor.
#[derive(Debug, Clone)]
pub struct TensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Proof that the ONNX Runtime shared library has been loaded; required to create sessions.
#[derive(Debug, Clone, Copy)]
pub struct OnnxRuntime {
    _private: (),
}

impl OnnxRuntime {
    /// Loads `onnxruntime.dll` (or the platform equivalent) from an explicit path. The library's
    /// own dependencies (DirectML.dll, CUDA and cuDNN) are expected in the same directory.
    pub fn load(library_path: &Path) -> Result<Self, EmbeddingError> {
        let runtime_library_error =
            |message: String| EmbeddingError::RuntimeLibrary { path: library_path.to_path_buf(), message };
        if !library_path.is_file() {
            return Err(EmbeddingError::ReadFile {
                path: library_path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "ONNX Runtime library not found"),
            });
        }
        if let Some(directory) = library_path.parent() {
            search_directory_for_dependencies(directory).map_err(|error| runtime_library_error(error.to_string()))?;
        }
        let builder = ort::init_from(library_path).map_err(|error| runtime_library_error(error.to_string()))?;
        builder.with_name("sems").commit();
        Ok(Self { _private: () })
    }
}

/// Makes `directory` part of the DLL search order for libraries loaded by name.
///
/// ONNX Runtime's CUDA provider loads cuDNN with a bare `LoadLibrary("cudnn64_9.dll")` at first
/// use, and Windows does not search the requesting DLL's directory for that, so a runtime folder
/// with everything side by side would still fail without this. Process-wide, which is fine: a
/// process only ever loads one ONNX Runtime.
#[cfg(windows)]
fn search_directory_for_dependencies(directory: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDllDirectoryW(path_name: *const u16) -> i32;
    }

    let wide: Vec<u16> = directory.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is a NUL-terminated UTF-16 string that outlives the call, which copies it.
    if unsafe { SetDllDirectoryW(wide.as_ptr()) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Other platforms resolve dependencies through the library's rpath.
#[cfg(not(windows))]
fn search_directory_for_dependencies(_directory: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Where a graph runs. GPU devices require the matching ONNX Runtime build: the CUDA pack
/// (onnxruntime-gpu with CUDA 13 and cuDNN 9 beside it) or the DirectML build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDevice {
    Cpu,
    Cuda,
    DirectMl,
}

impl std::str::FromStr for ExecutionDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "directml" => Ok(Self::DirectMl),
            other => Err(format!("unknown execution device '{other}' (expected cpu, cuda, or directml)")),
        }
    }
}

pub struct GraphSession {
    name: &'static str,
    session: Session,
}

impl GraphSession {
    pub fn load(
        _runtime: OnnxRuntime,
        name: &'static str,
        path: &Path,
        device: ExecutionDevice,
    ) -> Result<Self, EmbeddingError> {
        let onnx_error = |source: ort::Error| EmbeddingError::Onnx { graph: name, source };
        if !path.is_file() {
            return Err(EmbeddingError::ReadFile {
                path: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "model graph not found"),
            });
        }
        let mut builder = Session::builder()
            .map_err(onnx_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| onnx_error(error.into()))?;
        // error_on_failure: a missing GPU dependency must be an error, not a silent 10x slowdown on CPU.
        builder = match device {
            ExecutionDevice::Cpu => builder,
            ExecutionDevice::Cuda => builder
                .with_execution_providers([CUDA::default().build().error_on_failure()])
                .map_err(|error| onnx_error(error.into()))?,
            ExecutionDevice::DirectMl => builder
                // DirectML does not support memory patterns or parallel execution.
                .with_memory_pattern(false)
                .map_err(|error| onnx_error(error.into()))?
                .with_parallel_execution(false)
                .map_err(|error| onnx_error(error.into()))?
                // Laptops pair an integrated GPU with a discrete one; adapter 0 is often the weaker.
                .with_execution_providers([DirectML::default()
                    .with_device_filter(DeviceFilter::Gpu)
                    .with_performance_preference(PerformancePreference::HighPerformance)
                    .build()
                    .error_on_failure()])
                .map_err(|error| onnx_error(error.into()))?,
        };
        let session = builder.commit_from_file(path).map_err(onnx_error)?;
        Ok(Self { name, session })
    }

    /// Runs the graph and returns one named f32 output.
    pub fn run_single(
        &mut self,
        inputs: Vec<(&'static str, DynValue)>,
        output_name: &str,
    ) -> Result<TensorF32, EmbeddingError> {
        let name = self.name;
        let outputs = self.session.run(inputs).map_err(|source| EmbeddingError::Onnx { graph: name, source })?;
        let output = outputs.get(output_name).ok_or_else(|| EmbeddingError::UnexpectedOutput {
            graph: name,
            message: format!("missing output '{output_name}'"),
        })?;
        let (shape, data) =
            output.try_extract_tensor::<f32>().map_err(|source| EmbeddingError::Onnx { graph: name, source })?;
        Ok(TensorF32 { shape: shape.iter().map(|&dimension| dimension as usize).collect(), data: data.to_vec() })
    }
}

pub fn tensor_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<DynValue, EmbeddingError> {
    Tensor::from_array((shape, data))
        .map(Tensor::into_dyn)
        .map_err(|source| EmbeddingError::Onnx { graph: "input", source })
}

pub fn tensor_i64(shape: Vec<usize>, data: Vec<i64>) -> Result<DynValue, EmbeddingError> {
    Tensor::from_array((shape, data))
        .map(Tensor::into_dyn)
        .map_err(|source| EmbeddingError::Onnx { graph: "input", source })
}
