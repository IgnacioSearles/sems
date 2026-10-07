//! Thin wrapper that confines the `ort` API (still a release candidate) to one file.

use std::path::Path;

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
    /// Loads `onnxruntime.dll` (or the platform equivalent) from an explicit path.
    pub fn load(library_path: &Path) -> Result<Self, EmbeddingError> {
        if !library_path.is_file() {
            return Err(EmbeddingError::ReadFile {
                path: library_path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "ONNX Runtime library not found"),
            });
        }
        let builder = ort::init_from(library_path).map_err(|error| EmbeddingError::RuntimeLibrary {
            path: library_path.to_path_buf(),
            message: error.to_string(),
        })?;
        builder.with_name("sems").commit();
        Ok(Self { _private: () })
    }
}

pub struct GraphSession {
    name: &'static str,
    session: Session,
}

impl GraphSession {
    pub fn load(_runtime: OnnxRuntime, name: &'static str, path: &Path) -> Result<Self, EmbeddingError> {
        let onnx_error = |source: ort::Error| EmbeddingError::Onnx { graph: name, source };
        if !path.is_file() {
            return Err(EmbeddingError::ReadFile {
                path: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "model graph not found"),
            });
        }
        let session = Session::builder()
            .map_err(onnx_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| onnx_error(error.into()))?
            .commit_from_file(path)
            .map_err(onnx_error)?;
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
