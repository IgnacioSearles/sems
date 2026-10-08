//! Choosing an ONNX Runtime build and execution device.
//!
//! A process can load only one ONNX Runtime library, and the CUDA and DirectML builds are different
//! libraries, so the choice between them is made before loading, from what is installed and what
//! the NVIDIA driver supports. Within the loaded library, devices are then tried in order, ending
//! with the CPU, which every build provides.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Result, bail};

use crate::embedding::ExecutionDevice;

#[cfg(windows)]
pub const ONNXRUNTIME_LIBRARY: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
pub const ONNXRUNTIME_LIBRARY: &str = "libonnxruntime.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
pub const ONNXRUNTIME_LIBRARY: &str = "libonnxruntime.so";

/// What the user asked for with `--device`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The fastest device that works: CUDA, then DirectML, then the CPU.
    Auto,
    Exactly(ExecutionDevice),
}

impl FromStr for DeviceChoice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        value
            .parse()
            .map(Self::Exactly)
            .map_err(|_| format!("unknown device '{value}' (expected auto, cpu, cuda, or directml)"))
    }
}

/// Where installed ONNX Runtime builds live.
#[derive(Debug, Clone)]
pub struct RuntimeLayout {
    /// Directory of the sems executable; release builds ship the DirectML runtime there.
    pub executable_directory: PathBuf,
    /// `<data dir>/runtime`, holding `directml/` and `cuda/` (tools/runtime/install_runtime.py).
    pub runtimes_directory: PathBuf,
}

impl RuntimeLayout {
    fn directml_library(&self) -> Option<PathBuf> {
        [self.executable_directory.join(ONNXRUNTIME_LIBRARY), self.flavor_library("directml")]
            .into_iter()
            .find(|path| path.is_file())
    }

    fn cuda_library(&self) -> Option<PathBuf> {
        Some(self.flavor_library("cuda")).filter(|path| path.is_file())
    }

    fn flavor_library(&self, flavor: &str) -> PathBuf {
        self.runtimes_directory.join(flavor).join(ONNXRUNTIME_LIBRARY)
    }
}

/// The library to load and the devices to try in it, best first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlan {
    pub library: PathBuf,
    pub devices: Vec<ExecutionDevice>,
}

/// Decides which runtime to load. `cuda_usable` reports whether the driver can run the CUDA
/// build installed in the given directory (see [`cuda_driver_supports`]); it is a parameter so the
/// decision can be tested without a GPU.
pub fn plan_runtime(
    choice: DeviceChoice,
    explicit_library: Option<&Path>,
    layout: &RuntimeLayout,
    cuda_usable: &dyn Fn(&Path) -> bool,
) -> Result<RuntimePlan> {
    use ExecutionDevice::{Cpu, Cuda, DirectMl};

    if let Some(library) = explicit_library {
        if !library.is_file() {
            bail!("ONNX Runtime not found at {} (from --onnxruntime or SEMS_ONNXRUNTIME)", library.display());
        }
        let devices = match choice {
            DeviceChoice::Auto => vec![Cuda, DirectMl, Cpu],
            DeviceChoice::Exactly(device) => vec![device],
        };
        return Ok(RuntimePlan { library: library.to_path_buf(), devices });
    }

    let missing = |flavor: &str| {
        anyhow::anyhow!(
            "no {flavor} ONNX Runtime installed (looked in {} and {}); install it with \
             `python tools/runtime/install_runtime.py {flavor}` or set --onnxruntime",
            layout.executable_directory.display(),
            layout.runtimes_directory.join(flavor).display()
        )
    };
    let plan = |library: PathBuf, devices: Vec<ExecutionDevice>| Ok(RuntimePlan { library, devices });
    match choice {
        DeviceChoice::Exactly(Cuda) => plan(layout.cuda_library().ok_or_else(|| missing("cuda"))?, vec![Cuda]),
        DeviceChoice::Exactly(device) => {
            plan(layout.directml_library().ok_or_else(|| missing("directml"))?, vec![device])
        }
        DeviceChoice::Auto => {
            let usable_cuda = layout.cuda_library().filter(|library| library.parent().is_some_and(cuda_usable));
            if let Some(library) = usable_cuda {
                return plan(library, vec![Cuda, Cpu]);
            }
            match (layout.directml_library(), layout.cuda_library()) {
                (Some(library), _) => plan(library, vec![DirectMl, Cpu]),
                // The CUDA build also runs on the CPU when its GPU path is unusable.
                (None, Some(library)) => plan(library, vec![Cpu]),
                (None, None) => Err(missing("directml")),
            }
        }
    }
}

/// CUDA runtime shipped in the CUDA pack (pinned in tools/runtime/install_runtime.py).
#[cfg(windows)]
const CUDA_RUNTIME_LIBRARY: &str = "cudart64_13.dll";
#[cfg(not(windows))]
const CUDA_RUNTIME_LIBRARY: &str = "libcudart.so.13";
/// The pack is built against CUDA 13; drivers report their version as `major * 1000 + minor * 10`.
const MINIMUM_DRIVER_CUDA_VERSION: i32 = 13_000;

/// Asks the NVIDIA driver, through the pack's CUDA runtime, whether it supports CUDA 13 and sees
/// a GPU. False (never an error) when there is no driver, it is too old, or no GPU is present.
pub fn cuda_driver_supports(runtime_directory: &Path) -> bool {
    type CudaQuery = unsafe extern "C" fn(*mut i32) -> i32;
    const CUDA_SUCCESS: i32 = 0;

    // SAFETY: loading the CUDA runtime runs only its own initializers. Both functions take an out
    // pointer to an int and return a status code, per the CUDA runtime API; the out pointers are
    // valid locals.
    unsafe {
        let Ok(library) = libloading::Library::new(runtime_directory.join(CUDA_RUNTIME_LIBRARY)) else {
            return false;
        };
        let (Ok(driver_version), Ok(device_count)) =
            (library.get::<CudaQuery>(b"cudaDriverGetVersion\0"), library.get::<CudaQuery>(b"cudaGetDeviceCount\0"))
        else {
            return false;
        };
        let mut version = 0;
        let mut devices = 0;
        driver_version(&mut version) == CUDA_SUCCESS
            && version >= MINIMUM_DRIVER_CUDA_VERSION
            && device_count(&mut devices) == CUDA_SUCCESS
            && devices > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ExecutionDevice::{Cpu, Cuda, DirectMl};

    struct Installed {
        _directory: tempfile::TempDir,
        layout: RuntimeLayout,
    }

    fn installed(next_to_executable: bool, directml: bool, cuda: bool) -> Installed {
        let directory = tempfile::tempdir().unwrap();
        let layout = RuntimeLayout {
            executable_directory: directory.path().join("bin"),
            runtimes_directory: directory.path().join("runtime"),
        };
        let mut libraries = Vec::new();
        if next_to_executable {
            libraries.push(layout.executable_directory.join(ONNXRUNTIME_LIBRARY));
        }
        if directml {
            libraries.push(layout.flavor_library("directml"));
        }
        if cuda {
            libraries.push(layout.flavor_library("cuda"));
        }
        for library in libraries {
            std::fs::create_dir_all(library.parent().unwrap()).unwrap();
            std::fs::write(library, b"").unwrap();
        }
        Installed { _directory: directory, layout }
    }

    fn plan(choice: DeviceChoice, installed: &Installed, cuda_works: bool) -> Result<RuntimePlan> {
        plan_runtime(choice, None, &installed.layout, &|_| cuda_works)
    }

    #[test]
    fn auto_prefers_a_working_cuda_pack() {
        let installed = installed(false, true, true);
        let plan = plan(DeviceChoice::Auto, &installed, true).unwrap();
        assert_eq!(plan.library, installed.layout.flavor_library("cuda"));
        assert_eq!(plan.devices, [Cuda, Cpu]);
    }

    #[test]
    fn auto_uses_directml_when_the_driver_cannot_run_cuda() {
        let installed = installed(false, true, true);
        let plan = plan(DeviceChoice::Auto, &installed, false).unwrap();
        assert_eq!(plan.library, installed.layout.flavor_library("directml"));
        assert_eq!(plan.devices, [DirectMl, Cpu]);
    }

    #[test]
    fn auto_falls_back_to_the_cuda_build_on_cpu_without_directml() {
        let installed = installed(false, false, true);
        let plan = plan(DeviceChoice::Auto, &installed, false).unwrap();
        assert_eq!((plan.library, plan.devices), (installed.layout.flavor_library("cuda"), vec![Cpu]));
    }

    #[test]
    fn the_runtime_next_to_the_executable_wins() {
        let installed = installed(true, true, false);
        let plan = plan(DeviceChoice::Exactly(Cpu), &installed, false).unwrap();
        assert_eq!(plan.library, installed.layout.executable_directory.join(ONNXRUNTIME_LIBRARY));
        assert_eq!(plan.devices, [Cpu]);
    }

    #[test]
    fn requesting_cuda_without_the_pack_explains_how_to_install_it() {
        let installed = installed(false, true, false);
        let error = plan(DeviceChoice::Exactly(Cuda), &installed, true).unwrap_err().to_string();
        assert!(error.contains("install_runtime.py cuda"), "unexpected error: {error}");
    }

    #[test]
    fn an_explicit_library_is_used_as_is() {
        let installed = installed(false, true, true);
        let library = installed.layout.flavor_library("directml");
        let auto = plan_runtime(DeviceChoice::Auto, Some(&library), &installed.layout, &|_| true).unwrap();
        assert_eq!((auto.library, auto.devices), (library.clone(), vec![Cuda, DirectMl, Cpu]));
        let missing = installed.layout.runtimes_directory.join("nope.dll");
        assert!(plan_runtime(DeviceChoice::Auto, Some(&missing), &installed.layout, &|_| true).is_err());
    }

    #[test]
    fn parses_device_choices() {
        assert_eq!("AUTO".parse::<DeviceChoice>().unwrap(), DeviceChoice::Auto);
        assert_eq!("cuda".parse::<DeviceChoice>().unwrap(), DeviceChoice::Exactly(Cuda));
        assert!("tpu".parse::<DeviceChoice>().is_err());
    }

    #[test]
    fn cuda_probe_is_false_without_a_cuda_runtime() {
        let directory = tempfile::tempdir().unwrap();
        assert!(!cuda_driver_supports(directory.path()));
    }
}
