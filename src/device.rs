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

/// The runtime sems downloads by default (see runtime_download.rs) and the devices it serves: the
/// DirectML build on Windows, which also runs on the CPU, and the CPU build elsewhere.
#[cfg(windows)]
pub const DEFAULT_RUNTIME_FLAVOR: &str = "directml";
#[cfg(windows)]
const DEFAULT_RUNTIME_DEVICES: [ExecutionDevice; 2] = [ExecutionDevice::DirectMl, ExecutionDevice::Cpu];
#[cfg(not(windows))]
pub const DEFAULT_RUNTIME_FLAVOR: &str = "cpu";
#[cfg(not(windows))]
const DEFAULT_RUNTIME_DEVICES: [ExecutionDevice; 1] = [ExecutionDevice::Cpu];

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
    /// Directory of the sems executable; a runtime placed there is used as the default one.
    pub executable_directory: PathBuf,
    /// `<data dir>/runtime`, holding the default runtime (`directml/` or `cpu/`) and the optional
    /// `cuda/` pack (tools/runtime/install_runtime.py).
    pub runtimes_directory: PathBuf,
}

impl RuntimeLayout {
    /// A runtime placed next to the sems executable, which takes precedence over a download.
    pub fn bundled_library(&self) -> Option<PathBuf> {
        Some(self.executable_directory.join(ONNXRUNTIME_LIBRARY)).filter(|path| path.is_file())
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
/// build installed in the given directory (see [`cuda_driver_supports`]), and `default_runtime`
/// provides the default runtime's library, downloading it if needed; both are parameters so the
/// decision can be tested without a GPU or a network, and nothing is downloaded unless the plan
/// uses it.
pub fn plan_runtime(
    choice: DeviceChoice,
    explicit_library: Option<&Path>,
    layout: &RuntimeLayout,
    cuda_usable: &dyn Fn(&Path) -> bool,
    default_runtime: &dyn Fn() -> Result<PathBuf>,
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

    let plan = |library: PathBuf, devices: Vec<ExecutionDevice>| Ok(RuntimePlan { library, devices });
    match choice {
        DeviceChoice::Exactly(Cuda) => {
            let library = layout.cuda_library().ok_or_else(|| {
                anyhow::anyhow!(
                    "the CUDA pack is not installed in {}; install it with \
                     `python tools/runtime/install_runtime.py cuda` or set --onnxruntime",
                    layout.runtimes_directory.join("cuda").display()
                )
            })?;
            plan(library, vec![Cuda])
        }
        DeviceChoice::Exactly(device) => plan(default_runtime()?, vec![device]),
        DeviceChoice::Auto => {
            let usable_cuda = layout.cuda_library().filter(|library| library.parent().is_some_and(cuda_usable));
            if let Some(library) = usable_cuda {
                return plan(library, vec![Cuda, Cpu]);
            }
            match (default_runtime(), layout.cuda_library()) {
                (Ok(library), _) => plan(library, DEFAULT_RUNTIME_DEVICES.to_vec()),
                // The CUDA build also runs on the CPU, for example while offline before the
                // default runtime was ever downloaded.
                (Err(_), Some(library)) => plan(library, vec![Cpu]),
                (Err(error), None) => Err(error),
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

    fn installed(cuda: bool) -> Installed {
        let directory = tempfile::tempdir().unwrap();
        let layout = RuntimeLayout {
            executable_directory: directory.path().join("bin"),
            runtimes_directory: directory.path().join("runtime"),
        };
        if cuda {
            let library = layout.flavor_library("cuda");
            std::fs::create_dir_all(library.parent().unwrap()).unwrap();
            std::fs::write(library, b"").unwrap();
        }
        Installed { _directory: directory, layout }
    }

    fn default_library(installed: &Installed) -> PathBuf {
        installed.layout.flavor_library(DEFAULT_RUNTIME_FLAVOR)
    }

    /// Plans with a default runtime that is available (`default_available`) or cannot be had.
    fn plan(
        choice: DeviceChoice,
        installed: &Installed,
        cuda_works: bool,
        default_available: bool,
    ) -> Result<RuntimePlan> {
        let default = default_library(installed);
        let default_runtime =
            move || if default_available { Ok(default.clone()) } else { Err(anyhow::anyhow!("offline")) };
        plan_runtime(choice, None, &installed.layout, &|_| cuda_works, &default_runtime)
    }

    #[test]
    fn auto_prefers_a_working_cuda_pack() {
        let installed = installed(true);
        let plan = plan(DeviceChoice::Auto, &installed, true, true).unwrap();
        assert_eq!(plan.library, installed.layout.flavor_library("cuda"));
        assert_eq!(plan.devices, [Cuda, Cpu]);
    }

    #[test]
    fn auto_uses_the_default_runtime_when_the_driver_cannot_run_cuda() {
        let installed = installed(true);
        let plan = plan(DeviceChoice::Auto, &installed, false, true).unwrap();
        assert_eq!(plan.library, default_library(&installed));
        assert_eq!(plan.devices, DEFAULT_RUNTIME_DEVICES);
    }

    #[test]
    fn auto_falls_back_to_the_cuda_build_on_cpu_without_the_default_runtime() {
        let installed = installed(true);
        let plan = plan(DeviceChoice::Auto, &installed, false, false).unwrap();
        assert_eq!((plan.library, plan.devices), (installed.layout.flavor_library("cuda"), vec![Cpu]));
        assert!(self::plan(DeviceChoice::Auto, &self::installed(false), false, false).is_err());
    }

    #[test]
    fn a_working_cuda_pack_needs_no_default_runtime() {
        let installed = installed(true);
        let never = || -> Result<PathBuf> { panic!("the default runtime was requested") };
        plan_runtime(DeviceChoice::Auto, None, &installed.layout, &|_| true, &never).unwrap();
    }

    #[test]
    fn requesting_cuda_without_the_pack_explains_how_to_install_it() {
        let installed = installed(false);
        let error = plan(DeviceChoice::Exactly(Cuda), &installed, true, true).unwrap_err().to_string();
        assert!(error.contains("install_runtime.py cuda"), "unexpected error: {error}");
    }

    #[test]
    fn an_explicit_library_is_used_as_is() {
        let installed = installed(true);
        let library = installed.layout.flavor_library("cuda");
        let never = || -> Result<PathBuf> { panic!("the default runtime was requested") };
        let auto = plan_runtime(DeviceChoice::Auto, Some(&library), &installed.layout, &|_| true, &never).unwrap();
        assert_eq!((auto.library, auto.devices), (library.clone(), vec![Cuda, DirectMl, Cpu]));
        let missing = installed.layout.runtimes_directory.join("nope.dll");
        assert!(plan_runtime(DeviceChoice::Auto, Some(&missing), &installed.layout, &|_| true, &never).is_err());
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
