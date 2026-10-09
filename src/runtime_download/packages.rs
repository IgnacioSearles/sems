//! Pinned ONNX Runtime packages: where each archive comes from, its SHA-256, and which libraries
//! sems takes from it. Sizes and paths were read from the archives; the network tests in the
//! parent module download and check them.

use super::{ArchiveFormat, Member, RuntimeArchive, RuntimePackage};
#[cfg(any(all(windows, target_arch = "x86_64"), all(target_os = "linux", target_arch = "x86_64")))]
use crate::device::ONNXRUNTIME_LIBRARY;
use crate::download::PinnedFile;

const fn member(archive_path: &'static str, file_name: &'static str, size: u64) -> Member {
    Member { archive_path, file_name, size }
}

/// A Python wheel on PyPI; `url` ends with `name`. (macOS packages are all tarballs.)
#[cfg_attr(target_os = "macos", allow(dead_code))]
const fn wheel(
    url: &'static str,
    name: &'static str,
    size: u64,
    sha256: &'static str,
    members: &'static [Member],
) -> RuntimeArchive {
    RuntimeArchive { url, file: PinnedFile { name, size, sha256 }, format: ArchiveFormat::Zip, members }
}

// --- Default runtime ------------------------------------------------------------------------------

/// ONNX Runtime 1.24.4 with DirectML, the last DirectML release (see Cargo.toml).
#[cfg(all(windows, target_arch = "x86_64"))]
pub(super) const DEFAULT: Option<RuntimePackage> = Some(RuntimePackage {
    archives: &[wheel(
        "https://files.pythonhosted.org/packages/60/53/2bd2696fac19cf8ca55496a0bcfe431f3aff9579eabbb0e231dc238acf6f/onnxruntime_directml-1.24.4-cp313-cp313-win_amd64.whl",
        "onnxruntime_directml-1.24.4-cp313-cp313-win_amd64.whl",
        25_112_253,
        "2f1031cb2281e5b27cca9efe0b9399317c7286e4d226f7a79d4ab79bbd94d19e",
        &[
            member("onnxruntime/capi/onnxruntime.dll", ONNXRUNTIME_LIBRARY, 21_111_840),
            member("onnxruntime/capi/onnxruntime_providers_shared.dll", "onnxruntime_providers_shared.dll", 21_536),
            member("onnxruntime/capi/DirectML.dll", "DirectML.dll", 18_527_768),
        ],
    )],
});

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(super) const DEFAULT: Option<RuntimePackage> = Some(RuntimePackage {
    archives: &[RuntimeArchive {
        url: "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz",
        file: PinnedFile {
            name: "onnxruntime-linux-x64-1.30.0.tgz",
            size: 11_306_877,
            sha256: "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd",
        },
        format: ArchiveFormat::TarGz,
        members: &[member(
            "onnxruntime-linux-x64-1.30.0/lib/libonnxruntime.so.1.30.0",
            ONNXRUNTIME_LIBRARY,
            28_985_152,
        )],
    }],
});

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) const DEFAULT: Option<RuntimePackage> = Some(RuntimePackage {
    archives: &[RuntimeArchive {
        url: "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-osx-arm64-1.30.0.tgz",
        file: PinnedFile {
            name: "onnxruntime-osx-arm64-1.30.0.tgz",
            size: 42_373_116,
            sha256: "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012",
        },
        format: ArchiveFormat::TarGz,
        members: &[member(
            "onnxruntime-osx-arm64-1.30.0/lib/libonnxruntime.1.30.0.dylib",
            crate::device::ONNXRUNTIME_LIBRARY,
            43_879_424,
        )],
    }],
});

#[cfg(not(any(
    all(windows, target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
pub(super) const DEFAULT: Option<RuntimePackage> = None;

// --- CUDA pack: ONNX Runtime 1.30 with CUDA 13 and cuDNN 9 (NVIDIA driver 580+) ------------------
//
// Only the libraries inference loads. On Windows they were found by listing the modules loaded
// during CUDA inference, then removing cuDNN libraries one at a time until audio (convolutions)
// failed; cuFFT and cuRAND are delay-loaded there and never used. On Linux the CUDA provider links
// cuRAND directly, so it is included.

#[cfg(all(windows, target_arch = "x86_64"))]
pub(super) const CUDA: Option<RuntimePackage> = Some(RuntimePackage {
    archives: &[
        wheel(
            "https://files.pythonhosted.org/packages/7e/d3/50fba71f4174f25f1e677ffe4b2ebc9f6c2a2aa8093c12e82c93792fd4e8/onnxruntime_gpu-1.30.0-cp313-cp313-win_amd64.whl",
            "onnxruntime_gpu-1.30.0-cp313-cp313-win_amd64.whl",
            160_479_611,
            "cf494da233f3fc02bbeb2a3178e3ace7361b6b4fef7031b19c1e5aed5e9328ce",
            &[
                member("onnxruntime/capi/onnxruntime.dll", ONNXRUNTIME_LIBRARY, 18_036_576),
                member("onnxruntime/capi/onnxruntime_providers_shared.dll", "onnxruntime_providers_shared.dll", 21_816),
                member(
                    "onnxruntime/capi/onnxruntime_providers_cuda.dll",
                    "onnxruntime_providers_cuda.dll",
                    184_761_656,
                ),
            ],
        ),
        wheel(
            "https://files.pythonhosted.org/packages/86/00/d5436004268f049214193659ebc36550b5ef3925c3d13b4cc980e13be6f5/nvidia_cuda_runtime-13.4.92-py3-none-win_amd64.whl",
            "nvidia_cuda_runtime-13.4.92-py3-none-win_amd64.whl",
            2_778_543,
            "08dca5e4aba480c2fd5b55075c0fa71b84ef9dcf0521f2d58baa14a803a7311c",
            &[member("nvidia/cu13/bin/x86_64/cudart64_13.dll", "cudart64_13.dll", 551_024)],
        ),
        wheel(
            "https://files.pythonhosted.org/packages/a3/df/f1246959833e2c437db8be3e5b477f66b87f8817821ed40de6c7561c9a36/nvidia_cublas-13.8.0.4-py3-none-win_amd64.whl",
            "nvidia_cublas-13.8.0.4-py3-none-win_amd64.whl",
            423_266_897,
            "8c5494423bb8a46822cb6b0cb95d7fa4be2d7b96a31155dff083839ec8297910",
            &[
                member("nvidia/cu13/bin/x86_64/cublas64_13.dll", "cublas64_13.dll", 54_873_200),
                member("nvidia/cu13/bin/x86_64/cublasLt64_13.dll", "cublasLt64_13.dll", 493_474_416),
            ],
        ),
        wheel(
            "https://files.pythonhosted.org/packages/87/6a/e55ff0ac26a5c6e2b21f41c9d04ad096b4ed6da593fba7e25845c61b0532/nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl",
            "nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl",
            436_469_905,
            "7d96f634adafd55c72231eb0500ca77ab109ec8ebff7b33000b76e081bc4558e",
            &[
                member("nvidia/cudnn/bin/cudnn64_9.dll", "cudnn64_9.dll", 270_448),
                member("nvidia/cudnn/bin/cudnn_graph64_9.dll", "cudnn_graph64_9.dll", 116_132_464),
                member("nvidia/cudnn/bin/cudnn_ops64_9.dll", "cudnn_ops64_9.dll", 37_462_128),
                member("nvidia/cudnn/bin/cudnn_heuristic64_9.dll", "cudnn_heuristic64_9.dll", 94_795_376),
                member(
                    "nvidia/cudnn/bin/cudnn_engines_precompiled64_9.dll",
                    "cudnn_engines_precompiled64_9.dll",
                    231_660_656,
                ),
                member(
                    "nvidia/cudnn/bin/cudnn_engines_runtime_compiled64_9.dll",
                    "cudnn_engines_runtime_compiled64_9.dll",
                    39_266_928,
                ),
                member("nvidia/cudnn/bin/cudnn_engines_tensor_ir64_9.dll", "cudnn_engines_tensor_ir64_9.dll", 156_272),
            ],
        ),
    ],
});

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(super) const CUDA: Option<RuntimePackage> = Some(RuntimePackage {
    archives: &[
        wheel(
            "https://files.pythonhosted.org/packages/31/ae/cfec0c21d039e4125fdc333e2e2032ed0a49eb50f9dd70b7f3819b339018/onnxruntime_gpu-1.30.0-cp313-cp313-manylinux_2_28_x86_64.whl",
            "onnxruntime_gpu-1.30.0-cp313-cp313-manylinux_2_28_x86_64.whl",
            246_688_669,
            "9aba23f903f434cc55d851aeb801afee19e287c202f4ab8f59d02f04caa90aed",
            &[
                member("onnxruntime/capi/libonnxruntime.so.1.30.0", ONNXRUNTIME_LIBRARY, 29_062_976),
                member(
                    "onnxruntime/capi/libonnxruntime_providers_shared.so",
                    "libonnxruntime_providers_shared.so",
                    14_632,
                ),
                member(
                    "onnxruntime/capi/libonnxruntime_providers_cuda.so",
                    "libonnxruntime_providers_cuda.so",
                    272_054_000,
                ),
            ],
        ),
        wheel(
            "https://files.pythonhosted.org/packages/98/8a/3431271f6344874b8f1ac03f16b3d679c91493f8da63f716160403e6d0a0/nvidia_cuda_runtime-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "nvidia_cuda_runtime-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            2_494_438,
            "9641f797da20ce1dd8e779b6e96d08cf9ba564cec8e8225458811ee26423f3a5",
            &[member("nvidia/cu13/lib/libcudart.so.13", "libcudart.so.13", 798_496)],
        ),
        wheel(
            "https://files.pythonhosted.org/packages/7a/38/bdd540bf511d2c9b6f9efc71a81c60cb88e295be0b9312b61d19bbed2212/nvidia_cublas-13.8.0.4-py3-none-manylinux_2_27_x86_64.whl",
            "nvidia_cublas-13.8.0.4-py3-none-manylinux_2_27_x86_64.whl",
            439_317_144,
            "9f17797dfcc048694461f4e47de17d2e3c25adf172ef723d2db0a07cd8744b89",
            &[
                member("nvidia/cu13/lib/libcublas.so.13", "libcublas.so.13", 57_594_520),
                member("nvidia/cu13/lib/libcublasLt.so.13", "libcublasLt.so.13", 545_528_240),
            ],
        ),
        wheel(
            "https://files.pythonhosted.org/packages/07/73/3ee8e5b4cb891401e603ffd3a59b35c6afe785fd2de123afe7c7029603dc/nvidia_curand-10.4.4.72-py3-none-manylinux_2_27_x86_64.whl",
            "nvidia_curand-10.4.4.72-py3-none-manylinux_2_27_x86_64.whl",
            61_498_332,
            "25c3457ae7a224fdd484dab90b0fc5dc0e842fab5db3012afa4a5bd2af4eb7e5",
            &[member("nvidia/cu13/lib/libcurand.so.10", "libcurand.so.10", 126_468_312)],
        ),
        wheel(
            "https://files.pythonhosted.org/packages/af/75/96ea5c5368eb595c39d629cde08a66227a864e71ccb0e593add2612bb952/nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl",
            "nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl",
            536_771_498,
            "9677e76f21862eb5da7ee5ed69d544738b2d8b5c3ce7e5ec125c5592e6cdbdc8",
            &[
                member("nvidia/cudnn/lib/libcudnn.so.9", "libcudnn.so.9", 133_336),
                member("nvidia/cudnn/lib/libcudnn_graph.so.9", "libcudnn_graph.so.9", 134_437_112),
                member("nvidia/cudnn/lib/libcudnn_ops.so.9", "libcudnn_ops.so.9", 39_855_912),
                member("nvidia/cudnn/lib/libcudnn_heuristic.so.9", "libcudnn_heuristic.so.9", 96_473_904),
                member(
                    "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9",
                    "libcudnn_engines_precompiled.so.9",
                    272_654_912,
                ),
                member(
                    "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
                    "libcudnn_engines_runtime_compiled.so.9",
                    44_075_680,
                ),
                member(
                    "nvidia/cudnn/lib/libcudnn_engines_tensor_ir.so.9",
                    "libcudnn_engines_tensor_ir.so.9",
                    216_026_256,
                ),
            ],
        ),
    ],
});

#[cfg(not(any(all(windows, target_arch = "x86_64"), all(target_os = "linux", target_arch = "x86_64"))))]
pub(super) const CUDA: Option<RuntimePackage> = None;
