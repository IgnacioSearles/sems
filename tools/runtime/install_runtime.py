"""Install an ONNX Runtime flavor into sems' runtime directory.

    directml  ONNX Runtime 1.24.4 with DirectML (~40 MB). The default runtime: it also serves CPU,
              so searches and non-CUDA indexing use it.
    cuda      ONNX Runtime 1.30 with CUDA 13 and cuDNN 9 (~890 MB). Optional, for fast indexing on
              NVIDIA GPUs; needs driver 580+.

Downloads the pinned wheels with pip and extracts only the DLLs inference actually loads (measured
by listing modules loaded during CUDA inference; delay-loaded extras such as cuFFT are left out).
Prototype of a future `sems gpu install`.

Usage:
    python install_runtime.py directml|cuda [--destination DIR]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

RUNTIMES: dict[str, dict[str, list[str]]] = {
    "directml": {
        "onnxruntime-directml==1.24.4": ["onnxruntime.dll", "onnxruntime_providers_shared.dll", "DirectML.dll"],
    },
    "cuda": {
        "onnxruntime-gpu==1.30.0": [
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_cuda.dll",
        ],
        "nvidia-cuda-runtime==13.4.92": ["cudart64_13.dll"],
        "nvidia-cublas==13.8.0.4": ["cublas64_13.dll", "cublasLt64_13.dll"],
        "nvidia-cudnn-cu13==9.27.0.42": ["cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll"],
    },
}


def default_destination(flavor: str) -> Path:
    local_data = os.environ.get("LOCALAPPDATA")
    if not local_data:
        raise SystemExit("LOCALAPPDATA is not set; pass --destination")
    return Path(local_data) / "sems" / "runtime" / flavor


def download_wheel(requirement: str, directory: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--progress-bar", "off",
         "--disable-pip-version-check", "--dest", str(directory), requirement],
        check=True,
    )
    name = requirement.split("==")[0].replace("-", "_").lower()
    wheels = [wheel for wheel in directory.glob("*.whl") if wheel.name.lower().startswith(name + "-")]
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel for {requirement}, found {[wheel.name for wheel in wheels]}")
    return wheels[0]


def extract_files(wheel: Path, file_names: list[str], destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = {Path(member).name: member for member in archive.namelist()}
        for file_name in file_names:
            if file_name not in members:
                raise SystemExit(f"{wheel.name} does not contain {file_name}")
            with archive.open(members[file_name]) as source, open(destination / file_name, "wb") as target:
                shutil.copyfileobj(source, target)
            print(f"  {file_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flavor", choices=sorted(RUNTIMES))
    parser.add_argument("--destination", type=Path)
    arguments = parser.parse_args()
    destination = arguments.destination or default_destination(arguments.flavor)

    # Stage into a sibling directory and swap at the end, so a failed download never leaves a
    # half-populated runtime that sems would try to load.
    staging = destination.with_name(destination.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as download_directory:
        for requirement, file_names in RUNTIMES[arguments.flavor].items():
            print(f"{requirement}:")
            extract_files(download_wheel(requirement, Path(download_directory)), file_names, staging)
    shutil.rmtree(destination, ignore_errors=True)
    staging.rename(destination)
    total = sum(path.stat().st_size for path in destination.iterdir())
    print(f"installed {arguments.flavor} runtime into {destination} ({total / 1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
