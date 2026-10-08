"""Benchmark ONNX Runtime execution providers on the exported EmbeddingGemma 2 graphs.

Needs only onnxruntime (any flavor: cpu, gpu, directml) and numpy, so it can run in a separate
environment per provider. Correctness is checked against models/reference.json before any
timing is reported: a fast provider that drifts from the reference is useless.

Usage:
    python benchmark_providers.py --models ../../models --provider cpu|cuda|directml
"""

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime

MINIMUM_COSINE_SIMILARITY = 0.9999
DEFAULT_TEXT_BATCH_SIZE = 32
TEXT_SEQUENCE_LENGTH = 300
TIMED_REPETITIONS = 3


def provider_list(provider: str) -> list:
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    if provider == "directml":
        # Laptops expose an integrated GPU too; ask DirectML for the high-performance adapter.
        return [("DmlExecutionProvider", {"performance_preference": "high_performance"}), "CPUExecutionProvider"]
    raise ValueError(f"unknown provider {provider}")


def session_options(provider: str) -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    if provider == "directml":
        # Required by the DirectML provider.
        options.enable_mem_pattern = False
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return options


class GpuMemoryMonitor:
    """Samples NVIDIA GPU memory in the background; reports peak usage above the starting baseline."""

    def __init__(self) -> None:
        self.baseline = self._sample()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _sample() -> int:
        try:
            output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5, check=True).stdout
            return int(output.split()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return 0

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            self.peak = max(self.peak, self._sample())

    def __enter__(self) -> "GpuMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()


@dataclass
class Graphs:
    token_embedder: onnxruntime.InferenceSession
    text_encoder: onnxruntime.InferenceSession
    vision_encoder: onnxruntime.InferenceSession
    load_seconds: dict[str, float]


def load_graphs(model_directory: Path, provider: str) -> Graphs:
    load_seconds = {}

    def load(name: str, providers: list) -> onnxruntime.InferenceSession:
        started = time.perf_counter()
        session = onnxruntime.InferenceSession(str(model_directory / f"{name}.onnx"), session_options(provider),
                                               providers=providers)
        load_seconds[name] = time.perf_counter() - started
        return session

    graphs = Graphs(
        token_embedder=load("token_embedder", ["CPUExecutionProvider"]),
        text_encoder=load("text_encoder", provider_list(provider)),
        vision_encoder=load("vision_encoder", provider_list(provider)),
        load_seconds=load_seconds,
    )
    active = graphs.text_encoder.get_providers()[0]
    if provider != "cpu" and active == "CPUExecutionProvider":
        raise SystemExit(f"{provider} provider unavailable; session fell back to CPU")
    return graphs


def embed(graphs: Graphs, input_ids: np.ndarray, image_token_id: int,
          pixel_values: np.ndarray | None = None, position_ids: np.ndarray | None = None) -> np.ndarray:
    inputs_embeds = graphs.token_embedder.run(None, {"input_ids": input_ids})[0]
    if pixel_values is not None:
        soft_tokens = graphs.vision_encoder.run(None, {"pixel_values": pixel_values, "position_ids": position_ids})[0]
        inputs_embeds[input_ids == image_token_id] = soft_tokens
    attention_mask = np.ones_like(input_ids)
    return graphs.text_encoder.run(None, {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask})[0]


def read_tensor(models: Path, description: dict) -> np.ndarray:
    return np.fromfile(models / description["file"], dtype=np.dtype(description["dtype"]).newbyteorder("<")) \
        .reshape(description["shape"])


def check_parity(graphs: Graphs, models: Path, image_token_id: int) -> float:
    worst = 1.0
    for reference in json.loads((models / "reference.json").read_text(encoding="utf-8")):
        input_ids = np.array([reference["input_ids"]], dtype=np.int64)
        if reference["kind"] == "image":
            actual = embed(graphs, input_ids, image_token_id,
                           read_tensor(models, reference["pixel_values"]), read_tensor(models, reference["position_ids"]))
        else:
            actual = embed(graphs, input_ids, image_token_id)
        worst = min(worst, float(actual[0] @ np.array(reference["embedding"], dtype=np.float32)))
    return worst


def time_best_of(function) -> float:
    function()  # warm-up: first run includes kernel selection and allocations
    timings = []
    for _ in range(TIMED_REPETITIONS):
        started = time.perf_counter()
        function()
        timings.append(time.perf_counter() - started)
    return min(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda", "directml"], required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TEXT_BATCH_SIZE)
    arguments = parser.parse_args()
    batch_size = arguments.batch_size
    model_directory = arguments.models / "onnx"
    if arguments.provider == "cuda":
        # Loads CUDA/cuDNN DLLs installed as nvidia-* pip packages; no system CUDA install needed.
        onnxruntime.preload_dlls()
    image_token_id = json.loads((model_directory / "config.json").read_text())["image_token_id"]

    with GpuMemoryMonitor() as memory:
        graphs = load_graphs(model_directory, arguments.provider)
        worst = check_parity(graphs, arguments.models, image_token_id)
        status = "OK" if worst >= MINIMUM_COSINE_SIMILARITY else "FAIL"
        print(f"provider {arguments.provider} ({graphs.text_encoder.get_providers()[0]}), onnxruntime {onnxruntime.__version__}")
        print(f"parity: worst cosine similarity {worst:.6f} ({status})")
        if status == "FAIL":
            raise SystemExit("provider diverges from the reference; timings would be meaningless")
        for name, seconds in graphs.load_seconds.items():
            print(f"load {name:15} {seconds * 1000:7.0f} ms")

        rng = np.random.default_rng(0)
        query_ids = rng.integers(3, 200_000, (1, 12), dtype=np.int64)
        batch_ids = rng.integers(3, 200_000, (batch_size, TEXT_SEQUENCE_LENGTH), dtype=np.int64)
        query_seconds = time_best_of(lambda: embed(graphs, query_ids, image_token_id))
        batch_seconds = time_best_of(lambda: embed(graphs, batch_ids, image_token_id))
        tokens_per_second = batch_size * TEXT_SEQUENCE_LENGTH / batch_seconds
        print(f"query (12 tokens)        {query_seconds * 1000:7.0f} ms")
        print(f"text batch {batch_size:>2}x{TEXT_SEQUENCE_LENGTH}       {batch_seconds * 1000:7.0f} ms  "
              f"({tokens_per_second:,.0f} tokens/s)")

        image = next(r for r in json.loads((arguments.models / "reference.json").read_text(encoding="utf-8"))
                     if r["kind"] == "image")
        image_ids = np.array([image["input_ids"]], dtype=np.int64)
        pixels = read_tensor(arguments.models, image["pixel_values"])
        positions = read_tensor(arguments.models, image["position_ids"])
        image_seconds = time_best_of(lambda: embed(graphs, image_ids, image_token_id, pixels, positions))
        print(f"image (266 soft tokens)  {image_seconds * 1000:7.0f} ms  ({1 / image_seconds:.1f} images/s)")
    print(f"gpu memory peak          {memory.peak - memory.baseline:7,} MB above baseline")


if __name__ == "__main__":
    main()
