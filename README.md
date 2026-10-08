# sems

Semantic search for local files — like `grep`, but it matches meaning instead of exact text.

```console
$ sems index ~/projects/shop
indexed /home/me/projects/shop: 1,204 files (0 unchanged, 1,204 embedded into 5,310 chunks, 0 removed) in 41.2s [256d]

$ sems "try again with increasing delays when a request fails" ~/projects/shop
src/http/retry.ts:1-15  0.71
    export async function fetchWithBackoff(url: string, attempts = 5): Promise<Response> {
      let delayMs = 250;
```

Powered by [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), running locally on
ONNX Runtime. Nothing leaves your machine.

## Status

Milestone 1: text and code. Images, PDFs, video, and audio come next (the runtime already embeds
images; they are not indexed yet).

## Usage

```text
sems <QUERY> [PATH]           search under PATH (default: current directory)
    -n, --limit <N>           number of results (default 10)
    -l, --files-with-matches  print matching file paths only, best first
        --json                full results as JSON, for scripts and coding agents
sems index [PATH]             index or incrementally update PATH
        --device <cpu|cuda|directml>
        --rebuild             discard the index and start over
sems status [PATH]            what is indexed under PATH
```

Indexing respects `.gitignore`, skips hidden, binary and large (>1 MB) files, and reads a
`.semsignore` file (same syntax) for anything else to leave out. Re-indexing only embeds files whose
content changed.

Searches always run on the CPU (~1.2 s including model load). `--device` speeds up indexing:
roughly 3x with DirectML and 11x with CUDA compared to CPU on an RTX 3050 Ti.

## Setup (development)

sems keeps everything under `%LOCALAPPDATA%\sems` (`<local data dir>/sems` elsewhere):

```text
model\                 exported EmbeddingGemma 2 graphs, tokenizer and config
runtime\directml\      default ONNX Runtime (~40 MB); also serves CPU, so searches use it
runtime\cuda\          optional CUDA pack (~900 MB), used by `sems index --device cuda`
index.db               the index
```

1. **Export the model** (one time, needs Python 3.13):

   ```console
   python -m venv tools/export/.venv
   tools/export/.venv/Scripts/pip install -r tools/export/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   tools/export/.venv/Scripts/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-2', local_dir='models/embeddinggemma-2')"
   tools/export/.venv/Scripts/python tools/export/make_reference.py --model models/embeddinggemma-2 --out models/reference.json --images tests/fixtures/images/beach.png
   tools/export/.venv/Scripts/python tools/export/export_onnx.py --model models/embeddinggemma-2 --out models/onnx --reference models/reference.json --image tests/fixtures/images/beach.png
   ```

   The export fails unless every graph reproduces the official sentence-transformers embeddings.
   Then copy `models/onnx/*` into `%LOCALAPPDATA%\sems\model\`.

2. **Install the runtimes**: `python tools/runtime/install_runtime.py directml`, plus
   `python tools/runtime/install_runtime.py cuda` for NVIDIA GPUs (CUDA 13 needs driver 580+). Each
   runtime keeps its dependencies in its own folder; nothing goes on `PATH`.

3. **Build and run**: `cargo build --release`, then `target/release/sems index`.

Every location can be overridden (flags win over environment variables):

| Setting | Flag | Environment variable |
|---|---|---|
| Model directory | `--model-dir` | `SEMS_MODEL_DIR` |
| ONNX Runtime library (any 1.24+ build) | `--onnxruntime` | `SEMS_ONNXRUNTIME` |
| Index database | `--index` | `SEMS_INDEX` |
| Indexing device (`cpu`, `cuda`, `directml`) | `--device` | `SEMS_DEVICE` |

sems always loads ONNX Runtime by explicit path: Windows ships an outdated `onnxruntime.dll` in
System32 that loading by name could pick up.

## Tests

```console
cargo test --release                       # unit and indexing tests (fake encoder, no model needed)
SEMS_ONNXRUNTIME=$LOCALAPPDATA/sems/runtime/directml/onnxruntime.dll cargo test --release -- --ignored --test-threads=1
```

The ignored tests need the exported model: `reference_embeddings` checks numerical parity with the
official pipeline (set `SEMS_DEVICE` to test a GPU provider), and `search_quality` measures recall
on a labelled corpus in `tests/fixtures`. `tools/bench/benchmark_providers.py` compares execution
providers.
