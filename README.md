# sems

Semantic search for your files: like `grep`, but it matches meaning instead of exact text. It
searches code, documents, PDFs, photos, audio, and video, all locally.

```console
> sems index
> sems "how do we retry failed requests"
code\retry.ts:1-15  0.82
 1: export async function fetchWithBackoff(url: string, attempts = 5): Promise<Response> {
 2:   let delayMs = 250;
 ...

> sems "can I have a dog in my flat"
docs\lease_agreement.pdf  page 2  0.73
    2. Pets
    The Tenant may keep one cat or one small dog with the Landlord's written consent.
    ...

> sems "a rocket launch"
videos\clip_0001.mp4  0:09-0:18  0.74  [video]

photos\IMG_0003.jpg  0.73  [image 512x342]

> sems "when is the dentist appointment"
audio\memo_0003.m4a  0:00-0:30  0.69  [audio]
```

Powered by [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2). Nothing leaves your
machine.

## Usage

```text
sems <QUERY> [PATH]        search under PATH (default: current directory)
    -n, --limit <N>        number of results (default 10)
    -l                     print matching file paths only
        --json             results as JSON, for scripts and coding agents
        --kind <KIND>      only text, image, pdf, audio, or video
        --all              also show weak results
sems index [PATH]          index PATH, or update it after changes
        --skip-images, --skip-audio, --skip-video
        --rebuild          start the index over
sems status [PATH]         what is indexed under PATH
```

- Indexing skips whatever `.gitignore` or a `.semsignore` file lists, plus hidden and binary files.
  Re-indexing only processes files that changed.
- Searches show only the results that clearly stand out; `--all` shows everything.
- Result paths are clickable in terminals that support links.
- Audio and video need [ffmpeg](https://ffmpeg.org) on your `PATH` (`winget install Gyan.FFmpeg`).
- Indexing uses your GPU when it can. Photos, audio, and video are slow to index without one.

## Install (development)

1. **Export the model** (one time, needs Python 3.13):

   ```console
   python -m venv tools/export/.venv
   tools/export/.venv/Scripts/pip install -r tools/export/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   tools/export/.venv/Scripts/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-2', local_dir='models/embeddinggemma-2')"
   tools/export/.venv/Scripts/python tools/export/make_reference.py --model models/embeddinggemma-2 --out models/reference.json --images tests/fixtures/images/beach.png --audio tests/fixtures/eval_corpus/audio/memo_0001.wav tests/fixtures/eval_corpus/audio/memo_0003.m4a --videos tests/fixtures/eval_corpus/videos/clip_0001.mp4
   tools/export/.venv/Scripts/python tools/export/export_onnx.py --model models/embeddinggemma-2 --out models/onnx --reference models/reference.json --image tests/fixtures/images/beach.png --audio tests/fixtures/eval_corpus/audio/memo_0001.wav tests/fixtures/eval_corpus/audio/memo_0003.m4a --video tests/fixtures/eval_corpus/videos/clip_0001.mp4
   ```

2. **Install** (Windows): `powershell -ExecutionPolicy Bypass -File tools\install\install.ps1`.
   It builds sems, sets up the model and runtime, and puts `sems` on your PATH. Add `-Cuda` for
   faster indexing on NVIDIA GPUs, or `-Uninstall` to remove it.

Everything lives in `%LOCALAPPDATA%\sems`. To use other locations, set `SEMS_INDEX`,
`SEMS_MODEL_DIR`, `SEMS_ONNXRUNTIME`, or `SEMS_FFMPEG` (or the matching flags).

## Tests

```console
cargo test --release
SEMS_ONNXRUNTIME=$LOCALAPPDATA/sems/runtime/directml/onnxruntime.dll cargo test --release -- --ignored --test-threads=1
```

The first needs no model. The second checks the model against the official implementation and
measures search quality on the labelled files in `tests/fixtures`.
