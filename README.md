# sems

Semantic search for local files — like `grep`, but it matches meaning instead of exact text.

```console
> sems index
indexed C:\...\sems: 61 files (0 unchanged, 61 embedded (11 images) into 796 chunks, 0 removed) in 20.3s [cuda, 256d]

> sems -n 1 "how is the dll search path set for cudnn"
src\embedding\onnx.rs:46-67  0.82
46: /// Makes `directory` part of the DLL search order for libraries loaded by name.
47: ///
48: /// ONNX Runtime's CUDA provider loads cuDNN with a bare `LoadLibrary("cudnn64_9.dll")` at first
...
52: #[cfg(windows)]
53: fn search_directory_for_dependencies(directory: &Path) -> std::io::Result<()> {
...
```

Each result is the matching unit — a function, a markdown section, a paragraph — shown in full
with line numbers. Files are split at their own structure (blank lines before less-indented
code, headings), so results point at the lines that matched rather than a fixed-size window.

Powered by [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), running locally on
ONNX Runtime. Nothing leaves your machine.

Photos are searchable by what they show, in any of the model's 100+ languages:

```console
> sems --kind image "space"
tests\fixtures\eval_corpus\photos\IMG_0004.jpg  0.78  [image 512x446]

tests\fixtures\eval_corpus\photos\IMG_0003.jpg  0.72  [image 512x342]
...
> sems -n 1 "astronauta"
tests\fixtures\eval_corpus\photos\IMG_0010.jpg  0.77  [image 512x512]
```

PDFs are searched page by page:

```console
> sems "can I have a dog in my flat"
tests\fixtures\eval_corpus\docs\lease_agreement.pdf  page 2  0.73
    2. Pets
    The Tenant may keep one cat or one small dog with the Landlord's written consent.
    An additional pet deposit of 300 EUR applies and is refundable at move-out.
```

Audio and video are searched by what is said and what is shown, down to the moment:

```console
> sems "when is the dentist appointment"
tests\fixtures\eval_corpus\audio\memo_0003.m4a  0:00-0:30  0.69  [audio]

> sems "how do I repot a plant"
tests\fixtures\eval_corpus\videos\clip_0001.mp4  0:00-0:12  0.79  [video]

> sems --kind video "a rocket launch"
tests\fixtures\eval_corpus\videos\clip_0001.mp4  0:09-0:18  0.74  [video]
```

## Status

Text, code, images, PDFs, audio, and video. Scanned PDFs (no text layer, so they need OCR) come
next. Indexes built before audio and video support need `sems index --rebuild` once.

## Usage

```text
sems <QUERY> [PATH]           search under PATH (default: current directory)
    -n, --limit <N>           number of results (default 10)
    -l, --files-with-matches  print matching file paths only, best first
        --json                full results as JSON (with a "kind" per result), for scripts and agents
        --kind <text|image|pdf|audio|video>   only return this kind of content
        --all                 also show weak results (see below)
        --hyperlinks <auto|always|never>   clickable paths (default auto)
sems index [PATH]             index or incrementally update PATH
        --device <auto|cpu|cuda|directml>   (default auto)
        --skip-images         leave images out (see below)
        --skip-audio          leave audio files out
        --skip-video          leave videos out
        --rebuild             discard the index and start over
sems status [PATH]            what is indexed under PATH
```

Indexing respects `.gitignore`, skips hidden, binary and large (>1 MB text, >64 MB image or PDF) files
(audio and video have no size limit),
and reads a `.semsignore` file (same syntax) for anything else to leave out. Re-indexing only
embeds files whose content changed.

Images (JPEG, PNG, WebP, GIF, BMP, TIFF; not HEIC yet) are embedded from their pixels, upright
according to their EXIF orientation. Images under 64 px on a side are skipped as icons. Text and
images share one ranking: the model scores a photo query highest against the right photo and a
text query highest against text, so no `--kind` is needed to keep them apart.

Audio (MP3, WAV, M4A, FLAC, Ogg, Opus, AAC, WMA) and video (MP4, MOV, MKV, WebM, AVI, M4V, WMV)
are decoded with [ffmpeg](https://ffmpeg.org), which must be on `PATH` (`winget install
Gyan.FFmpeg`) or named by `SEMS_FFMPEG`. Without it, recordings are counted in the summary and
indexed by the first run that finds ffmpeg. Each result is the stretch of time that matched:

- audio is embedded in 30-second windows every 25 seconds (30 s is the model's limit), so speech
  cut at one window's edge is whole in the next; silent stretches are skipped;
- video is embedded from one frame every 3 seconds, three frames per 9-second segment, and its
  soundtrack is embedded like an audio file, so a video is found by what it shows or what is said.

Searches show only results that stand out: when the similarity curve has a cliff after the top
matches, everything below it is dropped, so a question with one answer gets one result. A smooth
curve (many related chunks) is not cut; exact keyword matches always stay; `--all` disables the
cutoff. On the labelled corpus a search returns 2.5 results on average and never hides the answer.

Result paths are clickable (OSC 8 hyperlinks; Ctrl+click in Windows Terminal) and open the file
in its default application. `auto` enables them only in terminals known to support them (Windows
Terminal, VS Code, WezTerm, iTerm2, kitty, Konsole, GNOME Terminal and other VTE terminals); use
`--hyperlinks always` elsewhere, or set `SEMS_HYPERLINKS`.

Searches always run on the CPU (~1.2 s including model load). Indexing picks the fastest device
that works (`--device auto`): CUDA when the CUDA pack is installed and the NVIDIA driver supports
it, otherwise DirectML on the high-performance GPU, otherwise the CPU. The summary line names the
device used. On an RTX 3050 Ti, text indexes about 11x faster with CUDA and 3x with DirectML than on
the CPU; a 12-megapixel photo takes about 0.3 s with CUDA and 2 s on the CPU. A minute of
audio takes about 1.5 s with CUDA and 7 s on the CPU; a minute of video about 5 s with CUDA and
50 s on the CPU, plus its soundtrack. Pass `--skip-images`, `--skip-audio` or `--skip-video` to
leave them out on CPU-only machines.

Images are reduced to 140 vision tokens rather than the model's default 280: about 2.3x faster on
CUDA and 2.6x on the CPU with no loss on the search-quality corpus. The budget is part of the index
identity, so changing it requires `sems index --rebuild`.

## Setup (development)

sems keeps everything under `%LOCALAPPDATA%\sems` (`<local data dir>/sems` elsewhere):

```text
bin\                   sems.exe, on the user PATH (tools\install\install.ps1)
model\                 exported EmbeddingGemma 2 graphs, tokenizer and config
runtime\directml\      default ONNX Runtime (~40 MB); also serves CPU, so searches use it
runtime\cuda\          optional CUDA pack (~1.2 GB), preferred by `sems index` when the driver supports it
index.db               the index
```

1. **Export the model** (one time, needs Python 3.13):

   ```console
   python -m venv tools/export/.venv
   tools/export/.venv/Scripts/pip install -r tools/export/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   tools/export/.venv/Scripts/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-2', local_dir='models/embeddinggemma-2')"
   tools/export/.venv/Scripts/python tools/export/make_reference.py --model models/embeddinggemma-2 --out models/reference.json --images tests/fixtures/images/beach.png --audio tests/fixtures/eval_corpus/audio/memo_0001.wav tests/fixtures/eval_corpus/audio/memo_0003.m4a --videos tests/fixtures/eval_corpus/videos/clip_0001.mp4
   tools/export/.venv/Scripts/python tools/export/export_onnx.py --model models/embeddinggemma-2 --out models/onnx --reference models/reference.json --image tests/fixtures/images/beach.png --audio tests/fixtures/eval_corpus/audio/memo_0001.wav tests/fixtures/eval_corpus/audio/memo_0003.m4a --video tests/fixtures/eval_corpus/videos/clip_0001.mp4
   ```

   The export fails unless every graph reproduces the official sentence-transformers embeddings.
   Both scripts decode the audio and video fixtures with ffmpeg, exactly as sems does.
   Then copy `models/onnx/*` into `%LOCALAPPDATA%\sems\model\`.

2. **Install the runtimes**: `python tools/runtime/install_runtime.py directml`, plus
   `python tools/runtime/install_runtime.py cuda` for NVIDIA GPUs (CUDA 13 needs driver 580+). Each
   runtime keeps its dependencies in its own folder; nothing goes on `PATH`.

3. **Install** (Windows): `powershell -ExecutionPolicy Bypass -File tools\install\install.ps1`
   builds sems, copies it to `%LOCALAPPDATA%\sems\bin`, installs the DirectML runtime and copies the
   model if they are missing, and adds that directory to your user PATH. Open a new terminal and
   run `sems index`. Add `-Cuda` for the CUDA pack; run it again to update; `-Uninstall` removes
   sems and the PATH entry but keeps the model and index.

Every location can be overridden (flags win over environment variables):

| Setting | Flag | Environment variable |
|---|---|---|
| Model directory | `--model-dir` | `SEMS_MODEL_DIR` |
| ONNX Runtime library (any 1.24+ build) | `--onnxruntime` | `SEMS_ONNXRUNTIME` |
| Index database | `--index` | `SEMS_INDEX` |
| Indexing device (`auto`, `cpu`, `cuda`, `directml`) | `--device` | `SEMS_DEVICE` |
| ffmpeg executable (ffprobe must be beside it) | | `SEMS_FFMPEG` |

sems always loads ONNX Runtime by explicit path: Windows ships an outdated `onnxruntime.dll` in
System32 that loading by name could pick up.

## Tests

```console
cargo test --release                       # unit and indexing tests (fake encoder, no model needed; recordings need ffmpeg)
SEMS_ONNXRUNTIME=$LOCALAPPDATA/sems/runtime/directml/onnxruntime.dll cargo test --release -- --ignored --test-threads=1
```

The ignored tests need the exported model: `reference_embeddings` checks numerical parity with the
official pipeline (set `SEMS_DEVICE` to test a GPU provider), and `search_quality` measures recall
on a labelled corpus in `tests/fixtures`. `tools/bench/benchmark_providers.py` compares execution
providers.
