---
license: apache-2.0
base_model: google/embeddinggemma-2
library_name: onnx
tags:
  - onnx
  - embeddings
  - multimodal
  - sems
---

# EmbeddingGemma 2 for sems (ONNX)

[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) exported to ONNX for
[sems](https://github.com/IgnacioSearles/sems), a local semantic search CLI. sems downloads these
files on first use; you do not need to fetch them yourself.

| File | Contents |
|---|---|
| `token_embedder.onnx` | token ids -> input embeddings |
| `text_encoder.onnx` | input embeddings -> pooled, normalized embedding |
| `vision_encoder.onnx` | image and video-frame patches -> soft tokens |
| `audio_encoder.onnx` | 30 s log-mel features -> soft tokens |
| `config.json`, `tokenizer.json` | from the original model |

Weights are stored in float16 and cast to float32 when the graph loads, so inference runs in
float32. Every graph reproduces the official sentence-transformers embeddings with cosine
similarity of at least 0.9999 (measured 1.000000 at six decimals). The export scripts are in
[tools/export](https://github.com/IgnacioSearles/sems/tree/main/tools/export).

These files are a derivative of google/embeddinggemma-2 by Google DeepMind and are distributed
under the same Apache 2.0 license.
