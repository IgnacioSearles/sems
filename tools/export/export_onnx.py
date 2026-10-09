"""Export EmbeddingGemma 2 into three ONNX graphs consumed by the sems Rust runtime.

    token_embedder.onnx   input_ids [B,T] int64                      -> inputs_embeds [B,T,512]
    vision_encoder.onnx   pixel_values [N,P,768], position_ids [N,P,2] -> soft_tokens [S,512]
    audio_encoder.onnx    input_features [1,3000,128], input_features_mask [1,3000] int64 -> soft_tokens [S,512]
    text_encoder.onnx     inputs_embeds [B,T,512], attention_mask [B,T] -> embedding [B,768] (L2-normalized)

Video frames use the vision encoder; their soft tokens fill <|video|> placeholders instead of
<|image|>. Audio soft tokens fill <|audio|> placeholders.

The split mirrors EmbeddingGemma2Model.forward: Rust tokenizes, embeds tokens, splices vision soft
tokens into the multimodal placeholder positions, then runs the text encoder. Mean pooling and
normalization (sentence-transformers modules 1 and 2) are folded into text_encoder.

Every graph is verified against the reference embeddings produced by make_reference.py, first as
PyTorch wrappers (proves the decomposition is faithful) and then through ONNX Runtime (proves the
export is faithful). Finally the weights are stored in float16 (half_precision.py), halving the
download, and the graphs that ship are verified again.

Usage:
    python export_onnx.py --model ../../models/embeddinggemma-2 --out ../../models/onnx \
        --reference ../../models/reference.json --image ../../tests/fixtures/images/beach.png
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch import nn

# Sibling module shared with make_reference.py; added explicitly so `python -I` (no script directory
# on sys.path) works too.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from half_precision import convert_directory  # noqa: E402
from media_decoding import decode_audio, decode_video_frames  # noqa: E402

MINIMUM_COSINE_SIMILARITY = 0.9999
# Smaller vision token budgets than the default 280 that the runtime may use; verified explicitly.
ALTERNATIVE_VISION_TOKEN_BUDGETS = (140, 70)
ONNX_OPSET = 20
# The audio graph takes a fixed 3000 mel frames: the model's 30 s cap. With a dynamic length, the
# exporter's symbolic shape simplification (sympy) on the conformer's chunked attention masks ran for
# over 20 minutes without finishing. Shorter clips are zero-padded and masked; verify() compares the
# padded ONNX path against the official unpadded embeddings.
AUDIO_FRAMES = 3000
# Non-graph files the Rust runtime reads from the model directory.
RUNTIME_FILES = ("tokenizer.json", "config.json")


class TokenEmbedder(nn.Module):
    def __init__(self, language_model: nn.Module, multimodal_token_ids: list[int], pad_token_id: int):
        super().__init__()
        self.embed_tokens = language_model.embed_tokens
        self.multimodal_token_ids = multimodal_token_ids
        self.pad_token_id = pad_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Placeholder positions are overwritten by soft tokens later; embed them as PAD like upstream.
        # Explicit equality instead of torch.isin, which the ONNX exporter does not support.
        is_placeholder = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.multimodal_token_ids:
            is_placeholder = is_placeholder | (input_ids == token_id)
        safe_ids = torch.where(is_placeholder, torch.full_like(input_ids, self.pad_token_id), input_ids)
        return self.embed_tokens(safe_ids)


class VisionEncoder(nn.Module):
    def __init__(self, vision_tower: nn.Module, embed_vision: nn.Module):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=position_ids).last_hidden_state
        return self.embed_vision(inputs_embeds=hidden_states)


class AudioEncoder(nn.Module):
    """Audio tower plus projection into the text model's space, padding stripped (like vision)."""

    def __init__(self, audio_tower: nn.Module, embed_audio: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        # An int64 mask keeps the runtime side simple; the tower expects booleans.
        output = self.audio_tower(input_features, input_features_mask != 0, return_dict=True)
        soft_tokens = self.embed_audio(inputs_embeds=output.last_hidden_state)
        return soft_tokens[output.attention_mask]


class TextEncoder(nn.Module):
    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_states = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_states.dtype)
        pooled = (token_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return nn.functional.normalize(pooled, p=2, dim=-1)


class Pipeline:
    """Runs the decomposition with interchangeable backends (torch modules or ORT sessions)."""

    def __init__(self, embed_tokens, encode_vision, encode_audio, encode_text, token_ids: dict[str, int]):
        self.embed_tokens = embed_tokens
        self.encode_vision = encode_vision
        self.encode_audio = encode_audio
        self.encode_text = encode_text
        self.token_ids = token_ids

    def embed(self, features: dict[str, torch.Tensor]) -> np.ndarray:
        input_ids = features["input_ids"]
        inputs_embeds = self.embed_tokens(input_ids)
        if "pixel_values" in features:
            soft_tokens = self.encode_vision(features["pixel_values"], features["image_position_ids"])
            self._splice(inputs_embeds, input_ids, "image", soft_tokens)
        if "pixel_values_videos" in features:
            soft_tokens = self.encode_vision(features["pixel_values_videos"], features["video_position_ids"])
            self._splice(inputs_embeds, input_ids, "video", soft_tokens)
        if "input_features" in features:
            audio, mask = pad_audio_features(features["input_features"], features["input_features_mask"])
            soft_tokens = self.encode_audio(audio, mask)
            self._splice(inputs_embeds, input_ids, "audio", soft_tokens)
        return self.encode_text(inputs_embeds, features["attention_mask"])

    def _splice(self, inputs_embeds, input_ids, modality: str, soft_tokens) -> None:
        positions = input_ids == self.token_ids[modality]
        if int(positions.sum()) != soft_tokens.shape[0]:
            raise ValueError(f"{int(positions.sum())} {modality} placeholders but {soft_tokens.shape[0]} soft tokens")
        inputs_embeds[positions] = soft_tokens


def pad_audio_features(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pads features [1,F,128] and mask [1,F] to AUDIO_FRAMES; padded frames are masked out."""
    missing = AUDIO_FRAMES - features.shape[1]
    if missing < 0:
        raise ValueError(f"{features.shape[1]} mel frames exceed the {AUDIO_FRAMES}-frame (30 s) audio window")
    features = torch.nn.functional.pad(features, (0, 0, 0, missing))
    mask = torch.nn.functional.pad(mask.to(torch.long), (0, missing))
    return features, mask


def make_pooler_exportable(pooling_kernel_size: int) -> None:
    """Replaces Gemma4VisionPooler's pooling with an equivalent that keeps the patch count symbolic.

    Upstream derives the kernel size with `int(sqrt(patches // length))` and pools with
    `F.one_hot(..., length)`; both need concrete integers, so torch.export silently fixed the patch
    dimension to the traced 2520 and every other vision token budget failed at runtime. The kernel
    size is a model constant, and one_hot is an equality test against `arange(length)`. Callers
    verify equivalence against references computed with the unpatched code.
    """
    from transformers.models.gemma4 import modeling_gemma4

    def avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, pooling_kernel_size, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // pooling_kernel_size) * kernel_idxs[..., 1]
        slots = torch.arange(length, device=kernel_idxs.device)
        weights = (kernel_idxs.unsqueeze(-1) == slots).float() / pooling_kernel_size**2
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
        # Upstream's size checks branch on the patch count; output_length < patches always holds here.
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        hidden_states, padding_positions = self._avg_pool_by_positions(hidden_states, pixel_position_ids, output_length)
        return hidden_states.float() * self.root_hidden_size, padding_positions

    modeling_gemma4.Gemma4VisionPooler._avg_pool_by_positions = avg_pool_by_positions
    modeling_gemma4.Gemma4VisionPooler.forward = forward


def budget_reference_cases(model: SentenceTransformer, image_path: Path) -> list[dict]:
    """Image cases at non-default token budgets, embedded by the official (unpatched) model."""
    backbone = model[0].model
    config = backbone.config
    image = Image.open(image_path).convert("RGB")
    cases = []
    for budget in ALTERNATIVE_VISION_TOKEN_BUDGETS:
        processed = model[0].processor(text=["<|image|>"], images=[image], return_tensors="pt", max_soft_tokens=budget)
        soft_tokens = int((processed["input_ids"] == config.image_token_id).sum())
        ids = [config.text_config.bos_token_id, config.boi_token_id] + [config.image_token_id] * soft_tokens
        ids += [config.eoi_token_id, config.text_config.eos_token_id]
        features = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": processed["pixel_values"],
            "image_position_ids": processed["image_position_ids"],
        }
        with torch.no_grad():
            hidden = backbone(**features).last_hidden_state
        expected = torch.nn.functional.normalize(hidden.mean(dim=1), dim=-1)[0].numpy()
        cases.append({"features": features, "expected": expected})
    return cases


def media_reference_cases(model: SentenceTransformer, audio_paths: list[Path], video_path: Path | None) -> list[dict]:
    """Audio and video cases embedded by the official sentence-transformers pipeline."""
    cases = []
    for path in audio_paths:
        item = {"audio": {"array": decode_audio(path), "sampling_rate": 16000}}
        cases.append({"features": model.preprocess([item]), "expected": model.encode(item, normalize_embeddings=True)})
    if video_path is not None:
        item = {"video": decode_video_frames(video_path, seconds_per_frame=8, width=480, height=320)}
        cases.append({"features": model.preprocess([item]), "expected": model.encode(item, normalize_embeddings=True)})
    return cases


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True,
                        help="clips of different lengths, to prove the audio graph's length is dynamic")
    parser.add_argument("--video", type=Path, required=True)
    return parser.parse_args()


def load_model(model_path: Path) -> SentenceTransformer:
    """The model as sentence-transformers users get it, including its default (SDPA) attention.

    Not eager attention: in transformers 5.19 the eager path of the audio encoder's chunked attention
    is wrong. A voice memo matched its own transcript at 0.69 with eager (unrelated text: 0.63) and
    0.87 with SDPA (unrelated: 0.57). Text and vision agree under both. Because these references
    come from the same model object as the export, they must use the default implementation, or a
    shared bug passes verification unseen (as eager audio did once).
    """
    model = SentenceTransformer(str(model_path), device="cpu", model_kwargs={"torch_dtype": torch.float32})
    model.eval()
    return model


Modules = tuple[TokenEmbedder, VisionEncoder, AudioEncoder, TextEncoder]


def build_modules(model: SentenceTransformer) -> Modules:
    backbone = model[0].model
    config = backbone.config
    multimodal_token_ids = [config.image_token_id, config.video_token_id, config.audio_token_id]
    return (
        TokenEmbedder(backbone.language_model, multimodal_token_ids, config.text_config.pad_token_id).eval(),
        VisionEncoder(backbone.vision_tower, backbone.embed_vision).eval(),
        AudioEncoder(backbone.audio_tower, backbone.embed_audio).eval(),
        TextEncoder(backbone.language_model).eval(),
    )


def placeholder_token_ids(model: SentenceTransformer) -> dict[str, int]:
    config = model[0].model.config
    return {"image": config.image_token_id, "video": config.video_token_id, "audio": config.audio_token_id}

def preprocess_reference_inputs(model: SentenceTransformer, references: list[dict], image_path: Path) -> list[dict]:
    """Text and image references; audio and video come from media_reference_cases."""
    cases = []
    for reference in references:
        if reference["kind"] not in ("text", "image"):
            continue
        model_input = reference["input"] if reference["kind"] == "text" else {"image": str(image_path)}
        if reference["kind"] == "image" and reference["input"] != image_path.name:
            raise ValueError(f"reference image {reference['input']} does not match --image {image_path.name}")
        features = model.preprocess([model_input])
        cases.append({"features": features, "expected": np.array(reference["embedding"], dtype=np.float32)})
    return cases


def verify(pipeline: Pipeline, cases: list[dict], label: str) -> None:
    worst = 1.0
    for case in cases:
        actual = np.asarray(pipeline.embed(case["features"]), dtype=np.float32)[0]
        worst = min(worst, float(actual @ case["expected"]))
    status = "OK" if worst >= MINIMUM_COSINE_SIMILARITY else "FAIL"
    print(f"[{label}] worst cosine similarity vs reference: {worst:.6f} ({status})")
    if worst < MINIMUM_COSINE_SIMILARITY:
        raise SystemExit(f"{label} diverges from the reference embeddings")


def export_graph(module: nn.Module, example_inputs: dict[str, torch.Tensor], dynamic_shapes: dict | None,
                 output_names: list[str], path: Path, optimize: bool = True) -> None:
    """Exports with the dynamo exporter and weights as external data.

    The legacy TorchScript exporter produced ~7k nodes of shape arithmetic for the text encoder,
    which made ONNX Runtime session creation take ~4 s; dynamo's graph loads in ~1 s. External
    data lets ONNX Runtime memory-map weights instead of copying them out of the protobuf.
    """
    torch.onnx.export(
        module,
        tuple(example_inputs.values()),
        str(path),
        input_names=list(example_inputs),
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=ONNX_OPSET,
        external_data=True,
        dynamo=True,
        optimize=optimize,
    )
    disable_reshape_allowzero(path)
    print(f"exported {path.name}")


def disable_reshape_allowzero(path: Path) -> None:
    """Rewrites Reshape(allowzero=1) to allowzero=0, which the DirectML provider requires.

    The two only differ when a target-shape entry is literally 0. Constant shapes are checked
    here; dynamic shapes come from Shape() of non-empty inputs, so they never contain 0.
    The onnx runtime parity checks that follow cover the rewritten graphs.
    """
    model = onnx.load(str(path), load_external_data=False)
    constant_shapes = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
        if initializer.data_type == onnx.TensorProto.INT64
        and initializer.data_location != onnx.TensorProto.EXTERNAL
    }
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        shape = constant_shapes.get(node.input[1])
        if shape is not None and (shape == 0).any():
            raise ValueError(f"{path.name}: {node.name} has a literal 0 in its target shape")
        for attribute in node.attribute:
            if attribute.name == "allowzero":
                attribute.i = 0
    onnx.save(model, str(path))  # weights stay in the existing external data file


def repeat_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Dynamo specializes size-1 dimensions to constants, so examples need a batch of at least 2."""
    return torch.cat([tensor, tensor], dim=0)


def export_all(modules: Modules, image_features: dict, audio_features: dict, out_dir: Path) -> None:
    token_embedder, vision_encoder, audio_encoder, text_encoder = modules
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch", min=1, max=1024)
    sequence = torch.export.Dim("sequence", min=2, max=8192)
    images = torch.export.Dim("images", min=1, max=64)
    patches = torch.export.Dim("patches", min=9, max=1120 * 9)

    input_ids = repeat_batch(image_features["input_ids"])
    with torch.no_grad():
        export_graph(token_embedder, {"input_ids": input_ids}, {"input_ids": {0: batch, 1: sequence}},
                     ["inputs_embeds"], out_dir / "token_embedder.onnx")
        export_graph(vision_encoder,
                     {"pixel_values": repeat_batch(image_features["pixel_values"]),
                      "position_ids": repeat_batch(image_features["image_position_ids"])},
                     {"pixel_values": {0: images, 1: patches}, "position_ids": {0: images, 1: patches}},
                     ["soft_tokens"], out_dir / "vision_encoder.onnx")
        padded_audio, padded_mask = pad_audio_features(audio_features["input_features"],
                                                       audio_features["input_features_mask"])
        # The onnxscript optimizer produced an invalid audio graph (a node read a value no node
        # produced: 'mul_10_min_1'); the unoptimized graph is valid, and ONNX Runtime optimizes at load.
        export_graph(audio_encoder, {"input_features": padded_audio, "input_features_mask": padded_mask},
                     None, ["soft_tokens"], out_dir / "audio_encoder.onnx", optimize=False)
        export_graph(text_encoder,
                     {"inputs_embeds": token_embedder(input_ids),
                      "attention_mask": repeat_batch(image_features["attention_mask"])},
                     {"inputs_embeds": {0: batch, 1: sequence}, "attention_mask": {0: batch, 1: sequence}},
                     ["embedding"], out_dir / "text_encoder.onnx")

def torch_pipeline(modules: Modules, token_ids: dict[str, int]) -> Pipeline:
    token_embedder, vision_encoder, audio_encoder, text_encoder = modules

    def no_grad(function):
        return lambda *arguments: torch.no_grad()(function)(*arguments)

    return Pipeline(no_grad(token_embedder), no_grad(vision_encoder), no_grad(audio_encoder),
                    lambda embeds, mask: no_grad(text_encoder)(embeds, mask).numpy(), token_ids)

def onnx_pipeline(out_dir: Path, token_ids: dict[str, int]) -> Pipeline:
    def session(name: str) -> onnxruntime.InferenceSession:
        return onnxruntime.InferenceSession(str(out_dir / name), providers=["CPUExecutionProvider"])

    token_embedder, vision_encoder, audio_encoder, text_encoder = (
        session("token_embedder.onnx"), session("vision_encoder.onnx"), session("audio_encoder.onnx"),
        session("text_encoder.onnx"))
    return Pipeline(
        lambda ids: torch.from_numpy(token_embedder.run(None, {"input_ids": ids.numpy()})[0]),
        lambda pixels, positions: torch.from_numpy(vision_encoder.run(
            None, {"pixel_values": pixels.numpy(), "position_ids": positions.numpy()})[0]),
        lambda features, mask: torch.from_numpy(audio_encoder.run(
            None, {"input_features": features.numpy(), "input_features_mask": mask.numpy()})[0]),
        lambda embeds, mask: text_encoder.run(None, {"inputs_embeds": embeds.numpy(), "attention_mask": mask.numpy()})[0],
        token_ids,
    )

def verify_generalization(model: SentenceTransformer, pipeline: Pipeline) -> None:
    """Check shapes the export never traced: another image aspect ratio and a padded multi-text batch."""
    portrait = Image.new("RGB", (300, 800), (40, 120, 60))
    portrait_case = {
        "features": model.preprocess([{"image": portrait}]),
        "expected": model.encode({"image": portrait}, normalize_embeddings=True),
    }
    verify(pipeline, [portrait_case], "onnx runtime, untraced image size")

    texts = ["title: none | text: short", "title: none | text: " + "a considerably longer document " * 40]
    batch_features = model.preprocess(texts)
    expected = model.encode(texts, normalize_embeddings=True)
    actual = np.asarray(pipeline.embed(batch_features), dtype=np.float32)
    worst = float(min((actual * expected).sum(axis=1)))
    print(f"[onnx runtime, padded batch] worst cosine similarity: {worst:.6f}")
    if worst < MINIMUM_COSINE_SIMILARITY:
        raise SystemExit("padded batch diverges from the reference embeddings")


def main() -> None:
    # The dynamo exporter prints emoji; Windows consoles default to cp1252 and would crash on them.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    arguments = parse_arguments()
    model = load_model(arguments.model)
    token_ids = placeholder_token_ids(model)
    references = json.loads(arguments.reference.read_text(encoding="utf-8"))
    cases = preprocess_reference_inputs(model, references, arguments.image)
    # References first, with the official pooler; the patch below must reproduce them exactly.
    budget_cases = budget_reference_cases(model, arguments.image)
    media_cases = media_reference_cases(model, arguments.audio, arguments.video)
    make_pooler_exportable(model[0].model.config.vision_config.pooling_kernel_size)
    modules = build_modules(model)

    decomposition = torch_pipeline(modules, token_ids)
    verify(decomposition, cases, "torch decomposition")
    verify(decomposition, budget_cases, "torch decomposition, other token budgets")
    verify(decomposition, media_cases, "torch decomposition, audio and video")
    image_case = next(case for case in cases if "pixel_values" in case["features"])
    audio_case = next(case for case in media_cases if "input_features" in case["features"])
    # Full-precision graphs go to a staging directory: verified there, then converted into --out.
    full_precision = arguments.out.with_name(arguments.out.name + ".float32")
    shutil.rmtree(full_precision, ignore_errors=True)
    export_all(modules, image_case["features"], audio_case["features"], full_precision)
    for file_name in RUNTIME_FILES:
        shutil.copy2(arguments.model / file_name, full_precision / file_name)
    verify_onnx(model, onnx_pipeline(full_precision, token_ids), cases, budget_cases, media_cases, "float32")

    shutil.rmtree(arguments.out, ignore_errors=True)
    convert_directory(full_precision, arguments.out, RUNTIME_FILES)
    verify_onnx(model, onnx_pipeline(arguments.out, token_ids), cases, budget_cases, media_cases, "float16 weights")
    shutil.rmtree(full_precision)


def verify_onnx(model: SentenceTransformer, exported: Pipeline, cases: list[dict], budget_cases: list[dict],
                media_cases: list[dict], variant: str) -> None:
    verify(exported, cases, f"onnx runtime, {variant}")
    verify(exported, budget_cases, f"onnx runtime, {variant}, other token budgets")
    verify(exported, media_cases, f"onnx runtime, {variant}, audio and video")
    verify_generalization(model, exported)


if __name__ == "__main__":
    main()
