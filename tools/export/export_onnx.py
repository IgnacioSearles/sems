"""Export EmbeddingGemma 2 into three ONNX graphs consumed by the sems Rust runtime.

    token_embedder.onnx   input_ids [B,T] int64                      -> inputs_embeds [B,T,512]
    vision_encoder.onnx   pixel_values [N,P,768], position_ids [N,P,2] -> soft_tokens [S,512]
    text_encoder.onnx     inputs_embeds [B,T,512], attention_mask [B,T] -> embedding [B,768] (L2-normalized)

The split mirrors EmbeddingGemma2Model.forward: Rust tokenizes, embeds tokens, splices vision soft
tokens into the multimodal placeholder positions, then runs the text encoder. Mean pooling and
normalization (sentence-transformers modules 1 and 2) are folded into text_encoder.

Every graph is verified against the reference embeddings produced by make_reference.py, first as
PyTorch wrappers (proves the decomposition is faithful) and then through ONNX Runtime (proves the
export is faithful).

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

MINIMUM_COSINE_SIMILARITY = 0.9999
ONNX_OPSET = 20
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
    """Runs the three-stage decomposition with interchangeable backends (torch modules or ORT sessions)."""

    def __init__(self, embed_tokens, encode_vision, encode_text, image_token_id: int):
        self.embed_tokens = embed_tokens
        self.encode_vision = encode_vision
        self.encode_text = encode_text
        self.image_token_id = image_token_id

    def embed(self, features: dict[str, torch.Tensor]) -> np.ndarray:
        input_ids = features["input_ids"]
        inputs_embeds = self.embed_tokens(input_ids)
        if "pixel_values" in features:
            soft_tokens = self.encode_vision(features["pixel_values"], features["image_position_ids"])
            image_positions = input_ids == self.image_token_id
            if int(image_positions.sum()) != soft_tokens.shape[0]:
                raise ValueError(f"{int(image_positions.sum())} placeholders but {soft_tokens.shape[0]} soft tokens")
            inputs_embeds[image_positions] = soft_tokens
        return self.encode_text(inputs_embeds, features["attention_mask"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def load_model(model_path: Path) -> SentenceTransformer:
    model = SentenceTransformer(
        str(model_path),
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"},
    )
    model.eval()
    return model


def build_modules(model: SentenceTransformer) -> tuple[TokenEmbedder, VisionEncoder, TextEncoder]:
    backbone = model[0].model
    config = backbone.config
    multimodal_token_ids = [config.image_token_id, config.video_token_id, config.audio_token_id]
    return (
        TokenEmbedder(backbone.language_model, multimodal_token_ids, config.text_config.pad_token_id).eval(),
        VisionEncoder(backbone.vision_tower, backbone.embed_vision).eval(),
        TextEncoder(backbone.language_model).eval(),
    )


def preprocess_reference_inputs(model: SentenceTransformer, references: list[dict], image_path: Path) -> list[dict]:
    cases = []
    for reference in references:
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


def export_graph(module: nn.Module, example_inputs: dict[str, torch.Tensor], dynamic_shapes: dict,
                 output_names: list[str], path: Path) -> None:
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
        optimize=True,
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


def export_all(modules: tuple[TokenEmbedder, VisionEncoder, TextEncoder], image_features: dict, out_dir: Path) -> None:
    token_embedder, vision_encoder, text_encoder = modules
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
        export_graph(text_encoder,
                     {"inputs_embeds": token_embedder(input_ids),
                      "attention_mask": repeat_batch(image_features["attention_mask"])},
                     {"inputs_embeds": {0: batch, 1: sequence}, "attention_mask": {0: batch, 1: sequence}},
                     ["embedding"], out_dir / "text_encoder.onnx")


def torch_pipeline(modules: tuple[TokenEmbedder, VisionEncoder, TextEncoder], image_token_id: int) -> Pipeline:
    token_embedder, vision_encoder, text_encoder = modules

    def no_grad(function):
        return lambda *arguments: torch.no_grad()(function)(*arguments)

    return Pipeline(no_grad(token_embedder), no_grad(vision_encoder),
                    lambda embeds, mask: no_grad(text_encoder)(embeds, mask).numpy(), image_token_id)


def onnx_pipeline(out_dir: Path, image_token_id: int) -> Pipeline:
    def session(name: str) -> onnxruntime.InferenceSession:
        return onnxruntime.InferenceSession(str(out_dir / name), providers=["CPUExecutionProvider"])

    token_embedder, vision_encoder, text_encoder = (
        session("token_embedder.onnx"), session("vision_encoder.onnx"), session("text_encoder.onnx"))
    return Pipeline(
        lambda ids: torch.from_numpy(token_embedder.run(None, {"input_ids": ids.numpy()})[0]),
        lambda pixels, positions: torch.from_numpy(vision_encoder.run(
            None, {"pixel_values": pixels.numpy(), "position_ids": positions.numpy()})[0]),
        lambda embeds, mask: text_encoder.run(None, {"inputs_embeds": embeds.numpy(), "attention_mask": mask.numpy()})[0],
        image_token_id,
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
    image_token_id = model[0].model.config.image_token_id
    references = json.loads(arguments.reference.read_text(encoding="utf-8"))
    cases = preprocess_reference_inputs(model, references, arguments.image)
    modules = build_modules(model)

    verify(torch_pipeline(modules, image_token_id), cases, "torch decomposition")
    image_case = next(case for case in cases if "pixel_values" in case["features"])
    export_all(modules, image_case["features"], arguments.out)
    for file_name in RUNTIME_FILES:
        shutil.copy2(arguments.model / file_name, arguments.out / file_name)
    exported = onnx_pipeline(arguments.out, image_token_id)
    verify(exported, cases, "onnx runtime")
    verify_generalization(model, exported)


if __name__ == "__main__":
    main()
