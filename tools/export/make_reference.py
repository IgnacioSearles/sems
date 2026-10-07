"""Generate golden reference embeddings with the official sentence-transformers pipeline.

The Rust/ONNX implementation is validated against these vectors: any divergence in
tokenization, pooling, or numerics shows up as a cosine similarity below threshold.

Usage:
    python make_reference.py --model ../../models/embeddinggemma-2 --out ../../models/reference.json
"""

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "

TEXT_CASES: list[dict[str, str]] = [
    {"prompt": QUERY_PROMPT, "text": "What causes the northern lights?"},
    {"prompt": DOCUMENT_PROMPT, "text": "The northern lights are caused by charged particles from the sun."},
    {"prompt": QUERY_PROMPT, "text": "receipt from the plumber"},
    {"prompt": DOCUMENT_PROMPT, "text": "fn retry_with_backoff(attempts: u32) -> Result<(), Error> { todo!() }"},
    {"prompt": DOCUMENT_PROMPT, "text": "Café — naïve façade, 日本語のテキスト, emoji 🚀"},
    {"prompt": QUERY_PROMPT, "text": ""},
    # Longer than the 512-token sliding window, so local and global attention masks diverge.
    {"prompt": DOCUMENT_PROMPT, "text": " ".join(
        f"Paragraph {index}: the quarterly report covers revenue, hiring, and infrastructure costs."
        for index in range(100)
    )},
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="*", default=[])
    return parser.parse_args()


def encode_texts(model: SentenceTransformer) -> list[dict[str, object]]:
    inputs = [case["prompt"] + case["text"] for case in TEXT_CASES]
    embeddings = model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True)
    return [
        {
            "kind": "text",
            "input": text,
            "input_ids": model.preprocess([text])["input_ids"][0].tolist(),
            "embedding": embedding.tolist(),
        }
        for text, embedding in zip(inputs, embeddings)
    ]


def write_tensor(tensor: torch.Tensor, path: Path) -> dict[str, object]:
    """Raw little-endian dump so Rust tests can load it without an npy dependency."""
    array = tensor.numpy()
    path.write_bytes(array.astype(array.dtype.newbyteorder("<")).tobytes())
    return {"file": path.name, "dtype": str(array.dtype), "shape": list(array.shape)}


def encode_images(model: SentenceTransformer, image_paths: list[Path], out_dir: Path) -> list[dict[str, object]]:
    references = []
    for image_path in image_paths:
        features = model.preprocess([{"image": str(image_path)}])
        embedding = model.encode({"image": str(image_path)}, convert_to_numpy=True, normalize_embeddings=True)
        references.append({
            "kind": "image",
            "input": image_path.name,
            "input_ids": features["input_ids"][0].tolist(),
            "pixel_values": write_tensor(features["pixel_values"], out_dir / f"{image_path.stem}.pixel_values.bin"),
            "position_ids": write_tensor(features["image_position_ids"], out_dir / f"{image_path.stem}.position_ids.bin"),
            "embedding": embedding.tolist(),
        })
    return references


def main() -> None:
    arguments = parse_arguments()
    model = SentenceTransformer(
        str(arguments.model),
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    references = encode_texts(model) + encode_images(model, arguments.images, arguments.out.parent)
    arguments.out.write_text(json.dumps(references, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(references)} reference embeddings to {arguments.out}")


if __name__ == "__main__":
    main()
