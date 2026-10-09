"""Stores exported graphs' weights in half precision, halving their size on disk.

Only storage changes: each float32 weight is saved as float16 and followed by a Cast back to
float32, which ONNX Runtime constant-folds when it loads the graph. Inference still computes in
float32, so speed and CPU support are unchanged and the only difference is the weights' rounding
(parity with the reference stays above 0.9999). Computing in float16 as well would be slow on the
CPU, which has few float16 kernels, and risks overflow in the normalization layers.

export_onnx.py runs this as its last step; it can also convert an existing export:
    python half_precision.py SOURCE_DIR DESTINATION_DIR
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Small tensors (norm scales, biases, constants) stay float32: they barely affect size, and
# keeping them exact avoids rounding where it is most likely to matter.
MIN_ELEMENTS = 4096
FLOAT16_MAX = float(np.finfo(np.float16).max)
GRAPHS = ("token_embedder.onnx", "text_encoder.onnx", "vision_encoder.onnx", "audio_encoder.onnx")


def element_count(tensor: TensorProto) -> int:
    return int(np.prod(tensor.dims)) if tensor.dims else 1


def store_weights_as_float16(source: Path, destination: Path) -> tuple[int, int]:
    """Converts one graph; returns (weights converted, weights kept as float32)."""
    model = onnx.load(str(source))  # with external data
    converted, kept = 0, 0
    casts = []
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type != TensorProto.FLOAT or element_count(initializer) < MIN_ELEMENTS:
            kept += 1
            continue
        # Read the name first: CopyFrom below overwrites this very message.
        name = initializer.name
        weights = numpy_helper.to_array(initializer)
        if np.abs(weights).max() > FLOAT16_MAX:
            raise ValueError(f"{source.name}: {name} exceeds the float16 range")
        stored_name = f"{name}__float16"
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(weights.astype(np.float16), stored_name))
        casts.append(helper.make_node("Cast", [stored_name], [name], to=TensorProto.FLOAT, name=f"{name}__to_float32"))
        converted += 1
    # Prepended so every Cast precedes its consumers, keeping the graph topologically sorted.
    nodes = casts + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    onnx.checker.check_model(model, full_check=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination), save_as_external_data=True, all_tensors_to_one_file=True,
              location=destination.name + ".data", size_threshold=1024)
    return converted, kept


def convert_directory(source: Path, destination: Path, companion_files: tuple[str, ...]) -> None:
    """Converts every graph in `source` into `destination` and copies `companion_files` along."""
    for graph in GRAPHS:
        converted, kept = store_weights_as_float16(source / graph, destination / graph)
        size = sum(path.stat().st_size for path in destination.glob(graph + "*"))
        print(f"{graph}: {converted} weights to float16, {kept} kept as float32, {size / 1e6:,.0f} MB")
    for name in companion_files:
        shutil.copy2(source / name, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert_directory(arguments.source, arguments.destination, ("config.json", "tokenizer.json"))


if __name__ == "__main__":
    main()
