from __future__ import annotations

import sys
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.model_loader import ASLModelLoader


def main() -> None:
    loader = ASLModelLoader().load()

    print("Model metadata:")
    for key, value in loader.metadata.items():
        print(f"  {key}: {value}")

    if loader.label_map is None:
        raise RuntimeError("Label map failed to load.")

    print(f"Label map loaded: {len(loader.label_map)} labels")
    print(f"First label: {loader.label_map.get_label(0)}")

    if loader.model is None:
        raise RuntimeError("Model failed to load.")

    dummy_input = torch.zeros(
        (1, loader.sequence_length, loader.input_dim),
        dtype=torch.float32,
        device=loader.device,
    )
    print(f"Dummy input shape: {tuple(dummy_input.shape)}")

    with torch.no_grad():
        output = loader.model(dummy_input)

    print(f"Forward pass output shape: {tuple(output.shape)}")
    if output.shape[-1] != len(loader.label_map):
        raise RuntimeError(
            "Output class count mismatch: "
            f"output has {output.shape[-1]}, label map has {len(loader.label_map)}."
        )
    print("Output class count matches label map length.")


if __name__ == "__main__":
    main()
