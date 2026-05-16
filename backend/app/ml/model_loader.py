"""Production model loader for the ASL temporal classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from app.ml.label_map import LabelMap
from app.ml.runtime_config import (
    INPUT_DIM,
    LABEL_MAP_PATH,
    MODEL_PATH,
    MODEL_SOURCE,
    SEQUENCE_LENGTH,
)
from app.ml.sequence_model import BiLSTMSignClassifier


@dataclass
class ASLModelLoader:
    """Load the production checkpoint and expose validated model metadata."""

    model_path: Path = MODEL_PATH
    label_map_path: Path = LABEL_MAP_PATH
    expected_input_dim: int = INPUT_DIM
    expected_sequence_length: int = SEQUENCE_LENGTH
    model_source: str = MODEL_SOURCE
    model: BiLSTMSignClassifier | None = field(default=None, init=False)
    label_map: LabelMap | None = field(default=None, init=False)
    checkpoint: dict[str, Any] | None = field(default=None, init=False)
    device: torch.device = field(default_factory=lambda: _select_device(), init=False)
    input_dim: int = field(default=INPUT_DIM, init=False)
    sequence_length: int = field(default=SEQUENCE_LENGTH, init=False)
    number_of_labels: int = field(default=0, init=False)
    model_loaded: bool = field(default=False, init=False)

    def load(self) -> "ASLModelLoader":
        self._validate_artifact_paths()
        self.label_map = LabelMap.from_path(self.label_map_path)
        self.number_of_labels = len(self.label_map)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Unexpected checkpoint format: expected a dictionary.")
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint key mismatch: missing 'model_state_dict'.")

        self.checkpoint = checkpoint
        self.input_dim = int(checkpoint.get("input_dim", self.expected_input_dim))
        self.sequence_length = int(
            checkpoint.get("sequence_length", self.expected_sequence_length)
        )
        hidden_dim = int(checkpoint.get("hidden_dim", 512))
        dropout = float(checkpoint.get("dropout", 0.5))
        num_classes = int(checkpoint.get("num_classes", self.number_of_labels))
        num_heads = int(checkpoint.get("num_heads", 1))
        use_motion_delta = bool(checkpoint.get("use_motion_delta", False))

        if self.input_dim != self.expected_input_dim:
            raise ValueError(
                "Architecture mismatch: "
                f"checkpoint input_dim={self.input_dim}, expected {self.expected_input_dim}."
            )
        if self.sequence_length != self.expected_sequence_length:
            raise ValueError(
                "Architecture mismatch: "
                f"checkpoint sequence_length={self.sequence_length}, "
                f"expected {self.expected_sequence_length}."
            )
        if num_classes != self.number_of_labels:
            raise ValueError(
                "Checkpoint key mismatch: "
                f"checkpoint num_classes={num_classes}, label map length={self.number_of_labels}."
            )

        model = BiLSTMSignClassifier(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=2,
            dropout=dropout,
            num_heads=num_heads,
            use_motion_delta=use_motion_delta,
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        self.model = model
        self.model_loaded = True
        return self

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "model_source": self.model_source,
            "number_of_labels": self.number_of_labels,
            "model_loaded": self.model_loaded,
        }

    def _validate_artifact_paths(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Missing label map file: {self.label_map_path}")


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
