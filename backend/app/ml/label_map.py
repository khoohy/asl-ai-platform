"""Helpers for loading and querying the production label map."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.runtime_config import LABEL_MAP_PATH


class LabelMap:
    """Simple index-to-label lookup wrapper."""

    def __init__(self, mapping: dict[int, str]) -> None:
        self._mapping = mapping

    @classmethod
    def from_path(cls, path: Path = LABEL_MAP_PATH) -> "LabelMap":
        if not path.exists():
            raise FileNotFoundError(f"Missing label map file: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw_mapping = json.load(handle)

        if not isinstance(raw_mapping, dict):
            raise ValueError("Unexpected label map format: expected a JSON object.")

        mapping = {int(index): str(label) for index, label in raw_mapping.items()}
        return cls(mapping)

    def get_label(self, index: int) -> str:
        return self._mapping[index]

    @property
    def mapping(self) -> dict[int, str]:
        return dict(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)
