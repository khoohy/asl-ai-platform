"""Per-session runtime buffer state for raw 30-frame ASL inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.ml.runtime_config import (
    INPUT_DIM,
    PEAK_HISTORY_WINDOW,
    SEQUENCE_LENGTH,
    STABILIZATION_WINDOW,
)


@dataclass
class RuntimeSessionState:
    """In-memory rolling state for one inference session."""

    session_id: str
    sequence_length: int = SEQUENCE_LENGTH
    input_dim: int = INPUT_DIM
    feature_buffer: deque[np.ndarray] = field(init=False)
    total_frames_received: int = 0
    valid_frames_collected: int = 0
    last_prediction: dict[str, Any] | None = None
    last_status: str = "idle"
    prediction_history: deque[str | None] = field(init=False)
    peak_history: deque[str | None] = field(init=False)
    last_stable_prediction: str | None = None
    last_stable_confidence: float = 0.0
    last_raw_prediction: str | None = None
    last_raw_confidence: float = 0.0
    stabilization_status: str = "raw_only"
    last_motion_score: float = 0.0

    def __post_init__(self) -> None:
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=STABILIZATION_WINDOW)
        self.peak_history = deque(maxlen=PEAK_HISTORY_WINDOW)

    def append(self, feature_vector: np.ndarray) -> None:
        vector = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
        if vector.shape != (self.input_dim,):
            raise ValueError(
                f"Expected feature vector shape ({self.input_dim},), got {vector.shape}."
            )
        self.feature_buffer.append(vector)
        self.valid_frames_collected = len(self.feature_buffer)

    def stack_sequence(self) -> np.ndarray:
        if len(self.feature_buffer) != self.sequence_length:
            raise ValueError(
                f"Sequence buffer is not ready: {len(self.feature_buffer)}/{self.sequence_length}."
            )
        return np.stack(self.feature_buffer, axis=0).astype(np.float32)

    def reset(self) -> None:
        self.feature_buffer.clear()
        self.total_frames_received = 0
        self.valid_frames_collected = 0
        self.last_prediction = None
        self.last_status = "reset"
        self.prediction_history.clear()
        self.peak_history.clear()
        self.last_stable_prediction = None
        self.last_stable_confidence = 0.0
        self.last_raw_prediction = None
        self.last_raw_confidence = 0.0
        self.stabilization_status = "raw_only"
        self.last_motion_score = 0.0
