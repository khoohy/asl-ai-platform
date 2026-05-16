from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.frame_processor import FrameProcessingResult
from app.ml.inference_engine import RuntimeInferenceEngine
from app.ml.runtime_config import (
    HAND_LOSS_GRACE_MS,
    HOLDING_CONTEXT_STATUS,
    INPUT_DIM,
    WAITING_FOR_HANDS_STATUS,
)


class FixedLogitModel(torch.nn.Module):
    def forward(self, batch):  # noqa: D401
        batch_size = batch.shape[0]
        logits = torch.tensor([[6.0, 1.2, 0.8, 0.4, 0.2]], dtype=torch.float32)
        return logits.repeat(batch_size, 1)


class FakeLabelMap:
    def __init__(self) -> None:
        self.labels = {
            0: "father",
            1: "mother",
            2: "hello",
            3: "good",
            4: "book",
        }

    def get_label(self, index: int) -> str:
        return self.labels[index]


@dataclass
class FakeModelLoader:
    model: torch.nn.Module
    label_map: FakeLabelMap
    device: torch.device
    sequence_length: int = 30
    input_dim: int = INPUT_DIM
    model_loaded: bool = True


class FakeFrameProcessor:
    def __init__(self, results: list[FrameProcessingResult]) -> None:
        self._results = list(results)

    def process_base64_image(self, image_base64: str) -> FrameProcessingResult:
        if not self._results:
            raise RuntimeError("FakeFrameProcessor ran out of queued results.")
        return self._results.pop(0)


class TestRuntimeInferenceEngine(RuntimeInferenceEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current_ms = 0.0

    def advance_ms(self, delta_ms: float) -> None:
        self.current_ms += delta_ms

    def _now_ms(self) -> float:
        return self.current_ms


def make_ok_frame() -> FrameProcessingResult:
    feature_vector = np.ones((INPUT_DIM,), dtype=np.float32)
    return FrameProcessingResult(
        status="ok",
        feature_vector=feature_vector,
        feature_dim=INPUT_DIM,
        expected_dim=INPUT_DIM,
        note="Synthetic valid hand frame.",
        hands_detected=True,
        any_landmarks_detected=True,
        keypoint_overlay={
            "left_hand": [[0.5, 0.5] for _ in range(21)],
            "right_hand": [[0.5, 0.5] for _ in range(21)],
            "pose": [],
            "face": [],
        },
        timing={},
    )


def make_no_hands_frame() -> FrameProcessingResult:
    return FrameProcessingResult(
        status="no_hands",
        feature_vector=None,
        feature_dim=0,
        expected_dim=INPUT_DIM,
        note="Synthetic no-hands frame.",
        hands_detected=False,
        any_landmarks_detected=True,
        keypoint_overlay={
            "left_hand": [],
            "right_hand": [],
            "pose": [[0.5, 0.3] for _ in range(7)],
            "face": [[0.5, 0.4] for _ in range(11)],
        },
        timing={},
    )


def main() -> None:
    valid_frames = [make_ok_frame() for _ in range(35)]
    missing_frames = [make_no_hands_frame() for _ in range(12)]
    engine = TestRuntimeInferenceEngine(
        model_loader=FakeModelLoader(
            model=FixedLogitModel(),
            label_map=FakeLabelMap(),
            device=torch.device("cpu"),
        ),
        frame_processor=FakeFrameProcessor(valid_frames + missing_frames),
    )

    session_id = "idle-state-test"
    last_valid_response = None
    for index in range(1, 36):
        response = engine.process_frame("synthetic", session_id=session_id)
        last_valid_response = response
        print(
            f"valid frame {index:02d}: status={response['status']}, "
            f"frames={response['frames_collected']}/{response['sequence_length']}, "
            f"prediction={response['prediction']}"
        )

    if last_valid_response is None:
        raise RuntimeError("No valid responses were produced during warmup.")

    if last_valid_response["status"] not in {
        "collecting_votes",
        "collecting_evidence",
        "transitioning",
        "stabilized",
        "raw_predicted",
        "low_confidence",
    }:
        raise RuntimeError("Expected a real inference status after valid frames filled the buffer.")

    holding_response = None
    final_idle_response = None
    missing_frame_index = 0
    while True:
        missing_frame_index += 1
        engine.advance_ms(250)
        response = engine.process_frame("synthetic", session_id=session_id)
        print(
            f"missing frame {missing_frame_index:02d}: status={response['status']}, "
            f"missing_hands={response['missing_hands_count']}, "
            f"grace_remaining_ms={response['grace_ms_remaining']}, "
            f"prediction={response['prediction']}"
        )
        if missing_frame_index == 1:
            holding_response = response
        final_idle_response = response
        if response["status"] == WAITING_FOR_HANDS_STATUS:
            break

    if holding_response is None or holding_response["status"] != HOLDING_CONTEXT_STATUS:
        raise RuntimeError("Expected holding_context on the first no-hands frame.")

    if final_idle_response is None or final_idle_response["status"] != WAITING_FOR_HANDS_STATUS:
        raise RuntimeError("Expected waiting_for_hands after the grace period expired.")

    if final_idle_response["prediction"] is not None:
        raise RuntimeError("Prediction should be cleared during waiting_for_hands.")
    if final_idle_response["stable_prediction"] is not None:
        raise RuntimeError("Stable prediction should be cleared during waiting_for_hands.")
    if final_idle_response["raw_prediction"] is not None:
        raise RuntimeError("Raw prediction should be cleared during waiting_for_hands.")
    if final_idle_response["frames_collected"] != 0:
        raise RuntimeError("Rolling buffer should be cleared during waiting_for_hands.")

    session = engine.session_manager.get_session(session_id) if engine.session_manager else None
    if session is None:
        raise RuntimeError("Session manager did not retain the test session.")
    if session.prediction_history:
        raise RuntimeError("Prediction history should be cleared during idle.")
    if session.peak_history:
        raise RuntimeError("Peak history should be cleared during idle.")

    print("test_idle_state: all checks passed")


if __name__ == "__main__":
    main()
