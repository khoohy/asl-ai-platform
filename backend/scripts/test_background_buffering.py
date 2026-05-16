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
from app.ml.runtime_config import HOLDING_CONTEXT_STATUS, INPUT_DIM, SEQUENCE_LENGTH, WAITING_FOR_HANDS_STATUS


class CountingLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, batch):  # noqa: D401
        self.call_count += 1
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
    sequence_length: int = SEQUENCE_LENGTH
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
    return FrameProcessingResult(
        status="ok",
        feature_vector=np.ones((INPUT_DIM,), dtype=np.float32),
        feature_dim=INPUT_DIM,
        expected_dim=INPUT_DIM,
        note="Synthetic valid frame.",
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
    print("starting recognition from a partially warm buffer")
    partial_model = CountingLogitModel()
    partial_engine = TestRuntimeInferenceEngine(
        model_loader=FakeModelLoader(
            model=partial_model,
            label_map=FakeLabelMap(),
            device=torch.device("cpu"),
        ),
        frame_processor=FakeFrameProcessor([make_ok_frame() for _ in range(20)]),
    )
    partial_session_id = "background-buffer-partial"
    for _ in range(12):
        partial_engine.process_frame(
            "synthetic",
            session_id=partial_session_id,
            recognition_active=False,
        )
    partial_response = partial_engine.process_frame(
        "synthetic",
        session_id=partial_session_id,
        recognition_active=True,
    )
    print(
        f"partial recognition frame: status={partial_response['status']}, "
        f"frames={partial_response['frames_collected']}/{partial_response['sequence_length']}"
    )
    if partial_response["frames_collected"] != 13:
        raise RuntimeError("Recognition should continue from the warmed background buffer, not restart from 0.")
    if partial_response["status"] != "warming_up":
        raise RuntimeError("A partially warm buffer should still report warming_up until 30 frames are ready.")
    if partial_model.call_count != 0:
        raise RuntimeError("Model forward should not run before the 30-frame sequence is full.")

    print()
    model = CountingLogitModel()
    engine = TestRuntimeInferenceEngine(
        model_loader=FakeModelLoader(
            model=model,
            label_map=FakeLabelMap(),
            device=torch.device("cpu"),
        ),
        frame_processor=FakeFrameProcessor(
            [make_ok_frame() for _ in range(SEQUENCE_LENGTH + 2)]
            + [make_no_hands_frame() for _ in range(12)]
        ),
    )

    session_id = "background-buffer-test"

    print("warming background buffer with recognition disabled")
    last_background_response = None
    for index in range(1, SEQUENCE_LENGTH + 1):
        response = engine.process_frame(
            "synthetic",
            session_id=session_id,
            recognition_active=False,
        )
        last_background_response = response
        print(
            f"background frame {index:02d}: status={response['status']}, "
            f"frames={response['frames_collected']}/{response['sequence_length']}, "
            f"buffer_ready={response['buffer_ready']}, prediction={response['prediction']}"
        )

    if last_background_response is None:
        raise RuntimeError("Background buffering produced no responses.")
    if model.call_count != 0:
        raise RuntimeError("Model forward should not run during background buffering.")
    if last_background_response["status"] != "buffer_ready":
        raise RuntimeError("Expected buffer_ready once 30 background frames were collected.")
    if last_background_response["frames_collected"] != SEQUENCE_LENGTH:
        raise RuntimeError("Background buffering did not fill the 30-frame sequence.")
    if last_background_response["prediction"] is not None:
        raise RuntimeError("Background buffering should not surface an accepted prediction.")

    print("\nstarting recognition on top of the hot buffer")
    recognition_response = engine.process_frame(
        "synthetic",
        session_id=session_id,
        recognition_active=True,
    )
    print(
        f"recognition frame: status={recognition_response['status']}, "
        f"frames={recognition_response['frames_collected']}/{recognition_response['sequence_length']}, "
        f"buffer_ready={recognition_response['buffer_ready']}, "
        f"prediction={recognition_response['prediction']}, raw={recognition_response['raw_prediction']}"
    )
    if model.call_count != 1:
        raise RuntimeError("Recognition should run the model immediately once the buffer is already warm.")
    if recognition_response["status"] == "warming_up":
        raise RuntimeError("Recognition incorrectly restarted a cold 30-frame warmup.")
    if recognition_response["sequence_length"] != SEQUENCE_LENGTH:
        raise RuntimeError("The 30-frame sequence contract changed unexpectedly.")

    print("\nstopping recognition while preserving the hot buffer")
    paused_response = engine.process_frame(
        "synthetic",
        session_id=session_id,
        recognition_active=False,
    )
    print(
        f"paused frame: status={paused_response['status']}, "
        f"frames={paused_response['frames_collected']}/{paused_response['sequence_length']}, "
        f"buffer_ready={paused_response['buffer_ready']}, prediction={paused_response['prediction']}"
    )
    if paused_response["status"] != "buffer_ready":
        raise RuntimeError("Paused recognition should keep the existing hot buffer ready.")
    if model.call_count != 1:
        raise RuntimeError("Pausing recognition should not keep calling the model.")

    print("\nchecking safe no-hands behavior while recognition is paused")
    holding_response = None
    final_idle_response = None
    missing_frame_index = 0
    while True:
        missing_frame_index += 1
        engine.advance_ms(250)
        response = engine.process_frame(
            "synthetic",
            session_id=session_id,
            recognition_active=False,
        )
        print(
            f"missing frame {missing_frame_index:02d}: status={response['status']}, "
            f"frames={response['frames_collected']}/{response['sequence_length']}, "
            f"prediction={response['prediction']}, grace_ms={response['grace_ms_remaining']}"
        )
        if missing_frame_index == 1:
            holding_response = response
        final_idle_response = response
        if response["status"] == WAITING_FOR_HANDS_STATUS:
            break

    if holding_response is None or holding_response["status"] != HOLDING_CONTEXT_STATUS:
        raise RuntimeError("Expected holding_context on the first missing-hands frame.")
    if holding_response["prediction"] is not None:
        raise RuntimeError("Background holding_context should not surface an accepted prediction.")
    if holding_response["frames_collected"] != SEQUENCE_LENGTH:
        raise RuntimeError("No-hands frames should not warm or mutate the buffered frame count.")
    if final_idle_response is None or final_idle_response["status"] != WAITING_FOR_HANDS_STATUS:
        raise RuntimeError("Expected waiting_for_hands after the hand-loss grace period.")
    if final_idle_response["frames_collected"] != 0:
        raise RuntimeError("Idle clearing should empty the rolling buffer.")
    if final_idle_response["stable_prediction"] is not None:
        raise RuntimeError("Idle clearing should remove the last stable prediction.")

    print("\nresetting the session to simulate camera stop")
    reset_response = engine.reset_session(session_id)
    session = engine.session_manager.get_session(session_id) if engine.session_manager else None
    print(
        f"reset response: status={reset_response['status']}, "
        f"session={reset_response['session_id']}"
    )
    if session is None:
        raise RuntimeError("Session manager lost the session during reset.")
    if session.valid_frames_collected != 0:
        raise RuntimeError("Stopping the camera should clear the buffered frames.")
    if session.camera_active:
        raise RuntimeError("Reset should mark the camera state inactive for the session.")
    if session.recognition_active:
        raise RuntimeError("Reset should mark recognition inactive for the session.")

    print("\ntest_background_buffering: all checks passed")


if __name__ == "__main__":
    main()
