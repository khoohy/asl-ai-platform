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
    HAND_LOSS_GRACE_FRAMES,
    HOLDING_CONTEXT_STATUS,
    INPUT_DIM,
    MIN_VOTE_COUNT,
    PEAK_HISTORY_WINDOW,
    STABILIZATION_WINDOW,
    TRANSITIONING_STATUS,
    WAITING_FOR_HANDS_STATUS,
)
from app.ml.runtime_state import RuntimeSessionState
from app.ml.stabilization import ASLStabilizer, StabilizationResult


def make_top_k(
    first_label: str,
    first_confidence: float,
    second_label: str = "other",
    second_confidence: float = 0.10,
) -> list[dict[str, float | str]]:
    return [
        {"label": first_label, "confidence": first_confidence},
        {"label": second_label, "confidence": second_confidence},
        {"label": "fallback_3", "confidence": 0.05},
        {"label": "fallback_4", "confidence": 0.03},
        {"label": "fallback_5", "confidence": 0.02},
    ]


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
    )


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_result(label: str, result: StabilizationResult) -> None:
    print(
        f"{label}: status={result.stabilization_status}, "
        f"raw={result.raw_prediction}:{result.raw_confidence:.2f}, "
        f"stable={result.stable_prediction}:{result.stable_confidence:.2f}, "
        f"votes={result.vote_count}/{result.vote_window_size}"
    )


def run_vote_window_test(stabilizer: ASLStabilizer) -> None:
    print_header("vote window and minimum votes")
    session = RuntimeSessionState(session_id="vote-window")
    last_result = None
    for index in range(1, MIN_VOTE_COUNT + 1):
        last_result = stabilizer.stabilize(
            session=session,
            top_k_predictions=make_top_k("father", 0.82, "mother", 0.12),
            motion_delta=0.02,
        )
        print_result(f"frame {index}", last_result)

    if last_result is None or last_result.stable_prediction != "father":
        raise RuntimeError("Vote-window test failed to stabilize after six votes.")
    if session.prediction_history.maxlen != STABILIZATION_WINDOW:
        raise RuntimeError("Prediction history window size is not 10.")


def run_low_confidence_test(stabilizer: ASLStabilizer) -> None:
    print_header("low confidence rejection")
    session = RuntimeSessionState(session_id="low-confidence")
    result = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("father", 0.41, "mother", 0.35),
        motion_delta=0.02,
    )
    print_result("low-confidence", result)
    if result.stabilization_status != "low_confidence":
        raise RuntimeError("Low-confidence prediction was not rejected.")
    if result.stable_prediction is not None:
        raise RuntimeError("Low-confidence prediction should not become stable.")


def run_adaptive_fallback_tests(stabilizer: ASLStabilizer) -> None:
    print_header("adaptive fallback")
    positive_session = RuntimeSessionState(session_id="adaptive-positive")
    positive = stabilizer.stabilize(
        session=positive_session,
        top_k_predictions=make_top_k("father", 0.58, "mother", 0.22),
        motion_delta=0.02,
    )
    print_result("adaptive-positive", positive)
    if positive.stabilization_status != "collecting_votes":
        raise RuntimeError("Adaptive fallback positive case did not enter vote collection.")

    negative_session = RuntimeSessionState(session_id="adaptive-negative")
    negative = stabilizer.stabilize(
        session=negative_session,
        top_k_predictions=make_top_k("father", 0.58, "mother", 0.50),
        motion_delta=0.02,
    )
    print_result("adaptive-negative", negative)
    if negative.stabilization_status != "low_confidence":
        raise RuntimeError("Adaptive fallback accepted a weak-margin prediction.")


def run_confusion_hold_test(stabilizer: ASLStabilizer) -> None:
    print_header("confusion-pair suppression")
    session = RuntimeSessionState(session_id="confusion")
    result = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("approve", 0.72, "draw", 0.69),
        motion_delta=0.02,
    )
    print_result("confusion", result)
    if result.stabilization_status != "held_confusion":
        raise RuntimeError("Confusion-pair near-tie was not held.")


def run_peak_preservation_test(stabilizer: ASLStabilizer) -> None:
    print_header("peak-sign preservation")
    session = RuntimeSessionState(session_id="peak")
    first = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("jacket", 0.82, "how", 0.20),
        motion_delta=0.02,
    )
    second = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("jacket", 0.83, "how", 0.18),
        motion_delta=0.02,
    )
    print_result("peak-first", first)
    print_result("peak-second", second)
    if first.stabilization_status != "collecting_votes":
        raise RuntimeError("First peak-sign frame should still collect votes.")
    if second.stabilization_status != "peak_accepted":
        raise RuntimeError("Peak-sign candidate was not preserved.")
    if session.peak_history.maxlen != PEAK_HISTORY_WINDOW:
        raise RuntimeError("Peak history window size is not 5.")


def run_transition_hold_tests() -> None:
    print_header("stable output hold and cooldown")
    engine = RuntimeInferenceEngine()
    session = RuntimeSessionState(session_id="transition-hold")
    session.last_stable_prediction = "father"
    session.last_stable_confidence = 0.82
    session.stable_output_hold_remaining = 4
    session.transition_cooldown_remaining = 4
    session.stable_output_cooldown_remaining = 3
    session.valid_frames_collected = session.sequence_length

    weak_transition = StabilizationResult(
        raw_prediction="cool",
        raw_confidence=0.39,
        stable_prediction=None,
        stable_confidence=0.0,
        stabilization_status="low_confidence",
        vote_count=0,
        vote_window_size=10,
        note="Synthetic weak transition frame.",
        prediction="cool",
        confidence=0.39,
    )
    weak_response = engine._build_prediction_response(
        session=session,
        stabilization=weak_transition,
        top_k=make_top_k("cool", 0.39, "good", 0.31),
        keypoint_overlay={"left_hand": [], "right_hand": [], "pose": [], "face": []},
    )
    print(
        "weak-transition: "
        f"status={weak_response['status']}, prediction={weak_response['prediction']}, "
        f"stable={weak_response['stable_prediction']}, raw={weak_response['raw_prediction']}"
    )
    if weak_response["status"] != TRANSITIONING_STATUS:
        raise RuntimeError("Weak transition did not enter the transition-hold state.")
    if weak_response["prediction"] != "father":
        raise RuntimeError("Weak transition replaced the last stable sign too early.")

    candidate_transition = StabilizationResult(
        raw_prediction="mother",
        raw_confidence=0.84,
        stable_prediction="mother",
        stable_confidence=0.84,
        stabilization_status="stable",
        vote_count=6,
        vote_window_size=10,
        note="Synthetic competing stable sign.",
        prediction="mother",
        confidence=0.84,
    )
    candidate_response = engine._build_prediction_response(
        session=session,
        stabilization=candidate_transition,
        top_k=make_top_k("mother", 0.84, "father", 0.10),
        keypoint_overlay={"left_hand": [], "right_hand": [], "pose": [], "face": []},
    )
    print(
        "cooldown-transition: "
        f"status={candidate_response['status']}, prediction={candidate_response['prediction']}, "
        f"stable={candidate_response['stable_prediction']}, raw={candidate_response['raw_prediction']}"
    )
    if candidate_response["status"] != TRANSITIONING_STATUS:
        raise RuntimeError("Cooldown transition did not hold the previous stable sign.")
    if candidate_response["prediction"] != "father":
        raise RuntimeError("A new stable sign replaced the old one before cooldown expired.")


def run_no_hands_grace_test() -> None:
    print_header("no-hands grace and idle clearing")
    valid_frames = [make_ok_frame() for _ in range(35)]
    missing_frames = [make_no_hands_frame() for _ in range(HAND_LOSS_GRACE_FRAMES + 1)]
    engine = RuntimeInferenceEngine(
        model_loader=FakeModelLoader(
            model=FixedLogitModel(),
            label_map=FakeLabelMap(),
            device=torch.device("cpu"),
        ),
        frame_processor=FakeFrameProcessor(valid_frames + missing_frames),
    )

    session_id = "no-hands-grace"
    for _ in range(35):
        engine.process_frame("synthetic", session_id=session_id)

    first_missing = engine.process_frame("synthetic", session_id=session_id)
    print(
        "first-missing: "
        f"status={first_missing['status']}, grace={first_missing['grace_frames_remaining']}, "
        f"prediction={first_missing['prediction']}"
    )
    if first_missing["status"] != HOLDING_CONTEXT_STATUS:
        raise RuntimeError("The first no-hands frame should enter holding_context.")

    final_missing = first_missing
    for _ in range(HAND_LOSS_GRACE_FRAMES):
        final_missing = engine.process_frame("synthetic", session_id=session_id)

    print(
        "final-missing: "
        f"status={final_missing['status']}, grace={final_missing['grace_frames_remaining']}, "
        f"prediction={final_missing['prediction']}"
    )
    if final_missing["status"] != WAITING_FOR_HANDS_STATUS:
        raise RuntimeError("Missing hands did not progress to waiting_for_hands.")
    if final_missing["prediction"] is not None:
        raise RuntimeError("Idle output should clear the displayed prediction.")
    if final_missing["stable_prediction"] is not None:
        raise RuntimeError("Idle output should clear the stable prediction.")
    if final_missing["raw_prediction"] is not None:
        raise RuntimeError("Idle output should clear the raw prediction.")
    if final_missing["frames_collected"] != 0:
        raise RuntimeError("Idle output should clear the rolling buffer.")


def main() -> None:
    stabilizer = ASLStabilizer()
    run_vote_window_test(stabilizer)
    run_low_confidence_test(stabilizer)
    run_adaptive_fallback_tests(stabilizer)
    run_confusion_hold_test(stabilizer)
    run_peak_preservation_test(stabilizer)
    run_transition_hold_tests()
    run_no_hands_grace_test()
    print("\ntest_runtime_tuning_parity: all checks passed")


if __name__ == "__main__":
    main()
