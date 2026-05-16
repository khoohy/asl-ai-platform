from __future__ import annotations

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.runtime_state import RuntimeSessionState
from app.ml.stabilization import ASLStabilizer


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


def print_result(title: str, result) -> None:
    print(title)
    print(f"  prediction: {result.prediction}")
    print(f"  confidence: {result.confidence:.2f}")
    print(f"  raw_prediction: {result.raw_prediction}")
    print(f"  raw_confidence: {result.raw_confidence:.2f}")
    print(f"  stable_prediction: {result.stable_prediction}")
    print(f"  stable_confidence: {result.stable_confidence:.2f}")
    print(f"  stabilization_status: {result.stabilization_status}")
    print(f"  vote_count: {result.vote_count}/{result.vote_window_size}")
    print(f"  note: {result.note}")


def run_vote_window_test(stabilizer: ASLStabilizer) -> None:
    session = RuntimeSessionState(session_id="vote-window")
    last_result = None
    for index in range(1, 7):
        last_result = stabilizer.stabilize(
            session=session,
            top_k_predictions=make_top_k("father", 0.82, "mother", 0.12),
            motion_delta=0.02,
        )
        print(
            f"vote_window frame {index}: "
            f"status={last_result.stabilization_status}, "
            f"votes={last_result.vote_count}/{last_result.vote_window_size}"
        )

    if last_result is None or last_result.stable_prediction != "father":
        raise RuntimeError("Stable output did not appear after enough repeated votes.")

    print_result("vote_window final", last_result)


def run_low_confidence_test(stabilizer: ASLStabilizer) -> None:
    session = RuntimeSessionState(session_id="low-confidence")
    result = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("father", 0.41, "mother", 0.35),
        motion_delta=0.02,
    )
    if result.stabilization_status != "low_confidence":
        raise RuntimeError("Low-confidence rejection test failed.")
    print_result("low_confidence", result)


def run_adaptive_fallback_test(stabilizer: ASLStabilizer) -> None:
    session = RuntimeSessionState(session_id="adaptive")
    result = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("father", 0.58, "mother", 0.22),
        motion_delta=0.02,
    )
    if result.stabilization_status != "collecting_votes":
        raise RuntimeError("Adaptive fallback acceptance did not enter vote collection.")
    print_result("adaptive_fallback", result)


def run_confusion_hold_test(stabilizer: ASLStabilizer) -> None:
    session = RuntimeSessionState(session_id="confusion")
    result = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("approve", 0.72, "draw", 0.69),
        motion_delta=0.02,
    )
    if result.stabilization_status != "held_confusion":
        raise RuntimeError("Confusion-pair hold test failed.")
    print_result("confusion_hold", result)


def run_motion_hold_test(stabilizer: ASLStabilizer) -> None:
    session = RuntimeSessionState(session_id="motion")
    result = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("again", 0.80, "doctor", 0.11),
        motion_delta=0.001,
    )
    if result.stabilization_status != "motion_required":
        raise RuntimeError("Motion hold test failed.")
    print_result("motion_required", result)


def run_peak_acceptance_test(stabilizer: ASLStabilizer) -> None:
    session = RuntimeSessionState(session_id="peak")
    result_one = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("jacket", 0.82, "how", 0.20),
        motion_delta=0.02,
    )
    result_two = stabilizer.stabilize(
        session=session,
        top_k_predictions=make_top_k("jacket", 0.83, "how", 0.18),
        motion_delta=0.02,
    )
    if result_one.stabilization_status != "collecting_votes":
        raise RuntimeError("Peak test frame one should still be collecting votes.")
    if result_two.stabilization_status != "peak_accepted":
        raise RuntimeError("Peak acceptance test failed.")
    print_result("peak_acceptance first", result_one)
    print_result("peak_acceptance second", result_two)


def main() -> None:
    stabilizer = ASLStabilizer()

    run_vote_window_test(stabilizer)
    run_low_confidence_test(stabilizer)
    run_adaptive_fallback_test(stabilizer)
    run_confusion_hold_test(stabilizer)
    run_motion_hold_test(stabilizer)
    run_peak_acceptance_test(stabilizer)

    print("test_stabilization: all checks passed")


if __name__ == "__main__":
    main()
