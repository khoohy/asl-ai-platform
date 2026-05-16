from __future__ import annotations

import argparse
import base64
import statistics
import sys
from pathlib import Path
from time import perf_counter

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.inference_engine import RuntimeInferenceEngine

WARMUP_FRAMES = 3
MEASURED_FRAMES = 10
MAX_PROFILE_SECONDS = 60.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile backend live-inference latency on one still image.",
    )
    parser.add_argument("image_path", type=Path, help="Path to a test image file.")
    args = parser.parse_args()

    profile_started_at = perf_counter()
    current_stage = "startup"

    def check_timeout(stage: str) -> None:
        elapsed = perf_counter() - profile_started_at
        if elapsed > MAX_PROFILE_SECONDS:
            raise TimeoutError(
                f"Latency profiling exceeded {MAX_PROFILE_SECONDS:.0f}s while {stage}."
            )

    image_path = args.image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    current_stage = "loading image"
    check_timeout(current_stage)
    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    print(f"[profile] image: {image_path}")
    print(f"[profile] payload bytes: {len(image_bytes)}")
    print(f"[profile] warmup frames: {WARMUP_FRAMES}")
    print(f"[profile] measured frames: {MEASURED_FRAMES}")

    current_stage = "initializing runtime inference engine"
    check_timeout(current_stage)
    engine = RuntimeInferenceEngine()

    session_id = "latency-profile"
    print("[profile] runtime engine initialized")

    for frame_index in range(1, WARMUP_FRAMES + 1):
        current_stage = f"warmup frame {frame_index}"
        check_timeout(current_stage)
        response = engine.process_frame(image_base64, session_id=session_id)
        print(
            f"[warmup {frame_index:02d}] "
            f"status={response['status']}, "
            f"total_backend_ms={response['timing']['total_backend_ms']:.2f}"
        )

    engine.reset_session(session_id)
    print("[profile] session reset before measured frames")

    measured_rows: list[dict[str, float | str]] = []
    for frame_index in range(1, MEASURED_FRAMES + 1):
        current_stage = f"measured frame {frame_index}"
        check_timeout(current_stage)
        response = engine.process_frame(image_base64, session_id=session_id)
        timing = response.get("timing", {})
        row = {
            "status": str(response.get("status")),
            "total_backend_ms": float(timing.get("total_backend_ms", 0.0)),
            "decode_ms": float(timing.get("base64_decode_ms", 0.0)),
            "image_decode_ms": float(timing.get("image_decode_ms", 0.0)),
            "mediapipe_ms": float(timing.get("mediapipe_ms", 0.0)),
            "feature_ms": float(timing.get("feature_ms", 0.0)),
            "model_ms": float(timing.get("model_ms", 0.0)),
            "stabilization_ms": float(timing.get("stabilization_ms", 0.0)),
        }
        measured_rows.append(row)
        print(
            f"[frame {frame_index:02d}] "
            f"status={row['status']}, "
            f"total_backend_ms={row['total_backend_ms']:.2f}, "
            f"decode_ms={row['decode_ms']:.2f}, "
            f"image_decode_ms={row['image_decode_ms']:.2f}, "
            f"mediapipe_ms={row['mediapipe_ms']:.2f}, "
            f"feature_ms={row['feature_ms']:.2f}, "
            f"model_ms={row['model_ms']:.2f}, "
            f"stabilization_ms={row['stabilization_ms']:.2f}"
        )

    if not measured_rows:
        print("[profile] no measured frames were collected")
        return

    print("\n[profile] averages")
    print_summary(measured_rows, mode="avg")
    print("\n[profile] maxima")
    print_summary(measured_rows, mode="max")
    print(
        "\n[profile] statuses: "
        + ", ".join(str(row["status"]) for row in measured_rows)
    )


def print_summary(rows: list[dict[str, float | str]], mode: str) -> None:
    metrics = [
        "total_backend_ms",
        "decode_ms",
        "image_decode_ms",
        "mediapipe_ms",
        "feature_ms",
        "model_ms",
        "stabilization_ms",
    ]

    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        if mode == "avg":
            value = statistics.mean(values)
        else:
            value = max(values)
        print(f"  {metric}: {value:.2f}")


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as exc:
        print(f"[profile] timeout: {exc}")
        raise
