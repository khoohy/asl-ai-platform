from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

import cv2

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.inference_engine import RuntimeInferenceEngine

TARGET_WIDTH = 640
TEST_SESSION_ID = "phase4c-sequence-test"


def resize_frame(frame):
    height, width = frame.shape[:2]
    if width <= TARGET_WIDTH:
        return frame
    scale = TARGET_WIDTH / float(width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)


def encode_image_to_base64(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"OpenCV could not load image: {image_path}")
    resized = resize_frame(frame)
    ok, buffer = cv2.imencode(".jpg", resized)
    if not ok:
        raise RuntimeError("Failed to encode the test image as JPEG.")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the same frame 30 times to the raw sequence inference engine.",
    )
    parser.add_argument("image_path", help="Path to a test image.")
    args = parser.parse_args()

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    engine = RuntimeInferenceEngine()
    engine.reset_session(TEST_SESSION_ID)
    payload = encode_image_to_base64(image_path)

    final_response = None
    for iteration in range(1, 31):
        response = engine.process_frame(payload, session_id=TEST_SESSION_ID)
        final_response = response
        print(
            f"frame {iteration:02d}: status={response['status']}, "
            f"frames_collected={response['frames_collected']}/{response['sequence_length']}"
        )
        if iteration < 30 and response["status"] not in {
            "warming_up",
            "no_landmarks",
            "holding_context",
            "waiting_for_hands",
            "idle",
        }:
            print("warning: early response was not warming_up/no_landmarks")

    if final_response is None:
        raise RuntimeError("Sequence test did not produce any responses.")

    print("final_response:")
    for key in (
        "status",
        "prediction",
        "confidence",
        "hands_detected",
        "missing_hands_count",
        "grace_frames_remaining",
        "raw_prediction",
        "raw_confidence",
        "stable_prediction",
        "stable_confidence",
        "stabilization_status",
        "vote_count",
        "vote_window_size",
        "frames_collected",
        "sequence_length",
        "note",
    ):
        print(f"  {key}: {final_response.get(key)}")

    if final_response["top_k"]:
        print("top_k:")
        for item in final_response["top_k"]:
            print(f"  - {item['label']}: {item['confidence']:.6f}")
    else:
        print("top_k: unavailable")


if __name__ == "__main__":
    main()
