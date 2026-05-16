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

from app.ml.frame_processor import FrameProcessor

MAX_FRAMES_TO_SCAN = 3
TARGET_FRAME_FRACTIONS = (0.25, 0.50, 0.75)
TARGET_WIDTH = 640


def resize_frame(frame):
    height, width = frame.shape[:2]
    if width <= TARGET_WIDTH:
        return frame
    scale = TARGET_WIDTH / float(width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)


def encode_frame_to_base64(frame, mime_type: str = "image/jpeg") -> str:
    resized = resize_frame(frame)
    ok, buffer = cv2.imencode(".jpg", resized)
    if not ok:
        raise RuntimeError("Failed to encode the test frame as JPEG.")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def encode_image_to_base64(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"OpenCV could not load image: {image_path}")
    return encode_frame_to_base64(frame, mime_type=mime_type)


def sample_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fractions = TARGET_FRAME_FRACTIONS[:MAX_FRAMES_TO_SCAN]
        for fraction in fractions:
            frame_index = int(frame_count * fraction) if frame_count > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok and frame is not None:
                yield fraction, frame_index, frame
    finally:
        cap.release()


def print_result(label: str, result) -> None:
    print(label)
    print(f"status: {result.status}")
    print(
        "feature_shape: "
        + (
            str(tuple(result.feature_vector.shape))
            if result.feature_vector is not None
            else "None"
        )
    )
    print(f"feature_dim: {result.feature_dim}")
    print(f"expected_dim: {result.expected_dim}")
    print(f"feature_dim_matches: {result.feature_dim == result.expected_dim}")
    print(
        "output_shape_is_180: "
        + str(result.feature_vector is not None and result.feature_vector.shape == (180,))
    )
    if result.feature_vector is not None:
        print(f"nonzero_values: {int((result.feature_vector != 0).sum())}")
    else:
        print("nonzero_values: unavailable")
    print(f"note: {result.note}")
    if result.status == "no_landmarks":
        print("warning: no landmarks were detected in the supplied frame.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one image frame into a production 180D feature vector.",
    )
    parser.add_argument("input_path", nargs="?", help="Path to a test image or video.")
    args = parser.parse_args()

    if not args.input_path:
        print("No input path provided.")
        print(
            "Usage: python backend\\scripts\\test_frame_preprocessing.py path\\to\\test_image.jpg"
        )
        return

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    processor = FrameProcessor()

    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        payload = encode_image_to_base64(input_path)
        result = processor.process_base64_image(payload)
        print_result(f"input: {input_path}", result)
        return

    if input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        print(f"input: {input_path}")
        print(
            f"video_sampling: max_frames_to_scan={MAX_FRAMES_TO_SCAN}, positions={TARGET_FRAME_FRACTIONS[:MAX_FRAMES_TO_SCAN]}"
        )
        any_frame = False
        for fraction, frame_index, frame in sample_video_frames(input_path):
            any_frame = True
            payload = encode_frame_to_base64(frame)
            result = processor.process_base64_image(payload)
            print_result(
                f"sampled_frame: index={frame_index}, fraction={fraction:.2f}",
                result,
            )
        if not any_frame:
            print("warning: no readable frames were found in the supplied video.")
        return

    raise ValueError(
        "Unsupported input type. Provide an image or video file path."
    )


if __name__ == "__main__":
    main()
