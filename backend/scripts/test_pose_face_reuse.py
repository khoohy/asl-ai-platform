from __future__ import annotations

import base64
from pathlib import Path
import sys

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.frame_processor import FrameProcessor
from app.ml.keypoint_extraction import KeypointExtractor


class FakeLandmark:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeLandmarkCollection:
    def __init__(self, points: np.ndarray) -> None:
        self.landmark = [FakeLandmark(float(x), float(y), float(z)) for x, y, z in points]


class FakeHandedness:
    def __init__(self, label: str) -> None:
        self.classification = [type("Classification", (), {"label": label})()]


class FakeHandsResult:
    def __init__(self, hands: list[tuple[str, np.ndarray]] | None = None) -> None:
        if hands:
            self.multi_hand_landmarks = [
                FakeLandmarkCollection(points) for _, points in hands
            ]
            self.multi_handedness = [
                FakeHandedness(label) for label, _ in hands
            ]
        else:
            self.multi_hand_landmarks = None
            self.multi_handedness = None


class FakePoseResult:
    def __init__(self, points: np.ndarray | None = None) -> None:
        self.pose_landmarks = (
            FakeLandmarkCollection(points) if points is not None else None
        )


class FakeFaceResult:
    def __init__(self, points: np.ndarray | None = None) -> None:
        self.multi_face_landmarks = (
            [FakeLandmarkCollection(points)] if points is not None else None
        )


class FakeProcessor:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.call_count = 0

    def process(self, _frame_rgb):
        if not self.outputs:
            raise RuntimeError("FakeProcessor ran out of outputs.")
        self.call_count += 1
        return self.outputs.pop(0)

    def close(self) -> None:
        return None


def make_hand_points(offset: float) -> np.ndarray:
    return np.array(
        [[offset + index * 0.001, 0.2 + index * 0.001, 0.01] for index in range(21)],
        dtype=np.float32,
    )


def make_pose_points(offset: float) -> np.ndarray:
    return np.array(
        [[offset + index * 0.002, 0.3 + index * 0.002, 0.02] for index in range(33)],
        dtype=np.float32,
    )


def make_face_points(offset: float) -> np.ndarray:
    return np.array(
        [[offset + index * 0.0005, 0.4 + index * 0.0005, 0.03] for index in range(478)],
        dtype=np.float32,
    )


def build_fake_extractor_for_reuse_test() -> KeypointExtractor:
    extractor = KeypointExtractor.__new__(KeypointExtractor)
    extractor.confidence_threshold = 0.5
    extractor.pose_face_reuse_enabled = True
    extractor.pose_face_reuse_stride = 3
    extractor.max_reused_pose_face_age = 5
    extractor.frames_since_pose_face_refresh = extractor.pose_face_reuse_stride
    extractor.cached_pose = np.zeros((33, 3), dtype=np.float32)
    extractor.cached_face = np.zeros((478, 3), dtype=np.float32)
    extractor.cached_pose_age = extractor.max_reused_pose_face_age + 1
    extractor.cached_face_age = extractor.max_reused_pose_face_age + 1
    extractor.last_extraction_meta = {}
    extractor.hands = FakeProcessor(
        [
            FakeHandsResult([("Right", make_hand_points(0.1))]),
            FakeHandsResult(None),
            FakeHandsResult([("Left", make_hand_points(0.2))]),
            FakeHandsResult([("Right", make_hand_points(0.25))]),
            FakeHandsResult([("Right", make_hand_points(0.3))]),
        ]
    )
    extractor.pose = FakeProcessor(
        [
            FakePoseResult(make_pose_points(0.1)),
            FakePoseResult(make_pose_points(0.3)),
        ]
    )
    extractor.face_mesh = FakeProcessor(
        [
            FakeFaceResult(make_face_points(0.1)),
            FakeFaceResult(make_face_points(0.3)),
        ]
    )
    return extractor


def build_fake_extractor_for_frame_processor() -> KeypointExtractor:
    extractor = KeypointExtractor.__new__(KeypointExtractor)
    extractor.confidence_threshold = 0.5
    extractor.pose_face_reuse_enabled = True
    extractor.pose_face_reuse_stride = 3
    extractor.max_reused_pose_face_age = 5
    extractor.frames_since_pose_face_refresh = extractor.pose_face_reuse_stride
    extractor.cached_pose = np.zeros((33, 3), dtype=np.float32)
    extractor.cached_face = np.zeros((478, 3), dtype=np.float32)
    extractor.cached_pose_age = extractor.max_reused_pose_face_age + 1
    extractor.cached_face_age = extractor.max_reused_pose_face_age + 1
    extractor.last_extraction_meta = {}
    extractor.hands = FakeProcessor(
        [FakeHandsResult([("Right", make_hand_points(0.5))])]
    )
    extractor.pose = FakeProcessor([FakePoseResult(make_pose_points(0.5))])
    extractor.face_mesh = FakeProcessor([FakeFaceResult(make_face_points(0.5))])
    return extractor


def test_keypoint_reuse() -> None:
    extractor = build_fake_extractor_for_reuse_test()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_one = extractor.extract_keypoints(blank_frame)
    frame_two = extractor.extract_keypoints(blank_frame)
    frame_three = extractor.extract_keypoints(blank_frame)
    frame_four = extractor.extract_keypoints(blank_frame)
    frame_five = extractor.extract_keypoints(blank_frame)

    if extractor.hands.call_count != 5:
        raise RuntimeError("Hands should still run on every frame.")
    if extractor.pose.call_count != 2 or extractor.face_mesh.call_count != 2:
        raise RuntimeError("Pose and face should be reused across the stride window.")

    if not np.any(frame_one["pose"]) or not np.any(frame_one["face"]):
        raise RuntimeError("First frame should populate pose and face caches.")

    if FrameProcessor._has_hand_landmarks(frame_two):
        raise RuntimeError("No-hands detection should depend on the current frame only.")
    if FrameProcessor._has_any_landmarks(frame_two):
        raise RuntimeError("Reused pose/face should not keep no-hands frames alive.")

    if not frame_three["_meta"].get("pose_reused") or not frame_three["_meta"].get("face_reused"):
        raise RuntimeError("Pose/face should be reused on skipped frames inside the stride window.")

    if not frame_four["_meta"].get("pose_reused") or not frame_four["_meta"].get("face_reused"):
        raise RuntimeError("Pose/face should still be reused until the valid-frame stride expires.")

    if frame_five["_meta"].get("pose_reused") or frame_five["_meta"].get("face_reused"):
        raise RuntimeError("Pose/face should refresh again when the reuse stride expires.")

    print("reuse test: hands every frame, pose/face reused by stride")


def test_frame_processor_contract() -> None:
    processor = FrameProcessor()
    processor.extractor = build_fake_extractor_for_frame_processor()

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank_frame)
    if not ok:
        raise RuntimeError("Could not encode the synthetic test image.")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    result = processor.process_base64_image(image_base64)

    if result.status != "ok":
        raise RuntimeError(f"Expected ok frame-processing result, got {result.status}.")
    if result.feature_dim != 180:
        raise RuntimeError(f"Expected 180D feature vector, got {result.feature_dim}.")
    if "mediapipe_ms" not in result.timing or "total_preprocess_ms" not in result.timing:
        raise RuntimeError("Timing fields are missing from the frame-processing result.")

    print(
        "frame processor test: "
        f"status={result.status}, feature_dim={result.feature_dim}, "
        f"mediapipe_ms={result.timing['mediapipe_ms']:.2f}"
    )


def main() -> None:
    test_keypoint_reuse()
    test_frame_processor_contract()
    print("test_pose_face_reuse: all checks passed")


if __name__ == "__main__":
    main()
