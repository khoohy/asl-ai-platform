"""Decode a base64 image frame and convert it into a production 180D feature vector."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from app.ml.keypoint_extraction import KeypointExtractor
from app.ml.preprocessing import WLASLFeatureEngineering
from app.ml.runtime_config import (
    INPUT_DIM,
    MAX_PROCESSING_WIDTH,
    SELECTED_FACE_LANDMARKS,
    SELECTED_POSE_LANDMARKS,
)


@dataclass
class FrameProcessingResult:
    status: str
    feature_vector: np.ndarray | None
    feature_dim: int
    expected_dim: int
    note: str
    hands_detected: bool
    any_landmarks_detected: bool
    keypoint_overlay: dict[str, list[list[float]]]
    timing: dict[str, float]


class FrameProcessor:
    """Convert one image frame into the exact 180D runtime feature representation."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self.extractor: KeypointExtractor | None = None

    def process_base64_image(self, image_base64: str) -> FrameProcessingResult:
        request_started_at = perf_counter()
        timing = self._create_timing()
        extractor = self._get_extractor()
        payload = self._strip_data_url_prefix(image_base64).strip()
        if not payload:
            return self._result(
                "invalid_base64",
                None,
                "The image_base64 field is empty.",
                timing=self._finalize_timing(timing, request_started_at),
            )

        try:
            decode_started_at = perf_counter()
            image_bytes = base64.b64decode(payload, validate=True)
            timing["base64_decode_ms"] = self._elapsed_ms(decode_started_at)
        except (binascii.Error, ValueError):
            return self._result(
                "invalid_base64",
                None,
                "The image_base64 payload is not valid base64.",
                timing=self._finalize_timing(timing, request_started_at),
            )

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_decode_started_at = perf_counter()
        frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        timing["image_decode_ms"] = self._elapsed_ms(image_decode_started_at)
        if frame_bgr is None:
            return self._result(
                "image_decode_failed",
                None,
                "OpenCV could not decode the provided image bytes.",
                timing=self._finalize_timing(timing, request_started_at),
            )

        resize_started_at = perf_counter()
        frame_bgr = self._resize_for_processing(frame_bgr)
        timing["image_resize_ms"] = self._elapsed_ms(resize_started_at)

        mediapipe_started_at = perf_counter()
        keypoints = extractor.extract_keypoints(frame_bgr)
        timing["mediapipe_ms"] = self._elapsed_ms(mediapipe_started_at)
        keypoint_overlay = self._build_keypoint_overlay(keypoints)
        hands_detected = self._has_hand_landmarks(keypoints)
        any_landmarks_detected = self._has_any_landmarks(keypoints)
        if not hands_detected:
            if any_landmarks_detected:
                return self._result(
                    "no_hands",
                    None,
                    "Pose or face landmarks were detected, but no hands are visible.",
                    hands_detected=False,
                    any_landmarks_detected=True,
                    keypoint_overlay=keypoint_overlay,
                    timing=self._finalize_timing(timing, request_started_at),
                )
            return self._result(
                "no_landmarks",
                None,
                "MediaPipe did not detect usable hand, pose, or face landmarks.",
                hands_detected=False,
                any_landmarks_detected=False,
                keypoint_overlay=keypoint_overlay,
                timing=self._finalize_timing(timing, request_started_at),
            )

        feature_started_at = perf_counter()
        feature_vector = self._build_feature_vector(keypoints)
        timing["feature_ms"] = self._elapsed_ms(feature_started_at)
        if feature_vector.shape != (INPUT_DIM,):
            return self._result(
                "wrong_feature_dim",
                feature_vector,
                f"Expected feature vector shape ({INPUT_DIM},), got {feature_vector.shape}.",
                hands_detected=True,
                any_landmarks_detected=any_landmarks_detected,
                keypoint_overlay=keypoint_overlay,
                timing=self._finalize_timing(timing, request_started_at),
            )

        return self._result(
            "ok",
            feature_vector,
            "Frame decoded and converted into a valid 180D feature vector.",
            hands_detected=True,
            any_landmarks_detected=any_landmarks_detected,
            keypoint_overlay=keypoint_overlay,
            timing=self._finalize_timing(timing, request_started_at),
        )

    def _build_feature_vector(self, keypoints_dict: dict[str, np.ndarray]) -> np.ndarray:
        left_hand = np.asarray(
            keypoints_dict.get("left_hand", np.zeros((21, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        right_hand = np.asarray(
            keypoints_dict.get("right_hand", np.zeros((21, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        hand_frame = np.vstack([left_hand, right_hand]).astype(np.float32)
        hand_features = WLASLFeatureEngineering.normalize_landmarks(hand_frame).reshape(-1)

        pose_frame = WLASLFeatureEngineering.extract_pose_frame(
            keypoints_dict.get("pose", np.zeros((33, 3), dtype=np.float32)),
        )
        pose_features = WLASLFeatureEngineering.normalize_pose_landmarks(
            pose_frame,
            shoulder_pair=(1, 2),
        ).reshape(-1)

        face_frame = WLASLFeatureEngineering.extract_face_frame(
            keypoints_dict.get("face", np.zeros((478, 3), dtype=np.float32)),
        )
        face_features = WLASLFeatureEngineering.normalize_face_landmarks(face_frame).reshape(-1)

        return np.concatenate([hand_features, pose_features, face_features]).astype(
            np.float32
        )

    def _get_extractor(self) -> KeypointExtractor:
        if self.extractor is None:
            self.extractor = KeypointExtractor(
                confidence_threshold=self.confidence_threshold,
            )
        return self.extractor

    @staticmethod
    def _resize_for_processing(frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.size == 0:
            return frame_bgr

        height, width = frame_bgr.shape[:2]
        if width <= MAX_PROCESSING_WIDTH:
            return frame_bgr

        scale = MAX_PROCESSING_WIDTH / float(width)
        target_size = (
            MAX_PROCESSING_WIDTH,
            max(1, int(round(height * scale))),
        )
        return cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _build_keypoint_overlay(
        keypoints_dict: dict[str, np.ndarray],
    ) -> dict[str, list[list[float]]]:
        left_hand = np.asarray(
            keypoints_dict.get("left_hand", np.zeros((21, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        right_hand = np.asarray(
            keypoints_dict.get("right_hand", np.zeros((21, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        pose = np.asarray(
            keypoints_dict.get("pose", np.zeros((33, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        face = np.asarray(
            keypoints_dict.get("face", np.zeros((478, 3), dtype=np.float32)),
            dtype=np.float32,
        )

        return {
            "left_hand": FrameProcessor._points_to_overlay(left_hand),
            "right_hand": FrameProcessor._points_to_overlay(right_hand),
            "pose": FrameProcessor._points_to_overlay(pose[SELECTED_POSE_LANDMARKS]),
            "face": FrameProcessor._points_to_overlay(face[SELECTED_FACE_LANDMARKS]),
        }

    @staticmethod
    def _points_to_overlay(points: np.ndarray) -> list[list[float]]:
        if points.size == 0 or not np.any(points):
            return []
        return [
            [float(point[0]), float(point[1])]
            for point in np.asarray(points, dtype=np.float32)
        ]

    @staticmethod
    def _strip_data_url_prefix(image_base64: str) -> str:
        if "," in image_base64 and image_base64.lower().startswith("data:"):
            return image_base64.split(",", maxsplit=1)[1]
        return image_base64

    @staticmethod
    def _has_any_landmarks(keypoints_dict: dict[str, np.ndarray]) -> bool:
        metadata = keypoints_dict.get("_meta", {})
        if metadata:
            return bool(
                metadata.get("hands_detected")
                or metadata.get("current_pose_detected")
                or metadata.get("current_face_detected")
            )
        return any(
            np.any(np.asarray(keypoints_dict.get(key), dtype=np.float32))
            for key in ("left_hand", "right_hand", "pose", "face")
        )

    @staticmethod
    def _has_hand_landmarks(keypoints_dict: dict[str, np.ndarray]) -> bool:
        metadata = keypoints_dict.get("_meta", {})
        if metadata:
            return bool(metadata.get("hands_detected"))
        return any(
            np.any(np.asarray(keypoints_dict.get(key), dtype=np.float32))
            for key in ("left_hand", "right_hand")
        )

    @staticmethod
    def _result(
        status: str,
        feature_vector: np.ndarray | None,
        note: str,
        hands_detected: bool = False,
        any_landmarks_detected: bool = False,
        keypoint_overlay: dict[str, list[list[float]]] | None = None,
        timing: dict[str, float] | None = None,
    ) -> FrameProcessingResult:
        feature_dim = int(feature_vector.shape[0]) if feature_vector is not None else 0
        return FrameProcessingResult(
            status=status,
            feature_vector=feature_vector,
            feature_dim=feature_dim,
            expected_dim=INPUT_DIM,
            note=note,
            hands_detected=hands_detected,
            any_landmarks_detected=any_landmarks_detected,
            keypoint_overlay=keypoint_overlay or {
                "left_hand": [],
                "right_hand": [],
                "pose": [],
                "face": [],
            },
            timing=timing or FrameProcessor._create_timing(),
        )

    @staticmethod
    def _create_timing() -> dict[str, float]:
        return {
            "base64_decode_ms": 0.0,
            "image_decode_ms": 0.0,
            "image_resize_ms": 0.0,
            "mediapipe_ms": 0.0,
            "feature_ms": 0.0,
            "total_preprocess_ms": 0.0,
        }

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (perf_counter() - started_at) * 1000.0

    @staticmethod
    def _finalize_timing(
        timing: dict[str, float],
        request_started_at: float,
    ) -> dict[str, float]:
        finalized = dict(timing)
        finalized["total_preprocess_ms"] = (perf_counter() - request_started_at) * 1000.0
        return finalized
