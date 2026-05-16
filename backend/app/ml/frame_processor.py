"""Decode a base64 image frame and convert it into a production 180D feature vector."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.keypoint_extraction import KeypointExtractor
from app.ml.preprocessing import WLASLFeatureEngineering
from app.ml.runtime_config import (
    INPUT_DIM,
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


class FrameProcessor:
    """Convert one image frame into the exact 180D runtime feature representation."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self.extractor: KeypointExtractor | None = None

    def process_base64_image(self, image_base64: str) -> FrameProcessingResult:
        extractor = self._get_extractor()
        payload = self._strip_data_url_prefix(image_base64).strip()
        if not payload:
            return self._result("invalid_base64", None, "The image_base64 field is empty.")

        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return self._result(
                "invalid_base64",
                None,
                "The image_base64 payload is not valid base64.",
            )

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return self._result(
                "image_decode_failed",
                None,
                "OpenCV could not decode the provided image bytes.",
            )

        keypoints = extractor.extract_keypoints(frame_bgr)
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
                )
            return self._result(
                "no_landmarks",
                None,
                "MediaPipe did not detect usable hand, pose, or face landmarks.",
                hands_detected=False,
                any_landmarks_detected=False,
                keypoint_overlay=keypoint_overlay,
            )

        feature_vector = self._build_feature_vector(keypoints)
        if feature_vector.shape != (INPUT_DIM,):
            return self._result(
                "wrong_feature_dim",
                feature_vector,
                f"Expected feature vector shape ({INPUT_DIM},), got {feature_vector.shape}.",
                hands_detected=True,
                any_landmarks_detected=any_landmarks_detected,
                keypoint_overlay=keypoint_overlay,
            )

        return self._result(
            "ok",
            feature_vector,
            "Frame decoded and converted into a valid 180D feature vector.",
            hands_detected=True,
            any_landmarks_detected=any_landmarks_detected,
            keypoint_overlay=keypoint_overlay,
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
        return any(
            np.any(np.asarray(keypoints_dict.get(key), dtype=np.float32))
            for key in ("left_hand", "right_hand", "pose", "face")
        )

    @staticmethod
    def _has_hand_landmarks(keypoints_dict: dict[str, np.ndarray]) -> bool:
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
        )
