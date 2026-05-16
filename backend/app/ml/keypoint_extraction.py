"""MediaPipe keypoint extraction for hands, pose, and compact face cues."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.ml.runtime_config import (
    MAX_REUSED_POSE_FACE_AGE,
    POSE_FACE_REUSE_ENABLED,
    POSE_FACE_REUSE_STRIDE,
)


class KeypointExtractor:
    """Extract MediaPipe landmarks from a single BGR frame."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self.pose_face_reuse_enabled = POSE_FACE_REUSE_ENABLED
        self.pose_face_reuse_stride = max(1, int(POSE_FACE_REUSE_STRIDE))
        self.max_reused_pose_face_age = max(0, int(MAX_REUSED_POSE_FACE_AGE))
        self.frames_since_pose_face_refresh = self.pose_face_reuse_stride
        self.cached_pose = np.zeros((33, 3), dtype=np.float32)
        self.cached_face = np.zeros((478, 3), dtype=np.float32)
        self.cached_pose_age = self.max_reused_pose_face_age + 1
        self.cached_face_age = self.max_reused_pose_face_age + 1

        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.last_extraction_meta = {
            "hands_detected": False,
            "current_pose_detected": False,
            "current_face_detected": False,
            "pose_reused": False,
            "face_reused": False,
            "pose_face_processed": False,
        }

    def extract_keypoints(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        hand_results = self.hands.process(frame_rgb)

        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        hands_detected = False

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for landmarks, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                    dtype=np.float32,
                )
                if handedness.classification[0].label == "Right":
                    right_hand = points
                else:
                    left_hand = points
            hands_detected = bool(np.any(left_hand) or np.any(right_hand))

        pose, face, pose_face_meta = self._extract_pose_and_face(
            frame_rgb,
            hands_detected=hands_detected,
        )
        self.last_extraction_meta = {
            "hands_detected": hands_detected,
            **pose_face_meta,
        }

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "pose": pose,
            "face": face,
            "_meta": self.last_extraction_meta,
        }

    def _extract_pose_and_face(
        self,
        frame_rgb: np.ndarray,
        hands_detected: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, bool]]:
        if not hands_detected:
            return (
                np.zeros((33, 3), dtype=np.float32),
                np.zeros((478, 3), dtype=np.float32),
                {
                    "current_pose_detected": False,
                    "current_face_detected": False,
                    "pose_reused": False,
                    "face_reused": False,
                    "pose_face_processed": False,
                },
            )

        should_process_pose_face = (
            not self.pose_face_reuse_enabled
            or self.frames_since_pose_face_refresh >= (self.pose_face_reuse_stride - 1)
        )

        pose = np.zeros((33, 3), dtype=np.float32)
        face = np.zeros((478, 3), dtype=np.float32)
        current_pose_detected = False
        current_face_detected = False
        pose_reused = False
        face_reused = False
        pose_face_processed = False

        if should_process_pose_face:
            pose_face_processed = True
            self.frames_since_pose_face_refresh = 0

            pose_results = self.pose.process(frame_rgb)
            face_results = self.face_mesh.process(frame_rgb)

            if pose_results.pose_landmarks:
                pose = np.array(
                    [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                current_pose_detected = bool(np.any(pose))
                if current_pose_detected:
                    self.cached_pose = pose.copy()
                    self.cached_pose_age = 0

            if face_results.multi_face_landmarks:
                points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face_results.multi_face_landmarks[0].landmark],
                    dtype=np.float32,
                )
                face[: len(points)] = points
                current_face_detected = bool(np.any(face))
                if current_face_detected:
                    self.cached_face = face.copy()
                    self.cached_face_age = 0

            if not current_pose_detected:
                pose, pose_reused = self._reuse_cached_pose()
            if not current_face_detected:
                face, face_reused = self._reuse_cached_face()
        else:
            self.frames_since_pose_face_refresh += 1
            pose, pose_reused = self._reuse_cached_pose()
            face, face_reused = self._reuse_cached_face()

        if pose_reused:
            self.cached_pose_age += 1
        elif current_pose_detected:
            self.cached_pose_age = 0

        if face_reused:
            self.cached_face_age += 1
        elif current_face_detected:
            self.cached_face_age = 0

        return (
            pose,
            face,
            {
                "current_pose_detected": current_pose_detected,
                "current_face_detected": current_face_detected,
                "pose_reused": pose_reused,
                "face_reused": face_reused,
                "pose_face_processed": pose_face_processed,
            },
        )

    def _reuse_cached_pose(self) -> tuple[np.ndarray, bool]:
        if (
            self.cached_pose_age <= self.max_reused_pose_face_age
            and np.any(self.cached_pose)
        ):
            return self.cached_pose.copy(), True
        return np.zeros((33, 3), dtype=np.float32), False

    def _reuse_cached_face(self) -> tuple[np.ndarray, bool]:
        if (
            self.cached_face_age <= self.max_reused_pose_face_age
            and np.any(self.cached_face)
        ):
            return self.cached_face.copy(), True
        return np.zeros((478, 3), dtype=np.float32), False

    def release(self) -> None:
        self.hands.close()
        self.pose.close()
        self.face_mesh.close()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
