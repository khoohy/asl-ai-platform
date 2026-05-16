"""MediaPipe keypoint extraction for hands, pose, and compact face cues."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class KeypointExtractor:
    """Extract MediaPipe landmarks from a single BGR frame."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

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

    def extract_keypoints(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        pose = np.zeros((33, 3), dtype=np.float32)
        face = np.zeros((478, 3), dtype=np.float32)

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

        if pose_results.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
                dtype=np.float32,
            )

        if face_results.multi_face_landmarks:
            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_results.multi_face_landmarks[0].landmark],
                dtype=np.float32,
            )
            face[: len(points)] = points

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "pose": pose,
            "face": face,
        }

    def release(self) -> None:
        self.hands.close()
        self.pose.close()
        self.face_mesh.close()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
