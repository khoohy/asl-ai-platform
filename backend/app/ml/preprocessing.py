"""Production-aligned WLASL300 frame preprocessing utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.ml.runtime_config import SELECTED_FACE_LANDMARKS, SELECTED_POSE_LANDMARKS


class WLASLFeatureEngineering:
    """Shared frame preprocessing ported from the old production runtime path."""

    WRIST_IDX = 0
    DEFAULT_POSE_JOINTS = tuple(SELECTED_POSE_LANDMARKS)
    DEFAULT_FACE_LANDMARKS = tuple(SELECTED_FACE_LANDMARKS)

    @staticmethod
    def _scale_hand_minmax(hand: np.ndarray) -> np.ndarray:
        if hand.shape != (21, 3):
            return np.zeros((21, 3), dtype=np.float32)
        if not np.any(hand):
            return hand.astype(np.float32)

        scaled = hand.astype(np.float32).copy()
        mins = scaled.min(axis=0)
        maxs = scaled.max(axis=0)
        spans = maxs - mins

        for coord_idx in range(3):
            if spans[coord_idx] > 1e-6:
                coord = (scaled[:, coord_idx] - mins[coord_idx]) / spans[coord_idx]
                scaled[:, coord_idx] = coord * 2.0 - 1.0
            else:
                scaled[:, coord_idx] = 0.0

        return scaled

    @staticmethod
    def normalize_landmarks(landmarks_frame: np.ndarray) -> np.ndarray:
        if landmarks_frame is None:
            return np.zeros((42, 3), dtype=np.float32)

        frame = np.asarray(landmarks_frame, dtype=np.float32)
        if frame.shape != (42, 3):
            return np.zeros((42, 3), dtype=np.float32)

        normalized_hands: list[np.ndarray] = []
        for start_idx in (0, 21):
            hand = frame[start_idx:start_idx + 21].copy()
            if np.any(hand):
                wrist = hand[WLASLFeatureEngineering.WRIST_IDX].copy()
                hand -= wrist
                hand = WLASLFeatureEngineering._scale_hand_minmax(hand)
            else:
                hand = np.zeros((21, 3), dtype=np.float32)
            normalized_hands.append(hand)

        return np.vstack(normalized_hands).astype(np.float32)

    @staticmethod
    def extract_pose_frame(
        frame_data: Sequence,
        selected_joints: Sequence[int] | None = None,
    ) -> np.ndarray:
        joints = tuple(selected_joints or WLASLFeatureEngineering.DEFAULT_POSE_JOINTS)
        frame = np.asarray(frame_data, dtype=np.float32)
        if frame.shape == (33, 3):
            pose = frame
        elif frame.ndim == 1 and frame.size == 99:
            pose = frame.reshape(33, 3)
        elif frame.ndim == 2 and frame.shape[1] >= 3 and frame.shape[0] >= 33:
            pose = frame[:33, :3]
        else:
            return np.zeros((len(joints), 3), dtype=np.float32)
        return pose[list(joints)].astype(np.float32)

    @staticmethod
    def extract_face_frame(
        frame_data: Sequence,
        selected_landmarks: Sequence[int] | None = None,
    ) -> np.ndarray:
        landmarks = tuple(
            selected_landmarks or WLASLFeatureEngineering.DEFAULT_FACE_LANDMARKS
        )
        frame = np.asarray(frame_data, dtype=np.float32)
        landmark_count = len(landmarks)

        if frame.ndim == 1 and frame.size % 3 == 0:
            frame = frame.reshape(-1, 3)

        if frame.ndim != 2 or frame.shape[1] < 3:
            return np.zeros((landmark_count, 3), dtype=np.float32)

        face = frame[:, :3]
        if face.shape[0] <= max(landmarks):
            return np.zeros((landmark_count, 3), dtype=np.float32)
        return face[list(landmarks)].astype(np.float32)

    @staticmethod
    def normalize_pose_landmarks(
        pose_frame: np.ndarray,
        shoulder_pair: tuple[int, int] = (1, 2),
    ) -> np.ndarray:
        frame = np.asarray(pose_frame, dtype=np.float32)
        if frame.ndim != 2 or frame.shape[1] != 3:
            return np.zeros_like(frame, dtype=np.float32)
        if not np.any(frame):
            return np.zeros_like(frame, dtype=np.float32)

        normalized = frame.copy()
        left_idx, right_idx = shoulder_pair
        if frame.shape[0] > max(left_idx, right_idx):
            center = (frame[left_idx] + frame[right_idx]) / 2.0
            scale = np.linalg.norm(frame[left_idx, :2] - frame[right_idx, :2])
        else:
            center = frame.mean(axis=0)
            scale = np.max(np.linalg.norm(frame[:, :2] - center[:2], axis=1))

        normalized -= center
        scale = float(scale) if scale > 1e-6 else 1.0
        normalized /= scale
        normalized = np.clip(normalized, -2.0, 2.0)
        return normalized.astype(np.float32)

    @staticmethod
    def normalize_face_landmarks(
        face_frame: np.ndarray,
        eye_pair: tuple[int, int] = (-2, -1),
    ) -> np.ndarray:
        frame = np.asarray(face_frame, dtype=np.float32)
        if frame.ndim != 2 or frame.shape[1] != 3:
            return np.zeros_like(frame, dtype=np.float32)
        if not np.any(frame):
            return np.zeros_like(frame, dtype=np.float32)

        normalized = frame.copy()
        left_idx, right_idx = eye_pair
        left_eye = frame[left_idx]
        right_eye = frame[right_idx]
        center = (left_eye + right_eye) / 2.0
        scale = np.linalg.norm(left_eye[:2] - right_eye[:2])
        if scale <= 1e-6:
            nose_idx = min(3, len(frame) - 1)
            center = frame[nose_idx]
            scale = np.max(np.linalg.norm(frame[:, :2] - center[:2], axis=1))

        normalized -= center
        scale = float(scale) if scale > 1e-6 else 1.0
        normalized /= scale
        normalized = np.clip(normalized, -2.0, 2.0)
        return normalized.astype(np.float32)
