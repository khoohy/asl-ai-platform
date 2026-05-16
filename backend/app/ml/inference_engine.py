"""Runtime raw 30-frame inference engine for the production ASL model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from app.ml.frame_processor import FrameProcessor
from app.ml.model_loader import ASLModelLoader
from app.ml.runtime_config import (
    HAND_LOSS_GRACE_FRAMES,
    HOLDING_CONTEXT_STATUS,
    IDLE_STATUS,
    INPUT_DIM,
    MODEL_SOURCE,
    SEQUENCE_LENGTH,
    TOP_K,
    WAITING_FOR_HANDS_STATUS,
)
from app.ml.session_manager import SessionManager
from app.ml.stabilization import ASLStabilizer


@dataclass
class RuntimeInferenceEngine:
    """Accept single frames, maintain a session buffer, and run raw model inference."""

    model_loader: ASLModelLoader | None = None
    frame_processor: FrameProcessor | None = None
    session_manager: SessionManager | None = None
    stabilizer: ASLStabilizer | None = None
    top_k: int = TOP_K

    def __post_init__(self) -> None:
        if self.session_manager is None:
            self.session_manager = SessionManager(
                sequence_length=SEQUENCE_LENGTH,
                input_dim=INPUT_DIM,
            )

    def process_frame(
        self,
        image_base64: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        loader = self._get_model_loader()
        frame_processor = self._get_frame_processor()

        if loader.model is None or loader.label_map is None:
            raise RuntimeError("Model load issue: production model is not ready.")

        if self.session_manager is None:
            raise RuntimeError("Session buffer issue: session manager is not initialized.")

        session = self.session_manager.get_session(session_id)
        session.total_frames_received += 1

        frame_result = frame_processor.process_base64_image(image_base64)
        if frame_result.status in {"no_hands", "no_landmarks"}:
            return self._handle_missing_hands(session, frame_result)

        if frame_result.status != "ok" or frame_result.feature_vector is None:
            session.is_idle = False
            session.missing_hands_count = 0
            session.hand_grace_remaining = HAND_LOSS_GRACE_FRAMES
            session.last_status = frame_result.status
            return self._base_response(
                session=session,
                status=frame_result.status,
                note=(
                    frame_result.note
                    + f" No valid frame was added. Buffer remains at "
                    f"{session.valid_frames_collected}/{session.sequence_length}."
                ),
                hands_detected=True,
            )

        session.is_idle = False
        session.missing_hands_count = 0
        session.hand_grace_remaining = HAND_LOSS_GRACE_FRAMES
        session.append(frame_result.feature_vector)

        if session.valid_frames_collected < session.sequence_length:
            session.last_status = "warming_up"
            return self._base_response(
                session=session,
                status="warming_up",
                note=(
                    f"Collecting valid frames: "
                    f"{session.valid_frames_collected}/{session.sequence_length}"
                ),
                hands_detected=True,
            )

        sequence = session.stack_sequence()
        motion_delta = self._compute_motion_delta(sequence)
        session.last_motion_score = motion_delta
        batch = (
            torch.from_numpy(sequence)
            .float()
            .unsqueeze(0)
            .to(loader.device)
        )

        with torch.no_grad():
            logits = loader.model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=min(self.top_k, probabilities.shape[0]),
            )

        top_k_predictions: list[dict[str, float | str]] = []
        for probability, index in zip(
            top_probabilities.cpu().tolist(),
            top_indices.cpu().tolist(),
        ):
            top_k_predictions.append(
                {
                    "label": loader.label_map.get_label(int(index)),
                    "confidence": float(probability),
                }
            )

        prediction = top_k_predictions[0] if top_k_predictions else None
        stabilizer = self._get_stabilizer()
        stabilization = stabilizer.stabilize(
            session=session,
            top_k_predictions=top_k_predictions,
            motion_delta=motion_delta,
        )

        session.last_prediction = {
            "prediction": stabilization.prediction,
            "confidence": stabilization.confidence,
            "top_k": top_k_predictions,
            "raw_prediction": stabilization.raw_prediction,
            "raw_confidence": stabilization.raw_confidence,
            "stable_prediction": stabilization.stable_prediction,
            "stable_confidence": stabilization.stable_confidence,
            "stabilization_status": stabilization.stabilization_status,
            "vote_count": stabilization.vote_count,
            "vote_window_size": stabilization.vote_window_size,
        }
        session.last_status = self._map_status(stabilization.stabilization_status)

        return self._base_response(
            session=session,
            status=session.last_status,
            note=stabilization.note,
            prediction=stabilization.prediction,
            confidence=stabilization.confidence,
            top_k=top_k_predictions,
            hands_detected=True,
            raw_prediction=stabilization.raw_prediction,
            raw_confidence=stabilization.raw_confidence,
            stable_prediction=stabilization.stable_prediction,
            stable_confidence=stabilization.stable_confidence,
            stabilization_status=stabilization.stabilization_status,
            vote_count=stabilization.vote_count,
            vote_window_size=stabilization.vote_window_size,
        )

    def reset_session(self, session_id: str | None = None) -> dict[str, str]:
        if self.session_manager is None:
            self.session_manager = SessionManager(
                sequence_length=SEQUENCE_LENGTH,
                input_dim=INPUT_DIM,
            )
        session = self.session_manager.reset_session(session_id)
        return {
            "status": "reset",
            "session_id": session.session_id,
            "note": "Session buffer cleared",
        }

    def _handle_missing_hands(self, session, frame_result) -> dict[str, Any]:
        session.missing_hands_count += 1
        session.hand_grace_remaining = max(
            HAND_LOSS_GRACE_FRAMES - session.missing_hands_count,
            0,
        )

        if session.valid_frames_collected > 0 and session.missing_hands_count <= HAND_LOSS_GRACE_FRAMES:
            session.last_status = HOLDING_CONTEXT_STATUS
            session.is_idle = False
            return self._base_response(
                session=session,
                status=HOLDING_CONTEXT_STATUS,
                note=(
                    f"{frame_result.note} "
                    "Holding context while hands are temporarily missing. "
                    f"Grace remaining: {session.hand_grace_remaining}/{HAND_LOSS_GRACE_FRAMES}. "
                    "No new frame was added."
                ),
                hands_detected=False,
                missing_hands_count=session.missing_hands_count,
                grace_frames_remaining=session.hand_grace_remaining,
            )

        session.clear_runtime_context()
        session.is_idle = True
        session.last_status = WAITING_FOR_HANDS_STATUS if session.missing_hands_count > 0 else IDLE_STATUS
        session.hand_grace_remaining = 0
        return self._base_response(
            session=session,
            status=session.last_status,
            note=(
                f"{frame_result.note} Waiting for hands. "
                "The rolling buffer and stabilization history were cleared."
            ),
            hands_detected=False,
            missing_hands_count=session.missing_hands_count,
            grace_frames_remaining=0,
        )

    def _get_model_loader(self) -> ASLModelLoader:
        if self.model_loader is None or not self.model_loader.model_loaded:
            self.model_loader = ASLModelLoader().load()
            if self.session_manager is not None:
                self.session_manager.sequence_length = self.model_loader.sequence_length
                self.session_manager.input_dim = self.model_loader.input_dim
        return self.model_loader

    def _get_frame_processor(self) -> FrameProcessor:
        if self.frame_processor is None:
            self.frame_processor = FrameProcessor()
        return self.frame_processor

    def _get_stabilizer(self) -> ASLStabilizer:
        if self.stabilizer is None:
            self.stabilizer = ASLStabilizer()
        return self.stabilizer

    @staticmethod
    def _compute_motion_delta(sequence) -> float:
        if sequence.ndim != 2 or len(sequence) < 2:
            return 0.0
        hand_dims = min(126, sequence.shape[1])
        deltas = sequence[1:, :hand_dims] - sequence[:-1, :hand_dims]
        return float(torch.from_numpy(deltas).abs().mean().item())

    @staticmethod
    def _map_status(stabilization_status: str) -> str:
        mapping = {
            "stable": "stabilized",
            "collecting_votes": "collecting_votes",
            "held_confusion": "held_confusion",
            "low_confidence": "low_confidence",
            "peak_accepted": "stabilized",
            "motion_required": "motion_required",
            "raw_only": "raw_predicted",
        }
        return mapping.get(stabilization_status, "raw_predicted")

    def _base_response(
        self,
        session,
        status: str,
        note: str,
        prediction: str | None = None,
        confidence: float = 0.0,
        top_k: list[dict[str, float | str]] | None = None,
        hands_detected: bool = False,
        missing_hands_count: int | None = None,
        grace_frames_remaining: int | None = None,
        raw_prediction: str | None = None,
        raw_confidence: float = 0.0,
        stable_prediction: str | None = None,
        stable_confidence: float = 0.0,
        stabilization_status: str = "raw_only",
        vote_count: int = 0,
        vote_window_size: int = 10,
    ) -> dict[str, Any]:
        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_k": top_k or [],
            "hands_detected": hands_detected,
            "missing_hands_count": (
                session.missing_hands_count
                if missing_hands_count is None
                else missing_hands_count
            ),
            "grace_frames_remaining": (
                session.hand_grace_remaining
                if grace_frames_remaining is None
                else grace_frames_remaining
            ),
            "raw_prediction": raw_prediction,
            "raw_confidence": raw_confidence,
            "stable_prediction": stable_prediction,
            "stable_confidence": stable_confidence,
            "stabilization_status": stabilization_status,
            "vote_count": vote_count,
            "vote_window_size": vote_window_size,
            "model_source": MODEL_SOURCE,
            "status": status,
            "frames_collected": session.valid_frames_collected,
            "sequence_length": session.sequence_length,
            "note": note,
        }
