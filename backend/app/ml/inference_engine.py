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
    MIN_FRAMES_BETWEEN_STABLE_OUTPUTS,
    MODEL_SOURCE,
    SEQUENCE_LENGTH,
    STABLE_OUTPUT_HOLD_FRAMES,
    TOP_K,
    TRANSITION_COOLDOWN_FRAMES,
    TRANSITIONING_STATUS,
    COLLECTING_EVIDENCE_STATUS,
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
        session.advance_output_timers()

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
                keypoint_overlay=frame_result.keypoint_overlay,
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
                keypoint_overlay=frame_result.keypoint_overlay,
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

        stabilizer = self._get_stabilizer()
        stabilization = stabilizer.stabilize(
            session=session,
            top_k_predictions=top_k_predictions,
            motion_delta=motion_delta,
        )

        response = self._build_prediction_response(
            session=session,
            stabilization=stabilization,
            top_k=top_k_predictions,
            keypoint_overlay=frame_result.keypoint_overlay,
        )
        session.last_prediction = response
        session.last_status = str(response["status"])
        return response

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
            held_prediction, held_confidence = self._get_held_display_prediction(session)
            return self._base_response(
                session=session,
                status=HOLDING_CONTEXT_STATUS,
                note=(
                    f"{frame_result.note} "
                    "Holding context while hands are temporarily missing. "
                    f"Grace remaining: {session.hand_grace_remaining}/{HAND_LOSS_GRACE_FRAMES}. "
                    "No new frame was added."
                ),
                prediction=held_prediction,
                confidence=held_confidence,
                hands_detected=False,
                missing_hands_count=session.missing_hands_count,
                grace_frames_remaining=session.hand_grace_remaining,
                raw_prediction=None,
                raw_confidence=0.0,
                stable_prediction=held_prediction,
                stable_confidence=held_confidence,
                stabilization_status=(
                    "holding_output" if held_prediction is not None else "holding_context"
                ),
                keypoint_overlay=frame_result.keypoint_overlay,
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
            keypoint_overlay=frame_result.keypoint_overlay,
        )

    def _build_prediction_response(
        self,
        session,
        stabilization,
        top_k: list[dict[str, float | str]],
        keypoint_overlay: dict[str, list[list[float]]],
    ) -> dict[str, Any]:
        new_stable_prediction = stabilization.stable_prediction
        new_stable_confidence = stabilization.stable_confidence

        if new_stable_prediction is not None:
            if (
                session.last_stable_prediction is not None
                and new_stable_prediction != session.last_stable_prediction
                and session.stable_output_cooldown_remaining > 0
            ):
                held_prediction, held_confidence = self._get_held_display_prediction(session)
                return self._base_response(
                    session=session,
                    status=TRANSITIONING_STATUS,
                    note=(
                        f"Holding {held_prediction} briefly before accepting a new sign. "
                        "Collecting evidence for the next stable output."
                    ),
                    prediction=held_prediction,
                    confidence=held_confidence,
                    top_k=top_k,
                    hands_detected=True,
                    raw_prediction=stabilization.raw_prediction,
                    raw_confidence=stabilization.raw_confidence,
                    stable_prediction=held_prediction,
                    stable_confidence=held_confidence,
                    stabilization_status="holding_output",
                    vote_count=stabilization.vote_count,
                    vote_window_size=stabilization.vote_window_size,
                    keypoint_overlay=keypoint_overlay,
                )

            session.last_stable_prediction = new_stable_prediction
            session.last_stable_confidence = new_stable_confidence
            session.stable_output_hold_remaining = STABLE_OUTPUT_HOLD_FRAMES
            session.stable_output_cooldown_remaining = MIN_FRAMES_BETWEEN_STABLE_OUTPUTS
            session.transition_cooldown_remaining = TRANSITION_COOLDOWN_FRAMES
            return self._base_response(
                session=session,
                status="stabilized",
                note=stabilization.note,
                prediction=new_stable_prediction,
                confidence=new_stable_confidence,
                top_k=top_k,
                hands_detected=True,
                raw_prediction=stabilization.raw_prediction,
                raw_confidence=stabilization.raw_confidence,
                stable_prediction=new_stable_prediction,
                stable_confidence=new_stable_confidence,
                stabilization_status=stabilization.stabilization_status,
                vote_count=stabilization.vote_count,
                vote_window_size=stabilization.vote_window_size,
                keypoint_overlay=keypoint_overlay,
            )

        held_prediction, held_confidence = self._get_held_display_prediction(session)
        if held_prediction is not None:
            return self._base_response(
                session=session,
                status=TRANSITIONING_STATUS,
                note=(
                    "Transitioning between signs. Holding the last accepted sign "
                    "briefly while collecting evidence for the next one."
                ),
                prediction=held_prediction,
                confidence=held_confidence,
                top_k=top_k,
                hands_detected=True,
                raw_prediction=stabilization.raw_prediction,
                raw_confidence=stabilization.raw_confidence,
                stable_prediction=held_prediction,
                stable_confidence=held_confidence,
                stabilization_status="holding_output",
                vote_count=stabilization.vote_count,
                vote_window_size=stabilization.vote_window_size,
                keypoint_overlay=keypoint_overlay,
            )

        return self._base_response(
            session=session,
            status=COLLECTING_EVIDENCE_STATUS,
            note="Collecting evidence for the next sign. Raw candidates remain secondary until stabilization accepts a new output.",
            prediction=None,
            confidence=0.0,
            top_k=top_k,
            hands_detected=True,
            raw_prediction=stabilization.raw_prediction,
            raw_confidence=stabilization.raw_confidence,
            stable_prediction=None,
            stable_confidence=0.0,
            stabilization_status=stabilization.stabilization_status,
            vote_count=stabilization.vote_count,
            vote_window_size=stabilization.vote_window_size,
            keypoint_overlay=keypoint_overlay,
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

    @staticmethod
    def _get_held_display_prediction(session) -> tuple[str | None, float]:
        if (
            session.last_stable_prediction is not None
            and (
                session.stable_output_hold_remaining > 0
                or session.transition_cooldown_remaining > 0
            )
        ):
            return session.last_stable_prediction, session.last_stable_confidence
        return None, 0.0

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
        keypoint_overlay: dict[str, list[list[float]]] | None = None,
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
            "keypoint_overlay": keypoint_overlay
            or {"left_hand": [], "right_hand": [], "pose": [], "face": []},
        }
