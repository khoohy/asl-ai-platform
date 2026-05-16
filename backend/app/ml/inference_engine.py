"""Runtime raw 30-frame inference engine for the production ASL model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from app.ml.frame_processor import FrameProcessor
from app.ml.model_loader import ASLModelLoader
from app.ml.runtime_config import INPUT_DIM, MODEL_SOURCE, SEQUENCE_LENGTH
from app.ml.session_manager import SessionManager


@dataclass
class RuntimeInferenceEngine:
    """Accept single frames, maintain a session buffer, and run raw model inference."""

    model_loader: ASLModelLoader | None = None
    frame_processor: FrameProcessor | None = None
    session_manager: SessionManager | None = None
    top_k: int = 5

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
        if frame_result.status != "ok" or frame_result.feature_vector is None:
            session.last_status = frame_result.status
            return self._base_response(
                session=session,
                status=frame_result.status,
                note=(
                    frame_result.note
                    + f" No valid frame was added. Buffer remains at "
                    f"{session.valid_frames_collected}/{session.sequence_length}."
                ),
            )

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
            )

        sequence = session.stack_sequence()
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
        session.last_prediction = {
            "prediction": prediction["label"] if prediction else None,
            "confidence": float(prediction["confidence"]) if prediction else 0.0,
            "top_k": top_k_predictions,
        }
        session.last_status = "predicted"

        return self._base_response(
            session=session,
            status="predicted",
            note="Raw model prediction from latest 30-frame buffer",
            prediction=prediction["label"] if prediction else None,
            confidence=float(prediction["confidence"]) if prediction else 0.0,
            top_k=top_k_predictions,
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

    def _base_response(
        self,
        session,
        status: str,
        note: str,
        prediction: str | None = None,
        confidence: float = 0.0,
        top_k: list[dict[str, float | str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_k": top_k or [],
            "model_source": MODEL_SOURCE,
            "status": status,
            "frames_collected": session.valid_frames_collected,
            "sequence_length": session.sequence_length,
            "note": note,
        }
