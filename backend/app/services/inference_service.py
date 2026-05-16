import base64
import binascii
from typing import TYPE_CHECKING, Any

from app.schemas.inference import (
    FrameDebugResponse,
    InferenceRequest,
    MockInferenceResponse,
    RealInferenceRequest,
    RealInferenceResponse,
    ResetSessionRequest,
    ResetSessionResponse,
)

if TYPE_CHECKING:
    from app.ml.frame_processor import FrameProcessor
    from app.ml.inference_engine import RuntimeInferenceEngine

_frame_processor: Any = None
_runtime_inference_engine: Any = None


def _extract_base64_payload(image_base64: str) -> str:
    if "," in image_base64 and image_base64.lower().startswith("data:"):
        return image_base64.split(",", maxsplit=1)[1]
    return image_base64


def _validate_base64_image(image_base64: str) -> None:
    payload = _extract_base64_payload(image_base64).strip()
    if not payload:
        raise ValueError("The image_base64 field cannot be empty.")

    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def run_mock_inference(payload: InferenceRequest) -> MockInferenceResponse:
    _validate_base64_image(payload.image_base64)

    return MockInferenceResponse(
        prediction="hello",
        confidence=0.98,
        model_source="mock-service",
        note=(
            "This is a placeholder response for Phase 1. "
            "Real ASL model inference has not been integrated yet."
        ),
    )


def run_frame_debug(payload: InferenceRequest) -> FrameDebugResponse:
    result = _get_frame_processor().process_base64_image(payload.image_base64)
    return FrameDebugResponse(
        status=result.status,
        feature_dim=result.feature_dim,
        expected_dim=result.expected_dim,
        note=result.note,
    )


def run_real_inference(payload: RealInferenceRequest) -> RealInferenceResponse:
    result = _get_runtime_inference_engine().process_frame(
        image_base64=payload.image_base64,
        session_id=payload.session_id,
    )
    return RealInferenceResponse(**result)


def reset_real_inference_session(
    payload: ResetSessionRequest,
) -> ResetSessionResponse:
    result = _get_runtime_inference_engine().reset_session(payload.session_id)
    return ResetSessionResponse(**result)


def _get_frame_processor() -> "FrameProcessor":
    global _frame_processor
    if _frame_processor is None:
        from app.ml.frame_processor import FrameProcessor

        _frame_processor = FrameProcessor()
    return _frame_processor


def _get_runtime_inference_engine() -> "RuntimeInferenceEngine":
    global _runtime_inference_engine
    if _runtime_inference_engine is None:
        from app.ml.inference_engine import RuntimeInferenceEngine

        _runtime_inference_engine = RuntimeInferenceEngine()
    return _runtime_inference_engine
