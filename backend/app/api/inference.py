from fastapi import APIRouter, HTTPException, status

from app.schemas.inference import (
    FrameDebugResponse,
    InferenceRequest,
    MockInferenceResponse,
    RealInferenceRequest,
    RealInferenceResponse,
    ResetSessionRequest,
    ResetSessionResponse,
)
from app.services.inference_service import (
    reset_real_inference_session,
    run_frame_debug,
    run_mock_inference,
    run_real_inference,
)

router = APIRouter()


@router.post("/mock", response_model=MockInferenceResponse)
def mock_inference(payload: InferenceRequest) -> MockInferenceResponse:
    try:
        return run_mock_inference(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/frame-debug", response_model=FrameDebugResponse)
def frame_debug(payload: InferenceRequest) -> FrameDebugResponse:
    try:
        return run_frame_debug(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/frame", response_model=RealInferenceResponse)
def frame_inference(payload: RealInferenceRequest) -> RealInferenceResponse:
    try:
        return run_real_inference(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/reset-session", response_model=ResetSessionResponse)
def reset_session(payload: ResetSessionRequest) -> ResetSessionResponse:
    try:
        return reset_real_inference_session(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
