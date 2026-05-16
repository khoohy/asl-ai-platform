from fastapi import APIRouter, HTTPException, status

from app.schemas.inference import (
    FrameDebugResponse,
    InferenceRequest,
    MockInferenceResponse,
)
from app.services.inference_service import run_frame_debug, run_mock_inference

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
