from fastapi import APIRouter, HTTPException, status

from app.schemas.inference import InferenceRequest, MockInferenceResponse
from app.services.inference_service import run_mock_inference

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
