import base64
import binascii

from app.schemas.inference import InferenceRequest, MockInferenceResponse


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
