from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=16,
        description="Base64-encoded image string, with or without a data URI prefix.",
    )


class MockInferenceResponse(BaseModel):
    prediction: str
    confidence: float
    model_source: str
    note: str


class FrameDebugResponse(BaseModel):
    status: str
    feature_dim: int
    expected_dim: int
    note: str
