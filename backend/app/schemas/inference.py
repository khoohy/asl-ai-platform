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


class RealInferenceRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=16,
        description="Base64-encoded image string, with or without a data URI prefix.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional in-memory session identifier for the rolling 30-frame buffer.",
    )


class TopKPrediction(BaseModel):
    label: str
    confidence: float


class KeypointOverlay(BaseModel):
    left_hand: list[list[float]] = Field(default_factory=list)
    right_hand: list[list[float]] = Field(default_factory=list)
    pose: list[list[float]] = Field(default_factory=list)
    face: list[list[float]] = Field(default_factory=list)


class FrameDebugResponse(BaseModel):
    status: str
    feature_dim: int
    expected_dim: int
    note: str
    hands_detected: bool = False
    keypoint_overlay: KeypointOverlay = Field(default_factory=KeypointOverlay)


class RealInferenceResponse(BaseModel):
    prediction: str | None
    confidence: float
    top_k: list[TopKPrediction]
    hands_detected: bool = False
    missing_hands_count: int = 0
    grace_frames_remaining: int = 0
    raw_prediction: str | None = None
    raw_confidence: float = 0.0
    stable_prediction: str | None = None
    stable_confidence: float = 0.0
    stabilization_status: str = "raw_only"
    vote_count: int = 0
    vote_window_size: int = 10
    model_source: str
    status: str
    frames_collected: int
    sequence_length: int
    note: str
    keypoint_overlay: KeypointOverlay = Field(default_factory=KeypointOverlay)


class ResetSessionRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier. Defaults to the shared local session.",
    )


class ResetSessionResponse(BaseModel):
    status: str
    session_id: str
    note: str
