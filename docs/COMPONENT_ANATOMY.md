# Component Anatomy

This document explains the current structure of `asl-ai-platform` as a junior AI engineer portfolio project. The repository is intentionally split into a frontend application layer, a FastAPI backend layer, and an ML runtime layer that ports only the minimum production pieces from the older ASL project.

The older ASL project reported that iterative live-runtime refinement improved operational success from about `82.7%` to `91.67%` during 300-sign live testing. In this platform repo, that improvement is treated as a runtime-layer concern rather than a model-retraining story. The production checkpoint, `180D` feature contract, and `30`-frame temporal input remain unchanged; the gain comes from portable decision logic, state handling, and live usability controls now housed in the FastAPI backend runtime layer.

## A. System overview

The current system is a browser-to-backend ASL inference prototype with these layers:

- Browser webcam frontend
  - opens the webcam
  - captures frames manually or on a timed loop
  - sends base64 image frames to the backend
  - runs a browser-side hand keypoint overlay loop for immediate visual feedback
- FastAPI backend
  - exposes health, mock inference, frame-debug, real frame inference, and session reset routes
  - keeps lightweight endpoints fast by avoiding model loading on module import
- MediaPipe preprocessing
  - extracts hands, selected pose landmarks, and selected face landmarks from an image
- 180D feature construction
  - produces the production-aligned per-frame vector:
    - 126 hand features
    - 21 pose features
    - 33 face features
- 30-frame rolling buffer
  - stores a session-local sliding window of valid 180D frame features
- PyTorch model inference
  - runs the real BiLSTM-based production model on a `1 x 30 x 180` tensor
- Raw Top-K prediction output
  - returns unstabilized Top-K labels and confidences from the model
- Frontend overlay path
  - draws hand keypoints directly from the live webcam feed in the browser
  - does not drive inference decisions
- Stabilization layer
  - converts raw model guesses into more usable live output through confidence, vote, peak, confusion, and motion checks
- Idle-state recovery
  - preserves context briefly when hands disappear
  - clears runtime state after prolonged hand loss
  - returns back to waiting instead of predicting from stale buffers
- Runtime-tuning parity
  - ports the old live deployment heuristics that helped raise operational success from about `82.7%` to `91.67%`
  - keeps that tuning in backend decision logic instead of changing the model checkpoint

## B. Frontend anatomy

### `frontend/src/App.jsx`

Responsibility:

- top-level orchestration for frontend data flow
- manages backend health state
- manages real inference response state, including raw and stabilized fields
- passes callbacks and result state into UI components

Important state and props:

- `health`
- `prediction`
- `healthError`
- `realInferenceError`
- loading flags for health, real inference, and session reset

API calls used:

- `fetchHealth()`
- `runRealInference()`
- `resetRealInferenceSession()`

How it interacts with inference flow:

- receives base64 frames from `WebcamPanel`
- sends them to the backend real frame endpoint
- stores the latest backend response so `PredictionCard` and `WebcamPanel` can reflect current session state

### `frontend/src/api/client.js`

Responsibility:

- provides all browser-side HTTP calls to the backend
- centralizes JSON parsing and backend error handling

Important state and props:

- no React state
- uses `VITE_API_BASE_URL`

API calls used:

- `GET /health`
- `POST /api/inference/mock`
- `POST /api/inference/frame-debug`
- `POST /api/inference/frame`
- `POST /api/inference/reset-session`

How it interacts with inference flow:

- converts frontend actions into backend requests
- ensures all UI components use the same backend URL and response parsing behavior

### `frontend/src/components/WebcamPanel.jsx`

Responsibility:

- manage browser webcam access
- run a browser-side hand keypoint overlay loop
- capture frames from the live video preview
- support manual real inference
- support continuous real inference loop
- expose runtime stats and session controls

Important state and props:

- local state:
  - camera status
  - capture status
  - whether the camera is active
  - whether the continuous inference session is running
  - whether a request is in flight
  - runtime counters:
    - frames sent
    - successful responses
    - failed responses
- refs:
  - `videoRef`
  - `canvasRef`
  - `streamRef`
  - `intervalRef`
  - `requestInFlightRef`
  - `runRealInferenceRef`
  - `overlayCanvasRef`
  - `handLandmarkerRef`
  - `animationFrameRef`
- props:
  - `isLoading`
  - `isResetting`
  - `error`
  - `inferenceResult`
  - `onCaptureAndRunRealInference`
  - `onResetRealInferenceSession`

API calls used:

- indirectly uses `POST /api/inference/frame`
- indirectly uses `POST /api/inference/reset-session`
- does not call backend debug routes during normal live overlay rendering

How it interacts with inference flow:

- runs browser-side hand landmark detection for immediate on-screen feedback
- captures the current video frame into a hidden canvas
- converts that frame to base64
- sends the frame into the app-level real inference callback
- on continuous mode, repeats that process at a fixed interval while skipping ticks if the previous request has not finished
- keeps the overlay loop independent from backend inference so keypoints do not lag behind the live video due to round-trip latency

### `frontend/src/components/PredictionCard.jsx`

Responsibility:

- display the latest inference result from either mock or real inference

Important state and props:

- `prediction`
- derived `top_k` list

API calls used:

- none directly

How it interacts with inference flow:

- displays:
  - status
  - prediction
  - confidence
  - raw prediction
  - raw confidence
  - stable prediction
  - stable confidence
  - stabilization status
  - vote count
  - frames collected
  - sequence length
  - model source
  - Top-K list
- acts as the main feedback surface for the user during warm-up, raw prediction, and stabilized output
- keeps raw Top-K and runtime details secondary under `Advanced details`

### `frontend/src/index.css`

Responsibility:

- defines the dashboard layout, typography, spacing, button styling, prediction surfaces, and webcam panel presentation

Important state and props:

- not a React component
- supports visual structure for:
  - hero area
  - health panel
  - webcam panel
  - prediction card
  - Top-K list

API calls used:

- none

How it interacts with inference flow:

- purely presentational
- makes warm-up status, stabilization state, and live inference feedback more understandable to the user

## C. Backend anatomy

### `backend/app/main.py`

Responsibility:

- create the FastAPI app
- register routers
- configure CORS
- expose the root metadata route

Endpoints, schemas, and services involved:

- includes health and inference routers

Lightweight or model-dependent:

- lightweight
- should import quickly after the lazy-loading Phase 4C fix

### `backend/app/api/health.py`

Responsibility:

- expose `GET /health`

Endpoints, schemas, and services involved:

- returns simple service metadata from config

Lightweight or model-dependent:

- lightweight
- no model dependency

### `backend/app/api/inference.py`

Responsibility:

- define inference-related API routes

Endpoints, schemas, and services involved:

- `POST /api/inference/mock`
- `POST /api/inference/frame-debug`
- `POST /api/inference/frame`
- `POST /api/inference/reset-session`
- uses request and response schemas from `backend/app/schemas/inference.py`
- delegates actual work to `backend/app/services/inference_service.py`

Lightweight or model-dependent:

- route module itself is lightweight
- real model work only happens inside the lazily invoked service path

### `backend/app/core/config.py`

Responsibility:

- hold app-level configuration such as app name, version, and environment

Endpoints, schemas, and services involved:

- read by `main.py` and `health.py`

Lightweight or model-dependent:

- lightweight
- no model dependency

### `backend/app/schemas/inference.py`

Responsibility:

- define request and response contracts for inference APIs

Endpoints, schemas, and services involved:

- mock request and response
- frame debug response
- real inference request and response
- Top-K prediction item
- session reset request and response

Lightweight or model-dependent:

- lightweight
- schema-only

### `backend/app/services/inference_service.py`

Responsibility:

- act as the orchestration layer between API routes and ML runtime modules
- perform lazy creation of heavy runtime objects

Endpoints, schemas, and services involved:

- mock inference path
- frame-debug path
- real inference path
- reset-session path

Lightweight or model-dependent:

- mixed
- the module is lightweight at import time
- model loading and MediaPipe initialization are delayed until the real paths are used

## D. ML runtime anatomy

### `backend/app/ml/model_loader.py`

Responsibility:

- load the production checkpoint
- load the label map
- select device
- validate checkpoint assumptions
- expose metadata

Input and output:

- input:
  - checkpoint path
  - label map path
- output:
  - loaded PyTorch model
  - label map object
  - metadata such as `input_dim`, `sequence_length`, `device`, and label count

Dependency on old ASL system:

- directly reflects the production checkpoint format from the older repo

Why it exists in the platform repo:

- the app needs a clean, production-facing model loading boundary independent of training scripts

### `backend/app/ml/sequence_model.py`

Responsibility:

- define the production BiLSTM plus attention-pooling classifier architecture

Input and output:

- input:
  - tensor shaped like `batch x seq_len x input_dim`
- output:
  - logits over the 300 ASL classes

Dependency on old ASL system:

- ported from the production-aligned model definition used in the legacy system

Why it exists in the platform repo:

- the platform must instantiate the exact architecture needed for the copied checkpoint

### `backend/app/ml/label_map.py`

Responsibility:

- load index-to-label mappings from JSON
- support index lookup during Top-K formatting

Input and output:

- input:
  - `label_map_300.json`
- output:
  - `LabelMap` wrapper for integer index to gloss lookup

Dependency on old ASL system:

- depends on the production label map exported from the older repo

Why it exists in the platform repo:

- inference responses need human-readable class labels

### `backend/app/ml/runtime_config.py`

Responsibility:

- centralize model and runtime constants

Input and output:

- input:
  - none at runtime beyond file paths
- output:
  - constants such as:
    - `MODEL_PATH`
    - `LABEL_MAP_PATH`
    - `INPUT_DIM`
    - `SEQUENCE_LENGTH`
    - selected pose and face landmarks
    - stabilization thresholds
    - confusion-pair definitions
    - sign-specific motion and confidence overrides
    - hand-loss grace and idle-state status codes

Dependency on old ASL system:

- mirrors the production assumptions and runtime heuristics documented in the old system
- carries the ported live-tuning values that were instrumental to the old runtime improvement from about `82.7%` to `91.67%`

Why it exists in the platform repo:

- keeps runtime assumptions explicit, centralized, and easy to audit

### `backend/app/ml/keypoint_extraction.py`

Responsibility:

- run MediaPipe hands, pose, and face mesh on a BGR image frame

Input and output:

- input:
  - decoded OpenCV frame
- output:
  - landmark dictionary containing:
    - left hand
    - right hand
    - pose
    - face

Dependency on old ASL system:

- derived from the legacy runtime landmark extraction path

Why it exists in the platform repo:

- the backend needs to transform raw image frames into model-ready landmarks

### `backend/app/ml/preprocessing.py`

Responsibility:

- reproduce the production-aligned WLASL feature engineering logic

Input and output:

- input:
  - hands, pose, and face landmark arrays
- output:
  - normalized slices and feature-ready arrays

Dependency on old ASL system:

- strongly tied to the old WLASL runtime preprocessing behavior

Why it exists in the platform repo:

- parity between the platform runtime and the old training and runtime assumptions is essential for meaningful predictions

### `backend/app/ml/frame_processor.py`

Responsibility:

- accept a base64 image string
- decode it
- build a 180D feature vector
- return structured status codes

Input and output:

- input:
  - base64 image string
- output:
  - `FrameProcessingResult` with:
    - status
    - feature vector or `None`
    - feature dimension
    - note
    - hand-presence flags

Dependency on old ASL system:

- uses the ported landmark extraction and preprocessing logic

Why it exists in the platform repo:

- the backend real inference route needs a clean single-frame preprocessing unit

### `backend/app/ml/runtime_state.py`

Responsibility:

- hold one session's rolling feature buffer and counters
- retain per-session stabilization history and last stable output metadata
- track missing-hand counts, grace frames remaining, and idle state

Input and output:

- input:
  - valid 180D frame vectors
- output:
  - stacked `30 x 180` sequence when ready
  - prediction histories used for stabilization

Dependency on old ASL system:

- indirect
- not copied from the old repo, but built to satisfy the model's 30-frame runtime requirement and the new backend session model

Why it exists in the platform repo:

- session-local buffering and prediction history are application concerns, not training concerns
- it also stores the hold, cooldown, and missing-hand state needed to preserve the old runtime's live usability behavior

### `backend/app/ml/session_manager.py`

Responsibility:

- manage in-memory runtime session states by `session_id`

Input and output:

- input:
  - session id
- output:
  - `RuntimeSessionState`
  - reset behavior
  - simple metadata

Dependency on old ASL system:

- not directly copied

Why it exists in the platform repo:

- the platform needs request-to-request continuity for rolling inference

### `backend/app/ml/inference_engine.py`

Responsibility:

- orchestrate the full real inference path
- lazily load the model
- lazily instantiate frame preprocessing
- append valid features into the session buffer
- run the model when the buffer reaches 30 valid frames
- pass raw predictions through the stabilization layer
- suspend or clear runtime context when hands are absent

Input and output:

- input:
  - base64 frame
  - optional session id
- output:
  - warming-up responses
  - no-landmark or no-hand responses
  - holding-context responses during short hand loss
  - waiting-for-hands responses after idle clearing
  - raw Top-K model outputs
  - stabilization-aware responses that include raw and stable fields

Dependency on old ASL system:

- depends on the production checkpoint, label map, and preprocessing parity assumptions

Why it exists in the platform repo:

- it is the platform-level runtime orchestrator that bridges API input to raw model output and stabilized live feedback
- this is where the unchanged model path is wrapped with the live decision logic that made the old system operationally stronger in practice

### `backend/app/ml/stabilization.py`

Responsibility:

- apply portable decision logic from the old runtime on top of raw Top-K outputs

Input and output:

- input:
  - session runtime state
  - raw Top-K predictions
  - recent motion score
- output:
  - stabilization result containing:
    - raw prediction
    - stable prediction
    - vote count
    - stabilization status
    - explanatory note

Dependency on old ASL system:

- directly inspired by the old desktop runtime logic in `src/main.py` and the associated config values in `src/utils/config.py`
- ports the runtime-layer heuristics that were cited in the old report as instrumental to improving live operational success from about `82.7%` to `91.67%`

Why it exists in the platform repo:

- the new backend needs the old runtime's usability logic, but without the old desktop UI coupling

## E. Inference lifecycle

The current real inference lifecycle is:

1. User starts the camera in the frontend.
2. A browser-side MediaPipe Tasks hand detector reads the live `<video>` frames and draws hand keypoints directly on the overlay canvas for low-latency feedback.
3. The frontend captures a frame from the `<video>` element for backend inference.
4. The frame is drawn to a hidden `<canvas>`.
5. The canvas is encoded as a base64 JPEG string.
6. The frontend sends `POST /api/inference/frame`.
7. The backend receives `image_base64` and an optional `session_id`.
8. The frame processor strips any data URL prefix.
9. The backend base64-decodes the image bytes.
10. OpenCV decodes the bytes into an image frame.
11. Backend MediaPipe extracts hand, pose, and face landmarks.
12. The preprocessing layer constructs the production-aligned 180D feature vector.
13. The session manager retrieves or creates the runtime session state.
14. If no hands are visible, the backend increments the missing-hands counter.
15. During the grace window, the backend returns `holding_context` and keeps the current buffer without appending invalid frames.
16. After the grace window expires, the backend clears the rolling buffer and stabilization history and returns `waiting_for_hands`.
17. If hands are visible, the runtime session appends the valid 180D vector to the rolling buffer.
18. If the buffer holds fewer than 30 valid frames, the backend returns `warming_up`.
19. Once the buffer reaches 30 valid frames, the engine stacks the sequence into shape `1 x 30 x 180`.
20. The PyTorch model runs a forward pass under `torch.no_grad()`.
21. Softmax probabilities are computed.
22. The Top-K class indices are mapped to labels with the label map.
23. The stabilization layer evaluates confidence, margin, votes, confusion pairs, peak history, and motion requirements.
24. The backend can briefly hold the last accepted stable sign during weak transition frames so random in-between guesses do not immediately replace it.
25. The backend returns raw and stabilized prediction data to the frontend.
26. The frontend updates status, prediction, confidence, stabilization state, Top-K list, and frame counters in the UI.

## F. Current limitations

Current limitations:

- no confidence-gated sentence segmentation or translation yet
- no WebSocket streaming yet
- no authentication or persistent session storage yet
- no Docker deployment yet
- session state is in-memory only
- frontend capture is browser-loop based rather than server-pushed streaming
- stabilization is still tuned for isolated sign recognition, not full continuous sentence translation
- no TTS or speech-style cooldown output layer
- runtime tuning parity is now strong for the portable backend decision logic, but it still does not turn the system into continuous sentence translation

## G. Next planned phase

The next planned phase after the current stabilization work is Phase 4F.

Possible Phase 4F topics:

- richer frontend explanation of stabilized state transitions
- optional backend warmup endpoint
- more debugging surfaces for motion and confusion holds
- groundwork for future streaming improvements

Why Phase 4E matters:

- Phase 4C and 4D proved that the raw end-to-end path worked
- Phase 4E is where the system becomes more usable for live signing by reducing flicker, suppressing unstable outputs, and making predictions behave more like the older deployed runtime
