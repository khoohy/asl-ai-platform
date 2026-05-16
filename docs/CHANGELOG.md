# Changelog

This document records the development history of `asl-ai-platform` phase by phase. The project was built as a production-oriented full-stack platform around an existing ASL recognition system, while intentionally keeping the platform repo separate from the older research and training repo.

## Initial repo setup

### Purpose

Create a clean repository for the product platform rather than copying the older ASL project wholesale.

### Files created or modified

- `README.md`
- `.gitignore`
- base directories:
  - `backend/`
  - `frontend/`
  - `docs/`

### Major design decisions

- keep the new repo separate from the legacy ASL model repo
- treat this repo as the application and integration layer, not the training codebase
- avoid copying datasets, archived checkpoints, notebooks, and experiment scripts

### Verification and tests performed

- confirmed the repo had a clean directory structure for backend, frontend, and docs work

### Known limitations

- no runnable backend or frontend yet
- no model integration yet

### Intentionally deferred

- all app implementation work
- all inference integration
- all UI work

## Phase 1: FastAPI backend scaffold

### Purpose

Establish a minimal backend skeleton with a health route and a mock inference route so the application architecture existed before real model integration.

### Files created or modified

- `backend/app/main.py`
- `backend/app/api/health.py`
- `backend/app/api/inference.py`
- `backend/app/core/config.py`
- `backend/app/schemas/inference.py`
- `backend/app/services/inference_service.py`
- `backend/requirements.txt`
- `frontend/README.md`
- `docs/ARCHITECTURE.md`
- `docs/INTEGRATION_PLAN.md`
- `README.md`
- `.gitignore`
- `docker-compose.yml`

### Major design decisions

- use FastAPI for a clean API-first backend
- keep `/health` lightweight and dependency-free
- add `/api/inference/mock` to validate request and response wiring without tying the system to the real model
- document the separation between platform code and model or training code early

### Verification and tests performed

- Python compile check over `backend/app`
- manual run instructions documented for `uvicorn`

### Known limitations

- no real model inference
- no image preprocessing
- no authentication
- no WebSockets

### Intentionally deferred

- actual ASL checkpoint loading
- frontend implementation
- Docker deployment details

## Phase 2: React frontend mock inference flow

### Purpose

Add a basic React dashboard that could reach the FastAPI backend, check health, and run the mock inference endpoint.

### Files created or modified

- `frontend/package.json`
- `frontend/vite.config.js`
- `frontend/.env.example`
- `frontend/index.html`
- `frontend/src/main.jsx`
- `frontend/src/App.jsx`
- `frontend/src/api/client.js`
- `frontend/src/components/HealthStatus.jsx`
- `frontend/src/components/MockInferencePanel.jsx`
- `frontend/src/components/PredictionCard.jsx`
- `frontend/src/index.css`
- `frontend/README.md`
- `README.md`
- `backend/app/main.py` for minimal CORS support

### Major design decisions

- use React plus Vite for a lightweight frontend scaffold
- keep the frontend intentionally narrow: health, mock inference, and result display
- add environment-based backend URL configuration
- add only the minimal backend CORS support needed for local browser development

### Verification and tests performed

- backend compile check after the CORS change
- structure verification for the Vite frontend

### Known limitations

- no webcam access yet
- no real model integration
- no rolling session state

### Intentionally deferred

- production React architecture
- camera capture
- real inference requests

## Phase 3: webcam capture UI

### Purpose

Prove that the browser could open a webcam, capture a frame, encode it as base64, and send it to the backend.

### Files created or modified

- `frontend/src/components/WebcamPanel.jsx`
- `frontend/src/App.jsx`
- `frontend/src/api/client.js`
- `frontend/src/components/MockInferencePanel.jsx`
- `frontend/src/index.css`
- `frontend/README.md`
- `README.md`

### Major design decisions

- keep webcam support frontend-only at first
- support both manual webcam capture and the older placeholder mock path
- explicitly handle browser camera permission, missing camera, and capture failures
- release the camera stream cleanly on stop and unmount

### Verification and tests performed

- code-level verification of webcam cleanup and capture logic
- manual browser workflow documented

### Known limitations

- backend still returned mock predictions only
- no real preprocessing
- no model-backed inference

### Intentionally deferred

- MediaPipe
- real checkpoint use
- streaming or automatic capture

## Phase 4A: real ASL model loading and smoke test

### Purpose

Port the minimum production model artifacts into the platform repo and prove the backend could load the real checkpoint safely.

### Files created or modified

- copied:
  - `backend/models/asl_wlasl300_realtime.pt`
  - `backend/artifacts/label_map_300.json`
- created:
  - `backend/app/ml/__init__.py`
  - `backend/app/ml/runtime_config.py`
  - `backend/app/ml/sequence_model.py`
  - `backend/app/ml/label_map.py`
  - `backend/app/ml/model_loader.py`
  - `backend/scripts/test_model_loading.py`
  - `backend/README.md`
- modified:
  - `backend/requirements.txt`
  - `.gitignore`

### Major design decisions

- copy only the production checkpoint, label map, and minimal architecture definition
- preserve the known production assumptions:
  - `sequence_length = 30`
  - `input_dim = 180`
- keep loading separate from actual runtime inference

### Verification and tests performed

- smoke test loading the checkpoint
- label map load verification
- dummy forward pass on shape `1 x 30 x 180`
- output shape verification `1 x 300`

### Known limitations

- no image preprocessing yet
- no live frame ingestion
- no API route using the real model

### Intentionally deferred

- MediaPipe runtime path
- rolling sequence buffering
- frontend real inference integration

## Phase 4B: MediaPipe frame preprocessing to 180D feature vector

### Purpose

Port the old runtime preprocessing logic so one image frame could become the exact 180D per-frame feature representation expected by the production model.

### Files created or modified

- created:
  - `backend/app/ml/keypoint_extraction.py`
  - `backend/app/ml/preprocessing.py`
  - `backend/app/ml/frame_processor.py`
  - `backend/scripts/test_frame_preprocessing.py`
- modified:
  - `backend/app/ml/runtime_config.py`
  - `backend/app/schemas/inference.py`
  - `backend/app/services/inference_service.py`
  - `backend/app/api/inference.py`
  - `backend/README.md`
  - `backend/requirements.txt`

### Major design decisions

- preserve preprocessing parity with the old production path instead of inventing new normalization rules
- port the selected pose and face landmark slices directly from the old WLASL runtime path
- keep `frame-debug` separate from real inference so preprocessing could be validated independently
- use bounded testing rather than unbounded frame scanning

### Verification and tests performed

- compile checks over backend modules
- single-frame preprocessing smoke test
- validation that a test frame produced:
  - `status = ok`
  - `feature_shape = (180,)`
  - `feature_dim = 180`

### Known limitations

- single-frame feature extraction only
- no rolling 30-frame buffer yet
- no model prediction yet

### Intentionally deferred

- raw model inference route
- stabilization logic
- continuous capture

## Phase 4C: 30-frame rolling buffer and raw model inference

### Purpose

Implement real inference by buffering 30 valid 180D frames per session and running the model on the latest sequence without any stabilization logic.

### Files created or modified

- created:
  - `backend/app/ml/runtime_state.py`
  - `backend/app/ml/session_manager.py`
  - `backend/app/ml/inference_engine.py`
  - `backend/scripts/test_sequence_inference.py`
- modified:
  - `backend/app/ml/keypoint_extraction.py`
  - `backend/app/ml/frame_processor.py`
  - `backend/app/schemas/inference.py`
  - `backend/app/services/inference_service.py`
  - `backend/app/api/inference.py`
  - `frontend/src/api/client.js`
  - `frontend/src/App.jsx`
  - `frontend/src/components/WebcamPanel.jsx`
  - `frontend/src/components/PredictionCard.jsx`
  - `frontend/src/index.css`
  - `backend/README.md`
  - `frontend/README.md`
  - `README.md`

### Major design decisions

- use an in-memory session manager keyed by `session_id`
- append only valid 180D feature vectors
- return `warming_up` until the session holds 30 valid frames
- return raw Top-K softmax outputs once the sequence is full
- make model loading lazy so importing `app.main` stays lightweight
- make MediaPipe initialization lazy so `/health` and `/mock` remain fast

### Verification and tests performed

- direct runtime engine smoke test
- verified progression from `warming_up` to `predicted`
- verified raw Top-5 output on the 30th frame
- verified `app.main` import finished quickly after lazy initialization changes

### Known limitations

- predictions are raw and may flicker
- no stabilization, voting, motion gating, or confusion handling
- in-memory session state only

### Intentionally deferred

- confidence squelch
- session persistence
- WebSockets
- automatic frontend loop

## Phase 4D: continuous frontend capture loop

### Purpose

Make raw real inference usable from the browser by automatically sending frames at a controlled interval instead of requiring 30 manual clicks.

### Files created or modified

- `frontend/src/App.jsx`
- `frontend/src/components/WebcamPanel.jsx`
- `frontend/src/components/PredictionCard.jsx`
- `frontend/src/api/client.js`
- `frontend/src/index.css`
- `frontend/README.md`
- `README.md`

### Major design decisions

- run the continuous loop in the frontend only
- use a safe starting interval of `200ms` per frame, about `5 FPS`
- prevent overlapping requests with an in-flight ref
- keep manual capture buttons and mock inference flows intact
- stop the loop when the camera stops or the component unmounts
- keep the backend session alive across predictions so raw output can keep updating

### Verification and tests performed

- logic review of interval cleanup and request overlap prevention
- browser test steps documented for start, stop, and reset behavior

### Known limitations

- still raw predictions only
- output may fluctuate frame to frame
- no backend stabilization or filtering yet

### Intentionally deferred

- stabilization and voting
- adaptive confidence logic
- confusion-pair suppression
- motion gating
- peak detection
- WebSocket streaming

## Phase 4E: runtime stabilization logic

### Purpose

Port the old runtime's portable stabilization logic into the FastAPI backend so live predictions become more usable and less flickery without changing the model, preprocessing, or raw Top-K path.

### Files created or modified

- created:
  - `backend/app/ml/stabilization.py`
  - `backend/scripts/test_stabilization.py`
- modified:
  - `backend/app/ml/runtime_config.py`
  - `backend/app/ml/runtime_state.py`
  - `backend/app/ml/inference_engine.py`
  - `backend/app/schemas/inference.py`
  - `backend/scripts/test_sequence_inference.py`
  - `frontend/src/App.jsx`
  - `frontend/src/components/PredictionCard.jsx`
  - `frontend/src/components/WebcamPanel.jsx`
  - `backend/README.md`
  - `frontend/README.md`
  - `README.md`
  - `docs/COMPONENT_ANATOMY.md`
  - `docs/CHANGELOG.md`

### Major design decisions

- keep raw Top-K predictions visible instead of replacing them
- port portable logic only:
  - base confidence threshold
  - adaptive fallback floor
  - runner-up margin logic
  - 10-prediction vote history
  - minimum vote count of 6
  - confusion-pair suppression
  - sign-specific motion requirements
  - peak-sign fallback
- keep stabilization state per session inside the runtime state object
- preserve lazy loading so app import remains fast

### Verification and tests performed

- direct stabilizer smoke test for:
  - vote collection to stable output
  - low-confidence rejection
  - adaptive fallback acceptance
  - confusion hold
  - motion hold
  - peak acceptance
- updated sequence inference test to print stabilization fields after the 30th frame

### Known limitations

- no WebSocket streaming yet
- no TTS integration in the platform repo
- motion logic is still lightweight and based on recent hand-feature deltas
- still isolated sign recognition rather than sentence-level translation

### Intentionally deferred

- backend speech or TTS suppression logic
- persistent session storage
- continuous language-level segmentation

## Phase 4F: idle-state clearing for missing hands

### Purpose

Restore the old deployed runtime behavior where brief hand loss preserves context temporarily, but prolonged hand absence clears buffered state and returns the system to an honest waiting-for-hands mode.

### Files created or modified

- created:
  - `backend/scripts/test_idle_state.py`
- modified:
  - `backend/app/ml/runtime_config.py`
  - `backend/app/ml/frame_processor.py`
  - `backend/app/ml/runtime_state.py`
  - `backend/app/ml/inference_engine.py`
  - `backend/app/schemas/inference.py`
  - `backend/scripts/test_sequence_inference.py`
  - `frontend/src/components/PredictionCard.jsx`
  - `backend/README.md`
  - `docs/COMPONENT_ANATOMY.md`
  - `docs/CHANGELOG.md`

### Major design decisions

- treat visible hands as the required signal for ASL inference
- distinguish `no_hands` from `no_landmarks`
- preserve rolling context for a short grace period
- clear the rolling buffer and stabilization history after sustained hand loss
- return null predictions during idle rather than showing stale or random signs

### Verification and tests performed

- dedicated idle-state smoke test with:
  - valid-hand warmup
  - missing-hand grace transition
  - waiting-for-hands clearing
- updated sequence inference test output to include hand-loss metadata

### Known limitations

- idle handling is still in-memory and session-local
- hand-loss timing is frame-count based rather than wall-clock based
- continuous sentence segmentation is still out of scope

### Intentionally deferred

- persistent session state
- WebSocket delivery
- speech or TTS integration

## Phase 4F frontend follow-up: capture latency tuning

### Purpose

Reduce perceived warm-up latency in the live webcam experience by making the continuous capture loop faster and user-configurable without changing backend inference logic.

### Files created or modified

- modified:
  - `frontend/src/App.jsx`
  - `frontend/src/components/WebcamPanel.jsx`
  - `frontend/src/components/PredictionCard.jsx`
  - `frontend/README.md`
  - `docs/CHANGELOG.md`

### Major design decisions

- add frontend capture presets instead of hard-coding one interval
- default to `Fast` at `100ms` per frame
- keep the in-flight guard so overlapping requests never stack
- count skipped ticks explicitly so operators can see when the frontend is outrunning backend response time

### Verification and tests performed

- code-path review of timer restart behavior and skip-on-in-flight handling
- runtime stats surface updated to show selected interval and skipped ticks

### Known limitations

- effective frame rate still depends on backend response speed
- this is still timer-based browser capture rather than streaming

### Intentionally deferred

- WebSocket transport
- backend-side queueing or adaptive sampling

## Release refinement: product-facing UI and transition hold

### Purpose

Polish the live ASL recognition experience so it behaves more like a released realtime product and less like a developer validation dashboard.

### Files created or modified

- modified:
  - `backend/app/ml/runtime_config.py`
  - `backend/app/ml/runtime_state.py`
  - `backend/app/ml/frame_processor.py`
  - `backend/app/ml/inference_engine.py`
  - `backend/app/schemas/inference.py`
  - `backend/app/services/inference_service.py`
  - `frontend/src/api/client.js`
  - `frontend/src/App.jsx`
  - `frontend/src/components/WebcamPanel.jsx`
  - `frontend/src/components/PredictionCard.jsx`
  - `frontend/src/index.css`
  - `backend/README.md`
  - `frontend/README.md`
  - `docs/CHANGELOG.md`

### Major design decisions

- remove visible mock inference and capture-speed controls from the main UI
- hardcode the live loop to the fastest validated interval of `100ms`
- shorten the top status cards to product-facing labels:
  - backend
  - mode
  - session
- add a lightweight keypoint overlay payload to backend frame responses so the webcam can show detection guidance by default
- keep raw Top-K output available, but demote it into advanced details
- stop using raw model guesses as the primary displayed output during weak transition periods
- add a short post-stable-output hold so the previous accepted sign remains visible while the next sign is still being formed

### Verification and tests performed

- backend compile verification
- existing idle-state logic preserved conceptually:
  - short hand loss still holds context
  - prolonged hand loss still clears state and returns to waiting-for-hands
- frontend code-path review for:
  - hidden mock UI
  - default keypoint overlay
  - fixed 100ms capture loop
  - advanced-details-only debug controls

### Known limitations

- keypoint overlay depends on backend frame processing and is not fully frontend-local
- preview overlay refresh while the camera is idle uses a lightweight debug request path
- raw candidates still exist internally and in advanced details, but the main output is now intentionally more conservative

### Intentionally deferred

- WebSockets
- sentence-level translation
- TTS
- deployment packaging

## Frontend overlay refinement: low-latency keypoints

### Purpose

Replace delayed backend-driven keypoint overlays with a browser-side overlay path so the visible hand landmarks track the live webcam feed more closely.

### Files created or modified

- created:
  - `frontend/src/lib/handLandmarker.js`
- modified:
  - `frontend/package.json`
  - `frontend/src/components/WebcamPanel.jsx`
  - `frontend/README.md`
  - `docs/COMPONENT_ANATOMY.md`
  - `docs/CHANGELOG.md`

### Major design decisions

- keep backend inference logic unchanged
- stop using backend `keypoint_overlay` data as the primary live webcam overlay source
- use browser-side MediaPipe Tasks hand landmarks for visual feedback only
- keep the overlay hand-only in the browser to reduce latency and CPU load
- keep backend MediaPipe preprocessing as the recognition source of truth
- stop polling `/api/inference/frame-debug` during normal live camera use

### Verification and tests performed

- code-path review of the new browser-side overlay loop
- ensured the visible overlay no longer depends on `/api/inference/frame` or `/api/inference/frame-debug` round trips

### Known limitations

- browser-side overlay depends on the frontend MediaPipe Tasks package and its model assets
- frontend overlay is hand-only, while backend inference still uses hands plus selected pose and face features

### Intentionally deferred

- browser-side pose and face overlays
- WebSocket streaming
- full offline packaging of frontend overlay model assets

## Runtime tuning parity audit and fix

### Purpose

Verify that the new FastAPI backend preserves the old live-runtime tuning that helped improve operational success from about `82.7%` to `91.67%`, and fix any remaining parity gaps without changing the model itself.

### Files created or modified

- created:
  - `backend/scripts/test_runtime_tuning_parity.py`
- modified:
  - `backend/app/ml/stabilization.py`
  - `backend/README.md`
  - `docs/COMPONENT_ANATOMY.md`
  - `docs/CHANGELOG.md`

### Major design decisions

- treat the old live-runtime improvement as a backend decision-logic concern rather than a retraining task
- preserve the existing production contract:
  - same checkpoint
  - same `180D` preprocessing
  - same `30`-frame sequence length
- audit old and new runtime behavior feature by feature before changing code
- fix only the confirmed parity gap:
  - the stable-output cooldown path now compares against the previously accepted stable sign before overwriting it

### Verification and tests performed

- `python backend\scripts\test_stabilization.py`
- `python backend\scripts\test_idle_state.py`
- `python backend\scripts\test_runtime_tuning_parity.py`
- `python -m compileall backend\app backend\scripts`

The parity script verifies:

- stable signs are not immediately replaced by weak random predictions
- no-hands transitions still go `holding_context -> waiting_for_hands`
- low-confidence raw predictions do not become stable outputs
- confusion-pair near-ties are held
- peak-sign preservation still works
- the `10`-prediction window and `6`-vote requirement still work
- adaptive fallback only accepts lower-confidence predictions when the runner-up margin is strong enough

### Known limitations

- the platform ports the portable runtime logic, not the old desktop OpenCV UI behavior
- the old live improvement also depended partly on user execution refinement, not only backend logic
- the system still remains an isolated-sign recognizer rather than a continuous sentence translator

### Intentionally deferred

- retraining the model
- changing the checkpoint
- changing the `180D` feature construction
- changing the `30`-frame sequence length
