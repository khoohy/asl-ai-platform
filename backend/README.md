# Backend

## Phase 4A: Real model loading

- Model path: `backend/models/asl_wlasl300_realtime.pt`
- Label map path: `backend/artifacts/label_map_300.json`
- Smoke test command:

```powershell
python backend\scripts\test_model_loading.py
```

- This phase only verifies that the production temporal model and label map can be loaded safely.
- Frame preprocessing, real image decoding, and real inference routes are intentionally deferred to Phase 4B and Phase 4C.

## Phase 4B: Frame preprocessing

- Added dependencies: `mediapipe`, `opencv-python`, and `numpy`
- One image frame can now be decoded and converted into the production-aligned 180D per-frame feature representation
- Optional preprocessing debug endpoint: `POST /api/inference/frame-debug`
- Smoke test command:

```powershell
python backend\scripts\test_frame_preprocessing.py path\to\test_image.jpg
```

- The smoke test now uses a bounded validation path:
  - one still image, or
  - at most 3 sampled video frames at `25%`, `50%`, and `75%`
  - test inputs are resized to width `640` before MediaPipe processing
- This phase validates single-frame decoding and feature construction only.
- Rolling 30-frame buffering and real model prediction remain deferred to Phase 4C.

## Phase 4C: Raw 30-frame inference

- Added a per-session in-memory rolling buffer with sequence length `30`
- Added raw real inference endpoint: `POST /api/inference/frame`
- Added session reset endpoint: `POST /api/inference/reset-session`
- The backend now:
  - accepts one base64 image frame per request
  - converts it into a `180D` feature vector
  - appends valid frames to the session buffer
  - returns `warming_up` until `30/30`
  - returns raw Top-K model predictions once the buffer is full
- Smoke test command:

```powershell
python backend\scripts\test_sequence_inference.py backend\artifacts\phase4b_test_frame.jpg
```

- Predictions in this phase are raw and intentionally not stabilized yet.
- Phase 4D adds continuous frontend capture, while Phase 4E adds backend stabilization.

## Phase 4E: Runtime stabilization

- Added portable stabilization logic on top of raw model inference
- Preserved the existing raw prediction path and Top-K output
- Stabilization now includes:
  - confidence squelch
  - adaptive fallback thresholding
  - runner-up margin checks
  - 10-prediction vote window
  - minimum vote count of `6`
  - confusion-pair suppression
  - sign-specific motion requirements
  - peak-sign fallback for short strong predictions
- New smoke test command:

```powershell
python backend\scripts\test_stabilization.py
```

- The API now returns both raw and stabilized fields.
- This phase still does not implement WebSocket streaming, TTS, or sentence-level translation.

## Phase 4F: Idle-state clearing

- Restored old-runtime hand-loss handling so stale buffered context does not keep producing random signs
- Hand presence is now required for live ASL inference
- Real inference now distinguishes:
  - `no_hands`
  - `no_landmarks`
  - `holding_context`
  - `waiting_for_hands`
- Behavior:
  - brief hand loss preserves context for a short grace period
  - prolonged hand loss clears the rolling buffer and stabilization history
  - idle responses return null predictions instead of stale sign output
- New smoke test command:

```powershell
python backend\scripts\test_idle_state.py
```

- This phase restores the old “Holding context” and “Waiting for hands” behavior without changing the model checkpoint or preprocessing format.
## Release refinement: transition hold and keypoint overlay

- The real inference response now includes a lightweight `keypoint_overlay` payload for:
  - left hand
  - right hand
  - selected pose landmarks
  - selected face landmarks
- The frontend uses that overlay to draw keypoints on the webcam by default.
- The backend now keeps the last accepted stable sign visible for a short transition window instead of immediately promoting weak in-between guesses.
- Added runtime constants for:
  - `STABLE_OUTPUT_HOLD_FRAMES`
  - `MIN_FRAMES_BETWEEN_STABLE_OUTPUTS`
  - `TRANSITION_COOLDOWN_FRAMES`
- Current behavior:
  - accepted signs remain visible briefly while the next sign is forming
  - `holding_context` still handles short hand loss
  - `waiting_for_hands` still clears stale state after the grace period
  - raw Top-K remains available as secondary output rather than the primary user-facing recognition

## Runtime tuning parity

- The old ASL project reported that iterative live-runtime refinement improved operational success from about `82.7%` to `91.67%` during 300-sign live testing.
- In this platform repo, that improvement is treated as runtime-layer behavior rather than a model retraining change.
- The following production assumptions remain unchanged:
  - checkpoint: `backend/models/asl_wlasl300_realtime.pt`
  - preprocessing contract: `180D` per frame
  - temporal input: `30` frames per sequence
- The FastAPI backend now carries the portable tuning logic from the old runtime layer, including:
  - per-sign confidence overrides
  - confusion-pair suppression
  - peak-sign preservation
  - sign-specific motion requirements
  - adaptive lower-confidence acceptance with runner-up margin checks
  - `10`-prediction vote history with a `6`-vote minimum
  - no-hands grace handling with `holding_context` and `waiting_for_hands`
  - stable-output hold and cooldown behavior to reduce random between-sign word flashes
- Focused parity smoke test:

```powershell
python backend\scripts\test_runtime_tuning_parity.py
```

- This parity work does not retrain the model or change the checkpoint. The practical improvement comes from runtime decision logic and live usability controls layered around the same production model path.

## Background buffer warming

- The runtime session now separates:
  - camera-driven background buffering
  - user-facing recognition mode
- `POST /api/inference/frame` accepts `recognition_active`:
  - `true`: maintain the buffer and run full recognition
  - `false`: maintain the buffer only, without running user-facing model inference
- This allows the frontend to:
  - start warming the rolling `30`-frame buffer as soon as the camera is on
  - start recognition on top of an already-hot buffer
  - stop recognition without destroying the warm buffer if the camera remains on
  - clear the session when the camera stops or when the user resets
- hand-loss grace is now time-based as well as safety-oriented:
  - `HAND_LOSS_GRACE_MS = 2000`
  - brief hand loss stays in `holding_context`
  - longer absence still clears to `waiting_for_hands`
- Focused regression test:

```powershell
python backend\scripts\test_background_buffering.py
```
