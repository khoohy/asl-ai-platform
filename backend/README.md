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
