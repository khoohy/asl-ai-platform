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
