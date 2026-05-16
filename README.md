# ASL AI Platform

A production-oriented full-stack platform for real-time **American Sign Language (ASL) recognition** using a **React frontend**, **FastAPI backend**, **MediaPipe-based landmark extraction**, and a **30-frame temporal PyTorch model** with live stabilization logic.

This repository is the **product/platform layer** of the ASL system. It is intentionally separate from the original training/research repository so the app, API, runtime serving logic, and frontend experience can evolve cleanly without dragging the full experimental codebase into production.

---

## Overview

**ASL AI Platform** turns a webcam stream into live ASL predictions through a browser-based interface.

The system captures webcam frames in the frontend, sends them to a FastAPI backend, extracts hand/pose/face landmarks, constructs the deployed **180-dimensional feature representation**, maintains a **30-frame rolling sequence buffer**, runs the trained temporal model, and applies **runtime stabilization logic** so predictions are more usable in live settings.

This platform is built around the deployed ASL recognition path, not just raw model inference. It includes:

- real webcam capture in the browser
- frontend hand landmark overlay
- backend MediaPipe preprocessing
- deployed **30 x 180** temporal inference path
- rolling session buffer
- runtime stabilization and confidence control
- hot-buffer behavior so recognition can start from already collected context
- idle / no-hands handling for live usability

---

## Key Features

- **Real-time webcam ASL recognition** in the browser
- **FastAPI backend** for model serving and runtime orchestration
- **React + Vite frontend** for live camera, controls, and prediction display
- **MediaPipe-powered feature extraction** using:
  - hands
  - selected upper-body pose joints
  - compact face landmarks
- **180D per-frame feature representation**
- **30-frame rolling temporal inference**
- **Runtime stabilization layer** including:
  - vote window
  - confidence thresholds
  - adaptive fallback
  - confusion-pair suppression
  - peak-sign preservation
  - no-hands grace handling
- **Hot-buffer recognition flow**
  - camera on starts background buffering
  - recognition can use an already warm buffer
- **Frontend keypoint overlay**
  - browser-side hand landmarks for low-lag visual feedback
- **Released-style UI**
  - cleaner user-facing interface
  - compact controls
  - sticky live prediction panel
  - Top-5 model guesses

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                            Frontend (React)                         │
│                                                                      │
│  Webcam Feed ──> Browser Capture ──> JPEG Frame ──> POST /frame      │
│       │                                                              │
│       └──> Frontend Hand Landmark Overlay (browser-side MediaPipe)   │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Backend (FastAPI)                            │
│                                                                      │
│  Base64 Decode                                                       │
│      ↓                                                               │
│  Image Decode / Resize                                               │
│      ↓                                                               │
│  MediaPipe Extraction                                                │
│    - hands every frame                                               │
│    - pose/face reused across short strides                           │
│      ↓                                                               │
│  180D Feature Construction                                           │
│      ↓                                                               │
│  Rolling Session Buffer (30 frames)                                  │
│      ↓                                                               │
│  PyTorch Temporal Model                                              │
│      ↓                                                               │
│  Runtime Stabilization Layer                                         │
│    - vote window                                                     │
│    - confidence gating                                               │
│    - adaptive fallback                                               │
│    - confusion suppression                                           │
│    - peak-sign preservation                                          │
│    - idle / no-hands handling                                        │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           Live UI Output                             │
│                                                                      │
│  Stable Prediction                                                   │
│  Raw Prediction                                                      │
│  Top-5 Guesses                                                       │
│  Buffer Progress / Readiness                                         │
│  Waiting / Holding / Recognition Status                              │
└──────────────────────────────────────────────────────────────────────┘
```

## Inference Flow

1. User starts the camera in the browser.
2. The frontend begins sending captured frames to the backend.
3. The backend decodes the image and runs MediaPipe extraction.
4. A valid frame is converted into a 180D feature vector.
5. The session buffer stores the latest 30 valid frames.
6. Once enough temporal context exists, the PyTorch model runs on the current `1 x 30 x 180` sequence.
7. Raw predictions are passed through the stabilization layer.
8. The frontend displays:
   - stable prediction
   - raw prediction
   - Top-5 guesses
   - runtime state (`warming_up`, `holding_context`, `waiting_for_hands`, etc.)

---

## Why This Project Exists

The original ASL repository focused on training, experiments, and model evolution.

This repository focuses on serving and productization:

- API boundary
- frontend UX
- runtime session logic
- live stabilization behavior
- browser integration
- maintainable deployment structure

In other words:

- old repo = research + training + model experimentation
- this repo = application platform + runtime serving + live user experience

---

## Current Scope

### Included

- FastAPI backend
- React + Vite frontend
- webcam capture UI
- frontend hand keypoint overlay
- real deployed model loading
- 180D frame preprocessing
- real 30-frame rolling ASL inference
- runtime stabilization
- hot-buffer recognition flow
- no-hands grace and idle handling
- Top-5 live guess display
- architecture and component documentation

### Not Included Yet

- authentication
- WebSocket streaming
- Docker deployment
- sentence-level translation
- TTS in the web app
- cloud deployment
- CI/CD pipeline

---

## Tech Stack

### Frontend

- React
- Vite
- JavaScript
- browser MediaPipe Tasks Vision for low-lag hand overlay

### Backend

- FastAPI
- Uvicorn
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Pydantic

### Model Runtime

- temporal sequence model
- deployed input contract: 30 frames x 180 features
- rolling buffer inference
- stabilization and live-session decision logic

---

## Project Structure

```text
asl-ai-platform/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── ml/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── artifacts/
│   ├── models/
│   ├── scripts/
│   ├── README.md
│   └── requirements.txt
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHANGELOG.md
│   ├── COMPONENT_ANATOMY.md
│   └── INTEGRATION_PLAN.md
├── frontend/
│   ├── src/
│   ├── scripts/
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
└── README.md
```

---

## Backend Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

## Frontend Setup

From the `frontend/` directory:

```powershell
npm install
npm run dev
```

Frontend runs at:

```text
http://127.0.0.1:5173
```

---

## How to Use

### Normal Live Flow

1. Start backend
2. Start frontend
3. Open the browser app
4. Click `Start Camera`
5. Put hands in frame
6. Let the buffer warm
7. Click `Start Recognition`
8. View:
   - stable prediction
   - raw prediction
   - Top-5 guesses
   - buffer readiness
   - waiting / holding / recognition status

### Runtime Behavior

- Camera on starts buffer warming in the background
- Recognition on uses the current hot buffer instead of starting from zero
- Brief hand loss stays in `holding_context`
- Longer absence transitions to `waiting_for_hands`
- No-hand frames do not warm the buffer
- Hand-present valid frames can warm the buffer naturally

---

## API Endpoints

### `GET /health`

Basic backend health check.

### `POST /api/inference/frame`

Primary live inference endpoint.

Used by the frontend for:

- background buffer warming
- active recognition
- runtime state updates

### `POST /api/inference/reset-session`

Clears current session state and rolling buffer.

### `POST /api/inference/mock`

Legacy/dev endpoint kept for internal testing.  
Not part of the main released UI flow.

---

## Runtime Design Notes

### 30-Frame Temporal Contract

The deployed model is not a single-frame classifier.  
It expects a rolling temporal input of:

- `30 frames x 180 features`

That is why the runtime is built around:

- session buffers
- warm-up logic
- context hold behavior
- stabilized output rather than naive per-frame guesses

### 180D Feature Representation

Each valid frame is encoded using:

- hand landmarks
- selected pose joints
- compact face landmarks

### Stabilization Philosophy

The runtime layer does more than just call the model. It includes:

- confidence thresholds
- adaptive fallback
- runner-up margin checks
- confusion-pair suppression
- peak-sign preservation
- vote window logic
- idle/no-hands safety

This is essential for live usability.

---

## Performance Notes

Recent optimizations include:

- frontend hand overlay moved fully into the browser
- pose/face reuse across short backend strides
- smaller JPEG capture payloads
- hot-buffer recognition flow
- cleaner control-state logic

The biggest remaining future architecture upgrade for even lower latency would likely be:

- WebSocket streaming instead of HTTP-per-frame

That is intentionally deferred for now.

---

## Documentation

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Change history: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Detailed component breakdown: [docs/COMPONENT_ANATOMY.md](docs/COMPONENT_ANATOMY.md)
- Integration planning: [docs/INTEGRATION_PLAN.md](docs/INTEGRATION_PLAN.md)
- Backend guide: [backend/README.md](backend/README.md)
- Frontend guide: [frontend/README.md](frontend/README.md)

---

## Roadmap

### Near-term

- refine final live UX
- tune latency further
- improve deployment readiness
- polish project presentation

### Later

- WebSocket streaming
- Dockerization
- deployment config
- CI/CD
- sentence-level or multi-sign extensions
- optional web-based TTS
