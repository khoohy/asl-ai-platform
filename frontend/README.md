# Frontend

Phase 4E adds runtime stabilization on top of the continuous 30-frame real inference flow.

## Features in this phase

- dashboard shell for `ASL AI Platform`
- responsive live inference layout with webcam and prediction visible together on desktop
- backend health status panel
- webcam panel with browser camera access
- single-frame capture to base64
- continuous automatic frame capture for real inference
- webcam-triggered real inference request
- in-memory real inference session reset
- stabilized prediction display on top of raw Top-K output
- separate manual mock inference test panel
- prediction display card
- loading and error states for API calls

## Required backend URL

Create a local environment file from the example and point it to the FastAPI backend:

```powershell
Copy-Item .env.example .env
```

Expected value:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Setup

From the `frontend/` directory:

```powershell
npm install
```

## Run

From the `frontend/` directory:

```powershell
npm run dev
```

The Vite dev server runs at `http://127.0.0.1:5173`.

## Backend requirement

The FastAPI backend must already be running on `http://127.0.0.1:8000` for the dashboard to load health data and run mock inference.

Run both services:

```powershell
# terminal 1, from repo root
uvicorn app.main:app --app-dir backend --reload

# terminal 2, from frontend/
npm run dev
```

## Browser webcam permission

When you click `Start Camera`, the browser will ask for webcam permission. You must allow access for the Phase 4D webcam capture flow to work.

The UI handles common camera states such as:

- permission denied
- no camera found
- camera already in use
- capture not ready
- raw session warming up
- no landmarks
- collecting votes
- held confusion
- low confidence
- stabilized output
- backend request failure

## Current mock request

The manual mock test button sends this placeholder base64 value to the backend mock endpoint:

```text
abcdefghijklmnop
```

The webcam flow now supports two paths:

- manual mock inference through the placeholder endpoint
- manual real inference through the 30-frame rolling model buffer
- continuous real inference through a timed capture loop

## Continuous real inference

The frontend now supports:

- `Start Camera`
- `Stop Camera`
- `Capture frame and run real inference`
- `Start Real Inference Session`
- `Stop Real Inference Session`
- `Reset Real Inference Session`

The continuous capture loop starts at a safe interval of `200ms` per frame, which is about `5 FPS`.

Only one request is allowed in flight at a time. If a frame request is still running, the next timer tick is skipped instead of queueing overlapping requests.

## What Phase 4E adds

This phase proves that the browser can:

- open the webcam
- show a live preview
- capture one frame
- encode it as base64
- send it to the raw sequence inference endpoint
- warm up a 30-frame rolling session buffer
- continue sending frames automatically after the session starts
- display raw Top-K model output once the buffer reaches 30 valid frames
- show when the backend is still collecting votes
- show when a stable output has been accepted
- show when the output is held for low confidence, confusion, or motion reasons

## Live dashboard layout

- desktop:
  - larger left column for webcam preview and capture controls
  - right column for sticky live prediction output, stabilization state, compact Top-K, and collapsible runtime stats
- mobile:
  - webcam and prediction stack vertically
  - the prediction card stays directly under the webcam section

## Raw prediction note

Phase 4E preserves raw predictions but adds a stabilization layer. The UI now shows both:

- raw prediction fields
- stabilized prediction fields

The backend still performs isolated sign recognition only. It does not yet implement sentence-level translation, WebSocket streaming, or TTS.

## Coming later

Phase 4F can build on the stabilized backend with further UX and runtime improvements.

Not included yet:

- authentication
- WebSockets
