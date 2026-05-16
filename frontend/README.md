# Frontend

Phase 4C adds raw 30-frame model inference on top of the existing webcam capture flow.

## Features in this phase

- dashboard shell for `ASL AI Platform`
- backend health status panel
- webcam panel with browser camera access
- single-frame capture to base64
- webcam-triggered raw real inference request
- in-memory real inference session reset
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

When you click `Start Camera`, the browser will ask for webcam permission. You must allow access for the Phase 4C webcam capture flow to work.

The UI handles common camera states such as:

- permission denied
- no camera found
- camera already in use
- capture not ready
- raw session warming up
- no landmarks
- backend request failure

## Current mock request

The manual mock test button sends this placeholder base64 value to the backend mock endpoint:

```text
abcdefghijklmnop
```

The webcam flow now supports two paths:

- manual mock inference through the placeholder endpoint
- raw real inference through the 30-frame rolling model buffer

## What Phase 4C adds

This phase proves that the browser can:

- open the webcam
- show a live preview
- capture one frame
- encode it as base64
- send it to the raw sequence inference endpoint
- warm up a 30-frame rolling session buffer
- display raw Top-K model output once the buffer reaches 30 valid frames

## Raw prediction note

Phase 4C returns raw model predictions only. Predictions are not stabilized yet, so you may need to capture multiple frames until the buffer reaches `30/30`, and the output may still flicker or be noisy.

## Coming later

Phase 4D will add stabilization, voting, and confusion-handling logic on top of the raw model output.

Not included yet:

- authentication
- WebSockets
