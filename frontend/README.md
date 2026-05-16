# Frontend

Phase 3 adds a React + Vite webcam capture flow on top of the existing mock-backend dashboard.

## Features in this phase

- dashboard shell for `ASL AI Platform`
- backend health status panel
- webcam panel with browser camera access
- single-frame capture to base64
- webcam-triggered mock inference request
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

When you click `Start Camera`, the browser will ask for webcam permission. You must allow access for the Phase 3 webcam capture flow to work.

The UI handles common camera states such as:

- permission denied
- no camera found
- camera already in use
- capture not ready
- backend request failure

## Current mock request

The manual mock test button sends this placeholder base64 value to the backend mock endpoint:

```text
abcdefghijklmnop
```

The webcam flow captures the current browser video frame, converts it to a base64 image string, and sends that payload to the same mock endpoint.

## What Phase 3 proves

This phase proves that the browser can:

- open the webcam
- show a live preview
- capture one frame
- encode it as base64
- send it to the existing mock FastAPI backend
- display the returned mock prediction

## Coming later

Real ASL model inference will be added in a later phase after the webcam capture path is validated.

Not included yet:

- real model inference
- authentication
- WebSockets
