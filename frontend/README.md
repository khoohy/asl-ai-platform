# Frontend

Phase 2 adds a React + Vite frontend scaffold that can talk to the existing FastAPI mock backend.

## Features in this phase

- dashboard shell for `ASL AI Platform`
- backend health status panel
- mock inference test panel
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

## Current mock request

The frontend currently sends this placeholder base64 value to the backend mock endpoint:

```text
abcdefghijklmnop
```

## Coming in Phase 3

Phase 3 will introduce real webcam integration and begin connecting the frontend flow to actual ASL model inference.

Not included yet:

- webcam access
- real model inference
- authentication
- WebSockets
