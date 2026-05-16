# ASL AI Platform

`asl-ai-platform` is a production-oriented full-stack application repository for serving an American Sign Language experience through a FastAPI backend and a future React frontend.

This repository is intentionally separate from the existing ASL recognition model repository. In this repo, we are building the product platform, API surface, integration boundary, and deployment structure without copying over the entire legacy codebase.

## Current scope

The repository now includes:

- FastAPI backend structure
- `GET /health`
- `POST /api/inference/mock`
- React + Vite frontend scaffold
- browser webcam capture UI
- production model loading and 180D frame preprocessing
- real 30-frame rolling ASL model inference in the backend
- runtime stabilization on top of raw real inference
- continuous frontend capture loop for real inference
- architecture and integration planning docs

Not included yet:

- authentication
- WebSockets or real-time streaming
- Docker deployment
- sentence-level translation
- TTS

## Project structure

```text
asl-ai-platform/
+-- backend/
|   +-- app/
|   |   +-- api/
|   |   +-- core/
|   |   +-- schemas/
|   |   +-- services/
|   |   `-- main.py
|   `-- requirements.txt
+-- docs/
|   +-- ARCHITECTURE.md
|   `-- INTEGRATION_PLAN.md
+-- frontend/
|   +-- src/
|   +-- package.json
|   +-- vite.config.js
|   `-- README.md
`-- README.md
```

## Backend setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

The API will then be available at `http://127.0.0.1:8000`.

## Frontend setup

From the `frontend/` directory:

```powershell
npm install
npm run dev
```

The frontend will then be available at `http://127.0.0.1:5173`.

## Example requests

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Mock inference:

```powershell
curl -X POST http://127.0.0.1:8000/api/inference/mock `
  -H "Content-Type: application/json" `
  -d "{\"image_base64\":\"aGVsbG8gd29ybGQ=\"}"
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Integration plan](docs/INTEGRATION_PLAN.md)
- [Changelog](docs/CHANGELOG.md)
- [Component anatomy](docs/COMPONENT_ANATOMY.md)
- [Frontend guide](frontend/README.md)

## Roadmap

1. Refine stabilization behavior and live-session UX.
2. Add streaming-oriented transport when needed.
3. Add containerization and environment-specific configuration.
4. Add testing, CI, and deployment workflows.
