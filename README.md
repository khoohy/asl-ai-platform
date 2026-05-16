# ASL AI Platform

`asl-ai-platform` is a production-oriented full-stack application repository for serving an American Sign Language experience through a FastAPI backend and a future React frontend.

This repository is intentionally separate from the existing ASL recognition model repository. In this repo, we are building the product platform, API surface, integration boundary, and deployment structure without copying over the entire legacy codebase.

## Phase 1 scope

Phase 1 focuses on scaffolding and documentation only.

Included now:

- FastAPI backend structure
- `GET /health`
- `POST /api/inference/mock`
- architecture and integration planning docs
- frontend planning notes

Not included yet:

- real ASL model inference
- authentication
- WebSockets or real-time streaming
- full React implementation
- large model artifacts

## Project structure

```text
asl-ai-platform/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   └── requirements.txt
├── docs/
│   ├── ARCHITECTURE.md
│   └── INTEGRATION_PLAN.md
├── frontend/
│   └── README.md
└── README.md
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
- [Frontend plan](frontend/README.md)

## Roadmap

1. Add real model integration behind the backend service layer.
2. Build the React webcam dashboard.
3. Add containerization and environment-specific configuration.
4. Add testing, CI, and deployment workflows.
