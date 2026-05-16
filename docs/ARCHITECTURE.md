# Architecture Overview

## Purpose

`asl-ai-platform` is the full-stack product repository for the future ASL application experience. It is separate from the existing ASL recognition model repository on purpose.

This repo will own:

- the FastAPI application surface
- frontend application code
- deployment-oriented configuration
- integration logic that prepares inputs and formats outputs for the product

The existing ASL model repo will continue to own:

- model training code
- experiment tracking
- evaluation workflows
- model weights and research-oriented assets

## Separation of concerns

Phase 1 keeps the product platform and model repository loosely coupled.

Instead of copying the old repository into this one, the platform will eventually consume the model through a controlled integration boundary such as:

- a packaged Python dependency published from the model repo
- a versioned model artifact exported by the model repo
- a model-serving interface shared across repositories

This keeps deployment cleaner and avoids mixing research code with production API code.

## High-level flow

1. The React frontend captures a webcam frame or uploaded image.
2. The frontend sends a request to the FastAPI backend.
3. The backend validates the payload and calls an inference service.
4. The inference service will later delegate to the ASL model integration layer.
5. The backend returns a structured prediction response to the frontend.

## Phase 1 backend shape

- `app/main.py`: FastAPI entrypoint
- `app/api/health.py`: service health route
- `app/api/inference.py`: mock inference route
- `app/core/config.py`: settings and environment configuration
- `app/schemas/inference.py`: request and response schemas
- `app/services/inference_service.py`: mock inference logic and future model boundary

## Future deployment direction

The likely production topology is:

- React frontend deployed independently
- FastAPI backend deployed as an API service
- model artifact or model package versioned separately from the app repo
- optional object storage or model registry for released model assets

That structure will let the platform evolve without forcing every frontend or API change to live inside the model training repository.
