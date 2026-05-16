# Frontend Plan

The React frontend is intentionally not implemented in Phase 1.

## Planned dashboard

This repository will eventually include a webcam-first dashboard that lets a user:

- open a live camera preview in the browser
- capture or stream frames for ASL prediction
- view the latest prediction, confidence, and system health
- switch between mock inference and real backend-powered model inference
- inspect request and response timing for debugging

## Planned frontend responsibilities

- React application shell and routing
- webcam access and frame capture flow
- upload and preview support for still images
- API integration with the FastAPI backend
- status panels for backend health and inference results

## Not included yet

- production React code
- authentication
- real-time streaming or WebSocket support
- design system implementation
