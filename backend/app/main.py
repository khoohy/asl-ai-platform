from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.inference import router as inference_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-oriented backend scaffold for serving ASL inference APIs. "
        "Phase 1 exposes only health and mock inference routes."
    ),
)

app.include_router(health_router)
app.include_router(inference_router, prefix="/api/inference", tags=["inference"])


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "message": f"{settings.app_name} backend is running.",
        "docs_url": "/docs",
    }
