"""Meta routes: home and health check."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["meta"])


@router.get("/", summary="Home")
async def home() -> dict:
    return {
        "name": settings.app_title,
        "version": settings.app_version,
        "docs": "/docs",
    }


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "version": settings.app_version}
