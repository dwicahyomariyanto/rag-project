"""Health / info endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_service
from app.schemas.models import HealthOut
from app.service import RAGService

router = APIRouter(prefix="/api", tags=["health"])


@router.get(
    "/health",
    response_model=HealthOut,
    summary="Status aplikasi, jumlah chunk, dan backend yang aktif",
)
def health(service: RAGService = Depends(get_service)) -> HealthOut:
    return HealthOut(**service.health_snapshot())
