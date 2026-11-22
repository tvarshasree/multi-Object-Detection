"""
TrafficVision - Health Check Route
Provides server and model readiness diagnostics.
"""

from fastapi import APIRouter, Depends
from app.schemas import HealthResponse
from app.dependencies import get_model_service
from app.services.model_service import ModelService, TARGET_CLASSES


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def check_health(model_service: ModelService = Depends(get_model_service)):
    """
    Returns server operational status, model loading state, and target classes.
    """
    is_loaded = model_service.is_loaded()
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_path=model_service.model_path or "uninitialized",
        is_custom_checkpoint=model_service.is_custom_4class,
        device=model_service.device,
        target_classes=TARGET_CLASSES
    )
