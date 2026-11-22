"""
TrafficVision - Model Information Route
Exposes metadata about the loaded YOLO11 backbone and inference parameters.
"""

from fastapi import APIRouter, Depends
from app.schemas import ModelInfoResponse
from app.dependencies import get_model_service
from app.services.model_service import ModelService, TARGET_CLASSES
from app.config import settings


router = APIRouter(tags=["Model Info"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(model_service: ModelService = Depends(get_model_service)):
    """
    Returns model metadata, target classes, input resolution, and confidence parameters.
    """
    return ModelInfoResponse(
        model_name="YOLO11 Traffic Density & Congestion Classifier",
        backbone="Ultralytics YOLO11 (C3k2 + SPPF + C2PSA)",
        weights_path=model_service.model_path or "uninitialized",
        is_custom_4class=model_service.is_custom_4class,
        input_size=settings.IMAGE_SIZE,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        iou_threshold=settings.IOU_THRESHOLD,
        device=model_service.device,
        target_classes=TARGET_CLASSES
    )
