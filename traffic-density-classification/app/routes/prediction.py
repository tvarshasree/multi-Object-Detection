"""
TrafficVision - Prediction & Media Retrieval Routes
Handles image upload, YOLO11 inference, counting, classification,
and safe retrieval of annotated images.
"""

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse

from app.schemas import PredictionResponse
from app.dependencies import get_detection_service
from app.services.detection_service import DetectionService
from app.utils.validation import validate_image_upload
from app.config import settings


router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_traffic_image(
    file: UploadFile = File(..., description="Traffic image file (JPG, PNG, WebP)"),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """
    Processes an uploaded traffic image:
    1. Validates file format and size
    2. Runs YOLO11 object detection on 4 target classes
    3. Counts vehicles (cars, buses, motorcycles) and pedestrians (persons) separately
    4. Classifies traffic density into Low, Medium, or High
    5. Returns annotated image URL and structured JSON metrics
    """
    image_bytes = await validate_image_upload(file)

    try:
        result = detection_service.process_image(
            image_bytes=image_bytes,
            original_filename=file.filename
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failure: {str(e)}")


@router.get("/results/{filename}")
async def get_annotated_image(filename: str):
    """
    Retrieves the generated annotated output image.
    Enforces path safety and prevents directory traversal.
    """
    safe_filename = Path(filename).name
    file_path = Path(settings.OUTPUT_DIRECTORY) / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Requested annotated image not found.")

    return FileResponse(path=str(file_path), media_type="image/jpeg")
