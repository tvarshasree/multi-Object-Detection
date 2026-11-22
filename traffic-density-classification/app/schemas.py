"""
TrafficVision - Pydantic Schemas
Strict API data contracts for validation, inference, counting, and classification.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate (pixels)")
    y1: float = Field(..., description="Top-left Y coordinate (pixels)")
    x2: float = Field(..., description="Bottom-right X coordinate (pixels)")
    y2: float = Field(..., description="Bottom-right Y coordinate (pixels)")


class DetectionItem(BaseModel):
    class_id: int = Field(..., description="Target class ID (0..3)")
    class_name: str = Field(..., description="Target class name (person, car, bus, motorcycle)")
    confidence: float = Field(..., description="Detection confidence score (0.00..1.00)")
    bbox: BoundingBox


class ClassCounts(BaseModel):
    person: int = Field(default=0, description="Pedestrians / persons on foot")
    car: int = Field(default=0, description="Passenger cars, taxis, sedans, SUVs")
    bus: int = Field(default=0, description="Buses and large public transport vehicles")
    motorcycle: int = Field(default=0, description="Two-wheelers, scooters, and motorcycles")


class DensityClassification(BaseModel):
    density_level: Literal["Low", "Medium", "High"]
    total_objects: int = Field(..., description="Total objects detected (vehicles + pedestrians)")
    vehicle_count: int = Field(..., description="Vehicular traffic volume (car + bus + motorcycle)")
    pedestrian_count: int = Field(..., description="Pedestrian volume (person)")
    low_threshold: int = Field(..., description="Configured T1 threshold for Low density")
    medium_threshold: int = Field(..., description="Configured T2 threshold for Medium density")
    explanation: str = Field(..., description="Human-readable explanation of the classification")
    badge_color: str = Field(..., description="Hex color code for UI badge display")


class PredictionResponse(BaseModel):
    status: str = Field(default="success")
    filename: str = Field(..., description="Sanitized uploaded filename")
    annotated_image_url: str = Field(..., description="Web-accessible URL for annotated image")
    total_processing_ms: float = Field(..., description="End-to-end request processing time in ms")
    inference_ms: float = Field(..., description="YOLO11 tensor inference time in ms")
    counts: ClassCounts
    total_objects: int
    vehicle_count: int
    pedestrian_count: int
    traffic_density: Literal["Low", "Medium", "High"]
    classification_details: DensityClassification
    detections: List[DetectionItem]


class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    model_loaded: bool
    model_path: str
    is_custom_checkpoint: bool
    device: str
    target_classes: Dict[int, str]


class ModelInfoResponse(BaseModel):
    model_name: str
    backbone: str = "Ultralytics YOLO11"
    weights_path: str
    is_custom_4class: bool
    input_size: int
    confidence_threshold: float
    iou_threshold: float
    device: str
    target_classes: Dict[int, str]
