"""
TrafficVision - Configuration Module
Loads and validates environment variables and application settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Model settings
    MODEL_PATH: str = Field(default="weights/best.pt", description="Path to fine-tuned YOLO11 weights")
    FALLBACK_MODEL_PATH: str = Field(default="yolo11n.pt", description="Pretrained fallback YOLO11 checkpoint")
    CONFIDENCE_THRESHOLD: float = Field(default=0.35, ge=0.05, le=0.95, description="Confidence score cutoff")
    IOU_THRESHOLD: float = Field(default=0.45, ge=0.1, le=0.9, description="NMS IoU threshold")
    IMAGE_SIZE: int = Field(default=640, description="Input inference image dimension")
    DEVICE: str = Field(default="cpu", description="'cpu' or '0' for CUDA GPU")

    # Traffic Density Thresholds
    LOW_THRESHOLD: int = Field(default=5, ge=1, description="Upper threshold for Low traffic (T1)")
    MEDIUM_THRESHOLD: int = Field(default=12, ge=2, description="Upper threshold for Medium traffic (T2)")

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Bind host")
    PORT: int = Field(default=8000, description="Listening port")
    MAX_UPLOAD_SIZE_MB: int = Field(default=25, description="Maximum permitted upload size in MB")
    
    # Storage paths
    OUTPUT_DIRECTORY: str = Field(default="outputs/annotated", description="Directory to store annotated images")
    REPORTS_DIRECTORY: str = Field(default="outputs/reports", description="Directory to store evaluation reports")
    FRONTEND_DIRECTORY: str = Field(default="frontend", description="Frontend static files directory")



# Instantiate global settings
settings = Settings()

# Ensure directories exist
Path(settings.OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(settings.REPORTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path("weights").mkdir(parents=True, exist_ok=True)
