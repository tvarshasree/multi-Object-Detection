"""
TrafficVision - Model Service
Manages YOLO11 lifecycle, model loading, device assignment, and inference execution.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from app.config import settings
from app.schemas import DetectionItem, BoundingBox


TARGET_CLASSES = {
    0: "person",
    1: "car",
    2: "bus",
    3: "motorcycle"
}

# COCO to Target 4-class mapping
COCO_TO_TARGET = {
    0: (0, "person"),
    2: (1, "car"),
    5: (2, "bus"),
    3: (3, "motorcycle")
}


class ModelService:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.is_custom_4class = False
        self.device = settings.DEVICE
        self._initialize_model()

    def _initialize_model(self):
        """
        Loads the fine-tuned YOLO11 model or falls back to pretrained yolo11n.pt.
        """
        weights_path = Path(settings.MODEL_PATH)
        if weights_path.exists():
            self.model_path = str(weights_path)
            print(f"[ModelService] Loading custom fine-tuned weights: {self.model_path}")
        else:
            self.model_path = settings.FALLBACK_MODEL_PATH
            print(f"[ModelService] Custom weights not found. Using checkpoint: {self.model_path}")

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            
            # Determine class schema
            num_classes = len(self.model.names)
            self.is_custom_4class = (num_classes == 4)
            print(f"[ModelService] YOLO11 successfully loaded with {num_classes} classes. Custom 4-class: {self.is_custom_4class}")
        except Exception as e:
            print(f"[ModelService] [WARNING] Failed to load model weights: {e}")
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image: np.ndarray) -> Tuple[List[DetectionItem], float]:
        """
        Executes YOLO11 inference on input OpenCV image (BGR).
        Filters detections down to the 4 target classes and returns items + latency.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized or loaded.")

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.IMAGE_SIZE,
            device=self.device,
            verbose=False
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[DetectionItem] = []
        if not results:
            return detections, round(inference_ms, 2)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections, round(inference_ms, 2)

        for box in boxes:
            raw_cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            if self.is_custom_4class:
                if raw_cls in TARGET_CLASSES:
                    detections.append(DetectionItem(
                        class_id=raw_cls,
                        class_name=TARGET_CLASSES[raw_cls],
                        confidence=round(conf, 3),
                        bbox=BoundingBox(
                            x1=round(xyxy[0], 1),
                            y1=round(xyxy[1], 1),
                            x2=round(xyxy[2], 1),
                            y2=round(xyxy[3], 1)
                        )
                    ))
            else:
                # Map COCO classes to our 4 target classes
                if raw_cls in COCO_TO_TARGET:
                    t_id, t_name = COCO_TO_TARGET[raw_cls]
                    detections.append(DetectionItem(
                        class_id=t_id,
                        class_name=t_name,
                        confidence=round(conf, 3),
                        bbox=BoundingBox(
                            x1=round(xyxy[0], 1),
                            y1=round(xyxy[1], 1),
                            x2=round(xyxy[2], 1),
                            y2=round(xyxy[3], 1)
                        )
                    ))

        return detections, round(inference_ms, 2)
