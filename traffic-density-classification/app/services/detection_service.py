"""
TrafficVision - End-to-End Detection Pipeline Service
Orchestrates image decoding, YOLO11 inference, counting, density classification,
and annotated image rendering.
"""

import time
from typing import Tuple
from app.config import settings
from app.schemas import PredictionResponse
from app.services.model_service import ModelService
from app.services.counting_service import CountingService
from app.services.classification_service import ClassificationService
from app.utils.image_utils import bytes_to_cv2_image, render_annotated_image, save_annotated_image
from app.utils.validation import sanitize_filename


class DetectionService:
    def __init__(
        self,
        model_service: ModelService,
        counting_service: CountingService = None,
        classification_service: ClassificationService = None
    ):
        self.model_service = model_service
        self.counting_service = counting_service or CountingService()
        self.classification_service = classification_service or ClassificationService()

    def process_image(self, image_bytes: bytes, original_filename: str) -> PredictionResponse:
        t_start = time.perf_counter()

        # 1. Decode Image
        cv2_image = bytes_to_cv2_image(image_bytes)

        # 2. Run YOLO11 Inference
        detections, inference_ms = self.model_service.predict(cv2_image)

        # 3. Object Counting
        class_counts = self.counting_service.count_objects(detections)
        veh_count, ped_count, total_objects = self.counting_service.compute_aggregates(class_counts)

        # 4. Traffic Density Classification
        density_details = self.classification_service.classify_traffic_density(
            vehicle_count=veh_count,
            pedestrian_count=ped_count,
            total_objects=total_objects
        )

        # 5. Render Annotated Image with HUD
        annotated_img = render_annotated_image(
            image=cv2_image,
            detections=detections,
            density=density_details,
            inference_ms=inference_ms
        )

        # 6. Save Output Image
        out_filename = sanitize_filename(original_filename)
        save_annotated_image(annotated_img, out_filename, settings.OUTPUT_DIRECTORY)

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return PredictionResponse(
            status="success",
            filename=out_filename,
            annotated_image_url=f"/results/{out_filename}",
            total_processing_ms=round(total_ms, 2),
            inference_ms=inference_ms,
            counts=class_counts,
            total_objects=total_objects,
            vehicle_count=veh_count,
            pedestrian_count=ped_count,
            traffic_density=density_details.density_level,
            classification_details=density_details,
            detections=detections
        )
