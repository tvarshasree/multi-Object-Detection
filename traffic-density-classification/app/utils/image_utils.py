"""
TrafficVision - Image Processing and HUD Rendering Utilities
Decodes images, applies bounding box annotations, and renders high-contrast HUD overlays.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np

from app.schemas import DetectionItem, DensityClassification


# Visually distinct high-contrast colors (BGR format for OpenCV)
CLASS_COLORS = {
    "person": (255, 140, 0),       # Cyan / Light Blue
    "car": (0, 200, 0),             # Vivid Green
    "bus": (0, 140, 255),           # Amber / Orange
    "motorcycle": (0, 230, 255)     # Bright Yellow
}

DENSITY_BGR = {
    "Low": (50, 180, 50),     # Emerald Green
    "Medium": (30, 160, 240), # Golden Amber
    "High": (50, 50, 230)     # Vivid Red
}


def bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an OpenCV BGR numpy array.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image with OpenCV.")
    return image


def draw_hud_banner(
    frame: np.ndarray,
    density: DensityClassification,
    inference_ms: float
) -> np.ndarray:
    """
    Renders an informational Heads-Up Display (HUD) at the top of the annotated image.
    Displays:
    - Traffic density status pill (LOW / MEDIUM / HIGH)
    - Vehicle count (Cars + Buses + Motorcycles)
    - Pedestrian count (Persons)
    - Inference latency (ms)
    """
    h, w = frame.shape[:2]
    banner_height = 68
    overlay = frame.copy()

    # Translucent dark banner
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 20, 28), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # Congestion Status Pill
    badge_color = DENSITY_BGR.get(density.density_level, (0, 255, 0))
    cv2.rectangle(frame, (16, 12), (185, 54), badge_color, -1, cv2.LINE_AA)
    cv2.putText(
        frame, f"TRAFFIC: {density.density_level.upper()}",
        (26, 40), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )

    # Vehicles KPI
    cv2.putText(
        frame, f"Vehicles: {density.vehicle_count}",
        (210, 40), cv2.FONT_HERSHEY_DUPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA
    )

    # Pedestrians KPI (held separate!)
    cv2.putText(
        frame, f"Pedestrians: {density.pedestrian_count}",
        (380, 40), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 205, 110), 2, cv2.LINE_AA
    )

    # Latency metric on the top right
    cv2.putText(
        frame, f"{inference_ms:.1f}ms",
        (max(10, w - 120), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA
    )

    return frame


def render_annotated_image(
    image: np.ndarray,
    detections: List[DetectionItem],
    density: DensityClassification,
    inference_ms: float
) -> np.ndarray:
    """
    Draws bounding boxes, labels, confidence scores, and HUD banner.
    """
    annotated = image.copy()

    for det in detections:
        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
        cname = det.class_name
        conf = det.confidence
        color = CLASS_COLORS.get(cname, (200, 200, 200))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Class label badge
        label = f"{cname} {int(conf * 100)}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + lw + 6, max(0, y1)), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA
        )

    # Draw top HUD banner
    draw_hud_banner(annotated, density, inference_ms)
    return annotated


def save_annotated_image(annotated_img: np.ndarray, output_filename: str, output_dir: str) -> str:
    """
    Saves the annotated image to the target directory and returns the file path.
    """
    out_path = Path(output_dir) / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated_img)
    return str(out_path)
