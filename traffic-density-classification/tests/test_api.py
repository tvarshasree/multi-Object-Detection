"""
TrafficVision - Integration Tests for REST API
Tests /health, /model-info, /predict input validation, and media serving.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.dependencies import get_model_service, get_detection_service
from app.services.model_service import ModelService
from app.services.detection_service import DetectionService
from app.schemas import DetectionItem, BoundingBox


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "target_classes" in data
    assert data["target_classes"]["0"] == "person"
    assert data["target_classes"]["1"] == "car"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["backbone"] == "Ultralytics YOLO11 (C3k2 + SPPF + C2PSA)"
    assert data["input_size"] == 640
    assert "confidence_threshold" in data


def test_predict_invalid_extension():
    # Uploading a text file should yield 400
    file_content = b"this is not an image"
    response = client.post(
        "/predict",
        files={"file": ("test.txt", file_content, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_predict_empty_file():
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_results_not_found():
    response = client.get("/results/non_existent_image_12345.jpg")
    assert response.status_code == 404


def test_results_path_traversal_prevention():
    response = client.get("/results/../../etc/passwd")
    assert response.status_code in (404, 400)


def create_mock_traffic_image():
    # Helper to generate a dummy valid JPEG in memory
    img = Image.new("RGB", (640, 480), color=(70, 70, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_pipeline_with_mocked_model():
    """
    Tests end-to-end route handling with mocked detections.
    """
    class MockModelService:
        model_path = "mock_model.pt"
        is_custom_4class = True
        device = "cpu"
        def is_loaded(self): return True
        def predict(self, img):
            # Return 2 cars and 1 person
            return [
                DetectionItem(class_id=1, class_name="car", confidence=0.92, bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150)),
                DetectionItem(class_id=1, class_name="car", confidence=0.88, bbox=BoundingBox(x1=160, y1=60, x2=240, y2=140)),
                DetectionItem(class_id=0, class_name="person", confidence=0.85, bbox=BoundingBox(x1=20, y1=200, x2=40, y2=280))
            ], 14.5

    mock_model = MockModelService()
    mock_detection_service = DetectionService(model_service=mock_model)

    app.dependency_overrides[get_model_service] = lambda: mock_model
    app.dependency_overrides[get_detection_service] = lambda: mock_detection_service

    try:
        raw_image = create_mock_traffic_image()
        response = client.post(
            "/predict",
            files={"file": ("test_road.jpg", raw_image, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_objects"] == 3
        assert data["vehicle_count"] == 2
        assert data["pedestrian_count"] == 1
        assert data["counts"]["car"] == 2
        assert data["counts"]["person"] == 1
        assert data["traffic_density"] == "Low"
        assert "annotated_image_url" in data
    finally:
        app.dependency_overrides.clear()
