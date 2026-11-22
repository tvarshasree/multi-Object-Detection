"""
TrafficVision - Application Dependency Injection
Provides singletons of the core services for FastAPI route injection.
"""

from app.services.model_service import ModelService
from app.services.counting_service import CountingService
from app.services.classification_service import ClassificationService
from app.services.detection_service import DetectionService

# Singleton instances initialized once at startup
_model_service = None
_counting_service = CountingService()
_classification_service = ClassificationService()
_detection_service = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_counting_service() -> CountingService:
    return _counting_service


def get_classification_service() -> ClassificationService:
    return _classification_service


def get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService(
            model_service=get_model_service(),
            counting_service=get_counting_service(),
            classification_service=get_classification_service()
        )
    return _detection_service
