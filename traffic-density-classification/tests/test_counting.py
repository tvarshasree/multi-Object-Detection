"""
TrafficVision - Unit Tests for Object Counting
Tests per-class counting, aggregation, and isolation of pedestrian volume.
"""

import pytest
from app.schemas import DetectionItem, BoundingBox
from app.services.counting_service import CountingService


def test_empty_detection_list():
    counts = CountingService.count_objects([])
    assert counts.person == 0
    assert counts.car == 0
    assert counts.bus == 0
    assert counts.motorcycle == 0

    veh, ped, total = CountingService.compute_aggregates(counts)
    assert veh == 0
    assert ped == 0
    assert total == 0


def test_single_detection():
    detections = [
        DetectionItem(
            class_id=1,
            class_name="car",
            confidence=0.88,
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150)
        )
    ]
    counts = CountingService.count_objects(detections)
    assert counts.car == 1
    assert counts.person == 0

    veh, ped, total = CountingService.compute_aggregates(counts)
    assert veh == 1
    assert ped == 0
    assert total == 1


def test_multiple_classes_and_aggregates():
    detections = [
        # 3 Cars
        DetectionItem(class_id=1, class_name="car", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        DetectionItem(class_id=1, class_name="car", confidence=0.85, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        DetectionItem(class_id=1, class_name="car", confidence=0.82, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        # 2 Buses
        DetectionItem(class_id=2, class_name="bus", confidence=0.91, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        DetectionItem(class_id=2, class_name="bus", confidence=0.89, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        # 1 Motorcycle
        DetectionItem(class_id=3, class_name="motorcycle", confidence=0.78, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        # 4 Persons (Pedestrians)
        DetectionItem(class_id=0, class_name="person", confidence=0.8, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        DetectionItem(class_id=0, class_name="person", confidence=0.79, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        DetectionItem(class_id=0, class_name="person", confidence=0.84, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        DetectionItem(class_id=0, class_name="person", confidence=0.81, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
    ]

    counts = CountingService.count_objects(detections)
    assert counts.car == 3
    assert counts.bus == 2
    assert counts.motorcycle == 1
    assert counts.person == 4

    veh, ped, total = CountingService.compute_aggregates(counts)
    assert veh == 6          # 3 + 2 + 1
    assert ped == 4          # 4 pedestrians
    assert total == 10       # 6 + 4
