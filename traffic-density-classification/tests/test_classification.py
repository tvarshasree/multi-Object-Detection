"""
TrafficVision - Unit Tests for Traffic Density Classification
Tests Low/Medium/High threshold boundaries and pedestrian segregation.
"""

import pytest
from app.services.classification_service import ClassificationService


def test_low_density_boundary():
    # T1 = 5, T2 = 12
    # At exact boundary 5: should be Low
    res = ClassificationService.classify_traffic_density(
        vehicle_count=5,
        pedestrian_count=2,
        total_objects=7,
        low_thresh=5,
        med_thresh=12
    )
    assert res.density_level == "Low"
    assert res.vehicle_count == 5
    assert "Low traffic density" in res.explanation


def test_medium_density_boundary():
    # 6: should be Medium
    res1 = ClassificationService.classify_traffic_density(
        vehicle_count=6,
        pedestrian_count=1,
        total_objects=7,
        low_thresh=5,
        med_thresh=12
    )
    assert res1.density_level == "Medium"

    # Exact upper boundary 12: should be Medium
    res2 = ClassificationService.classify_traffic_density(
        vehicle_count=12,
        pedestrian_count=0,
        total_objects=12,
        low_thresh=5,
        med_thresh=12
    )
    assert res2.density_level == "Medium"


def test_high_density_boundary():
    # 13: strictly greater than 12 -> High
    res = ClassificationService.classify_traffic_density(
        vehicle_count=13,
        pedestrian_count=3,
        total_objects=16,
        low_thresh=5,
        med_thresh=12
    )
    assert res.density_level == "High"
    assert "High traffic congestion" in res.explanation


def test_empty_scene():
    res = ClassificationService.classify_traffic_density(
        vehicle_count=0,
        pedestrian_count=0,
        total_objects=0,
        low_thresh=5,
        med_thresh=12
    )
    assert res.density_level == "Low"
    assert res.vehicle_count == 0


def test_pedestrian_isolation_prevents_false_congestion():
    # 40 pedestrians crossing street, but only 2 cars:
    # System MUST classify as Low, NOT High!
    res = ClassificationService.classify_traffic_density(
        vehicle_count=2,
        pedestrian_count=40,
        total_objects=42,
        low_thresh=5,
        med_thresh=12
    )
    assert res.density_level == "Low"
    assert res.pedestrian_count == 40
    assert "40 pedestrians tracked separately" in res.explanation
