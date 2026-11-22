"""
TrafficVision - Unit Tests for Dataset Parsing & Validation
Tests syntax checking, class ID verification, and coordinate normalization boundaries.
"""

import pytest
from pathlib import Path
from scripts.validate_dataset import TARGET_CLASSES


def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError("Expected 5 tokens per annotation line")
    cls_id = int(parts[0])
    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    
    if cls_id not in TARGET_CLASSES:
        raise ValueError(f"Invalid class ID: {cls_id}")
    
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        raise ValueError(f"Bounding box coordinates out of bounds: ({xc}, {yc}, {w}, {h})")
    
    return cls_id, xc, yc, w, h


def test_valid_annotation_line():
    line = "1 0.45 0.60 0.25 0.30"
    cls_id, xc, yc, w, h = parse_yolo_line(line)
    assert cls_id == 1  # Car
    assert xc == 0.45
    assert yc == 0.60
    assert w == 0.25
    assert h == 0.30


def test_invalid_token_count():
    line = "1 0.45 0.60 0.25"  # Only 4 tokens
    with pytest.raises(ValueError, match="Expected 5 tokens"):
        parse_yolo_line(line)


def test_invalid_class_id():
    line = "4 0.5 0.5 0.2 0.2"  # Class ID 4 not in 0..3
    with pytest.raises(ValueError, match="Invalid class ID: 4"):
        parse_yolo_line(line)


def test_out_of_bounds_coordinate():
    line = "0 1.25 0.50 0.20 0.20"  # xc > 1.0
    with pytest.raises(ValueError, match="out of bounds"):
        parse_yolo_line(line)


def test_zero_dimension():
    line = "2 0.50 0.50 0.00 0.20"  # width == 0
    with pytest.raises(ValueError, match="out of bounds"):
        parse_yolo_line(line)
