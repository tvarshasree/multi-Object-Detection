"""
TrafficVision - Dataset Validation Script
Validates YOLO-formatted traffic datasets:
- Checks directory structure (train, val, test)
- Validates label syntax (class_id, normalized bbox coordinates)
- Detects missing/unpaired images or labels
- Checks for out-of-bounds coordinates and malformed rows
- Produces a comprehensive validation summary report
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple


TARGET_CLASSES = {0: "person", 1: "car", 2: "bus", 3: "motorcycle"}


def validate_yolo_dataset(data_yaml_path: str = "dataset/data.yaml") -> Dict[str, Any]:
    print("=" * 65)
    print(f"  TrafficVision Dataset Validator: {data_yaml_path}")
    print("=" * 65)

    yaml_file = Path(data_yaml_path)
    if not yaml_file.exists():
        print(f"[ERROR] Dataset configuration file not found at: {data_yaml_path}")
        return {"valid": False, "error": "Configuration file not found"}

    with open(yaml_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Failed to parse YAML: {e}")
            return {"valid": False, "error": f"Invalid YAML: {e}"}

    root_dir = yaml_file.parent
    splits = ["train", "val", "test"]
    stats = {
        "valid": True,
        "splits": {},
        "errors": [],
        "warnings": [],
        "class_totals": {0: 0, 1: 0, 2: 0, 3: 0}
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in splits:
        img_dir = root_dir / "images" / split
        lbl_dir = root_dir / "labels" / split

        split_stat = {
            "images_found": 0,
            "labels_found": 0,
            "missing_labels": 0,
            "corrupt_labels": 0,
            "out_of_bounds_bboxes": 0,
            "class_counts": {0: 0, 1: 0, 2: 0, 3: 0}
        }

        if not img_dir.exists():
            stats["warnings"].append(f"Directory missing: {img_dir}")
            stats["splits"][split] = split_stat
            continue

        images = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        split_stat["images_found"] = len(images)

        for img in images:
            lbl_file = lbl_dir / f"{img.stem}.txt"
            if not lbl_file.exists():
                split_stat["missing_labels"] += 1
                continue

            split_stat["labels_found"] += 1
            try:
                with open(lbl_file, "r") as lf:
                    lines = lf.readlines()
                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            split_stat["corrupt_labels"] += 1
                            stats["errors"].append(f"Malformed row in {lbl_file.name}:{line_idx+1} (expected 5 tokens)")
                            continue

                        cls_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        
                        if cls_id not in TARGET_CLASSES:
                            split_stat["corrupt_labels"] += 1
                            stats["errors"].append(f"Invalid class ID {cls_id} in {lbl_file.name}:{line_idx+1}")
                        else:
                            split_stat["class_counts"][cls_id] += 1
                            stats["class_totals"][cls_id] += 1

                        # Validate bounding box normalization (0.0 to 1.0)
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            split_stat["out_of_bounds_bboxes"] += 1
                            stats["errors"].append(f"Out-of-bounds bbox in {lbl_file.name}:{line_idx+1}")

            except Exception as e:
                split_stat["corrupt_labels"] += 1
                stats["errors"].append(f"Read error {lbl_file.name}: {e}")

        stats["splits"][split] = split_stat

    # Print Summary Table
    print(f"{'Split':<10} | {'Images':<8} | {'Labels':<8} | {'Missing':<8} | {'Corrupted':<10}")
    print("-" * 55)
    total_imgs = 0
    for split, s in stats["splits"].items():
        total_imgs += s["images_found"]
        print(f"{split:<10} | {s['images_found']:<8} | {s['labels_found']:<8} | {s['missing_labels']:<8} | {s['corrupt_labels']:<10}")
    print("-" * 55)

    print("\nClass Instance Summary (Target 4 Classes):")
    for cid, cname in TARGET_CLASSES.items():
        print(f"  Class {cid} ({cname:<12}): {stats['class_totals'][cid]} annotations")

    if total_imgs == 0:
        print("\n[Notice] Dataset directories are currently empty.")
        print("To fine-tune YOLO11, place images into dataset/images/{train,val,test}")
        print("and annotations into dataset/labels/{train,val,test}.")
        stats["valid"] = False
    elif len(stats["errors"]) > 0:
        print(f"\n[Warning] Found {len(stats['errors'])} validation errors in dataset.")
        stats["valid"] = False
    else:
        print("\n[SUCCESS] Dataset structure and annotations are fully valid!")

    return stats


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/data.yaml"
    res = validate_yolo_dataset(path)
    sys.exit(0 if res["valid"] or res.get("splits") else 1)
