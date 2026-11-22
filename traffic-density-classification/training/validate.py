"""
TrafficVision - Validation Script
Evaluates model performance on the held-out validation split.
Computes Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
"""

import os
import sys
import argparse
from pathlib import Path


def run_validation(
    model_path: str = "weights/best.pt",
    data_yaml: str = "dataset/data.yaml",
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.35,
    iou: float = 0.45,
    device: str = "cpu"
):
    print("=" * 70)
    print("  TRAFFICVISION - MODEL VALIDATION")
    print("=" * 70)
    print(f"Checkpoint:  {model_path}")
    print(f"Dataset:     {data_yaml}")
    print(f"Parameters:  Conf={conf}, IoU={iou}, ImgSz={imgsz}, Device={device}")
    print("-" * 70)

    if not os.path.exists(model_path):
        print(f"[ERROR] Weights file '{model_path}' does not exist.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
        save_json=True
    )

    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Precision (mp):    {metrics.box.mp * 100:.2f}%")
    print(f"Overall Recall (mr):       {metrics.box.mr * 100:.2f}%")
    print(f"Overall mAP@0.5:           {metrics.box.map50:.4f}")
    print(f"Overall mAP@0.5:0.95:      {metrics.box.map:.4f}")
    print("=" * 70)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficVision Model Validation")
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    run_validation(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device
    )
