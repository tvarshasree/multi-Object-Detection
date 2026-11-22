"""
TrafficVision - VisDrone Evaluation and Metrics Calculator
Computes:
- Per-class and aggregate Precision
- Per-class and aggregate Recall
- Per-class and aggregate mAP@0.5
- Mean inference and processing latency
- Saves a formal Markdown evaluation report to outputs/reports/visdrone_evaluation.md
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

# Isolate Ultralytics & Matplotlib cache
ultralytics_dir = workspace_root / ".ultralytics"
ultralytics_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_dir)
os.environ["MPLCONFIGDIR"] = str(ultralytics_dir)

from app.dependencies import get_detection_service
from app.services.model_service import TARGET_CLASSES


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes AP using all-point interpolation (COCO standard)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate_dataset(
    dataset_root: Path,
    max_images: int = 150,
    iou_thresh: float = 0.50,
    output_report: Path = Path("outputs/reports/visdrone_evaluation.md")
):
    print("=" * 75)
    print("  TRAFFICVISION - VISDRONE EVALUATION & BENCHMARK")
    print("=" * 75)
    print(f"Dataset:            {dataset_root}")
    print(f"Max Test Images:    {max_images or 'All'}")
    print(f"Evaluation IoU:     {iou_thresh}")
    print("-" * 75)

    images_dir = dataset_root / "images" / "val"
    labels_dir = dataset_root / "labels" / "val"

    image_files = sorted(list(images_dir.glob("*.jpg")))
    if not image_files:
        print("[ERROR] No validation images found in", images_dir)
        sys.exit(1)

    if max_images:
        image_files = image_files[:max_images]

    svc = get_detection_service()

    # Data structures for metrics: class_id -> list of (conf, is_tp)
    class_preds: Dict[int, List[Tuple[float, bool]]] = {c: [] for c in TARGET_CLASSES}
    class_gt_counts: Dict[int, int] = {c: 0 for c in TARGET_CLASSES}

    latencies = []
    print(f"Evaluating {len(image_files)} surveillance images...")

    for idx, img_path in enumerate(image_files, 1):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Load ground truth
        gt_boxes_by_class: Dict[int, List[List[float]]] = {c: [] for c in TARGET_CLASSES}
        
        with Image.open(img_path) as im:
            im_w, im_h = im.size

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        if cid in TARGET_CLASSES:
                            xc, yc, w, h = map(float, parts[1:5])
                            x1 = (xc - w / 2.0) * im_w
                            y1 = (yc - h / 2.0) * im_h
                            x2 = (xc + w / 2.0) * im_w
                            y2 = (yc + h / 2.0) * im_h
                            gt_boxes_by_class[cid].append([x1, y1, x2, y2])
                            class_gt_counts[cid] += 1

        # Run inference
        with open(img_path, "rb") as bf:
            raw_bytes = bf.read()

        t0 = time.time()
        result = svc.process_image(raw_bytes, img_path.name)
        latencies.append((time.time() - t0) * 1000.0)

        # Match predictions to GT per class
        pred_boxes_by_class: Dict[int, List[Tuple[float, List[float]]]] = {c: [] for c in TARGET_CLASSES}
        for det in result.detections:
            cid = det.class_id
            if cid in TARGET_CLASSES:
                b = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                pred_boxes_by_class[cid].append((det.confidence, b))

        for cid in TARGET_CLASSES:
            preds = sorted(pred_boxes_by_class[cid], key=lambda x: x[0], reverse=True)
            gts = gt_boxes_by_class[cid]
            matched_gt = set()

            for conf, pbox in preds:
                best_iou = 0.0
                best_gt_idx = -1

                for gidx, gbox in enumerate(gts):
                    if gidx in matched_gt:
                        continue
                    iou = compute_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gidx

                if best_iou >= iou_thresh and best_gt_idx != -1:
                    class_preds[cid].append((conf, True))
                    matched_gt.add(best_gt_idx)
                else:
                    class_preds[cid].append((conf, False))

        if idx % 25 == 0 or idx == len(image_files):
            print(f"      Processed {idx}/{len(image_files)} images (avg {np.mean(latencies):.1f} ms/frame)...")

    # Compute metrics per class
    results_table = []
    ap_list = []
    prec_list = []
    rec_list = []

    print("\n" + "=" * 75)
    print("  VISDRONE EVALUATION METRICS (IoU >= 0.50)")
    print("=" * 75)
    print(f"{'Class Name':<16} | {'Images':<8} | {'GT Targets':<12} | {'Precision':<10} | {'Recall':<8} | {'mAP@0.5'}")
    print("-" * 75)

    for cid, cname in TARGET_CLASSES.items():
        preds = class_preds[cid]
        num_gt = class_gt_counts[cid]

        if not preds or num_gt == 0:
            p, r, ap = 0.0, 0.0, 0.0
        else:
            preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
            tps = np.array([1 if x[1] else 0 for x in preds_sorted])
            fps = np.array([0 if x[1] else 1 for x in preds_sorted])

            cum_tps = np.cumsum(tps)
            cum_fps = np.cumsum(fps)

            recalls = cum_tps / float(num_gt)
            precisions = cum_tps / (cum_tps + cum_fps)

            p = float(precisions[-1]) if len(precisions) > 0 else 0.0
            r = float(recalls[-1]) if len(recalls) > 0 else 0.0
            ap = compute_ap(recalls, precisions)

        prec_list.append(p)
        rec_list.append(r)
        ap_list.append(ap)

        print(f"{cname.capitalize():<16} | {len(image_files):<8} | {num_gt:<12} | {p*100:>8.2f}% | {r*100:>6.2f}% | {ap:>7.4f}")
        results_table.append((cname, len(image_files), num_gt, p, r, ap))

    mean_p = float(np.mean(prec_list))
    mean_r = float(np.mean(rec_list))
    map50 = float(np.mean(ap_list))
    mean_latency = float(np.mean(latencies))

    print("-" * 75)
    print(f"{'Aggregate Mean':<16} | {len(image_files):<8} | {sum(class_gt_counts.values()):<12} | {mean_p*100:>8.2f}% | {mean_r*100:>6.2f}% | {map50:>7.4f}")
    print("=" * 75)
    print(f"Mean Pipeline Latency: {mean_latency:.1f} ms / frame ({1000.0 / mean_latency:.1f} FPS)")

    # Generate Markdown Report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    report_content = [
        "# VisDrone2019-DET Evaluation Report\n",
        f"- **Evaluated Checkpoint**: `yolo11n.pt` with COCO-to-Target Dynamic Class Translation",
        f"- **Evaluation Dataset**: VisDrone2019-DET Validation Split ({len(image_files)} frames)",
        f"- **Matching IoU Cutoff**: {iou_thresh}",
        f"- **Mean Processing Latency**: {mean_latency:.1f} ms / frame\n",
        "## Performance Metrics\n",
        "| Class Name | Test Images | Ground Truth | Precision | Recall | mAP@0.5 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for cname, n_imgs, gt_cnt, p, r, ap in results_table:
        report_content.append(f"| **{cname.capitalize()}** | {n_imgs} | {gt_cnt} | {p*100:.2f}% | {r*100:.2f}% | {ap:.4f} |")

    report_content.append(f"| **Overall / Mean** | **{len(image_files)}** | **{sum(class_gt_counts.values())}** | **{mean_p*100:.2f}%** | **{mean_r*100:.2f}%** | **{map50:.4f}** |")

    with open(output_report, "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_content))

    print(f"\n[SUCCESS] Report successfully saved to: {output_report}")
    return results_table, mean_p, mean_r, map50


if __name__ == "__main__":
    from PIL import Image

    parser = argparse.ArgumentParser(description="Evaluate TrafficVision on VisDrone Dataset")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset root path")
    parser.add_argument("--max-images", type=int, default=100, help="Number of images to evaluate (default: 100)")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU cutoff for evaluation")
    parser.add_argument("--output", type=str, default="outputs/reports/visdrone_evaluation.md", help="Output report path")

    args = parser.parse_args()
    evaluate_dataset(
        dataset_root=Path(args.dataset),
        max_images=args.max_images,
        iou_thresh=args.iou,
        output_report=Path(args.output)
    )
