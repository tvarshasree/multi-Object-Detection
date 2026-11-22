"""
TrafficVision - Dataset Statistics & Distribution Analysis
Computes statistical profiles, bounding box metrics, and class distributions.
Generates an exportable markdown report in outputs/reports/
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

TARGET_CLASSES = {0: "person", 1: "car", 2: "bus", 3: "motorcycle"}


def generate_dataset_report(dataset_dir: str = "dataset", output_report_dir: str = "outputs/reports") -> str:
    root = Path(dataset_dir)
    lbl_dir = root / "labels"
    out_dir = Path(output_report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    overall_counts = {cid: 0 for cid in TARGET_CLASSES}
    split_details = {}

    widths, heights, aspect_ratios = [], [], []

    for split in splits:
        s_dir = lbl_dir / split
        split_details[split] = {cid: 0 for cid in TARGET_CLASSES}
        total_annotations = 0
        img_count = len(list((root / "images" / split).glob("*.*"))) if (root / "images" / split).exists() else 0

        if s_dir.exists():
            for f in s_dir.glob("*.txt"):
                try:
                    with open(f, "r") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cid = int(parts[0])
                                w, h = float(parts[3]), float(parts[4])
                                if cid in TARGET_CLASSES:
                                    split_details[split][cid] += 1
                                    overall_counts[cid] += 1
                                    total_annotations += 1
                                    widths.append(w)
                                    heights.append(h)
                                    if h > 0:
                                        aspect_ratios.append(w / h)
                except Exception:
                    pass

        split_details[split]["total_images"] = img_count
        split_details[split]["total_annotations"] = total_annotations

    # Build Markdown Report
    report = []
    report.append("# TrafficVision: Dataset Statistical Profile\n")
    report.append(f"**Target Classes:** 4 (`person`, `car`, `bus`, `motorcycle`)\n")
    report.append("## Split Summary\n")
    report.append("| Split | Images | Annotations | Person | Car | Bus | Motorcycle |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for split in splits:
        d = split_details[split]
        report.append(f"| **{split}** | {d['total_images']} | {d['total_annotations']} | {d[0]} | {d[1]} | {d[2]} | {d[3]} |")

    report.append("\n## Aggregate Class Distribution\n")
    total_annos = sum(overall_counts.values())
    for cid, cname in TARGET_CLASSES.items():
        cnt = overall_counts[cid]
        pct = (cnt / total_annos * 100) if total_annos > 0 else 0.0
        report.append(f"- **Class {cid} ({cname})**: {cnt} instances ({pct:.1f}%)")

    if widths:
        report.append("\n## Bounding Box Spatial Statistics\n")
        report.append(f"- **Mean Width (normalized)**: {np.mean(widths):.3f}")
        report.append(f"- **Mean Height (normalized)**: {np.mean(heights):.3f}")
        report.append(f"- **Mean Aspect Ratio (W/H)**: {np.mean(aspect_ratios):.3f}")

    report_str = "\n".join(report)
    out_file = out_dir / "dataset_summary.md"
    with open(out_file, "w") as f:
        f.write(report_str)

    print(report_str)
    print(f"\n[Report saved to: {out_file}]")
    return str(out_file)


if __name__ == "__main__":
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    generate_dataset_report(ds_dir)
