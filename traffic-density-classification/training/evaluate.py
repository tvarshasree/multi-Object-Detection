"""
TrafficVision - Evaluation & Benchmark Reproducibility Script
Implements Section 13 of Master Specification:
- Reports Precision, Recall, and mAP@0.5
- Validates against the student project report target metrics (Table 4.1)
- Generates a formal reproducibility report in outputs/reports/evaluation_report.md
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any


# Documented targets from Chapter 4 (Table 4.1) of the Student Project Report
REPORTED_TARGET_METRICS = {
    "Car": {"precision": 80.5, "recall": 81.0, "map50": 0.81},
    "Bus": {"precision": 81.2, "recall": 82.5, "map50": 0.78},
    "Motorcycle": {"precision": 79.8, "recall": 78.5, "map50": 0.73},
    "Person": {"precision": 78.5, "recall": 78.2, "map50": 0.69},
    "Overall": {"precision": 81.8, "recall": 80.05, "map50": 0.75}
}


def print_target_benchmark_table():
    print("=" * 72)
    print("  STUDENT PROJECT REPORT BENCHMARK TARGETS (Table 4.1)")
    print("=" * 72)
    print(f"{'Class':<25} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'mAP@0.5':<8}")
    print("-" * 72)
    for cname, vals in REPORTED_TARGET_METRICS.items():
        if cname == "Overall":
            print("-" * 72)
        print(f"{cname:<25} | {vals['precision']:>13.1f}% | {vals['recall']:>11.1f}% | {vals['map50']:>7.2f}")
    print("=" * 72)


def evaluate_and_generate_report(
    model_path: str = "weights/best.pt",
    data_yaml: str = "dataset/data.yaml",
    output_report_path: str = "outputs/reports/evaluation_report.md"
):
    print_target_benchmark_table()

    report_lines = [
        "# TrafficVision: Evaluation and Benchmark Reproducibility Report\n",
        "## 1. Reference Targets (Report Table 4.1)\n",
        "| Object Class | Precision (%) | Recall (%) | mAP@0.5 |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for cname, vals in REPORTED_TARGET_METRICS.items():
        report_lines.append(f"| **{cname}** | {vals['precision']}% | {vals['recall']}% | {vals['map50']} |")

    report_lines.append("\n## 2. Live Evaluation Execution\n")

    if not os.path.exists(model_path):
        note = (
            f"> [!NOTE]\n"
            f"> Trained checkpoint `{model_path}` is not present in the current workspace.\n"
            f"> If you have trained weights or an exported `.pt` file, copy it to `{model_path}` to execute live validation."
        )
        print(f"\n[Notice] Checkpoint '{model_path}' not found. Generated target benchmark reference.")
        report_lines.append(note)
    else:
        try:
            from ultralytics import YOLO
            print(f"\nRunning validation with '{model_path}'...")
            model = YOLO(model_path)
            metrics = model.val(data=data_yaml, split="val", verbose=False)
            
            p = metrics.box.mp * 100
            r = metrics.box.mr * 100
            m50 = metrics.box.map50
            
            report_lines.append("| Metric | Live Evaluated | Target Benchmark | Delta |")
            report_lines.append("| :--- | :---: | :---: | :---: |")
            report_lines.append(f"| **Precision** | {p:.2f}% | 81.80% | {p - 81.8:+.2f}% |")
            report_lines.append(f"| **Recall** | {r:.2f}% | 80.05% | {r - 80.05:+.2f}% |")
            report_lines.append(f"| **mAP@0.5** | {m50:.4f} | 0.7500 | {m50 - 0.75:+.4f} |")
        except Exception as e:
            report_lines.append(f"> [!WARNING]\n> Live evaluation could not run: {e}")

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReproducibility report generated at: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficVision Benchmark Evaluation")
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--output", type=str, default="outputs/reports/evaluation_report.md")

    args = parser.parse_args()
    evaluate_and_generate_report(args.model, args.data, args.output)
