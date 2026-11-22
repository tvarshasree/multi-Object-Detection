"""
TrafficVision - YOLO11 Model Training Pipeline
Implements Section 7 of the Master Specification:
- Transfer learning using YOLO11 detection backbone
- Configurable hyperparameters (epochs, batch, learning rate, optimizer, device)
- Domain-appropriate traffic data augmentation
- Automatic saving of best weights to weights/best.pt
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def run_training(
    data_yaml: str = "dataset/data.yaml",
    base_model: str = "yolo11n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    optimizer: str = "auto",
    weight_decay: float = 0.0005,
    device: str = "cpu",
    workers: int = 2,
    seed: int = 42,
    patience: int = 15,
    project: str = "runs/train",
    name: str = "yolo11_traffic",
    save_dir: str = "weights"
):
    print("=" * 70)
    print("  TRAFFICVISION - YOLO11 MODEL TRAINING")
    print("=" * 70)
    print(f"Data configuration: {data_yaml}")
    print(f"Base backbone:      {base_model}")
    print(f"Hyperparameters:    Epochs={epochs}, Batch={batch_size}, ImgSz={imgsz}, LR0={lr0}")
    print(f"Execution Device:   {device} (workers={workers}, seed={seed})")
    print("-" * 70)

    # Validate data yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config '{data_yaml}' not found.")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"[1/4] Loading base pretrained checkpoint '{base_model}'...")
    model = YOLO(base_model)

    print("[2/4] Initializing fine-tuning with traffic-specific data augmentations...")
    # Safe augmentations preserving semantic traffic flow meaning
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        optimizer=optimizer,
        weight_decay=weight_decay,
        device=device,
        workers=workers,
        seed=seed,
        patience=patience,
        project=project,
        name=name,
        save=True,
        plots=True,
        verbose=True,
        # Augmentations
        fliplr=0.5,        # Horizontal flipping (valid on symmetrical road views)
        flipud=0.0,        # NEVER flip upside down (cars do not drive upside down)
        mosaic=1.0,        # Mosaic for multi-scale vehicle context
        scale=0.5,         # Scale jittering for far/near vehicle sizes
        translate=0.1,     # Slight translation for shifted camera angles
        hsv_h=0.015,       # Slight hue variation
        hsv_s=0.7,         # Saturation variation for varying weather
        hsv_v=0.4          # Brightness variation for varying daylight
    )

    print("[3/4] Exporting best model checkpoint...")
    best_weights_path = Path(project) / name / "weights" / "best.pt"
    dest_dir = Path(save_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "best.pt"

    if best_weights_path.exists():
        shutil.copy(best_weights_path, dest_path)
        print(f"[SUCCESS] Best weights successfully saved to: {dest_path}")
    else:
        print(f"[WARNING] Expected best weights at {best_weights_path} not found.")

    print("[4/4] Training complete. Logs, curves, and confusion matrix available in:")
    print(f"      {Path(project) / name}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficVision YOLO11 Training")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="yolo11_traffic")
    parser.add_argument("--save-dir", type=str, default="weights")

    args = parser.parse_args()
    run_training(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=args.project,
        name=args.name,
        save_dir=args.save_dir
    )
