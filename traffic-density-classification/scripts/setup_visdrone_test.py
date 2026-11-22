"""
TrafficVision - VisDrone2019-DET Test Integration Script
Automates:
1. Downloading the VisDrone2019-DET validation/test split (81.6 MB)
2. Extracting images and converting annotations to our target 4-class YOLO format:
   - VisDrone pedestrian (1) and people (2) -> Target 0 (person)
   - VisDrone car (4) and van (5)           -> Target 1 (car)
   - VisDrone bus (9)                       -> Target 2 (bus)
   - VisDrone motor (10)                    -> Target 3 (motorcycle)
3. Storing images in dataset/images/val and annotations in dataset/labels/val
4. Running the validation and live inference pipeline
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image

# Point Ultralytics config to workspace
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))
ultralytics_dir = workspace_root / ".ultralytics"
ultralytics_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_dir)

VISDRONE_VAL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"

# Mapping from VisDrone categories (1-indexed in raw annotations) to our 4 classes
VISDRONE_TO_TARGET = {
    1: 0,   # pedestrian -> person
    2: 0,   # people -> person
    4: 1,   # car -> car
    5: 1,   # van -> car
    9: 2,   # bus -> bus
    10: 3   # motor -> motorcycle
}


def download_visdrone(dest_zip: Path):
    if dest_zip.exists() and dest_zip.stat().st_size > 80_000_000:
        print(f"[1/4] Archive '{dest_zip.name}' already downloaded ({dest_zip.stat().st_size / 1e6:.1f} MB).")
        return

    print(f"[1/4] Downloading VisDrone2019-DET validation split (81.6 MB) from official mirror...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(VISDRONE_VAL_URL, headers=headers)

    with urllib.request.urlopen(req) as response, open(dest_zip, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"      Downloaded {downloaded / 1e6:.1f} MB of {total_size / 1e6:.1f} MB ({percent:.1f}%)", end="\r")

    print(f"\n      Download complete: {dest_zip}")


def convert_and_populate_dataset(zip_path: Path, dataset_root: Path, max_samples: int = None):
    print(f"[2/4] Extracting and converting VisDrone annotations to 4-class YOLO format...")
    
    img_dest = dataset_root / "images" / "val"
    lbl_dest = dataset_root / "labels" / "val"
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    converted_count = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        # Find all image entries in the archive
        all_members = z.namelist()
        image_members = [m for m in all_members if m.startswith("VisDrone2019-DET-val/images/") and m.lower().endswith(".jpg")]
        
        if max_samples:
            image_members = image_members[:max_samples]

        print(f"      Found {len(image_members)} images to process.")

        for img_entry in image_members:
            img_filename = Path(img_entry).name
            base_stem = Path(img_entry).stem
            annot_entry = f"VisDrone2019-DET-val/annotations/{base_stem}.txt"

            # Extract image
            target_img_path = img_dest / img_filename
            with z.open(img_entry) as src, open(target_img_path, "wb") as dst:
                dst.write(src.read())
            extracted_count += 1

            # Convert annotation if it exists
            target_lbl_path = lbl_dest / f"{base_stem}.txt"
            yolo_lines = []

            if annot_entry in all_members:
                with Image.open(target_img_path) as img:
                    img_w, img_h = img.size

                dw = 1.0 / img_w
                dh = 1.0 / img_h

                with z.open(annot_entry) as annot_file:
                    content = annot_file.read().decode("utf-8")
                    for line in content.strip().splitlines():
                        if not line.strip():
                            continue
                        parts = line.split(",")
                        if len(parts) < 8:
                            continue

                        # Format: bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
                        score = parts[4].strip()
                        if score == "0":
                            # VisDrone score 0 = ignored region
                            continue

                        category_id = int(parts[5].strip())
                        if category_id not in VISDRONE_TO_TARGET:
                            continue

                        target_cls = VISDRONE_TO_TARGET[category_id]
                        bx = float(parts[0].strip())
                        by = float(parts[1].strip())
                        bw = float(parts[2].strip())
                        bh = float(parts[3].strip())

                        if bw <= 0 or bh <= 0:
                            continue

                        # Convert to normalized YOLO format
                        x_center = (bx + bw / 2.0) * dw
                        y_center = (by + bh / 2.0) * dh
                        w_norm = bw * dw
                        h_norm = bh * dh

                        # Clamp to [0, 1]
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        w_norm = max(0.0, min(1.0, w_norm))
                        h_norm = max(0.0, min(1.0, h_norm))

                        yolo_lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            with open(target_lbl_path, "w", encoding="utf-8") as out_lbl:
                out_lbl.write("\n".join(yolo_lines))
            converted_count += 1

    print(f"      Successfully extracted {extracted_count} images to: {img_dest}")
    print(f"      Successfully written {converted_count} YOLO label files to: {lbl_dest}")


def run_pipeline_test(dataset_root: Path, sample_count: int = 5):
    print(f"[3/4] Running live TrafficVision pipeline on VisDrone surveillance test frames...")
    from app.dependencies import get_detection_service

    svc = get_detection_service()
    val_images = sorted((dataset_root / "images" / "val").glob("*.jpg"))[:sample_count]

    print("-" * 75)
    print(f"{'Image':<22} | {'Time':<8} | {'Vehicles':<8} | {'Peds':<6} | {'Density':<8} | {'Status'}")
    print("-" * 75)

    for img_path in val_images:
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        res = svc.process_image(img_bytes, original_filename=img_path.name)
        v_str = f"{res.vehicle_count} (C:{res.counts.car}, B:{res.counts.bus}, M:{res.counts.motorcycle})"
        print(f"{img_path.name:<22} | {res.total_processing_ms:>6.1f}ms | {res.vehicle_count:>8} | {res.pedestrian_count:>6} | {res.traffic_density:<8} | OK")

    print("-" * 75)
    print(f"[SUCCESS] Sample annotated frames saved to: outputs/annotated/")


def run_model_benchmark(data_yaml: Path):
    print(f"[4/4] Running YOLO11 validation evaluation on VisDrone dataset split...")
    from ultralytics import YOLO

    model_path = Path("weights/best.pt") if Path("weights/best.pt").exists() else Path("yolo11n.pt")
    print(f"      Evaluating checkpoint: {model_path}")
    
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.45,
        device="cpu",
        plots=True,
        verbose=False
    )

    print("=" * 75)
    print("  VISDRONE VALIDATION BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Overall Precision (mp):  {metrics.box.mp * 100:.2f}%")
    print(f"Overall Recall (mr):     {metrics.box.mr * 100:.2f}%")
    print(f"Overall mAP@0.5:         {metrics.box.map50:.4f}")
    print(f"Overall mAP@0.5:0.95:    {metrics.box.map:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisDrone2019-DET Download and Test Pipeline")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of extracted images (default: all)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip full YOLO benchmark evaluation")
    args = parser.parse_args()

    zip_file = workspace_root / "VisDrone2019-DET-val.zip"
    dataset_dir = workspace_root / "dataset"
    yaml_file = dataset_dir / "data.yaml"

    download_visdrone(zip_file)
    convert_and_populate_dataset(zip_file, dataset_dir, max_samples=args.samples)
    run_pipeline_test(dataset_dir, sample_count=5)

    if not args.skip_eval:
        run_model_benchmark(yaml_file)
