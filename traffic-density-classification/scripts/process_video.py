"""
TrafficVision - Video Stream and Multi-Object Tracking Pipeline (Stage 4)
Processes video files or RTSP streams using YOLO11 and ByteTrack:
- Assigns persistent tracking IDs to road users across frames
- Segregates vehicle volume from pedestrian foot traffic
- Classifies instantaneous traffic density into Low, Medium, or High
- Renders bounding boxes, track labels, and top heads-up display (HUD)
- Exports the annotated video stream to disk
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Point Ultralytics config to workspace to avoid Windows AppData permissions
workspace_root = Path(__file__).resolve().parent.parent
ultralytics_dir = workspace_root / ".ultralytics"
ultralytics_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_dir)

TARGET_CLASSES = {
    0: "person",
    1: "car",
    2: "bus",
    3: "motorcycle"
}

COCO_TO_TARGET = {
    0: (0, "person"),
    2: (1, "car"),
    5: (2, "bus"),
    3: (3, "motorcycle")
}

CLASS_COLORS = {
    "person": (255, 140, 0),      # Cyan / Light Blue (BGR)
    "car": (0, 200, 0),           # Green
    "bus": (0, 140, 255),         # Orange
    "motorcycle": (0, 230, 255)   # Yellow
}

STATUS_COLORS = {
    "Low": (16, 185, 129),        # Green (BGR: 129, 185, 16)
    "Medium": (11, 158, 245),     # Amber (BGR: 245, 158, 11)
    "High": (68, 68, 239)         # Red (BGR: 239, 68, 68)
}


def classify_density(vehicle_count: int, t1: int = 5, t2: int = 12) -> str:
    if vehicle_count <= t1:
        return "Low"
    elif vehicle_count <= t2:
        return "Medium"
    return "High"


def draw_hud(frame: np.ndarray, vehicle_count: int, pedestrian_count: int, density: str) -> np.ndarray:
    h, w = frame.shape[:2]
    hud_height = max(45, int(h * 0.07))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, hud_height), (w, hud_height), (60, 70, 85), 1)

    status_color = STATUS_COLORS.get(density, (200, 200, 200))
    cv2.circle(frame, (25, hud_height // 2), 7, status_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 1800.0)
    thickness = 1 if scale < 0.7 else 2

    text_status = f"TRAFFIC DENSITY: {density.upper()}"
    cv2.putText(frame, text_status, (45, hud_height // 2 + 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    offset_vehicles = int(w * 0.40)
    text_vehicles = f"Vehicles: {vehicle_count}"
    cv2.putText(frame, text_vehicles, (offset_vehicles, hud_height // 2 + 5), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)

    offset_peds = int(w * 0.65)
    text_peds = f"Pedestrians: {pedestrian_count}"
    cv2.putText(frame, text_peds, (offset_peds, hud_height // 2 + 5), font, scale, (180, 220, 255), thickness, cv2.LINE_AA)

    return frame


def process_video_stream(
    source: str,
    output_path: str = "outputs/annotated/tracked_output.mp4",
    model_path: str = "weights/best.pt",
    fallback_model: str = "yolo11n.pt",
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.45,
    tracker_type: str = "bytetrack.yaml",
    low_thresh: int = 5,
    med_thresh: int = 12
):
    print("=" * 70)
    print("  TRAFFICVISION - VIDEO TRACKING & DENSITY PIPELINE (STAGE 4)")
    print("=" * 70)
    print(f"Video Source:   {source}")
    print(f"Output Path:    {output_path}")
    print(f"Tracker Engine: {tracker_type}")

    active_model = model_path if Path(model_path).exists() else fallback_model
    print(f"Loading Model:  {active_model}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(active_model)
    is_custom_4class = (len(model.names) == 4)

    # Resolve video source (integer if webcam, else string)
    src_input = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src_input)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source '{source}'.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    print(f"Video Specs:    {width}x{height} @ {fps:.1f} FPS (Total frames: {total_frames if total_frames > 0 else 'Stream'})")
    print("-" * 70)

    unique_tracked_vehicles = set()
    unique_tracked_pedestrians = set()
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run YOLO11 tracking
            results = model.track(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                tracker=tracker_type,
                persist=True,
                verbose=False
            )

            current_vehicle_count = 0
            current_pedestrian_count = 0

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    raw_cls = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    track_id = int(boxes.id[i].item()) if boxes.id is not None else -1

                    # Class filtering & mapping
                    if is_custom_4class:
                        if raw_cls not in TARGET_CLASSES:
                            continue
                        target_name = TARGET_CLASSES[raw_cls]
                    else:
                        if raw_cls not in COCO_TO_TARGET:
                            continue
                        _, target_name = COCO_TO_TARGET[raw_cls]

                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # Counting aggregation
                    if target_name in ["car", "bus", "motorcycle"]:
                        current_vehicle_count += 1
                        if track_id != -1:
                            unique_tracked_vehicles.add(track_id)
                    elif target_name == "person":
                        current_pedestrian_count += 1
                        if track_id != -1:
                            unique_tracked_pedestrians.add(track_id)

                    # Draw bounding box
                    color = CLASS_COLORS.get(target_name, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with Track ID
                    label = f"#{track_id} {target_name.upper()} {conf:.2f}" if track_id != -1 else f"{target_name.upper()} {conf:.2f}"
                    font_scale = max(0.4, min(x2 - x1, y2 - y1) / 100.0)
                    cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            density = classify_density(current_vehicle_count, t1=low_thresh, t2=med_thresh)
            frame = draw_hud(frame, current_vehicle_count, current_pedestrian_count, density)
            writer.write(frame)

            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx:05d}: Vehicles={current_vehicle_count}, Pedestrians={current_pedestrian_count}, Density={density}")

    finally:
        cap.release()
        writer.release()

    print("=" * 70)
    print("  TRACKING PIPELINE COMPLETED")
    print(f"Total Frames Processed:      {frame_idx}")
    print(f"Cumulative Unique Vehicles:  {len(unique_tracked_vehicles)}")
    print(f"Cumulative Unique Pedestrians:{len(unique_tracked_pedestrians)}")
    print(f"Output Video Saved:          {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficVision Video Stream & Tracking Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or RTSP stream URL")
    parser.add_argument("--output", type=str, default="outputs/annotated/tracked_output.mp4", help="Output annotated video path")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Fine-tuned weights path")
    parser.add_argument("--fallback", type=str, default="yolo11n.pt", help="Fallback pretrained checkpoint")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence cutoff")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker engine (bytetrack.yaml or botsort.yaml)")
    parser.add_argument("--t1", type=int, default=5, help="Low threshold")
    parser.add_argument("--t2", type=int, default=12, help="Medium threshold")

    args = parser.parse_args()
    process_video_stream(
        source=args.source,
        output_path=args.output,
        model_path=args.model,
        fallback_model=args.fallback,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        tracker_type=args.tracker,
        low_thresh=args.t1,
        med_thresh=args.t2
    )
