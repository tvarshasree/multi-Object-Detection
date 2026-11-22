# Real-Time Traffic Density and Congestion Classification

This repository provides an end-to-end computer vision system for traffic monitoring, object detection, vehicle counting, and road congestion classification. The core system, located in the `traffic-density-classification` directory, uses Ultralytics YOLO11 for object detection and a FastAPI backend with an integrated web dashboard to process roadway imagery in real time.

The project detects four primary road-user classes (Person, Car, Bus, and Motorcycle), isolates pedestrian movement from vehicular traffic, and assigns an instantaneous traffic density level (Low, Medium, or High) based on calibrated vehicle counts.

## Overview of Recent Updates

The repository has transitioned from an initial two-class prototype into a complete, modular traffic analytics platform:

- Upgraded model backbone to YOLO11 with support for fine-tuned weights and automatic fallback to pretrained weights with COCO class mapping.
- Expanded detection scope to four dedicated classes: Person, Car, Bus, and Motorcycle.
- Introduced pedestrian segregation logic so that high pedestrian volumes (e.g., crosswalks, sidewalks) do not trigger false vehicular congestion alerts.
- Implemented a three-tier traffic density classification engine using calibrated thresholds.
- Built an asynchronous FastAPI service providing endpoints for health monitoring, model metadata, predictions, and media retrieval.
- Integrated an OpenCV visual rendering pipeline that outputs class-specific bounding boxes and a top-mounted heads-up display (HUD) status banner.
- Added a lightweight, responsive web interface for image upload, real-time KPI card display, and annotated result visualization.
- Created standalone scripts for dataset structure validation, annotation integrity checks, class distribution statistics, and model evaluation against target benchmarks.
- Added a complete unit and integration test suite covering dataset schemas, counting logic, classification rules, and API endpoints.

## Target Classes and Classification Logic

### Target Classes

The system focuses strictly on four road-user categories:

- Person (Class 0): Pedestrians, people on foot, and cyclists.
- Car (Class 1): Sedans, hatchbacks, SUVs, taxis, and vans.
- Bus (Class 2): Public transport buses and coaches.
- Motorcycle (Class 3): Two-wheelers, scooters, and motorbikes.

### Counting Logic

Detections are separated into vehicle and pedestrian counts:

- vehicle_count = car_count + bus_count + motorcycle_count
- pedestrian_count = person_count
- total_objects = vehicle_count + pedestrian_count

Pedestrians are monitored separately from vehicles. A crowded sidewalk or zebra crossing increases pedestrian count but does not inflate vehicular congestion levels.

### Congestion Thresholds

Traffic density is categorized based on the active vehicle count:

- Low: vehicle_count <= 5. Free-flowing roadway with low density.
- Medium: 5 < vehicle_count <= 12. Moderate vehicle density with steady movement.
- High: vehicle_count > 12. Heavy vehicular density indicating road congestion.

Threshold values (T1 = 5, T2 = 12) are configurable through environment variables.

## Repository Layout

```
od/
  images/                           Sample input images
  notebooks/                        Historical exploratory notebooks (SSD-MobileNet baseline)
  src/                              Legacy prototype source files
  traffic-density-classification/   Current production system
    app/
      main.py                       FastAPI application and route wiring
      config.py                     Application settings and environment validation
      dependencies.py               Service singleton management
      schemas.py                    Pydantic request and response schemas
      routes/
        health.py                   Health check and readiness endpoint
        model_info.py               Model configuration and metadata endpoint
        prediction.py               Inference and annotated media retrieval
      services/
        model_service.py            YOLO11 model wrapper and class mapping
        detection_service.py        End-to-end inference and orchestration
        counting_service.py         Object aggregation and count separation
        classification_service.py   Density categorization engine
      utils/
        image_utils.py              OpenCV HUD drawing and image encoding
        validation.py               Upload validation and filename sanitization
    frontend/
      index.html                    Web dashboard layout
      style.css                     Dashboard styling
      script.js                     Client-side interaction and API requests
    training/
      train.py                      YOLO11 fine-tuning script
      validate.py                   Validation set evaluation script
      evaluate.py                   Benchmark evaluation and report generator
    scripts/
      validate_dataset.py           Dataset directory and label file validator
      dataset_statistics.py         Class distribution and dataset metrics script
    tests/
      test_api.py                   FastAPI integration tests
      test_classification.py        Density classification unit tests
      test_counting.py              Counting service unit tests
      test_dataset.py               Dataset schema and label format tests
    dataset/
      data.yaml                     Dataset configuration for Ultralytics
    weights/
      best.pt                       Target location for fine-tuned weights
    requirements.txt                Python package dependencies
    README.md                       Component documentation
  README.md                         Repository documentation
```

## System Requirements

- Operating System: Linux, macOS, or Windows
- Python: Version 3.10 or higher
- Hardware: Multi-core CPU; NVIDIA GPU with CUDA support recommended for training and high-throughput inference

## Installation and Setup

All primary application files reside within the `traffic-density-classification` folder.

1. Navigate to the project directory:

```bash
cd traffic-density-classification
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:
```bash
source .venv/bin/activate
```

4. Upgrade pip and install required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Set up the environment configuration:

```bash
cp .env.example .env
```

The default settings work immediately on CPU. If a CUDA GPU is available, set `DEVICE=0` in `.env`.

## Running the Application

### 1. Start the API and Web Dashboard

Run the server with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- Web Dashboard: http://localhost:8000
- Swagger API Documentation: http://localhost:8000/docs
- ReDoc API Documentation: http://localhost:8000/redoc

### 2. Testing via cURL

You can submit an image directly to the prediction endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_road.jpg"
```

The response includes:
- Processing and inference latencies in milliseconds
- Individual counts for car, bus, motorcycle, and person
- Total vehicle count and pedestrian count
- Assigned density level (Low, Medium, or High) with explanatory text
- URL to retrieve the annotated output image with HUD overlays
- Bounding box coordinates and confidence scores for each detected object

### 3. Running Automated Tests

Run the full test suite with pytest:

```bash
pytest tests/ -v
```

## Dataset Preparation and Model Training

### Dataset Structure

The training pipeline expects a standard YOLO dataset structure:

```
dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

The `data.yaml` file defines the dataset paths and four target classes:

```yaml
names:
  0: person
  1: car
  2: bus
  3: motorcycle
nc: 4
```

### Dataset Validation

Before starting training, verify the integrity of the dataset:

```bash
python scripts/validate_dataset.py dataset/data.yaml
python scripts/dataset_statistics.py dataset
```

These scripts verify directory structure, image/label pairings, label format validity, bounding box coordinates, and report class distribution counts.

### Fine-Tuning YOLO11

To fine-tune base YOLO11 weights on your traffic dataset:

```bash
python training/train.py --data dataset/data.yaml --model yolo11n.pt --epochs 50 --batch 16 --imgsz 640 --device cpu
```

When training completes, the script copies the best checkpoint to `weights/best.pt`. The FastAPI service automatically detects and loads `weights/best.pt` upon startup. If `weights/best.pt` is not present, the service falls back to `yolo11n.pt` with COCO class mapping.

### Evaluating Performance

To evaluate fine-tuned weights against the validation split and generate a performance report:

```bash
python training/evaluate.py --model weights/best.pt --data dataset/data.yaml
```

The generated report will be saved to `outputs/reports/`.

## Operational Boundaries and Limitations

1. Single-Frame Spatial Proxy: Frame-level vehicle counts measure instantaneous spatial occupancy. They do not calculate vehicle velocity, queue discharge rates, or travel delay times.
2. Static Analysis: The system classifies current traffic density from observed visual data; it does not forecast future traffic trends or upstream congestion accumulation.
3. Occlusion: In dense traffic conditions, motorcycles travelling close together or behind larger vehicles like buses can experience lower detection recall.
4. Lighting Conditions: The model is tuned for standard daytime urban roadway conditions. Severe rain, fog, or unlit nighttime environments may require additional domain-specific training data.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
