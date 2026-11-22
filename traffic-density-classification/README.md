# TrafficVision: Real-Time Traffic Density and Congestion Classification

TrafficVision is a vision-based traffic monitoring and density classification system built with Ultralytics YOLO11 and FastAPI. It processes roadway images to detect key road users, separate pedestrian counts from vehicular traffic, and classify overall traffic congestion into Low, Medium, or High states.

## Key Features

- Multi-Class Detection: Detects four targeted road-user classes: Person (0), Car (1), Bus (2), and Motorcycle (3).
- Pedestrian Segregation: Tracks pedestrian counts independently from vehicular counts to avoid false congestion alerts at pedestrian crossings or sidewalks.
- Three-Tier Classification: Categorizes road traffic density as Low, Medium, or High using configurable vehicle-count thresholds.
- Dual-Mode Serving: Operates with fine-tuned custom weights (`weights/best.pt`) or falls back to pretrained YOLO11 weights (`yolo11n.pt`) using an internal COCO class mapping layer.
- Modular FastAPI Service: Provides REST endpoints for health checks, model metadata, image inference, and annotated image retrieval.
- OpenCV Visualization: Generates annotated output images with class-colored bounding boxes and a top heads-up display (HUD) banner showing active counts and status.
- Web Dashboard: Includes a responsive browser interface for drag-and-drop image uploads, live KPI cards, and instant visual feedback.
- Dataset and Evaluation Tools: Provides validation and statistics scripts for YOLO-formatted datasets, training workflows, and report generation.
- Automated Tests: Contains unit and integration tests for API endpoints, counting logic, classification rules, and dataset formatting.

## Target Classes and Classification Rules

### Class Definitions

The model targets four road-user classes:
- Person (0): Pedestrians, cyclists, and individuals on foot.
- Car (1): Passenger vehicles including sedans, hatchbacks, SUVs, and taxis.
- Bus (2): Public transit buses and coaches.
- Motorcycle (3): Two-wheelers, scooters, and motorcycles.

### Counting and Segregation Logic

The system aggregates detections as follows:
- vehicle_count = car_count + bus_count + motorcycle_count
- pedestrian_count = person_count
- total_objects = vehicle_count + pedestrian_count

Pedestrians are counted separately from motorized vehicles so that foot traffic along sidewalks or crosswalks does not inflate vehicular congestion indicators.

### Congestion Thresholds

Traffic density is assigned using the following rules:
- Low: vehicle_count <= 5. The roadway has light traffic and free flow.
- Medium: 5 < vehicle_count <= 12. The roadway has moderate traffic with steady movement.
- High: vehicle_count > 12. The roadway has heavy traffic and noticeable congestion.

The threshold values (T1 = 5, T2 = 12) can be adjusted in the environment configuration.

## System Architecture

The request workflow follows a sequential, modular pipeline:

1. Client Request: An image is submitted via the web dashboard or directly to the `/predict` endpoint.
2. Validation: The upload is checked for allowed image types (JPEG, PNG, WebP), file size limits, and basic integrity.
3. Inference: The YOLO11 model detects objects and returns bounding boxes, class labels, and confidence scores.
4. Counting and Segregation: Detected objects are split into pedestrian and vehicle counts.
5. Density Classification: The vehicle count is evaluated against configured thresholds to assign a congestion level.
6. HUD Rendering: OpenCV draws high-contrast bounding boxes and superimposes a top HUD banner with status and metrics.
7. Response: The API returns structured JSON metrics along with a URL pointing to the annotated image.

## Directory Structure

```
traffic-density-classification/
  app/
    main.py                       FastAPI application entrypoint and middleware
    config.py                     Application settings and environment validation
    dependencies.py               Service singleton management
    schemas.py                    Pydantic request and response schemas
    routes/
      health.py                   Health check and diagnostic endpoint
      model_info.py               Model configuration and metadata endpoint
      prediction.py               Inference and annotated media endpoints
    services/
      model_service.py            YOLO11 wrapper with dual-mode class mapping
      detection_service.py        End-to-end detection orchestration
      counting_service.py         Vehicle and pedestrian counting logic
      classification_service.py   Three-tier congestion classification
    utils/
      image_utils.py              OpenCV image decoding, HUD rendering, and saving
      validation.py               File type, size, and integrity validation
  frontend/
    index.html                    Web dashboard layout
    style.css                     Dashboard styling
    script.js                     Client-side interaction and API communication
  training/
    train.py                      YOLO11 fine-tuning script
    validate.py                   Validation dataset evaluation script
    evaluate.py                   Benchmark evaluation and report generator
  scripts/
    validate_dataset.py           Dataset structure and annotation validator
    dataset_statistics.py         Class distribution and dataset metrics
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
```

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv
- Optional: CUDA-compatible GPU for accelerated training and inference

### 1. Create and Activate Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the sample environment file:

```bash
cp .env.example .env
```

Available environment variables:
- MODEL_PATH: Path to fine-tuned weights (default: `weights/best.pt`)
- FALLBACK_MODEL_PATH: Pretrained fallback model (default: `yolo11n.pt`)
- CONFIDENCE_THRESHOLD: Detection confidence threshold (default: `0.35`)
- IOU_THRESHOLD: NMS intersection over union threshold (default: `0.45`)
- IMAGE_SIZE: Input image dimension for inference (default: `640`)
- DEVICE: Compute device, either `cpu` or `0` for GPU (default: `cpu`)
- LOW_THRESHOLD: Upper limit for Low density (default: `5`)
- MEDIUM_THRESHOLD: Upper limit for Medium density (default: `12`)
- HOST: Server bind address (default: `0.0.0.0`)
- PORT: Server port (default: `8000`)
- MAX_UPLOAD_SIZE_MB: Maximum upload file size in megabytes (default: `25`)

## Running the Application

### Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Web Dashboard: http://localhost:8000
- Interactive API Documentation (Swagger): http://localhost:8000/docs
- Alternative API Documentation (ReDoc): http://localhost:8000/redoc

### Test with cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@road_sample.jpg"
```

### Run Automated Tests

Execute the test suite with pytest:

```bash
pytest tests/ -v
```

## Dataset Preparation and Training

### Dataset Layout

Datasets must follow the standard YOLO directory organization:

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

### Validation and Profiling

Run dataset checks before initiating training:

```bash
python scripts/validate_dataset.py dataset/data.yaml
python scripts/dataset_statistics.py dataset
```

### Fine-Tuning

Train the YOLO11 model on your custom dataset:

```bash
python training/train.py --data dataset/data.yaml --model yolo11n.pt --epochs 50 --batch 16 --imgsz 640 --device cpu
```

Once training completes, the best checkpoint is saved to `weights/best.pt`. The server automatically loads this checkpoint upon next startup.

### Performance Evaluation

Evaluate the trained checkpoint against the validation split:

```bash
python training/evaluate.py --model weights/best.pt --data dataset/data.yaml
```

The evaluation report is saved in Markdown format to `outputs/reports/`.

## Operational Boundaries and Practical Considerations

- Spatial Occupancy vs. Traffic Flow: Single-frame vehicle counting represents instantaneous spatial density. It does not measure vehicle speed, flow rate, or queue clearance times.
- Instantaneous Classification: The system reflects conditions at the exact moment an image was captured; it does not forecast future traffic trends.
- Occlusion in High Density: Two-wheelers travelling in tight clusters or shadowed by large vehicles such as buses may experience lower detection recall.
- Adverse Environmental Factors: The default models are optimized for daytime road footage. Extreme weather or unlit night conditions may require domain-adapted training data.
