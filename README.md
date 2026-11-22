# Real-Time Traffic Density and Congestion Classification

This repository provides an end-to-end computer vision and traffic analytics system for roadway monitoring, vehicle detection, object counting, and congestion level classification.

The repository is structured into two clean modules:
- `traffic-density-classification/`: The primary production system built with Ultralytics YOLO11 and FastAPI, featuring a real-time web dashboard, heads-up display (HUD) rendering, pedestrian segregation, and three-tier density classification.
- `baseline-mobilenet-ssd/`: The initial exploratory research prototype built with MobileNetV2 and TensorFlow/Keras on Open Images V6 for two-class object detection.

## Primary System: TrafficVision (YOLO11 + FastAPI)

The production application resides in `traffic-density-classification`. It detects four road-user classes, isolates pedestrian foot traffic from vehicular volume, and classifies roadway congestion into Low, Medium, or High states.

### Target Classes

- Person (Class 0): Pedestrians, people on foot, and cyclists.
- Car (Class 1): Sedans, hatchbacks, SUVs, taxis, and vans.
- Bus (Class 2): Public transit buses and coaches.
- Motorcycle (Class 3): Two-wheelers, scooters, and motorbikes.

### Counting and Segregation Logic

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
  requirements.txt                  Root dependencies pointing to production system
  README.md                         Master repository documentation
  LICENSE                           Project license

  traffic-density-classification/   Production System (YOLO11 + FastAPI)
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
    requirements.txt                PyTorch, Ultralytics, and FastAPI dependencies
    README.md                       Production component documentation

  baseline-mobilenet-ssd/           Research Prototype (MobileNetV2 SSD)
    notebooks/
      object_Detection_SSD_mobilenet.ipynb  Jupyter notebook for training and evaluation
    src/
      data_preprocessing.py         FiftyOne Open Images V6 ingestion pipeline
      train.py                      MobileNetV2 transfer learning and SSD head
      testing.py                    Convergence evaluation and confusion matrix
      results.py                    Inference and bounding box visualization
    images/                         Baseline performance curves and confusion matrix
    requirements.txt                TensorFlow and FiftyOne dependencies
    README.md                       Baseline documentation and evaluation findings
```

## Quick Start: Production System

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv

### 1. Installation

```bash
cd traffic-density-classification
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux or macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- Web Dashboard: http://localhost:8000
- Interactive Swagger API Documentation: http://localhost:8000/docs
- Alternative ReDoc API Documentation: http://localhost:8000/redoc

### 3. Test API via cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_road.jpg"
```

### 4. Run Automated Tests

```bash
pytest tests/ -v
```

## Exploratory Baseline (MobileNetV2 SSD)

For details on the initial two-class prototype using TensorFlow/Keras and FiftyOne, refer to the [baseline-mobilenet-ssd/README.md](file:///c:/Users/tvars/Documents/patchamomma2026/od/baseline-mobilenet-ssd/README.md).

## Operational Boundaries and Practical Considerations

1. Spatial Density vs Temporal Flow: Raw vehicle counts in a single frame provide an instantaneous measure of spatial occupancy. They do not calculate velocity, delay time, or queue discharge rates.
2. Static Analysis: The system classifies visual density from the current frame; it does not forecast future traffic build-up.
3. Occlusion: In dense traffic, motorcycles travelling close together or behind buses can experience lower detection recall.
4. Lighting: The model is tuned for standard daytime urban roadway conditions. Nighttime or severe weather may require domain-adapted training data.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
