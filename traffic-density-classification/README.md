# TrafficVision: Real-Time Traffic Density & Congestion Classification Using YOLO11

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Model-Ultralytics_YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, vision-only deep learning system designed to monitor urban roadway traffic, count road users, isolate pedestrian foot traffic from vehicular volume, and classify instantaneous traffic density into **Low**, **Medium**, or **High** states.

**Author:** Varshasree Tadiparthi (M.Tech 2nd Year, Computer Science & Data Processing, Roll: 25MA60R37)  
**Primary Reference:** Student Project Report (*TrafficVision*)  
**Reference Paper:** D. Maladkar et al., *"Traffic Flow Detection and Prediction Using YOLO11: A Threshold-Based Approach for Identifying Traffic Levels"*, INCOFT 2025 ([SCITEPRESS 135920](https://www.scitepress.org/Papers/2025/135920/135920.pdf)).

---

## 1. Project Overview & Problem Statement

Urban traffic congestion results in severe economic losses, air pollution, and fuel wastage. Conventional traffic monitoring infrastructure relies on expensive sensors such as inductive loops embedded in pavement or radar counters, requiring high installation and maintenance expenditures. Surveillance cameras are already ubiquitous in urban centers, but they lack autonomous visual intelligence to quantify traffic states.

**TrafficVision** bridges this gap by converting standard roadway video and image feeds into structured, actionable traffic intelligence.

### Key Objectives
1. **Multi-Class Road-User Detection**: Accurately detect four essential classes: **Person (0)**, **Car (1)**, **Bus (2)**, and **Motorcycle (3)** using the modern **YOLO11** architecture.
2. **Pedestrian Segregation**: Unlike traditional systems that count all detected bounding boxes indiscriminately, TrafficVision separates pedestrian volume from vehicle volume. Pedestrian-heavy crosswalks will **never** trigger false vehicular congestion alerts.
3. **Three-Tier Density Classification**: Classify vehicular congestion into actionable levels: `Low`, `Medium`, or `High` using calibrated threshold bounds ($T_1, T_2$).
4. **Full-Stack Serving Layer**: Deliver results via an asynchronous **FastAPI** service and a responsive web dashboard with live heads-up display (HUD) visualization.

---

## 2. Target Benchmark Results (Report Table 4.1)

The system targets reproduction of the validation performance metrics established in Chapter 4 of the project report:

| Object Class | Precision (%) | Recall (%) | mAP@0.5 |
| :--- | :---: | :---: | :---: |
| **Car** | 80.5% | 81.0% | 0.81 |
| **Bus** | 81.2% | 82.5% | 0.78 |
| **Motorcycle / Two-Wheeler** | 79.8% | 78.5% | 0.73 |
| **Person (Pedestrian)** | 78.5% | 78.2% | 0.69 |
| **Overall / Aggregate** | **81.8%** | **80.05%** | **0.75** |

### Benchmark Comparison (Report Table 3.1)

| Dimension | Maladkar et al. (2025) [1] | Shreya & Sunith Kumar [2] | TrafficVision (This Work) |
| :--- | :---: | :---: | :---: |
| **Backbone** | YOLO11 | YOLOv8 | **YOLO11** |
| **Classes** | 8 classes | 4 classes | **4 target classes** |
| **Serving Layer** | Not specified | Flask | **FastAPI + Uvicorn** |
| **Congestion Levels** | 2 (`Heavy` / `Light`) | 3 (`Low` / `Moderate` / `High`) | **3 (`Low` / `Medium` / `High`)** |
| **Pedestrian Segregation** | Mixed with vehicles | Not detected | **Explicitly segregated** |
| **Precision** | 0.88 | 0.925 | **0.818 (81.8%)** |
| **Recall** | 0.85 | 0.927 | **0.8005 (80.05%)** |
| **mAP@0.5** | 0.773 | 0.970 | **0.750 (75.0%)** |

---

## 3. System Architecture

```
User / Client Browser
       |
       v
Frontend Web Dashboard (HTML5 / Modern CSS / Vanilla JS)
       |
       v
FastAPI Asynchronous Gateway
       |
       +---> [GET  /health]               -> System & Model Diagnostics
       +---> [GET  /model-info]           -> Architecture & Threshold Metadata
       +---> [GET  /results/{filename}]   -> Annotated Image Asset Delivery
       +---> [POST /predict]              -> Full Pipeline Execution:
                   |
                   v
             Image Validator & Preprocessor (utils/validation.py)
                   |
                   v
             YOLO11 Detection Engine (services/model_service.py)
                   |
                   +---> Object Counter (services/counting_service.py)
                   |         - Computes N_car, N_bus, N_motorcycle -> Vehicle Count
                   |         - Computes N_person                   -> Pedestrian Count
                   |
                   +---> Traffic Classifier (services/classification_service.py)
                   |         - Low:    vehicle_count <= 5
                   |         - Medium: 5 < vehicle_count <= 12
                   |         - High:   vehicle_count > 12
                   |
                   +---> HUD Renderer (utils/image_utils.py)
                             - Draws high-contrast bounding boxes
                             - Injects top status HUD banner
                             - Encodes JPEG output to disk
```

---

## 4. Project Directory Layout

```
traffic-density-classification/
├── app/
│   ├── main.py                     # FastAPI application entrypoint & routing
│   ├── config.py                   # Pydantic environment configuration
│   ├── schemas.py                  # Pydantic request/response data models
│   ├── dependencies.py             # Service singletons & dependency injection
│   ├── routes/
│   │   ├── prediction.py           # /predict and /results/{filename} endpoints
│   │   ├── health.py               # /health status endpoint
│   │   └── model_info.py           # /model-info metadata endpoint
│   ├── services/
│   │   ├── model_service.py        # YOLO11 inference wrapper & class filtering
│   │   ├── detection_service.py    # End-to-end detection orchestrator
│   │   ├── counting_service.py     # Class aggregation & vehicle/pedestrian separation
│   │   └── classification_service.py # Three-tier density classification engine
│   └── utils/
│       ├── image_utils.py          # OpenCV decoding, HUD drawing, file saving
│       └── validation.py           # File extension, MIME, size, path sanitization
├── training/
│   ├── train.py                    # YOLO11 fine-tuning pipeline
│   ├── validate.py                 # Validation split evaluation
│   └── evaluate.py                 # Benchmark reproducibility & report generation
├── scripts/
│   ├── validate_dataset.py         # YOLO annotation & directory structure validator
│   └── dataset_statistics.py       # Dataset profiling & class distribution analysis
├── frontend/
│   ├── index.html                  # Responsive upload & metrics dashboard
│   ├── style.css                   # Modern dark-mode styling
│   └── script.js                   # Asynchronous fetch controller & UI updates
├── dataset/
│   ├── images/{train,val,test}/    # Roadway imagery
│   ├── labels/{train,val,test}/    # Normalized YOLO format txt annotations
│   └── data.yaml                   # Ultralytics dataset definition
├── weights/
│   └── best.pt                     # Target fine-tuned YOLO11 checkpoint
├── outputs/
│   ├── annotated/                  # Saved visual outputs
│   └── reports/                    # Generated Markdown & metric reports
├── tests/
│   ├── test_dataset.py             # Unit tests for dataset validation
│   ├── test_counting.py            # Unit tests for object counting logic
│   ├── test_classification.py      # Unit tests for density classification
│   └── test_api.py                 # Integration tests for FastAPI endpoints
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Complete Python dependencies
└── README.md                       # Master documentation
```

---

## 5. Installation & Environment Setup

### Prerequisites
- Python 3.10 (Recommended: 3.10.x)
- pip & virtualenv
- (Optional) CUDA-enabled NVIDIA GPU for hardware acceleration

### Step 1: Clone and Navigate
```bash
git clone https://github.com/tvarshasree/multi-Object-Detection.git
cd od/traffic-density-classification
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

---

## 6. How to Run

### 6.1 Launch the FastAPI Web Application
Start the Uvicorn server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Web Dashboard: Open [http://localhost:8000](http://localhost:8000)
- Interactive Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative ReDoc Documentation: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 6.2 Test API via cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_road.jpg"
```

### 6.3 Run Automated Tests
```bash
pytest tests/ -v
```

---

## 7. Model Training & Evaluation

### 7.1 Dataset Structure
The dataset follows standard YOLO formatting:
```
dataset/
├── data.yaml
├── images/
│   ├── train/  # Training images
│   ├── val/    # Validation images
│   └── test/   # Test images
└── labels/
    ├── train/  # class_id x_center y_center width height
    ├── val/
    └── test/
```

Validate your dataset before training:
```bash
python scripts/validate_dataset.py dataset/data.yaml
python scripts/dataset_statistics.py dataset
```

### 7.2 Fine-Tuning YOLO11
To fine-tune base YOLO11 on the 4 targeted traffic classes:
```bash
python training/train.py --data dataset/data.yaml --model yolo11n.pt --epochs 50 --batch 16 --imgsz 640 --device cpu
```
Upon completion, the best checkpoint is automatically copied to `weights/best.pt`.

### 7.3 Evaluating Against Baseline Target Metrics
```bash
python training/evaluate.py --model weights/best.pt --data dataset/data.yaml
```

---

## 8. Limitations & Conceptual Boundaries

1. **Spatial Density vs Temporal Flow**: Raw object counts in a single frame provide an instantaneous measure of spatial occupancy. They do not measure velocity, delay time, or queue discharge rates.
2. **No Predictive Forecasting**: The model classifies the observed state in the uploaded image; it does not predict future congestion or traffic build-up.
3. **Heavy Occlusion in Two-Wheelers**: In dense urban traffic, motorcycles riding closely in clusters or partially obscured by buses experience lower recall, as noted in the research report.
4. **Lighting Variability**: Performance is optimized for daytime urban road footage; severe low-light conditions may degrade detection accuracy without domain-specific data augmentation.

---

## 9. Troubleshooting

- **`ModuleNotFoundError: No module named 'ultralytics'`**: Make sure you have activated your virtual environment (`.\.venv\Scripts\Activate.ps1`) before executing scripts.
- **Port 8000 already in use**: Specify a different port using `--port 8080` or update `PORT=8080` in your `.env` file.
- **CUDA out of memory**: If running on GPU, reduce `--batch` size from 16 to 8 or 4 in `training/train.py`, or use `--device cpu`.
