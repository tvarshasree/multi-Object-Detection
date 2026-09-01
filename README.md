# Multi-Class Detection & Density Analytics

An end-to-end computer vision and traffic analytics application that performs multi-class object detection, vehicle/pedestrian counting, road-occupancy estimation, and real-time congestion classification from uploaded images.

---

## 📌 Features

- **Multi-Class Detection:** Detects only 4 key traffic classes: `Person`, `Car`, `Bus`, and `Motorcycle / Two-Wheeler`.
- **Targeted Counting:** Separates pedestrian counts from total motorized vehicle counts.
- **Traffic Density Estimation:** Uses Passenger Car Equivalent (PCE) weighted scoring alongside bounding box area coverage.
- **Congestion Level Classification:** Automatically categorizes traffic status as **Low**, **Medium**, or **High**.
- **Interactive Dashboard:** Upload UI, visual bounding-box overlays, KPI metric cards, and class distribution charts.

---
## 🛠️ Tech Stack

- **Model:** Ultralytics YOLOv8 / YOLOv11 (`yolov8n.pt` / `yolov11n.pt`)
- **Backend / Engine:** Python 3.10+, PyTorch, OpenCV (`cv2`)
- **Frontend / Dashboard:** Streamlit (or FastAPI + React)
- **Data Visualization:** Plotly / Matplotlib

---

## 🧮 Congestion Classification Logic

The system computes a **Passenger Car Equivalent (PCE) Score ($S$)** to account for differing road-space occupancy:

$$S = (1.0 \times N_{\text{car}}) + (0.5 \times N_{\text{motorcycle}}) + (2.5 \times N_{\text{bus}})$$

| Congestion Level | PCE Score ($S$) | Road Coverage Area | Visual Indicator |
|---|---|---|---|
| **Low** (Free Flow) | $S < 5$ | $< 15\%$ | 🟢 Green |
| **Medium** (Moderate) | $5 \le S < 12$ | $15\% - 35\%$ | 🟡 Yellow |
| **High** (Congested) | $S \ge 12$ | $> 35\%$ | 🔴 Red |

---

## 🚀 Quick Start

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone [https://github.com/your-username/traffic-density-analytics.git](https://github.com/your-username/traffic-density-analytics.git)
cd traffic-density-analytics

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
