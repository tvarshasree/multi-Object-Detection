# System Requirements Specification: TrafficVision

**Project Title:** Real-Time Traffic Density and Congestion Classification Using Deep Learning  
**Student:** Varshasree Tadiparthi (M.Tech 2nd Year, Computer Science and Data Processing, Roll: 25MA60R37)  
**Primary Reference:** Student Project Report (*TrafficVision*)  
**Secondary Reference:** Maladkar et al., *"Traffic Flow Detection and Prediction Using YOLO11: A Threshold-Based Approach for Identifying Traffic Levels"*, INCOFT 2025.

---

## 1. Executive Summary & Problem Statement

### 1.1 Background & Motivation
Urban traffic congestion leads to substantial economic losses, fuel waste, and increased carbon emissions. Traditional traffic monitoring relies heavily on fixed, hardware-intensive sensors such as inductive loop detectors and radar sensors, which carry high installation and maintenance overheads. Conversely, urban camera surveillance networks are already extensively deployed.

The goal of this project is to develop **TrafficVision**, an end-to-end, vision-only, deep-learning-based system capable of:
1. Detecting key road users with high precision in real-time.
2. Counting vehicles and pedestrians in isolated streams.
3. Classifying road traffic density into **Low**, **Medium**, or **High** states.
4. Serving results through a robust **FastAPI** backend and an intuitive web dashboard.

---

## 2. Critical Evidence Rules & Decision Provenance

Each requirement and technical decision is classified according to its evidentiary origin:
- `[CONFIRMED]`: Confirmed directly by the student's project report or repository files.
- `[DEFAULT]`: Chosen as a reasonable, documented engineering default.
- `[CONFIRMATION_REQUIRED]`: Requires confirmation from the user.
- `[NOT_REPRODUCIBLE]`: Cannot be reproduced without additional raw assets (e.g. original raw image dataset).

### Evidence Matrix

| Parameter / Decision | Value | Provenance | Notes |
| :--- | :--- | :--- | :--- |
| **Project Title** | Real-Time Traffic Density and Congestion Classification Using Deep Learning | `[CONFIRMED]` | Report Cover Page |
| **Model Backbone** | Ultralytics YOLO11 | `[CONFIRMED]` | Report Table 1.1, Section 3.1.1 |
| **Target Classes** | 4 classes: Person (0), Car (1), Bus (2), Motorcycle (3) | `[CONFIRMED]` | Report Table 5.2 |
| **Paper vs Project Difference** | Paper uses 8 classes; this project explicitly uses 4 | `[CONFIRMED]` | Report Table 3.1 & Section 3.1.3 |
| **Pedestrian Segregation** | Pedestrian count tracked separately from vehicle count | `[CONFIRMED]` | Report Section 1.2, 5.3, Table 3.1 |
| **Congestion Categories** | 3 Levels: `Low`, `Medium`, `High` | `[CONFIRMED]` | Report Section 2.3.2 |
| **Thresholds $T_1, T_2$** | $T_1 = 5$ (Low $\le 5$), $T_2 = 12$ ($5 <$ Medium $\le 12$, High $> 12$) | `[DEFAULT]` | Calibrated defaults based on typical urban frame densities |
| **Confidence Threshold** | 0.35 | `[DEFAULT]` | Balances false positives and recall |
| **NMS IoU Threshold** | 0.45 | `[DEFAULT]` | Standard YOLO NMS threshold |
| **Serving Architecture** | FastAPI + Uvicorn + Jinja2 | `[CONFIRMED]` | Report Table 2.1, Appendix A & B |
| **Frame Processing** | OpenCV (frame decode, draw, re-encode) | `[CONFIRMED]` | Report Section 5.2, Appendix B |
| **Validation Targets** | P: 81.8%, R: 80.05%, mAP@0.5: 0.75 | `[CONFIRMED]` | Report Table 4.1 |
| **Original Raw Dataset** | Proprietary annotated road frames | `[NOT_REPRODUCIBLE]` | Raw training images/labels were not bundled with git repo |
| **Pretrained Checkpoint** | `yolo11n.pt` with class mapping to target 4 | `[DEFAULT]` | Provides immediate fully-functional inference out of the box |

---

## 3. Class Definitions & Mapping

The project focuses exclusively on the four primary urban road-user classes:

```
Class ID 0: person        (Pedestrians, cyclists, people on foot)
Class ID 1: car           (Sedans, SUVs, hatchbacks, taxis, vans)
Class ID 2: bus           (Public transit buses, coaches, minibuses)
Class ID 3: motorcycle    (Motorcycles, mopeds, scooters, two-wheelers)
```

### Pretrained Backbone Mapping
When operating with official YOLO11 weights (trained on 80 COCO classes), detections are dynamically filtered and mapped:
- COCO Class `0` (`person`) $\rightarrow$ Target Class `0` (`person`)
- COCO Class `2` (`car`) $\rightarrow$ Target Class `1` (`car`)
- COCO Class `5` (`bus`) $\rightarrow$ Target Class `2` (`bus`)
- COCO Class `3` (`motorcycle`) $\rightarrow$ Target Class `3` (`motorcycle`)
- All other COCO classes (e.g. truck, dog, traffic light) are discarded.

When operating with custom fine-tuned weights (`model/yolo11_traffic.pt`), classes map directly 1:1.

---

## 4. Object Counting & Classification Methodology

### 4.1 Counting Engine
For each image or video frame:
$$\text{vehicle\_count} = N_{\text{car}} + N_{\text{bus}} + N_{\text{motorcycle}}$$
$$\text{pedestrian\_count} = N_{\text{person}}$$
$$\text{total\_objects} = \text{vehicle\_count} + \text{pedestrian\_count}$$

### 4.2 Classification Logic
$$\text{Traffic State} = \begin{cases}
\textbf{Low}, & \text{if } \text{vehicle\_count} \le T_1 \\
\textbf{Medium}, & \text{if } T_1 < \text{vehicle\_count} \le T_2 \\
\textbf{High}, & \text{if } \text{vehicle\_count} > T_2
\end{cases}$$

### 4.3 Conceptual Boundary & Explicit Limitations
- **Image Proxy Only**: An instantaneous count of vehicles in a single frame is a spatial density proxy; it does **not** directly measure speed, flow rate, or queue accumulation time.
- **No Forecasting**: The model describes current visual density; it does **not** predict future congestion.
- **Occlusion**: Dense clusters of two-wheelers may experience reduced recall under heavy occlusion.

---

## 5. API Specification

| Method | Endpoint | Description | Expected Payload | Response Format |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server and model health status | None | JSON (`status`, `model_loaded`, `device`, `classes`) |
| `GET` | `/model-info` | Metadata on loaded model backbone | None | JSON (`backbone`, `input_size`, `thresholds`, `classes`) |
| `POST` | `/predict` | Full detection, counting, & classification | Multipart Image/Video | JSON with counts, density, bounding boxes, media URL |
| `GET` | `/results/{filename}` | Retrieve annotated media output | None | Image/Video binary stream |
| `GET` | `/api/benchmark` | Retrieve target validation baseline metrics | None | JSON matching Report Table 4.1 |

---

## 6. Target Validation Metrics (Table 4.1)

To evaluate and reproduce the reported performance:
- **Car**: Precision = 80.5%, Recall = 81.0%, mAP@0.5 = 0.81
- **Bus**: Precision = 81.2%, Recall = 82.5%, mAP@0.5 = 0.78
- **Motorcycle**: Precision = 79.8%, Recall = 78.5%, mAP@0.5 = 0.73
- **Person**: Precision = 78.5%, Recall = 78.2%, mAP@0.5 = 0.69
- **Aggregate**: **Precision = 81.8%**, **Recall = 80.05%**, **mAP@0.5 = 0.75**
