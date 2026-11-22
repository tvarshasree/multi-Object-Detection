# Baseline Prototype: MobileNetV2 SSD

This directory preserves the initial research baseline and proof-of-concept for the multi-object detection project. It implements a custom Single Shot MultiBox Detector (SSD) head built on top of a transfer-learned MobileNetV2 backbone using TensorFlow/Keras and FiftyOne.

## Overview

The purpose of this baseline was to evaluate lightweight two-class object detection on edge-oriented architectures prior to developing the four-class YOLO11 production system.

Target Classes:
- Person (Class 0)
- Car (Class 1)

Dataset:
- Source: Google Open Images V6 via the FiftyOne Zoo (`fiftyone.zoo`).
- Size: 600 training images (300 Person, 300 Car) and 200 validation images (100 Person, 100 Car).
- Preprocessing: Images resized to 320 x 320 x 3 and normalized between 0.0 and 1.0.

## Directory Layout

```
baseline-mobilenet-ssd/
  notebooks/
    object_Detection_SSD_mobilenet.ipynb    Interactive training, evaluation, and visualization
  src/
    data_preprocessing.py                   FiftyOne ingestion and tf.data pipeline
    train.py                                MobileNetV2 backbone and SSD head definition
    testing.py                              Convergence plots and confusion matrix evaluation
    results.py                              Visual inference utilities
  images/
    confusion_matrix1.png                   Confusion matrix heatmap on validation split
    training_curve_3.png                    Localization loss convergence (Huber loss)
    training_curve_4.png                    Classification loss convergence (Cross-Entropy)
    image_recog3.png                        Sample detection output
    image_recog4.png                        Sample detection output
    image_recog5.png                        Sample detection output
  requirements.txt                          TensorFlow and FiftyOne dependencies
  README.md                                 Component documentation
```

## Architectural Design

- Feature Extractor: MobileNetV2 pre-trained on ImageNet. Foundational feature layers are frozen, and the top 40 layers are unfrozen for fine-tuning.
- Detection Head: Parallel 2D convolutional layers mapped via Global Average Pooling into 10 spatial object slots per frame.
  - Bounding Box Regression: Outputs 4 normalized coordinates per slot with Sigmoid activation.
  - Classification: Outputs class probabilities across target classes via Softmax.
- Loss Function: Multi-task loss combining Huber Loss for bounding boxes (weight 40.0) and Sparse Categorical Cross-Entropy for class labels (weight 1.0).

## Findings and Transition to YOLO11

While MobileNetV2 SSD provided a fast and lightweight foundation for basic two-class detection, evaluation on dense roadway scenes revealed key architectural limitations:
1. Fixed Slot Constraint: The custom head restricted detections to a maximum of 10 objects per image, which is insufficient for crowded urban intersections.
2. Small Object Recall: Small, distant vehicles and motorcycles suffered from lower recall due to the lack of an integrated feature pyramid network.
3. Class Scope: Traffic monitoring required separate counting of buses and two-wheelers, as well as pedestrian segregation.

These findings motivated the transition to the production system located in `traffic-density-classification`, which uses Ultralytics YOLO11 and FastAPI.

## Setup and Running

To run this baseline separately from the production system:

```bash
cd baseline-mobilenet-ssd
python -m venv .venv

# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook notebooks/object_Detection_SSD_mobilenet.ipynb
```
