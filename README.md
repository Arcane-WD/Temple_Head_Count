# Temple Head Count System

A modular, GPU-accelerated computer vision pipeline for automated entry/exit analytics and demographic classification using YOLO-based detection, ByteTrack multi-object tracking, dynamic gate calibration, ONNX-based gender recognition, and event-based counting.

Designed for fixed CCTV installations monitoring physical entry gates.

---

## Overview

This system provides:

* Person detection using YOLOv8
* Multi-object tracking via ByteTrack
* Directional entry/exit counting using line-crossing logic
* Track-level gender classification using a ConvNeXt-Tiny ONNX model
* Optional motion-based gate auto-calibration
* CUDA acceleration support
* Modular backend structure for future analytics extensions

The system performs event-based counting and attribute classification without facial recognition or biometric storage.

---

## System Architecture

### Core Components

* Detection: YOLOv8 (default: yolov8s)
* Tracking: ByteTrack (custom configuration)
* Counting: Ultralytics ObjectCounter
* Gender Classification: ConvNeXt-Tiny (ONNX, 82.44% accuracy on PA-100K)
* Calibration: Kinematic motion-vector PCA
* Configuration-driven execution via `config.py`

### Pipeline Flow

1. Video selection
2. Optional dynamic gate calibration
3. Person detection
4. Multi-object tracking
5. Directional line-crossing detection
6. Per-track gender classification with majority voting
7. Annotated video generation

---

## Project Structure

```
temple_proj/
├─ core/
│  ├─ counter.py
│  └─ gender.py
├─ utils/
│  └─ video_io.py
├─ data/
│  ├─ input_vids/
│  └─ output_vids/
│     ├─ annotated_vids/
│     └─ logs/
├─ calibrate.py
├─ config.py
├─ custom_bytetrack.yaml
├─ main.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

Model weights (`.pt`, `.onnx`) and runtime artifacts are excluded from version control.

---

## Installation

Using uv (recommended):

```
uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Ensure CUDA-enabled PyTorch is installed for GPU acceleration.
ONNX Runtime with CUDA support is required for GPU-accelerated gender classification.

---

## Configuration

All runtime parameters are defined in `config.py`.

### Detection and Tracking

* `MODEL_PATH` -- YOLO model weight file
* `CONF_THRESH` -- Detection confidence threshold
* `IOU_THRESH` -- NMS IoU threshold
* `TARGET_CLASSES` -- Object class indices to track
* `TRACKER_CONFIG` -- ByteTrack configuration file
* `GATE_LINE` -- Manual gate line coordinates

### Calibration

* `AUTO_CALIBRATE` -- Enable automatic gate calibration
* `MAX_CALIBRATION_FRAMES` -- Upper frame limit for calibration
* `CALIBRATION_FRACTION` -- Fraction of video used for calibration
* `MIN_FRAMES_FOR_CALIBRATION` -- Minimum frames required

### Gender Classification

* `GENDER_MODEL_PATH` -- Path to the ONNX gender classification model
* `GENDER_REQUIRED_VOTES` -- Number of inference votes before locking a prediction
* `GENDER_CONF_THRESH` -- Minimum confidence threshold for a valid classification
* `STALE_TRACK_TIMEOUT` -- Frames before an unseen track is evicted from the cache

The configuration file remains declarative.
All procedural logic is executed in `main.py`.

---

## Usage

1. Place input videos inside:

```
data/input_vids/
```

2. Place the ONNX gender model in the project root (or update `GENDER_MODEL_PATH` in `config.py`).

3. Run:

```
python main.py
```

4. Select the desired input video number when prompted.

5. Output is saved to:

```
data/output_vids/annotated_vids/
```

---

## Gender Classification

The gender classification module operates at the track level using the following process:

1. For each tracked person, the bounding box crop is extracted from the original frame.
2. The crop is preprocessed to match training conditions (85% vertical crop, resize to 256x128, center crop to 224x112, ImageNet normalization).
3. The ConvNeXt-Tiny ONNX model produces a two-class softmax output (Female, Male).
4. Predictions are accumulated per track over multiple frames.
5. After reaching the configured vote threshold, the final classification is locked using averaged probabilities.
6. Tracks that are no longer visible are evicted after a configurable timeout to prevent memory growth.

Annotated output displays gender labels above each bounding box with color coding:

* Yellow -- Female
* Blue -- Male
* Gray -- Unknown (below confidence threshold)

If the ONNX model file is not found at the configured path, gender classification is automatically disabled. The counting pipeline continues to operate without interruption.

---

## Automatic Gate Calibration

When enabled:

```
AUTO_CALIBRATE = True
```

The system:

* Evaluates video length
* Selects dynamic calibration frame count
* Computes dominant crowd flow using motion vectors
* Generates a perpendicular gate line
* Applies calibration before tracking begins

Short clips automatically bypass calibration to avoid unstable geometry.

Recommended: Calibrate once per fixed camera installation.

---

## Performance

Tested on 576x768 CCTV footage:

* ~14-20 ms tracking time
* ~2-3 ms counting overhead
* ~45-50 FPS on RTX 3050 (CUDA)
* Stable IN/OUT directional counts
* Low GPU overhead

Gender classification adds minimal per-frame latency due to lightweight ONNX inference and per-track caching.

---

## Design Principles

* Modular backend separation
* GPU-aware inference
* Track-level event logic and demographic caching
* Calibration safety checks
* Production-ready file structure
* Graceful degradation when optional components are unavailable
* Extendable analytics framework

---

## Limitations

* Designed for fixed-camera gate scenarios
* Not optimized for open-field crowd density estimation
* Performance depends on camera angle and occlusion level
* Gender classification accuracy is dependent on crop quality and camera angle
* Model weights are not included in repository

---

## Roadmap

Planned enhancements:

* JSON analytics export with demographic breakdown
* Batch processing for large datasets
* Multi-camera aggregation
* Dashboard integration
* Ground-truth benchmarking

---
