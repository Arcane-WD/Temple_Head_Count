# Temple Head Count System

A modular, GPU-accelerated computer vision pipeline for automated entry/exit analytics using YOLO-based detection, ByteTrack multi-object tracking, dynamic gate calibration, and event-based counting.

Designed for fixed CCTV installations monitoring physical entry gates.

---

## Overview

This system provides:

* Person detection using YOLOv8
* Multi-object tracking via ByteTrack
* Directional entry/exit counting using line-crossing logic
* Optional motion-based gate auto-calibration
* CUDA acceleration support
* Modular backend structure for future demographic and analytics extensions

The system performs event-based counting without facial recognition or biometric storage.

---

## System Architecture

### Core Components

* Detection: YOLOv8 (default: yolov8s)
* Tracking: ByteTrack (custom configuration)
* Counting: Ultralytics ObjectCounter
* Calibration: Kinematic motion-vector PCA
* Configuration-driven execution via `config.py`

### Pipeline Flow

1. Video selection
2. Optional dynamic gate calibration
3. Person detection
4. Multi-object tracking
5. Directional line-crossing detection
6. Annotated video generation

---

## Project Structure

```
temple_proj/
├─ core/
│  └─ counter.py
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

Model weights and runtime artifacts are excluded from version control.

---

## Installation

Using uv (recommended):

```
uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Ensure CUDA-enabled PyTorch is installed for GPU acceleration.

---

## Configuration

All runtime parameters are defined in `config.py`.

Key fields:

* `MODEL_PATH`
* `AUTO_CALIBRATE`
* `MAX_CALIBRATION_FRAMES`
* `CALIBRATION_FRACTION`
* `MIN_FRAMES_FOR_CALIBRATION`
* `GATE_LINE`
* `CONF_THRESH`
* `IOU_THRESH`
* `TRACKER_CONFIG`

The configuration file remains declarative.
All procedural logic is executed in `main.py`.

---

## Usage

1. Place input videos inside:

```
data/input_vids/
```

2. Run:

```
python main.py
```

3. Select the desired input video number when prompted.

4. Output is saved to:

```
data/output_vids/annotated_vids/
```

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

Tested on 576×768 CCTV footage:

* ~14–20 ms tracking time
* ~2–3 ms counting overhead
* ~45–50 FPS on RTX 3050 (CUDA)
* Stable IN/OUT directional counts
* Low GPU overhead

---

## Design Principles

* Modular backend separation
* GPU-aware inference
* Track-level event logic
* Calibration safety checks
* Production-ready file structure
* Extendable analytics framework

---

## Limitations

* Designed for fixed-camera gate scenarios
* Not optimized for open-field crowd density estimation
* Performance depends on camera angle and occlusion level
* Model weights are not included in repository

---

## Roadmap

Planned enhancements:

* ONNX-based pedestrian attribute recognition (gender classification)
* Track-level demographic caching
* JSON analytics export
* Batch processing for large datasets
* Multi-camera aggregation
* Dashboard integration
* Ground-truth benchmarking

---
