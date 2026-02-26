# Temple Head Count System

A modular, GPU-accelerated computer vision pipeline for estimating daily footfall at temple entry gates using YOLO-based detection, ByteTrack multi-object tracking, and vector-based line-crossing logic.

---

## Overview

This project implements a robust entry-counting system designed for fixed CCTV camera installations at physical entry gates.

The system:

* Detects persons using YOLOv8
* Tracks individuals using ByteTrack
* Counts entries and exits using trajectory–line intersection
* Supports automatic gate line calibration using PCA
* Runs efficiently on CUDA-enabled GPUs
* Is structured for scalable deployment on large video datasets

The architecture avoids facial recognition or biometric identification and focuses purely on event-based counting.

---

## Architecture

Core Components:

* Detection: YOLOv8 (default: yolov8s)
* Tracking: ByteTrack (custom tuned YAML)
* Counting: Ultralytics ObjectCounter (vector intersection logic)
* Calibration: PCA-based motion analysis
* Configuration-driven runtime via `config.py`

Pipeline:

1. Optional gate calibration (offline PCA)
2. Video ingestion
3. Detection + tracking
4. Line-crossing event detection
5. Annotated video generation

---

## Project Structure

```
temple_proj/
├─ core/
│  └─ counter.py
├─ utils/
│  └─ video_io.py
├─ calibrate.py
├─ config.py
├─ custom_bytetrack.yaml
├─ main.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

Runtime artifacts (videos, logs, weights) are excluded via `.gitignore`.

---

## Installation

Using uv (recommended):

```
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

Ensure CUDA-enabled PyTorch is installed if using GPU.

---

## Usage

1. Place input video in:

```
data/input_vids/
```

2. Configure parameters in `config.py`:

   * MODEL_PATH
   * GATE_LINE (or enable AUTO_CALIBRATE)
   * thresholds

3. Run:

```
python main.py
```

Annotated output will be saved to:

```
data/output_vids/annotated_vids/
```

---

## Automatic Gate Calibration

To enable motion-based gate estimation:

Set in `config.py`:

```
AUTO_CALIBRATE = True
```

The system will:

* Analyze initial frames
* Compute dominant crowd flow using PCA
* Generate a perpendicular gate line
* Apply it before counting begins

For production stability, calibration is recommended once per camera setup.

---

## Performance

Tested on 576x768 CCTV footage:

* ~18–20 ms tracking time
* ~2–3 ms counting solution time
* ~45–50 FPS on CUDA-enabled GPU
* Stable IN/OUT directional counts

---

## Notes

* Designed for fixed camera installations.
* Optimized for entry-gate chokepoint scenarios.
* Not intended for dense open-field crowd occupancy estimation.
* Model weights are not stored in repository.

---

## Future Extensions

* ROI-based detection optimization
* CSV event logging
* Multi-camera aggregation
* Batch processing for large datasets
* Accuracy benchmarking against manual ground truth

---
