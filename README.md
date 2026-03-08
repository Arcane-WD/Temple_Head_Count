# Temple Head Count System

A modular, GPU-accelerated computer vision pipeline for automated entry/exit analytics and demographic classification, exposed as a real-time analytics web dashboard.

The system combines YOLO-based person detection, ByteTrack multi-object tracking, dynamic gate calibration, ONNX-based gender classification, and entry-event-anchored demographic counting — all served through a FastAPI backend and visualized in a Next.js frontend.

Designed for fixed CCTV installations monitoring physical entry gates such as temple doorways, stadium entrances, or turnstiles.

---

## System Architecture

```
temple_proj/
├─ backend/              Python pipeline + FastAPI API server
│  ├─ core/
│  │  ├─ counter.py      ObjectCounter wrapper with demographic tracking
│  │  └─ gender.py       ConvNeXt-Tiny ONNX inference with majority voting
│  ├─ utils/
│  │  └─ video_io.py     Video capture and writer utilities
│  ├─ data/
│  │  ├─ input_vids/     Input video files
│  │  └─ output_vids/    Annotated output and logs
│  ├─ calibrate.py       Kinematic motion-vector PCA gate calibration
│  ├─ config.py          Central configuration
│  ├─ custom_bytetrack.yaml
│  ├─ server.py          FastAPI server
│  └─ requirements.txt
│
├─ frontend/             Next.js analytics dashboard
│  ├─ app/
│  │  ├─ page.tsx        Main dashboard page
│  │  ├─ layout.tsx      Root layout and metadata
│  │  └─ globals.css     Design system and Tailwind setup
│  ├─ components/
│  │  ├─ StatCard.tsx    Live analytics cards (IN, OUT, Male, Female, Unknown)
│  │  ├─ FlowChart.tsx   Flow-over-time Recharts line graph
│  │  ├─ ModelSidebar.tsx   Model metadata panel
│  │  ├─ VideoPanel.tsx  Video player with high-traffic timeline markers
│  │  └─ AlertBanner.tsx    Error and warning banners
│  ├─ lib/
│  │  └─ api.ts          Type-safe API client with SSE subscription
│  └─ package.json
│
├─ main.py               Standalone CLI runner (no server required)
├─ README.md
└─ .gitignore
```

Model weights (`.pt`, `.onnx`) and runtime video data are excluded from version control.

---

## Core Components

| Component | Technology | Notes |
|---|---|---|
| Person Detection | YOLOv8s | Configured via `config.py` |
| Multi-Object Tracking | ByteTrack | Custom tracker config |
| Directional Counting | Ultralytics ObjectCounter | Entry/exit line-crossing events |
| Demographic Counting | Custom centroid tracking | Anchored to ObjectCounter `in_count` delta |
| Gender Classification | ConvNeXt-Tiny (ONNX) | 82.44% accuracy on PA-100K |
| Gate Calibration | Kinematic PCA | Motion-vector-based, auto-bypass on short clips |
| Backend API | FastAPI + SSE | Real-time frame streaming to frontend |
| Frontend | Next.js + Tailwind CSS v4 + Recharts | Dark-mode analytics dashboard |

---

## Backend API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/videos` | List available `.mp4` files |
| POST | `/api/upload` | Upload a new video to the input directory |
| POST | `/api/process` | Start pipeline for a given filename, returns `job_id` |
| GET | `/api/status/{job_id}` | Server-Sent Events stream (frame, counts, demographics) |
| GET | `/api/results/{job_id}` | Final analytics and timeline JSON |
| GET | `/api/video/{filename}` | Serve processed or input video for playback |

---

## Pipeline Flow

1. Video selection (via dashboard upload or local file placement)
2. Frame count validation
3. Optional kinematic gate calibration
4. Frame-by-frame: detection, tracking, line-crossing detection
5. Per-entry demographic classification with majority-vote locking
6. Annotated video write and H.264 re-encode for browser playback
7. Timeline and final analytics returned to dashboard

### Demographic Counting Logic

Gender demographics are counted **only on entry** — a track is tallied at most once, only after it has both:
- Triggered a line-crossing event in the IN direction (verified by centroid sign-change + ObjectCounter `in_count` delta)
- Accumulated enough inference votes to resolve a confident classification

---

## Installation

### Backend (Python venv)

```
cd backend
# Activate virtual environment first:
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
python server.py
```

Backend runs on `http://localhost:8000`.

Requires CUDA-enabled PyTorch for GPU acceleration. ONNX Runtime with CUDA support enables GPU-accelerated gender classification.

### Frontend (separate terminal, no venv)

```
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`.

---

## CLI Usage (No Server)

For offline standalone use without the web dashboard:

```
python main.py
```

Select the desired input video when prompted. Output is saved to `data/output_vids/annotated_vids/`.

---

## Configuration

All runtime parameters are in `backend/config.py`.

### Detection and Tracking

| Key | Description |
|---|---|
| `MODEL_PATH` | YOLO model weight file |
| `CONF_THRESH` | Detection confidence threshold |
| `IOU_THRESH` | NMS IoU threshold |
| `TARGET_CLASSES` | Object class indices to track (default: `[0]` for persons) |
| `TRACKER_CONFIG` | ByteTrack configuration file |
| `GATE_LINE` | Manual gate line coordinates (overridden by auto-calibration) |

### Calibration

| Key | Description |
|---|---|
| `AUTO_CALIBRATE` | Enable automatic gate calibration |
| `MAX_CALIBRATION_FRAMES` | Upper frame limit for calibration analysis |
| `CALIBRATION_FRACTION` | Fraction of video used for calibration |
| `MIN_FRAMES_FOR_CALIBRATION` | Minimum frames required to attempt calibration |

### Gender Classification

| Key | Description |
|---|---|
| `GENDER_MODEL_PATH` | Path to the ONNX gender classification model |
| `GENDER_REQUIRED_VOTES` | Inference frames before locking a classification |
| `GENDER_CONF_THRESH` | Minimum confidence for a valid classification |
| `STALE_TRACK_TIMEOUT` | Frames before an unseen track is evicted from cache |

---

## Error Handling

The pipeline surfaces three structured error conditions to the dashboard:

| Code | Cause | User Message |
|---|---|---|
| `LOW_FRAME_COUNT` | Video too short for calibration | Clip is too short for the system to analyze movement patterns. |
| `CHAOTIC_MOTION` | Motion ratio below rejection threshold | Crowd movement is too random for reliable line placement. |
| `CHAOTIC_MOTION_WARN` | Motion ratio below warning threshold | Results may be less accurate. Verify the counting line manually. |

---

## Performance

Tested on 576x768 CCTV footage:

- 14-20 ms tracking time per frame
- 45-50 FPS on RTX 3050 (CUDA)
- Stable IN/OUT directional counts
- Gender classification adds minimal latency due to ONNX inference and per-track vote caching

---

## Design Principles

- Modular backend with clean separation of detection, tracking, classification, and calibration
- Entry-event-anchored demographic counting prevents false positives
- Graceful degradation when optional components (ONNX model, ffmpeg) are unavailable
- Structured error propagation from pipeline to UI with layman-friendly explanations
- No facial recognition or biometric data storage

---

## Limitations

- Designed for fixed-camera, single-gate scenarios
- Not optimized for open-field crowd density estimation
- Gender classification accuracy depends on crop quality and camera angle
- Model weights are not included in the repository

---

## Roadmap

- JSON analytics export per session
- Batch processing for large archive datasets
- Multi-camera feed aggregation
- Ground-truth benchmarking harness
