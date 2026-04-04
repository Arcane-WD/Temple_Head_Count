# Temple Head Count System

A high-performance, GPU-accelerated computer vision pipeline for automated visitor counting and demographic classification at temple entrances using CCTV footage.

The system uses **YOLOv8** for detection, **OSNet Re-ID** embeddings for identity tracking, and **ConvNeXt-Tiny** for gender classification. It is specifically optimized to handle "non-flowing" crowds (people sitting, praying, or milling) without traditional gate-line logic.

---

## Architecture & Performance

The system is engineered for maximum throughput by decoupling heavy ML inference from I/O and reporting.

-   **Decoupled ML Loop**: The main processing loop runs at full GPU speed, isolated from disk I/O and network reporting.
-   **Threaded Reporting**: Progress updates (SSE) and timeline logging run on a separate heartbeat thread to prevent GIL contention.
-   **Async Video Writer**: Annotated frames are written via a threaded producer-consumer queue.
-   **Timelapse Output**: The system generates a high-efficiency timelapse video (`fps / SKIP_FRAMES`), drastically reducing disk usage and encoding time.
-   **Zero-Copy Skip-Frames**: Detections are performed every $k$ frames; skipped frames skip all memory allocations, copies, and drawing operations.

```
temple_proj/
├── backend/                     Python ML pipeline + FastAPI server
│   ├── core/
│   │   ├── counter.py           Orchestrator: YOLO → ZoneFilter → ReID → Gender
│   │   ├── reid_engine.py       OSNet-x1_0 feature extraction (torchreid)
│   │   ├── reid_tracker.py      Hungarian matching, EMA updates, grace period
│   │   ├── zone_filter.py       Polygon-based worker exclusion zones
│   │   └── gender.py            ConvNeXt-Tiny ONNX gender classifier
│   ├── utils/
│   │   └── video_io.py          Video capture/writer helpers
│   ├── data/
│   │   ├── input_vids/          Source video files (.mp4)
│   │   └── output_vids/         Annotated output and logs
│   ├── config.py                All thresholds, model paths, zone definitions
│   ├── server.py                Highly optimized FastAPI server (Async I/O + Threaded Reporting)
│   ├── main_cli.py              Standalone CLI runner (no server needed)
│   ├── requirements.txt
│   ├── yolov8s.pt               YOLO weights (not committed)
│   └── convnext_tiny_*.onnx     Gender model (not committed)
│
├── frontend/                    Next.js analytics dashboard
│   ├── app/                     Page routes and global styles
│   ├── components/              StatCard, FlowChart, VideoPanel, etc.
│   ├── lib/                     API client and TypeScript types
│   └── package.json
│
├── aggregate_code.py            Codebase export utility
├── test_run.py                  Diagnostic script (PCA, similarity plots)
└── README.md
```

---

## Pipeline Flow

```
Video Frame (every 5th frame)
    │
    ▼
YOLOv8s Detection ─── conf > 0.50, class 0 (person)
    │
    ▼
Zone Filter ─── skip detections inside worker exclusion polygons
    │
    ▼
OSNet-x1_0 Re-ID ─── 512-dim L2-normalized embeddings
    │
    ▼
Hungarian Matching ─── cosine distance, θ = 0.65
    ├── Matched → update active track (EMA embedding)
    ├── Matched (departed) → re-activate (no new count)
    └── Unmatched → new visitor (+1 cumulative)
    │
    ▼
ConvNeXt-Tiny Gender ─── majority vote, locked after 5 frames
    │
    ▼
Terminal / Dashboard Output
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- `ffmpeg` (required for browser-compatible video re-encoding)
- YOLO weights (`yolov8s.pt`) in `backend/`
- Gender ONNX model (`convnext_tiny_gender_82.44acc.onnx`) in `backend/`

### Install Dependencies

```bash
# Backend (activate your venv first)
cd backend
uv pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### CLI Testing (No Server)

```bash
cd backend
python main_cli.py
```

### Full Stack (Server + Dashboard)

**Terminal 1 (Backend):**
```bash
cd backend
python server.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

---

## Configuration

Settings are centralized in [`backend/config.py`](backend/config.py):

| Parameter | Default | Description |
|---|---|---|
| `CONF_THRESH` | `0.50` | YOLO detection confidence |
| `REID_MODEL` | `osnet_x1_0` | Re-ID model variant |
| `REID_SKIP_FRAMES` | `5` | Process every k-th frame |
| `REID_MATCH_THRESHOLD` | `0.65` | Cosine similarity cutoff |
| `REID_EVICTION_TIMEOUT` | `500` | Frames before evicting unseen tracks |
| `REID_GRACE_PERIOD` | `7500` | Frames before forgetting departed tracks |
| `EXCLUSION_ZONES` | `{}` | Per-camera worker exclusion polygons |
| `GENDER_REQUIRED_VOTES` | `5` | Frames before locking gender |

---

## Diagnostics

Run `test_run.py` from the project root to validate YOLO detection rates, cluster quality (PCA), and similarity distribution. This is recommended before running on large datasets to ensure the $\theta$ threshold is optimal for your camera angles.

---

## Limitations

- Single-camera counting is verified; cross-camera deduplication is in development.
- Re-entry after the grace period (~5 mins @ 25fps) counts as a new visit.
- Gender accuracy depends on visibility and resolution of the face/body.
