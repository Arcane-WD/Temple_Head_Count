# Temple Head Count System

A high-performance, GPU-accelerated computer vision pipeline for automated visitor counting and demographic classification at temple entrances using CCTV footage.

The system uses **YOLOv8** for detection, **OSNet Re-ID** embeddings for identity tracking, and **ConvNeXt-Tiny** for gender classification. It is specifically optimized to handle "non-flowing" crowds (people sitting, praying, or milling) without traditional gate-line logic.

---

## 🏛️ Pipeline Versions

This repository contains two parallel backends serving a single unified frontend dashboard.

| Version | Status | Description |
|---|---|---|
| **V1 (Skip-Frame)** | 🟢 Stable MVP | Traditional approach: performs heavy inference strictly every $k$ frames and uses Hungarian matching to map visitor persistence. Best for stability and baseline measurements. |
| **V2 (Adaptive)** | 🟡 In Development | High-performance approach: utilizes **YOLOv8n + ByteTrack always-on**, removing tracker drift. Expensive ML operations (Re-ID, Gender) are only selectively queued natively on targeted events (new tracks, confidence drops, occlusion warnings). |

---

## 🏗️ Architecture 

The codebase heavily decouples UI components, AI orchestration, and memory-heavy machine learning models.

```
temple_proj/
├── global_assets/               Shared heavy dependencies (ignored in git)
│   ├── input_vids/              Source video files (.mp4)
│   └── models/                  yolov8n.pt, yolov8s.pt, convnext_tiny.onnx
│
├── frontend/                    Global Next.js analytics dashboard
│
├── v1_skipframe/                [STABLE] Original skip-frame MVP pipeline
│   └── backend/                 FastAPI server + Logic
│
├── v2_adaptive/                 [IN DEVELOPMENT] High-performance hybrid tracking pipeline
│   └── backend/                 FastAPI server + Identity Vault Logic
│
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- `ffmpeg` (required for browser-compatible video re-encoding)
- Downloaded Models: Place `yolov8s.pt`, `yolov8n.pt`, and the Gender ONNX model inside `/global_assets/models/`.

### 1. Install Dependencies

```bash
# Frontend
cd frontend
npm install

# Backend (activate your venv first, required for both V1 and V2)
cd v1_skipframe/backend  # (or v2_adaptive/backend)
pip install -r requirements.txt
```

### 2. Run the Full Stack

Both versions share the exact same UI interface. You just need to run the global frontend alongside the backend version of your preference.

**Terminal 1 (The Global Frontend):**
```bash
cd frontend
npm run dev
```

**Terminal 2 (Run Stable V1 Backend):**
```bash
cd v1_skipframe/backend
python server.py
```
*(The FastAPI server will boot on `localhost:8000`. Navigate to `localhost:3000` to interact with V1).*

### 3. Testing V2 (In Development)

If you'd like to test the experimental adaptive tracker:
```bash
cd v2_adaptive/backend
python server.py
# Or run the CLI GPU diagnostic tool:
# python demo_gpu.py --video your_video.mp4
```

---

## ⚙️ Configuration

Settings for both pipelines are safely scoped inside their respective `backend/config.py` files.

| Parameter | Used By | Description |
|---|---|---|
| `REID_SKIP_FRAMES` | V1 Only | How often to process frames (e.g., 5). |
| `DETECTION_INTERVAL`| V2 Only | Frames between active trigger queuing. |
| `MAX_REID_PER_FRAME`| V2 Only | Upper limit constraint on Re-ID extractions. |
| `REID_MODEL` | Both | Re-ID model variant. |
| `GENDER_REQUIRED_VOTES` | Both | Confidence votes needed before caching gender. |

---

## 🔍 Limitations & Diagnostics

- **Cross-camera deduplication:** Single-camera counting is heavily verified; full multi-camera meshing is under active development.
- **Grace Periods:** Re-entry after the defined config grace period (~5 mins) will trigger a fresh count sequence. 
- You can run `test_run.py` to validate baseline YOLO detection speeds and verify Re-ID embedding clustering quality (PCA) on native hardware before scaling.
