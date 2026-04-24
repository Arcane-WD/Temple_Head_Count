import os
import torch

# Dynamically resolve the absolute path to the backend/ folder
# so all model/data paths work regardless of where python is invoked from.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))

# Global Shared Assets path
GLOBAL_ASSETS = os.path.join(PROJECT_ROOT, "global_assets")

# Data Paths
INPUT_DIR = os.path.join(GLOBAL_ASSETS, "input_vids")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "data", "output_vids")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_vids")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Device Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DETECTION CONFIG ---
MODEL_PATH = os.path.join(GLOBAL_ASSETS, "models", "yolov8n.pt")
CONF_THRESH = 0.50          # Increased from 0.35 to ignore low-confidence noise
IOU_THRESH = 0.65
TARGET_CLASSES = [0]

# --- RE-ID CONFIG ---
REID_MODEL = "osnet_x1_0"
DETECTION_INTERVAL = 3       # Frames between event triggers check
MAX_REID_PER_FRAME = 5       # Max people to extract embeddings for on trigger
REID_MATCH_THRESHOLD = 0.55  # Lowered from 0.88 to prevent track splintering and identity explosion
REID_EVICTION_TIMEOUT = 500
REID_GRACE_PERIOD = 7500

# Worker Exclusion Zones
EXCLUSION_ZONES = {
    "channel_1_main": [],
    "channel_5_main": [],
}

# --- GENDER MODEL CONFIG ---
GENDER_MODEL_PATH = os.path.join(GLOBAL_ASSETS, "models", "convnext_tiny_gender_82.44acc.onnx")
GENDER_REQUIRED_VOTES = 5
GENDER_CONF_THRESH = 0.65
STALE_TRACK_TIMEOUT = 100
