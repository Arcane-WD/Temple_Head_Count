import os
import torch

# Paths
INPUT_VIDEO = "data/input_vids/PETS09-S2L2-raw.mp4"
OUTPUT_DIR = "data/output_vids"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_vids")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Calibration
AUTO_CALIBRATE = True  # If True, compute gate before running pipeline
CALIBRATION_FRAMES = 400

# Device Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config
MODEL_PATH = "yolov8s.pt"  
CONF_THRESH = 0.35
IOU_THRESH = 0.65
TARGET_CLASSES = [0]  # Person only

# Counting Gate (Auto-calibrated or manual)
# These are the coordinates you'll update after running the calibrator
GATE_LINE = [
    (511, 89),
    (448, 546)
]

# Tracker Config
TRACKER_CONFIG = "custom_bytetrack.yaml"