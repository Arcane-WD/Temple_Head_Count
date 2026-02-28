import os
import torch

# Paths
INPUT_VIDEO = "data/input_vids/InpVid"
OUTPUT_DIR = "data/output_vids"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_vids")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Calibration Constants
AUTO_CALIBRATE = True  
MAX_CALIBRATION_FRAMES = 400
CALIBRATION_FRACTION = 0.8
MIN_FRAMES_FOR_CALIBRATION = 300  # Avoid calibrating on 9-second noise

# Device Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config
MODEL_PATH = "yolov8s.pt"  
CONF_THRESH = 0.35
IOU_THRESH = 0.65
TARGET_CLASSES = [0]  # Person only

# Counting Gate (Fallback/Manual coordinates)
GATE_LINE = [
    (89, 511),
    (546,448)
]

# Tracker Config
TRACKER_CONFIG = "custom_bytetrack.yaml"