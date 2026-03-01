import os
import torch

# Paths
INPUT_DIR = "data/input_vids"  # Normalized to directory
OUTPUT_DIR = "data/output_vids"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_vids")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODEL_DIR = "models"

# Calibration Constants
AUTO_CALIBRATE = True  
MAX_CALIBRATION_FRAMES = 400
CALIBRATION_FRACTION = 0.8
MIN_FRAMES_FOR_CALIBRATION = 300  

# Device Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tracking & Detection Config
MODEL_PATH = "yolov8s.pt"  
CONF_THRESH = 0.35
IOU_THRESH = 0.65
TARGET_CLASSES = [0]  
TRACKER_CONFIG = "custom_bytetrack.yaml"
GATE_LINE = [(89, 511), (546, 448)]

# --- GENDER MODEL CONFIG ---
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, "gender_classifier.onnx")
GENDER_REQUIRED_VOTES = 5
GENDER_CONF_THRESH = 0.65
STALE_TRACK_TIMEOUT = 100