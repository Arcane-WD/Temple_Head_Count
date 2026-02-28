import cv2
import numpy as np
from ultralytics import YOLO
import config

def auto_calibrate_gate(video_path, frames_to_analyze=400):
    print(f"Starting Strict Kinematic Calibration on {video_path}...")
    
    # --- PRODUCTION THRESHOLDS ---
    MIN_VECTOR_THRESHOLD = 100  # Require a statistically significant sample size
    REJECT_RATIO = 1.5          # If variance ratio is below this, reject completely
    WARN_RATIO = 3.0            # If variance ratio is below this, warn the user

    model = YOLO(config.MODEL_PATH)
    if config.DEVICE == "cuda":
        model.to("cuda")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    anchor_points = []
    motion_vectors = []
    track_history = {} 
    
    frame_count = 0

    while frame_count < frames_to_analyze:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=config.TARGET_CLASSES,
            conf=config.CONF_THRESH,
            iou=config.IOU_THRESH,
            verbose=False
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                anchor_x = int((x1 + x2) / 2)
                anchor_y = int(y2)
                
                anchor_points.append([anchor_x, anchor_y])
                
                if track_id in track_history:
                    prev_x, prev_y = track_history[track_id]
                    dx = anchor_x - prev_x
                    dy = anchor_y - prev_y
                    
                    if abs(dx) > 2 or abs(dy) > 2:
                        motion_vectors.append([dx, dy])
                
                track_history[track_id] = (anchor_x, anchor_y)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Analyzed {frame_count}/{frames_to_analyze} frames...")

    cap.release()

    # 🛡️ FAIL-SAFE 1: Insufficient Data
    if len(motion_vectors) < MIN_VECTOR_THRESHOLD:
        print("\n" + "❌ "*15)
        print(f"CALIBRATION FAILED: Insufficient motion data.")
        print(f"Found {len(motion_vectors)} vectors. Required: {MIN_VECTOR_THRESHOLD}.")
        print("ACTION: Do not update config.py. Use your manual GATE_LINE.")
        print("❌ "*15)
        return None

    spatial_data = np.array(anchor_points, dtype=np.float32)
    spatial_mean = np.mean(spatial_data, axis=0)
    center = (int(spatial_mean[0]), int(spatial_mean[1]))

    motion_data = np.array(motion_vectors, dtype=np.float32)
    _, eigenvectors, eigenvalues = cv2.PCACompute2(motion_data, np.empty((0)))

    flow_dir = eigenvectors[0] 
    ratio = eigenvalues[0][0] / (eigenvalues[1][0] + 1e-6)
    
    print(f"\nMotion Confidence Ratio: {ratio:.2f}")

    # 🛡️ FAIL-SAFE 2: Chaotic Crowd Rejection
    if ratio < REJECT_RATIO:
        print("\n" + "❌ "*15)
        print("CALIBRATION FAILED: Crowd motion is too chaotic/multi-directional.")
        print(f"Ratio {ratio:.2f} is below strict rejection threshold of {REJECT_RATIO}.")
        print("ACTION: Do not update config.py. Use your manual GATE_LINE.")
        print("❌ "*15)
        return None
    elif ratio < WARN_RATIO:
        print("⚠️ WARNING: Motion is somewhat dispersed. Verify the output line visually before trusting it.")

    gate_vector = np.array([-flow_dir[1], flow_dir[0]])
    line_length = min(width, height) * 0.45

    raw_pt1_x = int(center[0] - gate_vector[0] * line_length)
    raw_pt1_y = int(center[1] - gate_vector[1] * line_length)
    raw_pt2_x = int(center[0] + gate_vector[0] * line_length)
    raw_pt2_y = int(center[1] + gate_vector[1] * line_length)

    pt1 = (max(0, min(width, raw_pt1_x)), max(0, min(height, raw_pt1_y)))
    pt2 = (max(0, min(width, raw_pt2_x)), max(0, min(height, raw_pt2_y)))

    print("\n" + "="*40)
    print("✅ KINEMATIC CALIBRATION SUCCESSFUL")
    print("="*40)
    print("Paste this into config.py:\n")
    print(f"GATE_LINE = [\n    {pt1},\n    {pt2}\n]")
    print("="*40)

    return [pt1, pt2]

if __name__ == "__main__":
    auto_calibrate_gate(config.INPUT_VIDEO)