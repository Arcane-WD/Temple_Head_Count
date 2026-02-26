import cv2
import numpy as np
from ultralytics import YOLO
import config

def auto_calibrate_gate(video_path, frames_to_analyze=400):
    print(f"Starting calibration on first {frames_to_analyze} frames of {video_path}...")
    
    model = YOLO(config.MODEL_PATH)
    model.to(config.DEVICE)

    cap = cv2.VideoCapture(video_path)
    
    # 🐛 FIX APPLIED: Grab width and height BEFORE releasing the capture
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    motion_points = []
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

            for box in boxes:
                x1, y1, x2, y2 = box
                anchor_x = int((x1 + x2) / 2)
                anchor_y = int(y2)
                motion_points.append([anchor_x, anchor_y])

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Analyzed {frame_count}/{frames_to_analyze} frames...")

    cap.release()

    if len(motion_points) < 20:
        print("❌ Not enough motion data detected to calibrate.")
        return None

    # --- PCA Math ---
    data = np.array(motion_points, dtype=np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(data, np.empty((0)))

    center = (int(mean[0, 0]), int(mean[0, 1]))

    # Perpendicular vector (gate line direction)
    gate_vector = eigenvectors[1]

    line_length = min(width, height) * 0.4

    # Calculate raw points
    raw_pt1_x = int(center[0] - gate_vector[0] * line_length)
    raw_pt1_y = int(center[1] - gate_vector[1] * line_length)
    raw_pt2_x = int(center[0] + gate_vector[0] * line_length)
    raw_pt2_y = int(center[1] + gate_vector[1] * line_length)

    # 🛡️ ENHANCEMENT: Clamp to frame dimensions to avoid Ultralytics out-of-bounds errors
    pt1 = (max(0, min(width, raw_pt1_x)), max(0, min(height, raw_pt1_y)))
    pt2 = (max(0, min(width, raw_pt2_x)), max(0, min(height, raw_pt2_y)))

    return [pt1, pt2]

if __name__ == "__main__":
    auto_calibrate_gate(config.INPUT_VIDEO)