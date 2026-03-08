import cv2
import numpy as np
from ultralytics import YOLO
import config

def auto_calibrate_gate(video_path, frames_to_analyze=400):
    """
    Returns a dict with:
      - 'status': 'success' | 'low_frame_count' | 'chaotic_motion' | 'chaotic_motion_warn'
      - 'gate_line': [(x1,y1), (x2,y2)] or None
      - 'ratio': float or None
      - 'vectors_found': int
      - 'message': str (technical)
      - 'layman': str (user-friendly)
    """
    MIN_VECTOR_THRESHOLD = 100
    REJECT_RATIO = 1.5
    WARN_RATIO = 3.0

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

    cap.release()

    # FAIL-SAFE 1: Insufficient Data
    if len(motion_vectors) < MIN_VECTOR_THRESHOLD:
        return {
            "status": "chaotic_motion",
            "gate_line": None,
            "ratio": None,
            "vectors_found": len(motion_vectors),
            "message": f"Insufficient motion data. Found {len(motion_vectors)} vectors, required {MIN_VECTOR_THRESHOLD}.",
            "layman": "Not enough movement was detected in the video for the system to determine a reliable counting line. Try using a longer video with more visible foot traffic."
        }

    spatial_data = np.array(anchor_points, dtype=np.float32)
    spatial_mean = np.mean(spatial_data, axis=0)
    center = (int(spatial_mean[0]), int(spatial_mean[1]))

    motion_data = np.array(motion_vectors, dtype=np.float32)
    _, eigenvectors, eigenvalues = cv2.PCACompute2(motion_data, np.empty((0)))

    flow_dir = eigenvectors[0] 
    ratio = eigenvalues[0][0] / (eigenvalues[1][0] + 1e-6)

    # FAIL-SAFE 2: Chaotic Crowd Rejection
    if ratio < REJECT_RATIO:
        return {
            "status": "chaotic_motion",
            "gate_line": None,
            "ratio": float(ratio),
            "vectors_found": len(motion_vectors),
            "message": f"Crowd motion is too chaotic. Ratio {ratio:.2f} is below rejection threshold {REJECT_RATIO}.",
            "layman": "The crowd movement in this video is too random for the computer to draw a reliable counting line. Try a video with a clearer flow direction."
        }

    gate_vector = np.array([-flow_dir[1], flow_dir[0]])
    line_length = min(width, height) * 0.45

    raw_pt1_x = int(center[0] - gate_vector[0] * line_length)
    raw_pt1_y = int(center[1] - gate_vector[1] * line_length)
    raw_pt2_x = int(center[0] + gate_vector[0] * line_length)
    raw_pt2_y = int(center[1] + gate_vector[1] * line_length)

    pt1 = (max(0, min(width, raw_pt1_x)), max(0, min(height, raw_pt1_y)))
    pt2 = (max(0, min(width, raw_pt2_x)), max(0, min(height, raw_pt2_y)))

    result = {
        "status": "success",
        "gate_line": [pt1, pt2],
        "ratio": float(ratio),
        "vectors_found": len(motion_vectors),
        "message": f"Calibration successful. Ratio: {ratio:.2f}",
        "layman": "The system successfully determined the optimal counting line based on crowd flow patterns."
    }

    if ratio < WARN_RATIO:
        result["status"] = "chaotic_motion_warn"
        result["layman"] = "The crowd movement is somewhat scattered. Results may be less accurate than usual. Consider verifying the counting line manually."

    return result

if __name__ == "__main__":
    result = auto_calibrate_gate("data/input_vids/InpVid1.mp4")
    print(result)
