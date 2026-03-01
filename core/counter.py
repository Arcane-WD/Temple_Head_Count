from ultralytics import solutions
import cv2
import config
from core.gender import GenderClassifier

class TempleCounter:
    def __init__(self):
        self.counter = solutions.ObjectCounter(
            model=config.MODEL_PATH,
            region=config.GATE_LINE,
            classes=config.TARGET_CLASSES,
            conf=config.CONF_THRESH,
            iou=config.IOU_THRESH,
            tracker=config.TRACKER_CONFIG,
            show=False,
            verbose=False
        )

        if config.DEVICE == "cuda":
            self.counter.model.to(config.DEVICE)

        # Initialize Gender Module with confidence threshold
        self.gender_classifier = GenderClassifier(
            model_path=config.GENDER_MODEL_PATH,
            required_votes=config.GENDER_REQUIRED_VOTES,
            stale_timeout=config.STALE_TRACK_TIMEOUT,
            confidence_thresh=config.GENDER_CONF_THRESH, # Added from config
            device=config.DEVICE
        )

    def process_frame(self, frame, frame_idx):
        results = self.counter(frame)

        if isinstance(results, list):
            res = results[0]
            # Use .copy() to avoid modifying the original frame in memory
            annotated_frame = res.plot_im.copy()
        else:
            res = results
            annotated_frame = res.plot_im.copy()

        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                h_frame, w_frame = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)

                crop = frame[y1:y2, x1:x2]
                
                gender = self.gender_classifier.get_gender(track_id, crop, frame_idx)

                if gender:
                    # Determine color: Yellow (Female), Blue (Male), Gray (Unknown)
                    if gender == "Female":
                        color = (0, 255, 255)
                    elif gender == "Male":
                        color = (255, 200, 0)
                    else: # "Unknown"
                        color = (200, 200, 200)

                    cv2.putText(
                        annotated_frame,
                        gender,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

        # Prevent memory leaks
        self.gender_classifier.clean_stale_tracks(frame_idx)

        return annotated_frame, self.counter.in_count, self.counter.out_count