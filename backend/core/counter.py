from ultralytics import YOLO
import cv2
import numpy as np

import config
from core.gender import GenderClassifier
from core.reid_engine import ReIDEngine
from core.reid_tracker import ReIDTracker
from core.zone_filter import ZoneFilter

class TempleCounter:
    def __init__(self, camera_id="channel_1_main"):
        # We use raw YOLO predict now, not ObjectCounter
        self.detector = YOLO(config.MODEL_PATH)
        if config.DEVICE == "cuda":
            self.detector.to(config.DEVICE)

        self.reid_engine = ReIDEngine(model_name=config.REID_MODEL, device=config.DEVICE)
        self.reid_tracker = ReIDTracker(
            match_threshold=config.REID_MATCH_THRESHOLD,
            eviction_timeout=config.REID_EVICTION_TIMEOUT,
            grace_period=config.REID_GRACE_PERIOD
        )
        
        self.zone_filter = ZoneFilter(config.EXCLUSION_ZONES.get(camera_id, []))

        self.gender_classifier = GenderClassifier(
            model_path=config.GENDER_MODEL_PATH,
            required_votes=config.GENDER_REQUIRED_VOTES,
            stale_timeout=config.STALE_TRACK_TIMEOUT,
            confidence_thresh=config.GENDER_CONF_THRESH,
            device=config.DEVICE,
        )

        # Track demographics for Unique Visitors
        self.male_count = 0
        self.female_count = 0
        self.unknown_count = 0
        self._counted_genders = set() 

    def process_frame(self, frame, frame_idx):
        # Skip non-processed frames entirely — no copy, no draw
        if frame_idx % config.REID_SKIP_FRAMES != 0:
            return None, self.reid_tracker.cumulative_visitors

        annotated_frame = frame.copy()
        annotated_frame = self.zone_filter.draw_zones(annotated_frame)

        # 1. Detection
        results = self.detector.predict(
            frame, 
            classes=config.TARGET_CLASSES, 
            conf=config.CONF_THRESH, 
            iou=config.IOU_THRESH,
            verbose=False
        )
        
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self.reid_tracker.update([], frame_idx)
            return annotated_frame, self.reid_tracker.cumulative_visitors

        valid_crops = []
        valid_boxes = []

        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()

        # 2. Filtering and Cropping
        h, w = frame.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Skip if inside worker exclusion zone
            if self.zone_filter.is_excluded(cx, cy):
                continue
                
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                valid_crops.append(crop)
                valid_boxes.append((x1, y1, x2, y2))

        if not valid_crops:
            self.reid_tracker.update([], frame_idx)
            return annotated_frame, self.reid_tracker.cumulative_visitors

        # 3. Embedding Extraction
        embeddings = self.reid_engine.extract_features(valid_crops)
        
        # 4. Re-ID Tracking & Matching
        assigned_ids = self.reid_tracker.update(embeddings, frame_idx)
        
        # 5. Gender Tallying (Only fully locked predictions)
        for i, tid in enumerate(assigned_ids):
            # If not assigned (e.g. invalid embedding), skip
            if tid == -1: continue
                
            x1, y1, x2, y2 = valid_boxes[i]
            crop = valid_crops[i]
            
            # Continuously vote on gender whenever target is visible
            gender = self.gender_classifier.get_gender(tid, crop, frame_idx)
            
            if gender:
                # We only want to tally a visitor's demographic ONCE per unique visit
                if tid not in self._counted_genders:
                    self._counted_genders.add(tid)
                    if gender == "Male":
                        self.male_count += 1
                    elif gender == "Female":
                        self.female_count += 1
                    else:
                        self.unknown_count += 1
                
                # Draw Box and Label
                color = (
                    (255, 200, 0) if gender == "Male"
                    else (0, 255, 255) if gender == "Female"
                    else (200, 200, 200)
                )
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{tid} {gender}"
                cv2.putText(
                    annotated_frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )
            else:
                # Still voting, draw gray box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
                cv2.putText(
                    annotated_frame, f"#{tid} [...]",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA,
                )

        self.gender_classifier.clean_stale_tracks(frame_idx)
        
        # Micro-optimization: prevent _counted_genders from growing infinitely over weeks of uptime
        expired_genders = [tid for tid in self._counted_genders 
                           if tid not in self.reid_tracker.active_tracks 
                           and tid not in self.reid_tracker.departed_tracks]
        for tid in expired_genders:
            self._counted_genders.remove(tid)
            
        return annotated_frame, self.reid_tracker.cumulative_visitors
