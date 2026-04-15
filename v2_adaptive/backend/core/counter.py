from ultralytics import YOLO
import cv2
import numpy as np

import config
from core.gender import GenderClassifier
from core.reid_engine import ReIDEngine
from core.reid_tracker import ReIDTracker
from core.zone_filter import ZoneFilter

class TempleCounter:
    def __init__(self, camera_id="channel_1_main", override_device=None):
        device = override_device if override_device else config.DEVICE
        
        # Load specific tracking model (nano instead of small)
        self.detector = YOLO(config.MODEL_PATH)
        if device == "cuda":
            self.detector.to(device)

        self.reid_engine = ReIDEngine(model_name=config.REID_MODEL, device=device)
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
            device=device,
        )

        # Track demographics for Unique Visitors
        self.male_count = 0
        self.female_count = 0
        self.unknown_count = 0
        self._counted_genders = set() 
        self.healed_switches = 0
        
        # ByteTrack State
        # tid -> {"global_id": int, "last_conf": float, "last_area": float}
        self.track_data = {}

    def process_frame(self, frame, frame_idx):
        annotated_frame = frame.copy()
        annotated_frame = self.zone_filter.draw_zones(annotated_frame)

        # 1. ALWAYS Run Tracking (ByteTrack)
        results = self.detector.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=config.TARGET_CLASSES, 
            conf=config.CONF_THRESH, 
            iou=config.IOU_THRESH,
            verbose=False
        )
        
        boxes_data = results[0].boxes
        if boxes_data is None or len(boxes_data) == 0 or boxes_data.id is None:
            self.reid_tracker.cleanup(frame_idx)
            return annotated_frame, self.reid_tracker.cumulative_visitors

        boxes = boxes_data.xyxy.cpu().numpy()
        track_ids = boxes_data.id.cpu().numpy().astype(int)
        confs = boxes_data.conf.cpu().numpy()

        # 2. Filtering and Action Queuing
        h, w = frame.shape[:2]
        reid_queue = []
        valid_detections = []

        for i, box in enumerate(boxes):
            tid = track_ids[i]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if self.zone_filter.is_excluded(cx, cy):
                continue
                
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue

            conf = confs[i]
            area = (x2 - x1) * (y2 - y1)
            trigger_reason = None
            
            if tid not in self.track_data:
                trigger_reason = "new_track"
                self.track_data[tid] = {"global_id": None, "last_conf": conf, "last_area": area}
            else:
                prev_data = self.track_data[tid]
                
                # Triggers for partial occlusion / drift
                if prev_data["last_conf"] - conf > 0.2:
                    trigger_reason = "conf_drop"
                elif prev_data["last_area"] > 0 and abs(area - prev_data["last_area"]) / prev_data["last_area"] > 0.3:
                    trigger_reason = "size_change"
                
                # Update state
                prev_data["last_conf"] = conf
                prev_data["last_area"] = area
                
                # Update Vault timestamp
                gid = prev_data["global_id"]
                if gid is not None:
                    self.reid_tracker.update_last_seen(gid, frame_idx, [x1, y1, x2, y2])

            if trigger_reason:
                reid_queue.append({
                    "track_id": tid,
                    "crop": crop,
                    "box": [x1, y1, x2, y2],
                    "conf": conf,
                    "area": area,
                    "reason": trigger_reason
                })
                
            valid_detections.append({
                "track_id": tid,
                "crop": crop,
                "box": [x1, y1, x2, y2]
            })

        # 3. Intelligent Re-ID Execution
        if frame_idx % config.DETECTION_INTERVAL == 0 and reid_queue:
            def get_priority(item):
                p_map = {"new_track": 3, "conf_drop": 2, "size_change": 1}
                return (p_map.get(item["reason"], 0), item["conf"], item["area"])
                
            reid_queue.sort(key=get_priority, reverse=True)
            reid_queue = reid_queue[:config.MAX_REID_PER_FRAME]
            
            crops_to_extract = [item["crop"] for item in reid_queue]
            embeddings = self.reid_engine.extract_features(crops_to_extract)
            
            for i, emb in enumerate(embeddings):
                tid = reid_queue[i]["track_id"]
                box = reid_queue[i]["box"]
                
                gid = self.reid_tracker.match_identity(emb, frame_idx)
                if gid is None:
                    gid = self.reid_tracker.register_identity(emb, frame_idx, box)
                else:
                    # Track fragmentation healed!
                    if self.track_data[tid].get("global_id") != gid:
                        self.healed_switches += 1
                
                self.track_data[tid]["global_id"] = gid

        # 4. Gender Tallying and Rendering
        for det in valid_detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = det["box"]
            crop = det["crop"]
            
            gid = self.track_data.get(tid, {}).get("global_id")
            
            if gid is not None:
                # Use Global ID for gender to preserve identity coherence
                gender = self.gender_classifier.get_gender(gid, crop, frame_idx)
                
                if gender:
                    if gid not in self._counted_genders:
                        self._counted_genders.add(gid)
                        if gender == "Male": self.male_count += 1
                        elif gender == "Female": self.female_count += 1
                        else: self.unknown_count += 1
                    
                    color = (255, 200, 0) if gender == "Male" else (0, 255, 255) if gender == "Female" else (200, 200, 200)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"ID:{gid} {gender}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100, 100, 100), 2)
                    cv2.putText(annotated_frame, f"ID:{gid} [...]", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
            else:
                # Still initializing global ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (50, 50, 50), 2)
                cv2.putText(annotated_frame, f"WAIT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

        # 5. Maintenance
        self.gender_classifier.clean_stale_tracks(frame_idx)
        self.reid_tracker.cleanup(frame_idx)
        
        # Micro-optimization: Cleanup counted genders and track data
        active_gids = set(self.reid_tracker.active_identities.keys()) | set(self.reid_tracker.departed_identities.keys())
        self._counted_genders = {g for g in self._counted_genders if g in active_gids}
        
        current_tids = {det["track_id"] for det in valid_detections}
        stale_tids = [t for t in self.track_data if t not in current_tids]
        for t in stale_tids:
            # We can pop t immediately since ByteTrack assigns a new ID if completely lost
            self.track_data.pop(t, None)

        return annotated_frame, self.reid_tracker.cumulative_visitors
