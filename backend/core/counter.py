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
            verbose=False,
        )

        if config.DEVICE == "cuda":
            self.counter.model.to(config.DEVICE)

        self.gender_classifier = GenderClassifier(
            model_path=config.GENDER_MODEL_PATH,
            required_votes=config.GENDER_REQUIRED_VOTES,
            stale_timeout=config.STALE_TRACK_TIMEOUT,
            confidence_thresh=config.GENDER_CONF_THRESH,
            device=config.DEVICE,
        )

        # Demographic accumulators
        self.male_count = 0
        self.female_count = 0
        self.unknown_count = 0

        # State for entry-anchored demographic counting
        self._prev_in_count = 0
        self._pending_gender = set()    # entered but gender not yet resolved
        self._counted_genders = set()   # already tallied

    def process_frame(self, frame, frame_idx):
        res = self.counter(frame)
        annotated_frame = res.plot_im.copy()

        boxes = self.counter.boxes
        ids = self.counter.track_ids

        current_ids = set()

        if boxes is not None and ids is not None and len(boxes) > 0 and len(ids) == len(boxes):
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()

            # Build per-track info for this frame
            track_info = []
            for box, tid in zip(boxes, ids):
                tid = int(tid)
                current_ids.add(tid)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                track_info.append((tid, cx, cy, x1, y1, x2, y2))

            # -----------------------------------------------------------
            # When in_count increases, mark the N closest-to-line tracks
            # that haven't been counted yet as "entered".
            # This uses ObjectCounter as the sole authority on who entered.
            # -----------------------------------------------------------
            new_entries = self.counter.in_count - self._prev_in_count
            if new_entries > 0:
                p1, p2 = config.GATE_LINE
                lmx, lmy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

                candidates = []
                for tid, cx, cy, *_ in track_info:
                    if tid not in self._counted_genders and tid not in self._pending_gender:
                        dist = (cx - lmx) ** 2 + (cy - lmy) ** 2
                        candidates.append((dist, tid))

                candidates.sort()  # closest first
                for i in range(min(new_entries, len(candidates))):
                    self._pending_gender.add(candidates[i][1])

            self._prev_in_count = self.counter.in_count

            # -----------------------------------------------------------
            # Gender inference + tally (for ALL visible tracks)
            # -----------------------------------------------------------
            for tid, cx, cy, x1, y1, x2, y2 in track_info:
                h, w = frame.shape[:2]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                gender = None
                if crop.size > 0:
                    gender = self.gender_classifier.get_gender(tid, crop, frame_idx)

                # Tally only if this track entered AND has a resolved gender
                if gender and tid in self._pending_gender and tid not in self._counted_genders:
                    self._counted_genders.add(tid)
                    self._pending_gender.discard(tid)
                    if gender == "Male":
                        self.male_count += 1
                    elif gender == "Female":
                        self.female_count += 1
                    else:
                        self.unknown_count += 1

                # Draw gender label on bounding box regardless
                if gender:
                    color = (
                        (255, 200, 0) if gender == "Male"
                        else (0, 255, 255) if gender == "Female"
                        else (200, 200, 200)
                    )
                    cv2.putText(
                        annotated_frame, gender,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
                    )

        # Evict lost tracks — force-tally pending ones as Unknown
        for t in list(self._pending_gender):
            if t not in current_ids:
                self._pending_gender.discard(t)
                if t not in self._counted_genders:
                    self._counted_genders.add(t)
                    self.unknown_count += 1

        self.gender_classifier.clean_stale_tracks(frame_idx)
        return annotated_frame, self.counter.in_count, self.counter.out_count
