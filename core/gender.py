import numpy as np
import cv2
import onnxruntime as ort

class GenderClassifier:
    def __init__(self, model_path: str, required_votes: int = 5, stale_timeout: int = 100, confidence_thresh: float = 0.5, device: str = "cpu"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            
            # 🚨 FIX 1: Explicit Output Shape Logging
            output_details = self.session.get_outputs()[0]
            self.output_name = output_details.name
            print(f"✅ Gender ONNX Loaded. Expected Output Shape: {output_details.shape}")
            print("⚠️ NOTE: Ensure output index 0=Female, 1=Male. If different, update GenderClassifier logic.")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load ONNX model at {model_path}. Gender classification disabled.")
            self.session = None

        self.required_votes = required_votes
        self.stale_timeout = stale_timeout
        self.confidence_thresh = confidence_thresh # 🚨 FIX 2: Confidence Threshold
        
        # State Management
        self.track_cache = {}       
        self.track_buffer = {}      
        self.track_last_seen = {}   

    def get_gender(self, track_id: int, crop: np.ndarray, frame_idx: int) -> str:
        if self.session is None:
            return None

        self.track_last_seen[track_id] = frame_idx

        if track_id in self.track_cache:
            return self.track_cache[track_id]

        if track_id not in self.track_buffer:
            self.track_buffer[track_id] = []

        logits = self._infer_probs(crop)
        if logits is not None:
            self.track_buffer[track_id].append(logits)

        # Lock in prediction
        if len(self.track_buffer[track_id]) >= self.required_votes:
            avg_probs = np.mean(self.track_buffer[track_id], axis=0)
            
            # 🚨 FIX 1 & 2: Explicit mapping and confidence check
            # We will explicitly map this when we export the Torchreid model
            female_prob = avg_probs[0]
            male_prob = avg_probs[1]

            max_prob = max(female_prob, male_prob)

            if max_prob < self.confidence_thresh:
                gender = "Unknown"
            else:
                gender = "Male" if male_prob > female_prob else "Female"
            
            self.track_cache[track_id] = gender
            del self.track_buffer[track_id] 
            return gender

        return None

    def clean_stale_tracks(self, current_frame: int):
        stale_ids = [
            tid for tid, last_seen in self.track_last_seen.items() 
            if (current_frame - last_seen) > self.stale_timeout
        ]
        for tid in stale_ids:
            self.track_cache.pop(tid, None)
            self.track_buffer.pop(tid, None)
            self.track_last_seen.pop(tid, None)

    def _infer_probs(self, crop: np.ndarray) -> np.ndarray:
        processed = self._preprocess(crop)
        if processed is None:
            return None

        outputs = self.session.run([self.output_name], {self.input_name: processed})
        logits = outputs[0].flatten()
        if logits.shape[0] != 2:
            print("⚠️ Unexpected gender output shape:", logits.shape)
            return None

        # Apply softmax (if model does NOT already include softmax)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        return probs

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return None

        # 🚨 FIX 4: 85% Crop to preserve face/chin on steep CCTV angles
        crop = crop[: int(h * 0.85), :]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (224, 224))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop = crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std

        crop = np.transpose(crop, (2, 0, 1))
        crop = np.expand_dims(crop, axis=0)

        return crop