import torch
import numpy as np
import torchreid
from torchreid.utils import FeatureExtractor


class ReIDEngine:
    def __init__(self, model_name: str = "osnet_x1_0", device: str = "cpu"):
        self.device = device
        print(f"Loading Re-ID Model '{model_name}'. This may download weights on first run...")
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=self.device,
        )
        print("Re-ID Engine loaded successfully.")

    def extract_features(self, crops: list) -> np.ndarray:
        """
        Extracts L2-normalized 512-dim Re-ID embeddings from a list of BGR numpy crops.
        Returns: np.ndarray of shape (N, 512), or empty (0,) array if no crops.
        """
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        features = self.extractor(crops)              # returns torch.Tensor on CPU
        features = features.detach().cpu().numpy()    # (N, 512)

        # L2-normalize each row (critical for cosine distance to work correctly)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero on degenerate crops
        features = features / norms

        return features
