import numpy as np
from scipy.spatial.distance import cdist

class ReIDTracker:
    """
    Identity Vault Model.
    Stores robust embeddings as global identities and retrieves them via pure distance search.
    This replaces the continuous Hungarian matcher which was overkill for tracking frames.
    """

    def __init__(self, match_threshold: float = 0.65,
                 eviction_timeout: int = 500, grace_period: int = 7500):
        # Cosine distance threshold (1 - similarity)
        self.max_distance = 1.0 - match_threshold
        self.eviction_timeout = eviction_timeout
        self.grace_period = grace_period

        self.next_global_id = 1
        
        # Identity Vault: global_id -> { "embedding": ndarray, "first_seen": int, "last_seen_time": int, "track_history": list }
        self.active_identities: dict = {}
        self.departed_identities: dict = {}
        self.cumulative_visitors = 0

    def match_identity(self, new_embedding: np.ndarray, frame_idx: int):
        """
        Compare a single embedding against all identities.
        Returns the matched global_id, or None if no match below threshold.
        """
        if not isinstance(new_embedding, np.ndarray):
            new_embedding = np.array(new_embedding)
            
        if new_embedding.ndim == 1:
            new_embedding = new_embedding.reshape(1, -1)

        # 1. Gather candidates
        cand_ids = []
        cand_embs = []
        for gid, d in self.active_identities.items():
            cand_ids.append(gid)
            cand_embs.append(d["embedding"])
        for gid, d in self.departed_identities.items():
            cand_ids.append(gid)
            cand_embs.append(d["embedding"])

        if not cand_embs:
            return None

        # 2. Match
        cand_matrix = np.array(cand_embs)                          # (M, 512)
        dist_mat = cdist(new_embedding, cand_matrix, "cosine")[0]  # (M,)
        
        best_idx = np.argmin(dist_mat)
        best_dist = dist_mat[best_idx]

        if best_dist <= self.max_distance:
            matched_id = cand_ids[best_idx]

            # EMA embedding update (adapts to slow appearance changes)
            alpha = 0.9
            updated = alpha * cand_matrix[best_idx] + (1 - alpha) * new_embedding[0]
            norm = np.linalg.norm(updated)
            if norm > 0:
                updated /= norm

            if matched_id in self.departed_identities:
                # Returned from departed buffer
                track = self.departed_identities.pop(matched_id)
                track["embedding"] = updated
                track["last_seen_time"] = frame_idx
                self.active_identities[matched_id] = track
            else:
                self.active_identities[matched_id]["embedding"] = updated
                self.active_identities[matched_id]["last_seen_time"] = frame_idx
                
            return matched_id

        return None

    def register_identity(self, embedding: np.ndarray, frame_idx: int, initial_bbox=None) -> int:
        """
        Store a truly new person.
        """
        gid = self.next_global_id
        self.next_global_id += 1
        
        self.active_identities[gid] = {
            "embedding": np.array(embedding).flatten(),
            "first_seen": frame_idx,
            "last_seen_time": frame_idx,
            "track_history": [initial_bbox] if initial_bbox else []
        }
        self.cumulative_visitors += 1
        return gid

    def update_last_seen(self, global_id: int, frame_idx: int, bbox=None):
        """
        Update the timestamp and append track history without touching embeddings.
        To be called continuously during YOLO tracking frames.
        """
        if global_id in self.active_identities:
            self.active_identities[global_id]["last_seen_time"] = frame_idx
            if bbox:
                self.active_identities[global_id]["track_history"].append(bbox)
                # Keep history short to avoid memory bloat
                if len(self.active_identities[global_id]["track_history"]) > 60:
                    self.active_identities[global_id]["track_history"].pop(0)

    def cleanup(self, frame_idx: int):
        """ Move active identities to departed if not seen, or forget departed after grace period. """
        # Active -> Departed
        evict = [gid for gid, d in self.active_identities.items()
                 if frame_idx - d["last_seen_time"] > self.eviction_timeout]
        for gid in evict:
            self.departed_identities[gid] = self.active_identities.pop(gid)

        # Departed -> Forget
        forget = [gid for gid, d in self.departed_identities.items()
                  if frame_idx - d["last_seen_time"] > self.grace_period]
        for gid in forget:
            del self.departed_identities[gid]
