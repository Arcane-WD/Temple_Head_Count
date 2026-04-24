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

    def match_identity(self, new_embedding: np.ndarray, frame_idx: int, bbox=None, current_gid=None):
        """
        Compare a single embedding against identities with temporal and spatial constraints.
        Returns the matched global_id, or None if no match below threshold.
        """
        if not isinstance(new_embedding, np.ndarray):
            new_embedding = np.array(new_embedding)
            
        if new_embedding.ndim == 1:
            new_embedding = new_embedding.reshape(1, -1)

        valid_cands = []
        for gid, d in self.active_identities.items():
            valid_cands.append((gid, d))
        for gid, d in self.departed_identities.items():
            valid_cands.append((gid, d))
            
        # Optimization: Cap search to the 100 most recently updated identities
        valid_cands.sort(key=lambda x: x[1]["last_seen_time"], reverse=True)
        valid_cands = valid_cands[:100]

        # 1. Gather candidates with constraints
        cand_ids = []
        cand_embs = []
        
        if bbox:
            qx = (bbox[0] + bbox[2]) / 2
            qy = (bbox[1] + bbox[3]) / 2

        for gid, d in valid_cands:
            # CONSTRAINT 1: Same-Frame Clash
            # If identity was already recorded THIS frame and it's not our own, 
            # it's impossible to match (a person cannot be in two boxes at once).
            if d["last_seen_time"] == frame_idx and gid != current_gid:
                continue
                
            # CONSTRAINT 2: Physical Plausibility (Spatial-Temporal)
            if bbox and d["track_history"]:
                last_box = d["track_history"][-1]
                cx = (last_box[0] + last_box[2]) / 2
                cy = (last_box[1] + last_box[3]) / 2
                dist_px = ((qx - cx)**2 + (qy - cy)**2)**0.5
                
                time_gap = max(1, frame_idx - d["last_seen_time"])
                
                # Assume max speed of 40 pixels per frame jump
                max_allowed_dist = max(250.0, time_gap * 40.0)
                if dist_px > max_allowed_dist:
                    continue

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

    def force_update_embedding(self, global_id: int, new_embedding: np.ndarray, frame_idx: int):
        """
        Manually push an EMA update to an existing identity's embedding.
        Useful when an event re-validates a tracker ID but misses the match threshold.
        """
        if global_id in self.active_identities:
            if not isinstance(new_embedding, np.ndarray):
                new_embedding = np.array(new_embedding)
            if new_embedding.ndim == 1:
                new_embedding = new_embedding.reshape(1, -1)
                
            old = self.active_identities[global_id]["embedding"]
            alpha = 0.9
            updated = alpha * old + (1 - alpha) * new_embedding[0]
            norm = np.linalg.norm(updated)
            if norm > 0:
                updated /= norm
            self.active_identities[global_id]["embedding"] = updated
            self.active_identities[global_id]["last_seen_time"] = frame_idx

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
