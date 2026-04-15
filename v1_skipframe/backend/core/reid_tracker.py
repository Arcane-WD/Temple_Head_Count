import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class ReIDTracker:
    """
    Embedding-set tracker with Hungarian matching and a grace period buffer.
    
    active_tracks:   Currently visible people (matched this frame or recently).
    departed_tracks: Recently evicted people — if they re-appear within the
                     grace period they are NOT counted as new visitors.
    """

    def __init__(self, match_threshold: float = 0.75,
                 eviction_timeout: int = 500, grace_period: int = 7500):
        # Cosine distance threshold (1 - similarity)
        self.max_distance = 1.0 - match_threshold
        self.eviction_timeout = eviction_timeout
        self.grace_period = grace_period

        self.next_id = 1
        # track_id -> {"embedding": ndarray, "first_seen": int, "last_seen": int}
        self.active_tracks: dict = {}
        self.departed_tracks: dict = {}
        self.cumulative_visitors = 0

    # ------------------------------------------------------------------
    def update(self, new_embeddings: np.ndarray, frame_idx: int) -> list:
        """
        Match new_embeddings (N×512 ndarray) against active + departed tracks.
        Returns list[int] of assigned track IDs (length = N).
        """
        if not isinstance(new_embeddings, np.ndarray):
            new_embeddings = np.array(new_embeddings)

        if new_embeddings.ndim != 2 or new_embeddings.shape[0] == 0:
            self._cleanup(frame_idx)
            return []

        n_det = new_embeddings.shape[0]
        assigned_ids = [-1] * n_det

        # 1. Gather candidate embeddings from active + departed
        cand_ids = []
        cand_embs = []
        for tid, d in self.active_tracks.items():
            cand_ids.append(tid)
            cand_embs.append(d["embedding"])
        for tid, d in self.departed_tracks.items():
            cand_ids.append(tid)
            cand_embs.append(d["embedding"])

        unmatched = set(range(n_det))

        # 2. Hungarian matching
        if cand_embs:
            cand_matrix = np.array(cand_embs)                          # (M, 512)
            dist_mat = cdist(new_embeddings, cand_matrix, "cosine")    # (N, M)
            row_ind, col_ind = linear_sum_assignment(dist_mat)

            for r, c in zip(row_ind, col_ind):
                if dist_mat[r, c] > self.max_distance:
                    continue  # too far apart — treat as new person

                matched_id = cand_ids[c]
                assigned_ids[r] = matched_id
                unmatched.discard(r)

                # EMA embedding update (adapts to slow appearance changes)
                alpha = 0.9
                updated = alpha * cand_matrix[c] + (1 - alpha) * new_embeddings[r]
                norm = np.linalg.norm(updated)
                if norm > 0:
                    updated /= norm

                if matched_id in self.departed_tracks:
                    # Person returned from departed buffer — move back to active
                    track = self.departed_tracks.pop(matched_id)
                    track["embedding"] = updated
                    track["last_seen"] = frame_idx
                    self.active_tracks[matched_id] = track
                else:
                    self.active_tracks[matched_id]["embedding"] = updated
                    self.active_tracks[matched_id]["last_seen"] = frame_idx

        # 3. New visitors (unmatched detections)
        for r in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.active_tracks[tid] = {
                "embedding": new_embeddings[r],
                "first_seen": frame_idx,
                "last_seen": frame_idx,
            }
            assigned_ids[r] = tid
            self.cumulative_visitors += 1

        # 4. Evict stale tracks
        self._cleanup(frame_idx)
        return assigned_ids

    # ------------------------------------------------------------------
    def _cleanup(self, frame_idx: int):
        # Active -> Departed
        evict = [t for t, d in self.active_tracks.items()
                 if frame_idx - d["last_seen"] > self.eviction_timeout]
        for t in evict:
            self.departed_tracks[t] = self.active_tracks.pop(t)

        # Departed -> Forget
        forget = [t for t, d in self.departed_tracks.items()
                  if frame_idx - d["last_seen"] > self.grace_period]
        for t in forget:
            del self.departed_tracks[t]
