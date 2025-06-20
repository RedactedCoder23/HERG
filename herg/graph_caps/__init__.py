import numpy as np
from .capsule import Capsule

EdgeId = int
Weight = int  # int16


class EdgeCOO:
    """Ultra-thin sparse COO table (src, dst, w16)."""

    def __init__(self):
        self.src = []      # list[int]
        self.dst = []      # list[int]
        self.wts = []      # list[int16]

    # ------------------------------------------------------------------ #
    def add_edge(self, src: int, dst: int, w: int):
        self.src.append(src)
        self.dst.append(dst)
        self.wts.append(int(np.clip(w, -32767, 32767)))

    def neighbors(self, node_id: int):
        idx = [i for i, s in enumerate(self.src) if s == node_id]
        n_ids = [self.dst[i] for i in idx]
        n_wts = [self.wts[i] for i in idx]
        return n_ids, n_wts

    def prune_edges(self, threshold: int):
        keep = [abs(w) >= threshold for w in self.wts]
        self.src = [s for s, k in zip(self.src, keep) if k]
        self.dst = [d for d, k in zip(self.dst, keep) if k]
        self.wts = [w for w, k in zip(self.wts, keep) if k]

