try:
    import faiss
except ModuleNotFoundError:  # optional for simple demos
    faiss = None
import numpy as np
import os
from .distance import hybrid_hamming


class HybridIndex:
    """Simple in-memory index using hybrid Hamming distance."""

    def __init__(self, lane_split=(4096, 2048, 2048)):
        self.lane_split = lane_split
        self.vecs = []

    @property
    def ntotal(self) -> int:
        return len(self.vecs)

    def add(self, vecs: np.ndarray):
        arr = np.asarray(vecs, dtype=np.uint8)
        if arr.ndim == 1:
            self.vecs.append(arr.copy())
        else:
            for v in arr:
                self.vecs.append(np.array(v, dtype=np.uint8))

    def search(self, vec: np.ndarray, k: int):
        if self.ntotal == 0:
            return np.empty((1, 0), dtype="float32"), np.empty((1, 0), dtype="int64")
        vec = np.asarray(vec, dtype=np.uint8)
        dists = [hybrid_hamming(vec, v, self.lane_split) for v in self.vecs]
        idx = np.argsort(dists)[:k]
        D = np.array(dists, dtype=np.float32)[idx][None, :]
        I = np.array(idx, dtype=np.int64)[None, :]
        return D, I


def make_index(dim: int):
    """Return FAISS index if available; else raise ImportError."""
    if faiss is None:
        raise ImportError("faiss library not installed")
    if os.getenv("USE_FLAT", "") == "1":
        return faiss.IndexFlatL2(dim)
    return faiss.IndexFlatL2(dim)

