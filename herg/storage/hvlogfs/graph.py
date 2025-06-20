import os
import pickle
from typing import List

import numpy as np

try:
    import hnswlib
    _HAVE_HNSW = True
except Exception:  # pragma: no cover - optional
    _HAVE_HNSW = False


class DiskHNSW:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        if _HAVE_HNSW:
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.index.init_index(max_elements=10000, ef_construction=100, M=16)
        else:
            self.index: List[tuple[int, np.ndarray]] = []

    def add(self, vec: np.ndarray, label: int) -> None:
        if _HAVE_HNSW:
            self.index.add_items(vec.reshape(1, -1), [label])
        else:
            self.index.append((label, vec.astype(np.float32)))
        self._dump()

    def query(self, vec: np.ndarray, k: int) -> List[int]:
        if _HAVE_HNSW:
            ids, _ = self.index.knn_query(vec.reshape(1, -1), k=k)
            return ids[0].tolist()
        dists = [float(np.dot(vec, v) / (np.linalg.norm(vec) * np.linalg.norm(v) + 1e-9))
                 for _, v in self.index]
        order = np.argsort(dists)[::-1][:k]
        return [self.index[i][0] for i in order]

    def _dump(self) -> None:
        with open(self.path, 'wb') as f:
            pickle.dump(self.index, f)

