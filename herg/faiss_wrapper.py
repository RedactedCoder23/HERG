import faiss
import numpy as np


def safe_search(index: faiss.Index, vec, k: int):
    """Search FAISS index safely when index might be empty."""
    if index.ntotal == 0:
        return np.empty((1, 0), dtype="float32"), np.empty((1, 0), dtype="int64")
    return index.search(vec, k)

