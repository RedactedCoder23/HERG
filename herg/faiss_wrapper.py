import faiss
import numpy as np
import os


def safe_search(index: faiss.Index, vec, k: int):
    """Search FAISS index safely when index might be empty."""
    if index.ntotal == 0:
        return np.empty((1, 0), dtype="float32"), np.empty((1, 0), dtype="int64")
    return index.search(vec, k)


def make_index(dim: int) -> faiss.Index:
    """Return FAISS index; USE_FLAT=1 forces simple FlatL2."""
    if os.getenv("USE_FLAT", "") == "1":
        return faiss.IndexFlatL2(dim)
    return faiss.IndexFlatL2(dim)

