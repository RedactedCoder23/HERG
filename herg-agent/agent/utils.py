"""Shared utilities for the herg agent."""

import os
import orjson
from .encoder_ext import encode, prefix as _prefix


def add_prefix(payload: dict) -> bytes:
    """Return JSON payload prefixed with two-hex shard key."""
    _, h = encode(payload["seed"])
    p = os.getenv("SHARD_KEY") or _prefix(h)
    return p.encode() + orjson.dumps(payload)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="HERG utility commands")
    parser.add_argument("cmd", choices=["prefix"], help="operation to run")
    parser.add_argument("json", help="payload JSON")
    args = parser.parse_args()

    if args.cmd == "prefix":
        data = orjson.loads(args.json)
        sys.stdout.buffer.write(add_prefix(data))

import numpy as np
try:
    import faiss
except ModuleNotFoundError:
    faiss = None
from herg.faiss_wrapper import HybridIndex


def safe_search(index, xb, k: int):
    """Search wrapper that handles empty HybridIndex or faiss index."""
    if getattr(index, "ntotal", 0) == 0:
        return np.empty((1, 0)), np.empty((1, 0), dtype=np.int64)
    if isinstance(index, HybridIndex):
        D, I = index.search(np.asarray(xb, np.uint8), k)
        return D.astype(np.float32), I
    if faiss is None:
        raise ImportError("faiss library not installed")
    D, I = index.search(np.asarray(xb, np.float32), k)
    return D.astype(np.float32), I


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Tiny cosine similarity helper."""
    num = float(np.dot(a, b))
    den = (float(np.dot(a, a)) ** 0.5) * (float(np.dot(b, b)) ** 0.5) + 1e-8
    return num / den

