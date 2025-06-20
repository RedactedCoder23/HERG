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
import faiss


def safe_search(index: faiss.Index, xb, k: int):
    if index.ntotal == 0:
        return np.empty((1, 0)), np.empty((1, 0), dtype=np.int64)
    return index.search(xb, k)

