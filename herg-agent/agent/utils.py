"""Shared utilities for the herg agent."""

import os
import orjson
from .encoder_ext import encode, prefix as _prefix


def add_prefix(payload: dict) -> bytes:
    """Return JSON payload prefixed with two-hex shard key."""
    _, h = encode(payload["seed"])
    p = os.getenv("SHARD_KEY") or _prefix(h)
    return p.encode() + orjson.dumps(payload)
