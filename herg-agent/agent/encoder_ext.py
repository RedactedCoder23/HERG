"""
Wrapper around your existing HERG encoder that adds:
  • 128-bit cityhash of the raw int8 vector (as bytes)
  • prefix() helper for sharding
"""
from hashlib import blake2b
from typing import Tuple
import numpy as np
from herg.encoder import seed_to_hyper

# --- Constants --------------------------------------------------------------

NUM_DIM = 2048
SHARD_BITS = 8          # prefix length → 256 shards max

# --- API --------------------------------------------------------------------

def encode(seed: str) -> Tuple[np.ndarray, int]:
    """
    Returns (vector[int8], 128-bit_hash_int)
    """
    vec = seed_to_hyper(seed, dim=NUM_DIM)        # int8 bipolar ±1
    h = blake2b(vec.tobytes(), digest_size=16).digest()
    h_int = int.from_bytes(h, "big")
    return vec, h_int

def prefix(hash_int: int) -> str:
    """
    Returns two-hex-digit shard key, e.g. '7a'
    """
    return f"{hash_int >> (128 - SHARD_BITS):02x}"
