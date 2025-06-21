"""
Wrapper around your existing HERG encoder that adds:
  • 64-bit blake2 hash of the raw int8 vector (as bytes)
  • prefix() helper for sharding
"""
from hashlib import blake2b
from typing import Tuple
import numpy as np


def _philox(seed: bytes) -> np.random.Generator:
    """Return Philox32 generator seeded with bytes."""
    digest = blake2b(seed, digest_size=16).digest()
    seed_int = int.from_bytes(digest, "big")
    return np.random.Generator(np.random.Philox(seed_int))

# --- Constants --------------------------------------------------------------

NUM_DIM = 2048
SHARD_BITS = 8          # prefix length → 256 shards max

# --- API --------------------------------------------------------------------

def expand_seed(seed: bytes,
                lane_split: tuple[int, int, int] = (4096, 2048, 2048),
                dtype=np.uint8) -> Tuple[np.ndarray, int]:
    """Return packed mixed-radix vector and 64-bit blake2 hash."""
    if isinstance(seed, str):
        seed = seed.encode()

    d1, d2, d3 = lane_split
    rng = _philox(seed)
    bin_bits = rng.integers(0, 2, size=d1, dtype=np.uint8)
    quads = rng.integers(0, 4, size=d2, dtype=np.uint8)
    nibbles = rng.integers(0, 16, size=d3, dtype=np.uint8)

    total_bits = d1 + 2 * d2 + 4 * d3
    vec_len = (total_bits + 7) // 8
    vec = np.zeros(vec_len, dtype=dtype)

    bit_idx = 0
    for b in bin_bits:
        vec[bit_idx >> 3] |= b << (bit_idx & 7)
        bit_idx += 1
    for q in quads:
        for i in range(2):
            vec[bit_idx >> 3] |= ((q >> i) & 1) << (bit_idx & 7)
            bit_idx += 1
    for n in nibbles:
        for i in range(4):
            vec[bit_idx >> 3] |= ((n >> i) & 1) << (bit_idx & 7)
            bit_idx += 1

    h = blake2b(vec.tobytes(), digest_size=8).digest()
    return vec, int.from_bytes(h, "big")

def encode(seed: str | bytes) -> Tuple[np.ndarray, int]:
    """Wrapper for backward compatibility."""
    return expand_seed(seed, LANE_SPLIT)

def prefix(hash_int: int) -> str:
    """
    Returns two-hex-digit shard key, e.g. '7a'
    """
    return f"{hash_int >> (64 - SHARD_BITS):02x}"

LANE_SPLIT = (4096, 2048, 2048)   # (binary, quaternary, hex)
