import hashlib
import os
import numpy as np
from herg import backend as B
from herg.sinc_kernel import flavor_coords


def sinc_sep(x):
    return np.prod(np.sinc(x / np.pi), axis=-1)

DIM_DEFAULT = 2048          # can be overridden per-call
_LCG_A = 1103515245
_LCG_C = 12345
_LCG_M = 2 ** 31


def _lcg(value):
    """One step of a 31-bit LCG (same constants as glibc rand)."""
    return (_LCG_A * value + _LCG_C) % _LCG_M


def seed_to_hyper(seed: bytes,
                  dim: int = DIM_DEFAULT,
                  ternary: bool = False,
                  device: str | None = None,
                  alpha: float = 0.0):
    """
    Deterministic ±1 (or ternary) hyper-vector from an arbitrary seed.

    Algorithm (branch-free, CPU/GPU friendly):
    1. If seed ≠ 32 bytes → SHA-256 hash it.
    2. Interpret digest as big-endian int S.
    3. For each i:   h = LCG( S XOR i ) ; bit = (h >> 15) & 1
                     val = +1 if bit else −1
    4. If ternary: zero indices where i % 3 != 0  (~33 % density)
    """
    if len(seed) != 32:
        seed = hashlib.sha256(seed).digest()
    S = int.from_bytes(seed, "big")

    vals = []
    x = S & (_LCG_M - 1)
    for i in range(dim):
        x = _lcg(x ^ i)
        vals.append(1 if (x >> 15) & 1 else -1)

    if ternary:
        for i in range(dim):
            if i % 3:
                vals[i] = 0

    hv = np.asarray(vals, dtype=np.float32)
    if dim % 6 == 0:
        coords = flavor_coords(seed) / 255.0 * np.pi * 2 - np.pi
        weights = np.sinc((alpha * coords) / np.pi).reshape(6, 1)
        hv = hv.reshape(6, dim // 6) * weights
        hv = hv.reshape(dim)
    hv = np.sign(hv).astype(np.int8)
    return B.tensor(hv, dtype=np.int8, device=device)


# convenience wrapper used by CapsuleStore
def sha_vector_of_token(token: str,
                        dim: int = DIM_DEFAULT,
                        ternary: bool = False,
                        device: str | None = None):
    return seed_to_hyper(token.encode(), dim, ternary, device)
