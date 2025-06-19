"""
GPU-only encoder with ECC and 6-D sinc modulation.
"""

import hashlib
import numpy as np
from herg import backend as B
from herg import sinc_kernel


def _ecc_encode(digest: bytes) -> bytes:
    w = [int.from_bytes(digest[i*8:(i+1)*8], "big") for i in range(4)]
    p = w[0] ^ w[1] ^ w[2] ^ w[3]
    return digest + p.to_bytes(8, "big")


def _ecc_decode(code: bytes, simulate_error: bool=False):
    w = [int.from_bytes(code[i*8:(i+1)*8], "big") for i in range(4)]
    p = int.from_bytes(code[32:40], "big")
    if simulate_error:
        w[0] ^= 1
    parity = w[0] ^ w[1] ^ w[2] ^ w[3]
    if parity != p:
        err = parity ^ p
        w[0] ^= err
    return tuple(w)


def seed_to_cupy(seed: bytes, dim: int = 6000, simulate_error: bool = False):
    if len(seed) != 32:
        digest = hashlib.sha256(seed).digest()
    else:
        digest = seed
    code = _ecc_encode(digest)
    w0, w1, w2, w3 = _ecc_decode(code, simulate_error)

    xp = np
    if getattr(B, "IS_CUPY", False):
        import cupy as cp
        xp = cp
    elif getattr(B, "_TORCH", False):
        import torch
        xp = torch

    signs = []
    for i in range(dim):
        bit = ((w0 ^ w1) >> (i % 64)) & 1
        sign = 0.7071 if bit else -0.7071
        signs.append(sign)
    arr = xp.asarray(signs, dtype=xp.float32)
    if xp is np:
        noise = xp.random.normal(0.0, 0.01, size=dim).astype(xp.float32)
    else:
        noise = xp.random.normal(0.0, 0.01, size=dim, dtype=xp.float32)
    arr = arr + noise
    arr = sinc_kernel.modulate(arr, digest)
    return B.tensor(arr, dtype=np.float32)

