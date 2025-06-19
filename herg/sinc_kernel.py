import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None


def _get_xp(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def weighted_sinc(x, alpha: float = 0.75):
    xp = _get_xp(x)
    arr = xp.asarray(x, dtype=xp.float32)
    w = xp.sinc(arr) * (1 + alpha * (1 - xp.abs(arr))) / (1 + alpha)
    w = xp.where(arr == 0, xp.array(1, dtype=xp.float32), w)
    w = xp.where(xp.abs(arr) == 1, xp.array(0, dtype=xp.float32), w)
    return w


def flavor_coords(digest: bytes) -> np.ndarray:
    arr = np.frombuffer(digest[:6], dtype=np.uint8).astype(np.float32)
    return arr


def modulate(hv, digest: bytes, alpha=0.75):
    xp = _get_xp(hv)
    coords = flavor_coords(digest).astype(np.float32)
    coords = xp.asarray(coords / 255.0 * 2 - 1, dtype=xp.float32)
    weight = weighted_sinc(coords, alpha=alpha)
    w_prod = xp.prod(weight)
    return hv * w_prod

