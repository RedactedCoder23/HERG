import numpy as np
from typing import Sequence, Union, Literal
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


def sinc_kernel(x: np.ndarray,
                alpha: Union[float, Sequence[float]],
                mode: Literal['separable', 'radial'] = 'separable') -> np.ndarray:
    """Generalized sinc kernel supporting per-axis scaling."""
    xp = _get_xp(x)
    arr = xp.asarray(x, dtype=xp.float32)

    if isinstance(alpha, Sequence) and not isinstance(alpha, (bytes, bytearray, str)):
        alpha_vec = xp.asarray(list(alpha), dtype=xp.float32)
        if alpha_vec.size != arr.shape[-1]:
            raise ValueError(f"alpha length {alpha_vec.size} does not match dimensions {arr.shape[-1]}")
    else:
        alpha_vec = xp.full(arr.shape[-1], float(alpha), dtype=xp.float32)

    if mode == 'separable':
        out = xp.ones(arr.shape[:-1], dtype=xp.float32)
        for i in range(arr.shape[-1]):
            out = out * xp.sinc(alpha_vec[i] * arr[..., i])
        return out[..., None] * xp.ones_like(arr)
    elif mode == 'radial':
        r = xp.linalg.norm(alpha_vec * arr, axis=-1)
        return xp.sinc(r)[..., None] * xp.ones_like(arr)
    else:
        raise ValueError(f"unknown mode '{mode}'")

