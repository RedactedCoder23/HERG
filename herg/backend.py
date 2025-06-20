"""
Light-weight backend wrapper – NumPy by default, optional Torch or CuPy.

Set   HERG_BACKEND=torch   or   HERG_BACKEND=cupy   to opt-in to GPU backends.
"""

import os
import numpy as np

_CUPY = False
if os.getenv("HERG_BACKEND") == "cupy":
    try:
        import cupy as cp  # type: ignore
        _CUPY = True
    except Exception:
        cp = None

IS_CUPY = _CUPY and 'cp' in globals() and cp is not None

_TORCH = False
if os.getenv("HERG_BACKEND") == "torch":
    try:
        import torch
        _TORCH = True
    except ImportError:
        pass  # stay NumPy-only

# --------------------------------------------------------------------------- #
# Public helpers                                                              #
# --------------------------------------------------------------------------- #
def _to_torch_dtype(np_dtype):
    """Map NumPy dtype → torch dtype (int8 / int16 / int32 / float32)."""
    if not _TORCH:
        raise RuntimeError("Torch backend not active")
    import torch  # local import to silence linters

    return {
        np.int8:  torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.float32: torch.float32,
    }[np_dtype]


def tensor(data, dtype=np.int8, device=None):
    """
    Return a 1-D tensor (NumPy ndarray **or** torch.Tensor, depending on backend).
    • dtype must be NumPy dtype for portability.
    • device is ignored for NumPy; for Torch defaults to 'cuda' if available.
    """
    if _TORCH:
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(data, torch.Tensor):
            return data.to(device=dev, dtype=_to_torch_dtype(dtype))
        else:
            return torch.tensor(data, dtype=_to_torch_dtype(dtype), device=dev)
    elif _CUPY:
        arr = cp.asarray(data, dtype=dtype)
        return arr
    else:
        return np.asarray(data, dtype=dtype)


def as_numpy(t):
    """Always produce a NumPy view (no copy if already ndarray)."""
    if _TORCH:
        import torch

        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
    if IS_CUPY and isinstance(t, cp.ndarray):
        return cp.asnumpy(t)
    return np.asarray(t)


def dot(a, b):
    if _TORCH and hasattr(a, "dot"):
        return float(a.float().dot(b.float()))
    if IS_CUPY and isinstance(a, cp.ndarray):
        return float(cp.dot(a.astype(cp.float32), b.astype(cp.float32)))
    return float(np.dot(as_numpy(a).astype(np.float32), as_numpy(b).astype(np.float32)))


def cosine(a, b, eps=1e-8):
    num = dot(a, b)
    den = (dot(a, a) ** 0.5) * (dot(b, b) ** 0.5) + eps
    return num / den


def stack(seq, axis=0):
    if _TORCH and any(hasattr(v, "device") for v in seq):
        import torch

        return torch.stack(seq, dim=axis)
    if IS_CUPY and any(isinstance(v, cp.ndarray) for v in seq):
        return cp.stack(seq, axis=axis)
    return np.stack([as_numpy(v) for v in seq], axis=axis)


def device_of(t):
    if _TORCH and hasattr(t, "device"):
        return "cuda" if t.device.type == "cuda" else "cpu"
    if IS_CUPY and isinstance(t, cp.ndarray):
        return "cuda"
    return "cpu"
