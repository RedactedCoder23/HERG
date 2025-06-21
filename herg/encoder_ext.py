"""Minimal encoder wrapper used by CLI demos."""
from typing import Tuple, Sequence

import numpy as np

from herg import config
from herg.encoder import seed_to_hyper
from herg.sinc_kernel import flavor_coords, sinc_kernel


def encode(seed: str | bytes,
           alpha: Sequence[float] | float | None = None,
           mode: str | None = None) -> Tuple[np.ndarray, int]:
    """Encode seed with optional sinc-kernel modulation."""
    if isinstance(seed, str):
        seed = seed.encode()
    cfg = config.load()
    alpha = alpha if alpha is not None else cfg.kernel_alpha
    mode = mode or cfg.kernel_mode
    vec = seed_to_hyper(seed)
    coords = flavor_coords(seed).astype(np.float32)
    coords = coords / 255.0 * 2 - 1
    k = sinc_kernel(coords, alpha, mode=mode)
    weight = float(np.prod(np.asarray(k)))
    vec = vec * weight
    return vec, 0
