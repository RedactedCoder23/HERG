"""Minimal encoder wrapper used by CLI demos."""
from herg.encoder import seed_to_hyper
from typing import Tuple
import numpy as np


def encode(seed: str | bytes) -> Tuple[np.ndarray, int]:
    if isinstance(seed, str):
        seed = seed.encode()
    vec = seed_to_hyper(seed)
    return vec, 0
