from typing import Any
import numpy as np
from herg import backend as B

BLOCK_SIZE = 512


def bind_block(i: int, vec: Any) -> Any:
    """Rotate vector by block index for binding."""
    arr = B.as_numpy(vec)
    rolled = np.roll(arr, i * BLOCK_SIZE)
    return B.tensor(rolled, dtype=arr.dtype, device=B.device_of(vec))


def bundle_block(i: int, vec: Any) -> Any:
    """Move block i to the front and zero original segment."""
    arr = B.as_numpy(vec).copy()
    start = i * BLOCK_SIZE
    seg = arr[start:start + BLOCK_SIZE]
    arr[start:start + BLOCK_SIZE] = 0
    arr[:BLOCK_SIZE] += seg
    return B.tensor(arr, dtype=arr.dtype, device=B.device_of(vec))
