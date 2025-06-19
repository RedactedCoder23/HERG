import numpy as np
from herg import backend as B

# fixed random signed int8 weight matrix (dim x dim)
def _rand_W(dim: int, device=None, seed: int = 0xFEED):
    rng = np.random.default_rng(seed)
    arr = rng.integers(-3, 4, size=(dim, dim), dtype=np.int8)
    return B.tensor(arr, dtype=np.int8, device=device)

_W_CACHE = {}

# cache vectorized erf for NumPy path
import math as _math
_ERF = np.vectorize(_math.erf)


def gelu(x):
    if hasattr(x, "device"):
        import torch
        return 0.5 * x * (1.0 + torch.erf(x / _math.sqrt(2)))
    else:
        return 0.5 * x * (1.0 + _ERF(x / _math.sqrt(2)))


def gnn_step(center_vec, neigh_vecs, weights):
    """
    Very tiny GNN:   out = center + GELU(W · mean(neigh_vecs * weights))
    where W is a fixed random int8→int8 matrix.
    """
    if not neigh_vecs:
        return center_vec
    dim = B.as_numpy(center_vec).shape[0]
    dev = B.device_of(center_vec)
    key = (dim, dev)
    W = _W_CACHE.get(key)
    if W is None:
        W = _rand_W(dim, device=dev)
        _W_CACHE[key] = W

    stack = B.stack(neigh_vecs, axis=0)
    if hasattr(stack, "to"):
        import torch
        stack = stack.to(torch.int16)
        w_arr = torch.tensor(weights, dtype=torch.int16, device=stack.device).view(-1, 1)
        mean_vec = (stack * w_arr).mean(dim=0)
        lin = (W.to(torch.int16) @ mean_vec)
        result = torch.as_tensor(center_vec, dtype=torch.int16) + gelu(lin).to(torch.int16)
        return result.to(torch.int8)
    else:
        stack = np.asarray(stack, dtype=np.int16)
        w_arr = np.asarray(weights, dtype=np.int16).reshape(-1, 1)
        mean_vec = (stack * w_arr).mean(axis=0)
        lin = np.matmul(B.as_numpy(W).astype(np.int16), mean_vec)
        result = B.as_numpy(center_vec).astype(np.int16) + gelu(lin).astype(np.int16)
        return B.tensor(result, dtype=np.int8, device=dev)
