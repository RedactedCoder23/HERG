"""Prototype hypervector encoder."""

import numpy as np

ENC_DIM = 2048


def seed_to_hyper(seed: int, dim: int = ENC_DIM) -> np.ndarray:
    """Generate a deterministic int8 hypervector from a seed."""
    rng = np.random.RandomState(seed)
    vec = rng.randint(-128, 128, size=dim, dtype=np.int8)
    return vec

# ❑ CODEX: scaffold Capsule dataclass in graph_caps/capsule.py
#     fields = (id: int, vec: torch.IntTensor, last_used: int, edge_ids: list[int], edge_wts: list[int])
#     include .to(device) helper

# ❑ CODEX: implement sparse EdgeCOO tensor in graph_caps/edge_table.py
#     api: add_edge(src, dst, w), neighbors(id) -> (ids, wts)

# ❑ CODEX: write CapsuleStore with spawn/read/update/prune using sticky_pool policy
