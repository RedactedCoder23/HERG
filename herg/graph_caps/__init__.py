import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from herg import backend as B

EdgeId = int
Weight = int  # int16


@dataclass
class Capsule:
    """Ephemeral / persistent node in the HERG capsule graph."""
    id: int
    vec: Any
    last_used: int
    edge_ids: list[EdgeId] = field(default_factory=list)
    edge_wts: list[Weight] = field(default_factory=list)
    mean: Any | None = None
    L: Any | None = None
    entropy: float = 0.0

    def to(self, device=None):
        self.vec = B.tensor(B.as_numpy(self.vec), dtype=np.int8, device=device)

    def promote(self, dim: int = 6000) -> None:
        if B.device_of(self.vec) != "cpu":
            return
        from herg.cupy_encoder import seed_to_cupy
        self.vec = seed_to_cupy(self.id.to_bytes(32, "big"), dim=dim)

    def demote(self) -> None:
        import numpy as np
        from herg import backend as B
        self.vec = B.tensor(B.as_numpy(self.vec), dtype=np.int8, device="cpu")


class EdgeCOO:
    """Ultra-thin sparse COO table (src, dst, w16)."""

    def __init__(self):
        self.src = []      # list[int]
        self.dst = []      # list[int]
        self.wts = []      # list[int16]

    # ------------------------------------------------------------------ #
    def add_edge(self, src: int, dst: int, w: int):
        self.src.append(src)
        self.dst.append(dst)
        self.wts.append(int(np.clip(w, -32767, 32767)))

    def neighbors(self, node_id: int):
        idx = [i for i, s in enumerate(self.src) if s == node_id]
        n_ids = [self.dst[i] for i in idx]
        n_wts = [self.wts[i] for i in idx]
        return n_ids, n_wts

    def prune_edges(self, threshold: int):
        keep = [abs(w) >= threshold for w in self.wts]
        self.src = [s for s, k in zip(self.src, keep) if k]
        self.dst = [d for d, k in zip(self.dst, keep) if k]
        self.wts = [w for w, k in zip(self.wts, keep) if k]

