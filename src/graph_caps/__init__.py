"""Capsule structures and sparse edge table."""

from __future__ import annotations

import dataclasses
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import torch


@dataclasses.dataclass
class Capsule:
    """Minimal capsule node."""

    id: int
    vec: torch.Tensor  # int8
    last_used: int
    edge_ids: List[int]
    edge_wts: List[int]

    def to(self, device: str) -> "Capsule":
        self.vec = self.vec.to(device)
        return self


class EdgeCOO:
    """Sparse COO-style adjacency using torch tensors internally."""

    def __init__(self) -> None:
        self._ids: Dict[int, deque[int]] = defaultdict(deque)
        self._wts: Dict[int, deque[int]] = defaultdict(deque)

    def add_edge(self, src: int, dst: int, w: int) -> None:
        ids_q = self._ids[src]
        wts_q = self._wts[src]
        ids_q.append(dst)
        wts_q.append(int(w))
        if len(ids_q) > 5:
            ids_q.popleft()
            wts_q.popleft()

    def neighbors(self, node_id: int) -> Tuple[List[int], List[int]]:
        ids = list(self._ids.get(node_id, []))
        wts = list(self._wts.get(node_id, []))
        return ids, wts

    def prune_edges(self, threshold: int) -> None:
        for nid in list(self._ids.keys()):
            ids_q = self._ids[nid]
            wts_q = self._wts[nid]
            kept_ids = deque()
            kept_wts = deque()
            for i, w in zip(ids_q, wts_q):
                if abs(int(w)) >= abs(int(threshold)):
                    kept_ids.append(i)
                    kept_wts.append(w)
            self._ids[nid] = kept_ids
            self._wts[nid] = kept_wts

