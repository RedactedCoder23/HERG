"""In-memory capsule store with LRU + sticky eviction policy."""

from __future__ import annotations

import pickle
import sqlite3
import time
from collections import OrderedDict
from typing import Dict, Optional

import torch

from . import Capsule
from ..encoder import seed_to_hyper, ENC_DIM


class CapsuleStore:
    """Manage active capsules within a VRAM budget."""

    def __init__(self, dim: int = ENC_DIM, budget: int = 100_000) -> None:
        self.dim = dim
        self.budget = budget
        self.active: "OrderedDict[int, Capsule]" = OrderedDict()
        self.sticky: Dict[int, int] = {}
        self.update_counter = 0

        self.conn = sqlite3.connect("capsules.sqlite")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS caps (id INTEGER PRIMARY KEY, data BLOB)"
        )

    # ---------------------------- basic ops ----------------------------
    def spawn(self, seed: bytes, ts: int) -> Capsule:
        cid = int.from_bytes(seed[:8], "big")
        if cid in self.active:
            cap = self.active.pop(cid)
            self.active[cid] = cap
            cap.last_used = ts
            return cap

        vec = seed_to_hyper(seed, self.dim).clone()
        cap = Capsule(id=cid, vec=vec, last_used=ts, edge_ids=[], edge_wts=[])
        self.active[cid] = cap
        self._ensure_budget(ts)
        return cap

    def read(self, id: int) -> Optional[Capsule]:
        return self.active.get(id)

    def update(self, id: int, delta_vec: torch.Tensor, ts: int) -> None:
        cap = self.active.get(id)
        if cap is None:
            return
        cap.vec = (cap.vec + delta_vec.to(cap.vec.device)).to(torch.int8)
        cap.last_used = ts
        self.active.move_to_end(id)
        self.update_counter += 1
        if self.update_counter % 1000 == 0:
            self.prune(ts)

    # ---------------------------- eviction ----------------------------
    def prune(self, now: int) -> None:
        for cid in list(self.active.keys()):
            if len(self.active) <= self.budget:
                break

            cap = self.active[cid]
            sticky_until = self.sticky.get(cid, 0)
            if sticky_until > now:
                self.active.move_to_end(cid)
                continue

            if self._qualifies_sticky(cap):
                self.sticky[cid] = now + 600
                self.active.move_to_end(cid)
                continue

            self._evict(cid)

    # ---------------------------- helpers ----------------------------
    def _qualifies_sticky(self, cap: Capsule) -> bool:
        active_neighbors = [nid for nid in cap.edge_ids if nid in self.active]
        if len(active_neighbors) <= 1:
            return False
        sim_count = 0
        a = cap.vec.float()
        a_norm = torch.norm(a) + 1e-8
        for nid in active_neighbors:
            b = self.active[nid].vec.float()
            cos = torch.dot(a, b) / (a_norm * (torch.norm(b) + 1e-8))
            if cos > 0.95:
                sim_count += 1
                if sim_count > 1:
                    return True
        return False

    def _evict(self, cid: int) -> None:
        cap = self.active.pop(cid)
        self.sticky.pop(cid, None)
        data = pickle.dumps(cap)
        self.conn.execute(
            "INSERT OR REPLACE INTO caps (id, data) VALUES (?, ?)",
            (cap.id, sqlite3.Binary(data)),
        )
        self.conn.commit()

    def _ensure_budget(self, now: int) -> None:
        if len(self.active) > self.budget:
            self.prune(now)

