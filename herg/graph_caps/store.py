"""
CapsuleStore – LRU + “sticky” pool of Capsule objects.
Persistently swaps evicted capsules to SQLite (pickle BLOB).
"""

import os, sqlite3, pickle, time, hashlib
import numpy as np
from collections import OrderedDict
from herg.graph_caps import Capsule, EdgeCOO
from herg.encoder import seed_to_hyper
from herg import backend as B

VRAM_BUDGET = 100_000      # max capsules resident
STICKY_TTL  = 600          # seconds capsule is immune after sticky mark


class CapsuleStore:
    def __init__(self, dim=2048, db_path="capsules.sqlite"):
        self.dim = dim
        self.caps: OrderedDict[int, Capsule] = OrderedDict()
        self.edges = EdgeCOO()
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS caps (id INTEGER PRIMARY KEY, blob BLOB)"
        )

    # ------------------------------------------------------------ #
    def _evict_if_needed(self):
        while len(self.caps) > VRAM_BUDGET:
            cid, cap = self.caps.popitem(last=False)           # LRU
            blob = pickle.dumps(cap, protocol=4)
            self.conn.execute("REPLACE INTO caps VALUES (?,?)", (cid, blob))
            self.conn.commit()

    # ------------------------------------------------------------ #
    def spawn(self, seed: bytes, ts=None) -> Capsule:
        digest = seed if len(seed) == 32 else hashlib.sha256(seed).digest()
        cid = int.from_bytes(digest, "big", signed=False) & ((1<<64)-1)
        cap = self.caps.get(cid)
        if cap is None:
            fast = seed_to_hyper(digest, dim=self.dim, device="cpu")
            mu = B.as_numpy(fast).astype(np.float32)
            L = np.zeros((1, self.dim), dtype=np.float32)
            cap = Capsule(cid, fast, mu, L)
            self.caps[cid] = cap
            self._evict_if_needed()
        self.caps.move_to_end(cid, last=True)
        return cap

    # ------------------------------------------------------------ #
    def read(self, cid: int):
        cap = self.caps.get(cid)
        if cap is None:
            cur = self.conn.execute("SELECT blob FROM caps WHERE id=?", (cid,))
            row = cur.fetchone()
            if row:
                cap = pickle.loads(row[0])
                self.caps[cid] = cap
                self._evict_if_needed()
        return cap

    # ------------------------------------------------------------ #
    def update(self, cid: int, delta_vec, sign: float = 1.0, eta: float = 0.05):
        cap = self.read(cid)
        if cap is None:
            return
        from .step import adf_update
        adf_update(cap, delta_vec, sign, eta)
        self.caps.move_to_end(cid, last=True)

    # ------------------------------------------------------------ #
    def prune(self) -> None:
        from .prune import sticky_pool_prune
        sticky_pool_prune(self)
