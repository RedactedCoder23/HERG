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
        ts = ts or int(time.time())
        digest = hashlib.sha256(seed).digest()
        cid = int.from_bytes(digest, "big", signed=False) & ((1<<64)-1)  # low 64 bits of full hash
        cap = self.caps.get(cid)
        if cap is None:
            vec = seed_to_hyper(seed, dim=self.dim, device="cpu")
            cap = Capsule(cid, vec, ts)
            self.caps[cid] = cap
            self._evict_if_needed()
        cap.last_used = ts
        self.caps.move_to_end(cid, last=True)   # mark as recently used
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
    def update(self, cid: int, delta_vec, ts=None):
        ts = ts or int(time.time())
        cap = self.read(cid)
        if cap is None:
            return
        new_vec = B.tensor(
            B.as_numpy(cap.vec) + B.as_numpy(delta_vec), dtype=np.int8
        )
        cap.vec = new_vec
        cap.last_used = ts
        self.caps.move_to_end(cid, last=True)

    # ------------------------------------------------------------ #
    def prune(self, now: int | None = None):
        now = now or int(time.time())
        to_drop = []
        for cid, cap in list(self.caps.items()):
            age = now - cap.last_used
            # cosine stickiness check (≥0.95 with any neighbour)
            neigh_ids, _ = self.edges.neighbors(cid)
            sticky = False
            for nid in neigh_ids:
                ncap = self.caps.get(nid)
                if ncap is None:
                    continue
                if B.cosine(cap.vec, ncap.vec) > 0.95:
                    sticky = True
                    break
            if not sticky and age > STICKY_TTL:
                to_drop.append(cid)
        for cid in to_drop:
            blob = pickle.dumps(self.caps.pop(cid), protocol=4)
            self.conn.execute("REPLACE INTO caps VALUES (?,?)", (cid, blob))
        if to_drop:
            self.conn.commit()
