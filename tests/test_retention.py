import numpy as np
from herg.graph_caps.store import CapsuleStore
from herg.graph_caps.step import adf_update
from herg.encoder import seed_to_hyper
from herg import backend as B


def test_retention_50tasks():
    rng = np.random.default_rng(0)
    store = CapsuleStore(dim=240)
    seeds = [
        rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
        for _ in range(50)
    ]
    ids = []
    for s in seeds:
        cap = store.spawn(s)
        adf_update(cap, cap.fast, 1.0, 0.1)
        ids.append(cap.id)
    hits = 0
    for s, cid in zip(seeds[:10], ids[:10]):
        query = seed_to_hyper(s, dim=240)
        best = max(store.caps.values(), key=lambda c: B.cosine(query, c.fast))
        if best.id == cid:
            hits += 1
    assert hits / 10 >= 0.70
