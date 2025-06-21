import numpy as np
from herg.encoder import seed_to_hyper
from herg.graph_caps.store import CapsuleStore
from herg import backend as B


def test_noise_recall():
    rng = np.random.default_rng(0)
    # fmt: off
    seeds = [
        rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        for _ in range(8)
    ]
    # fmt: on
    store = CapsuleStore(dim=240)
    ids = []
    for s in seeds:
        cap = store.spawn(s)
        ids.append(cap.id)
    hits = 0
    for s, cid in zip(seeds, ids):
        hv = seed_to_hyper(s, dim=240)
        arr = B.as_numpy(hv).copy()
        mask = rng.random(arr.shape) < 0.2
        arr[mask] *= -1
        query = B.tensor(arr, dtype=np.int8)
        best = max(store.caps.values(), key=lambda c: B.cosine(query, c.fast))
        if best.id == cid:
            hits += 1
    recall = hits / len(seeds)
    assert recall >= 0.95
