import numpy as np
from herg import backend as B
from .capsule import Capsule


def sticky_pool_prune(store) -> None:
    """Prune capsules with depleted energy and merge similar ones."""
    to_remove = []
    caps = list(store.caps.values())
    for cap in caps:
        if cap.energy < 0:
            to_remove.append(cap.id)
    # merge based on mu cosine similarity
    ids = list(store.caps.keys())
    for i, cid in enumerate(ids):
        if cid not in store.caps:
            continue
        cap_i = store.caps[cid]
        for j in range(i + 1, len(ids)):
            cj = ids[j]
            if cj not in store.caps:
                continue
            cap_j = store.caps[cj]
            if B.cosine(cap_i.mu, cap_j.mu) > 0.98:
                cap_i.mu = (B.as_numpy(cap_i.mu) + B.as_numpy(cap_j.mu)) / 2
                cap_i.fast = B.tensor(np.sign(B.as_numpy(cap_i.mu)), dtype=np.int8)
                cap_i.energy += cap_j.energy
                to_remove.append(cj)
    for cid in to_remove:
        store.caps.pop(cid, None)
