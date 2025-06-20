import numpy as np
from herg import backend as B


def gap_junction_gossip(store, ε: float = 0.1) -> None:
    """Share slow state across all capsules via mean-field."""
    if not store.caps:
        return
    mus = [B.as_numpy(c.mu).astype(np.float32) for c in store.caps.values()]
    cluster_mu = np.mean(mus, axis=0)
    for cap in store.caps.values():
        mu = (1 - ε) * B.as_numpy(cap.mu).astype(np.float32) + ε * cluster_mu
        cap.mu = B.tensor(mu, dtype=np.float32, device=B.device_of(cap.fast))
        cap.fast = B.tensor(np.sign(mu).astype(np.int8), dtype=np.int8, device=B.device_of(cap.fast))
