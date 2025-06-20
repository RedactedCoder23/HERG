from collections import deque
import numpy as np
from herg import backend as B
from .capsule import Capsule


def adf_update(capsule: Capsule, incoming_fast, sign: float, eta: float) -> None:
    """Slow ADF-style update of capsule statistics."""
    mu = B.as_numpy(capsule.mu).astype(np.float32)
    x = B.as_numpy(incoming_fast).astype(np.float32)
    delta = x - mu
    mu = mu + eta * sign * delta
    L = B.as_numpy(capsule.L).astype(np.float32)
    if L.size == 0:
        L = np.zeros((1, mu.size), dtype=np.float32)
    r = L.shape[0]
    d_hat = np.outer(np.ones(r, dtype=np.float32), delta)
    L = L + eta * sign * d_hat
    capsule.mu = B.tensor(mu, dtype=np.float32, device=B.device_of(capsule.fast))
    capsule.L = B.tensor(L, dtype=np.float32, device=B.device_of(capsule.fast))
    capsule.fast = B.tensor(np.sign(mu).astype(np.int8), dtype=np.int8, device=B.device_of(capsule.fast))
    capsule.energy -= float(np.linalg.norm(delta)) * eta


def k_radius_pass(store, radius: int) -> None:
    """Propagate fast state across k-hop neighborhood."""
    for cid, cap in list(store.caps.items()):
        agg = []
        wts = []
        visited = {cid}
        q = deque([(cid, 0)])
        while q:
            node, dist = q.popleft()
            if dist >= radius:
                continue
            neigh_ids, neigh_wts = store.edges.neighbors(node)
            for n, w in zip(neigh_ids, neigh_wts):
                if n in visited:
                    continue
                visited.add(n)
                q.append((n, dist + 1))
                ncap = store.caps.get(n)
                if ncap is None:
                    continue
                weight = 1.0 / (dist + 1)
                agg.append(ncap.fast)
                wts.append(weight)
        if agg:
            stack = B.stack(agg, axis=0)
            if hasattr(stack, "to"):
                import torch
                w = torch.tensor(wts, dtype=torch.float32, device=stack.device).view(-1, 1)
                new = (stack.to(torch.float32) * w).sum(dim=0)
                cap.fast = torch.sign(new).to(torch.int8)
            else:
                arr = B.as_numpy(stack).astype(np.float32)
                w_arr = np.asarray(wts, dtype=np.float32).reshape(-1, 1)
                new = (arr * w_arr).sum(axis=0)
                cap.fast = B.tensor(np.sign(new).astype(np.int8), dtype=np.int8, device=B.device_of(cap.fast))
