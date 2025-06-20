from herg.graph_caps.store import CapsuleStore
from herg.graph_caps.step import k_radius_pass, adf_update
from herg.graph_caps.gossip import gap_junction_gossip
from herg.graph_caps.prune import sticky_pool_prune
from herg.encoder import seed_to_hyper


def dual_clock_loop(ticks: int, radius: int, gossip_every: int = 8) -> CapsuleStore:
    store = CapsuleStore()
    for tick in range(ticks):
        seed = f"{tick}".encode()
        cap = store.spawn(seed)
        k_radius_pass(store, radius)
        if tick % gossip_every == 0:
            for c in list(store.caps.values()):
                adf_update(c, c.fast, 1.0, 0.1)
            gap_junction_gossip(store)
            sticky_pool_prune(store)
    return store
