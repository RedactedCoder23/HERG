"""
Background GC: drop PQ-quantised capsule if energy≈0 && parent merged.
"""
import time, logging
from herg.hvlogfs import HVLogFS
from agent.encoder_ext import prefix

log = logging.getLogger(__name__)
hvlog = HVLogFS("/data/hvlog")

def sweep():
    freed = 0
    for chunk in hvlog.chunks():
        for cap in chunk.capsules():
            if cap.meta.get("energy", 1.0) < 0.05:
                chunk.tombstone(cap.id_int)
                freed += 1
        chunk.flush()
    log.info("GC removed %d capsules", freed)

if __name__ == "__main__":
    sweep()
