"""
Offline job: pick cold capsules (energy<THR, age>days) →
             replace mu with PQ-code → zstd compress chunk.
Run weekly via GitHub Action.
"""
import os, time, zstandard as zstd
from pathlib import Path
from herg.hvlogfs import HVLogFS

ENERGY_THR = 0.3
AGE_DAYS   = 7
HVLOG_DIR = Path(os.getenv("HVLOG_DIR", "/data/hvlog"))

hvlog = HVLogFS(str(HVLOG_DIR))
for chunk in hvlog.chunks():
    modified = False
    for cap in chunk.capsules():
        age = time.time() - cap.meta["ts"]
        if cap.meta["energy"] < ENERGY_THR and age > AGE_DAYS * 86400:
            cap.quantise_pq(m=16)           # 16-byte code
            modified = True
    if modified:
        chunk.flush()
        # compress older closed chunks
        if chunk.is_closed() and not chunk.path.suffix == ".zst":
            c_path = chunk.path
            with open(c_path, "rb") as src, open(c_path.with_suffix(".zst"), "wb") as dst:
                dst.write(zstd.compress(src.read(), level=19))
            c_path.unlink()
