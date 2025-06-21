import time
import csv
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# TODO: clean up once herg-agent becomes an installable package
sys.path.append(str(ROOT / "herg-agent"))
from agent.encoder_ext import expand_seed

splits = [3072, 3584, 4096, 4608, 5120, 5632]
rows = []
for d1 in splits:
    lane = (d1, d1 // 2, d1 // 2)
    seeds = [np.random.bytes(16) for _ in range(1000)]
    t0 = time.time()
    vecs = [expand_seed(s, lane)[0] for s in seeds]
    dur = time.time() - t0
    mb = (len(vecs[0]) * len(vecs)) / 1e6
    mbps = mb / dur
    uniq = len({v.tobytes() for v in vecs})
    fp = 1 - uniq / len(vecs)
    rows.append((lane, mbps, fp))

writer = csv.writer(sys.stdout)
writer.writerow(["d1", "d2", "d3", "mbps", "fp_rate"])
for lane, mbps, fp in rows:
    writer.writerow([lane[0], lane[1], lane[2], f"{mbps:.2f}", f"{fp:.4f}"])

