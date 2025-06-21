import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "herg-agent"))

from agent.encoder_ext import expand_seed
from herg.distance import hybrid_hamming
from herg.faiss_wrapper import HybridIndex
from agent.utils import safe_search


def _brute(a: np.ndarray, b: np.ndarray, lane_split):
    d1, d2, d3 = lane_split
    idx = 0
    dist = 0
    b1 = d1 // 8
    if b1:
        ba = np.unpackbits(a[idx:idx+b1])[:d1]
        bb = np.unpackbits(b[idx:idx+b1])[:d1]
        dist += np.count_nonzero(ba != bb)
        idx += b1
    b2 = (2 * d2) // 8
    if b2:
        qa = []
        qb = []
        for byte in a[idx:idx+b2]:
            qa.extend([(byte >> (2*i)) & 3 for i in range(4)])
        for byte in b[idx:idx+b2]:
            qb.extend([(byte >> (2*i)) & 3 for i in range(4)])
        dist += np.count_nonzero(np.array(qa[:d2]) != np.array(qb[:d2]))
        idx += b2
    b3 = (4 * d3) // 8
    if b3:
        na = []
        nb = []
        for byte in a[idx:idx+b3]:
            na.extend([(byte >> (4*i)) & 0xF for i in range(2)])
        for byte in b[idx:idx+b3]:
            nb.extend([(byte >> (4*i)) & 0xF for i in range(2)])
        dist += np.count_nonzero(np.array(na[:d3]) != np.array(nb[:d3]))
    return int(dist)


def test_hybrid_distance_matches_brute():
    lane = (8, 4, 4)
    a, _ = expand_seed(b"A"*32, lane)
    b, _ = expand_seed(b"B"*32, lane)
    d1 = hybrid_hamming(a, b, lane)
    d2 = _brute(a, b, lane)
    assert d1 == d2


def test_safe_search_empty():
    idx = HybridIndex((8,4,4))
    q, _ = expand_seed(b"A"*32, (8,4,4))
    D, I = safe_search(idx, q, 1)
    assert D.size == 0 and I.size == 0


def test_safe_search_result():
    lane = (8,4,4)
    idx = HybridIndex(lane)
    vecs = [expand_seed(bytes([i])*16, lane)[0] for i in range(5)]
    idx.add(np.vstack(vecs))
    q = vecs[2]
    D, I = safe_search(idx, q, 1)
    assert I[0,0] == 2 and D[0,0] == 0
