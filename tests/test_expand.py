import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "herg-agent"))

from agent.encoder_ext import expand_seed

def test_expand_len():
    vec, _ = expand_seed(b'A'*32, (8,4,4))
    assert len(vec) == 4

def test_expand_determinism():
    s = b'B'*32
    v1, _ = expand_seed(s, (8,4,4))
    v2, _ = expand_seed(s, (8,4,4))
    assert np.array_equal(v1, v2)

def test_expand_collision_rate():
    seen = set()
    collisions = 0
    for _ in range(1000):
        seed = np.random.bytes(16)
        v, _ = expand_seed(seed, (8,4,4))
        if v.tobytes() in seen:
            collisions += 1
        else:
            seen.add(v.tobytes())
    assert collisions / 1000 < 0.05
