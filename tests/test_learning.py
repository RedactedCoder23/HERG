import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "herg-agent"))

import numpy as np
from agent.memory import MemoryCapsule, maybe_branch


class DummyGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
    def add(self, cap):
        self.nodes.append(cap)
    def add_edge(self, s, d, route=""):
        self.edges.append((s, d, route))

def test_hebbian():
    c = MemoryCapsule(1, np.zeros(3), {})
    v = np.ones(3, np.float32)
    for _ in range(50):
        c.update(v, 1.0)
    assert np.allclose(c.mu, v, atol=0.1)


def test_energy_decay():
    c = MemoryCapsule(2, np.zeros(1), {})
    e0 = c.energy
    for _ in range(50):
        c.update(np.zeros(1), 0.0)
    assert c.energy < e0


def test_branch_spawn():
    g = DummyGraph()
    parent = MemoryCapsule(3, np.zeros(3), {})
    child = maybe_branch(g, parent, np.ones(3), 0.0)
    assert child and g.edges

