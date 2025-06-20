import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "herg-agent"))

from agent.memory import MemoryCapsule, maybe_branch
import numpy as np


class DummyGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
    def add(self, cap):
        self.nodes.append(cap)
    def add_edge(self, s, d, route=""):
        self.edges.append((s, d, route))

def test_hebbian():
    c = MemoryCapsule(1, np.zeros(3, np.float32), {})
    v = np.ones(3, np.float32)
    for _ in range(19):
        c.update(v, 1.0)
    prev = c.mu.copy()
    c.update(v, 1.0)
    delta = np.linalg.norm(c.mu - prev)
    assert delta < 0.1


def test_energy_decay():
    c = MemoryCapsule(1, np.zeros(1, np.float32), {})
    e0 = c.energy
    for _ in range(5):
        c.update(np.zeros(1, np.float32), 0.0)
    assert c.energy < e0


def test_maybe_branch():
    g = DummyGraph()
    parent = MemoryCapsule(1, np.zeros(3, np.float32), {})
    child = maybe_branch(g, parent, np.ones(3, np.float32), 0.0)
    assert child is not None
    assert g.edges and g.edges[0][2] == "branch"

