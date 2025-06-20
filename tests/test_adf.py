import numpy as np
from herg.graph_caps.capsule import Capsule
from herg.graph_caps.step import adf_update


def test_adf_update():
    cap = Capsule(1, np.zeros(8, dtype=np.int8), np.zeros(8, dtype=np.float32), np.zeros((1, 8), dtype=np.float32))
    incoming = np.ones(8, dtype=np.int8)
    adf_update(cap, incoming, 1.0, 0.5)
    assert np.all(cap.mu != 0)
    assert np.any(cap.L != 0)
