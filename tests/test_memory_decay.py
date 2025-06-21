import numpy as np
from herg.memory import MemoryCapsule


def test_memory_update_no_change_when_same_vec():
    mu = np.random.rand(4).astype(np.float32)
    cap = MemoryCapsule(1, mu.copy(), {})
    cap.update(mu)
    assert np.allclose(cap.mu, mu)

