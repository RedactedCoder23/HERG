import numpy as np
from herg.encoder import seed_to_hyper
from herg import backend as B

def test_determinism():
    seed = b"A" * 32
    v1 = seed_to_hyper(seed)
    v2 = seed_to_hyper(seed)
    assert np.array_equal(B.as_numpy(v1), B.as_numpy(v2))

def test_bipolar():
    seed = b"B" * 32
    v = seed_to_hyper(seed, ternary=False)
    arr = B.as_numpy(v)
    assert set(arr.tolist()) <= {-1, 1}

def test_ternary_density():
    seed = b"C" * 32
    v = seed_to_hyper(seed, ternary=True)
    arr = B.as_numpy(v)
    density = np.count_nonzero(arr) / arr.size
    assert 0.31 <= density <= 0.35
