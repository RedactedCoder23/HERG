import hashlib

import torch

from src.encoder import seed_to_hyper


def test_determinism():
    seed = hashlib.sha256(b"foo").digest()
    v1 = seed_to_hyper(seed, dim=128)
    v2 = seed_to_hyper(seed, dim=128)
    assert torch.equal(v1, v2)


def test_bipolar_values():
    seed = hashlib.sha256(b"bar").digest()
    vec = seed_to_hyper(seed, dim=256)
    assert set(vec.tolist()) <= {1, -1}


def test_ternary_density():
    seed = hashlib.sha256(b"baz").digest()
    dim = 300
    vec = seed_to_hyper(seed, dim=dim, ternary=True)
    density = (vec != 0).float().mean().item()
    assert 0.32 <= density <= 0.35

