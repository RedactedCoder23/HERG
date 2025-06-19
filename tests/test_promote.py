# ◇ CODEX_IMPLEMENT: test promote/demote device toggling
import numpy as np
from herg.graph_caps import Capsule
from herg.graph_caps.store import CapsuleStore
from herg import backend as B
import pytest


def test_promote_demote(tmp_path):
    store = CapsuleStore()
    cap = store.spawn(b"x", ts=0)
    cap.demote()
    assert B.device_of(cap.vec) == "cpu"
    cap.promote(dim=2048)
    if B.device_of(cap.vec) == "cpu":
        pytest.skip("No GPU available")
    assert B.device_of(cap.vec) != "cpu"
