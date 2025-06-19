# ◇ CODEX_IMPLEMENT: test save/load round-trip
import pickle, pytest
from herg.graph_caps.store import CapsuleStore
from herg.snapshot import save_snapshot, load_snapshot


def test_save_and_load(tmp_path):
    store = CapsuleStore()
    store.spawn(b"x", ts=0)
    path = str(tmp_path / "brain.pkl")
    save_snapshot(store, path)
    new = load_snapshot(path)
    assert isinstance(new, CapsuleStore)
    assert len(new.caps) == 1
