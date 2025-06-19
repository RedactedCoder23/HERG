# ◇ CODEX_IMPLEMENT: test viz_dot output
from herg.graph_caps.store import CapsuleStore
from herg.viz import viz_dot


def test_viz_dot(tmp_path):
    store = CapsuleStore()
    a = store.spawn(b"a", ts=0)
    b = store.spawn(b"b", ts=0)
    store.edges.add_edge(a.id, b.id, 1)
    dot = viz_dot(store, last_n=2)
    assert str(a.id) in dot and str(b.id) in dot and "->" in dot
