from herg.graph_caps.store import CapsuleStore
from herg.graph_caps.step import k_radius_pass
from herg.graph_caps.blocks import bind_block, BLOCK_SIZE
from herg import backend as B
import numpy as np


def test_edge_blocks_isolation():
    store = CapsuleStore(dim=BLOCK_SIZE * 4)
    root = store.spawn(b'r')
    n1 = store.spawn(b'a')
    n2 = store.spawn(b'b')
    store.edges.add_edge(root.id, n1.id, 1)
    store.edges.add_edge(n1.id, n2.id, 1)
    expected = (
        B.as_numpy(bind_block(1, n1.fast)).astype(float) * 1.0
        + B.as_numpy(bind_block(2, n2.fast)).astype(float) * 0.5
    )
    expected = np.sign(expected).astype(np.int8)
    k_radius_pass(store, radius=2)
    cap = store.caps[root.id]
    assert np.array_equal(B.as_numpy(cap.fast), expected)
