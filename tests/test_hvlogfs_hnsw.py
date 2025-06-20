import numpy as np
from herg.storage.hvlogfs.graph import DiskHNSW


def test_topk_equiv(tmp_path):
    dim = 16
    idx = DiskHNSW(dim, str(tmp_path / 'g.idx'))
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((10, dim)).astype(np.float32)
    for i, v in enumerate(vecs):
        idx.add(v, i)

    query = vecs[0]
    res = idx.query(query, k=3)

    # brute force
    sims = vecs @ query / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(query) + 1e-9)
    brute = np.argsort(sims)[::-1][:3].tolist()
    assert res == brute

