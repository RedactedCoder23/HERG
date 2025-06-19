import numpy as np
from herg import backend as B
from herg.sinc_kernel import weighted_sinc, modulate
from herg.graph_caps.store import CapsuleStore


def test_weighted_sinc_values():
    assert weighted_sinc(0) == 1
    assert weighted_sinc(1) == 0


def test_modulate_shape_and_dtype():
    hv = np.ones(8, dtype=np.float32)
    digest = b"\x00\x01\x02\x03\x04\x05"
    out = modulate(hv, digest)
    assert out.shape == hv.shape
    assert out.dtype == hv.dtype
    assert not np.allclose(out, hv)


def test_adf_update_shrinks_distance(tmp_path):
    store = CapsuleStore(dim=8, db_path=str(tmp_path / "db.sqlite"))
    cap = store.spawn(b"seed")
    cid = cap.id
    x = np.ones(8, dtype=np.float32)
    store.update(cid, x)
    dist1 = np.linalg.norm(x - cap.mean)
    store.update(cid, x)
    dist2 = np.linalg.norm(x - cap.mean)
    assert dist2 < dist1

