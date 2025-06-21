import numpy as np
import pytest
from herg import backend as B
from herg.sinc_kernel import weighted_sinc, modulate, sinc_kernel
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


@pytest.mark.parametrize(
    "x,alpha",
    [
        (np.array([[0.5, -0.5]], dtype=np.float32), [2.0, 0.5]),
        (np.array([[1.0, 0.0]], dtype=np.float32), [1.0, 1.0]),
    ],
)
def test_sinc_kernel_separable(x, alpha):
    out = sinc_kernel(x, alpha, mode='separable')
    expected = np.sinc(alpha[0] * x[..., 0]) * np.sinc(alpha[1] * x[..., 1])
    expected = expected[..., None] * np.ones_like(x)
    assert np.allclose(out, expected)


def test_sinc_kernel_radial_symmetry():
    pts = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = sinc_kernel(pts, 1.0, mode='radial')
    assert np.allclose(out[0], out[1])


def test_sinc_kernel_radial_alpha_mean():
    x = np.array([[1.0, 0.5]], dtype=np.float32)
    alpha = [2.0, 0.5]
    out = sinc_kernel(x, alpha, mode='radial')
    r = np.linalg.norm(x, axis=-1)
    expected = np.sinc(np.mean(alpha) * r)[..., None] * np.ones_like(x)
    assert np.allclose(out, expected)

