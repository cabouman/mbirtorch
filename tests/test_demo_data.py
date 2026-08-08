"""Gates for get_ct_model and the demo-data generators.

The reference phantom shares its numpy code with mbirjax, so it gates on
exact equality.  The demo sinograms have the projectors in the loop, so
they gate at the projector tolerance with a small allowance for phantom
voxels that sit exactly on an ellipsoid boundary.  get_ct_model gates on
class and recon shape against the mbirjax golden.
"""

import os

import numpy as np
import pytest

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_npz_path),
    reason="no preprocess goldens: run tests/generate_preprocess_goldens.py in the mbirjax env")


@pytest.fixture(scope="module")
def golden():
    return np.load(_npz_path)


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def test_reference_phantom_matches_exactly(golden):
    out = mbirtorch.generate_3d_shepp_logan_reference((32, 30, 28))
    assert np.array_equal(out, golden["ref_phantom"])


def test_get_ct_model_classes_and_shapes(golden):
    par = mbirtorch.get_ct_model('parallel', (8, 10, 12),
                                 np.linspace(0, np.pi, 8, endpoint=False))
    cone = mbirtorch.get_ct_model('cone', (8, 10, 12),
                                  np.linspace(0, 2 * np.pi, 8, endpoint=False),
                                  source_detector_dist=100.0, source_iso_dist=50.0)
    assert type(par).__name__ == 'ParallelBeamModel'
    assert type(cone).__name__ == 'ConeBeamModel'
    shapes = np.array([par.get_params('recon_shape'), cone.get_params('recon_shape')],
                      dtype=np.int64)
    assert np.array_equal(shapes, golden["gcm_shapes"])


def test_get_ct_model_rejects_bad_geometry():
    with pytest.raises(ValueError):
        mbirtorch.get_ct_model('spiral', (8, 10, 12), np.zeros(8))


def test_get_ct_model_warns_on_parallel_z_shifts():
    with pytest.warns(UserWarning):
        mbirtorch.get_ct_model('parallel', (8, 10, 12),
                               np.linspace(0, np.pi, 8, endpoint=False),
                               helical_z_shifts=np.zeros(8))


@pytest.mark.parametrize("tag,kwargs", [
    ("par", dict(model_type='parallel')),
    ("cone", dict(model_type='cone')),
    ("hel", dict(model_type='cone', use_helical=True, helical_pitch=0.5,
                 helical_z_range=16.0)),
])
def test_generate_demo_data_matches_golden(golden, tag, kwargs):
    phantom, sino, params = mbirtorch.generate_demo_data(
        num_views=12, num_det_rows=24, num_det_channels=32, **kwargs)
    ref_phantom = golden[f"demo_{tag}_phantom"]
    ref_sino = golden[f"demo_{tag}_sino"]
    assert tuple(phantom.shape) == tuple(ref_phantom.shape)
    assert tuple(sino.shape) == tuple(ref_sino.shape)
    # mbirjax builds angles in float32; match at that precision.
    assert np.allclose(params['angles'], golden[f"demo_{tag}_angles"], atol=1e-6)
    if tag == "hel":
        assert np.allclose(params['helical_z_shifts'], golden["demo_hel_z_shifts"], atol=1e-5)

    # A few phantom voxels sit exactly on an ellipsoid boundary and can differ between
    # the two builds; everywhere else the phantoms must agree.
    frac_diff = float(np.mean(~np.isclose(np.asarray(phantom), ref_phantom, atol=1e-6)))
    err = _rel_max(sino, ref_sino)
    print(f"demo {tag}: phantom boundary-voxel fraction = {frac_diff:.2e}, "
          f"sino rel_max = {err:.2e}")
    assert frac_diff < 5e-3
    assert err < 1e-3


def test_gen_cube_phantom_is_float32():
    """mbirjax's jnp.array downcasts to float32; torch.as_tensor keeps
    numpy's float64, which doubles the memory and mps cannot hold at all."""
    import torch
    phantom = mbirtorch.gen_cube_phantom((8, 8, 4))
    assert phantom.dtype == torch.float32


def test_generate_demo_data_cube():
    phantom, sino, params = mbirtorch.generate_demo_data(
        model_type='cone', object_type='cube', num_views=12,
        num_det_rows=24, num_det_channels=32)
    # Host numpy float32 for either object type, as the docstring promises.
    assert isinstance(phantom, np.ndarray) and phantom.dtype == np.float32
    assert phantom.max() > 0 and np.isfinite(np.asarray(sino)).all()


def test_generate_demo_data_translation_needs_translation_model():
    # The translation path is written against TranslationModel, which is not ported yet.
    with pytest.raises(AttributeError):
        mbirtorch.generate_demo_data(model_type='translation',
                                     num_det_rows=24, num_det_channels=32)


def test_gen_translation_vectors_grid():
    vecs = mbirtorch.gen_translation_vectors(3, 2, 10.0, 5.0)
    assert vecs.shape == (6, 3)
    assert np.allclose(vecs[:, 1], 0.0)                       # no y motion
    assert np.allclose(sorted(set(vecs[:, 0])), [-10.0, 0.0, 10.0])
    assert np.allclose(sorted(set(vecs[:, 2])), [-2.5, 2.5])
