"""Gates for the two public model-construction helpers -- get_ct_model and
copy_ct_model -- and the demo-data generators.

The reference phantom shares its numpy code with mbirjax, so it gates on
exact equality.  The demo sinograms have the projectors in the loop, so
they gate at the projector tolerance with a small allowance for phantom
voxels that sit exactly on an ellipsoid boundary.  get_ct_model gates on
class and recon shape against the mbirjax golden.

Both helpers accept four geometries, and each geometry names its per-view
parameters differently, so the tests at the end of this file gate every
geometry through both helpers rather than assuming the angle-based form
carries over.
"""

import os

import numpy as np
import pytest

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")
_REGENERATE = "run tests/generate_preprocess_goldens.py in the mbirjax env"


class _Archive:
    """The golden archive, read one key at a time.

    An archive written before a golden was added is missing that key, which
    is the same situation as having no archive at all, so a lookup that
    misses skips with the regeneration message rather than raising KeyError.
    """

    def __init__(self, npz):
        self._npz = npz

    def __getitem__(self, key):
        if key not in self._npz:
            pytest.skip(f"golden '{key}' predates this archive: {_REGENERATE}")
        return self._npz[key]


@pytest.fixture(scope="module")
def golden():
    """The archive, for the tests that gate against mbirjax.

    Requesting this fixture is what makes a test need the archive; the tests
    below that check mbirtorch against itself take no golden and run
    whether or not one has been generated.
    """
    if not os.path.exists(_npz_path):
        pytest.skip(f"no preprocess goldens: {_REGENERATE}")
    return _Archive(np.load(_npz_path))


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


@pytest.mark.goldens
def test_reference_phantom_matches_exactly(golden):
    out = mbirtorch.generate_3d_shepp_logan_reference((32, 30, 28))
    assert np.array_equal(out, golden["ref_phantom"])


@pytest.mark.goldens
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
@pytest.mark.goldens
def test_generate_demo_data_matches_golden(golden, tag, kwargs):
    phantom, sino, params = mbirtorch.generate_demo_data(
        num_views=12, num_det_rows=24, num_det_channels=32, **kwargs)
    ref_phantom = golden[f"demo_{tag}_phantom"]
    ref_sino = golden[f"demo_{tag}_sino"]
    assert tuple(phantom.shape) == tuple(ref_phantom.shape)
    assert tuple(sino.shape) == tuple(ref_sino.shape)
    # mbirjax builds angles in float32, so the two sets differ by the float32
    # rounding of the same linspace: the largest measured gap is 1.21e-6, for
    # the helical case.  An absolute gate alone holds that, with no rtol term
    # to let a large angle through on a relative allowance.
    assert np.allclose(params['angles'], golden[f"demo_{tag}_angles"],
                       rtol=0, atol=5e-6)
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


def test_generate_demo_data_translation():
    # Structure only: with translation's thin recon volume the generic demo
    # phantoms are empty (mbirjax behaves identically); translation demos use
    # gen_translation_phantom for content.
    phantom, sino, params = mbirtorch.generate_demo_data(
        model_type='translation', object_type='cube',
        num_det_rows=24, num_det_channels=32)
    assert isinstance(phantom, np.ndarray) and phantom.dtype == np.float32
    assert sino.shape[0] == params['translation_vectors'].shape[0]
    assert np.isfinite(np.asarray(sino)).all()


def test_gen_translation_vectors_grid():
    vecs = mbirtorch.gen_translation_vectors(3, 2, 10.0, 5.0)
    assert vecs.shape == (6, 3)
    assert np.allclose(vecs[:, 1], 0.0)                       # no y motion
    assert np.allclose(sorted(set(vecs[:, 0])), [-10.0, 0.0, 10.0])
    assert np.allclose(sorted(set(vecs[:, 2])), [-2.5, 2.5])


# ── the two helpers across all four geometries ───────────────────────────────
MA_CELL = (16, 24, 20)
TCT_DETS = (40, 32)


def _multiaxis_angles(num_views=MA_CELL[0]):
    """(azimuth, elevation) pairs -- multiaxis's angles are a (num_views, 2)
    array, not the 1D vector parallel and cone take."""
    azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
    elevation = np.linspace(-0.5, 0.5, num_views)
    return np.stack([azimuth, elevation], axis=1)


def _multiaxis_model():
    model = mbirtorch.MultiAxisParallelModel(MA_CELL, _multiaxis_angles())
    model.set_params(no_warning=True, verbose=0)
    return model


def _translation_model():
    vectors = mbirtorch.gen_translation_vectors(4, 4, x_spacing=3.0, z_spacing=2.0)
    model = mbirtorch.TranslationModel((vectors.shape[0],) + TCT_DETS, vectors,
                                       source_detector_dist=128.0, source_iso_dist=32.0)
    model.set_params(no_warning=True, verbose=0)
    return model


@pytest.mark.parametrize("make_model,view_key", [
    (_multiaxis_model, 'angles'),
    (_translation_model, 'translation_vectors'),
], ids=["multiaxis", "translation"])
def test_copy_ct_model_with_no_changes_reproduces_the_model(make_model, view_key):
    """The copy is the same reconstruction, not merely the same class.

    The per-view array is gated EXACTLY, because carrying it through
    get_all_params and build_model unchanged is the whole job here: an array
    dropped, reshaped or read under the wrong key shows up as an inexact
    round trip rather than as a different repr.

    The sinogram is gated at float level instead.  Two separately built
    models sum their views in their own order, so a copy of an unchanged
    PARALLEL or CONE model already differs from its original by 5e-8 to
    1.3e-7 on this phantom; multiaxis measures 6.8e-8 and translation 0.
    Bitwise is therefore the wrong bar for every geometry, not just the new
    ones, and 1e-5 leaves roughly two orders of headroom over the largest
    measured difference.
    """
    model = make_model()
    copy = mbirtorch.copy_ct_model(model)

    assert type(copy) is type(model)
    assert tuple(copy.get_params('sinogram_shape')) == tuple(model.get_params('sinogram_shape'))
    assert tuple(copy.get_params('recon_shape')) == tuple(model.get_params('recon_shape'))
    assert np.array_equal(np.asarray(copy.get_params(view_key)),
                          np.asarray(model.get_params(view_key)))

    recon_shape = tuple(model.get_params('recon_shape'))
    phantom = mbirtorch.gen_translation_phantom(recon_shape, 'dots', None, fill_rate=0.05)
    err = _rel_max(copy.forward_project(phantom), model.forward_project(phantom))
    print(f"copy_ct_model forward rel_max = {err:.2e}")
    assert err < 1e-5


def test_copy_ct_model_multiaxis_takes_new_angles_and_detector_rows():
    """Multiaxis reaches copy_ct_model through the angles key like parallel
    and cone; the only difference is that each entry is a pair, so the view
    count is the number of ROWS of the array."""
    model = _multiaxis_model()

    fewer_rows = mbirtorch.copy_ct_model(model, new_num_det_rows=12)
    assert tuple(fewer_rows.get_params('sinogram_shape')) == (MA_CELL[0], 12, MA_CELL[2])
    assert np.asarray(fewer_rows.get_params('angles')).shape == (MA_CELL[0], 2)

    new_angles = _multiaxis_angles(num_views=8)
    fewer_views = mbirtorch.copy_ct_model(model, new_angles=new_angles)
    assert tuple(fewer_views.get_params('sinogram_shape')) == (8,) + MA_CELL[1:]
    assert np.allclose(np.asarray(fewer_views.get_params('angles')), new_angles)


def test_copy_ct_model_translation_takes_new_translation_vectors():
    """Translation's required parameters carry translation_vectors and no
    angles at all, so the copy has to read and write that key -- and the
    source/detector distances have to survive the round trip with it."""
    model = _translation_model()

    fewer_rows = mbirtorch.copy_ct_model(model, new_num_det_rows=20)
    assert tuple(fewer_rows.get_params('sinogram_shape')) == (16, 20, TCT_DETS[1])
    assert np.asarray(fewer_rows.get_params('translation_vectors')).shape == (16, 3)
    assert float(fewer_rows.get_params('source_detector_dist')) == 128.0
    assert float(fewer_rows.get_params('source_iso_dist')) == 32.0

    new_vectors = mbirtorch.gen_translation_vectors(3, 3, x_spacing=3.0, z_spacing=2.0)
    fewer_views = mbirtorch.copy_ct_model(model, new_translation_vectors=new_vectors)
    assert tuple(fewer_views.get_params('sinogram_shape')) == (9,) + TCT_DETS
    assert np.allclose(np.asarray(fewer_views.get_params('translation_vectors')), new_vectors)


def test_copy_ct_model_rejects_the_per_view_argument_that_does_not_apply():
    """Silently ignoring the wrong argument would return an unchanged copy
    and look like success, so each geometry refuses the other's."""
    with pytest.raises(ValueError, match='translations rather than angles'):
        mbirtorch.copy_ct_model(_translation_model(), new_angles=np.linspace(0, np.pi, 4))
    with pytest.raises(ValueError, match='TranslationModel only'):
        mbirtorch.copy_ct_model(_multiaxis_model(),
                                new_translation_vectors=np.zeros((4, 3), dtype=np.float32))


def test_copy_ct_model_still_rejects_a_class_it_does_not_support():
    """The refusal names the four supported classes, so a caller learns what
    to reach for instead of only what failed."""
    denoiser = mbirtorch.QGGMRFDenoiser((8, 8, 4))
    with pytest.raises(TypeError, match='MultiAxisParallelModel and TranslationModel'):
        mbirtorch.copy_ct_model(denoiser)


def test_get_ct_model_builds_multiaxis_and_translation():
    """Both new branches build the class their geometry_type names, with the
    same reconstruction geometry the constructor would have chosen."""
    angles = _multiaxis_angles()
    multiaxis = mbirtorch.get_ct_model('multiaxis', MA_CELL, angles)
    assert type(multiaxis).__name__ == 'MultiAxisParallelModel'
    assert tuple(multiaxis.get_params('recon_shape')) == \
        tuple(_multiaxis_model().get_params('recon_shape'))

    vectors = mbirtorch.gen_translation_vectors(4, 4, x_spacing=3.0, z_spacing=2.0)
    translation = mbirtorch.get_ct_model('translation', (vectors.shape[0],) + TCT_DETS,
                                         translation_vectors=vectors,
                                         source_detector_dist=128.0, source_iso_dist=32.0)
    assert type(translation).__name__ == 'TranslationModel'
    assert tuple(translation.get_params('recon_shape')) == \
        tuple(_translation_model().get_params('recon_shape'))


def test_get_ct_model_translation_needs_translation_vectors():
    """A translation geometry has no angles, so the omission has to be named
    rather than surfacing as a constructor TypeError about a positional."""
    with pytest.raises(ValueError, match='needs translation_vectors'):
        mbirtorch.get_ct_model('translation', (16,) + TCT_DETS,
                               source_detector_dist=128.0, source_iso_dist=32.0)


def test_get_ct_model_warns_on_multiaxis_z_shifts():
    """Same as parallel: axial shifts are a cone-only mode, and ignoring one
    silently would drop part of the caller's geometry."""
    with pytest.warns(UserWarning):
        mbirtorch.get_ct_model('multiaxis', MA_CELL, _multiaxis_angles(),
                               helical_z_shifts=np.zeros(MA_CELL[0]))
