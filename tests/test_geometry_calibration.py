"""Tests for mbirtorch.preprocess.geometry_calibration.

The channel offset estimator recovers a known channel offset from synthetic data and leaves the
caller's model and sinogram as they were.  The alignment estimator recovers known offsets, per-view
shifts, and a rotation when the phantom itself is the reference, and the resampler undoes them.
The deprecated align_sino_views warns and still works.

Everything runs on CPU with compile_mode='off', which keeps the suite fast.
"""

import warnings

import numpy as np
import pytest
import mbirtorch
import mbirtorch.preprocess as mtp
from mbirtorch.preprocess.geometry_calibration import estimate_det_channel_offset


# The models are small, but they cover a full rotation of views with enough channels to leave an
# interior after the edge margin.  One forward projection takes a fraction of a second.

def _parallel_model(det_channel_offset):
    """A 64-view parallel model over a full rotation, at the given channel offset in ALU."""
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    model = mbirtorch.ParallelBeamModel((64, 16, 64), angles, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset)
    return model


def _cone_model(det_channel_offset):
    """A 128-view cone model over a full rotation, at the given channel offset in ALU.  The source
    distance puts the 64 channels of the detector across a full fan angle of 20 degrees."""
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    source_detector_dist = 64 / 2 / np.tan(np.deg2rad(10.0))
    model = mbirtorch.ConeBeamModel((128, 16, 64), angles,
                                    source_detector_dist=source_detector_dist,
                                    source_iso_dist=source_detector_dist / 2, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset)
    return model


_SINOGRAMS = {}


def _sinogram(geometry, true_offset):
    """The Shepp-Logan sinogram of one geometry at one true channel offset, projected once per run."""
    key = (geometry, true_offset)
    if key not in _SINOGRAMS:
        model = (_parallel_model if geometry == 'parallel' else _cone_model)(true_offset)
        phantom = mbirtorch.gen_shepp_logan_3d(model.get_params('recon_shape'))
        _SINOGRAMS[key] = np.asarray(model.forward_project(phantom), dtype=np.float32)
    return _SINOGRAMS[key]


def _copy_params(all_params):
    """A deep enough copy of a get_all_params result to compare against later."""
    return [{name: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
             for name, value in group.items()} for group in all_params]


def _params_equal(before, after):
    """True when two get_all_params results hold the same names and equal values."""
    for group_before, group_after in zip(before, after):
        if set(group_before) != set(group_after):
            return False
        for name, value in group_before.items():
            other = group_after[name]
            if value is None or other is None:
                if value is not other:
                    return False
            elif not np.array_equal(value, other):
                return False
    return True


@pytest.mark.parametrize('true_offset', [1.3, -2.2])
def test_estimate_det_channel_offset_on_parallel_beam(true_offset):
    """On parallel-beam data the estimator recovers the offset the data were simulated with, to
    within a tenth of a channel, from clean data and from data with 2 percent Gaussian noise, and
    it raises no warning."""
    sino = _sinogram('parallel', true_offset)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        estimate = estimate_det_channel_offset(_parallel_model(0.0), sino)
    print(f'parallel offset {true_offset}: estimate {estimate:.4f}, error {estimate - true_offset:.4f} channels')
    assert isinstance(estimate, float)
    assert abs(estimate - true_offset) < 0.1

    noise = np.random.default_rng(0).normal(0.0, 0.02 * float(sino.max()), sino.shape)
    noisy_sino = (sino + noise).astype(np.float32)
    noisy_estimate = estimate_det_channel_offset(_parallel_model(0.0), noisy_sino)
    print(f'parallel offset {true_offset} with noise: error {noisy_estimate - true_offset:.4f} channels')
    assert abs(noisy_estimate - true_offset) < 0.1


@pytest.mark.parametrize('true_offset', [3.0, -1.7])
def test_estimate_det_channel_offset_on_cone_beam(true_offset):
    """On cone-beam data with a 20 degree fan the estimator recovers the offset the data were
    simulated with, to within a tenth of a channel, starting from zero and from ten channels off."""
    sino = _sinogram('cone', true_offset)
    for start in (0.0, true_offset - 10.0):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            estimate = estimate_det_channel_offset(_cone_model(start), sino)
        print(f'cone offset {true_offset} from {start}: estimate {estimate:.4f}, error {estimate - true_offset:.4f} channels')
        assert abs(estimate - true_offset) < 0.1


def test_estimator_does_not_change_the_caller_state():
    """The estimator leaves the model's parameters and the sinogram as they were."""
    model = _cone_model(0.0)
    sino = _sinogram('cone', 0.0)
    params_before = _copy_params(model.get_all_params())
    sino_before = sino.copy()

    estimate_det_channel_offset(model, sino)

    assert _params_equal(params_before, model.get_all_params())
    assert np.array_equal(sino, sino_before)
    assert model.get_params('det_channel_offset') == 0.0


def test_unsuitable_scan_returns_the_unchanged_value_with_a_warning():
    """On a scan over half a rotation, which has no opposite views, the estimator returns the
    model's current offset with a warning."""
    angles = np.linspace(0, np.pi, 32, endpoint=False)
    model = mbirtorch.ParallelBeamModel((32, 16, 64), angles, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=0.7)
    sino = np.zeros((32, 16, 64), dtype=np.float32)
    with pytest.warns(UserWarning, match='opposite view'):
        assert estimate_det_channel_offset(model, sino) == 0.7


# ── alignment from reprojection ───────────────────────────────────────────────────────────────────

# A taller detector than the models above, since the fit works on whole views.  The phantom itself
# is the reference, so the tests check the signs, the units, and the rotation center of the two
# functions, not the quality of a first reconstruction.

_ALIGN_SHAPE = (48, 32, 64)


def _align_model(det_channel_offset=0.0, det_row_offset=0.0):
    """A 48-view cone model over a full rotation at the given offsets in ALU."""
    angles = np.linspace(0, 2 * np.pi, _ALIGN_SHAPE[0], endpoint=False)
    source_detector_dist = _ALIGN_SHAPE[2] / 2 / np.tan(np.deg2rad(10.0))
    model = mbirtorch.ConeBeamModel(_ALIGN_SHAPE, angles, source_detector_dist=source_detector_dist,
                                    source_iso_dist=source_detector_dist / 2, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset,
                     det_row_offset=det_row_offset)
    return model


@pytest.fixture(scope='module')
def align_case():
    """The phantom and its sinogram at zero offsets."""
    model = _align_model()
    phantom = np.asarray(mbirtorch.gen_shepp_logan_3d(model.get_params('recon_shape')), dtype=np.float32)
    sino = np.asarray(model.forward_project(phantom), dtype=np.float32)
    return phantom, sino


def test_alignment_recovers_global_offsets(align_case):
    # Data from a model at offsets (3, 2) fit against a model at (0, 0): the global values come
    # back and the per-view deviations stay small.
    phantom, _ = align_case
    sino = np.asarray(_align_model(3.0, 2.0).forward_project(phantom), dtype=np.float32)
    model_params, view_params = mtp.fit_det_alignment(_align_model(), sino, phantom)
    assert model_params['det_channel_offset'] == pytest.approx(3.0, abs=0.1)
    assert model_params['det_row_offset'] == pytest.approx(2.0, abs=0.1)
    assert set(view_params) == {'det_channel_offset', 'det_row_offset'}
    assert np.abs(view_params['det_channel_offset']).max() < 0.15
    assert np.abs(view_params['det_row_offset']).max() < 0.15


def test_alignment_recovers_a_rotation_and_the_resampler_undoes_it(align_case):
    # Views rotated with correct_det_rotation come back with that angle, and after the correction a
    # second fit finds nothing left.
    phantom, sino = align_case
    theta = 0.02
    rotated = mtp.correct_det_rotation(sino, det_rotation=-theta)
    model = _align_model()
    model_params, view_params = mtp.fit_det_alignment(model, rotated, phantom, rotation=True)
    assert np.median(view_params['det_rotation']) == pytest.approx(theta, abs=0.002)
    assert abs(model_params['det_channel_offset']) < 0.1 and abs(model_params['det_row_offset']) < 0.1
    corrected = mtp.correct_det_alignment(model, rotated, view_params)
    assert corrected.shape == sino.shape and corrected.dtype == np.float32
    _, residual = mtp.fit_det_alignment(model, corrected, phantom, rotation=True)
    assert abs(np.median(residual['det_rotation'])) < 0.0005
    assert np.abs(residual['det_channel_offset']).max() < 0.05


def test_alignment_recovers_per_view_shifts(align_case):
    # Each view moved by its own (row, channel) shift, zero mean over views, comes back as that
    # deviation in ALU, and align_sino_views puts the views back.
    from scipy.ndimage import shift as nd_shift
    phantom, sino = align_case
    rng = np.random.default_rng(1)
    shifts = rng.uniform(-1.5, 1.5, size=(sino.shape[0], 2))
    shifts -= shifts.mean(axis=0)
    moved = np.stack([nd_shift(view, s, order=1, mode='nearest') for view, s in zip(sino, shifts)])
    model = _align_model()
    model_params, view_params = mtp.fit_det_alignment(model, moved, phantom)
    assert abs(model_params['det_channel_offset']) < 0.15 and abs(model_params['det_row_offset']) < 0.2
    assert np.abs(view_params['det_row_offset'] - shifts[:, 0]).max() < 0.25
    assert np.abs(view_params['det_channel_offset'] - shifts[:, 1]).max() < 0.25
    with pytest.warns(DeprecationWarning, match='align_sino_views is deprecated'):
        aligned = mtp.align_sino_views(model, moved, phantom)
    interior = (slice(None), slice(3, -3), slice(3, -3))
    before = np.sqrt(np.mean((moved - sino)[interior] ** 2))
    after = np.sqrt(np.mean((aligned - sino)[interior] ** 2))
    assert after < 0.4 * before


def test_alignment_warns_on_a_view_that_does_not_match(align_case):
    # A view replaced by noise cannot be aligned: it is named in a warning and gets no per-view
    # correction, while the other views are unaffected.
    phantom, sino = align_case
    broken = sino.copy()
    broken[5] = np.random.default_rng(2).normal(size=sino.shape[1:]).astype(np.float32)
    with pytest.warns(UserWarning, match='failed for 1 of 48 views, starting with view 5'):
        model_params, view_params = mtp.fit_det_alignment(_align_model(), broken, phantom, rotation=True)
    assert view_params['det_channel_offset'][5] == 0.0 and view_params['det_row_offset'][5] == 0.0
    assert view_params['det_rotation'][5] == np.median(view_params['det_rotation'])
    assert abs(model_params['det_channel_offset']) < 0.1


def test_alignment_leaves_the_model_unchanged(align_case):
    phantom, sino = align_case
    model = _align_model(0.5, -0.25)
    before = _copy_params(model.get_all_params())
    mtp.fit_det_alignment(model, sino, phantom, rotation=True)
    mtp.correct_det_alignment(model, sino, {'det_rotation': np.full(sino.shape[0], 0.01)})
    assert _params_equal(before, model.get_all_params())
