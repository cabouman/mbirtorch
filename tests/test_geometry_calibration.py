"""Tests for mbirtorch.preprocess.geometry_calibration.

The module builds a reduced problem, reduces a sinogram to match it, sweeps a geometry parameter,
checks the rotation direction, and estimates the detector channel offset and rotation.  Four things
are checked here: a candidate equal to the model's current value reproduces the full direct
reconstruction, nothing except apply_calibration changes the caller's model or sinogram, the
rotation check recovers the direction the data were simulated with, and the two estimators recover a
known offset and a known rotation.

Everything runs on CPU with compile_mode='off', which keeps the suite fast and is inherited by the
reduced models the module builds.
"""

import numpy as np
import pytest
import mbirtorch
from mbirtorch.preprocess.geometry_calibration import (build_reduced_problem,
                                                      check_rotation_direction,
                                                      conjugate_difference,
                                                      estimate_det_channel_offset,
                                                      estimate_det_rotation,
                                                      parameter_sweep,
                                                      reduce_sinogram)
from mbirtorch.preprocess.utilities import correct_det_rotation

NUM_VIEWS = 32
NUM_ROWS = 16
NUM_CHANNELS = 32


def _rel_max(out, ref):
    """The largest absolute difference divided by the largest absolute reference value."""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def _make_model(geometry, compile_mode='off'):
    """Build a small CPU model of the requested geometry."""
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    sinogram_shape = (NUM_VIEWS, NUM_ROWS, NUM_CHANNELS)
    if geometry == 'parallel':
        model = mbirtorch.ParallelBeamModel(sinogram_shape, angles, compile_mode=compile_mode)
    else:
        model = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                        source_detector_dist=4 * NUM_CHANNELS,
                                        source_iso_dist=2 * NUM_CHANNELS,
                                        compile_mode=compile_mode)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


def _phantom_sinogram(model):
    """Forward project the Shepp-Logan phantom that fits the model's recon shape."""
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(model.get_params('recon_shape'))
    return np.asarray(model.forward_project(phantom), dtype=np.float32)


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


@pytest.fixture(scope='module')
def cone_model():
    return _make_model('cone')


@pytest.fixture(scope='module')
def parallel_model():
    return _make_model('parallel')


@pytest.fixture(scope='module')
def cone_sino(cone_model):
    return _phantom_sinogram(cone_model)


@pytest.fixture(scope='module')
def parallel_sino(parallel_model):
    return _phantom_sinogram(parallel_model)


@pytest.fixture(params=['parallel', 'cone'])
def model_and_sino(request):
    """The base model of one geometry and its sinogram, built once per module."""
    model = request.getfixturevalue(f'{request.param}_model')
    sino = request.getfixturevalue(f'{request.param}_sino')
    return request.param, model, sino


# ── sweep parity with the full direct reconstruction ──────────────────────────────────────────────

def test_sweep_matches_full_reconstruction(model_and_sino):
    """The candidate that equals the model's current parameter value reproduces the full model's
    direct reconstruction of the requested slice."""
    geometry, model, sino = model_and_sino
    num_slices = model.get_params('recon_shape')[2]
    middle = (num_slices - 1) // 2
    full_recon = np.asarray(model.recon_direct(sino))

    channel_stack = parameter_sweep(model, sino, 'det_channel_offset', [-1.0, 0.0, 1.0])
    channel_error = _rel_max(channel_stack[:, :, 1], full_recon[:, :, middle])
    print(f'{geometry} det_channel_offset parity rel_max = {channel_error:.2e}')
    assert channel_error <= 1e-5

    rotation_stack = parameter_sweep(model, sino, 'det_rotation', [0.0, 0.02])
    rotation_error = _rel_max(rotation_stack[:, :, 0], full_recon[:, :, middle])
    print(f'{geometry} det_rotation parity rel_max = {rotation_error:.2e}')
    assert rotation_error <= 1e-5

    if geometry == 'cone':
        row_stack = parameter_sweep(model, sino, 'det_row_offset', [-1.0, 0.0, 1.0])
        row_error = _rel_max(row_stack[:, :, 1], full_recon[:, :, middle])
        print(f'{geometry} det_row_offset parity rel_max = {row_error:.2e}')
        assert row_error <= 1e-5

    chosen_stack = parameter_sweep(model, sino, 'det_channel_offset', [0.0], slice_index=3)
    chosen_error = _rel_max(chosen_stack[:, :, 0], full_recon[:, :, 3])
    print(f'{geometry} slice_index=3 parity rel_max = {chosen_error:.2e}')
    assert chosen_error <= 1e-5


# ── no state change ───────────────────────────────────────────────────────────────────────────────

def test_functions_do_not_change_the_caller_state(cone_model, cone_sino):
    """Only apply_calibration changes state.  The reduction, the sweep, the rotation check, the
    conjugate-view estimator, and the difference image all leave the model's parameters and the
    sinogram as they were."""
    params_before = _copy_params(cone_model.get_all_params())
    sino_before = cone_sino.copy()

    _, reduction = build_reduced_problem(cone_model, view_stride=2, bin_factor=2,
                                         num_slab_slices=3)
    reduce_sinogram(cone_sino, reduction)
    parameter_sweep(cone_model, cone_sino, 'det_channel_offset', [-1.0, 0.0, 1.0])
    check_rotation_direction(cone_model, cone_sino, view_stride=2, bin_factor=2)
    estimate_det_channel_offset(cone_model, cone_sino)
    conjugate_difference(cone_model, cone_sino)

    assert _params_equal(params_before, cone_model.get_all_params())
    assert np.array_equal(cone_sino, sino_before)
    assert cone_model.get_params('det_channel_offset') == 0.0


# ── the rotation-direction check ──────────────────────────────────────────────────────────────────

def test_check_rotation_direction_on_cone_data(cone_model, cone_sino):
    """The check picks the direction the data were simulated with, for the angles as given and for
    the angles negated, and the two scores are separated by at least a factor of 1.5."""
    result = check_rotation_direction(cone_model, cone_sino, view_stride=2, bin_factor=1)
    print(f'forward scores = {result.scores}, ratio = {result.scores[1] / result.scores[0]:.2f}')
    assert result.value == 1.0
    assert np.array_equal(result.candidates, np.array([1.0, -1.0]))
    assert result.method == 'direct_residual'
    assert result.scores[1] / result.scores[0] >= 1.5

    angles = np.asarray(cone_model.get_all_params()[0]['angles'])
    reversed_model = mbirtorch.copy_ct_model(cone_model, new_angles=-angles,
                                             new_helical_z_shifts=np.zeros_like(angles))
    reversed_model.compile_mode = 'off'
    reversed_model.configure_devices(devices=['cpu'])
    reversed_model.set_params(no_warning=True, verbose=0)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        reversed_model.get_params('recon_shape'))
    reversed_sino = np.asarray(reversed_model.forward_project(phantom), dtype=np.float32)

    reversed_result = check_rotation_direction(cone_model, reversed_sino, view_stride=2,
                                               bin_factor=1)
    print(f'reversed scores = {reversed_result.scores}, '
          f'ratio = {reversed_result.scores[0] / reversed_result.scores[1]:.2f}')
    assert reversed_result.value == -1.0


# The models below are larger than the ones above, because the conjugate-view comparison needs a
# full rotation of views and enough channels to leave an interior after the edge margin.  They are
# still small enough that one forward projection takes a fraction of a second.

def _conjugate_parallel_model(det_channel_offset):
    """A 64-view parallel model over a full rotation, at the given channel offset in ALU."""
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    model = mbirtorch.ParallelBeamModel((64, 16, 64), angles, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset)
    return model


def _conjugate_cone_model(det_channel_offset, use_curved_detector=False):
    """A 128-view cone model over a full rotation, at the given channel offset in ALU.  The source
    distance puts the 64 channels of the detector across a full fan angle of 20 degrees."""
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    source_detector_dist = 64 / 2 / np.tan(np.deg2rad(10.0))
    model = mbirtorch.ConeBeamModel((128, 16, 64), angles,
                                    source_detector_dist=source_detector_dist,
                                    source_iso_dist=source_detector_dist / 2,
                                    use_curved_detector=use_curved_detector, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, det_channel_offset=det_channel_offset)
    return model


_CONJUGATE_SINOGRAMS = {}


def _conjugate_sinogram(geometry, true_offset):
    """The phantom sinogram of one geometry at one true channel offset, projected once per run."""
    key = (geometry, true_offset)
    if key not in _CONJUGATE_SINOGRAMS:
        build = _conjugate_parallel_model if geometry == 'parallel' else _conjugate_cone_model
        _CONJUGATE_SINOGRAMS[key] = _phantom_sinogram(build(true_offset))
    return _CONJUGATE_SINOGRAMS[key]


def _check_offset_result(result, true_offset, tolerance):
    """Check that an offset estimate is within tolerance of the true value and that the search
    record is well formed.  Returns the signed error in ALU."""
    assert result.parameter == 'det_channel_offset'
    assert result.method == 'conjugate'
    assert np.all(np.diff(result.candidates) > 0)
    assert result.candidates.size == result.scores.size
    assert result.score == result.scores.min()
    assert result.reduction['search_notes'] == []
    error = result.value - true_offset
    assert abs(error) < tolerance
    return error


@pytest.mark.parametrize('true_offset', [1.3, -2.2])
def test_estimate_det_channel_offset_on_parallel_beam(true_offset):
    """On parallel-beam data the conjugate-view estimator recovers the offset the data were
    simulated with, from clean data and from data with 2 percent Gaussian noise."""
    sino = _conjugate_sinogram('parallel', true_offset)
    result = estimate_det_channel_offset(_conjugate_parallel_model(0.0), sino)
    error = _check_offset_result(result, true_offset, 0.1)
    print(f'parallel offset {true_offset}: estimate {result.value:.4f}, error {error:.4f} channels')

    noise = np.random.default_rng(0).normal(0.0, 0.02 * float(sino.max()), sino.shape)
    noisy_sino = (sino + noise).astype(np.float32)
    noisy_result = estimate_det_channel_offset(_conjugate_parallel_model(0.0), noisy_sino)
    noisy_error = _check_offset_result(noisy_result, true_offset, 0.1)
    print(f'parallel offset {true_offset} with noise: error {noisy_error:.4f} channels')


# ── the rotation estimate ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('geometry', ['parallel', 'cone'])
def test_estimate_det_rotation_recovers_a_rotation(geometry):
    """A rotation of 2 or -3 degrees applied to the sinogram is recovered to within 12 percent,
    which is the accuracy the cubic resampling gives when the rotation displaces the edge pixel by
    more than half a pixel.  A rotation of zero is recovered to within 0.05 degrees.  A rotation
    of 0.3 degrees displaces the edge pixel of this detector by 0.17 pixels, and the estimate then
    carries a warning about the sub-pixel regime."""
    if geometry == 'parallel':
        model = _conjugate_parallel_model(0.0)
    else:
        model = _conjugate_cone_model(0.0)
    sino = _conjugate_sinogram(geometry, 0.0)
    for true_degrees in (2.0, -3.0):
        tilted = correct_det_rotation(sino, -np.radians(true_degrees))
        result = estimate_det_rotation(model, tilted)
        estimate = np.degrees(result.value)
        print(f'{geometry} rotation {true_degrees:+.1f} degrees: estimate {estimate:+.3f}')
        assert result.parameter == 'det_rotation' and result.method == 'conjugate'
        assert abs(estimate - true_degrees) < 0.12 * abs(true_degrees)
        assert result.reduction['search_notes'] == []
    result = estimate_det_rotation(model, sino)
    print(f'{geometry} rotation 0.0 degrees: estimate {np.degrees(result.value):+.4f}')
    assert abs(np.degrees(result.value)) < 0.05
    with pytest.warns(UserWarning, match='edge channels'):
        result = estimate_det_rotation(model, correct_det_rotation(sino, -np.radians(0.3)))
    print(f'{geometry} rotation 0.3 degrees: estimate {np.degrees(result.value):+.3f}, with a warning')
