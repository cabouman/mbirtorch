"""Tests for mbirtorch.preprocess.geometry_calibration.

The estimator recovers a known channel offset from synthetic data and leaves the caller's model
and sinogram as they were.

Everything runs on CPU with compile_mode='off', which keeps the suite fast.
"""

import warnings

import numpy as np
import pytest
import mbirtorch
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
