"""Recon smoke gates: the VCD loop runs on every available backend and reduces
the forward loss on a simple synthetic object, and it warns when its update
direction is too short."""

import warnings

import numpy as np
import pytest

import mbirtorch
from mbirtorch import tomography_model


def test_recon_reduces_loss(device):
    sino_shape = (40, 32, 32)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    recon_shape = model.get_params('recon_shape')

    # A simple centered-box phantom and its sinogram.
    phantom = np.zeros(tuple(recon_shape), dtype=np.float32)
    r0, c0, s0 = [n // 4 for n in recon_shape]
    phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    sinogram = model.forward_project(phantom)

    np.random.seed(0)
    recon, recon_dict = model.recon(sinogram, max_iterations=4,
                                    stop_threshold_change_pct=0.0)
    fm_rmse = recon_dict['recon_params']['fm_rmse']
    assert fm_rmse[-1] < fm_rmse[0], fm_rmse
    assert recon.shape == tuple(recon_shape)
    nrmse = float(np.linalg.norm(recon - phantom) / np.linalg.norm(phantom))
    assert nrmse < 0.5, nrmse


def test_recon_warns_when_the_direction_is_too_short(monkeypatch):
    """When the update direction is divided by c, the line search computes
    about c times the step size alpha.  recon must warn when, in more than
    half of the subsets of an iteration, alpha before the clamp is above 10.

    Here the update direction is divided by 1000, and alpha before the clamp
    is above 300 in every subset.  The same recon with the direction unchanged
    gives an alpha of at most about 1 and must not warn."""
    _, sinogram, params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='shepp-logan', num_views=24,
        num_det_rows=8, num_det_channels=32)
    model = mbirtorch.ParallelBeamModel(sinogram.shape, params['angles'])
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)

    def run():
        np.random.seed(0)
        model.recon(sinogram, max_iterations=2, stop_threshold_change_pct=0.0,
                    logfile_path=None, print_logs=False)

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='In more than half of the subsets')
        run()
    direction = model._get_update_direction
    monkeypatch.setattr(model, '_get_update_direction',
                        lambda *args, **kwargs: direction(*args, **kwargs) / 1000)
    with pytest.warns(RuntimeWarning, match='In more than half of the subsets'):
        run()


def test_the_warning_needs_more_than_half_of_the_subsets():
    """The warning counts subsets, so that a few subsets with a large alpha do
    not trigger it.  Of 16 subsets, 8 with a large alpha must not warn, and 9
    must."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='In more than half of the subsets')
        tomography_model._warn_if_most_alphas_large(8, 16)
    with pytest.warns(RuntimeWarning, match='In more than half of the subsets'):
        tomography_model._warn_if_most_alphas_large(9, 16)
