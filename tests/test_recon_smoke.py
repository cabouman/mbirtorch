"""Recon smoke gate: the VCD loop runs on every available backend and reduces
the forward loss on a simple synthetic object."""

import numpy as np

import mbirtorch


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
