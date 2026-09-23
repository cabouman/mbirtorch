"""Cone-beam gates: adjointness on every backend and a recon smoke."""

import numpy as np
import torch

import mbirtorch


def _small_cone(device="cpu"):
    cell = (24, 16, 16)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    m = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                source_iso_dist=2 * cell[2])
    m.configure_devices(devices=[device])
    m.set_params(no_warning=True, verbose=0)
    return m


def test_cone_adjointness(device):
    torch.manual_seed(0)
    m = _small_cone(device)
    rs = m.get_params('recon_shape')
    idx = torch.as_tensor(mbirtorch.gen_full_indices(rs), dtype=torch.int64,
                          device=m.torch_device)
    x = torch.rand((idx.shape[0], rs[2]), device=m.torch_device)
    y = torch.rand(tuple(m.get_params('sinogram_shape')), device=m.torch_device)
    lhs = float(torch.sum(m.sparse_forward_project(x, idx) * y))
    rhs = float(torch.sum(x * m.sparse_back_project(y, idx)))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (lhs, rhs)


def test_cone_recon_smoke(device):
    m = _small_cone(device)
    rs = m.get_params('recon_shape')
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rs)
    sino = m.forward_project(phantom)
    np.random.seed(0)
    recon, rd = m.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0)
    fm = rd['recon_params']['fm_rmse']
    assert fm[-1] < fm[0]
    assert recon.shape == tuple(rs)
