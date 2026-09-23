"""Multiaxis-parallel gates: adjointness on every backend, agreement with the
parallel-beam model at zero elevation, and a recon smoke."""

import numpy as np
import torch

import mbirtorch


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def _small_ma(devices=("cpu",)):
    n_views = 16
    az = np.linspace(0, np.pi, n_views, endpoint=False)
    el = np.linspace(-0.5, 0.5, n_views)
    m = mbirtorch.MultiAxisParallelModel((n_views, 24, 20),
                                         np.stack([az, el], axis=1))
    m.configure_devices(devices=list(devices))
    m.set_params(no_warning=True, verbose=0)
    return m


def test_multiaxis_adjointness(device):
    torch.manual_seed(0)
    m = _small_ma([device])
    rs = m.get_params('recon_shape')
    idx = torch.as_tensor(mbirtorch.gen_full_indices(
        rs, use_ror_mask=m.get_params('use_ror_mask')), dtype=torch.int64,
        device=m.torch_device)
    x = torch.rand((idx.shape[0], rs[2]), device=m.torch_device)
    y = torch.rand(tuple(m.get_params('sinogram_shape')), device=m.torch_device)
    lhs = float(torch.sum(m.sparse_forward_project(x, idx) * y))
    rhs = float(torch.sum(x * m.sparse_back_project(y, idx)))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (lhs, rhs)


def test_multiaxis_zero_elevation_matches_parallel(device):
    """At zero elevation the geometry is parallel beam; the forward
    projections must agree at float level."""
    n_views = 12
    az = np.linspace(0, np.pi, n_views, endpoint=False)
    cell = (n_views, 16, 16)
    pm = mbirtorch.ParallelBeamModel(cell, az)
    pm.configure_devices(devices=[device])
    pm.set_params(no_warning=True, verbose=0)
    rs = tuple(pm.get_params('recon_shape'))
    mm = mbirtorch.MultiAxisParallelModel(cell, np.stack([az, np.zeros(n_views)], axis=1))
    mm.configure_devices(devices=[device])
    mm.set_params(no_warning=True, verbose=0, recon_shape=rs)
    vol = np.random.RandomState(0).rand(*rs).astype(np.float32)
    sino_p = np.asarray(pm.forward_project(vol))
    sino_m = np.asarray(mm.forward_project(vol))
    rel = _rel_max(sino_m, sino_p)
    print(f"multiaxis vs parallel at zero elevation rel_max = {rel:.2e}")
    assert rel < 1e-5


def test_multiaxis_recon_smoke(device):
    m = _small_ma([device])
    rs = m.get_params('recon_shape')
    phantom = mbirtorch.gen_translation_phantom(rs, 'dots', None, fill_rate=0.05)
    sino = m.forward_project(phantom)
    np.random.seed(0)
    recon, rd = m.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0,
                        logfile_path=None)
    fm = rd['recon_params']['fm_rmse']
    assert fm[-1] < fm[0]
    assert recon.shape == tuple(rs)
