"""Translation (TCT) gates: adjointness on every backend and a recon smoke."""

import numpy as np
import torch

import mbirtorch


def _small_tct(devices=("cpu",)):
    tvecs = mbirtorch.gen_translation_vectors(4, 4, x_spacing=3.0, z_spacing=2.0)
    cell = (tvecs.shape[0], 40, 32)
    m = mbirtorch.TranslationModel(cell, tvecs, source_detector_dist=128.0,
                                   source_iso_dist=32.0)
    m.configure_devices(devices=list(devices))
    m.set_params(no_warning=True, verbose=0)
    return m


def test_translation_adjointness(device):
    torch.manual_seed(0)
    m = _small_tct([device])
    rs = m.get_params('recon_shape')
    idx = torch.as_tensor(mbirtorch.gen_full_indices(
        rs, use_ror_mask=m.get_params('use_ror_mask')), dtype=torch.int64,
        device=m.torch_device)
    x = torch.rand((idx.shape[0], rs[2]), device=m.torch_device)
    y = torch.rand(tuple(m.get_params('sinogram_shape')), device=m.torch_device)
    lhs = float(torch.sum(m.sparse_forward_project(x, idx) * y))
    rhs = float(torch.sum(x * m.sparse_back_project(y, idx)))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (lhs, rhs)


def test_translation_recon_smoke(device):
    m = _small_tct([device])
    rs = m.get_params('recon_shape')
    phantom = mbirtorch.gen_translation_phantom(rs, 'dots', None, fill_rate=0.05)
    sino = m.forward_project(phantom)
    np.random.seed(0)
    recon, rd = m.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0,
                        logfile_path=None)
    fm = rd['recon_params']['fm_rmse']
    assert fm[-1] < fm[0]
    assert recon.shape == tuple(rs)
