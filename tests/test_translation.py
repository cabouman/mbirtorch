"""Translation (TCT) gates: adjointness on every backend, cross-framework
goldens against mbirjax (single ops, FDK, auto geometry, and seeded
convergence parity), a recon smoke, and 2-shard vs 1-device parity."""

import glob
import os

import numpy as np
import pytest
import torch

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "golden_*.npz")))
_have_tct = bool(_paths) and "tct_sino" in np.load(_paths[0]).files


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


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


def test_translation_sharded_recon_matches_single_device():
    """2 CPU shards vs 1 device on the same seeded problem (the iterated
    comparison gate)."""
    ref_m = _small_tct(['cpu'])
    rs = ref_m.get_params('recon_shape')
    phantom = mbirtorch.gen_translation_phantom(rs, 'dots', None, fill_rate=0.05)
    sino = np.asarray(ref_m.forward_project(phantom))
    np.random.seed(0)
    ref, _ = ref_m.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0,
                         logfile_path=None)
    sh_m = _small_tct(['cpu', 'cpu'])
    np.random.seed(0)
    out, _ = sh_m.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0,
                        logfile_path=None)
    rel = _rel_max(out, ref)
    print(f"translation sharded vs single recon rel_max = {rel:.2e}")
    assert rel < 1e-4


tct_golden = pytest.mark.skipif(
    not _have_tct, reason="no translation goldens: rerun tests/generate_goldens.py")


@pytest.fixture(scope="module")
def golden():
    return np.load(_paths[0])


@pytest.fixture(scope="module")
def tct_model(golden):
    cell = tuple(int(x) for x in golden["tct_cell"])
    m = mbirtorch.TranslationModel(cell, golden["tct_tvecs"],
                                   source_detector_dist=float(golden["tct_sdd"]),
                                   source_iso_dist=float(golden["tct_sid"]))
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    return m


@pytest.mark.goldens
@tct_golden
def test_translation_auto_geometry(golden, tct_model):
    assert tuple(tct_model.get_params('recon_shape')) == \
        tuple(int(x) for x in golden["tct_recon_shape"])
    assert float(tct_model.get_params('delta_voxel')) == \
        pytest.approx(float(golden["tct_delta_voxel"]), rel=1e-6)
    assert float(tct_model.get_params('voxel_row_aspect')) == \
        pytest.approx(float(golden["tct_voxel_row_aspect"]), rel=1e-6)


@pytest.mark.goldens
@tct_golden
def test_translation_sparse_forward(golden, tct_model):
    out = tct_model.sparse_forward_project(golden["tct_vals"], golden["tct_subset"])
    err = _rel_max(out.numpy(), golden["tct_sp_fwd"])
    print(f"translation sparse_fwd rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@tct_golden
def test_translation_sparse_back(golden, tct_model):
    out = tct_model.sparse_back_project(golden["tct_sino"], golden["tct_subset"])
    err = _rel_max(out.numpy(), golden["tct_sp_back"])
    print(f"translation sparse_back rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@tct_golden
def test_translation_full_forward(golden, tct_model):
    out = tct_model.forward_project(golden["tct_phantom"])
    err = _rel_max(out, golden["tct_sino"])
    print(f"translation forward rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@tct_golden
def test_translation_fdk(golden, tct_model):
    out = tct_model.fdk_recon(golden["tct_sino"])
    err = _rel_max(out, golden["tct_fdk"])
    print(f"translation fdk rel_max = {err:.2e}")
    assert err < 1e-3


@pytest.mark.goldens
@tct_golden
def test_translation_recon_convergence_parity(golden, tct_model):
    np.random.seed(int(golden["recon_seed"]))
    recon, rd = tct_model.recon(golden["tct_sino"], max_iterations=3,
                                stop_threshold_change_pct=0.0, logfile_path=None)
    rp = rd['recon_params']
    alpha_rel = np.max(np.abs(np.array(rp['alpha_values']) - golden["tct_alpha"])
                       / np.abs(golden["tct_alpha"]))
    fm_rel = np.max(np.abs(np.array(rp['fm_rmse']) - golden["tct_fm_rmse"])
                    / np.abs(golden["tct_fm_rmse"]))
    final_rel = _rel_max(recon, golden["tct_recon"])
    print(f"translation recon parity: alpha rel = {alpha_rel:.2e}, "
          f"fm rel = {fm_rel:.2e}, final rel_max = {final_rel:.2e}")
    assert alpha_rel < 1e-2
    assert fm_rel < 1e-3
    assert final_rel < 1e-3
