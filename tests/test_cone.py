"""Cone-beam gates: adjointness on every backend, cross-framework goldens
against mbirjax (single ops, FDK, auto geometry, and seeded convergence
parity), and a recon smoke."""

import glob
import os

import numpy as np
import pytest
import torch

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "golden_*.npz")))
_have_cone = bool(_paths) and "cone_sino" in np.load(_paths[0]).files


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


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


cone_golden = pytest.mark.skipif(
    not _have_cone, reason="no cone goldens: rerun tests/generate_goldens.py")


@pytest.fixture(scope="module")
def golden():
    return np.load(_paths[0])


@pytest.fixture(scope="module")
def cone_model(golden):
    cell = tuple(int(x) for x in golden["cone_cell"])
    m = mbirtorch.ConeBeamModel(cell, golden["cone_angles"],
                                source_detector_dist=float(golden["cone_sdd"]),
                                source_iso_dist=float(golden["cone_sid"]))
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    return m


@pytest.mark.goldens
@cone_golden
def test_cone_auto_geometry(golden, cone_model):
    assert tuple(cone_model.get_params('recon_shape')) == \
        tuple(int(x) for x in golden["cone_recon_shape"])
    rel = abs(cone_model.get_params('recon_slice_offset')
              - float(golden["cone_slice_offset"]))
    assert rel < 1e-6


@pytest.mark.goldens
@cone_golden
def test_cone_sparse_forward(golden, cone_model):
    out = cone_model.sparse_forward_project(golden["cone_vals"], golden["cone_subset"])
    err = _rel_max(out.numpy(), golden["cone_sp_fwd"])
    print(f"cone sparse_fwd rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@cone_golden
def test_cone_sparse_back(golden, cone_model):
    out = cone_model.sparse_back_project(golden["cone_sino"], golden["cone_subset"])
    err = _rel_max(out.numpy(), golden["cone_sp_back"])
    print(f"cone sparse_back rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@cone_golden
def test_cone_full_forward(golden, cone_model):
    out = cone_model.forward_project(golden["cone_phantom"])
    err = _rel_max(out, golden["cone_sino"])
    print(f"cone forward rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@cone_golden
def test_cone_hessian(golden, cone_model):
    out = cone_model.compute_hessian_diagonal(
        weights=torch.as_tensor(golden["cone_weights"]))
    err = _rel_max(out, golden["cone_hess"])
    print(f"cone hessian rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@cone_golden
def test_cone_fdk(golden, cone_model):
    out = cone_model.fdk_recon(golden["cone_sino"])
    err = _rel_max(out, golden["cone_fdk"])
    print(f"cone fdk rel_max = {err:.2e}")
    assert err < 1e-3


@pytest.mark.goldens
@cone_golden
def test_cone_recon_convergence_parity(golden, cone_model):
    np.random.seed(int(golden["recon_seed"]))
    recon, rd = cone_model.recon(golden["cone_sino"],
                                 weights=golden["cone_weights"],
                                 max_iterations=int(golden["max_iterations"]),
                                 stop_threshold_change_pct=0.0)
    rp = rd["recon_params"]
    alpha_rel = np.max(np.abs(np.array(rp["alpha_values"]) - golden["cone_alpha"])
                       / np.abs(golden["cone_alpha"]))
    fm_rel = np.max(np.abs(np.array(rp["fm_rmse"]) - golden["cone_fm_rmse"])
                    / np.abs(golden["cone_fm_rmse"]))
    final_rel = _rel_max(recon, golden["cone_recon"])
    print(f"cone parity: alpha {alpha_rel:.2e}, fm {fm_rel:.2e}, "
          f"final {final_rel:.2e}")
    assert alpha_rel < 1e-2
    assert fm_rel < 1e-3
    assert final_rel < 1e-3


def test_helical_z_weight_zeroes_padded_slices():
    # A sharding port zero-pads the recon slice axis; the z-weight must force
    # padded slices to zero (the forced-zero invariant) even if an upstream bug
    # leaves them nonzero, and must leave the real slices' weights unchanged.
    cell = (24, 16, 16)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    z_shifts = np.linspace(-4.0, 4.0, cell[0]).astype(np.float32)
    m = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                source_iso_dist=2 * cell[2],
                                helical_z_shifts=z_shifts)
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    rs = tuple(m.get_params('recon_shape'))
    sino = torch.zeros(tuple(m.get_params('sinogram_shape')))
    torch.manual_seed(0)
    recon = torch.rand(rs)
    ref = m.helical_fdk_z_weight(recon.clone(), sino)

    extra = 3
    padded = torch.ones(rs[:2] + (rs[2] + extra,))
    padded[:, :, :rs[2]] = recon
    out = m.helical_fdk_z_weight(padded, sino)
    assert float(torch.abs(out[:, :, rs[2]:]).max()) == 0.0
    assert torch.allclose(out[:, :, :rs[2]], ref)


# ── helical and curved-detector golden coverage ──────────────────────────────
_have_hel = bool(_paths) and "chel_sino" in np.load(_paths[0]).files
hel_golden = pytest.mark.skipif(
    not _have_hel, reason="no helical/curved goldens: rerun tests/generate_goldens.py")


@pytest.fixture(scope="module")
def helical_model(golden):
    cell = tuple(int(x) for x in golden["cone_cell"])
    m = mbirtorch.ConeBeamModel(cell, golden["cone_angles"],
                                source_detector_dist=float(golden["cone_sdd"]),
                                source_iso_dist=float(golden["cone_sid"]),
                                helical_z_shifts=golden["chel_shifts"])
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    return m


@pytest.fixture(scope="module")
def curved_model(golden):
    cell = tuple(int(x) for x in golden["cone_cell"])
    m = mbirtorch.ConeBeamModel(cell, golden["cone_angles"],
                                source_detector_dist=float(golden["cone_sdd"]),
                                source_iso_dist=float(golden["cone_sid"]),
                                use_curved_detector=True)
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    return m


@hel_golden
def test_helical_auto_geometry(golden, helical_model):
    assert tuple(helical_model.get_params('recon_shape')) == \
        tuple(int(x) for x in golden["chel_recon_shape"])


@hel_golden
def test_helical_sparse_forward(golden, helical_model):
    out = helical_model.sparse_forward_project(golden["chel_vals"],
                                               golden["chel_subset"])
    err = _rel_max(out.numpy(), golden["chel_sp_fwd"])
    print(f"helical sparse_fwd rel_max = {err:.2e}")
    assert err < 1e-4


@hel_golden
def test_helical_sparse_back(golden, helical_model):
    out = helical_model.sparse_back_project(golden["chel_sino"],
                                            golden["chel_subset"])
    err = _rel_max(out.numpy(), golden["chel_sp_back"])
    print(f"helical sparse_back rel_max = {err:.2e}")
    assert err < 1e-4


@hel_golden
def test_helical_full_forward(golden, helical_model):
    out = helical_model.forward_project(golden["chel_phantom"])
    err = _rel_max(out, golden["chel_sino"])
    print(f"helical forward rel_max = {err:.2e}")
    assert err < 1e-4


@hel_golden
def test_helical_fdk(golden, helical_model):
    # Exercises the helical z-weight (nonuniform per-slice coverage).
    out = helical_model.fdk_recon(golden["chel_sino"])
    err = _rel_max(out, golden["chel_fdk"])
    print(f"helical fdk rel_max = {err:.2e}")
    assert err < 1e-3


@hel_golden
def test_helical_recon_convergence_parity(golden, helical_model):
    np.random.seed(int(golden["recon_seed"]))
    recon, rd = helical_model.recon(golden["chel_sino"],
                                    weights=golden["chel_weights"],
                                    max_iterations=3,
                                    stop_threshold_change_pct=0.0)
    rp = rd["recon_params"]
    alpha_rel = np.max(np.abs(np.array(rp["alpha_values"]) - golden["chel_alpha"])
                       / np.abs(golden["chel_alpha"]))
    fm_rel = np.max(np.abs(np.array(rp["fm_rmse"]) - golden["chel_fm_rmse"])
                    / np.abs(golden["chel_fm_rmse"]))
    final_rel = _rel_max(recon, golden["chel_recon"])
    print(f"helical parity: alpha {alpha_rel:.2e}, fm {fm_rel:.2e}, "
          f"final {final_rel:.2e}")
    assert alpha_rel < 1e-2
    assert fm_rel < 1e-3
    assert final_rel < 1e-3


@hel_golden
def test_curved_sparse_forward(golden, curved_model):
    out = curved_model.sparse_forward_project(golden["ccurv_vals"],
                                              golden["ccurv_subset"])
    err = _rel_max(out.numpy(), golden["ccurv_sp_fwd"])
    print(f"curved sparse_fwd rel_max = {err:.2e}")
    assert err < 1e-4


@hel_golden
def test_curved_sparse_back(golden, curved_model):
    out = curved_model.sparse_back_project(golden["ccurv_sino"],
                                           golden["ccurv_subset"])
    err = _rel_max(out.numpy(), golden["ccurv_sp_back"])
    print(f"curved sparse_back rel_max = {err:.2e}")
    assert err < 1e-4


@hel_golden
def test_curved_full_forward(golden, curved_model):
    out = curved_model.forward_project(golden["ccurv_phantom"])
    err = _rel_max(out, golden["ccurv_sino"])
    print(f"curved forward rel_max = {err:.2e}")
    assert err < 1e-4


@hel_golden
def test_curved_fdk(golden, curved_model):
    out = curved_model.fdk_recon(golden["ccurv_sino"])
    err = _rel_max(out, golden["ccurv_fdk"])
    print(f"curved fdk rel_max = {err:.2e}")
    assert err < 1e-3
