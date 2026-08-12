"""Multiaxis-parallel gates: adjointness on every backend, cross-framework
goldens against mbirjax (single ops, FBP, auto geometry, and seeded
convergence parity), a recon smoke, and 2-shard vs 1-device parity.

The two seeded-reconstruction gates are each set from the parity MEASURED at
the configuration that gate runs on, rather than sharing one number, because
the two configurations differ by more than an order of magnitude:

  * The GOLDEN configuration (24 views, elevations to +-0.4 rad) matches
    mbirjax to 1.1e-5 max on the volume at 3 iterations, decaying to 6.8e-6
    by 10.  Its volume gate is 2e-4, about 18x the measured value.
  * The SHARDED comparison runs the dividing case (16 views, elevations to
    29 deg), where three VCD iterations amplify float summation-order
    differences of order 1e-7 into 9.4e-4 between 2 shards and 1 device --
    trajectory float noise around one fixed point, the same recorded pattern
    as parallel 1024, and the same size as this configuration's own 1.2e-3
    difference from mbirjax at 3 iterations (4.2e-4 by 10).  Its volume gate
    stays 5e-3, a 5.3x margin over that measurement.

The golden test's per-iteration traces (alpha, fm_rmse) measure about 6.5e-6
and 4.7e-6 and are gated further above that than the volume is: a trace is one
scalar per iteration, so a single late step size can move without the
reconstruction moving with it.
"""

import glob
import os

import numpy as np
import pytest
import torch

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "golden_*.npz")))
_have_ma = bool(_paths) and "ma_sino" in np.load(_paths[0]).files


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


def test_multiaxis_sharded_recon_matches_single_device():
    """2 CPU shards vs 1 device on the same seeded problem.

    This runs the dividing configuration, where the reconstruction
    trajectory amplifies float summation-order differences: the measured
    spread is 9.4e-4, so the 5e-3 gate below is a 5.3x margin.  That is a
    much looser number than the golden test's, and deliberately so -- see
    the module docstring for why the two configurations cannot share one
    tolerance.
    """
    ref_m = _small_ma(['cpu'])
    rs = ref_m.get_params('recon_shape')
    phantom = mbirtorch.gen_translation_phantom(rs, 'dots', None, fill_rate=0.05)
    sino = np.asarray(ref_m.forward_project(phantom))
    np.random.seed(0)
    ref, _ = ref_m.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0,
                         logfile_path=None)
    sh_m = _small_ma(['cpu', 'cpu'])
    np.random.seed(0)
    out, _ = sh_m.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0,
                        logfile_path=None)
    rel = _rel_max(out, ref)
    print(f"multiaxis sharded vs single recon rel_max = {rel:.2e}")
    assert rel < 5e-3


ma_golden = pytest.mark.skipif(
    not _have_ma, reason="no multiaxis goldens: rerun tests/generate_goldens.py")


@pytest.fixture(scope="module")
def golden():
    return np.load(_paths[0])


@pytest.fixture(scope="module")
def ma_model(golden):
    cell = tuple(int(x) for x in golden["ma_cell"])
    m = mbirtorch.MultiAxisParallelModel(cell, golden["ma_angles"])
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    return m


@pytest.mark.goldens
@ma_golden
def test_multiaxis_auto_geometry(golden, ma_model):
    assert tuple(ma_model.get_params('recon_shape')) == \
        tuple(int(x) for x in golden["ma_recon_shape"])
    assert float(ma_model.get_params('delta_voxel')) == \
        pytest.approx(float(golden["ma_delta_voxel"]), rel=1e-6)


@pytest.mark.goldens
@ma_golden
def test_multiaxis_sparse_forward(golden, ma_model):
    out = ma_model.sparse_forward_project(golden["ma_vals"], golden["ma_subset"])
    err = _rel_max(out.numpy(), golden["ma_sp_fwd"])
    print(f"multiaxis sparse_fwd rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@ma_golden
def test_multiaxis_sparse_back(golden, ma_model):
    out = ma_model.sparse_back_project(golden["ma_sino"], golden["ma_subset"])
    err = _rel_max(out.numpy(), golden["ma_sp_back"])
    print(f"multiaxis sparse_back rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@ma_golden
def test_multiaxis_full_forward(golden, ma_model):
    out = ma_model.forward_project(golden["ma_phantom"])
    err = _rel_max(out, golden["ma_sino"])
    print(f"multiaxis forward rel_max = {err:.2e}")
    assert err < 1e-4


@pytest.mark.goldens
@ma_golden
def test_multiaxis_fbp(golden, ma_model):
    out = ma_model.fbp_recon(golden["ma_sino"])
    err = _rel_max(out, golden["ma_fbp"])
    print(f"multiaxis fbp rel_max = {err:.2e}")
    assert err < 1e-3


@pytest.mark.goldens
@ma_golden
def test_multiaxis_recon_convergence_parity(golden, ma_model):
    """Seeded 3-iteration parity with mbirjax on the GOLDEN configuration.

    The volume gate is set from what this configuration measures, not from
    the sharded test's number: 24 views with elevations to +-0.4 rad agree
    with mbirjax to 1.1e-5 at 3 iterations and 6.8e-6 at 10, so 2e-4 is
    about 18x the measurement -- room for another platform's arithmetic,
    while still catching a regression an order of magnitude smaller than the
    5e-3 this test used to share with the sharded comparison.
    """
    np.random.seed(int(golden["recon_seed"]))
    recon, rd = ma_model.recon(golden["ma_sino"], max_iterations=3,
                               stop_threshold_change_pct=0.0, logfile_path=None)
    rp = rd['recon_params']
    alpha_rel = np.max(np.abs(np.array(rp['alpha_values']) - golden["ma_alpha"])
                       / np.abs(golden["ma_alpha"]))
    fm_rel = np.max(np.abs(np.array(rp['fm_rmse']) - golden["ma_fm_rmse"])
                    / np.abs(golden["ma_fm_rmse"]))
    final_rel = _rel_max(recon, golden["ma_recon"])
    print(f"multiaxis recon parity: alpha rel = {alpha_rel:.2e}, "
          f"fm rel = {fm_rel:.2e}, final rel_max = {final_rel:.2e}")
    assert alpha_rel < 1e-2
    assert fm_rel < 1e-3
    assert final_rel < 2e-4
