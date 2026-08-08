"""Golden parity gates for mbirtorch.preprocess MAR and the coupled functions.

Golden data comes from tests/generate_preprocess_goldens.py (run in the
mbirjax env), with the mbirjax direct recon saved as the SHARED input to the
correction, so the theta / corrected-sinogram gates do not depend on recon
parity.  Per the plan, the MAR end-to-end path gates on the corrected
sinogram and the fitted theta at a documented looser tolerance (projections
and OSQP are in the loop), not bitwise; the one-BH-pass recon parity is
measured and printed.

The OSQP guard tests exercise the infeasible-fit behavior directly: a
non-solved status returns None, and near-zero metal-support pixels are never
selected as residual constraints.
"""

import os

import numpy as np
import pytest
import torch

import mbirtorch
import mbirtorch.preprocess as mtp
import mbirtorch.preprocess.utilities as mtpu
import mbirtorch.preprocess.mar as mtmar

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_npz_path),
    reason="no preprocess goldens: run tests/generate_preprocess_goldens.py in the mbirjax env")


@pytest.fixture(scope="module")
def golden():
    return np.load(_npz_path)


@pytest.fixture(scope="module")
def mar_model(golden):
    cell = tuple(int(v) for v in golden["mar_cell"])
    model = mbirtorch.ConeBeamModel(cell, golden["mar_angles"],
                                    source_detector_dist=float(golden["mar_sdd"]),
                                    source_iso_dist=float(golden["mar_sid"]),
                                    device="cpu")
    model.set_params(no_warning=True, verbose=0)
    return model


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def test_gen_huber_weights(golden):
    out = mtp.gen_huber_weights(golden["hub_w"].copy(), golden["hub_e"].copy(), T=1.0, delta=0.7)
    err = _rel_max(out, golden["hub_out"])
    print(f"huber rel_max = {err:.2e}")
    assert err < 1e-5


def test_bh_correction(golden):
    out = mtp.BH_correction(golden["mar_sino"].copy(), list(golden["bhc_alpha"]), batch_size=7)
    err = _rel_max(out, golden["bhc_out"])
    print(f"BH_correction rel_max = {err:.2e}")
    assert err < 1e-5


def test_gen_weights_mar_sino_path(golden, mar_model):
    out = mbirtorch.gen_weights_mar(mar_model, golden["mar_sino"].copy())
    err = _rel_max(out, golden["gwm_sino_path"])
    print(f"gen_weights_mar (sino path) rel_max = {err:.2e}")
    assert err < 1e-5


def test_gen_weights_mar_recon_path(golden, mar_model):
    # The metal mask comes from each framework's own forward projection, so boundary entries of the
    # binary mask can flip where a grazing ray is ~0 in one projector and exactly 0 in the other.
    # Gate: mask flips are rare, and the weights agree closely where the masks agree.
    out = mbirtorch.gen_weights_mar(mar_model, golden["mar_sino"].copy(),
                                    init_recon=golden["mar_recon_input"].copy())
    ref = golden["gwm_recon_path"]
    agree = np.isclose(out, ref, rtol=1e-4, atol=1e-6)
    frac_disagree = 1.0 - agree.mean()
    print(f"gen_weights_mar (recon path) disagree fraction = {frac_disagree:.2e}")
    assert frac_disagree < 5e-3


def test_mar_theta(golden, mar_model):
    num_metal, order = 1, 3
    metal_exp = mtmar._generate_metal_exponent_list(num_metal, order)
    cross_exp = mtmar._generate_metal_exponent_list(num_metal, order - 1)
    H_exp = ([(1,) + (0,) * num_metal] + [(1, *t) for t in cross_exp] + [(0, *t) for t in metal_exp])
    p_est, m_est = mtmar._est_plastic_metal_sinos_from_recon(
        golden["mar_recon_input"].copy(), num_metal, mar_model)
    p_scale = float(torch.max(torch.abs(p_est)))
    m_scales = [float(torch.max(torch.abs(m))) for m in m_est]
    p_est_n = p_est / p_scale
    m_est_n = [m / s for m, s in zip(m_est, m_scales)]
    theta = mtmar._estimate_BH_model_params(
        p_est_n, m_est_n, mar_model.prepare_sino_for_devices(golden["mar_sino"].copy()),
        H_exp, len(cross_exp), alpha=1, beta=0.002)
    err = _rel_max(theta, golden["mar_theta"])
    print(f"MAR theta rel_max = {err:.2e}")
    assert err < 1e-3


def test_mar_corrected_sino(golden, mar_model):
    out = mtp.correct_sino_plastic_metal(mar_model, golden["mar_sino"].copy(),
                                         golden["mar_recon_input"].copy(), num_metal=1, order=3)
    err = _rel_max(out, golden["mar_corrected"])
    print(f"MAR corrected sino rel_max = {err:.2e}")
    assert err < 1e-3


def test_mar_recon_measured(golden, mar_model):
    # Recon in the loop: parity is measured and printed, gated loosely.
    np.random.seed(11)
    out = mtp.recon_plastic_metal(mar_model, golden["mar_sino"].copy(),
                                  golden["mar_weights"].copy(), num_BH_iterations=1,
                                  max_iterations=5, num_metal=1, verbose=0)
    err = _rel_max(out, golden["mar_recon_out"])
    print(f"MAR one-pass recon rel_max = {err:.2e} (measured; loose gate)")
    assert err < 0.1


def test_align_sino_views(golden, mar_model):
    # The two stages gate SEPARATELY, because the end-to-end tolerance is set by the
    # estimator, not the port.  The shift estimator is iterative and model-coupled, so
    # cross-framework projector differences reach it and a few e-3 px of spread is
    # legitimate (measured 3.3e-3 px here).  The aligned output then differs by about
    # shift difference times the local edge gradient, so a coupled output gate tighter
    # than the shift gate would fail on estimator spread the shift gate itself permits.
    shifts = mtp.estimate_sino_view_offset(mar_model, golden["mar_sino"].copy(),
                                           golden["mar_recon_input"].copy())
    max_shift_diff = float(np.max(np.abs(shifts - golden["align_shifts"])))

    # Stage 2 in isolation: the torch interpolator fed mbirjax's OWN golden shifts must
    # reproduce mbirjax's aligned output at float precision (measured 1.0e-7).  This is
    # the sharp assertion on the shifting convention.
    interp_only = mtpu._translate_views_bilinear(
        golden["mar_sino"].copy(), golden["align_shifts"].copy()).cpu().numpy()
    interp_err = _rel_max(interp_only, golden["align_out"])

    # End to end, as a sanity bound consistent with the shift gate.
    out = mtp.align_sino_views(mar_model, golden["mar_sino"].copy(),
                               golden["mar_recon_input"].copy())
    coupled_err = _rel_max(out, golden["align_out"])
    print(f"align shifts max diff = {max_shift_diff:.2e} px; interp-only rel_max = "
          f"{interp_err:.2e}; coupled rel_max = {coupled_err:.2e}")
    assert max_shift_diff < 1e-2
    assert interp_err < 1e-5
    assert coupled_err < 1e-2


def test_median_filter3d(golden):
    out = mbirtorch.median_filter3d(golden["med_in"].copy(), max_block_gb=0.0001)
    assert np.array_equal(out, golden["med_out"])
    med, mn, mx = mbirtorch.median_filter3d(golden["med_in"].copy(), max_block_gb=0.0001,
                                            return_min_max=True)
    assert np.array_equal(med, golden["med_out"])
    assert np.array_equal(mn, golden["med_min"])
    assert np.array_equal(mx, golden["med_max"])


def test_osqp_infeasible_returns_none():
    # x <= -1 and -x <= -1 cannot both hold: OSQP must report non-solved and the guard returns None.
    P = np.eye(1)
    q = np.zeros(1)
    A = np.array([[1.0], [-1.0]])
    u = np.array([-1.0, -1.0])
    assert mtmar._estimate_BH_model_params_using_OSQP(P, q, A, u) is None


def test_osqp_unconstrained_solves():
    P = np.diag([2.0, 4.0])
    q = np.array([-2.0, -8.0])
    theta = mtmar._estimate_BH_model_params_using_OSQP(P, q, None, None)
    assert np.allclose(theta, [1.0, 2.0], atol=1e-6)


def test_metal_support_floor_excludes_unactionable_pixels():
    # A very negative measured value at a pixel with NO metal support must never be selected as the
    # residual constraint (its H_m row is ~0, making the QP structurally infeasible).
    shape = (4, 5, 6)
    measured = torch.zeros(shape)
    measured[1, 2, 3] = -50.0                       # no-support pixel: huge violation, must be ignored
    measured[3, 1, 1] = -0.5                        # supported pixel: the real violator
    plastic = torch.ones(shape)
    metal = torch.zeros(shape)
    metal[3, 1, 1] = 1.0                            # support only at the real violator
    theta = np.zeros(2, dtype=np.float32)
    H_exp = [(1, 0), (0, 1)]                        # p and m columns, no cross terms
    idx_sp, v_sp, idx_res, v_res = mtmar._find_most_violated_constraints(
        measured, plastic, [metal], theta, H_exp, num_cross_terms=0)
    assert idx_res == (3, 1, 1)
    assert float(v_res) == pytest.approx(-0.5)
