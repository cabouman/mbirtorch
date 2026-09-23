"""Smoke tests for the NNAL solvers in mbirtorch.hsnt.

Self-contained: a small random nonnegative factorization stands in for the material phantom, so
no external basis file is needed. Each test runs in a few seconds on a GPU and is skipped without
one (the streaming path pins host memory for CUDA transfers).
"""
import sys

import numpy as np
import pytest
import torch

hsnt = pytest.importorskip("mbirtorch.hsnt")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
LBFGSB_GAP = 1e-3        # relative loss gap L-BFGS-B may leave against joint Newton on _problem() (measured below 1e-5 on a laptop GPU)


def _compile_works():
    """torch.compile needs Inductor (no Windows support) and a C++ toolchain; probe once on a trivial function."""
    if sys.platform == "win32":
        return False
    try:
        torch.compile(lambda x: x + 1)(torch.zeros(2, device="cuda" if torch.cuda.is_available() else "cpu"))
        return True
    except Exception:
        return False


compiled = pytest.mark.skipif(not (torch.cuda.is_available() and _compile_works()), reason="needs CUDA and a working torch.compile (Inductor)")


def _problem(P=2048, K=200, R=3, dose=10.0, seed=0, noisy=True, dtype=torch.float32):
    """T = counts / dose for X = W_true @ H_true with a sparse nonnegative W (one material per pixel)."""
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 1.0, size=(R, K)); H[:, K // 3:] *= 0.5                # rough edge structure
    W = np.zeros((P, R)); m = rng.integers(0, R, P); W[np.arange(P), m] = rng.uniform(0.2, 2.0, P)
    W[: P // 8] = 0                                                                 # background pixels
    X = W @ H
    T = rng.poisson(dose * np.exp(-X)) / dose if noisy else np.exp(-X)
    return (torch.tensor(T, dtype=dtype, device="cuda"), torch.tensor(W, dtype=torch.float64, device="cuda"),
            torch.tensor(H, dtype=torch.float64, device="cuda"))


def _loss(W, H, T):
    return hsnt.stable_nnal(W.double() @ H.double(), T.double()).item()


@cuda
@pytest.mark.parametrize("method", ["joint_newton", "block_newton", "multiplicative", "lbfgsb"])
def test_every_method_decreases_the_loss_and_stays_nonnegative(method):
    T, _, _ = _problem()
    W, H, steps = hsnt.nnal_factorization(T, method=method, num_materials=3, max_steps=300, rel_tol=1e-6)
    assert steps > 0 and W.min() >= 0 and H.min() >= 0
    # the initialization alone, for reference
    real = T > 1e-12; floor = 0.5 * T[real].min()
    W0, H0 = hsnt.nndsvda(-torch.log(torch.where(real, T, floor)), n_components=3)
    assert _loss(W, H, T) < _loss(W0, H0, T)


@cuda
def test_lbfgsb_converges_to_the_joint_newton_loss():
    T, _, _ = _problem()
    Wj, Hj, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=300, rel_tol=1e-8)
    Wl, Hl, it = hsnt.nnal_factorization(T, method="lbfgsb", num_materials=3, max_steps=3000, rel_tol=1e-9)
    assert it > 10 and Wl.min() >= 0 and Hl.min() >= 0
    assert _loss(Wl, Hl, T) <= (1 + LBFGSB_GAP) * _loss(Wj, Hj, T)


@cuda
def test_joint_newton_reaches_machine_precision_on_exact_data():
    T, _, _ = _problem(noisy=False, dtype=torch.float64)
    W, H, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=300, rel_tol=1e-6)
    assert _loss(W, H, T) < 1e-8 * T.numel()


@cuda
def test_default_method_is_joint_newton():
    assert hsnt.nnal_factorization.__defaults__[0] == "joint_newton"


@compiled
def test_compiled_matches_eager():
    T, _, _ = _problem()
    W1, H1, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    W2, H2, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8,
                                        compile_mode="default")
    assert abs(_loss(W1, H1, T) - _loss(W2, H2, T)) <= 1e-6 * _loss(W1, H1, T)


@cuda
def test_spectra_estimators_run_and_keep_w_nonnegative():
    T, _, _ = _problem()
    W, H, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    Wu, Hu, _ = hsnt.unconstrained_spectra(T, W, H)
    assert Wu.min() >= 0 and Hu.shape == H.shape
    Ws, Hs, support, _ = hsnt.support_selected_spectra(T, W, H, dose=10.0)
    assert Ws.min() >= 0 and support.dtype == torch.bool and support.shape == W.shape
    assert bool((Ws[~support] == 0).all())                                          # off-support coefficients stay zero


@cuda
def test_streaming_matches_monolithic_within_a_decibel_of_loss():
    T, _, _ = _problem(P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    W_chunks, H, passes = hsnt.stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024)
    assert passes >= 1 and all(w.min() >= 0 for w in W_chunks)
    W = torch.cat([w.cuda() for w in W_chunks])
    Wm, Hm, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    assert _loss(W, H, T) <= 1.01 * _loss(Wm, Hm, T)


def test_sphere_phantom_geometry():
    """Chord lengths, the 10-unit diameter, in-plane overlaps, and the noiseless data equal to W @ H."""
    rng = np.random.default_rng(0); basis = rng.uniform(0.05, 1.0, size=(3, 50))
    noisy, angles, gt, maps = hsnt.generate_sphere_data(basis, num_angles=4, detector_rows=64, detector_columns=64,
                                                        material_density={"Ni": 1.0, "Cu": 1.0, "Al": 1.0}, noisy=False, verbose=0)
    assert noisy.shape == (4, 64, 64, 50) and maps.shape == (4, 64, 64, 3) and len(angles) == 4
    assert abs(maps.max() - 10.0) < 0.05                                            # a diameter is 10 thickness units
    assert np.allclose(gt.reshape(-1, 50), maps.reshape(-1, 3) @ basis, atol=1e-5)  # the data are exactly rank 3
    assert np.allclose(noisy, gt, atol=1e-6)
    n = (maps > 0).sum(-1)
    assert (n[0] == 2).sum() > 100 and (n == 3).sum() < 0.01 * (n > 0).sum()         # view 0: Cu and Al coincide; triple overlaps only where their disc grazes Ni's
    for a in range(4):
        for m in range(3):
            assert (maps[a, :, :, m] > 0).any()                                      # every material visible in every view


@cuda
def test_support_search_methods_agree_and_scale():
    """Branch and bound reproduces the enumeration's supports on nearly every pixel of the test problem at the same
    criterion; the greedy search runs; both run at a rank the enumeration cannot (12)."""
    from mbirtorch.hsnt import select_supports
    T, W_true, Ht = _problem(dose=10.0)
    W, H, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    s_enum, W_enum, f_enum = select_supports(T, W, H, dose=10.0, method="enumerate")
    s_bb, W_bb, f_bb = select_supports(T, W, H, dose=10.0, method="branch_bound")
    s_gr, W_gr, f_gr = select_supports(T, W, H, dose=10.0, method="greedy")
    lam = 2 * np.log(T.shape[1])
    crit = lambda s, f: (10.0 * f + lam * s.sum(1)).sum().item()
    assert (s_bb == s_enum).all(1).double().mean() > 0.95 and crit(s_bb, f_bb) <= crit(s_enum, f_enum) * (1 + 2e-3)
    assert (s_gr == s_enum).all(1).double().mean() > 0.8 and crit(s_gr, f_gr) <= crit(s_enum, f_enum) * (1 + 2e-2)
    for s, W0 in ((s_bb, W_bb), (s_gr, W_gr)):
        assert W0.min() >= 0 and bool((W0[~s] == 0).all()) and s.dtype == torch.bool
    Ws, Hs, support, _ = hsnt.support_selected_spectra(T, W, H, dose=10.0, method="branch_bound")   # the full estimator
    assert Ws.min() >= 0 and bool((Ws[~support] == 0).all())
    with pytest.raises(ValueError, match="R <= 8"):
        select_supports(T, torch.zeros(T.shape[0], 12, device="cuda"), torch.rand(12, T.shape[1], device="cuda"), dose=10.0, method="enumerate")
    rng = np.random.default_rng(5); H12 = torch.tensor(rng.uniform(0.05, 1.0, (12, T.shape[1])), dtype=torch.float32, device="cuda")
    W12 = torch.zeros(T.shape[0], 12, device="cuda")
    for method in ("branch_bound", "greedy"):
        s12, W0, f12 = select_supports(T, W12, H12, dose=10.0, method=method)                 # rank 12: no enumeration possible
        assert s12.shape == (T.shape[0], 12) and s12.sum(1).max() <= 4 and torch.isfinite(f12).all()


@cuda
def test_free_refit_keeps_the_supports_and_w_nonnegative():
    T, _, _ = _problem()
    W, H, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    Ws, Hs, support, _ = hsnt.support_selected_spectra(T, W, H, dose=10.0)
    Wf, Hf, support_f, _ = hsnt.support_selected_spectra(T, W, H, dose=10.0, free_refit=True)
    assert bool((support_f == support).all())                                       # same selection, different refit
    assert Wf.min() >= 0 and bool((Wf[~support_f] == 0).all()) and Hf.min() >= 0
    assert abs(_loss(Wf, Hf, T) - _loss(Ws, Hs, T)) <= 1e-3 * _loss(Ws, Hs, T)     # the two refits fit the data alike
    # with no penalty the selection keeps every material the constrained pixel fit uses, and the free refit then fits
    # the data about as well as the unconstrained estimator (whose W is free everywhere)
    Wz, Hz, support_z, _ = hsnt.support_selected_spectra(T, W, H, dose=10.0, penalty=0.0, free_refit=True)
    Wu, Hu, _ = hsnt.unconstrained_spectra(T, W, H)
    assert support_z.sum() >= support.sum() and abs(_loss(Wz, Hz, T) - _loss(Wu, Hu, T)) <= 2e-3 * _loss(Wu, Hu, T)


@cuda
def test_free_sign_pixel_fits_are_stationary():
    from mbirtorch.hsnt.spectra import _fit_free_sets
    from mbirtorch.hsnt._loss import _nnal_prep, stable_nnal_derivatives
    T, Wt, Ht = _problem(P=512)
    H = Ht.float(); idx = torch.arange(3, device="cuda").expand(512, 3).contiguous(); valid = torch.ones_like(idx, dtype=torch.bool)
    w, _ = _fit_free_sets(T, H, idx, valid, torch.zeros(512, 3, device="cuda"), steps=40, nonneg=False)
    G, _ = stable_nnal_derivatives(w @ H, T, _nnal_prep(T))
    g = G @ H.T                                                                     # per-pixel gradient, all coefficients free
    assert (g.norm(dim=1) / (T.shape[1] ** 0.5)).max() < 1e-3
    assert bool((w < 0).any())                                                      # some coefficients go negative: the bound is off


@cuda
def test_component_guard_reverts_an_empty_component():
    from mbirtorch.hsnt.spectra import _guard_components
    P, R = 5000, 3
    support = torch.rand(P, R, device="cuda") > 0.5; support[:, 1] = False; support[:3, 1] = True
    W_mle = torch.rand(P, R, device="cuda"); W0 = torch.zeros(P, R, device="cuda")
    weak = _guard_components(support, W_mle, W0)
    assert weak.tolist() == [False, True, False]
    assert bool(support[:, 1].all()) and torch.equal(W0[:, 1], W_mle[:, 1]) and bool((W0[:, 0] == 0).all())


@cuda
def test_streamed_support_selection_matches_the_monolithic_estimator():
    T, _, _ = _problem(P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    for free in (False, True):
        stats = {}
        W_chunks, H, passes = hsnt.stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, stats=stats,
                                                        support_selection=dict(dose=10.0, free_refit=free))
        W = torch.cat([w.cuda() for w in W_chunks]); S = torch.cat(stats["support_chunks"]).cuda()
        assert S.shape == W.shape and bool((W[~S] == 0).all()) and W.min() >= 0 and "loss_refit" in stats
        Wm, Hm, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
        Ws, Hs, Sm, _ = hsnt.support_selected_spectra(T, Wm, Hm, dose=10.0, free_refit=free)
        assert abs(S.sum(1).double().mean().item() - Sm.sum(1).double().mean().item()) < 0.1
        assert _loss(W, H, T) <= 1.01 * _loss(Ws, Hs, T)
