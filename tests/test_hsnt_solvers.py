"""Smoke tests for the NNAL solvers in mbirtorch.hsnt.

Self-contained: a small random nonnegative factorization stands in for the material phantom, so no external basis
file is needed. Each test runs on every device of the repository's ``device`` fixture except MPS, which lacks the
float64 the solvers accumulate in.
"""
import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt import _newton
from mbirtorch.hsnt._loss import _nnal_prep, stable_nnal_derivatives
from mbirtorch.hsnt.factorization import _initial_factors
from mbirtorch.hsnt.spectra import _fit_free_sets, _guard_components, auto_penalty, select_supports

LBFGSB_GAP = 1e-3        # relative loss gap L-BFGS-B may leave against joint Newton on _problem()


@pytest.fixture
def dev(device):
    if device == "mps":
        pytest.skip("the hsnt solvers accumulate in float64, which MPS does not support")
    return device


def _problem(device, P=2048, K=200, R=3, dose=10.0, seed=0, noisy=True, dtype=torch.float32):
    """T = counts / dose for X = W_true @ H_true with a sparse nonnegative W (one material per pixel)."""
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 1.0, size=(R, K))
    H[:, K // 3:] *= 0.5                                                            # rough edge structure
    W = np.zeros((P, R))
    W[np.arange(P), rng.integers(0, R, P)] = rng.uniform(0.2, 2.0, P)
    W[: P // 8] = 0                                                                 # background pixels
    X = W @ H
    T = rng.poisson(dose * np.exp(-X)) / dose if noisy else np.exp(-X)
    return (torch.tensor(T, dtype=dtype, device=device), torch.tensor(W, dtype=torch.float64, device=device),
            torch.tensor(H, dtype=torch.float64, device=device))


def _loss(W, H, T):
    return hsnt.stable_nnal(W.double() @ H.double(), T.double()).item()


def _mle(T, max_steps=200, rel_tol=1e-8):
    return hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=max_steps, rel_tol=rel_tol)


@pytest.mark.parametrize("method", ["joint_newton", "block_newton", "multiplicative", "lbfgsb"])
def test_every_method_decreases_the_loss_and_stays_nonnegative(dev, method):
    T, _, _ = _problem(dev)
    W, H, steps = hsnt.nnal_factorization(T, method=method, num_materials=3, max_steps=300, rel_tol=1e-6)
    assert steps > 0 and W.min() >= 0 and H.min() >= 0
    W0, H0 = _initial_factors(T, 3)
    assert _loss(W, H, T) < _loss(W0, H0, T)


def test_lbfgsb_converges_to_the_joint_newton_loss(dev):
    T, _, _ = _problem(dev)
    Wj, Hj, _ = _mle(T, max_steps=300)
    Wl, Hl, it = hsnt.nnal_factorization(T, method="lbfgsb", num_materials=3, max_steps=3000, rel_tol=1e-9)
    assert it > 10 and Wl.min() >= 0 and Hl.min() >= 0
    assert _loss(Wl, Hl, T) <= (1 + LBFGSB_GAP) * _loss(Wj, Hj, T)


def test_joint_newton_reaches_machine_precision_on_exact_data(dev):
    T, _, _ = _problem(dev, noisy=False, dtype=torch.float64)
    W, H, _ = _mle(T, max_steps=300, rel_tol=1e-6)
    assert _loss(W, H, T) < 1e-8 * T.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("method", ["joint_newton", "block_newton", "multiplicative"])
def test_compiled_matches_eager(method):
    from mbirtorch.kernel_availability import triton_available
    usable, reason = triton_available()
    if not usable:
        pytest.skip(f"torch.compile needs Triton on CUDA: {reason}")
    T, _, _ = _problem("cuda")
    W1, H1, _ = hsnt.nnal_factorization(T, method=method, num_materials=3, max_steps=200, rel_tol=1e-8)
    W2, H2, _ = hsnt.nnal_factorization(T, method=method, num_materials=3, max_steps=200, rel_tol=1e-8,
                                        compile_mode="on")
    assert abs(_loss(W1, H1, T) - _loss(W2, H2, T)) <= 1e-6 * _loss(W1, H1, T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_and_eager_block_steps_take_the_same_zero_decisions():
    """Fused rounding in the compiled step once left ulp-sized residues where the eager step lands on zero."""
    from mbirtorch.kernel_availability import triton_available
    usable, reason = triton_available()
    if not usable:
        pytest.skip(f"torch.compile needs Triton on CUDA: {reason}")
    T, _, _ = _problem("cuda", dose=3.0)
    prep = _nnal_prep(T)
    W, H = _initial_factors(T, 3)
    eager, compiled = _newton._kernels("off")[3], _newton._kernels("on")[3]
    for _ in range(3):
        X = W @ H
        We, _, _ = eager(W, H, X, T, prep, 0)
        Wc, _, _ = compiled(W, H, X, T, prep, 0)
        assert torch.equal(We == 0, Wc == 0)
        W = We
        H, _, _ = eager(H, W, W @ H, T, prep, 1)


def test_solves_from_nearby_starts_stop_at_the_same_point(dev):
    """The joint solve stops after several quiet steps in a row: from starts 1e-7 apart it ends at the same loss
    (a one-step stop spread these by about 1e-6 on this problem)."""
    basis, _ = hsnt.load_material_basis()
    _, _, _, maps = hsnt.generate_sphere_data(np.ones((3, 1), np.float32), num_angles=4, detector_rows=32,
                                              detector_columns=32, dosage_rate=1.0,
                                              material_density={"Ni": 0.25, "Cu": 0.25, "Al": 0.75}, noisy=False,
                                              verbose=0)
    B = torch.tensor(basis[:, ::4].copy(), device=dev)
    g = torch.Generator(device=dev).manual_seed(7)
    T = torch.poisson(100.0 * torch.exp(-(torch.tensor(maps.reshape(-1, 3), device=dev) @ B)), generator=g) / 100.0
    W0, H0 = _initial_factors(T, 3)
    rng = torch.Generator(device=dev).manual_seed(0)
    losses = []
    for k in range(3):
        eps = 0.0 if k == 0 else 1e-7
        W, H, _ = hsnt.nnal_factorization(T, num_materials=3, compile_mode="off",
                                          W_init=W0 * (1 + eps * torch.randn(W0.shape, generator=rng, device=dev)),
                                          H_init=H0 * (1 + eps * torch.randn(H0.shape, generator=rng, device=dev)))
        losses.append(_loss(W, H, T))
    assert (max(losses) - min(losses)) / min(losses) < 1e-7


def test_compile_mode_is_validated():
    T, _, _ = _problem("cpu", P=64, K=20)
    with pytest.raises(ValueError, match="compile_mode"):
        hsnt.nnal_factorization(T, num_materials=3, compile_mode="default")


def test_spectra_estimators_keep_w_nonnegative_and_off_support_zero(dev):
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    Wu, Hu, _ = hsnt.unconstrained_spectra(T, W, H)
    assert Wu.min() >= 0 and Hu.shape == H.shape
    for method in ("branch_bound", "greedy"):
        Ws, Hs, support, _ = hsnt.support_selected_spectra(T, W, H, dose=10.0, method=method)
        assert Ws.min() >= 0 and support.dtype == torch.bool and support.shape == W.shape
        assert bool((Ws[~support] == 0).all())                                      # off-support coefficients stay zero


def test_streaming_matches_the_monolithic_loss_within_one_percent(dev):
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    W_chunks, H, passes = hsnt.stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024,
                                                    device=dev)
    assert passes >= 1 and all(w.min() >= 0 for w in W_chunks)
    W = torch.cat([w.to(dev) for w in W_chunks])
    Wm, Hm, _ = _mle(T)
    assert _loss(W, H, T) <= 1.01 * _loss(Wm, Hm, T)


def test_streamed_unconstrained_spectra_return_nonnegative_maps(dev):
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    W_chunks, H, _ = hsnt.stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev,
                                               nonneg_W=False)
    W = torch.cat([w.to(dev) for w in W_chunks])
    Wm, Hm, _ = _mle(T)
    Wu, Hu, _ = hsnt.unconstrained_spectra(T, Wm, Hm)
    assert W.min() >= 0 and H.shape == Hu.shape
    assert abs(_loss(W, H, T) - _loss(Wu, Hu, T)) <= 1e-2 * _loss(Wu, Hu, T)


def test_sphere_phantom_geometry():
    """Chord lengths, the 10-unit diameter, in-plane overlaps, and the noiseless data equal to W @ H."""
    rng = np.random.default_rng(0)
    basis = rng.uniform(0.05, 1.0, size=(3, 50))
    noisy, angles, gt, maps = hsnt.generate_sphere_data(basis, num_angles=4, detector_rows=64, detector_columns=64,
                                                        material_density={"Ni": 1.0, "Cu": 1.0, "Al": 1.0},
                                                        noisy=False, verbose=0)
    assert noisy.shape == (4, 64, 64, 50) and maps.shape == (4, 64, 64, 3) and len(angles) == 4
    assert abs(maps.max() - 10.0) < 0.05                                            # a diameter is 10 thickness units
    assert np.allclose(gt.reshape(-1, 50), maps.reshape(-1, 3) @ basis, atol=1e-5)  # the data are exactly rank 3
    assert np.allclose(noisy, gt, atol=1e-6)
    n = (maps > 0).sum(-1)
    # view 0: Cu and Al coincide; triple overlaps only where their disc grazes Ni's
    assert (n[0] == 2).sum() > 100 and (n == 3).sum() < 0.01 * (n > 0).sum()
    for a in range(4):
        for m in range(3):
            assert (maps[a, :, :, m] > 0).any()                          # every material visible in every view


def test_packaged_material_basis_loads():
    basis, wavelengths = hsnt.load_material_basis()
    assert basis.shape == (3, 1200) and wavelengths.shape == (1200,) and basis.min() >= 0
    assert np.all(np.diff(wavelengths) > 0)


def test_support_search_methods_agree_and_scale(dev):
    """Branch and bound reproduces the enumeration's supports on nearly every pixel of the test problem at the same
    criterion; the greedy search runs; both run at a rank the enumeration cannot (12)."""
    T, _, _ = _problem(dev, dose=10.0)
    W, H, _ = _mle(T)
    s_enum, W_enum, f_enum = select_supports(T, W, H, dose=10.0, method="enumerate")
    s_bb, W_bb, f_bb = select_supports(T, W, H, dose=10.0, method="branch_bound")
    s_gr, W_gr, f_gr = select_supports(T, W, H, dose=10.0, method="greedy")
    lam = 2 * np.log(T.shape[1])

    def criterion(s, f):
        return (10.0 * f + lam * s.sum(1)).sum().item()

    assert (s_bb == s_enum).all(1).double().mean() > 0.95
    assert criterion(s_bb, f_bb) <= criterion(s_enum, f_enum) * (1 + 2e-3)
    assert (s_gr == s_enum).all(1).double().mean() > 0.8
    assert criterion(s_gr, f_gr) <= criterion(s_enum, f_enum) * (1 + 2e-2)
    for s, W0 in ((s_bb, W_bb), (s_gr, W_gr)):
        assert W0.min() >= 0 and bool((W0[~s] == 0).all()) and s.dtype == torch.bool
    with pytest.raises(ValueError, match="R <= 8"):
        select_supports(T, torch.zeros(T.shape[0], 12, device=dev), torch.rand(12, T.shape[1], device=dev), dose=10.0,
                        method="enumerate")
    rng = np.random.default_rng(5)
    H12 = torch.tensor(rng.uniform(0.05, 1.0, (12, T.shape[1])), dtype=torch.float32, device=dev)
    W12 = torch.zeros(T.shape[0], 12, device=dev)
    for method in ("branch_bound", "greedy"):
        s12, W0, f12 = select_supports(T, W12, H12, dose=10.0, method=method)   # rank 12: no enumeration possible
        assert s12.shape == (T.shape[0], 12) and s12.sum(1).max() <= 4 and torch.isfinite(f12).all()


def test_auto_penalty_moves_from_half_to_twice_log_k_with_the_counts(dev):
    K = 200
    low = torch.full((64, K), 0.5, device=dev)
    assert auto_penalty(low, dose=2.0) == pytest.approx(0.5 * np.log(K))           # 1 count per bin
    assert auto_penalty(low, dose=1000.0) == pytest.approx(2 * np.log(K))          # 500 counts per bin
    mid = auto_penalty(low, dose=60.0)
    assert 0.5 * np.log(K) < mid < 2 * np.log(K)
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    s_auto = select_supports(T, W, H, dose=10.0, penalty="auto")[0]
    s_fixed = select_supports(T, W, H, dose=10.0, penalty=auto_penalty(T, 10.0))[0]
    assert torch.equal(s_auto, s_fixed)
    with pytest.raises(ValueError):
        select_supports(T, W, H, dose=10.0, penalty="strong")


def test_free_refit_keeps_the_supports_and_w_nonnegative(dev):
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
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


def test_free_sign_pixel_fits_are_stationary(dev):
    T, _, Ht = _problem(dev, P=512)
    H = Ht.float()
    idx = torch.arange(3, device=dev).expand(512, 3).contiguous()
    valid = torch.ones_like(idx, dtype=torch.bool)
    w, _ = _fit_free_sets(T, H, idx, valid, torch.zeros(512, 3, device=dev), steps=40, nonneg=False)
    G, _ = stable_nnal_derivatives(w @ H, T, _nnal_prep(T))
    g = G @ H.T                                                     # per-pixel gradient, all coefficients free
    assert (g.norm(dim=1) / (T.shape[1] ** 0.5)).max() < 1e-3
    assert bool((w < 0).any())                                      # some coefficients go negative: the bound is off


def test_component_guard_reverts_an_empty_component(dev):
    P, R = 5000, 3
    support = torch.rand(P, R, device=dev) > 0.5
    support[:, 1] = False
    support[:3, 1] = True
    W_mle = torch.rand(P, R, device=dev)
    W0 = torch.zeros(P, R, device=dev)
    with pytest.warns(UserWarning):
        weak = _guard_components(support, W_mle, W0)
    assert weak.tolist() == [False, True, False]
    assert bool(support[:, 1].all()) and torch.equal(W0[:, 1], W_mle[:, 1]) and bool((W0[:, 0] == 0).all())


def test_streamed_support_selection_matches_the_monolithic_estimator(dev):
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    Wm, Hm, _ = _mle(T)
    for free in (False, True):
        stats = {}
        W_chunks, H, _ = hsnt.stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024,
                                                   device=dev, stats=stats,
                                                   support_selection=dict(dose=10.0, free_refit=free))
        W = torch.cat([w.to(dev) for w in W_chunks])
        S = torch.cat(stats["support_chunks"]).to(dev)
        assert S.shape == W.shape and bool((W[~S] == 0).all()) and W.min() >= 0 and "loss_refit" in stats
        Ws, Hs, Sm, _ = hsnt.support_selected_spectra(T, Wm, Hm, dose=10.0, free_refit=free)
        assert abs(S.sum(1).double().mean().item() - Sm.sum(1).double().mean().item()) < 0.1
        assert _loss(W, H, T) <= 1.01 * _loss(Ws, Hs, T)


def test_l2_baseline_reproduces_exactly_low_rank_data():
    """The scikit-learn NMF baseline: on noiseless rank-3 attenuation its subspace has safety_factor x 3 dimensions and
    the rehydrated data match the input."""
    rng = np.random.default_rng(0)
    W = rng.uniform(0.0, 1.0, (16, 16, 3))
    H = rng.uniform(0.05, 1.0, (3, 60))
    A = (W @ H).astype(np.float32)
    sub, basis, dtype = hsnt.l2_dehydrate(A, num_materials=3, safety_factor=2, random_state=0, verbose=0)
    assert sub.shape == (16, 16, 6) and basis.shape == (6, 60) and dtype == "attenuation"
    assert sub.min() >= 0 and basis.min() >= 0
    den = hsnt.l2_hyper_denoise(A, num_materials=3, random_state=0, verbose=0)
    assert den.shape == A.shape and np.linalg.norm(den - A) / np.linalg.norm(A) < 1e-2
