"""Smoke tests for the NNAL solvers in mbirtorch.hsnt.

Self-contained: a small random nonnegative factorization stands in for the material phantom, so no external basis
file is needed. Each test runs on every device of the repository's ``device`` fixture except MPS, which lacks the
float64 the solvers accumulate in.
"""
import itertools

import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt import _newton
from mbirtorch.hsnt._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives
from mbirtorch.hsnt._streaming import _stream_factorization
from mbirtorch.hsnt.factorization import _initial_factors, _nnal_factorization
from mbirtorch.hsnt.spectra import (_auto_penalty, _empty_fit_loss, _fit_free_sets, _guard_components, _select_supports,
                                    _support_selected_spectra, _unconstrained_spectra)


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


def _sphere_problem(device, n=64, K=300, dose=100.0, seed=7):
    """Three overlapping spheres seen along one axis, with the packaged spectra: most pixels mix two materials."""
    basis, _ = hsnt.load_material_basis()
    yy, xx = np.mgrid[:n, :n] + 0.5
    maps = []
    for (cy, cx), density in zip(((0.38, 0.38), (0.38, 0.62), (0.6, 0.5)), (0.25, 0.25, 0.75)):
        d2 = ((yy - cy * n) ** 2 + (xx - cx * n) ** 2) / (0.22 * n) ** 2
        maps.append(10.0 * density * np.sqrt(np.clip(1.0 - d2, 0.0, None)))       # a chord of a diameter-10 sphere
    X = torch.tensor(np.stack(maps, -1).reshape(-1, 3), dtype=torch.float32, device=device) @ torch.tensor(
        basis[:, ::basis.shape[1] // K].copy(), device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.poisson(dose * torch.exp(-X), generator=g) / dose


def _loss(W, H, T):
    return stable_nnal(W.double() @ H.double(), T.double()).item()


def _mle(T, max_steps=200, rel_tol=1e-8):
    return _nnal_factorization(T, 3, max_steps=max_steps, rel_tol=rel_tol, compile_mode="off")


def test_mle_decreases_the_loss_and_stays_nonnegative(dev):
    T, _, _ = _problem(dev)
    W, H, steps = _mle(T, max_steps=300, rel_tol=1e-6)
    assert steps > 0 and W.min() >= 0 and H.min() >= 0
    W0, H0 = _initial_factors(T, 3)
    assert _loss(W, H, T) < _loss(W0, H0, T)


def test_joint_newton_reaches_machine_precision_on_exact_data(dev):
    T, _, _ = _problem(dev, noisy=False, dtype=torch.float64)
    W, H, _ = _mle(T, max_steps=300, rel_tol=1e-6)
    assert _loss(W, H, T) < 1e-8 * T.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_matches_eager():
    from mbirtorch.kernel_availability import triton_available
    usable, reason = triton_available()
    if not usable:
        pytest.skip(f"torch.compile needs Triton on CUDA: {reason}")
    T, _, _ = _problem("cuda")
    W1, H1, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="off")
    W2, H2, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="on")
    assert abs(_loss(W1, H1, T) - _loss(W2, H2, T)) <= 1e-6 * _loss(W1, H1, T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_and_eager_block_steps_take_the_same_zero_decisions():
    """Fused rounding in the compiled step can leave ulp-sized residues where the eager step lands on zero; the snap
    to zero makes both steps take the same zero decisions."""
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
    (stopping at the first quiet step spreads these by 2e-8 on the CPU and 4e-7 on CUDA on this problem)."""
    T = _sphere_problem(dev)
    W0, H0 = _initial_factors(T, 3)
    rng = torch.Generator(device=dev).manual_seed(0)
    losses = []
    for k in range(3):
        eps = 0.0 if k == 0 else 1e-7
        W, H, _ = _nnal_factorization(T, 3, compile_mode="off",
                                      W_init=W0 * (1 + eps * torch.randn(W0.shape, generator=rng, device=dev)),
                                      H_init=H0 * (1 + eps * torch.randn(H0.shape, generator=rng, device=dev)))
        losses.append(_loss(W, H, T))
    assert (max(losses) - min(losses)) / min(losses) < 1e-8


def test_compile_mode_is_validated():
    T, _, _ = _problem("cpu", P=64, K=20)
    with pytest.raises(ValueError, match="compile_mode"):
        _nnal_factorization(T, 3, compile_mode="default")


def test_spectra_estimators_keep_w_nonnegative_and_off_support_zero(dev):
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    Wu, Hu, _ = _unconstrained_spectra(T, W, H)
    assert Wu.min() >= 0 and Hu.shape == H.shape
    Ws, Hs, support, _ = _support_selected_spectra(T, W, H, dose=10.0)
    assert Ws.min() >= 0 and support.dtype == torch.bool and support.shape == W.shape
    assert bool((Ws[~support] == 0).all())                                          # off-support coefficients stay zero


def test_streaming_matches_the_monolithic_loss_within_one_percent(dev):
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    W_chunks, H, passes = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev)
    assert passes >= 1 and all(w.min() >= 0 for w in W_chunks)
    W = torch.cat([w.to(dev) for w in W_chunks])
    Wm, Hm, _ = _mle(T)
    assert _loss(W, H, T) <= 1.01 * _loss(Wm, Hm, T)


def test_streamed_unconstrained_spectra_return_nonnegative_maps(dev):
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    W_chunks, H, _ = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev,
                                           nonneg_W=False)
    W = torch.cat([w.to(dev) for w in W_chunks])
    Wm, Hm, _ = _mle(T)
    Wu, Hu, _ = _unconstrained_spectra(T, Wm, Hm)
    assert W.min() >= 0 and H.shape == Hu.shape
    assert abs(_loss(W, H, T) - _loss(Wu, Hu, T)) <= 1e-2 * _loss(Wu, Hu, T)


def test_rank_one_below_max_rank_is_found(dev):
    """With the true rank one below max_rank a real component is among the last three gains; it must not raise the
    noise floor and collapse the estimate."""
    rng = np.random.default_rng(0)
    P, K, R = 2000, 300, 5
    x = np.linspace(0, 1, K)
    H = np.stack([0.1 + 0.9 * np.exp(-((x - (r + 0.5) / R) / (0.6 / R)) ** 2) for r in range(R)])
    W = rng.dirichlet(np.full(R, 0.5), P) * rng.uniform(0.3, 2.0, (P, 1))
    T = (rng.poisson(50.0 * np.exp(-W @ H)) / 50.0).astype(np.float32)
    rank, _, detail = hsnt.estimate_rank(T, device=dev, max_rank=6)
    assert rank == R and detail["full"]["noise_tail"]


def test_packaged_material_basis_loads():
    basis, wavelengths = hsnt.load_material_basis()
    assert basis.shape == (3, 1200) and wavelengths.shape == (1200,) and basis.min() >= 0
    assert np.all(np.diff(wavelengths) > 0)


def _enumerated_supports(T, W, H, dose, lam):
    """The reference search: every subset of the R materials, fitted by a constrained W solve."""
    R = H.shape[0]
    prep = _nnal_prep(T)
    rowwise = _newton._kernels("off")[2]
    subsets = [list(c) for r in range(1, R + 1) for c in itertools.combinations(range(R), r)]
    f0 = _empty_fit_loss(T, prep)
    crit, fits, W_sub = [f0 * dose], [f0], []
    for S in subsets:
        Ws = _newton.solve_W(T, H[S].contiguous(), W[:, S].contiguous(), 100, 1e-12)
        f = rowwise(Ws @ H[S], T, prep, 1, dtype=torch.float64)
        crit.append(f * dose + lam * len(S))
        fits.append(f)
        W_sub.append(Ws)
    best = torch.stack(crit, 1).argmin(1)
    W0 = torch.zeros_like(W)
    for j, S in enumerate(subsets):
        m = (best == j + 1).nonzero().squeeze(1)
        W0[m[:, None], torch.tensor(S, device=T.device)[None, :]] = W_sub[j][m]
    return W0 > 0, torch.stack(fits, 1).gather(1, best[:, None]).squeeze(1)


def test_branch_and_bound_matches_the_enumeration_and_scales(dev):
    """Branch and bound reproduces the exhaustive search's supports on nearly every pixel of the test problem at the
    same criterion, and runs at a rank the enumeration cannot reach (12)."""
    T, _, _ = _problem(dev, dose=10.0)
    W, H, _ = _mle(T)
    lam = 2 * np.log(T.shape[1])
    s_enum, f_enum = _enumerated_supports(T, W, H, 10.0, lam)
    s_bb, W_bb, f_bb = _select_supports(T, W, H, dose=10.0, penalty=2.0)

    def criterion(s, f):
        return (10.0 * f + lam * s.sum(1)).sum().item()

    assert (s_bb == s_enum).all(1).double().mean() > 0.95
    assert criterion(s_bb, f_bb) <= criterion(s_enum, f_enum) * (1 + 2e-3)
    assert W_bb.min() >= 0 and bool((W_bb[~s_bb] == 0).all()) and s_bb.dtype == torch.bool
    rng = np.random.default_rng(5)
    H12 = torch.tensor(rng.uniform(0.05, 1.0, (12, T.shape[1])), dtype=torch.float32, device=dev)
    s12, _, f12 = _select_supports(T, torch.zeros(T.shape[0], 12, device=dev), H12, dose=10.0, penalty=2.0)
    assert s12.shape == (T.shape[0], 12) and s12.sum(1).max() <= 4 and torch.isfinite(f12).all()


def test_auto_penalty_moves_from_half_to_twice_log_k_with_the_counts(dev):
    means = torch.full((64,), 0.5, device=dev)
    assert _auto_penalty(means, dose=2.0) == pytest.approx(0.5)                     # 1 count per bin
    assert _auto_penalty(means, dose=1000.0) == pytest.approx(2.0)                  # 500 counts per bin
    assert 0.5 < _auto_penalty(means, dose=60.0) < 2.0
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    s_auto = _select_supports(T, W, H, dose=10.0, penalty="auto")[0]
    s_fixed = _select_supports(T, W, H, dose=10.0, penalty=_auto_penalty(T.mean(1), 10.0))[0]
    assert torch.equal(s_auto, s_fixed)
    for bad in ("strong", -1.0):
        with pytest.raises(ValueError):
            _select_supports(T, W, H, dose=10.0, penalty=bad)


def test_free_refit_keeps_the_supports_and_w_nonnegative(dev):
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    Ws, Hs, support, _ = _support_selected_spectra(T, W, H, dose=10.0)
    Wf, Hf, support_f, _ = _support_selected_spectra(T, W, H, dose=10.0, free_refit=True)
    assert bool((support_f == support).all())                                       # same selection, different refit
    assert Wf.min() >= 0 and bool((Wf[~support_f] == 0).all()) and Hf.min() >= 0
    assert abs(_loss(Wf, Hf, T) - _loss(Ws, Hs, T)) <= 1e-3 * _loss(Ws, Hs, T)     # the two refits fit the data alike
    # with no penalty the selection keeps every material the constrained pixel fit uses, and the free refit then fits
    # the data about as well as the unconstrained estimator (whose W is free everywhere)
    Wz, Hz, support_z, _ = _support_selected_spectra(T, W, H, dose=10.0, penalty=0.0, free_refit=True)
    Wu, Hu, _ = _unconstrained_spectra(T, W, H)
    assert support_z.sum() >= support.sum() and abs(_loss(Wz, Hz, T) - _loss(Wu, Hu, T)) <= 2e-3 * _loss(Wu, Hu, T)


def test_pixel_fits_meet_the_kkt_conditions(dev):
    """From a uniform start, where the coupled Newton step pushes entries at zero outward, every pixel's w >= 0 fit
    reaches its KKT point; the free-sign fit reaches a zero gradient with some coefficients negative."""
    T, _, Ht = _problem(dev, P=512)
    H = Ht.float()
    idx = torch.arange(3, device=dev).expand(512, 3).contiguous()
    valid = torch.ones_like(idx, dtype=torch.bool)
    w, _ = _fit_free_sets(T, H, idx, valid, torch.full((512, 3), 0.5, device=dev), steps=8)
    g = stable_nnal_derivatives(w @ H, T, _nnal_prep(T))[0] @ H.T / T.shape[1]
    assert w.min() >= 0 and torch.where(w > 0, g.abs(), (-g).clamp(min=0)).max() < 1e-6
    w, _ = _fit_free_sets(T, H, idx, valid, torch.zeros(512, 3, device=dev), steps=40, nonneg=False)
    g = stable_nnal_derivatives(w @ H, T, _nnal_prep(T))[0] @ H.T / T.shape[1]
    assert g.abs().max() < 1e-3 and bool((w < 0).any())            # the bound is off


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
        W_chunks, H, _ = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev,
                                               stats=stats, support_selection=dict(dose=10.0, free_refit=free))
        W = torch.cat([w.to(dev) for w in W_chunks])
        S = torch.cat(stats["support_chunks"]).to(dev)
        assert S.shape == W.shape and bool((W[~S] == 0).all()) and W.min() >= 0 and "loss_refit" in stats
        Ws, Hs, Sm, _ = _support_selected_spectra(T, Wm, Hm, dose=10.0, free_refit=free)
        assert abs(S.sum(1).double().mean().item() - Sm.sum(1).double().mean().item()) < 0.1
        assert _loss(W, H, T) <= 1.01 * _loss(Ws, Hs, T)
