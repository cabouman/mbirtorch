"""Tests for the NNAL solvers in mbirtorch.hsnt: the maximum-likelihood fit, its stop and compiled kernels, the rank
estimate, the spectra estimators and streaming.

Self-contained: small random nonnegative factorizations with mixed pixels stand in for a phantom. Each test runs on
every device of the repository's ``device`` fixture except MPS, which lacks the float64 the solvers accumulate in.
"""
import itertools

import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt import _linalg, _newton
from mbirtorch.hsnt._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives
from mbirtorch.hsnt._streaming import _stream_factorization
from mbirtorch.hsnt.factorization import _initial_factors, _nnal_factorization
from mbirtorch.hsnt.spectra import (_auto_penalty, _empty_fit_loss, _fit_free_sets, _guard_components, _select_supports,
                                    _support_selected_spectra, _unconstrained_spectra)


@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    """One intra-op thread, so parallel test workers do not each start one per core."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture
def dev(device):
    if device == "mps":
        pytest.skip("the hsnt solvers accumulate in float64, which MPS does not support")
    return device


def _problem(device, P=2048, K=200, R=3, dose=10.0, seed=0, noisy=True, dtype=torch.float32):
    """T = counts / dose for X = W_true @ H_true: background pixels, pure pixels, and pixels mixing two or three
    materials, as at the boundaries of a real sample."""
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 1.0, size=(R, K))
    H[:, K // 3:] *= 0.5                                                            # rough edge structure
    W = rng.dirichlet(np.full(R, 0.5), P) * rng.uniform(0.2, 2.0, (P, 1))
    W[rng.uniform(size=(P, R)) < 0.4] = 0.0                                         # absent materials
    W[: P // 8] = 0                                                                 # background pixels
    X = W @ H
    T = rng.poisson(dose * np.exp(-X)) / dose if noisy else np.exp(-X)
    return (torch.tensor(T, dtype=dtype, device=device), torch.tensor(W, dtype=torch.float64, device=device),
            torch.tensor(H, dtype=torch.float64, device=device))


def _sphere_problem(device, n=48, K=150, dose=100.0, seed=7):
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


def _skip_without_triton():
    from mbirtorch.kernel_availability import triton_available
    usable, reason = triton_available()
    if not usable:
        pytest.skip(f"torch.compile needs Triton on CUDA: {reason}")


def test_mle_fits_noisy_data_and_reaches_machine_precision_on_exact_data(dev):
    T, _, _ = _problem(dev)
    W, H, steps = _mle(T, max_steps=300, rel_tol=1e-6)
    W0, H0 = _initial_factors(T, 3)
    assert steps > 0 and W.min() >= 0 and H.min() >= 0 and _loss(W, H, T) < _loss(W0, H0, T)
    T, _, _ = _problem(dev, noisy=False, dtype=torch.float64)
    W, H, _ = _mle(T, max_steps=300, rel_tol=1e-6)
    assert _loss(W, H, T) < 1e-8 * T.numel()
    with pytest.raises(ValueError, match="compile_mode"):
        _nnal_factorization(T, 3, compile_mode="default")


def test_a_dead_component_is_revived(dev):
    """A component whose spectrum or map is zero gets no gradient and would stay out of the fit; the solve re-seeds
    it and reaches the loss of an ordinary start."""
    T, _, _ = _problem(dev)
    W0, H0 = _initial_factors(T, 3)
    W_ref, H_ref, _ = _mle(T)
    W0[:, 2] = 0
    H0[2] = 0                                                                       # dead in both factors
    W, H, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="off", W_init=W0, H_init=H0)
    assert H[2].norm() > 0 and abs(_loss(W, H, T) - _loss(W_ref, H_ref, T)) <= 1e-6 * _loss(W_ref, H_ref, T)
    Wd, Hd = W_ref.clone(), H_ref.clone()
    Hd[1] = 0                                                                       # only the spectrum dead
    Wn, Hn, n = _linalg._reseed_dead(Wd, Hd)
    assert n == 1 and Hn[1].min() > 0 and Wn[:, 1].min() > 0 and torch.equal(Hn[0], Hd[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_and_eager_solves_agree():
    """The compiled solve ends at the eager loss. Fused rounding in the compiled block step can leave ulp-sized
    residues where the eager step lands on zero; the snap to zero makes both steps take the same zero decisions."""
    _skip_without_triton()
    T, _, _ = _problem("cuda")
    W1, H1, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="off")
    W2, H2, _ = _nnal_factorization(T, 3, max_steps=200, rel_tol=1e-8, compile_mode="on")
    assert abs(_loss(W1, H1, T) - _loss(W2, H2, T)) <= 1e-6 * _loss(W1, H1, T)
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
    (stopping at the first quiet step spreads these by 2e-5 on the CPU and 5e-7 on CUDA on this problem)."""
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


def test_rank_estimate_finds_the_rank_with_pooling_and_near_max_rank(dev):
    """The sphere phantom has rank 3 at full resolution and pooled, given as transmission or as attenuation; with the
    true rank one below max_rank a real component is among the last three gains, and it must not raise the noise floor
    and collapse the estimate."""
    T = _sphere_problem(dev).reshape(48, 48, -1)
    n, _, detail = hsnt.estimate_rank(T, "transmission", max_rank=5, device=dev, pool=2)
    assert n == 3 and detail["rank_full"] == 3 and detail["pool_block"] == 2 and detail["pooled"]["pixels"] == 24 * 24
    assert hsnt.estimate_rank(-torch.log(T), max_rank=5, device=dev, pool=2)[0] == 3             # attenuation
    rng = np.random.default_rng(0)
    P, K, R = 2000, 300, 5
    x = np.linspace(0, 1, K)
    H = np.stack([0.1 + 0.9 * np.exp(-((x - (r + 0.5) / R) / (0.6 / R)) ** 2) for r in range(R)])
    W = rng.dirichlet(np.full(R, 0.5), P) * rng.uniform(0.3, 2.0, (P, 1))
    T = (rng.poisson(50.0 * np.exp(-W @ H)) / 50.0).astype(np.float32)
    rank, _, detail = hsnt.estimate_rank(T, "transmission", max_rank=6, device=dev)
    assert rank == R and detail["full"]["noise_tail"]


def test_spectra_estimators(dev):
    """The unconstrained estimate keeps W >= 0. Support selection holds the coefficients off its supports at zero;
    the free refit keeps the same supports and fits the data alike, and with no penalty about as well as the
    unconstrained estimate. A component selected almost nowhere reverts to the maximum-likelihood treatment."""
    T, _, _ = _problem(dev)
    W, H, _ = _mle(T)
    Wu, Hu, _ = _unconstrained_spectra(T, W, H)
    Ws, Hs, support, _ = _support_selected_spectra(T, W, H, dose=10.0)
    Wf, Hf, support_f, _ = _support_selected_spectra(T, W, H, dose=10.0, free_refit=True)
    assert Wu.min() >= 0 and Hu.shape == H.shape and Ws.min() >= 0 and Wf.min() >= 0 and Hf.min() >= 0
    assert bool((Ws[~support] == 0).all()) and torch.equal(support_f, support) and bool((Wf[~support] == 0).all())
    assert abs(_loss(Wf, Hf, T) - _loss(Ws, Hs, T)) <= 1e-3 * _loss(Ws, Hs, T)
    Wz, Hz, support_z, _ = _support_selected_spectra(T, W, H, dose=10.0, penalty=0.0, free_refit=True)
    assert support_z.sum() >= support.sum() and abs(_loss(Wz, Hz, T) - _loss(Wu, Hu, T)) <= 2e-3 * _loss(Wu, Hu, T)
    P = 5000
    guard = torch.rand(P, 3, device=dev) > 0.5
    guard[:, 1] = False
    guard[:3, 1] = True
    W_mle, W0 = torch.rand(P, 3, device=dev), torch.zeros(P, 3, device=dev)
    with pytest.warns(UserWarning):
        weak = _guard_components(guard, W_mle, W0)
    assert weak.tolist() == [False, True, False] and bool(guard[:, 1].all()) and torch.equal(W0[:, 1], W_mle[:, 1])


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
    same criterion, and runs at a rank the enumeration cannot reach (12). 'auto' is the penalty it names."""
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
    s_auto = _select_supports(T, W, H, dose=10.0, penalty="auto")[0]
    assert torch.equal(s_auto, _select_supports(T, W, H, dose=10.0, penalty=_auto_penalty(T.mean(1), 10.0))[0])


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


def test_streaming_matches_the_full_solve(dev):
    """Streamed by chunks of pixels, the MLE, the unconstrained estimate and support selection (with either refit)
    land within 1% of the loss of the solve held whole, and keep W >= 0."""
    T, _, _ = _problem(dev, P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    Wm, Hm, _ = _mle(T)
    W_chunks, H, passes = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev)
    W = torch.cat([w.to(dev) for w in W_chunks])
    assert passes >= 1 and W.min() >= 0 and _loss(W, H, T) <= 1.01 * _loss(Wm, Hm, T)
    W_chunks, H, _ = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev,
                                           nonneg_W=False)
    W = torch.cat([w.to(dev) for w in W_chunks])
    Wu, Hu, _ = _unconstrained_spectra(T, Wm, Hm)
    assert W.min() >= 0 and abs(_loss(W, H, T) - _loss(Wu, Hu, T)) <= 1e-2 * _loss(Wu, Hu, T)
    for free in (False, True):
        stats = {}
        W_chunks, H, _ = _stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024, device=dev,
                                               stats=stats, support_selection=dict(dose=10.0, free_refit=free))
        W = torch.cat([w.to(dev) for w in W_chunks])
        S = torch.cat(stats["support_chunks"]).to(dev)
        assert S.shape == W.shape and bool((W[~S] == 0).all()) and W.min() >= 0
        Ws, Hs, Sm, _ = _support_selected_spectra(T, Wm, Hm, dose=10.0, free_refit=free)
        assert abs(S.sum(1).double().mean().item() - Sm.sum(1).double().mean().item()) < 0.1
        assert _loss(W, H, T) <= 1.01 * _loss(Ws, Hs, T)
