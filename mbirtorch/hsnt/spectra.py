import math

import torch

import itertools
import logging

from ._loss import _nnal_prep, _nnal_rowwise, stable_nnal_derivatives
from ._newton import _joint_newton_pcg, _kernels, solve_W

log = logging.getLogger("mbirtorch.hsnt")


def unconstrained_spectra(T, W, H, max_steps=100, cg_max=10, rel_tol=1e-8, w_max_steps=100, compile_mode=None):
    """Re-estimate the spectra with the bound on the pixel coefficients dropped, then re-solve W >= 0.

    The maximum-likelihood spectra are biased by the truncation of pixel
    coefficients at zero: a coefficient whose true value is zero is estimated
    positive half the time and clipped the other half, an O(1/sqrt m) effect per
    pixel (m = information per pixel) that does not average out over pixels.
    Dropping the bound removes it -- the pixel problem stays strictly convex for
    any real w; estimating H with W free is a semi-NMF step (Ding, Li & Jordan
    2010) and the relax-where-it-truncates logic of NEG-ML in PET -- at the price
    of the variance reduction the constraint provides (the implicit regularisation
    of a sign constraint, Slawski & Hein 2013),
    so it wins once pixels are plentiful (above ~10^5 at dose 3; the crossover
    moves to larger P at lower dose) and loses a little below that. Cost: a
    continuation of the joint solve from the ML point, cheaper than the ML solve.
    Not for coefficients that are physically nonnegative and mostly zero, such as
    fractions over a dictionary of many similar atoms: there the unconstrained fit
    is ill-conditioned and the bound carries real information; see
    support_selected_spectra.

    With W free the loss is the same for (W A^-1, A H) and (W, H) whenever A H >= 0,
    so the solve determines the row space of H but not which mixture of its rows
    to call a material, and once the loss has converged the iteration only wanders
    along that gauge: at low dose it can reach mixtures whose W >= 0 re-solve fits
    worse than the MLE and whose maps are poor (the re-solved loss against the MLE's
    is the check). The stop at rel_tol = 1e-8 within 100 steps ends the solve before
    most of that wandering; nonneg-preserving re-mixing cannot undo it afterwards.

    Returns (W, H, steps) with W >= 0 re-solved for the returned H.
    """
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    _, Hu, steps, _ = _joint_newton_pcg(T, W, H, max_steps=max_steps, cg_max=cg_max, rel_tol=rel_tol,
                                        prep=prep, nnal=nnal_fn, deriv=deriv, nonneg_W=False)
    Wc = solve_W(T, Hu, W, w_max_steps, 1e-12, compile_mode=compile_mode)
    return Wc, Hu, steps


def _fit_free_sets(T, H, idx, valid, w0, steps=8, nonneg=True):
    """Newton fit of every pixel on its own small set of materials, batched over pixels; w >= 0 unless nonneg=False.

    idx (P, m) holds material indices (anything where valid is False is padding), w0 (P, m) the start. The per-pixel
    Hessian is the m x m block of the free set, so one Newton step costs a (P, m, K) gather-product whatever R is;
    entries at zero with an outward gradient are frozen (two-metric projection) and the step is the largest feasible
    one, checked by a per-pixel Armijo test on the float64 row loss. Returns (w, f) with f the per-pixel loss."""
    P, m = idx.shape
    dev = T.device
    prep = _nnal_prep(T)
    Hf = H[idx.clamp(min=0)] * valid[:, :, None]                                       # (P, m, K)
    w = (w0 * valid).clone()
    X = torch.einsum('pm,pmk->pk', w, Hf); f = _nnal_rowwise(X, T, prep, 1, dtype=torch.float64)
    eye = torch.eye(m, device=dev, dtype=T.dtype)
    for _ in range(steps):
        G, Z = stable_nnal_derivatives(X, T, prep)
        g = torch.einsum('pk,pmk->pm', G, Hf)
        Hs = torch.einsum('pk,pmk,pnk->pmn', Z, Hf, Hf) + 1e-6 * eye
        free = valid & ~((w <= 0) & (g > 0)) if nonneg else valid
        Hs = torch.where(free[:, :, None] & free[:, None, :], Hs, eye.expand_as(Hs))
        rhs = torch.where(free, g, torch.zeros_like(g))
        d = (torch.linalg.pinv(Hs) @ rhs.unsqueeze(-1)).squeeze(-1) * free            # pinv: identical or zero spectra make Hs singular
        slope = (g * d).sum(1)
        d = torch.where((slope <= 0)[:, None], rhs / Hs.diagonal(dim1=1, dim2=2).clamp(min=1e-12), d)
        if nonneg:
            ratio = torch.where(d > 0, w / d.clamp(min=1e-30), torch.full_like(d, float('inf')))
            alpha = ratio.amin(1).clamp(max=1.0)
        else:
            alpha = torch.ones_like(slope)
        accepted = torch.zeros_like(alpha); done = torch.zeros_like(alpha, dtype=torch.bool)
        slope = (g * d).sum(1).clamp(min=0)
        for _ in range(8):
            trial = torch.where(done, torch.zeros_like(alpha), alpha)
            wt = (w - trial[:, None] * d); wt = (wt.clamp(min=0) if nonneg else wt) * valid
            ft = _nnal_rowwise(torch.einsum('pm,pmk->pk', wt, Hf), T, prep, 1, dtype=torch.float64)
            ok = (ft <= f - 1e-4 * trial * slope + 4 * torch.finfo(T.dtype).eps * f.abs()) | (trial == 0)
            accepted = torch.where(ok & ~done, trial, accepted); done |= ok
            if bool(done.all()):
                break
            alpha = alpha * 0.5
        w_new = (w - accepted[:, None] * d); w_new = (w_new.clamp(min=0) if nonneg else w_new) * valid
        X = torch.einsum('pm,pmk->pk', w_new, Hf); f_new = _nnal_rowwise(X, T, prep, 1, dtype=torch.float64)
        moved = (f - f_new).abs() > 1e-9 * f.abs()
        w, f = w_new, f_new
        if not bool(moved.any()):
            break
    return w, f


def _scatter_support(idx, valid, w, R):
    """(P, R) support mask and coefficients from padded per-pixel sets; only valid slots are written (padding
    indices must not collide with a real material)."""
    P = idx.shape[0]
    pp, slot = valid.nonzero(as_tuple=True)
    support = torch.zeros(P, R, dtype=torch.bool, device=idx.device); support[pp, idx[pp, slot]] = True
    W = torch.zeros(P, R, dtype=w.dtype, device=idx.device); W[pp, idx[pp, slot]] = w[pp, slot]
    return support, W


def _support_sets(support, W):
    """Padded per-pixel index sets (idx, valid, w0) of a (P, R) bool support, for _fit_free_sets."""
    valid, idx = torch.sort(support, dim=1, descending=True, stable=True)
    return idx, valid, W.gather(1, idx)


def _solve_W_on_support(T, H, W, support, steps=30, nonneg=True):
    """W given H, restricted to each pixel's support: exact per-pixel Newton, W >= 0 unless nonneg=False."""
    idx, valid, w0 = _support_sets(support, W.clamp(min=0) if nonneg else W)
    w, _ = _fit_free_sets(T, H, idx, valid, w0, steps=steps, nonneg=nonneg)
    return _scatter_support(idx, valid, w, W.shape[1])[1]


def _guard_components(support, W_mle, W0, min_support=None):
    """Components selected in fewer than min_support pixels (default max(2R, P / 10^4)) revert to the MLE's treatment:
    every pixel free for them, W >= 0 alone deciding their zeros. A component the data cannot place is otherwise
    refit from a handful of pixels, and its spectrum and the gauge of the others with it. Returns the weak mask."""
    P, R = support.shape
    if min_support is None:
        min_support = max(2 * R, P // 10000)
    weak = support.sum(0) < min_support
    if bool(weak.any()):
        log.warning("support selection: component(s) %s selected in fewer than %d pixels; kept free in every pixel",
                    weak.nonzero().flatten().tolist(), min_support)
        support[:, weak] = True; W0[:, weak] = W_mle[:, weak]
    return weak


def _select_branch_bound(T, H, dose, lam, f_full, k_top=6, m_max=4, plausible=None):
    """Per-pixel subset search: exact single-material fits for every material (R batched one-dimensional solves,
    restricted to `plausible` pixels when given), then pairs and triples among each pixel's k_top best singletons for
    the pixels the lower bound leaves open. The pixel loss is monotone in the subset, so the full-model loss f_full
    bounds every subset from below: a set of `size` materials can beat the current best only if
    dose * (f_best - f_full) > lam * (size - |best|). Returns (idx, valid, w, f) as padded per-pixel sets."""
    P, K = T.shape; R = H.shape[0]; dev = T.device
    k_top = min(k_top, R); m_max = min(m_max, R)
    prep = _nnal_prep(T)
    f0 = _nnal_rowwise(torch.zeros_like(T), T, prep, 1, dtype=torch.float64)
    F1 = torch.full((P, R), float('inf'), device=dev, dtype=torch.float64); W1 = torch.zeros(P, R, device=dev, dtype=T.dtype)
    for r in range(R):
        rows = torch.arange(P, device=dev) if plausible is None else plausible[:, r].nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        idx1 = torch.full((rows.numel(), 1), r, dtype=torch.long, device=dev); valid1 = torch.ones_like(idx1, dtype=torch.bool)
        w1, f1 = _fit_free_sets(T[rows], H, idx1, valid1, torch.full((rows.numel(), 1), 0.5, device=dev, dtype=T.dtype))
        F1[rows, r] = f1; W1[rows, r] = w1[:, 0]
    best_f1, r1 = F1.min(1)
    one = best_f1 * dose + lam < f0 * dose                                                  # best singleton beats the empty set
    best_idx = torch.full((P, m_max), -1, dtype=torch.long, device=dev); best_valid = torch.zeros(P, m_max, dtype=torch.bool, device=dev)
    best_w = torch.zeros(P, m_max, device=dev, dtype=T.dtype)
    best_idx[one, 0] = r1[one]; best_valid[one, 0] = True; best_w[one, 0] = W1[one, r1[one]]
    best_f = torch.where(one, best_f1, f0)
    order = F1.argsort(1)[:, :k_top]                                                       # candidate materials per pixel
    n_cand = torch.isfinite(F1).sum(1)
    for size in range(2, m_max + 1):
        can = ((best_f - f_full) * dose > lam * (size - best_valid.sum(1).double())) & (n_cand >= size)
        cp = can.nonzero().squeeze(1)
        if cp.numel() == 0:
            break
        for combo in itertools.combinations(range(k_top), size):
            idx_c = order[cp][:, list(combo)]
            ok = torch.isfinite(F1[cp[:, None], idx_c]).all(1)
            if not bool(ok.any()):
                continue
            cq = cp[ok]; idx_c = idx_c[ok]
            valid_c = torch.ones(cq.numel(), size, dtype=torch.bool, device=dev)
            w_c, f_c = _fit_free_sets(T[cq], H, idx_c, valid_c, (W1[cq[:, None], idx_c] / size).clamp(min=1e-3))
            better = f_c * dose + lam * size < best_f[cq] * dose + lam * best_valid[cq].sum(1).double()
            bp = cq[better]
            best_idx[bp] = -1; best_valid[bp] = False; best_w[bp] = 0
            best_idx[bp, :size] = idx_c[better]; best_valid[bp, :size] = True; best_w[bp, :size] = w_c[better]; best_f[bp] = f_c[better]
    return best_idx, best_valid, best_w, best_f


def _select_greedy(T, H, dose, lam, m_max=4, screen=0.2, n_cand=2):
    """Forward selection: each active pixel takes the material with the largest estimated gain (the one-material
    Newton estimate g^2 / 2M from one gradient pass), the n_cand best estimates are fitted exactly on the enlarged
    set, the exact gain must exceed lam, and one backward pass drops members whose removal costs less than lam.
    Its cost is a few gradient passes whatever R is; it is a heuristic (about 0.5% worse in criterion than the
    exact search on the phantoms). Returns (idx, valid, w, f)."""
    P, K = T.shape; R = H.shape[0]; dev = T.device
    m_max = min(m_max, R)
    prep = _nnal_prep(T)
    idx = torch.full((P, m_max), -1, dtype=torch.long, device=dev); valid = torch.zeros(P, m_max, dtype=torch.bool, device=dev)
    w = torch.zeros(P, m_max, device=dev, dtype=T.dtype); f = _nnal_rowwise(torch.zeros_like(T), T, prep, 1, dtype=torch.float64)
    H2 = H * H
    for step in range(m_max):
        X = torch.einsum('pm,pmk->pk', w, H[idx.clamp(min=0)] * valid[:, :, None]) if step else torch.zeros_like(T)
        G, Z = stable_nnal_derivatives(X, T, prep); g = G @ H.T; Md = (Z @ H2.T).clamp(min=1e-12)
        in_set = _scatter_support(idx, valid, w, R)[0]
        gain_est = torch.where((g < 0) & ~in_set, 0.5 * g * g / Md, torch.zeros_like(g)) * dose
        est, cands = gain_est.topk(min(n_cand, R), dim=1)
        active = est[:, 0] > screen * lam
        if not bool(active.any()):
            break
        ap = active.nonzero().squeeze(1)
        best_f, best_w, best_idx, best_valid = f[ap].clone(), w[ap].clone(), idx[ap].clone(), valid[ap].clone()
        improved = torch.zeros(ap.numel(), dtype=torch.bool, device=dev)
        for c in range(cands.shape[1]):
            ok = est[ap, c] > screen * lam
            if not bool(ok.any()):
                break
            idx_c, valid_c, w_c = idx[ap].clone(), valid[ap].clone(), w[ap].clone()
            idx_c[:, step] = cands[ap, c]; valid_c[:, step] = True; w_c[:, step] = (-g[ap, cands[ap, c]] / Md[ap, cands[ap, c]]).clamp(min=0)
            w_new, f_new = _fit_free_sets(T[ap], H, idx_c, valid_c, w_c)
            better = ok & ((f[ap] - f_new) * dose > lam) & (f_new < best_f)
            best_f = torch.where(better, f_new, best_f); best_w[better] = w_new[better]; best_idx[better] = idx_c[better]; best_valid[better] = valid_c[better]
            improved |= better
        acc = ap[improved]; idx[acc] = best_idx[improved]; valid[acc] = best_valid[improved]; w[acc] = best_w[improved]; f[acc] = best_f[improved]
    for slot in range(1, m_max):                                                           # backward pass
        cand = (valid.sum(1) >= 2) & valid[:, slot]
        if not bool(cand.any()):
            continue
        cp = cand.nonzero().squeeze(1); valid_c = valid[cp].clone(); valid_c[:, slot] = False
        w_c, f_c = _fit_free_sets(T[cp], H, idx[cp], valid_c, w[cp])
        drop = (f_c - f[cp]) * dose < lam; dp = cp[drop]
        valid[dp] = valid_c[drop]; w[dp] = w_c[drop]; f[dp] = f_c[drop]
    return idx, valid, w, f


def _select_enumerate(T, H, W, dose, lam, w_max_steps, compile_mode):
    """Every nonempty subset as one batched constrained solve (2^R - 1 solves): the reference search, R <= 8."""
    R, K = H.shape
    if R > 8:
        raise ValueError("enumeration of 2^R - 1 subsets is limited to R <= 8; use method='branch_bound'")
    _, _, rowwise, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    subsets = [list(c) for r in range(1, R + 1) for c in itertools.combinations(range(R), r)]
    crit = [rowwise(torch.zeros_like(T), T, prep, 1, dtype=torch.float64) * dose]
    fits = [rowwise(torch.zeros_like(T), T, prep, 1, dtype=torch.float64)]
    W_sub = []
    for S in subsets:
        idx = torch.tensor(S, device=T.device)
        Ws = solve_W(T, H[idx].contiguous(), W[:, idx].contiguous(), w_max_steps, 1e-12, compile_mode=compile_mode)
        f = rowwise(Ws @ H[idx], T, prep, 1, dtype=torch.float64)
        crit.append(f * dose + lam * len(S)); fits.append(f); W_sub.append(Ws)
    C = torch.stack(crit, 1); best = C.argmin(1); F = torch.stack(fits, 1)
    W0 = torch.zeros_like(W)
    for j, S in enumerate(subsets):
        m = best == j + 1
        if m.any():
            W0[m.nonzero().squeeze(1)[:, None], torch.tensor(S, device=T.device)[None, :]] = W_sub[j][m]
    return W0 > 0, W0, F.gather(1, best[:, None]).squeeze(1)


def select_supports(T, W, H, dose, penalty=None, method="branch_bound", k_top=6, m_max=4, wald_screen=0.0,
                    w_max_steps=100, compile_mode=None):
    """Choose each pixel's material subset by penalised likelihood: minimise dose * f_p(subset) + penalty * |subset|.

    method 'branch_bound' (default; any R): exact single-material fits, then subsets of 2..m_max among each pixel's
    k_top best singletons, pruned by the full-model lower bound. On the three-material phantom it returns the
    enumeration's supports on every pixel (37k pixels, doses 30 and 300) and on a synthetic eight-material problem
    with sparse (1% of pixels) and faint (0.1-0.2 of the common amplitude) materials it matches the enumeration's
    recall of every material at k_top = 6, within 1e-4 of its criterion, in about one full W solve, linear in R.
    It presumes distinct spectra and supports of at most m_max materials: with spectra far from identified (a rank
    well above the number of materials) the enumeration can prefer larger, poorly conditioned subsets that this
    search never visits. 'greedy': forward selection on gradient-based gain estimates, a few gradient passes whatever
    R; it misses about a third of the sparse materials on that problem, so it is a speed option only. 'enumerate':
    the 2^R - 1 subsets (R <= 8), the reference. wald_screen > 0 skips the single-material fit of material r for the
    pixels whose full-fit Wald statistic dose * w_pr^2 * M_pr / 2 is below wald_screen * penalty; it saves the
    singleton stage for sparse supports (2-3x at rank 3) but a faint material whose coefficient the constrained full
    fit truncated to zero is then never considered: at 0.5 it cost a quarter of the sparse-and-faint detections on
    the synthetic problem and nothing on the phantom.

    Returns (support (P, R) bool, W0 (P, R) the coefficients on the supports, f (P,) the per-pixel loss)."""
    R, K = H.shape
    lam = 2.0 * math.log(K) if penalty is None else float(penalty)
    if method == "enumerate":
        return _select_enumerate(T, H, W, dose, lam, w_max_steps, compile_mode)
    if method == "greedy":
        idx, valid, w, f = _select_greedy(T, H, dose, lam, m_max=max(m_max, 1))
    elif method == "branch_bound":
        prep = _nnal_prep(T)
        X = W @ H
        f_full = _nnal_rowwise(X, T, prep, 1, dtype=torch.float64)
        plausible = None
        if wald_screen > 0:
            _, Z = stable_nnal_derivatives(X, T, prep)
            wald = 0.5 * dose * (W.double() ** 2) * (Z @ (H * H).T).double()
            plausible = wald > wald_screen * lam
        idx, valid, w, f = _select_branch_bound(T, H, dose, lam, f_full, k_top=k_top, m_max=m_max, plausible=plausible)
    else:
        raise ValueError(f"method must be 'branch_bound', 'greedy' or 'enumerate', got {method!r}")
    support, W0 = _scatter_support(idx, valid, w, R)
    return support, W0, f


def _warn_collinear_rows(H, cosine=0.999):
    """Two spectra that came out (nearly) proportional split their pixels' coefficients arbitrarily: say so."""
    Hn = H.double() / H.double().norm(dim=1, keepdim=True).clamp_min(1e-300)
    C = (Hn @ Hn.T).fill_diagonal_(0)
    if bool((C > cosine).any()):
        i, j = (C > cosine).nonzero()[0].tolist()
        log.warning("spectra %d and %d are proportional (cosine %.4f): their pixel coefficients are not separately "
                    "identified; consider a smaller rank", i, j, C[i, j].item())


def support_selected_spectra(T, W, H, dose, penalty=None, max_steps=300, cg_max=10, rel_tol=1e-10,
                             w_max_steps=100, compile_mode=None, verbose=False, method="branch_bound", k_top=6, m_max=4,
                             wald_screen=0.0, free_refit=False, min_support=None):
    """Choose each pixel's material subset by penalised likelihood, then refit with the supports fixed.

    The truncation bias of the ML spectra (see unconstrained_spectra) comes from
    coefficients whose true value is zero. If those are identified and held at
    zero, the remaining coefficients sit in the interior and only the much
    smaller curvature bias is left, while W >= 0 -- and the variance reduction it
    brings -- is kept. Each pixel takes the subset minimising
    dose * loss + penalty * |subset| (the empty subset allowed: a pixel with no
    material), found by `select_supports` (branch and bound by default, any R),
    and (W on the selected supports, H) is then refit jointly. The selection is a
    model-selection step and carries its own errors; a stronger penalty helped
    monotonically up to the default 2 log K, and a single select/refit round is
    the optimum (iterating degrades). It lifts the maps a little as well.

    Args:
        dose: open-beam counts per pixel and bin, which converts the loss to
            log-likelihood units for the penalty.
        penalty: per selected coefficient, in log-likelihood units. Default 2 log K.
        method, k_top, m_max, wald_screen: the subset search, see `select_supports`.
        free_refit: True drops the bound on the selected coefficients during the joint
            refit (as unconstrained_spectra does for all of them) and re-solves W >= 0
            on the supports afterwards. The penalty then only has to zero the
            coefficients that are clearly absent; a coefficient admitted by mistake
            contributes zero-mean noise instead of the truncation bias, so a smaller
            penalty can be used where the selection lacks power (low dose, faint
            materials). With penalty 0 every material the constrained pixel fit uses is
            kept and the estimator approaches unconstrained_spectra; with the default
            penalty and free_refit=False it is the constrained estimator above.
        min_support: components selected in fewer pixels than this revert to the MLE's
            treatment (free in every pixel, W >= 0 deciding); default max(2R, P / 10^4).

    Returns (W, H, support, steps) with support a bool mask of W's shape (all True in a
    column that reverted).
    """
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    support, W0, _ = select_supports(T, W, H, dose, penalty=penalty, method=method, k_top=k_top, m_max=m_max,
                                     wald_screen=wald_screen, w_max_steps=w_max_steps, compile_mode=compile_mode)
    _guard_components(support, W, W0, min_support)
    Wn, Hn, steps, _ = _joint_newton_pcg(T, W0, H, max_steps=max_steps, cg_max=cg_max, rel_tol=rel_tol,
                                         prep=prep, nnal=nnal_fn, deriv=deriv, w_mask=support, nonneg_W=not free_refit)
    if free_refit:
        Wn = _solve_W_on_support(T, Hn, Wn, support)
    _warn_collinear_rows(Hn)
    if verbose:
        print(f'  supports ({method}): mean size {support.sum(1).double().mean().item():.2f}; joint refit {steps} steps', flush=True)
    return Wn, Hn, support, steps
