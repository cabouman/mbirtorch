"""The maximum-likelihood NNAL factorization of a transmission tensor, the solver behind dehydrate."""

import torch

from ._linalg import _attenuation_for_start, _nonneg_least_squares_start, nndsvda
from ._newton import _resolve_compile, joint_newton_optimize


def _initial_factors(T, num_materials):
    """NNDSVDa initialization in the attenuation domain.

    The model is X = W @ H with X = -log(T), so T itself is not low rank. A zero count only says the attenuation
    exceeds that of the faintest pixel that did register, so it is floored at half the smallest positive
    transmission; transmissions below 1e-12 count as zero counts (generate_hyper_data marks them 1e-30).
    """
    return nndsvda(_attenuation_for_start(T), n_components=num_materials)


def _nnal_factorization(T, num_materials, max_steps=1000, rel_tol=1e-8, compile_mode='auto', W_init=None,
                        H_init=None):
    """Factorize the transmission ratio T ~= exp(-W @ H), W, H >= 0, by minimizing the non-negative attenuation loss.

    The loss sum[exp(-X) + T X], X = W @ H, is the Poisson negative log-likelihood of the counts up to a constant
    and a factor of the dose. The solver runs a few block-Newton steps, then a matrix-free truncated Newton solve on
    (W, H) jointly, and stops after five consecutive steps whose relative loss change is at most rel_tol. At 1e-8 the
    stop is insensitive to rounding: compiled and eager solves, and solves from nearby starts, end within about 5e-7
    of each other in loss (1M pixels; 3e-10 for compiled against eager at dose 3). The factors can still differ where
    the loss is flat, and the problem is not convex, so a start far from the NNDSVDa one can reach another local
    optimum. At a few counts per bin the loss can also keep falling without limit along a component that grows on
    zero counts (_zero_count_divergence reports it); the factors then depend on max_steps.

    Args:
        T (torch.Tensor): Transmission ratio, (pixels, bins): counts divided by the open-beam counts, zero counts
            allowed.
        num_materials (int): Rank of the factorization, at most min(pixels, bins).
        max_steps (int, optional): Iteration cap. Defaults to 1000.
        rel_tol (float, optional): Relative change in the float64 loss per step at which to stop. Defaults to 1e-8.
        compile_mode (str, optional): 'auto' compiles the hot kernels with torch.compile on CUDA with a working
            Triton when T has at least 5e8 entries (about 417k pixels at 1200 bins; at 1M pixels a compiled step
            takes about a third of the eager time, and the break-even against a first compile is not measured);
            'on' always compiles; 'off' never does. Defaults to 'auto'.
        W_init, H_init (torch.Tensor, optional): A start; the missing factor is fitted to the attenuation by
            nonnegative least squares. Defaults to None: an NNDSVDa start.

    Returns:
        (W, H, steps): W (pixels, num_materials) and H (num_materials, bins) on T's device and in T's dtype, and the
        number of steps taken.
    """
    if not 1 <= int(num_materials) <= min(T.shape):
        raise ValueError(f"num_materials must be between 1 and min(pixels, bins) = {min(T.shape)}, got {num_materials}")
    compile_mode = _resolve_compile(compile_mode, T)
    if W_init is None and H_init is None:
        W_init, H_init = _initial_factors(T, num_materials)
    elif W_init is None:
        W_init = _nonneg_least_squares_start(_attenuation_for_start(T), H_init)
    elif H_init is None:
        H_init = _nonneg_least_squares_start(_attenuation_for_start(T).T, W_init.T).T
    return joint_newton_optimize(T, num_materials, max_steps, rel_tol, W_init=W_init, H_init=H_init,
                                 compile_mode=compile_mode)


# An attenuation above this on an entry with no counts is not constrained by the data (exp(-50) is 2e-22 of one count).
_ZERO_COUNT_BOUND = 50.0
# A component whose attenuation lies at least this share on zero-count entries models the zero counts, not a material.
# On the 65,536-pixel sphere phantom at rank 3 and 1 to 100 counts per bin, the component that took the zero counts
# held 0.9994 to 1.0 of its attenuation there (7 fits of 22), and no other component more than 0.78.
_CAPTURED_SHARE = 0.99


def _zero_count_divergence(W, H, T, chunk=2 ** 23):
    """(largest fitted attenuation on a zero-count entry, number of zero-count entries above _ZERO_COUNT_BOUND), by
    blocks of rows. The loss exp(-X) of a zero count has no minimum: at a few counts per bin the fit can grow a
    component on the zero counts for as long as it runs, and these numbers show it."""
    rows = max(1, chunk // max(T.shape[1], 1))
    x_max, n_above = 0.0, 0
    for i in range(0, T.shape[0], rows):
        zero = T[i:i + rows] <= 1e-12
        if not bool(zero.any()):
            continue
        X = W[i:i + rows] @ H
        if bool((zero & ~torch.isfinite(X)).any()):          # a fit gone non-finite counts as diverged
            x_max = float('inf')
        else:
            x_max = max(x_max, float(torch.where(zero, X, torch.zeros_like(X)).max()))
        n_above += int((zero & ~(X <= _ZERO_COUNT_BOUND)).sum())
    return x_max, n_above


def _zero_count_mass(W, H, T, chunk=2 ** 23):
    """Each component's attenuation summed over the zero-count entries, sum over zero (p, b) of W_pk H_kb, (rank,) in
    float64, by blocks of rows: divided by the component's total, (sum_p W_pk)(sum_b H_kb), the share of it the zero
    counts hold."""
    rows = max(1, chunk // max(T.shape[1], 1))
    out = torch.zeros(W.shape[1], dtype=torch.float64, device=W.device)
    Hd = H.to(torch.float64)
    for i in range(0, T.shape[0], rows):
        zero = (T[i:i + rows] <= 1e-12).to(torch.float64)
        if bool(zero.any()):
            out += (W[i:i + rows].to(torch.float64) * (zero @ Hd.T)).sum(0)
    return out
