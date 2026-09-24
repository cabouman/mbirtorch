import torch

from ._linalg import nndsvda
from ._loss import _nnal_prep, stable_nnal
from ._multiplicative import multiplicative_update
from ._newton import _resolve_compile, block_newton_optimize, joint_newton_optimize
from ._lbfgsb import lbfgsb_optimize


def _initial_factors(T, num_materials):
    """NNDSVDa initialization in the attenuation domain.

    The model is X = W @ H with X = -log(T), so T itself is not low rank. A zero count only says the attenuation
    exceeds that of the faintest pixel that did register, so it is floored at half the smallest positive
    transmission; transmissions below 1e-12 count as zero counts (generate_hyper_data marks them 1e-30).
    """
    real = T > 1e-12
    if bool(real.any()) and not bool(real.all()):
        T_for_init = torch.where(real, T, 0.5 * T[real].min())
    else:
        T_for_init = T.clamp_min(torch.finfo(T.dtype).tiny)
    return nndsvda(-torch.log(T_for_init), n_components=num_materials)


def optimize(T, update, num_materials, max_steps, rel_tol, update_H=True, W_init=None, H_init=None,
             compile_mode=None, extrapolate=False, extrapolate_check_every=5):
    """Run one solver from the default or a given start. Returns (W, H, steps).

    update is one of the solver entry points below or multiplicative_update, which this function iterates; with
    extrapolate=True it adds Nesterov momentum to that sweep (see the comment in the loop).
    """
    if W_init is None and H_init is None:
        W_init, H_init = _initial_factors(T, num_materials)
    elif W_init is None:
        W_init = torch.linalg.lstsq(H_init.T, T.T)[0].T.clamp(min=0)
    elif H_init is None:
        H_init = torch.linalg.lstsq(W_init, T)[0].clamp(min=0)

    if update in (block_newton_optimize, joint_newton_optimize):
        return update(T, num_materials, max_steps, rel_tol, update_H=update_H, W_init=W_init, H_init=H_init,
                      compile_mode=compile_mode)
    if update is lbfgsb_optimize:
        return update(T, num_materials, max_steps, rel_tol, update_H=update_H, W_init=W_init, H_init=H_init)

    prep = _nnal_prep(T)
    W, H = W_init, H_init
    num_steps = 0
    if extrapolate:
        # Nesterov momentum around the multiplicative map, beta_k = (k - 1) / (k + 2), restarted whenever a checked
        # sweep raises the loss. The extrapolated point is only an input: the answer is the best plain iterate, so
        # the fixed points are unchanged. The loss along a momentum sequence is not monotone, so the solve stops
        # after two consecutive checks without a relative improvement of rel_tol per sweep.
        log_T, positive, _, _ = prep
        const = torch.sum(torch.where(positive, T * (1.0 - log_T), torch.zeros_like(T)), dtype=torch.float64)
        cheap = lambda X: torch.sum(torch.exp(-X) + T * X, dtype=torch.float64) - const
        noise = 1e-9                                  # float32 noise on the raw sum is ~1e-10 relative
        Wp, Hp = W, H                                 # last plain iterate
        Wy, Hy = W, H                                 # extrapolated input to the sweep
        prev, j = None, 0
        best, Wb, Hb, stalls = None, W, H, 0
        for i in range(max_steps):
            Wn, Hn = update(Wy, Hy, T, update_H=update_H)
            num_steps = i + 1
            j += 1
            if num_steps % extrapolate_check_every == 0:
                L = cheap(Wn @ Hn)
                if prev is not None and (not torch.isfinite(L) or bool(L > prev * (1.0 + noise))):
                    Wy, Hy, j = Wp, Hp, 0
                    continue
                first = best is None
                if torch.isfinite(L) and (first or bool(L < best)):
                    improved = first or bool(best - L > rel_tol * extrapolate_check_every * torch.abs(L))
                    best, Wb, Hb = L, Wn, Hn
                else:
                    improved = False
                stalls = 0 if improved else stalls + 1
                if rel_tol > 0 and prev is not None and stalls >= 2:
                    break
                prev = L
            beta = (j - 1) / (j + 2) if j > 1 else 0.0
            Wy = (Wn + beta * (Wn - Wp)).clamp_(min=0)
            Hy = (Hn + beta * (Hn - Hp)).clamp_(min=0)
            Wp, Hp = Wn, Hn
        return Wb, Hb, num_steps

    # A float64 sum: in float32 the loss is quantized coarser than the per-step progress at low dose.
    prev_loss = stable_nnal(W @ H, T, prep, dtype=torch.float64)
    for i in range(max_steps):
        W, H = update(W, H, T, update_H=update_H)
        num_steps = i + 1
        if rel_tol > 0:
            loss_new = stable_nnal(W @ H, T, prep, dtype=torch.float64)
            converged = torch.abs(loss_new - prev_loss) / (prev_loss + 1e-30) < rel_tol
            prev_loss = loss_new
            if converged:
                break
    return W, H, num_steps


def nnal_factorization(T, method='joint_newton', num_materials=3, max_steps=1000, rel_tol=1e-8, compile_mode='auto',
                       **kwargs):
    """Factorize the transmission ratio T ~= exp(-W @ H), W, H >= 0, by minimizing the non-negative attenuation loss.

    The loss sum[exp(-X) + T X], X = W @ H, is the Poisson negative log-likelihood of the counts up to a constant
    and a factor of the dose.

    Args:
        T (torch.Tensor): Transmission ratio, shape (pixels, bins), counts divided by the open-beam counts. Zero
            counts are allowed.
        method (str): 'joint_newton' (default): a few block-Newton steps, then a matrix-free truncated Newton solve
            on (W, H) jointly; the fastest to a given loss. 'block_newton': alternating exact projected Newton on
            each factor (linear convergence). 'multiplicative': the damped, shifted multiplicative update with
            Nesterov extrapolation (sublinear). 'lbfgsb': scipy's L-BFGS-B over both factors, a generic baseline.
        num_materials (int): Rank of the factorization. Defaults to 3.
        max_steps (int): Iteration cap. Defaults to 1000.
        rel_tol (float): Relative change in the float64 loss per step at which to stop. joint_newton stops after
            five consecutive steps below it, which makes the result reproducible across starts and compilation at
            1e-8; the other methods stop at the first such step. Defaults to 1e-8. On data the model fits exactly a
            projected-gradient test takes over and runs to machine precision.
        compile_mode (str): 'auto' (default) compiles the hot kernels with torch.compile on CUDA with a working
            Triton when T has at least 5e8 entries (about 400k pixels at 1200 bins), where one solve repays the
            compile; 'on' always compiles; 'off' never does. Ignored by lbfgsb.
        **kwargs: W_init and H_init (start from given factors); extrapolate for the multiplicative method.

    Returns:
        (W, H, steps): W of shape (pixels, num_materials), H of shape (num_materials, bins), on T's device and in
        T's dtype, and the number of steps taken.
    """
    if method == 'multiplicative':
        update = multiplicative_update
        # Decided before torch.compile replaces `update` with a wrapper, which the identity test would miss.
        kwargs.setdefault('extrapolate', True)
    elif method == 'block_newton':
        update = block_newton_optimize
    elif method == 'joint_newton':
        update = joint_newton_optimize
    elif method == 'lbfgsb':
        update = lbfgsb_optimize
    else:
        raise ValueError("Invalid method. Choose 'joint_newton', 'block_newton', 'multiplicative' or 'lbfgsb'.")

    compile_mode = _resolve_compile(compile_mode, T)
    if update is multiplicative_update and compile_mode == 'on':
        from ..projectors import maybe_compile
        update = maybe_compile(update, True)
    return optimize(T, update, num_materials, max_steps, rel_tol, compile_mode=compile_mode, **kwargs)
