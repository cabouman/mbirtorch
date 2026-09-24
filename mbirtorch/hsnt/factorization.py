"""The maximum-likelihood NNAL factorization of a transmission tensor, the solver behind dehydrate."""

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
    (W, H) jointly, and stops after five consecutive steps whose relative loss change is at most rel_tol; at 1e-8
    the result is reproducible across starts and compilation.

    Args:
        T (torch.Tensor): Transmission ratio, (pixels, bins): counts divided by the open-beam counts, zero counts
            allowed.
        num_materials (int): Rank of the factorization, at most min(pixels, bins).
        max_steps (int, optional): Iteration cap. Defaults to 1000.
        rel_tol (float, optional): Relative change in the float64 loss per step at which to stop. Defaults to 1e-8.
        compile_mode (str, optional): 'auto' compiles the hot kernels with torch.compile on CUDA with a working
            Triton when T has at least 5e8 entries (about 400k pixels at 1200 bins), where one solve repays the
            compile; 'on' always compiles; 'off' never does. Defaults to 'auto'.
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
