"""All-at-once L-BFGS-B on (W, H) >= 0: the generic bound-constrained baseline for the NNAL factorization."""
import numpy as np
import torch

from ._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives


def lbfgsb_optimize(T, num_materials, max_steps, rel_tol, update_H=True, W_init=None, H_init=None,
                    memory=20, max_evals=None, verbose=False):
    """Minimize the NNAL over both factors at once with L-BFGS-B under the bounds W, H >= 0.

    This is the generic solver for a bound-constrained low-rank likelihood, the
    one generalized CP decomposition uses for the same Poisson log-link loss
    (Hong, Kolda & Duersch 2020), and the baseline any second-order method must
    beat on the same gradient. scipy's Fortran L-BFGS-B runs on the CPU in
    float64 over the (P + K) R parameters; every function and gradient
    evaluation is one pass over T on the device -- exp(-X) and two GEMMs -- in
    T's dtype with a float64 sum, exactly the kernels the Newton solvers use.
    The parameters cross to the device as one float64 vector per evaluation.

    rel_tol maps onto L-BFGS-B's ftol, which stops when the relative decrease of
    the loss per iteration falls below it: the same meaning rel_tol has for every
    other method here. max_steps maps onto maxiter; the projected-gradient test is
    disabled so ftol decides. Near the float32 noise floor of the loss the Fortran
    line search can fail to find a decrease and stop early; the message is printed
    when verbose. memory is the number of correction pairs; with scipy's default of
    10 the relative-decrease test can fire on a slow crawl short of the optimum.
    update_H=False is not supported.

    Returns (W, H, iterations).
    """
    from scipy.optimize import Bounds, minimize
    if not update_H:
        raise NotImplementedError("lbfgsb_optimize updates both factors; use solve_W for a fixed H")
    if W_init is None or H_init is None:
        raise ValueError("lbfgsb_optimize needs W_init and H_init (optimize supplies the default initialization)")
    P, K = T.shape
    R = num_materials
    nW = P * R
    dev, dt = T.device, T.dtype
    prep = _nnal_prep(T)
    x0 = np.concatenate([W_init.detach().to(torch.float64).cpu().numpy().ravel(),
                         H_init.detach().to(torch.float64).cpu().numpy().ravel()])
    np.maximum(x0, 0.0, out=x0)
    evals = [0]

    def fg(x):
        evals[0] += 1
        xt = torch.from_numpy(x).to(dev)
        W = xt[:nW].view(P, R).to(dt)
        H = xt[nW:].view(R, K).to(dt)
        X = W @ H
        f = stable_nnal(X, T, prep, dtype=torch.float64).item()
        G, _ = stable_nnal_derivatives(X, T, prep)              # dL/dX = T - exp(-X)
        g = torch.cat([(G @ H.T).reshape(-1), (W.T @ G).reshape(-1)]).to(torch.float64).cpu().numpy()
        return f, g

    res = minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=Bounds(0.0, np.inf),
                   options=dict(maxiter=max_steps, maxfun=max_evals or 4 * max_steps + 100, ftol=rel_tol,
                                gtol=0.0, maxcor=memory, maxls=20))
    x = torch.from_numpy(res.x).to(dev)
    W = x[:nW].view(P, R).to(dt).clamp_(min=0).contiguous()
    H = x[nW:].view(R, K).to(dt).clamp_(min=0).contiguous()
    if verbose:
        print(f'  lbfgsb: {res.nit} iterations, {evals[0]} evaluations, {res.message}', flush=True)
    return W, H, int(res.nit)
