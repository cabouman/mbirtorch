import torch

from . import _newton
from ._loss import _nnal_prep
from ._device import _default_device
from ._newton import _kernels, _resolve_compile, solve_W
from .factorization import _nnal_factorization


def _h_stats_accumulate(W, H, T, prep, rows, cols, deriv, rowwise):
    """Per-chunk sufficient statistics for one Newton step on H.

    The H-step of block_newton needs, for every wavelength bin, the gradient
    W^T G[:, k] and the Hessian W^T diag(Z[:, k]) W. Both are sums over pixels, so
    they accumulate across chunks: this returns one chunk's share, and the caller
    adds. The Hessian's upper triangle for all K bins comes out of a single GEMM
    against the Khatri-Rao product (W[:, rows] * W[:, cols]), exactly as in
    block_newton_step. The per-bin loss is the line search's baseline.
    """
    X = W @ H
    G, Z = deriv(X, T, prep)
    return W.T @ G, (W[:, rows] * W[:, cols]).T @ Z, rowwise(X, T, prep, 0, dtype=torch.float64)


def _h_direction(H, grad, flat, rows, cols, jitter_rel=1e-9):
    """Projected-Newton direction on H from accumulated statistics: the H axis of
    block_newton_step on the (K, R) transpose. Returns (d, slope, alpha_max) with
    one row/entry per bin; see _newton._two_metric_direction."""
    d, slope, alpha, _, _ = _newton._two_metric_direction(H.T, grad.T, flat.T, rows, cols, jitter_rel)
    return d, slope, alpha


# Each chunk's W solve (block Newton, stopping tolerance and cap), and the step lengths tried per H line search.
_W_REL_TOL, _W_MAX_STEPS, _LS_TRIALS = 1e-8, 300, 4


def _stream_factorization(chunks, num_materials, max_passes=5, rel_tol=1e-6, warmup_pixels=16384, device=None,
                          compile_mode='off', verbose=0, stats=None, nonneg_W=True, support_selection=None):
    """Factorize a dataset too large for device memory, one chunk of pixels at a time.

    W is separable over pixels, so it is solved chunk by chunk and never held whole on the device. H holds only R * K
    values, and its Newton step needs only sums over pixels (gradient, per-bin R x R Hessian, per-bin loss), which
    accumulate across chunks: one pass over the data gives one exact Newton step on H, and a second pass evaluates
    its line search at a few step lengths at once. H starts from a joint-Newton fit on a subsample, so a handful of
    passes polish it. The joint Newton solver is not streamed: each of its CG iterations would be a full pass.

    Args:
        chunks (sequence of torch.Tensor): Chunks of the transmission ratio, each (pixels, bins), together making
            up T; any indexable sequence works, so chunks may be loaded lazily.
        num_materials (int): Rank R.
        max_passes (int, optional): Polish passes over the data; 0 keeps the subsample fit. Defaults to 5.
        rel_tol (float, optional): Stop when a pass changes the total loss by less than this, relatively.
            Defaults to 1e-6.
        warmup_pixels (int, optional): Pixels from the leading chunks for the initial fit. Defaults to 16384.
        device (str, optional): Torch device. Defaults to None, meaning CUDA if available, else CPU.
        compile_mode (str, optional): 'auto', 'on' or 'off', as for the in-memory solver, judged on one chunk.
            Defaults to 'off'.
        verbose (int, optional): 1 prints the loss and KKT residual of every pass. Defaults to 0.
        stats (dict, optional): Receives 'loss' and 'kkt' lists, one entry per pass; the KKT residual of H is
            ||P(grad_H L)|| / ||W^T T||. Defaults to None.
        nonneg_W (bool, optional): False estimates H with the bound on W dropped during the polish passes (the
            unconstrained spectra), then re-solves W >= 0 for every chunk. Defaults to True.
        support_selection (dict, optional): Support selection after the passes: 'dose' (required) and optionally
            'penalty' ('auto' or a multiple of log K; 'auto' is judged once from every chunk), 'wald_screen',
            'free_refit' and 'max_passes' (the refit's pass budget). One pass selects each chunk's supports; the
            polish loop then runs again with W confined to them. The supports are returned in
            stats['support_chunks'], the refit's losses in stats['loss_refit'] and stats['kkt_refit'].
            Defaults to None.

    Returns:
        (W_chunks, H, passes): W as a list of CPU tensors aligned with the chunks, H, and the polish passes made.
    """
    device = _default_device(device)
    compile_mode = _resolve_compile(compile_mode, chunks[0], device)
    _, deriv, rowwise, _ = _kernels(compile_mode)
    R = num_materials
    W_chunks = [None] * len(chunks)

    # H from a subsample of the leading chunks.
    parts, n = [], 0
    for c in chunks:
        parts.append(c[: warmup_pixels - n])
        n += parts[-1].shape[0]
        if n >= warmup_pixels:
            break
    T_sub = torch.cat(parts, 0).to(device)
    _, H, _ = _nnal_factorization(T_sub, R, max_steps=300, rel_tol=1e-6, compile_mode=compile_mode)
    del T_sub
    rows, cols = torch.triu_indices(R, R, device=H.device)
    pin = torch.device(device).type == 'cuda'

    def to_device(c):
        if c.device.type == 'cpu' and pin:
            c = c.pin_memory().to(device, non_blocking=True)
        else:
            c = c.to(device)
        return c

    def polish(solve_chunk, passes_max, tag):
        """Passes of: W per chunk (solve_chunk), H statistics accumulated over chunks, one exact Newton step on H
        with a per-bin line search evaluated in a second pass. Updates H and W_chunks; returns the passes made."""
        nonlocal H
        prev_loss = None
        passes = 0
        for p in range(passes_max + 1):
            # Pass A: W per chunk with H fixed; H's statistics summed over chunks in float64, since the per-bin
            # loss must resolve improvements far below the float32 ulp of a sum near 1e8.
            grad = torch.zeros(H.shape, dtype=torch.float64, device=H.device)
            flat = torch.zeros(rows.numel(), H.shape[1], dtype=torch.float64, device=H.device)
            base = torch.zeros(H.shape[1], dtype=torch.float64, device=H.device)
            scale = torch.zeros(H.shape, dtype=torch.float64, device=H.device)     # W^T T, the gradient's natural scale
            nxt = to_device(chunks[0])
            for i in range(len(chunks)):
                Tc = nxt
                if i + 1 < len(chunks):
                    nxt = to_device(chunks[i + 1])            # prefetch overlaps the solve below
                prep = _nnal_prep(Tc)
                W0 = W_chunks[i].to(device=device, dtype=H.dtype) if W_chunks[i] is not None else None
                W = solve_chunk(Tc, W0, i)
                W_chunks[i] = W.cpu()
                g_c, f_c, b_c = _h_stats_accumulate(W, H, Tc, prep, rows, cols, deriv, rowwise)
                grad += g_c
                flat += f_c
                base += b_c
                scale += (W.T @ Tc).to(torch.float64)
                del Tc, W
            loss = base.sum(dtype=torch.float64)
            # Projected gradient: where H is zero only a negative gradient (a wish to grow) counts.
            pg = torch.where(H > 0, grad, grad.clamp(max=0))
            kkt = (pg.norm() / scale.norm()).item()
            if stats is not None:
                stats.setdefault('loss' + tag, []).append(loss.item())
                stats.setdefault('kkt' + tag, []).append(kkt)
            if verbose:
                print(f'  pass {p}{tag}: full-data loss {loss.item():.6e}  KKT residual {kkt:.2e}', flush=True)
            if prev_loss is not None and rel_tol > 0 and bool(torch.abs(loss - prev_loss) <= rel_tol * torch.abs(loss)):
                break
            prev_loss = loss
            if p == passes_max:
                break

            # One exact Newton step on H from the accumulated statistics.
            d, slope, alpha_max = _h_direction(H, grad.to(H.dtype), flat.to(H.dtype), rows, cols)
            alphas = alpha_max[None, :] * (0.5 ** torch.arange(_LS_TRIALS, dtype=H.dtype, device=H.device))[:, None]

            # Pass B: the per-bin loss at every trial step, summed over chunks.
            trial = torch.zeros(_LS_TRIALS, H.shape[1], dtype=torch.float64, device=H.device)
            nxt = to_device(chunks[0])
            for i in range(len(chunks)):
                Tc = nxt
                if i + 1 < len(chunks):
                    nxt = to_device(chunks[i + 1])
                prep = _nnal_prep(Tc)
                W = W_chunks[i].to(device)
                X = W @ H
                B = W @ d.T
                for t in range(_LS_TRIALS):
                    trial[t] += rowwise(X - alphas[t][None, :] * B, Tc, prep, 0, dtype=torch.float64)
                del Tc, W, X, B
            # Same floor as block_newton_step (see _ARMIJO_FLOOR): the float32 sums are
            # gone, but elements whose step falls below ulp(X) still do not move.
            noise = _newton._ARMIJO_FLOOR * torch.finfo(H.dtype).eps * base.abs()
            # Armijo, per bin and trial
            ok = trial <= base[None, :] - 1e-4 * alphas.double() * slope.double()[None, :] + noise[None, :]
            # largest accepted trial per bin, else zero
            accepted = torch.where(ok.any(0), alphas.gather(0, ok.float().argmax(0, keepdim=True)).squeeze(0),
                                   torch.zeros_like(alpha_max))
            Ht = (H.T - accepted[:, None] * d).clamp_(min=0)
            # Same epsilon-active snap as block_newton_step: a bin component at the
            # bound with an outward gradient becomes exactly zero, not a residue.
            eps_active = _newton._ACTIVE_TOL * Ht.abs().amax(-1, keepdim=True).mean()
            Ht = torch.where((Ht <= eps_active) & (grad.T.to(Ht.dtype) > 0), torch.zeros_like(Ht), Ht)
            H = Ht.T.contiguous()
            passes = p + 1
        return passes

    def solve_mle(Tc, W0, i):
        return solve_W(Tc, H, W0, _W_MAX_STEPS, _W_REL_TOL, nonneg=nonneg_W, compile_mode=compile_mode)

    passes = polish(solve_mle, max_passes, '')
    if not nonneg_W:
        # The physical coefficients: one more pass, W >= 0 given the final H.
        for i in range(len(chunks)):
            Tc = to_device(chunks[i])
            W = solve_W(Tc, H, W_chunks[i].to(device=device, dtype=H.dtype).clamp(min=0), _W_MAX_STEPS, _W_REL_TOL,
                        compile_mode=compile_mode)
            W_chunks[i] = W.cpu()
            del Tc, W

    if support_selection is not None:
        from .spectra import (_auto_penalty, _select_supports, _solve_W_on_support, _warn_collinear_rows,
                              _weak_components)
        opt = dict(support_selection)
        dose = opt.pop('dose')
        free = bool(opt.pop('free_refit', False))
        refit_passes = opt.pop('max_passes', max_passes)
        penalty = opt.pop('penalty', 'auto')
        wald_screen = opt.pop('wald_screen', 0.0)
        if opt:
            raise TypeError(f"unknown support_selection keys: {sorted(opt)}")
        if penalty == 'auto':                    # judged once from the whole data, not per chunk
            penalty = _auto_penalty(torch.cat([c.float().mean(1) for c in chunks]), dose)
        # One pass: each chunk's supports given the polished H (the MLE W is kept for the component guard).
        S_chunks = [None] * len(chunks)
        W_mle = list(W_chunks)
        counts = 0
        P_total = 0
        for i in range(len(chunks)):
            Tc = to_device(chunks[i])
            Wc = W_chunks[i].to(device=device, dtype=H.dtype)
            support, W0, _ = _select_supports(Tc, Wc, H, dose, penalty=penalty, wald_screen=wald_screen)
            S_chunks[i] = support.cpu()
            W_chunks[i] = W0.cpu()
            counts = counts + support.sum(0)
            P_total += Tc.shape[0]
            del Tc, Wc, support, W0
        weak = _weak_components(counts, P_total)
        if bool(weak.any()):
            weak_cpu = weak.cpu()
            for i in range(len(chunks)):
                S_chunks[i][:, weak_cpu] = True
                W_chunks[i][:, weak_cpu] = W_mle[i][:, weak_cpu]
        del W_mle

        # The polish loop again, W confined to the supports (free-signed there if free_refit).
        def solve_masked(Tc, W0, i):
            return _solve_W_on_support(Tc, H, W0, S_chunks[i].to(device), nonneg=not free)

        refit_passes = polish(solve_masked, refit_passes, '_refit')
        if free:
            for i in range(len(chunks)):
                Tc = to_device(chunks[i])
                W_c = W_chunks[i].to(device=device, dtype=H.dtype)
                W_chunks[i] = _solve_W_on_support(Tc, H, W_c, S_chunks[i].to(device)).cpu()
                del Tc
        _warn_collinear_rows(H)
        if stats is not None:
            stats['support_chunks'] = S_chunks
            stats['refit_passes'] = refit_passes
    return W_chunks, H, passes
