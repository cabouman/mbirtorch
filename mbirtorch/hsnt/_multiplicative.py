import torch


def _shifted(V, ratio, shift, mean_dim):
    """V <- max(V * ratio + d * (ratio - 1), 0), the shifted multiplicative step with offset d = shift * scale.

    A plain multiplicative step cannot move an entry off zero. With the offset, an interior fixed point needs
    ratio = 1 and a zero entry stays at zero only while ratio <= 1, i.e. while its gradient is not negative, so the
    fixed points are the KKT points for any d > 0 (the inadmissible-zero offset of Chi & Kolda 2012). The scale is
    the per-component mean, floored at 1 % of the factor's mean so a collapsed component still gets an offset.
    """
    per_component = V.mean(dim=mean_dim, keepdim=True)
    floor = 1e-2 * V.mean()
    scale = shift * torch.maximum(per_component, floor).clamp_min(torch.finfo(V.dtype).tiny)
    return (V * ratio + scale * (ratio - 1.0)).clamp_(min=0)


def _rebalance(W, H):
    """Equalize each component's scale between W and H, leaving W @ H unchanged.

    Nothing in a multiplicative update fixes the scaling W -> W D, H -> D^-1 H, and at low dose it drifts to
    extremes that starve the shifted step's offset and strain float32.
    """
    w = W.norm(dim=0)
    h = H.norm(dim=1)
    tiny = torch.finfo(W.dtype).tiny
    d = torch.sqrt((h.clamp_min(tiny) / w.clamp_min(tiny)))
    d = torch.where((w > tiny) & (h > tiny), d, torch.ones_like(d))
    return W * d, H / d[:, None]


def _reseed_dead(W, H, rel_tol=1e-6):
    """Re-seed any component that is zero in both factors, with small random values from a fixed generator.

    Its ratio is zero in either update, so no multiplicative or Newton step can revive it; it is a degenerate
    stationary point. A random rather than constant seed keeps the revived spectrum from being flat.
    """
    w = W.norm(dim=0)
    h = H.norm(dim=1)
    dead = (w <= rel_tol * w.max()) & (h <= rel_tol * h.max())
    n_dead = int(dead.sum())
    if n_dead == 0:
        return W, H
    live = ~dead
    W = W.clone()
    H = H.clone()
    w_ref = W[:, live].mean() if bool(live.any()) else W.new_tensor(1.0)
    h_ref = H[live].mean() if bool(live.any()) else H.new_tensor(1.0)
    g = torch.Generator(device=W.device).manual_seed(0)
    W[:, dead] = 1e-2 * w_ref * torch.rand(W.shape[0], n_dead, generator=g, dtype=W.dtype, device=W.device)
    H[dead] = 1e-2 * h_ref * torch.rand(n_dead, H.shape[1], generator=g, dtype=H.dtype, device=H.device)
    return W, H


def multiplicative_update(W: torch.Tensor, H: torch.Tensor, T: torch.Tensor, update_H: bool = True):
    """One damped, shifted multiplicative sweep of the NNAL factorization.

    Args:
        W: Pixel coefficients (pixels x materials).
        H: Spectra (materials x bins).
        T: Transmission ratio (pixels x bins).
        update_H: If False, keep H fixed and update only W.

    Returns:
        (W, H) after the sweep.
    """
    shift, ratio_max = 1e-2, 2.0
    Z = torch.exp(-W @ H)
    # The square-root damping alone does not bound the ratio: at very low dose T @ H.T reaches the clip and the
    # ratio ~1e15, which the offset term turns into NaN; cap it.
    W = _shifted(W, (((Z @ H.T) / torch.clip(T @ H.T, min=1e-30)) ** 0.5).clamp(max=ratio_max), shift, 0)
    if update_H:
        H = _shifted(H, (((W.T @ Z) / torch.clip(W.T @ T, min=1e-30)) ** 0.5).clamp(max=ratio_max), shift, 1)
        W, H = _reseed_dead(W, H)
        W, H = _rebalance(W, H)
    return W, H
