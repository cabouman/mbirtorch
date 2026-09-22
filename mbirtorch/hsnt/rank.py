"""Rank (number of materials) estimation by sequential likelihood-ratio tests on the NNAL factorization.

`estimate_rank` is what `dehydrate` and the command line use when no rank is given. The singular-value estimate of
the L2 baseline (`l2_baseline._estimate_subspace_dimension`) is unreliable for this (1 at dose 3, 161 at dose 30 on
the three-material phantom) and is kept only for comparison.
"""
import logging

import numpy as np

log = logging.getLogger("mbirtorch.hsnt")


def _lrt_rank(T, device, seed, max_rank, label):
    """Sequential likelihood-ratio rank test on the pixels of T (a torch tensor on the device). Returns (rank, detail)."""
    import torch
    from .factorization import nnal_factorization
    from ._loss import stable_nnal
    P, K = T.shape
    torch.manual_seed(seed)
    losses, resid = [], []
    for r in range(1, max_rank + 1):
        W, H, _ = nnal_factorization(T, method="joint_newton", num_materials=r, max_steps=200, rel_tol=1e-6, random_state=seed)
        Xd = W.double() @ H.double(); Th = torch.exp(-Xd); Td = T.double()
        losses.append(stable_nnal(Xd, Td).item()); resid.append((((Td - Th) ** 2) / Th.clamp_min(1e-12)).mean().item())
        log.debug("  %s rank %d: loss %.6g, mean chi-square term %.4g", label, r, losses[-1], resid[-1])
    dose_eff = 1.0 / resid[-1]
    gains = [dose_eff * (losses[i - 1] - losses[i]) for i in range(1, len(losses))]     # gains[i - 1] belongs to component i + 1
    floor = max(0.5 * (P + K), float(np.median(gains[-3:])))
    threshold = 2.0 * floor
    rank = 1
    for r, g in zip(range(2, max_rank + 1), gains):
        if g > threshold:
            rank = r
        else:
            break
    table = ", ".join(f"{r}: {g:,.0f}" for r, g in zip(range(2, max_rank + 1), gains))
    log.info("rank search (%s, %s pixels x %d bins): effective dose %.3g; log-likelihood gain of component %s; noise floor %.0f, "
             "threshold %.0f -> rank %d", label, f"{P:,}", K, dose_eff, table, floor, threshold, rank)
    return rank, dict(pixels=P, losses=losses, gains=gains, effective_dose=dose_eff, noise_floor=floor, threshold=threshold)


def pool_pixels(T, spatial_shape, block):
    """Block-average a (pixels, bins) array over block x block detector pixels within each view; rows and columns are
    cropped to multiples of the block. The averaged transmission is the summed count over the block divided by the
    block's summed dose, so it is a valid transmission ratio at block^2 times the dose."""
    V, rows, cols = spatial_shape
    r, c = rows // block * block, cols // block * block
    X = np.asarray(T).reshape(V, rows, cols, -1)[:, :r, :c]
    X = X.reshape(V, r // block, block, c // block, block, -1).mean(axis=(2, 4))
    return X.reshape(-1, X.shape[-1])


def estimate_rank(T, spatial_shape=None, device=None, seed=0, max_rank=6, subsample=16384, pool="auto"):
    """Choose the rank by sequential likelihood-ratio tests, at full resolution and on spatially pooled pixels.

    Ranks 1..max_rank are fitted in turn and the loss gain of each added component is converted to log-likelihood
    units with a dose calibrated from the residual of the most flexible fit (mean (T - e^-X)^2 / e^-X = 1 / dose for
    Poisson noise), so a nominal or unknown open-beam dose does not matter. A component that only fits noise gains
    about (P + K) / 2, its parameter count, because every pixel gives it a free coefficient; the noise floor is the
    larger of that and the median gain of the last three ranks, and a component is accepted while its gain exceeds
    twice the floor.

    That floor grows with the pixel count as fast as a faint material's evidence does, so at low dose the test at
    full resolution misses the weakest material (aluminium in the phantoms below dose ~18). Pooling blocks of
    neighbouring pixels keeps the evidence, the summed counts stay Poisson, but divides the nuisance count, so the
    same test on pooled pixels has far more power: on the sphere phantom pooling 8x8 recovers the true rank 3 at
    dose 1 where full resolution gives 1, without over-estimating up to dose 1e4. The block is chosen so the pooled
    pixel count falls to about the bin count, below which the floor is dominated by the spectrum's own K parameters
    and pooling buys nothing more. The larger of the two ranks is returned: over-estimation costs a fraction of a
    decibel while under-estimation caps the SNR. pool='auto' picks the block from the calibrated dose so pooled pixels
    hold about 64 counts per bin (no pooling above dose 64, where pooled mixed pixels start to add spurious rank);
    an integer fixes it; 0 disables it.

    Args:
        T: transmission ratio, (pixels, bins), numpy or torch.
        spatial_shape: (views, rows, cols) of the pixels, needed for pooling; None disables pooling.
        device: torch device for the solves (default: cuda if available).

    Returns (rank, note, detail)."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    T_np = T.detach().cpu().numpy() if torch.is_tensor(T) else np.asarray(T)
    pixels, K = T_np.shape
    stride = max(1, pixels // subsample)
    Tt = torch.from_numpy(np.ascontiguousarray(T_np[::stride], dtype=np.float32)).to(device)
    log.info("estimating the rank: ranks 1..%d on %s pixels (every %d-th) at full resolution", max_rank, f"{Tt.shape[0]:,}", stride)
    rank_full, d_full = _lrt_rank(Tt, device, seed, max_rank, "full resolution")
    block = 0
    if spatial_shape is not None and pool:
        V, rows, cols = spatial_shape
        if pool == "auto":
            # Pool only as much as the counts require: to about 64 counts per pooled pixel and bin (8x8 at dose 1, 2x2
            # at dose 30, none above dose 64). Pooling at high dose over-estimates the rank: pooled pixels of mixed
            # composition are not exactly low-rank (the exponential is applied to the block-averaged transmission) and
            # that misfit's likelihood gain grows with dose; on the sphere phantom it chose rank 4 from dose 32 upward
            # with an 8x8 block.
            block = int(np.ceil(np.sqrt(64.0 / max(d_full["effective_dose"], 1e-9))))
            block = int(min(block, np.ceil(np.sqrt(2.0 * pixels / K)), max(1, min(rows, cols) // 4)))   # never below ~K/2 pooled pixels
        else:
            block = int(pool)
    detail = dict(full=d_full, pool_block=block, max_rank=max_rank)
    rank, source = rank_full, "full resolution"
    if block > 1:
        Tp = torch.from_numpy(np.ascontiguousarray(pool_pixels(T_np, spatial_shape, block), dtype=np.float32)).to(device)
        rank_pool, d_pool = _lrt_rank(Tp, device, seed, max_rank, f"pooled {block}x{block}")
        detail["pooled"] = d_pool
        if rank_pool > rank_full:
            rank, source = rank_pool, f"pooled {block}x{block}"
    if rank == max_rank:
        log.warning("every rank up to max_rank %d was accepted; the search may be capped, raise max_rank", max_rank)
    parts = [f"full resolution gave {rank_full}"]
    if block > 1:
        parts.append(f"pooled {block}x{block} ({detail['pooled']['pixels']:,} pixels) gave {rank_pool}")
    note = f"rank {rank} estimated by likelihood-ratio tests ({'; '.join(parts)}); give the rank to override"
    detail.update(gains=(detail.get("pooled") or d_full)["gains"], effective_dose=d_full["effective_dose"], threshold=(detail.get("pooled") or d_full)["threshold"])
    return rank, note, detail
