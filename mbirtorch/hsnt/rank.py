"""Rank (number of components) estimation by sequential likelihood-ratio tests on the NNAL factorization."""
import math
import warnings

import numpy as np
import torch

from ._device import _default_device


def _lrt_rank(T, max_rank, label, verbose=0):
    """Sequential likelihood-ratio rank test on the pixels of T, a tensor on the device. Returns (rank, detail)."""
    from .factorization import _nnal_factorization
    from ._loss import stable_nnal
    P, K = T.shape
    losses, resid = [], []
    for r in range(1, max_rank + 1):
        # uncompiled: the search solves small problems at six ranks, and each new shape would recompile
        W, H, _ = _nnal_factorization(T, r, max_steps=200, rel_tol=1e-6, compile_mode="off")
        Xd = W.double() @ H.double()
        Th = torch.exp(-Xd)
        Td = T.double()
        losses.append(stable_nnal(Xd, Td).item())
        resid.append((((Td - Th) ** 2) / Th.clamp_min(1e-12)).mean().item())
    dose_eff = 1.0 / max(resid[-1], np.finfo(np.float64).tiny)
    gains = [dose_eff * (losses[i - 1] - losses[i]) for i in range(1, len(losses))]     # gains[i - 1]: component i + 1
    edge = 0.5 * (math.sqrt(P) + math.sqrt(K)) ** 2          # the gain of a component fitted to noise alone
    quiet = [g for g in gains[-3:] if g < 2.0 * edge]
    floor = max(0.5 * (P + K), float(np.median(quiet))) if quiet else edge
    threshold = 2.0 * floor
    rank = 1
    for r, g in zip(range(2, max_rank + 1), gains):
        if g > threshold:
            rank = r
        else:
            break
    if verbose >= 1:
        table = ", ".join(f"{r}: {g:,.0f}" for r, g in zip(range(2, max_rank + 1), gains))
        print(f"rank search ({label}, {P:,} pixels x {K} bins): effective dose {dose_eff:.3g}; log-likelihood gain of "
              f"component {table}; noise floor {floor:.0f}, threshold {threshold:.0f} -> rank {rank}")
    return rank, dict(pixels=P, losses=losses, gains=gains, effective_dose=dose_eff, noise_floor=floor,
                      threshold=threshold, noise_tail=bool(quiet) or max_rank == 1)


def _pool_pixels(T, spatial_shape, block):
    """Block-average a (pixels, bins) array over block x block detector pixels within each view.

    Rows and columns are cropped to multiples of the block. The averaged transmission is the summed count over the
    block divided by the block's summed dose, so it is a valid transmission ratio at block^2 times the dose.

    Args:
        T (numpy.ndarray): Transmission ratio, shape (pixels, bins), pixels in (views, rows, cols) order.
        spatial_shape (tuple): (views, rows, cols).
        block (int): Block size in pixels along rows and columns.

    Returns:
        numpy.ndarray: The pooled transmission, shape (views * (rows // block) * (cols // block), bins).
    """
    V, rows, cols = spatial_shape
    r, c = rows // block * block, cols // block * block
    X = np.asarray(T).reshape(V, rows, cols, -1)[:, :r, :c]
    X = X.reshape(V, r // block, block, c // block, block, -1).mean(axis=(2, 4))
    return X.reshape(-1, X.shape[-1])


def _subsample(T, n):
    """At most n rows of T, a seeded random subset in their original order, as contiguous float32."""
    rows = np.sort(np.random.default_rng(0).choice(T.shape[0], n, replace=False)) if T.shape[0] > n else slice(None)
    return np.ascontiguousarray(T[rows], dtype=np.float32)


def estimate_rank(data, dataset_type="attenuation", max_rank=6, device=None, pool="auto", verbose=0):
    """Estimate the number of components of a hyperspectral dataset, the rank dehydrate uses when num_materials is
    not given, by sequential likelihood-ratio tests at full resolution and on spatially pooled pixels.

    The data are taken as :func:`~mbirtorch.hsnt.dehydrate` takes them: the spectral axis last and any leading axes;
    when those are (views, rows, cols) or (rows, cols), blocks of neighboring pixels are also pooled.

    Ranks 1 to max_rank are fitted in turn. The loss gain of each added component is converted to log-likelihood
    units with a dose calibrated from the residual of the most flexible fit (the mean of (T - e^-X)^2 / e^-X is
    1 / dose for Poisson noise), so the nominal dose does not matter. A component that fits only noise gains at most
    about 0.5 (sqrt(P) + sqrt(K))^2, the top of the noise's singular spectrum. The noise floor is the median gain of
    those of the last three ranks that stay below twice that, and at least (P + K) / 2; a component is accepted
    while its gain exceeds twice the floor. When none of the last three gains is that small, the search is capped
    and a warning says to raise max_rank.

    That floor grows with the pixel count as fast as a faint material's evidence, so at low dose the full-resolution
    test misses weak materials. Pooling blocks of neighboring pixels keeps the evidence (summed counts stay Poisson)
    but divides the nuisance count, so the same test on pooled pixels has more power. The larger of the two ranks is
    returned: over-estimating the rank costs little, under-estimating it caps the fit.

    Args:
        data (numpy.ndarray or torch.Tensor): Hyperspectral data with any leading axes and the spectral axis last.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        max_rank (int, optional): Largest rank considered. Defaults to 6.
        device (str, optional): Torch device for the solves. Defaults to None, meaning CUDA if available, else CPU.
        pool (str or int, optional): Pooling block size. 'auto' (default) chooses the block from the calibrated dose so
            that pooled pixels hold about 64 counts per bin, with blocks of at most ceil(sqrt(2 P / K)) pixels a side
            (so about K / 2 pooled pixels, somewhat fewer after the rounding) and no pooling above 64 counts per bin
            (pooled mixed pixels are not exactly low rank and would add spurious rank at high dose). An integer fixes
            the block; 0 disables pooling.
        verbose (int, optional): 1 prints each search's gains and decision. Defaults to 0.

    Returns:
        (rank, note, detail): the rank, a one-line account of how it was chosen, and a dict with the searches'
        numbers ('full', and 'pooled' when pooling ran, each with gains, effective_dose and threshold; 'pool_block';
        'rank_full'; 'rank_pooled').
    """
    from .denoise import _spatial_shape, _to_transmission
    T, shape = _to_transmission(data, dataset_type)
    return _estimate_rank(T, _spatial_shape(shape[:-1]), device, max_rank, pool=pool, verbose=verbose)


def _estimate_rank(T, spatial_shape=None, device=None, max_rank=6, subsample=16384, pool="auto", verbose=0):
    """estimate_rank on a (pixels, bins) transmission ratio whose pixels are in (views, rows, cols) order when
    spatial_shape is given; subsample caps the pixels each test uses (a seeded random subset)."""
    device = _default_device(device)
    T_np = T.detach().cpu().numpy() if torch.is_tensor(T) else np.asarray(T)
    if not bool((T_np > 0).any()):
        raise ValueError("the data hold no counts: the transmission is zero everywhere")
    pixels, K = T_np.shape
    Tt = torch.from_numpy(_subsample(T_np, subsample)).to(device)
    rank_full, d_full = _lrt_rank(Tt, max_rank, "full resolution", verbose)
    block = 0
    if spatial_shape is not None and pool:
        rows, cols = spatial_shape[1], spatial_shape[2]
        if pool == "auto":
            block = int(np.ceil(np.sqrt(64.0 / max(d_full["effective_dose"], 1e-9))))
            block = int(min(block, np.ceil(np.sqrt(2.0 * pixels / K)), max(1, min(rows, cols) // 4)))
        else:
            block = int(pool)
    detail = dict(full=d_full, pool_block=block, max_rank=max_rank, rank_full=rank_full, rank_pooled=None)
    rank = rank_full
    parts = [f"full resolution gave {rank_full}"]
    if block > 1:
        Tp = torch.from_numpy(_subsample(_pool_pixels(T_np, spatial_shape, block), subsample)).to(device)
        rank_pool, d_pool = _lrt_rank(Tp, max_rank, f"pooled {block}x{block}", verbose)
        detail.update(pooled=d_pool, rank_pooled=rank_pool)
        rank = max(rank_full, rank_pool)
        parts.append(f"pooled {block}x{block} ({d_pool['pixels']:,} pixels) gave {rank_pool}")
    decisive = detail.get("pooled") or d_full
    if rank == max_rank or not decisive["noise_tail"]:
        warnings.warn(f"the largest ranks tried still gain more than noise does; the rank may exceed max_rank "
                      f"{max_rank}, raise max_rank")
    note = f"rank {rank} estimated by likelihood-ratio tests ({'; '.join(parts)})"
    detail.update(gains=decisive["gains"], effective_dose=d_full["effective_dose"], threshold=decisive["threshold"])
    return rank, note, detail
