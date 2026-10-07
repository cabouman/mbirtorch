"""Dehydration and rehydration of hyperspectral neutron data by the maximum-likelihood factorization X = W H."""
import warnings

import numpy as np

from ..utilities import _to_host

# The keywords of MBIRJAX's scikit-learn dehydrate that this package does not take.
_L2_KEYWORDS = ("safety_factor", "beta_loss", "max_iter", "tolerance", "batch_size", "random_state")


def _reject_unknown_keywords(name, kwargs):
    if not kwargs:
        return
    legacy = sorted(set(kwargs) & set(_L2_KEYWORDS))
    if legacy:
        raise TypeError(f"{name}() fits the Poisson likelihood and does not take {', '.join(legacy)}, which belong to "
                        "MBIRJAX's scikit-learn NMF dehydrate; MBIRTorch does not include that method")
    raise TypeError(f"{name}() got unexpected keyword argument(s) {', '.join(sorted(kwargs))}")


def _spatial_shape(lead):
    """(views, rows, cols) for leading axes of that form or (rows, cols); None for anything else (no pooling)."""
    return tuple(lead) if len(lead) == 3 else (1,) + tuple(lead) if len(lead) == 2 else None


def _clean_transmission(T):
    """Set every non-finite entry of a float32 transmission to zero (a zero count, which the likelihood handles) and
    clip the negative ones at zero, in place. Returns the shares of the entries changed, 'nonfinite_frac' and
    'negative_frac', for those that occur; the loader and the functions here share it, so the files and the arrays
    are treated alike."""
    info = {}
    bad = ~np.isfinite(T)
    if bad.any():
        info["nonfinite_frac"] = float(bad.mean())
        T[bad] = 0.0
    negative = T < 0
    if negative.any():
        info["negative_frac"] = float(negative.mean())
        T[negative] = 0.0
    return info


def _warn_cleaning(info):
    if info.get("nonfinite_frac"):
        warnings.warn(f"{100 * info['nonfinite_frac']:.3g}% of the entries are NaN or have an infinite transmission; "
                      "treated as zero counts")
    if info.get("negative_frac"):
        warnings.warn(f"{100 * info['negative_frac']:.3g}% of the entries have a negative transmission; clipped at "
                      "zero")


def _to_transmission(data, dataset_type):
    """(pixels, bins) float32 transmission ratio from an array with any leading axes, and the input shape.

    Every non-finite transmission becomes zero and negative transmissions are clipped at zero (_clean_transmission).
    """
    if dataset_type not in ("attenuation", "transmission"):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")
    a = _to_host(data)
    if a.ndim < 2:
        raise ValueError(f"data must have at least one leading axis and a spectral last axis; got shape {a.shape}")
    shape = a.shape
    a = a.reshape(-1, shape[-1]).astype(np.float32, copy=False)
    if dataset_type == "attenuation":
        with np.errstate(over="ignore", invalid="ignore"):
            T = np.exp(-a)
    else:
        T = a.copy()
    _warn_cleaning(_clean_transmission(T))
    if not bool((T > 0).any()):
        raise ValueError("the data hold no counts: the transmission is zero everywhere")
    return np.ascontiguousarray(T, dtype=np.float32), shape


def dehydrate(data, dataset_type="attenuation", num_materials=None, *, subspace_basis=None, spectra="mle", dose=None,
              penalty="auto", max_steps=1000, rel_tol=1e-8, max_rank=6, device=None,
              compile_mode="auto", mode="auto", chunk_pixels=None, max_passes=5, verbose=1, **kwargs):
    """Dehydrate a hyperspectral dataset: factor its attenuation as X = W H, W, H >= 0, by maximum likelihood.

    The spectral axis is last; the leading axes are kept. The fit maximizes the Poisson likelihood of the counts,
    written in terms of the transmission T = exp(-X), so zero counts are handled directly. When num_materials is
    not given, the rank is estimated by likelihood-ratio tests (see estimate_rank). The components are a
    nonnegative basis of the data, not necessarily the pure materials. Data too large for the device are fitted
    by chunks of pixels. Given a subspace_basis, only the maps W are fitted.

    Args:
        data (numpy.ndarray or torch.Tensor): Hyperspectral data, spectral axis of length :math:`N_k` last. NaN or
            infinite transmissions count as zero counts; negative transmissions are clipped at zero.
        dataset_type (str, optional): 'attenuation' or 'transmission', with attenuation = -log(transmission).
            Defaults to 'attenuation'.
        num_materials (int, optional): Rank :math:`N_m` of the factorization. Defaults to None, which estimates it.
        subspace_basis (numpy.ndarray or torch.Tensor, optional): Spectra of shape :math:`(N_m, N_k)` to hold
            fixed; only the maps are then fitted. Defaults to None.
        spectra (str, optional): 'mle', the maximum-likelihood spectra; 'unconstrained', a re-estimate that removes
            the bias of the nonnegativity constraint at low dose; 'support', which selects the components present in
            each pixel and refits (needs dose). Defaults to 'mle'.
        dose (float, optional): Open-beam counts per pixel and bin, for spectra='support'. Defaults to None.
        penalty (str or float, optional): Charge per component for spectra='support', as a multiple of log(N_k), or
            'auto'. Defaults to 'auto'.
        max_steps (int, optional): Solver iteration cap. Defaults to 1000.
        rel_tol (float, optional): Stop when the relative loss change stays below this. Defaults to 1e-8.
        max_rank (int, optional): Largest rank the estimate considers. Defaults to 6.
        device (str, optional): Torch device. Defaults to None: CUDA if available, else CPU.
        compile_mode (str, optional): 'auto', 'on' or 'off' for torch.compile. Defaults to 'auto'.
        mode (str, optional): 'full' fits on the device at once, 'stream' by chunks of pixels, 'auto' picks from
            the available memory. Defaults to 'auto'.
        chunk_pixels (int, optional): Pixels per chunk when streamed. Defaults to None, from the available memory.
        max_passes (int, optional): Passes over the chunks when streamed. Defaults to 5.
        verbose (int, optional): 0 prints nothing; 1 a summary; 2 also the rank search. Defaults to 1.

    Returns:
        list: [subspace_data, subspace_basis, dataset_type]: W reshaped to the leading axes plus :math:`N_m`, H of
        shape :math:`(N_m, N_k)`, both float32, and the input's dataset_type, so that rehydrate returns the same
        quantity as the input.
    """
    from ._device import _default_device
    from ._fit import _fit
    from .rank import _estimate_rank
    _reject_unknown_keywords("dehydrate", kwargs)
    if mode not in ("auto", "full", "stream"):
        raise ValueError(f"mode must be 'auto', 'full' or 'stream', got {mode!r}")
    if chunk_pixels is not None and int(chunk_pixels) < 1:
        raise ValueError(f"chunk_pixels must be a positive number of pixels, got {chunk_pixels}")
    if int(max_passes) < 0:
        raise ValueError(f"max_passes must be at least 0, got {max_passes}")
    T, shape = _to_transmission(data, dataset_type)
    device = _default_device(device)
    lead = shape[:-1]
    if subspace_basis is not None:
        from ._fit import _fit_fixed_basis
        H = _check_basis(subspace_basis, shape[-1], num_materials, spectra)
        W, H, rep = _fit_fixed_basis(T, H, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps,
                                     rel_tol=rel_tol, compile_mode=compile_mode)
        if verbose >= 1:
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", H.shape[0], "(the given subspace_basis)")
            print("   -Pixels: ", T.shape[0], f"; maps for the given basis ({rep['chunks']} chunk(s))")
        return [W.reshape(*lead, H.shape[0]), H, dataset_type]
    note = f"rank {num_materials} given"
    if num_materials is None:
        num_materials, note, _ = _estimate_rank(T, spatial_shape=_spatial_shape(lead), device=device, max_rank=max_rank,
                                                verbose=max(0, verbose - 1))
    W, H, rep = _fit(T, int(num_materials), spectra=spectra, dose=dose, penalty=penalty, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps, rel_tol=rel_tol,
                     max_passes=int(max_passes), compile_mode=compile_mode)
    subspace_data = W.reshape(*lead, int(num_materials))
    if verbose >= 1:
        solve = f"{rep['steps']} steps" if "steps" in rep else f"streamed, {rep['passes']} polish passes"
        print("dehydrate(): ")
        print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
        print("   -Pixels: ", T.shape[0], f"; spectra: {spectra} ({solve})")
    return [subspace_data, H, dataset_type]


def _check_basis(subspace_basis, bins, num_materials, spectra):
    """A given subspace_basis as float32 (rank, bins), checked against the data and the other arguments."""
    H = np.asarray(_to_host(subspace_basis), dtype=np.float32)
    if H.ndim != 2 or H.shape[1] != bins:
        raise ValueError(f"subspace_basis must have shape (rank, {bins}) for data with {bins} bins; got {H.shape}")
    if not np.isfinite(H).all() or (H < 0).any():
        raise ValueError("subspace_basis must be finite and nonnegative")
    if num_materials is not None and int(num_materials) != H.shape[0]:
        raise ValueError(f"num_materials={num_materials} disagrees with the {H.shape[0]} rows of subspace_basis")
    if spectra != "mle":
        raise ValueError(f"spectra={spectra!r} re-estimates the spectra, which a given subspace_basis holds fixed")
    return H


def hyper_denoise(data, dataset_type="attenuation", num_materials=None, **kwargs):
    """Denoise a hyperspectral dataset: dehydrate, then rehydrate.

    Args:
        data (numpy.ndarray or torch.Tensor): Hyperspectral data with the spectral axis last.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        num_materials (int, optional): Rank of the factorization. Defaults to None, which estimates it.
        **kwargs: The keyword arguments of :func:`~mbirtorch.hsnt.dehydrate`.

    Returns:
        numpy.ndarray: The denoised data, float32, with the shape and dataset_type of the input.
    """
    _reject_unknown_keywords("hyper_denoise", {k: v for k, v in kwargs.items() if k in _L2_KEYWORDS})
    return rehydrate(dehydrate(data, dataset_type=dataset_type, num_materials=num_materials, **kwargs))


def rehydrate(dehydrated_data, hyperspectral_idx=None):
    """Rehydrate dehydrated hyperspectral data: multiply the maps by the spectra, for all or selected bins.

    The method is described in M. S. N. Chowdhury et al., "Fast Hyperspectral Neutron Tomography," IEEE Trans.
    Computational Imaging, vol. 11, pp. 663-677, 2025, doi:10.1109/TCI.2025.3567854.

    Args:
        dehydrated_data: [subspace_data, subspace_basis, dataset_type]: the maps, with the component axis of length
            :math:`N_s` last; the spectra, shape :math:`(N_s, N_k)`; and 'attenuation' or 'transmission'.
        hyperspectral_idx: Indices of the spectral bins to rehydrate. Defaults to None, all :math:`N_k` bins.

    Returns:
        numpy.ndarray: Hyperspectral data with the shape of subspace_data, except that the last axis holds the
        selected bins; attenuation or transmission as dataset_type says.
    """
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data

    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)

    return rehydrated_data
