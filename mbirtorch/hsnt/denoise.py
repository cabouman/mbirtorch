"""This module compresses and restores hyperspectral neutron data with a factorization X = W H.
The function ``dehydrate`` computes the factors, and the function ``rehydrate`` multiplies them."""
import warnings

import numpy as np

from ..utilities import _to_host

# These keywords belong to the MBIRJAX version of dehydrate, which used scikit-learn.  mbirtorch rejects them.
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
    """This function returns the leading axes as (views, rows, cols).  It sets views to 1 when
    there are two leading axes, and it returns None when there are not two or three."""
    return tuple(lead) if len(lead) == 3 else (1,) + tuple(lead) if len(lead) == 2 else None


def _clean_transmission(T):
    """This function sets each NaN, infinite, or negative entry of a float32 transmission array to
    zero, in place.  It returns a dict with the fraction of entries that it changed."""
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
    """This function converts the data to a float32 transmission array of shape (pixels, bins).
    It cleans the transmission array with ``_clean_transmission``.  It returns the transmission
    array and the shape of the input."""
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


def dehydrate(data, dataset_type="transmission", num_materials=None, *, subspace_basis=None,
              max_steps=1000, rel_tol=1e-8, max_rank=6, device=None,
              compile_mode="auto", mode="auto", chunk_pixels=None, max_passes=5, sample_mask=None,
              verbose=1, **kwargs):
    """This function dehydrates hyperspectral data.  It computes a non-negative factorization
    of the data in the attenuation domain.  The spectral axis of the data is always the last axis.

    For transmission data :math:`T`, this function computes the non-negative factorization
    :math:`T = \\exp(-W H)`, where ``W`` has one row per pixel and :math:`N_m` columns, and ``H``
    is :math:`N_m \\times N_k`.  The rows of ``H`` are the spectral basis, and the columns of
    ``W`` are the maps.  The function ``rehydrate`` turns the result back into data.

    The algorithm fits ``W`` and ``H`` by maximum likelihood.  It assumes that :math:`T` comes
    from Poisson counts, so zero counts are handled correctly.

    The input may be either 'attenuation' or 'transmission'.  The output ``W`` and ``H`` always
    describe attenuation.

    Args:
        data (numpy.ndarray or torch.Tensor): The data, with the :math:`N_k` spectral bins on
            the last axis and the pixels on the other axes, for example (views, rows, cols).
            A NaN or infinite transmission counts as zero counts.  A negative transmission is
            clipped to zero.
        dataset_type (str, optional): 'attenuation' or 'transmission'.  Defaults to
            'transmission'.
        num_materials (int, optional): The number of spectra :math:`N_m`.  Defaults to None,
            which estimates it from the data.
        subspace_basis (numpy.ndarray or torch.Tensor, optional): Fixed spectra ``H`` of shape
            :math:`(N_m, N_k)`.  When given, only the maps ``W`` are computed.  Defaults to None.
        max_steps (int, optional): The maximum number of solver iterations.  Defaults to 1000.
        rel_tol (float, optional): The solver stops when the relative change in the loss falls
            below this value.  Defaults to 1e-8.
        max_rank (int, optional): The largest :math:`N_m` that the estimate of
            ``num_materials`` tries.  Defaults to 6.
        device (str, optional): The torch device.  Defaults to None, which uses CUDA when
            available and the CPU otherwise.
        compile_mode (str, optional): 'auto', 'on', or 'off' for torch.compile.  Defaults to
            'auto'.
        mode (str, optional): 'full' holds all pixels in device memory.  'stream' processes
            the pixels in chunks.  'auto' chooses by the available memory.  Defaults to 'auto'.
        chunk_pixels (int, optional): The pixels per chunk in 'stream' mode.  Defaults to
            None, which sets it from the available memory.
        max_passes (int, optional): The passes over the chunks in 'stream' mode.  Defaults
            to 5.
        sample_mask (numpy.ndarray or torch.Tensor, optional): A boolean array, True where the
            sample may be.  Its shape is (rows, cols) for one region shared by all views, or
            (views, rows, cols) for a region per view.  ``W`` is set to zero outside the mask
            and fit only inside it.  Defaults to None.
        verbose (int, optional): 0 prints nothing.  1 prints a summary.  2 also prints the
            search for ``num_materials``.  Defaults to 1.

    Returns:
        list: The list [subspace_data, subspace_basis].  subspace_data is the maps ``W``, with
            the pixel axes of the data followed by an axis of length :math:`N_m`.
            subspace_basis is the spectra ``H``, with shape :math:`(N_m, N_k)`.  Both are
            float32, and ``W H`` is the attenuation at each pixel and bin.
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
    T_all, shape = _to_transmission(data, dataset_type)
    device = _default_device(device)
    lead = shape[:-1]
    mask = _sample_mask(sample_mask, lead)
    T = T_all if mask is None else np.ascontiguousarray(T_all[mask])
    pixels = f"{T.shape[0]}" if mask is None else f"{T.shape[0]} of {T_all.shape[0]} (sample_mask)"
    if subspace_basis is not None:
        from ._fit import _fit_fixed_basis
        H = _check_basis(subspace_basis, shape[-1], num_materials)
        W, H, rep = _fit_fixed_basis(T, H, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps,
                                     rel_tol=rel_tol, compile_mode=compile_mode)
        if verbose >= 1:
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", H.shape[0], "(the given subspace_basis)")
            print("   -Pixels: ", pixels, f"; maps for the given basis ({rep['chunks']} chunk(s))")
    else:
        note = f"rank {num_materials} given"
        if num_materials is None:
            num_materials, note, _ = _estimate_rank(T_all, spatial_shape=_spatial_shape(lead), device=device,
                                                    max_rank=max_rank, verbose=max(0, verbose - 1))
        W, H, rep = _fit(T, int(num_materials), device=device, mode=mode,
                         chunk_pixels=chunk_pixels, max_steps=max_steps, rel_tol=rel_tol, max_passes=int(max_passes),
                         compile_mode=compile_mode)
        if verbose >= 1:
            solve = f"{rep['steps']} steps" if "steps" in rep else f"streamed, {rep['passes']} polish passes"
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
            print("   -Pixels: ", pixels, f" ({solve})")
    if mask is not None:
        W_all = np.zeros((T_all.shape[0], W.shape[1]), dtype=np.float32)
        W_all[mask] = W
        W = W_all
    return [W.reshape(*lead, H.shape[0]), H]


def _sample_mask(sample_mask, lead):
    """This function converts ``sample_mask`` to a flat boolean array over the pixels, in the C
    order of the leading axes.  It returns None when ``sample_mask`` is None."""
    if sample_mask is None:
        return None
    m = np.asarray(_to_host(sample_mask))
    if m.dtype != bool:
        if not np.isin(m, (0, 1)).all():
            raise ValueError("sample_mask must be boolean (or hold only 0 and 1)")
        m = m.astype(bool)
    try:
        m = np.broadcast_to(m, tuple(lead))
    except ValueError:
        raise ValueError(f"sample_mask of shape {m.shape} does not broadcast to the leading axes "
                         f"{tuple(lead)}") from None
    if not m.any():
        raise ValueError("sample_mask is False everywhere: there are no pixels to fit")
    return np.ascontiguousarray(m).reshape(-1)


def _check_basis(subspace_basis, bins, num_materials):
    """This function checks ``subspace_basis`` against the data and returns it as a float32 array
    of shape (rank, bins)."""
    H = np.asarray(_to_host(subspace_basis), dtype=np.float32)
    if H.ndim != 2 or H.shape[1] != bins:
        raise ValueError(f"subspace_basis must have shape (rank, {bins}) for data with {bins} bins; got {H.shape}")
    if not np.isfinite(H).all() or (H < 0).any():
        raise ValueError("subspace_basis must be finite and nonnegative")
    if num_materials is not None and int(num_materials) != H.shape[0]:
        raise ValueError(f"num_materials={num_materials} disagrees with the {H.shape[0]} rows of subspace_basis")
    return H


def hyper_denoise(data, dataset_type="attenuation", num_materials=None, **kwargs):
    """This function denoises hyperspectral data.  It calls ``dehydrate`` and then ``rehydrate``.
    The spectral axis of the data is always the last axis.

    Args:
        data (numpy.ndarray or torch.Tensor): The data.  The last axis is the spectral axis.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        num_materials (int, optional): The rank.  Defaults to None, in which case
            ``dehydrate`` estimates the rank.
        **kwargs: The keyword arguments of :func:`~mbirtorch.hsnt.dehydrate`.

    Returns:
        numpy.ndarray: The denoised float32 data, with the shape and dataset_type of the input.
    """
    _reject_unknown_keywords("hyper_denoise", {k: v for k, v in kwargs.items() if k in _L2_KEYWORDS})
    return rehydrate(dehydrate(data, dataset_type=dataset_type, num_materials=num_materials, **kwargs),
                     dataset_type=dataset_type)


def rehydrate(dehydrated_data, dataset_type="attenuation", hyperspectral_idx=None):
    """This function restores hyperspectral data from the output of ``dehydrate``.  It multiplies
    the maps by the spectra, for all bins or for selected bins.  The spectral axis of the
    returned data is always the last axis.  The method is described in
    M. S. N. Chowdhury et al., "Fast Hyperspectral Neutron Tomography," IEEE TCI, vol. 11,
    pp. 663-677, 2025, doi:10.1109/TCI.2025.3567854.

    Args:
        dehydrated_data: The list [subspace_data, subspace_basis].  subspace_data holds the
            maps ``W``, with the component axis last.  subspace_basis holds the spectra ``H``,
            with shape :math:`(N_m, N_k)`.
        dataset_type (str, optional): The quantity to return.  'attenuation' returns ``W H``.
            'transmission' returns ``exp(-W H)``.  Defaults to 'attenuation'.
        hyperspectral_idx: The indices of the bins to restore.  Defaults to None, which
            restores all bins.

    Returns:
        numpy.ndarray: The restored data.  Its last axis holds the bins, and its other axes match
            subspace_data.
    """
    if dataset_type not in ("attenuation", "transmission"):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")
    subspace_data, subspace_basis = dehydrated_data[0], dehydrated_data[1]

    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)

    return rehydrated_data
