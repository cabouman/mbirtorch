"""Dehydration and rehydration of hyperspectral neutron data with the maximum-likelihood NNAL factorization.

``dehydrate`` fits the factorization X = W H of the attenuation and returns the package's dehydrated form
``[subspace_data, subspace_basis, dataset_type]``; ``rehydrate`` multiplies it back and ``hyper_denoise`` chains the
two.
"""
import warnings

import numpy as np

from ..utilities import _to_host

# The keywords of MBIRJAX's scikit-learn dehydrate, which this package does not include.
_L2_KEYWORDS = ("safety_factor", "beta_loss", "max_iter", "tolerance", "batch_size", "subspace_basis", "random_state")


def _reject_unknown_keywords(name, kwargs):
    if not kwargs:
        return
    legacy = sorted(set(kwargs) & set(_L2_KEYWORDS))
    if legacy:
        raise TypeError(f"{name}() fits the Poisson likelihood and does not take {', '.join(legacy)}, which belong to "
                        "MBIRJAX's scikit-learn NMF dehydrate; MBIRTorch does not include that method")
    raise TypeError(f"{name}() got unexpected keyword argument(s) {', '.join(sorted(kwargs))}")


def _to_transmission(data, dataset_type):
    """(pixels, bins) float32 transmission ratio from an array with any leading axes, and the input shape.

    Every non-finite value becomes zero transmission (a zero count, which the likelihood handles) and negative
    transmissions are clipped at zero.
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
    bad = ~np.isfinite(T)
    if bad.any():
        warnings.warn(f"{100 * bad.mean():.3g}% of the data are NaN or infinite; treated as zero counts")
        T[bad] = 0.0
    np.maximum(T, 0.0, out=T)
    return np.ascontiguousarray(T, dtype=np.float32), shape


def dehydrate(data, dataset_type="attenuation", num_materials=None, *, spectra="mle", dose=None, penalty="auto",
              free_refit=False, max_steps=1000, rel_tol=1e-8, max_rank=6, device=None, compile_mode="auto", verbose=1,
              **kwargs):
    """Dehydrate a hyperspectral dataset by the maximum-likelihood factorization X = W H of its attenuation.

    The spectral axis must be the last axis; the leading axes are kept. The fit minimizes the non-negative
    attenuation loss sum[exp(-X) + T X] of the transmission T = exp(-attenuation), with W >= 0 and H >= 0. It is the
    Poisson log-likelihood of the counts, up to a constant, when the open beam is the same in every pixel and bin. The
    rank is the number of components; when it is not given it is estimated by likelihood-ratio tests
    (:func:`~mbirtorch.hsnt.estimate_rank`), which also pool pixels spatially when the leading axes are
    (views, rows, cols) or (rows, cols). The components are a nonnegative basis of the data, not necessarily the
    pure materials. Data that do not fit the device are factorized by chunks of pixels.

    Args:
        data (numpy.ndarray or torch.Tensor): Hyperspectral data with any leading axes and the spectral axis of
            length :math:`N_k` last. NaN and infinite values are treated as zero counts, with a warning.
        dataset_type (str, optional): 'attenuation' or 'transmission', where attenuation = -log(transmission).
            Defaults to 'attenuation'.
        num_materials (int, optional): Rank of the factorization :math:`N_m`. Defaults to None, which estimates it.
        spectra (str, optional): 'mle', the maximum-likelihood spectra; 'unconstrained', a re-estimate without the
            bias the nonnegativity of W gives the spectra at low dose, which pays from about 10^5 pixels; 'support',
            which decides the components present in each pixel and refits, and needs the dose: it corrects the same
            bias, and in the maps mostly zeroes the background, since a pixel of one material usually needs several
            of the fitted components. Defaults to 'mle'.
        dose (float, optional): Open-beam counts per pixel and bin, for spectra='support'. Defaults to None.
        penalty (str or float, optional): Support selection's charge per component, as a multiple of log(N_k), or
            'auto', which moves from 0.5 to 2 with the counts per pixel and bin. Defaults to 'auto'.
        free_refit (bool, optional): With spectra='support', leave the selected coefficients free of sign during the
            refit, then re-solve W >= 0 on the supports. Defaults to False.
        max_steps (int, optional): Solver iteration cap. Defaults to 1000.
        rel_tol (float, optional): The solver stops after five consecutive steps whose relative loss change is at most
            this. Defaults to 1e-8.
        max_rank (int, optional): Largest rank the estimate considers. Defaults to 6.
        device (str, optional): Torch device. Defaults to None, meaning CUDA if available, else CPU.
        compile_mode (str, optional): 'auto' compiles the solver with torch.compile on CUDA for data of at least 5e8
            entries, where it pays; 'on' always; 'off' never. The rank estimate always runs uncompiled. Defaults to
            'auto'.
        verbose (int, optional): 0 prints nothing; 1 prints a summary; 2 also prints the rank search. Defaults to 1.

    Returns:
        list: [subspace_data, subspace_basis, dataset_type], where subspace_data is W reshaped to the leading axes
        plus :math:`N_m` (float32), subspace_basis is H of shape :math:`(N_m, N_k)` (float32), and dataset_type is
        the input's, so that :func:`~mbirtorch.hsnt.rehydrate` returns the same quantity as the input.

    Example:
        >>> [subspace_data, subspace_basis, dataset_type] = dehydrate(data, num_materials=3)
        >>> data.shape, subspace_data.shape, subspace_basis.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., 3), (3, N_k))
    """
    from ._device import _default_device
    from ._fit import _fit
    from .rank import estimate_rank
    _reject_unknown_keywords("dehydrate", kwargs)
    T, shape = _to_transmission(data, dataset_type)
    device = _default_device(device)
    lead = shape[:-1]
    spatial = (tuple(lead) if len(lead) == 3 else (1,) + tuple(lead) if len(lead) == 2 else None)
    note = f"rank {num_materials} given"
    if num_materials is None:
        num_materials, note, _ = estimate_rank(T, spatial_shape=spatial, device=device, max_rank=max_rank,
                                               verbose=max(0, verbose - 1))
    W, H, rep = _fit(T, int(num_materials), spectra=spectra, dose=dose, penalty=penalty, free_refit=free_refit,
                     device=device, max_steps=max_steps, rel_tol=rel_tol, compile_mode=compile_mode)
    subspace_data = W.reshape(*lead, int(num_materials))
    if verbose >= 1:
        solve = f"{rep['steps']} steps" if "steps" in rep else f"streamed, {rep['passes']} polish passes"
        print("dehydrate(): ")
        print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
        print("   -Pixels: ", T.shape[0], f"; spectra: {spectra} ({solve})")
    return [subspace_data, H, dataset_type]


def hyper_denoise(data, dataset_type="attenuation", num_materials=None, **kwargs):
    """Denoise a hyperspectral dataset by dehydration and rehydration.

    Args:
        data (numpy.ndarray or torch.Tensor): Hyperspectral data with the spectral axis last.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        num_materials (int, optional): Rank of the factorization. Defaults to None, which estimates it.
        **kwargs: The other arguments of :func:`~mbirtorch.hsnt.dehydrate`.

    Returns:
        numpy.ndarray: The rank-:math:`N_m` fit, with the shape of the input and the input's dataset_type, float32.
    """
    _reject_unknown_keywords("hyper_denoise", {k: v for k, v in kwargs.items() if k in _L2_KEYWORDS})
    return rehydrate(dehydrate(data, dataset_type=dataset_type, num_materials=num_materials, **kwargs))


def rehydrate(dehydrated_data, hyperspectral_idx=None):
    """
    Rehydrate/decompress selected spectral bins from dehydrated hyperspectral data as described in:

    M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, "Fast Hyperspectral Neutron Tomography," IEEE Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854

    Args:
        dehydrated_data: Dehydrated hyperspectral data in the form [subspace_data, subspace_basis, dataset_type]:

            - subspace_data: ndarray with arbitrary axes and a subspace axis of length :math:`N_s` in the last position.
            - subspace_basis: ndarray of shape :math:`(N_s, N_k)`, where rows are subspace basis spectra.
            - dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission).
        hyperspectral_idx: A list of :math:`N_h` indices along the original spectral axis to rehydrate. If None, all :math:`N_k`
            spectral bins are rehydrated. Defaults to None.

    Returns:
        Rehydrated/decompressed hyperspectral data with the same shape as the input subspace_data except the last axis
        length is :math:`N_h (N_h <= N_k)`.

    Example:
        >>> hyper_data = rehydrate([subspace_data, subspace_basis, dataset_type], hyperspectral_idx=[5, 10, 15])
        >>> subspace_data.shape, subspace_basis.shape, hyper_data.shape
        ((N_x, N_y, N_z, ..., N_s), (N_s, N_k), (N_x, N_y, N_z, ..., 3))
    """
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data

    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)

    return rehydrated_data
