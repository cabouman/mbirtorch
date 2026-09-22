"""Dehydration and rehydration of hyperspectral neutron data with the NNAL factorization.

``dehydrate`` fits the maximum-likelihood factorization X = W H of the attenuation (Poisson counts through the
non-negative attenuation loss on the transmission ratio) and returns the package's dehydrated form
``[subspace_data, subspace_basis, dataset_type]``: the material maps with the spectral axis replaced by the rank,
the spectra as rows, and the input's type. ``rehydrate`` multiplies them back (for a subset of bins if asked) and
``hyper_denoise`` chains the two. The rank is estimated by likelihood-ratio tests when not given (``rank``).
``export_hsnt_data_hdf5`` / ``import_hsnt_data_hdf5`` write and read the dehydrated form; the command line
(``mbirtorch-hsnt dehydrate | rehydrate | denoise``) wraps these with data loading and checks.

The earlier scikit-learn L2 dehydration is kept as a baseline in ``l2_baseline`` (``l2_dehydrate``,
``l2_hyper_denoise``).
"""
import logging

import numpy as np

log = logging.getLogger("mbirtorch.hsnt")


def _to_transmission(data, dataset_type):
    """(pixels, bins) float32 transmission ratio from an array of arbitrary leading axes; non-finite entries become
    zero transmission (zero counts, which the likelihood handles) and negative transmissions are clipped at zero."""
    if dataset_type not in ("attenuation", "transmission"):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")
    try:
        import torch
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
    except ImportError:                                                               # pragma: no cover
        pass
    a = np.asarray(data)
    if a.ndim < 2:
        raise ValueError(f"data must have at least one leading axis and a spectral last axis; got shape {a.shape}")
    shape = a.shape
    a = a.reshape(-1, shape[-1]).astype(np.float32, copy=False)
    if dataset_type == "attenuation":
        with np.errstate(over="ignore", invalid="ignore"):
            T = np.exp(-a)
    else:
        T = a.copy()
    T = np.nan_to_num(T, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    np.maximum(T, 0.0, out=T)
    return np.ascontiguousarray(T, dtype=np.float32), shape


def dehydrate(data, dataset_type="attenuation", num_materials=None, method="joint_newton", max_steps=300, rel_tol=1e-6,
              max_rank=6, device=None, random_state=0, verbose=1):
    """Dehydrate a hyperspectral dataset: the maximum-likelihood NNAL factorization X = W H of its attenuation.

    The spectral axis must be the last axis; any leading axes are kept. Attenuation is converted to the transmission
    ratio exp(-X) and the factorization minimises the non-negative attenuation loss sum_k [exp(-X_k) + T_k X_k], the
    Poisson log-likelihood of the counts up to a constant, with W >= 0 and H >= 0. Unlike the L2 baseline no
    safety factor inflates the rank: the subspace dimension is the number of materials, estimated by
    likelihood-ratio tests when not given (:func:`~mbirtorch.hsnt.estimate_rank`, which pools pixels spatially when
    the leading axes are (views, rows, cols) or (rows, cols)).

    Args:
        data: Hyperspectral data, numpy or torch, arbitrary leading axes, spectral axis of length N_k last.
        dataset_type: 'attenuation' or 'transmission' where attenuation = -log(transmission). Defaults to 'attenuation'.
        num_materials: Rank of the factorization, the number of materials N_m. None estimates it (see max_rank).
        method: solver, 'joint_newton' (default), 'block_newton', 'multiplicative' or 'lbfgsb'.
        max_steps, rel_tol: the solver's iteration cap and relative loss change per step at which to stop.
        max_rank: largest rank the estimate considers when num_materials is None.
        device: torch device (default: cuda if available).
        random_state: seed of the initialization and of the rank estimate's pixel subsample.
        verbose: 0 prints nothing; 1 prints a summary.

    Returns:
        [subspace_data, subspace_basis, dataset_type]: the maps W reshaped to the leading axes plus N_m (float32),
        the spectra H of shape (N_m, N_k) (float32), and the input's dataset_type, so that
        :func:`~mbirtorch.hsnt.rehydrate` returns the same quantity as the input.

    Example:
        >>> [subspace_data, subspace_basis, dataset_type] = dehydrate(data, num_materials=3)
        >>> data.shape, subspace_data.shape, subspace_basis.shape
        ((N_x, N_y, N_z, ..., N_k), (N_x, N_y, N_z, ..., 3), (3, N_k))
    """
    import torch
    from .factorization import nnal_factorization
    from .rank import estimate_rank
    T, shape = _to_transmission(data, dataset_type)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lead = shape[:-1]
    spatial = (tuple(lead) if len(lead) == 3 else (1,) + tuple(lead) if len(lead) == 2 else None)
    note = f"rank {num_materials} given"
    if num_materials is None:
        num_materials, note, _ = estimate_rank(T, spatial_shape=spatial, device=device, seed=random_state, max_rank=max_rank)
    Tt = torch.from_numpy(T).to(device)
    W, H, steps = nnal_factorization(Tt, method=method, num_materials=int(num_materials), max_steps=max_steps, rel_tol=rel_tol,
                                     random_state=random_state)
    subspace_data = W.cpu().numpy().astype(np.float32).reshape(*lead, int(num_materials))
    subspace_basis = H.cpu().numpy().astype(np.float32)
    if verbose >= 1:
        print("dehydrate(): ")
        print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
        print("   -Pixels: ", T.shape[0], "; solver: ", method, f"({steps} steps)")
    return [subspace_data, subspace_basis, dataset_type]


def hyper_denoise(data, dataset_type="attenuation", num_materials=None, **kwargs):
    """Denoise a hyperspectral dataset by dehydration and rehydration: the rank-N_m maximum-likelihood fit X = W H
    of its attenuation (see :func:`~mbirtorch.hsnt.dehydrate` for the arguments), rehydrated to the input's shape and
    type. Returns an array with the shape of the input."""
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
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data  # Unpack data

    # Retrieve original data dimensions
    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)  # Convert to transmission

    return rehydrated_data
