"""This module writes the results of a factorization ``X = W H`` to HDF5 files.

One file holds the maps W and the spectra H, which are the output of ``dehydrate``.  Another
file holds the denoised data, which is the product of W and H.  The module also computes the
fit diagnostics that are reported with these files.
"""
import logging

import h5py
import numpy as np

from .io import _create_hyperspectral, _write_metadata, _written_atomically, export_hsnt_data_hdf5

log = logging.getLogger(__name__)

_CHUNK_ELEMENTS = 2 ** 23       # Entries per block in fit_quality: 64 MiB per float64 array.


def fit_quality(T, W, H, dose=None, device="cpu", dose_per_bin=None, open_beam_observations=0):
    """This function computes the reduced chi-square of the fit under a Poisson noise model.
    The reduced chi-square is the mean of (T - e^-X)^2 / var over the data.

    A value near 1 means that the residual is at the noise level.  A value well above 1 means
    that the rank is too small, the model is wrong, or the dose is overestimated.  A value well
    below 1 means that the dose is underestimated.

    Args:
        T (numpy.ndarray): The transmission ratio, with shape (pixels, bins).
        W (numpy.ndarray or torch.Tensor): The maps, with shape (pixels, rank).
        H (numpy.ndarray or torch.Tensor): The spectra, with shape (rank, bins).
        dose (float, optional): The open beam counts per pixel and bin.  If dose is None, the
            function reports only the relative residual in transmission.  Defaults to None.
        device (str, optional): The torch device for the matrix products.  Defaults to 'cpu'.
        dose_per_bin (numpy.ndarray, optional): The open beam count in each bin.  The function
            uses it in place of the scalar dose.  Defaults to None.
        open_beam_observations (float, optional): The effective number of observations in a
            measured open beam.  The value 0 treats the open beam as exact.  Defaults to 0.

    Returns:
        dict: The key 'relative_residual', and also the key 'reduced_chi2' when the dose is given.
    """
    import torch
    W = torch.as_tensor(W, device=device)
    H = torch.as_tensor(H, device=device).double()
    d_k = None if dose is None else torch.as_tensor(dose if dose_per_bin is None else dose_per_bin, device=device,
                                                    dtype=torch.float64).clamp_min(1e-12)
    chunk = max(1, _CHUNK_ELEMENTS // T.shape[1])
    num = den = res = tot = 0.0
    for i in range(0, T.shape[0], chunk):
        Tb = torch.from_numpy(T[i:i + chunk]).to(device).double()
        Th = torch.exp(-(W[i:i + chunk].double() @ H))
        d = Tb - Th
        res += (d * d).sum().item()
        tot += (Tb * Tb).sum().item()
        if d_k is not None:
            var = Th.clamp_min(1e-12) / d_k
            if open_beam_observations:
                var = var + Th * Th / (open_beam_observations * d_k)
            num += ((d * d) / var).sum().item()
        den += Tb.numel()
    out = dict(relative_residual=float(np.sqrt(res / tot)))
    if d_k is not None:
        out["reduced_chi2"] = num / den
    return out


def component_check(W, H, cos_warn=0.99):
    """This function uses the angle between the spectra to check whether the components can be
    told apart.  If two spectra are nearly proportional, the two maps divide one spectral shape
    between them.

    Returns:
        dict: The dict has four keys.  'max_map_correlation' is the largest correlation between
            two maps.  'max_spectral_cosine' is the largest cosine between two spectra.
            'proportional_pairs' lists the tuples (i, j, cosine) with cosine above cos_warn.
            'row_noise_rel' gives the noise level of each spectrum relative to its median.
    """
    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = H.shape[0]
    out = dict(max_map_correlation=0.0, max_spectral_cosine=0.0, proportional_pairs=[], row_noise_rel=[])
    for h in H:
        level = float(np.median(h)) if np.median(h) > 0 else float(h.max()) or 1.0
        out["row_noise_rel"].append(float(np.std(np.diff(h, 2)) / np.sqrt(6) / level) if h.size > 3 else 0.0)
    if R < 2:
        return out
    C = np.nan_to_num(np.corrcoef(W.T))
    np.fill_diagonal(C, 0.0)
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-300)
    S = Hn @ Hn.T
    np.fill_diagonal(S, 0.0)
    out["max_map_correlation"], out["max_spectral_cosine"] = float(C.max()), float(S.max())
    out["proportional_pairs"] = [(i, j, round(float(S[i, j]), 4)) for i in range(R) for j in range(i + 1, R)
                                 if S[i, j] > cos_warn]
    return out


def mean_pixel_spectrum(W, H, frac=0.25):
    """This function computes the attenuation spectrum of the average material pixel.  This
    spectrum is sum_k mean(W_pk) H_k.  The function also computes the contribution of each
    component to this spectrum.  A pixel counts as material when the sum of its map values
    exceeds frac times the 99th percentile of that sum.

    Returns:
        tuple: (total, contributions, pixels).  total is the spectrum, with shape (bins,).
            contributions has shape (rank, bins).  pixels is the number of pixels averaged.
    """
    total = W.sum(1)
    material = total > frac * np.percentile(total, 99)
    if material.sum() < 10:
        material = np.ones_like(material)
    wm = W[material].mean(0)
    return wm @ H, wm[:, None] * H, int(material.sum())


def _provenance(path, bin_indices, attrs, metadata=None):
    """This function adds the source bin indices, the hsnt metadata, and the run attributes to a file."""
    with h5py.File(path, "a") as f:
        if "bin_indices" not in f:
            f.create_dataset("bin_indices", data=np.asarray(bin_indices))
        _write_metadata(f, metadata or {})
        f.attrs.update(attrs)


def write_dehydrated(path, W4, H, bin_indices, attrs, metadata=None):
    """This function writes the maps and spectra to an hsnt file in the format that
    ``dehydrate`` returns.  It also records the source bin indices, the metadata, and the run
    attributes.

    Args:
        path (str): The output .h5 path.
        W4 (numpy.ndarray): The maps, with shape (views, rows, cols, rank).
        H (numpy.ndarray): The spectra, with shape (rank, bins).
        bin_indices (array): The source spectral index of each column of H.
        attrs (dict): The HDF5 attributes to record.
        metadata (dict, optional): The hsnt metadata of the input.  Defaults to None.

    Returns:
        str: The path.
    """
    total, contrib, n_mat = mean_pixel_spectrum(W4.reshape(-1, H.shape[0]), H)
    with _written_atomically(path) as tmp:
        export_hsnt_data_hdf5(tmp, [W4, H], {"dataset_modality": "hyperspectral neutron"})
        _provenance(tmp, bin_indices, attrs, metadata)
        with h5py.File(tmp, "a") as f:
            d = f.create_dataset("mean_pixel_spectrum", data=total.astype(np.float32))
            d.attrs["description"] = (f"attenuation of the average material pixel ({n_mat} pixels), sum_k mean(W_pk) "
                                      "H_k; independent of the split among components")
            f.create_dataset("mean_pixel_contributions", data=contrib.astype(np.float32))
    log.info("wrote %s: subspace_data %s (maps), subspace_basis %s (spectra), mean_pixel_spectrum over %s pixels",
             path, W4.shape, H.shape, f"{n_mat:,}")
    return path


def write_denoised(path, spatial_shape, W, H, out_type, bin_indices, attrs, metadata=None, block_pixels=16384):
    """This function writes the denoised data W @ H to an hsnt file.  It computes and writes the
    data in blocks of whole rows, so it never holds the full array in memory.

    Args:
        path (str): The output .h5 path.
        spatial_shape (tuple): The pixel dimensions (views, rows, cols).
        W (numpy.ndarray): The maps, with shape (pixels, rank).
        H (numpy.ndarray): The spectra, with shape (rank, bins).
        out_type (str): 'attenuation' (W @ H) or 'transmission' (exp(-W @ H)).
        bin_indices (array): The source spectral index of each column of H.
        attrs (dict): The HDF5 attributes to record.
        metadata (dict, optional): The hsnt metadata of the input.  Defaults to None.
        block_pixels (int, optional): The approximate number of pixels in each block.
            Defaults to 16384.

    Returns:
        str: The path.
    """
    R, K = H.shape
    V, rows, cols = spatial_shape
    chunk_rows = min(rows, 64)
    block_rows = max(chunk_rows, block_pixels // cols // chunk_rows * chunk_rows)
    with _written_atomically(path) as tmp:
        with h5py.File(tmp, "w") as f:
            d = _create_hyperspectral(f, (V, rows, cols, K), out_type, chunks=(1, chunk_rows, cols, min(16, K)))
            for v in range(V):
                for r0 in range(0, rows, block_rows):
                    r1 = min(r0 + block_rows, rows)
                    p0 = (v * rows + r0) * cols
                    X = W[p0:p0 + (r1 - r0) * cols] @ H
                    d[v, r0:r1] = (np.exp(-X) if out_type == "transmission" else X).astype(np.float32).reshape(
                        r1 - r0, cols, K)
        _provenance(tmp, bin_indices, dict(attrs, rank=R, rehydrated="1"), metadata)
    log.info("wrote %s: data %s %s, %.2f GiB", path, (V, rows, cols, K), out_type, V * rows * cols * K * 4 / 2**30)
    return path
