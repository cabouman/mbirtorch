"""Writing a factorization: the dehydrated file, the rehydrated (denoised) hyperspectral data, and the diagnostics
reported with them (fit quality, distinguishability of the components, mean-pixel spectrum)."""
import logging

import h5py
import numpy as np

from .io import _create_hyperspectral, _write_metadata, _written_atomically, export_hsnt_data_hdf5

log = logging.getLogger(__name__)

_CHUNK_ELEMENTS = 2 ** 23       # entries per block of the float64 diagnostics: 64 MiB per array


def fit_quality(T, W, H, dose=None, device="cpu", dose_per_bin=None, open_beam_observations=0):
    """Reduced chi-square of the fit against Poisson noise, the mean of (T - e^-X)^2 / var over the data.

    For counts c ~ Poisson(d e^-X) over an open beam of d counts per bin, var = e^-X / d; an open beam measured as the
    mean of n observations adds its own noise, e^-2X / (n d). Near 1 the residual is at the noise level; well above 1
    the rank is too small or the model is wrong; well below 1 the fit follows the noise.

    Args:
        T (numpy.ndarray): Transmission ratio, shape (pixels, bins).
        W (numpy.ndarray or torch.Tensor): Maps, shape (pixels, rank).
        H (numpy.ndarray or torch.Tensor): Spectra, shape (rank, bins).
        dose (float, optional): Open-beam counts per pixel and bin. Defaults to None, which reports only the relative
            residual in transmission.
        device (str, optional): Torch device for the products. Defaults to 'cpu'.
        dose_per_bin (numpy.ndarray, optional): The open beam's count in each bin, used instead of the scalar dose.
            Defaults to None.
        open_beam_observations (int, optional): Observations averaged into a measured open beam; 0 treats the open
            beam as exact. Defaults to 0.

    Returns:
        dict: 'relative_residual', and 'reduced_chi2' when the dose is given.
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


def component_check(W, H, corr_warn=0.8):
    """Whether the components are distinguishable, from the correlation of their maps.

    When two maps are nearly proportional the data determine only the weighted sum of their spectra, so each row of H
    is an arbitrary, noisy slice of it; on a one-material sample every extra component behaves so.

    Returns:
        dict: 'max_map_correlation', 'proportional_pairs' (i, j, correlation) above corr_warn, and 'row_noise_rel',
        each row's noise level from second differences relative to its median.
    """
    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = H.shape[0]
    out = dict(max_map_correlation=0.0, proportional_pairs=[], row_noise_rel=[])
    for h in H:
        level = float(np.median(h)) if np.median(h) > 0 else float(h.max()) or 1.0
        out["row_noise_rel"].append(float(np.std(np.diff(h, 2)) / np.sqrt(6) / level) if h.size > 3 else 0.0)
    if R < 2:
        return out
    C = np.nan_to_num(np.corrcoef(W.T))
    np.fill_diagonal(C, 0.0)
    out["max_map_correlation"] = float(C.max())
    out["proportional_pairs"] = [(i, j, round(float(C[i, j]), 3)) for i in range(R) for j in range(i + 1, R)
                                 if C[i, j] > corr_warn]
    return out


def mean_pixel_spectrum(W, H, frac=0.25):
    """Attenuation of the average material pixel, sum_k mean(W_pk) H_k, and each component's share of it.

    Unlike the rows of H it does not depend on how the solver split the spectrum among components, so it shows the
    Bragg edges at any rank. Pixels count as material when their total map value exceeds frac of the 99th percentile.

    Returns:
        (total, contributions, pixels): the spectrum (bins,), the shares (rank, bins), and the pixels averaged.
    """
    total = W.sum(1)
    material = total > frac * np.percentile(total, 99)
    if material.sum() < 10:
        material = np.ones_like(material)
    wm = W[material].mean(0)
    return wm @ H, wm[:, None] * H, int(material.sum())


def _provenance(path, bin_indices, attrs, metadata=None):
    """Add the source bin index of each column, the input's hsnt metadata and the run's attributes to an HDF5 file."""
    with h5py.File(path, "a") as f:
        if "bin_indices" not in f:
            f.create_dataset("bin_indices", data=np.asarray(bin_indices))
        _write_metadata(f, metadata or {})
        f.attrs.update(attrs)


def write_dehydrated(path, W4, H, out_type, bin_indices, attrs, metadata=None):
    """Write maps and spectra in the dehydrated layout, with the provenance and the mean-pixel spectrum.

    Args:
        path (str): Output .h5 path.
        W4 (numpy.ndarray): Maps, shape (views, rows, cols, rank).
        H (numpy.ndarray): Spectra, shape (rank, bins).
        out_type (str): 'attenuation' or 'transmission', the quantity rehydrate reconstructs.
        bin_indices (array): Source spectral index of each column of H.
        attrs (dict): HDF5 attributes to record.
        metadata (dict, optional): hsnt metadata of the input (angles, wavelengths, geometry). Defaults to None.

    Returns:
        str: the path.
    """
    total, contrib, n_mat = mean_pixel_spectrum(W4.reshape(-1, H.shape[0]), H)
    with _written_atomically(path) as tmp:
        export_hsnt_data_hdf5(tmp, [W4, H, out_type], {"dataset_type": out_type,
                                                       "dataset_modality": "hyperspectral neutron"})
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
    """Write the rehydrated product W @ H in the hsnt hyperspectral layout, by blocks of whole rows.

    The full array is never held in memory, and each block fills whole HDF5 chunks.

    Args:
        path (str): Output .h5 path.
        spatial_shape (tuple): (views, rows, cols) of the pixels.
        W (numpy.ndarray): Maps, shape (pixels, rank).
        H (numpy.ndarray): Spectra, shape (rank, bins).
        out_type (str): 'attenuation' (W @ H) or 'transmission' (exp(-W @ H)).
        bin_indices (array): Source spectral index of each column of H.
        attrs (dict): HDF5 attributes to record.
        metadata (dict, optional): hsnt metadata of the input (angles, wavelengths, geometry). Defaults to None.
        block_pixels (int, optional): Approximate pixels per block. Defaults to 16384.

    Returns:
        str: the path.
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
