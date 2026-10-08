# -*- coding: utf-8 -*-
"""
Direct (filtered backprojection) reconstruction for translation CT (TCT) with a planar source trajectory.

The source and the detector translate together: along each translation path the source steps by
``d_tau_x`` in x, and the translation paths are ``d_tau_z`` apart in height z.  The reconstruction is the
discrete TCT FBP formula (Yang, Bouman, Buzzard, direct reconstruction theory for TCT, eq. 53):

    h(n)            = (1 / du^2) (sinc(n) / 2 - sinc^2(n / 2) / 4)                    ramp filter
    g_pad(k, j)     = g(u_k, v_j) for |u_k| <= U,  g(+-U, v_j) for U < |u_k| <= 2U      pad each row with its edge value
    g_tilde(k, j)   = D_sd / sqrt(u_k^2 + v_j^2 + D_sd^2) * g_pad(k, j)                  pre-weight
    q[k, j]         = du * (h * g_tilde(., j))[k],   kept only for |u_k| <= U            filter each row along u, crop
    f(x, y, z)      = d_tau_x * D_sd / (D_sp - y)^2 * (1 / N_S(y, z))
                      * sum_l sum_m q_{m,l}( D_sd (x - tau_x,m) / (D_sp - y),  D_sd (z - tau_z,l) / (D_sp - y) )

where D_sp is the source-to-iso distance, D_sd the source-to-detector distance, U the half width of the valid
detector panel, and N_S(y, z) the number of translation paths whose detector row for the point lies on the
panel.  The lookup is read by linear interpolation in u and v and is zero outside the panel.

The geometry follows :class:`TranslationModel` exactly: a voxel at recon index (i, j, k) is at
x = dv (j - (Nc - 1)/2) - t_x, y = dvr (i - (Nr - 1)/2) - t_y, z = dvs (k - (Ns - 1)/2) - t_z in view t, its
magnification is 1 / (1/M - y / D_sd), and detector coordinates map to indices as in
:meth:`ConeBeamModel.detector_uv_to_mn`.
"""
import numpy as np
import torch

from . import _sharding


def _to_host(model, array, kind):
    """Return a sinogram-like array as a host numpy float32 array."""
    if array is None:
        return None
    if isinstance(array, _sharding.Shards):
        array = model._gather_sinogram(array)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32)


def _levels(values):
    """Cluster nearly equal translation coordinates into levels (tolerates sub-step jitter).
    Returns the sorted level centers and the level index of every value."""
    order = np.sort(values)
    span = order[-1] - order[0]
    tol = 1e-3 * span if span > 0 else 1.0
    starts = np.concatenate([[0], np.where(np.diff(order) > tol)[0] + 1])
    centers = np.array([order[s:e].mean() for s, e in zip(starts, np.append(starts[1:], len(order)))])
    index = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    return centers, index


def tct_direct_recon(model, sinogram, weights=None, view_batch=8, slab_rows=2):
    """
    Direct reconstruction for translation CT with a planar source trajectory (see the module docstring).

    Args:
        model (TranslationModel): The model whose geometry and reconstruction grid are used.
        sinogram (numpy or torch array): Sinogram with shape (num_views, num_det_rows, num_det_channels).
        weights (numpy or torch array, optional): Weight array of the sinogram shape.  The valid detector panel
            is the bounding box of the detector elements with a positive weight in any view (for example the
            dark-boundary mask of :func:`mbirtorch.preprocess.zeiss_tct.get_sino_and_model`).  Defaults to None,
            which uses the whole detector.
        view_batch (int, optional): Number of views filtered at once.  Defaults to 8.
        slab_rows (int, optional): Number of recon rows backprojected at once.  Defaults to 2.

    Returns:
        numpy array: The reconstruction with shape ``recon_shape`` (rows, columns, slices).

    Note:
        The translation vectors must form a planar raster: translation paths along x at a set of heights z
        (a single translation path, i.e. a straight source trajectory, is the special case of one height).
        The step d_tau_x is the median spacing of the x positions.
    """
    device = model.torch_device
    sino = _to_host(model, sinogram, 'sinogram')
    num_views, num_rows, num_channels = sino.shape

    sdd, sid = (float(a) for a in model.get_params(['source_detector_dist', 'source_iso_dist']))
    d_ch, d_row, off_ch, off_row = (float(a) for a in model.get_params(
        ['delta_det_channel', 'delta_det_row', 'det_channel_offset', 'det_row_offset']))
    dv, row_aspect, slice_aspect = (float(a) for a in model.get_params(
        ['delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect']))
    num_recon_rows, num_recon_cols, num_recon_slices = (int(a) for a in model.get_params('recon_shape'))
    tv = np.asarray(model.get_params('translation_vectors'), dtype=np.float64)
    magnification = sdd / sid

    # ---- valid detector panel
    if weights is None:
        r0, r1, c0, c1 = 0, num_rows - 1, 0, num_channels - 1
    else:
        on = _to_host(model, weights, 'weights') > 0
        rows = np.where(on.any(axis=(0, 2)))[0]
        cols = np.where(on.any(axis=(0, 1)))[0]
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    nrow, ncol = r1 - r0 + 1, c1 - c0 + 1

    # ---- filter: pad with edge values to twice the panel width (E = U), pre-weight, ramp filter, crop
    pad = ncol // 2
    npad = ncol + 2 * pad
    u_all = (np.arange(num_channels) - (num_channels - 1) / 2) * d_ch - off_ch
    v_all = (np.arange(num_rows) - (num_rows - 1) / 2) * d_row - off_row
    u_pad = np.concatenate([u_all[c0] - d_ch * np.arange(pad, 0, -1), u_all[c0:c1 + 1],
                            u_all[c1] + d_ch * np.arange(1, pad + 1)])
    pre_weight = torch.tensor(sdd / np.sqrt(u_pad[None, :] ** 2 + v_all[r0:r1 + 1, None] ** 2 + sdd ** 2),
                              dtype=torch.float32, device=device)
    n = np.arange(-(npad - 1), npad)
    h = (1 / d_ch ** 2) * (0.5 * np.sinc(n) - 0.25 * np.sinc(n / 2) ** 2)
    fft_len = int(2 ** np.ceil(np.log2(2 * npad - 1)))           # linear (not circular) convolution
    h_circ = np.zeros(fft_len)
    h_circ[:npad] = h[npad - 1:]
    h_circ[fft_len - (npad - 1):] = h[:npad - 1]
    h_hat = torch.fft.rfft(torch.tensor(h_circ, dtype=torch.float32, device=device))
    q = torch.empty((num_views, nrow, ncol), dtype=torch.float32, device=device)
    for v0 in range(0, num_views, view_batch):
        block = torch.tensor(sino[v0:v0 + view_batch, r0:r1 + 1, c0:c1 + 1], device=device)
        block = torch.cat([block[:, :, :1].expand(-1, -1, pad), block, block[:, :, -1:].expand(-1, -1, pad)], dim=2)
        filtered = torch.fft.irfft(torch.fft.rfft(block * pre_weight[None], n=fft_len, dim=2) * h_hat[None, None, :],
                                   n=fft_len, dim=2) * d_ch
        q[v0:v0 + view_batch] = filtered[:, :, pad:pad + ncol]
    del sino

    # ---- translation lattice
    x_levels, _ = _levels(tv[:, 0])
    z_levels, path = _levels(tv[:, 2])
    d_tau_x = float(np.median(np.diff(x_levels))) if len(x_levels) > 1 else 1.0
    tz_path = [float(tv[path == l, 2].mean()) for l in range(len(z_levels))]
    ty_path = [float(tv[path == l, 1].mean()) for l in range(len(z_levels))]

    # ---- backprojection with the depth weight and the average over translation paths
    ch_center = (num_channels - 1) / 2 - c0
    row_center = (num_rows - 1) / 2 - r0
    x = torch.tensor((np.arange(num_recon_cols) - (num_recon_cols - 1) / 2) * dv,
                     dtype=torch.float32, device=device)[None, :, None]
    z = torch.tensor((np.arange(num_recon_slices) - (num_recon_slices - 1) / 2) * dv * slice_aspect,
                     dtype=torch.float32, device=device)[None, None, :]
    y_all = (np.arange(num_recon_rows) - (num_recon_rows - 1) / 2) * dv * row_aspect
    zero = torch.zeros((), device=device)
    recon = np.zeros((num_recon_rows, num_recon_cols, num_recon_slices), dtype=np.float32)
    for i0 in range(0, num_recon_rows, slab_rows):
        y = torch.tensor(y_all[i0:i0 + slab_rows], dtype=torch.float32, device=device)[:, None, None]
        acc = torch.zeros((y.shape[0], num_recon_cols, num_recon_slices), dtype=torch.float32, device=device)
        for view in range(num_views):
            t = tv[view]
            mag = 1.0 / (1.0 / magnification - (y - float(t[1])) / sdd)
            fn = (mag * (x - float(t[0])) + off_ch) / d_ch + ch_center          # fractional channel, (s, Nx, 1)
            fm = (mag * (z - float(t[2])) + off_row) / d_row + row_center       # fractional row, (s, 1, Nz)
            on = ((fn >= -0.5) & (fn <= ncol - 0.5)) & ((fm >= -0.5) & (fm <= nrow - 0.5))
            fn = fn.clamp(0, ncol - 1)                                          # outer half pixel: edge sample
            fm = fm.clamp(0, nrow - 1)
            n0 = fn.floor().long().clamp(max=ncol - 2)
            m0 = fm.floor().long().clamp(max=nrow - 2)
            a, b = fn - n0, fm - m0
            img = q[view]
            m0e, m1e = m0.expand(-1, num_recon_cols, -1), (m0 + 1).expand(-1, num_recon_cols, -1)
            n0e, n1e = n0.expand(-1, -1, num_recon_slices), (n0 + 1).expand(-1, -1, num_recon_slices)
            val = ((1 - b) * (1 - a) * img[m0e, n0e] + (1 - b) * a * img[m0e, n1e]
                   + b * (1 - a) * img[m1e, n0e] + b * a * img[m1e, n1e])
            acc += torch.where(on, val, zero) * (mag * mag / sdd)
        num_paths = torch.zeros((y.shape[0], 1, num_recon_slices), dtype=torch.float32, device=device)
        for l in range(len(z_levels)):
            mag = 1.0 / (1.0 / magnification - (y - ty_path[l]) / sdd)
            fm = (mag * (z - tz_path[l]) + off_row) / d_row + row_center
            num_paths += ((fm >= -0.5) & (fm <= nrow - 0.5)).float()
        recon[i0:i0 + slab_rows] = (acc * d_tau_x / num_paths.clamp(min=1.0)).cpu().numpy()
    return recon
