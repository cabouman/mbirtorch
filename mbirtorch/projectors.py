"""Shared horizontal-fan projector kernels (portable forms) and the batched drivers.

Ported from mbirjax.projectors, Phase 1 scope (port_plan.md section 5): the
per-tap scatter-add forward and per-tap gather back only -- no sorted channel
reduction, no stacked gather, no tile policy.  Values match mbirjax's kernels
by construction: the same trapezoid weight rule on the same (n_p, center, W)
inputs, validated cross-framework at <= 1.1e-5 rel-max in the Phase 0 spikes.

Torch semantics note (port_plan.md section 7, item 1): jax DROPS out-of-bounds
scatter indices and CLAMPS out-of-bounds gathers, and mbirjax's per-tap loops
rely on both.  Torch index ops assert on out-of-bounds instead, so every tap
here uses the clip-plus-zero-weight pattern: the weight is zeroed where the
unclipped tap fell outside the detector, then the index is clamped in range.

The drivers batch over VIEWS with a plain python loop (the eager transient is
(view_batch, num_pixels, S) floats), matching the Phase 0 spike formulation.
``view_batch_size`` is the single memory/speed knob; torch.compile of these
bodies is the Phase 2 performance pass.
"""

import torch

_F32 = torch.float32


def tap_weights(n_p, n, W_p_c, weight_scale, L_max, num_channels):
    """The shared trapezoid weight for tap ``n``, with torch-safe OOB handling.

    Weight rule (identical to mbirjax.projectors):
        A = weight_scale * clip((W_p_c + 1)/2 - |n_p - n|, 0, min(1, W_p_c))
    zeroed outside the detector; the returned index is clamped into range.

    Args:
        n_p: continuous projected channel coordinate, (Vb, P).
        n: integer tap channel (center + offset), (Vb, P) int64.
        W_p_c: projected voxel width in channel units, (Vb, 1) (or broadcastable).
        weight_scale: geometry weight scale, (Vb, 1) (or broadcastable).
        L_max: precomputed min(1, W_p_c), same shape as W_p_c.
        num_channels (int): detector channel count.

    Returns:
        (A, n_clamped): weight (Vb, P) float32 and in-range index (Vb, P) int64.
    """
    A = torch.clamp((W_p_c + 1.0) / 2.0 - (n_p - n.to(_F32)).abs(), min=0.0)
    A = torch.minimum(A, L_max) * weight_scale
    A = A * ((n >= 0) & (n < num_channels)).to(_F32)
    return A, n.clamp(0, num_channels - 1)


def fan_forward_batch(hfan_data, values, num_channels, psf_radius):
    """Forward horizontal fan for one view batch: scatter weighted pixel rows
    into channels.

    Args:
        hfan_data: (n_p, centers, W_p_c, weight_scale, L_max) for this view
            batch -- n_p/centers are (Vb, P); the scalars are (Vb, 1).
        values: (P, num_cols) voxel cylinders (num_cols = slices).
        num_channels (int): detector channels.
        psf_radius (int): tap radius (psf_width = 2*psf_radius + 1).

    Returns:
        (Vb, num_channels, num_cols) channel-major partial views (the caller
        transposes to the sinogram's (Vb, rows, channels) layout).
    """
    n_p, centers, W_p_c, weight_scale, L_max = hfan_data
    vb, num_pixels = n_p.shape
    num_cols = values.shape[1]
    dev = values.device
    acc = torch.zeros((vb * num_channels, num_cols), dtype=_F32, device=dev)
    row_base = torch.arange(vb, device=dev)[:, None] * num_channels
    for offset in range(-psf_radius, psf_radius + 1):
        A, n = tap_weights(n_p, centers + offset, W_p_c, weight_scale, L_max,
                           num_channels)
        idx = (row_base + n).reshape(-1)
        src = (A.unsqueeze(-1) * values).reshape(-1, num_cols)
        acc.index_add_(0, idx, src)
    return acc.view(vb, num_channels, num_cols)


def fan_back_batch(sino_batch_T, hfan_data, num_channels, psf_radius, coeff_power=1):
    """Back (adjoint) horizontal fan for one view batch: gather each pixel's
    weighted channel rows, summed over the batch's views.

    Args:
        sino_batch_T: (Vb, num_channels, num_rows) channel-major views.
        hfan_data: as in fan_forward_batch.
        num_channels (int): detector channels.
        psf_radius (int): tap radius.
        coeff_power (int): weights raised to this power (2 = Hessian diagonal).

    Returns:
        (P, num_rows): this batch's contribution (caller accumulates batches).
    """
    n_p, centers, W_p_c, weight_scale, L_max = hfan_data
    vb, num_pixels = n_p.shape
    dev = sino_batch_T.device
    v_idx = torch.arange(vb, device=dev)[:, None]
    out = torch.zeros((num_pixels, sino_batch_T.shape[2]), dtype=_F32, device=dev)
    for offset in range(-psf_radius, psf_radius + 1):
        A, n = tap_weights(n_p, centers + offset, W_p_c, weight_scale, L_max,
                           num_channels)
        if coeff_power != 1:
            A = A ** coeff_power
        gathered = sino_batch_T[v_idx, n]                    # (Vb, P, num_rows)
        out += torch.einsum("vp,vpr->pr", A, gathered)
    return out


class Projectors:
    """The batched sparse projector drivers for one model.

    Holds the runtime view-parameter array and calls back into the model for
    the per-view-batch geometry (``model.compute_hfan_data_batched``), so the
    drivers stay geometry-agnostic (the mbirjax structure).
    """

    def __init__(self, model):
        self.model = model
        view_params_name = model.get_params('view_params_name')
        self.view_params_array = torch.as_tensor(
            model.get_params(view_params_name), dtype=_F32, device=model.torch_device)

    def sparse_forward_project(self, voxel_values, pixel_indices):
        """Project voxel cylinders at ``pixel_indices`` into a full sinogram.

        Args:
            voxel_values: (P, num_slices) tensor (or array-like) of cylinders.
            pixel_indices: (P,) flat indices into the (rows, cols) grid.

        Returns:
            (num_views, num_det_rows, num_det_channels) tensor.
        """
        m = self.model
        dev = m.torch_device
        num_views, num_rows, num_channels = m.get_params('sinogram_shape')
        voxel_values = torch.as_tensor(voxel_values, dtype=_F32, device=dev)
        pixel_indices = torch.as_tensor(pixel_indices, dtype=torch.int64, device=dev)
        psf_radius = m.get_psf_radius()
        vb_size = m.view_batch_size
        sinogram = torch.empty((num_views, num_rows, num_channels), dtype=_F32,
                               device=dev)
        for v0 in range(0, num_views, vb_size):
            params_batch = self.view_params_array[v0:v0 + vb_size]
            hfan = m.compute_hfan_data_batched(pixel_indices, params_batch)
            block = fan_forward_batch(hfan, voxel_values, num_channels, psf_radius)
            # channel-major (Vb, C, S) -> the sinogram's (Vb, rows=S, C) layout
            sinogram[v0:v0 + params_batch.shape[0]] = block.permute(0, 2, 1)
        return sinogram

    def sparse_back_project(self, sinogram, pixel_indices, coeff_power=1):
        """Back-project a sinogram onto the cylinders at ``pixel_indices``.

        Args:
            sinogram: (num_views, num_det_rows, num_det_channels).
            pixel_indices: (P,) flat indices into the (rows, cols) grid.
            coeff_power (int): 1 normally; 2 for the Hessian diagonal.

        Returns:
            (P, num_det_rows) tensor of per-pixel cylinders.
        """
        m = self.model
        dev = m.torch_device
        num_views, num_rows, num_channels = m.get_params('sinogram_shape')
        sinogram = torch.as_tensor(sinogram, dtype=_F32, device=dev)
        pixel_indices = torch.as_tensor(pixel_indices, dtype=torch.int64, device=dev)
        psf_radius = m.get_psf_radius()
        vb_size = m.view_batch_size
        out = torch.zeros((pixel_indices.shape[0], num_rows), dtype=_F32, device=dev)
        for v0 in range(0, num_views, vb_size):
            params_batch = self.view_params_array[v0:v0 + vb_size]
            hfan = m.compute_hfan_data_batched(pixel_indices, params_batch)
            sino_T = sinogram[v0:v0 + params_batch.shape[0]].permute(0, 2, 1).contiguous()
            out += fan_back_batch(sino_T, hfan, num_channels, psf_radius,
                                  coeff_power=coeff_power)
        return out
