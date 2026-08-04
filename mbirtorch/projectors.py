"""Shared horizontal-fan projector kernels (portable forms) and the batched drivers.

Ported from mbirjax.projectors, Phase 1 scope (port_plan.md section 5): the
per-tap scatter-add forward and per-tap gather back only -- no sorted channel
reduction, no stacked gather, no tile policy (those are the jax perf layer;
their torch analogs are Phase 2/5 work).

Every geometry's horizontal fan applies the same trapezoid rule; the geometry
enters ONLY through (n_p, n_p_center, W_p_c) -- the continuous projected
channel coordinate, its rounded center, and the projected voxel width in
channel units -- plus a per-geometry weight scale.  The weight rule (identical
across geometries): tap n = n_p_center + offset receives

    A = weight_scale * clip((W_p_c + 1) / 2 - |n_p - n|, 0, min(1, W_p_c))

-- the trapezoid overlap of the projected voxel with detector cell n -- zeroed
outside the detector.  Values match mbirjax's kernels by construction
(validated cross-framework at <= 1.6e-6 rel-max on the goldens).

Torch semantics note (port_plan.md section 7, item 1): jax DROPS out-of-bounds
scatter indices and CLAMPS out-of-bounds gathers, and mbirjax's per-tap loops
rely on both.  Torch index ops assert on out-of-bounds instead, so every tap
here uses the clip-plus-zero-weight pattern (the same pattern mbirjax's sorted
branch already used): the weight is zeroed where the unclipped tap fell
outside the detector, then the index is clamped in range.

Layout note (from mbirjax): the forward fan produces CHANNEL-MAJOR partial
views, (channels, cols), so the scatter writes contiguous rows rather than
strided columns; the drivers transpose to the sinogram's (rows, channels)
layout on write.  The back fan likewise gathers from channel-major views (the
drivers transpose each view batch up front).

The drivers batch over VIEWS with a plain python loop; the eager transient is
(view_batch, num_pixels, S) floats, so ``view_batch_size`` is the single
memory/speed knob.  torch.compile of these bodies is the Phase 2 performance
pass.
"""

import torch

_F32 = torch.float32


def tap_weights(n_p, n, W_p_c, weight_scale, L_max, num_channels):
    """The shared trapezoid weight for tap ``n``, with torch-safe OOB handling.

    Args:
        n_p: continuous projected channel coordinate, (Vb, P).
        n: integer tap channel (center + offset), (Vb, P) int64.
        W_p_c: projected voxel width in channel units, (Vb, 1) (or broadcastable).
        weight_scale: geometry weight scale (scalar or per-view; e.g. in-plane
            voxel area / footprint length), (Vb, 1) (or broadcastable).
        L_max: precomputed min(1, W_p_c), same shape as W_p_c.
        num_channels (int): detector channel count.

    Returns:
        (A, n_clamped): weight (Vb, P) float32, zeroed where the unclipped tap
        was out of range, and the in-range index (Vb, P) int64.
    """
    A = torch.clamp((W_p_c + 1.0) / 2.0 - (n_p - n.to(_F32)).abs(), min=0.0)
    A = torch.minimum(A, L_max) * weight_scale
    A = A * ((n >= 0) & (n < num_channels)).to(_F32)
    return A, n.clamp(0, num_channels - 1)


def fan_forward_batch(hfan_data, values, num_channels, psf_radius):
    """Forward horizontal fan for one view batch: bin weighted per-pixel rows
    into their detector channels (the per-tap scatter-add loop, mbirjax's
    portable/CPU formulation).

    Args:
        hfan_data: (n_p, centers, W_p_c, weight_scale, L_max) for this view
            batch -- n_p/centers are (Vb, P); the per-view scalars are (Vb, 1).
        values: (P, num_cols) rows to weight and bin (voxel cylinders;
            num_cols = slices for parallel beam).
        num_channels (int, static): number of detector channels.
        psf_radius (int, static): tap radius (psf_width = 2*psf_radius + 1 taps).

    Returns:
        (Vb, num_channels, num_cols) CHANNEL-MAJOR partial views (contiguous
        scatter rows; the caller transposes to the sinogram layout).
    """
    n_p, centers, W_p_c, weight_scale, L_max = hfan_data
    vb, num_pixels = n_p.shape
    num_cols = values.shape[1]
    dev = values.device
    # One flat (Vb*C, num_cols) accumulator so a single index_add_ covers the
    # whole batch: row v*C + n receives pixel p's contribution to view v,
    # channel n.
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
    weighted channel rows (the per-tap gather + multiply-accumulate loop),
    summed over the batch's views.

    Args:
        sino_batch_T: (Vb, num_channels, num_rows) CHANNEL-MAJOR views (the
            caller transposes up front so the per-pixel gather reads contiguous
            rows -- the adjoint of the forward fan's channel-major scatter).
        hfan_data: as in :func:`fan_forward_batch`.
        num_channels (int, static): number of detector channels.
        psf_radius (int, static): tap radius.
        coeff_power (int, static): weights raised to this power (2 = the
            Hessian diagonal).

    Returns:
        (P, num_rows): this batch's contribution (the caller accumulates
        batches; summing over views here keeps the transient bounded).
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

    Holds the runtime view-parameter array (angles for parallel beam) and calls
    back into the model for the per-view-batch fan geometry
    (``model.compute_hfan_data_batched``), so the drivers stay geometry-agnostic
    -- the mbirjax structure, minus its jit/static-argument machinery.

    Center-consistency contract (adapted from mbirjax's rounding-fix design):
    forward and back consume the SAME deterministic center computation for each
    (view, pixel), so the pair stays exactly adjoint even at rounding ties.  In
    mbirjax the centers are computed once outside the jitted programs (an XLA
    miscompile workaround); here there is no compiler hazard, and recomputing
    the same deterministic chain per call preserves the consistency property.
    """

    def __init__(self, model):
        self.model = model
        view_params_name = model.get_params('view_params_name')
        self.view_params_array = torch.as_tensor(
            model.get_params(view_params_name), dtype=_F32, device=model.torch_device)

    def sparse_forward_project(self, voxel_values, pixel_indices):
        """Forward project the given voxel cylinders into a full sinogram.

        Args:
            voxel_values: (P, num_recon_slices) tensor (or array-like) of voxel
                values, where voxel_values[i, j] is the value of the voxel in
                slice j at the location determined by pixel_indices[i].
            pixel_indices: (P,) indices into the flattened array of size
                num_rows x num_cols.

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
            # channel-major (Vb, C, S) -> the sinogram's (Vb, rows=S, C) layout.
            sinogram[v0:v0 + params_batch.shape[0]] = block.permute(0, 2, 1)
        return sinogram

    def sparse_back_project(self, sinogram, pixel_indices, coeff_power=1):
        """Back project the sinogram onto the voxel cylinders at ``pixel_indices``.

        Args:
            sinogram: (num_views, num_det_rows, num_det_channels).
            pixel_indices: (P,) indices into the flattened array of size
                num_rows x num_cols.
            coeff_power (int): backproject using the coefficients of
                (A_ij ** coeff_power).  Normally 1, but 2 when computing the
                Hessian diagonal.

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
