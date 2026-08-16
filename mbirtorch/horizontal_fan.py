"""The shared horizontal-fan kernels and the hfan data contract.

Every geometry's horizontal fan applies the same trapezoid rule; the
geometry enters ONLY through the HFAN DATA TUPLE

    (n_p, centers, W_p_c, weight_scale)

computed per (view, pixel) by the geometry's own chain (parallel:
_parallel_hfan_math; cone: _cone_horizontal_data, which also returns
pixel_mag for its vertical fan): the continuous projected channel
coordinate, its rounded center (int32 -- 32-bit indices halve the
contract's bytes and match what GPU kernels prefer; the torch index ops
upcast to int64 inside the compiled region, where the cast fuses), the
projected voxel width in channel units, and the per-geometry weight
scale.  The tap clip bound min(1, W_p_c) is DERIVED at the use site (one
instruction), not carried.  The tap axis is NEVER materialized: taps
expand inside the kernel loops, so the precomputed contract stays
per-(view, pixel) at any psf width -- a future kernel path must not
reintroduce a per-(view, tap, pixel) transient.

The weight rule (identical across geometries): tap n = center + offset
receives

    A = weight_scale * clip((W_p_c + 1) / 2 - |n_p - n|, 0, min(1, W_p_c))

-- the trapezoid overlap of the projected voxel with detector cell n --
zeroed outside the detector.  The weights reproduce the golden reference
to <= 1.6e-6 rel-max.

Torch semantics note: torch index ops assert on out-of-bounds indices
rather than dropping or clamping them, so every tap here uses the
zero-the-weight, clamp-the-index pattern: the weight is zeroed where the
unclipped tap fell outside the detector, then the index is clamped in
range.  This convention is load-bearing beyond correctness: a sorted/CSR
consumer's static stream bound (taps per view == psf_width * P exactly)
rests on no tap ever being dropped.

Layout note: the forward fan produces CHANNEL-MAJOR partial views,
(channels, cols), so the scatter writes contiguous rows rather than
strided columns; the geometry BODY transposes to the
sinogram's (rows, channels) layout on return.  The back fan likewise
gathers from channel-major views (the body transposes each view batch up
front).
"""

import torch

_F32 = torch.float32


def tap_weights(n_p, n, W_p_c, weight_scale, num_channels):
    """The shared trapezoid weight for tap ``n``, with torch-safe OOB
    handling (see the module docstring's zero-and-clamp note).

    Args:
        n_p: continuous projected channel coordinate, (Vb, P).
        n: integer tap channel (center + offset), (Vb, P) int64.
        W_p_c: projected voxel width in channel units, (Vb, P) or
            broadcastable.
        weight_scale: geometry weight scale (e.g. in-plane voxel area /
            footprint length), (Vb, P) or broadcastable.
        num_channels (int): detector channel count.

    Returns:
        (A, n_clamped): weight (Vb, P) float32, zeroed where the unclipped
        tap was out of range, and the in-range index (Vb, P) int64.
    """
    A = torch.clamp((W_p_c + 1.0) / 2.0 - (n_p - n.to(_F32)).abs(), min=0.0)
    A = torch.minimum(A, torch.clamp(W_p_c, max=1.0)) * weight_scale
    A = A * ((n >= 0) & (n < num_channels)).to(_F32)
    return A, n.clamp(0, num_channels - 1)


def fan_forward_batch(hfan_data, values, num_channels, psf_radius):
    """Forward horizontal fan for one view batch: bin weighted per-pixel rows
    into their detector channels with a per-tap scatter-add loop.

    Args:
        hfan_data: (n_p, centers, W_p_c, weight_scale) for this view batch;
            n_p (Vb, P) float32, centers (Vb, P) int32.
        values: (P, num_cols) rows to weight and bin (voxel cylinders,
            broadcast over views -- parallel beam), or (Vb, P, num_cols)
            per-view rows (a two-fan geometry's vertical-fan output -- cone).
        num_channels (int, static): number of detector channels.
        psf_radius (int, static): tap radius (psf_width = 2*psf_radius + 1).

    Returns:
        (Vb, num_channels, num_cols) CHANNEL-MAJOR partial views (contiguous
        scatter rows; the geometry body transposes to the sinogram layout).
    """
    n_p, centers, W_p_c, weight_scale = hfan_data
    centers = centers.to(torch.int64)
    vb, num_pixels = n_p.shape
    num_cols = values.shape[-1]
    dev = values.device
    # One flat (Vb*C, num_cols) accumulator so a single index_add_ covers the
    # whole batch: row v*C + n receives pixel p's contribution to view v,
    # channel n.
    acc = torch.zeros((vb * num_channels, num_cols), dtype=_F32, device=dev)
    row_base = torch.arange(vb, device=dev)[:, None] * num_channels
    for offset in range(-psf_radius, psf_radius + 1):
        A, n = tap_weights(n_p, centers + offset, W_p_c, weight_scale,
                           num_channels)
        idx = (row_base + n).reshape(-1)
        src = (A.unsqueeze(-1) * values).reshape(-1, num_cols)
        acc.index_add_(0, idx, src)
    return acc.view(vb, num_channels, num_cols)


def fan_back_batch(sino_batch_T, hfan_data, num_channels, psf_radius,
                   coeff_power=1, reduce_views=True):
    """Back (adjoint) horizontal fan for one view batch: gather each pixel's
    weighted channel rows (the per-tap gather + multiply-accumulate loop).

    Args:
        sino_batch_T: (Vb, num_channels, num_rows) CHANNEL-MAJOR views (the
            geometry body transposes up front so the per-pixel gather reads
            contiguous rows -- the adjoint of the forward fan's channel-major
            scatter).
        hfan_data: as in :func:`fan_forward_batch`.
        num_channels (int, static): number of detector channels.
        psf_radius (int, static): tap radius.
        coeff_power (int, static): weights raised to this power (2 = the
            Hessian diagonal).
        reduce_views (bool, static): True sums this batch's views (the
            single-fan back projection, returning (P, num_rows)); False keeps
            the view axis (a two-fan geometry feeds the per-view columns to
            its vertical fan, returning (Vb, P, num_rows)).

    Returns:
        (P, num_rows) with reduce_views, else (Vb, P, num_rows); the caller
        accumulates batches.
    """
    n_p, centers, W_p_c, weight_scale = hfan_data
    centers = centers.to(torch.int64)
    vb, num_pixels = n_p.shape
    dev = sino_batch_T.device
    v_idx = torch.arange(vb, device=dev)[:, None]
    num_rows = sino_batch_T.shape[2]
    if reduce_views:
        out = torch.zeros((num_pixels, num_rows), dtype=_F32, device=dev)
    else:
        out = torch.zeros((vb, num_pixels, num_rows), dtype=_F32, device=dev)
    for offset in range(-psf_radius, psf_radius + 1):
        A, n = tap_weights(n_p, centers + offset, W_p_c, weight_scale,
                           num_channels)
        if coeff_power != 1:
            A = A ** coeff_power
        gathered = sino_batch_T[v_idx, n]                    # (Vb, P, num_rows)
        if reduce_views:
            out += torch.einsum("vp,vpr->pr", A, gathered)
        else:
            out = out + A.unsqueeze(-1) * gathered
    return out
