"""The hand-written Triton parallel-beam kernels -- the parallel BACK and
FORWARD projections as alternative view-batch bodies.

A kernel here is an ALTERNATIVE BODY, never a new driver, exactly as in
triton_cone.py: each wrapper below has the same signature as the torch body it
replaces (:func:`mbirtorch.parallel_beam._parallel_back_view_batch`,
:func:`mbirtorch.parallel_beam._parallel_forward_view_batch`), so the driver's
view-range loop, its lazy assembly, the row-aligned banded seams and the
``plan`` slot all pass through unchanged, and the torch bodies stay compiled in
everywhere as the value reference and the fallback.

Parallel beam is the DEGENERATE case of the cone pair, and the kernels are the
cone kernels with the vertical fan deleted rather than a second design:

  - No vertical fan at all.  Detector row r IS recon slice r, so the row axis
    rides through both kernels as the vector axis: the back kernel's gathered
    sinogram row band is already the output's slice band, and the forward
    kernel's voxel cylinder is already the detector column.  Everything the
    cone kernels spend on the (m0, W_p_r) affine -- the row/slice tap loop, the
    ``1/cos(phi)`` divisor, the ``floor(x + 0.5)`` center and its inertness
    argument, the slice_start anchor -- is simply absent.  With it goes the
    entire rounding carve-out: these kernels contain no atan2-vs-sqrt divisor
    and no round-vs-floor tie, so they reproduce their torch bodies to float
    summation order alone.
  - Only the horizontal fan remains, from the same hfan contract of
    horizontal_fan.py that the cone kernels use (n_p, centers, W_p_c,
    weight_scale), built eagerly by ``_parallel_hfan_math``.  Tap weights are
    derived IN-KERNEL by the same trapezoid rule, so the tap axis is never
    materialized.
  - The parallel contract is CHEAPER than the cone one in a way worth
    exploiting: W_p_c and weight_scale come from the projected voxel footprint,
    which for parallel beam depends on the view ANGLE alone (see
    ``_parallel_hfan_math``: they are (Vb, 1)).  Both wrappers therefore pass
    them as (Vb,) per-view scalars, so the kernels load two floats per view
    instead of two (Vb, P) planes -- at a 1024-class cell that is a couple of
    hundred MB of traffic and allocation per call that the cone form cannot
    avoid.  The reshape to (Vb,) is the check as well as the conversion: a
    contract that ever became per-pixel would raise here rather than broadcast
    silently.

What the BACK kernel fuses (the mechanism behind the pallas campaign's
9x-class parallel back numbers): each program accumulates its (BLOCK_P,
BLOCK_R) tile of the output IN REGISTERS across the view reduction and the
channel taps, so the per-view (Vb, P, R) gather transient the torch body
materializes is never written at all.

What the FORWARD kernel fuses: the torch forward body materializes a
(Vb, P, R) scaled-value copy per tap and index_adds it into a (Vb*C, R)
accumulator.  The kernel loads its (BLOCK_P, BLOCK_R) tile of the voxel
cylinders ONCE per program and scatters it straight into the sinogram with
per-tap atomics, so no transient is written.  This is the PLAIN-ATOMIC form
the design mandates first (measure, then specialize); the sorted-stream
variant, if it is ever needed, builds its streams from this same contract.

Atomics and determinism (forward only): float atomic_add is commutative but
not associative, so the summation order over pixels and taps varies between
launches and the forward kernel is not bit-reproducible.  It is reproducible
to float rounding, and the repeat-consistency test in
tests/test_triton_parallel.py measures the run-to-run spread directly.

The module imports WITHOUT triton (a CPU/MPS install has none), so the test
suite and the availability self-check can import it anywhere; only calling a
wrapper needs a working triton.
"""

import contextlib
import os

import torch

from ._utils import padded_kernel_width
from .parallel_beam import _parallel_back_view_batch, _parallel_hfan_math
from .projectors import compile_serialized
# The Triton language shims live in the cone module.  _COMPILED_LAUNCH_KEYS is
# shared by all four kernels, and every key leads with its kernel's name.
from .triton_cone import (_COMPILED_LAUNCH_KEYS, _jit, _tap_range, _tile_size,
                          _tl_abs, tl, triton)

_F32 = torch.float32

# The tile constants below were chosen by a sweep on an H100.
PARALLEL_BACK_BLOCK_P = 8
PARALLEL_BACK_BLOCK_R = 256
PARALLEL_BACK_NUM_WARPS = 4
PARALLEL_BACK_NUM_STAGES = 1
# A row band or pixel subset smaller than this tile is padded, not shrunk.
PARALLEL_BACK_MIN_TILE = 8
# The view batch this body asks for when the model's view_batch_size is None.
# The driver's transient budget may cap the realized batch below it.
PARALLEL_BACK_VIEW_CHUNK = 128

PARALLEL_FWD_BLOCK_P = 8
PARALLEL_FWD_BLOCK_R = 128
PARALLEL_FWD_NUM_WARPS = 8
PARALLEL_FWD_NUM_STAGES = 1
PARALLEL_FWD_MIN_TILE = 8
PARALLEL_FWD_VIEW_CHUNK = 128

# Constants for the sorted contraction forward kernel.
PARALLEL_SORTED_BLOCK_P = 32
PARALLEL_SORTED_WINDOW = 16
PARALLEL_SORTED_BLOCK_R = 128
PARALLEL_SORTED_NUM_WARPS = 8
PARALLEL_SORTED_NUM_STAGES = 1
# tl.dot needs every dimension at 16 or more.
PARALLEL_SORTED_MIN_R = 16
PARALLEL_SORTED_VIEW_CHUNK = 16


def sorted_forward_enabled():
    """Whether the parallel forward routes through the sorted-contraction
    kernel (the default) or the original per-tap kernel.

    Read per call, like the other environment switches, so a test or a
    measurement can flip it around one block.  MBIRTORCH_SORTED_FORWARD=0
    restores the per-tap kernel; the switch is the same escape-hatch
    pattern the column-gather flip used while its gate ran.  Both kernels
    compute the same sums in a different order, inside the standing 1e-5
    value gates.
    """
    return os.environ.get('MBIRTORCH_SORTED_FORWARD', '1').strip().lower() \
        not in ('0', 'false', 'no', 'off')


@_jit
def _parallel_back_kernel(n_p_ptr, centers_ptr, w_p_c_ptr, weight_scale_ptr,
                          sino_ptr, out_ptr,
                          num_views, num_pixels, num_channels, num_band_rows,
                          sino_view_stride,
                          PSF_RADIUS: tl.constexpr, COEFF_POWER: tl.constexpr,
                          BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr):
    """One program per (pixel block, row chunk) of the output partial.

        out[p, r] = sum over views v, channel taps tc of
                    Wchan[v, p, tc] ** coeff_power * sino[v, c(v, p) + tc, r]

    Detector row r is recon slice r, so the row axis is the vector axis and the
    gathered row band is the output's slice band.  The pixel block is the fast
    grid axis, so programs that run together gather the same detector rows.

    Pixel lanes beyond ``num_pixels`` load zeroed contract values, which zeroes
    their weights, and their stores are masked.  ``num_band_rows`` is the
    padded row count the wrapper launches at.  Row lanes past the real rows
    read zeros from the padded sinogram copy and store into extra output
    columns that the wrapper slices off.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)     # (BLOCK_R,)
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_band_rows
    tile_mask = p_mask[:, None] & r_mask[None, :]

    acc = tl.zeros((BLOCK_P, BLOCK_R), dtype=tl.float32)
    for v in range(num_views):
        pix_base = v.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        # These are per view scalars.  Under parallel beam the projected
        # footprint of a voxel depends on the view angle alone.
        w_p_c = tl.load(w_p_c_ptr + v)
        weight_scale = tl.load(weight_scale_ptr + v)
        clip = tl.minimum(w_p_c, 1.0)
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
            # The trapezoid rule zeroes the weight where the unclipped tap
            # left the detector, then clamps the index.
            n_tap = centers + (tc - PSF_RADIUS)
            w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
            w_chan = tl.minimum(w_chan, clip) * weight_scale
            w_chan = tl.where((n_tap >= 0) & (n_tap < num_channels),
                              w_chan, 0.0)
            if COEFF_POWER == 2:
                w_chan = w_chan * w_chan
            n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
            # The views are channel major, so the row axis is contiguous and
            # a tile's gather is a unit stride read.
            row_ptr = sino_view_ptr + n_chan.to(tl.int64) * num_band_rows
            vals = tl.load(row_ptr[:, None] + r_offs[None, :], mask=tile_mask,
                           other=0.0)
            acc = acc + w_chan[:, None] * vals

    out_ptrs = (out_ptr + p_offs.to(tl.int64)[:, None] * num_band_rows
                + r_offs[None, :])
    tl.store(out_ptrs, acc, mask=tile_mask)


@torch.compiler.disable
def _parallel_back_view_batch_triton(sino_batch, pixel_indices,
                                     view_params_batch, num_rows, num_cols,
                                     num_channels, delta_det_channel,
                                     det_channel_offset, delta_voxel,
                                     delta_voxel_row, psf_radius,
                                     coeff_power=1, slice_start=0,
                                     band_slices=None, plan=None):
    """Triton parallel back body, replacing
    :func:`mbirtorch.parallel_beam._parallel_back_view_batch`.

    It has the same signature and the same (P, rows) return, written fresh
    each call so the driver may accumulate into it in place.  The row count is
    rounded up to a multiple of 16 for the launch, the channel major sinogram
    copy is made at that padded count with zeros in the extra rows, and the
    return is the real row slice.  This body must stay eager, which the
    ``torch.compiler.disable`` decorator and the ``_mbirtorch_no_compile``
    marker below enforce for the two kinds of caller.  A row aligned geometry
    carries its band in the sinogram's row axis, so the band keywords stay at
    their defaults.  ``plan`` is ignored.
    """
    if triton is None:
        raise RuntimeError('the Triton parallel back body was called without '
                           'triton installed; the selection in '
                           'ParallelBeamModel._view_batch_bodies should have '
                           'kept the torch body (see kernel_availability).')
    assert slice_start == 0 and band_slices is None
    # The kernel has static branches for powers 1 and 2 only.
    if coeff_power not in (1, 2):
        return _parallel_back_view_batch(
            sino_batch, pixel_indices, view_params_batch, num_rows, num_cols,
            num_channels, delta_det_channel, det_channel_offset, delta_voxel,
            delta_voxel_row, psf_radius, coeff_power=coeff_power,
            slice_start=slice_start, band_slices=band_slices, plan=plan)

    n_p, centers, w_p_c, weight_scale = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)

    num_views, num_pixels = n_p.shape
    num_band_rows = int(sino_batch.shape[1])
    # The kernel is launched at a row count rounded up to a multiple of 16.
    # Every use of the row argument takes this value.
    launch_rows = padded_kernel_width(num_band_rows)
    # The views are made channel major, so the per tile gather is contiguous.
    if launch_rows == num_band_rows:
        sino_t = sino_batch.permute(0, 2, 1).contiguous()
    else:
        # A padded row lane would read past the last real row, so the copy is
        # made at the padded row count with zeros in the extra rows.
        sino_t = torch.empty(
            (int(sino_batch.shape[0]), int(sino_batch.shape[2]), launch_rows),
            dtype=sino_batch.dtype, device=sino_batch.device)
        sino_t[:, :, :num_band_rows] = sino_batch.permute(0, 2, 1)
        sino_t[:, :, num_band_rows:] = 0.0
    contract = [t.contiguous() for t in (n_p, centers)]
    # The reshape both converts to per view scalars and checks that they are
    # per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    out = torch.empty((num_pixels, launch_rows), dtype=_F32,
                      device=sino_batch.device)

    block_p = _tile_size(PARALLEL_BACK_BLOCK_P, num_pixels,
                         PARALLEL_BACK_MIN_TILE)
    block_r = _tile_size(PARALLEL_BACK_BLOCK_R, launch_rows,
                         PARALLEL_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_rows // block_r))
    launch_key = ('pback', sino_batch.device.index, int(psf_radius),
                  int(coeff_power), block_p, block_r,
                  int(num_views), int(num_pixels), int(num_channels),
                  launch_rows)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch runs on the tensors' own device, and the device leads the
    # launch key.
    with torch.cuda.device(sino_batch.device), guard:
        _parallel_back_kernel[grid](
            *contract, sino_t, out,
            int(num_views), int(num_pixels), int(num_channels), launch_rows,
            int(num_channels) * launch_rows,
            PSF_RADIUS=int(psf_radius), COEFF_POWER=int(coeff_power),
            BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=PARALLEL_BACK_NUM_WARPS,
            num_stages=PARALLEL_BACK_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_rows == num_band_rows:
        return out
    # The padded columns are sliced off.  The result is a strided view whose
    # rows are each still contiguous.
    return out[:, :num_band_rows]


# The driver reads this marker in maybe_compile.
_parallel_back_view_batch_triton._mbirtorch_no_compile = True


def _parallel_back_view_batch_cost(num_pixels, band_rows, args):
    """Return the bytes resident per view in one back kernel batch, and this
    kernel's nominal view chunk.

    The driver reads this through the ``_view_batch_cost`` attribute.  One view
    holds the horizontal fan contract at 16 bytes per (view, pixel), plus the
    channel major copy of its sinogram plane at the padded row count.  Tensors
    whose size does not depend on the batch are not charged.
    """
    plane_bytes = (4 * int(args['num_channels'])
                   * padded_kernel_width(band_rows))
    return 16 * int(num_pixels) + plane_bytes, PARALLEL_BACK_VIEW_CHUNK


_parallel_back_view_batch_triton._view_batch_cost = \
    _parallel_back_view_batch_cost


@_jit
def _parallel_forward_kernel(n_p_ptr, centers_ptr, w_p_c_ptr,
                             weight_scale_ptr, values_ptr, out_ptr,
                             num_pixels, num_channels, num_cols,
                             out_view_stride,
                             PSF_RADIUS: tl.constexpr, BLOCK_P: tl.constexpr,
                             BLOCK_R: tl.constexpr):
    """One program per (pixel block, column chunk, view) of the sinogram.

        out[v, c, r] += sum over pixels p, channel taps tc of
                        Wchan[v, p, tc] * values[p, r]

    Under parallel beam a voxel cylinder's column r lands on detector row r in
    every view, so each program loads its tile of ``values`` once before the
    tap loop.  The view is a grid axis because each view writes its own output
    plane.  The pixel block is the fast axis, so programs that run together
    read the same rows of ``values`` and hit neighboring channels.

    The output is channel major (Vb, C, R), which puts the row axis on the
    contiguous stride.  The wrapper transposes the view on return.

    Pixel lanes beyond ``num_pixels`` have their atomics masked off.
    ``num_cols`` is the padded column count the wrapper launches at.  Column
    lanes past the real columns read zeros from the padded ``values`` copy and
    add into extra output columns that the wrapper slices off.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)     # (BLOCK_R,)
    v = tl.program_id(2)
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_cols
    tile_mask = p_mask[:, None] & r_mask[None, :]

    pix_base = v.to(tl.int64) * num_pixels + p_offs
    n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
    centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
    # These are per view scalars.
    w_p_c = tl.load(w_p_c_ptr + v)
    weight_scale = tl.load(weight_scale_ptr + v)
    clip = tl.minimum(w_p_c, 1.0)

    # The voxel cylinders are read once and held in registers across the taps.
    vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * num_cols
                   + r_offs[None, :], mask=tile_mask, other=0.0)

    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
        # The trapezoid rule again.  Taps outside the detector drop out of the
        # atomic mask instead of being added as zeros.
        n_tap = centers + (tc - PSF_RADIUS)
        w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                            - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
        w_chan = tl.minimum(w_chan, clip) * weight_scale
        n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
        out_ptrs = (out_view_ptr + n_chan.to(tl.int64)[:, None] * num_cols
                    + r_offs[None, :])
        tl.atomic_add(out_ptrs, w_chan[:, None] * vals,
                      mask=tile_mask & ((n_tap >= 0)
                                        & (n_tap < num_channels))[:, None])


@_jit
def _parallel_forward_sorted_kernel(n_p_ptr, centers_ptr, w_p_c_ptr,
                                    weight_scale_ptr, values_ptr, perm_ptr,
                                    out_ptr, num_views, num_pixels,
                                    num_channels, num_cols, out_view_stride,
                                    VIEW_CHUNK: tl.constexpr,
                                    WINDOW: tl.constexpr,
                                    PSF_RADIUS: tl.constexpr,
                                    BLOCK_P: tl.constexpr,
                                    BLOCK_R: tl.constexpr):
    """One program per (pixel block, column chunk, view chunk), with the
    pixels sorted per view by channel center.

    Sorting puts a tile's taps in a narrow channel window, so the scatter
    becomes a small dense contraction.  The tile's trapezoid weights form a
    (BLOCK_P, WINDOW) matrix, transpose(W) @ values accumulates the tile, and
    the window lands with one atomic add per (channel, column).  The
    contraction runs in full precision input mode, because the tensor core
    default rounds inputs to a 10 bit mantissa and fails the value checks.

    A tile whose sorted span exceeds the window falls back to the per tap
    block below.  A view chunk past the end of the batch clamps its view index
    and masks its stores.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    v0 = tl.program_id(2) * VIEW_CHUNK
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_cols
    tile_mask = p_mask[:, None] & r_mask[None, :]

    for dv in range(VIEW_CHUNK):
        v = v0 + dv
        v_ok = v < num_views
        # The clamp keeps every address in bounds for a tail chunk.  The store
        # masks carry v_ok, so a clamped iteration writes nothing.
        v_safe = tl.minimum(v, num_views - 1)
        pix_base = v_safe.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        w_p_c = tl.load(w_p_c_ptr + v_safe)
        weight_scale = tl.load(weight_scale_ptr + v_safe)
        clip = tl.minimum(w_p_c, 1.0)
        out_view_ptr = out_ptr + v_safe.to(tl.int64) * out_view_stride
        row_idx = tl.load(perm_ptr + pix_base, mask=p_mask, other=0)
        vals = tl.load(values_ptr + row_idx.to(tl.int64)[:, None] * num_cols
                       + r_offs[None, :], mask=tile_mask, other=0.0)

        big = 2147483647
        c_lo = tl.min(tl.where(p_mask, centers, big)) - PSF_RADIUS
        c_hi = tl.max(tl.where(p_mask, centers, -big)) + PSF_RADIUS
        span = c_hi - c_lo + 1
        if span <= WINDOW:
            # The window weights are the same trapezoid formula evaluated at every
            # window channel.  The window adds exact zeros to the tap path weights.
            j = tl.arange(0, WINDOW)
            c = c_lo + j
            w = tl.maximum(
                (w_p_c + 1.0) / 2.0
                - _tl_abs(n_p[:, None] - c.to(tl.float32)[None, :]), 0.0)
            w = tl.minimum(w, clip) * weight_scale
            w = tl.where(((c >= 0) & (c < num_channels))[None, :], w, 0.0)
            w = tl.where(p_mask[:, None], w, 0.0)
            out_window = tl.dot(tl.trans(w), vals,
                                input_precision="ieee")
            c_addr = tl.minimum(tl.maximum(c, 0), num_channels - 1)
            win_ptrs = (out_view_ptr
                        + c_addr.to(tl.int64)[:, None] * num_cols
                        + r_offs[None, :])
            # The span mask keeps lanes past the tile's real span from
            # issuing an atomic.
            tl.atomic_add(win_ptrs, out_window,
                          mask=(v_ok
                                & ((j <= (c_hi - c_lo))
                                   & (c >= 0)
                                   & (c < num_channels))[:, None]
                                & r_mask[None, :]))
        else:
            for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
                n_tap = centers + (tc - PSF_RADIUS)
                w_chan = tl.maximum(
                    (w_p_c + 1.0) / 2.0
                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, clip) * weight_scale
                n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
                out_ptrs = (out_view_ptr
                            + n_chan.to(tl.int64)[:, None] * num_cols
                            + r_offs[None, :])
                tl.atomic_add(out_ptrs, w_chan[:, None] * vals,
                              mask=(v_ok & tile_mask
                                    & ((n_tap >= 0)
                                       & (n_tap < num_channels))[:, None]))


@torch.compiler.disable
def _parallel_forward_view_batch_triton(values, pixel_indices,
                                        view_params_batch, num_rows, num_cols,
                                        num_channels, delta_det_channel,
                                        det_channel_offset, delta_voxel,
                                        delta_voxel_row, psf_radius,
                                        slice_start=0, plan=None):
    """Triton parallel forward body, replacing
    :func:`mbirtorch.parallel_beam._parallel_forward_view_batch`.

    It has the same signature and the same (Vb, rows, C) return, zeroed each
    call because the kernel accumulates into it with atomics.  The column count
    is rounded up to a multiple of 16 for the launch, ``values`` is copied into
    a zero padded array of that width, and the return is the real column slice.
    ``values`` is (P, cols), where cols is the detector row count of the block
    this call produces.  Rows track slices, so a slice band is a row band and
    needs no z anchor.  ``plan`` is ignored.
    """
    if triton is None:
        raise RuntimeError('the Triton parallel forward body was called '
                           'without triton installed; the selection in '
                           'ParallelBeamModel._view_batch_bodies should have '
                           'kept the torch body (see kernel_availability).')
    assert slice_start == 0
    n_p, centers, w_p_c, weight_scale = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)

    num_views, num_pixels = n_p.shape
    num_value_cols = int(values.shape[1])
    # The kernel is launched at a column count rounded up to a multiple of 16.
    # Every use of the column argument takes this value.
    launch_cols = padded_kernel_width(num_value_cols)
    if launch_cols == num_value_cols:
        values = values.contiguous()
    else:
        # A padded column lane would read past the last real column, so the
        # copy is made at the padded width with zeros in the extra columns.
        padded_values = torch.empty((int(values.shape[0]), launch_cols),
                                    dtype=values.dtype, device=values.device)
        padded_values[:, :num_value_cols] = values
        padded_values[:, num_value_cols:] = 0.0
        values = padded_values
    if sorted_forward_enabled():
        # On the sorted route the pixels are sorted per view by channel center.  The
        # permutation maps each sorted position back to its row of ``values``.
        order = torch.argsort(n_p, dim=1)
        contract = [torch.gather(n_p, 1, order).contiguous(),
                    torch.gather(centers, 1, order).contiguous(),
                    w_p_c.reshape(num_views).contiguous(),
                    weight_scale.reshape(num_views).contiguous()]
        perm = order.to(torch.int32).contiguous()
        out = torch.zeros((num_views, num_channels, launch_cols), dtype=_F32,
                          device=values.device)
        block_p = PARALLEL_SORTED_BLOCK_P
        block_r = max(PARALLEL_SORTED_MIN_R,
                      _tile_size(PARALLEL_SORTED_BLOCK_R, launch_cols,
                                 PARALLEL_SORTED_MIN_R))
        grid = (-(-num_pixels // block_p), -(-launch_cols // block_r),
                -(-num_views // PARALLEL_SORTED_VIEW_CHUNK))
        launch_key = ('pfwd_sorted', values.device.index, int(psf_radius),
                      block_p, block_r, int(num_views),
                      int(num_pixels), int(num_channels), launch_cols)
        first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
        guard = (compile_serialized() if first_launch
                 else contextlib.nullcontext())
        with torch.cuda.device(values.device), guard:
            _parallel_forward_sorted_kernel[grid](
                *contract, values, perm, out,
                int(num_views), int(num_pixels), int(num_channels),
                launch_cols, int(num_channels) * launch_cols,
                VIEW_CHUNK=PARALLEL_SORTED_VIEW_CHUNK,
                WINDOW=PARALLEL_SORTED_WINDOW,
                PSF_RADIUS=int(psf_radius), BLOCK_P=block_p,
                BLOCK_R=block_r,
                num_warps=PARALLEL_SORTED_NUM_WARPS,
                num_stages=PARALLEL_SORTED_NUM_STAGES)
        _COMPILED_LAUNCH_KEYS.add(launch_key)
        if launch_cols == num_value_cols:
            return out.permute(0, 2, 1)
        return out[:, :, :num_value_cols].permute(0, 2, 1)

    contract = [t.contiguous() for t in (n_p, centers)]
    # The reshape both converts to per view scalars and checks that they are
    # per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    # The output is channel major and zeroed, because the atomics accumulate
    # into it.  The return transposes each view.
    out = torch.zeros((num_views, num_channels, launch_cols), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(PARALLEL_FWD_BLOCK_P, num_pixels,
                         PARALLEL_FWD_MIN_TILE)
    block_r = _tile_size(PARALLEL_FWD_BLOCK_R, launch_cols,
                         PARALLEL_FWD_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_cols // block_r),
            num_views)
    launch_key = ('pfwd', values.device.index, int(psf_radius), block_p,
                  block_r, int(num_views),
                  int(num_pixels), int(num_channels), launch_cols)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch runs on the tensors' own device, and the device leads the
    # launch key.
    with torch.cuda.device(values.device), guard:
        _parallel_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), launch_cols,
            int(num_channels) * launch_cols,
            PSF_RADIUS=int(psf_radius), BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=PARALLEL_FWD_NUM_WARPS,
            num_stages=PARALLEL_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_cols == num_value_cols:
        return out.permute(0, 2, 1)
    # The extra columns are sliced off before the transpose.
    return out[:, :, :num_value_cols].permute(0, 2, 1)


# The driver reads this marker in maybe_compile.
_parallel_forward_view_batch_triton._mbirtorch_no_compile = True


def _parallel_forward_view_batch_cost(num_pixels, num_value_cols, args):
    """Return the bytes resident per view in one forward kernel batch, and
    this kernel's nominal view chunk.

    One view holds the horizontal fan contract at 16 bytes per (view, pixel),
    plus its zeroed channel major output plane at the padded column count.
    The sorted route adds 20 more bytes per (view, pixel) for the sort order,
    the permutation, and the gathered contract copies.
    """
    plane_bytes = (4 * int(args['num_channels'])
                   * padded_kernel_width(num_value_cols))
    per_view = 16 * int(num_pixels) + plane_bytes
    if sorted_forward_enabled():
        per_view += 20 * int(num_pixels)
    return per_view, PARALLEL_FWD_VIEW_CHUNK


_parallel_forward_view_batch_triton._view_batch_cost = \
    _parallel_forward_view_batch_cost
