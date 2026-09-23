"""The hand-written Triton cone-beam kernels -- the cone BACK and FORWARD
projections as alternative
view-batch bodies.

A kernel here is an ALTERNATIVE BODY, never a new driver.  Each wrapper below
has the same signature as the torch body it replaces
(:func:`mbirtorch.cone_beam._cone_back_view_batch`,
:func:`mbirtorch.cone_beam._cone_forward_view_batch`), so the driver's
view-range loop, its lazy assembly, the banded seams (``slice_start``,
``band_slices``, ``dev_index``) and the ``plan`` slot all pass through
unchanged, and the torch bodies stay compiled in everywhere as the value
reference and the fallback.

Both kernels stand on the same two eager precomputes, which is why one module
covers both directions: the hfan contract of horizontal_fan.py (n_p, centers,
W_p_c, weight_scale) and the (m0, W_p_r) pair from
:func:`_cone_vertical_affine`, the sanctioned geometry bridge.  Every tap
weight is derived from them IN-KERNEL, so the tap axis is never materialized.
Curved detectors and helical z shifts therefore need no kernel code at all:
they enter through those two builders alone (curved through
_cone_horizontal_data's arc parameterisation, helical through the z_shifts
that set z_offset, which the cone-angle chain reads).

What the BACK kernel fuses (the mechanism behind the pallas campaign's 9.07x
on this same math): each program accumulates its (BLOCK_P, BLOCK_L) tile of
the output IN REGISTERS across the view reduction and both tap loops, so the
per-view (Vb, P, R) gather transient the torch body materializes is never
written at all.

What the FORWARD kernel fuses: the torch forward body materializes a
(Vb, P, S) scaled-value copy and a (Vb, P, R) detector-column transient, then
scatters that second one channel by channel.  The kernel keeps its
(BLOCK_P, BLOCK_R) detector column IN REGISTERS through the vertical tap loop
and scatters it straight into the sinogram with per-tap atomics, so neither
transient is written.  This is the PLAIN-ATOMIC form the design mandates
first: the pallas evidence says atomics are the forward's limiter, but
Triton's atomics and scheduling differ enough that the plain form is measured
before the sorted-stream specialization (measure, then specialize).

Three arithmetic deviations from the torch bodies, all inherited from the
pallas cone kernels and all covered by the design's value gate (rel 1e-5 on
the gradient path, 1e-4 at coeff_power=2, with the rounding carve-out):

  - The cone-angle divisor is formed as ``inv_cos_phi = sqrt(1 + (v/SDD)^2)``
    and MULTIPLIED, where the torch bodies divide by ``cos(atan2(v, SDD))``.
    The forms are algebraically identical and differ by a ULP or two of
    rounding; the sqrt form needs no atan2 (whose Triton lowering is
    backend-specific) and is Inf-safe at SDD = inf, where inv_sdd = 0 gives
    exactly 1.
  - Row centers (back) use ``floor(m + 0.5)`` where the torch body uses
    ``torch.round`` (half to even).  The two differ only at an exact .5 tie,
    and there the tap that enters or leaves the window carries weight exactly
    zero: its trapezoid weight is clip((W_p_r + 1)/2 - (psf_radius + 0.5), 0)
    and W_p_r <= 2 * psf_radius holds by construction of psf_radius.
  - Slice centers (forward) use ``floor(k + 0.5)`` for the same reason, and
    the same tie argument holds one level down: the entering/leaving tap sits
    at |k - k_center| = bp_psf_radius + 0.5 slices, so its trapezoid weight is
    clip(1/2 - W_p_r * bp_psf_radius, 0), zero whenever
    2 * bp_psf_radius >= 1 / W_p_r -- which is what bp_psf_radius is built to
    guarantee (it is ceil(ceil(voxels per detector row) / 2)).  Where an
    anisotropic detector breaks that guarantee the torch body is ALREADY
    truncating a nonzero tap, so the two forms differ there by exactly that
    pre-existing truncation, at a measure-zero tie.

Atomics and determinism (forward only): float atomic_add is commutative but
not associative, so the summation order over pixels and taps varies between
launches and the forward kernel is not bit-reproducible.  It is reproducible
to float rounding -- the value gates use rel 1e-5, and the repeat-consistency
test in tests/test_triton_cone.py measures the run-to-run spread directly.
Nothing downstream depends on bit reproducibility (the torch body's own view
batching already reorders the same sums).

The module imports WITHOUT triton (a CPU/MPS install has none), so the test
suite and the availability self-check can import it anywhere; only calling a
wrapper needs a working triton.
"""

import contextlib
import math

import torch

from ._utils import padded_kernel_width
from .cone_beam import (_cone_back_view_batch, _cone_horizontal_data,
                        _cone_vertical_affine)
from .projectors import compile_serialized

_F32 = torch.float32

try:
    import triton
    import triton.language as tl
except ImportError:                        # no triton (e.g. a macOS/CPU build)
    triton = None

    class _NoTritonLanguage:
        """Stand-in for ``triton.language`` so this module imports with no
        triton installed.  Only ``constexpr`` is read at import time.  Every
        other name is reached at kernel compile time, which never runs here."""
        constexpr = None

    tl = _NoTritonLanguage()


def _jit(fn):
    """Apply ``triton.jit`` where triton is importable.  Without triton the
    function is returned undecorated and is not callable as a kernel."""
    return fn if triton is None else triton.jit(fn)


def _tl_builtin(name):
    """Return the ``triton.language`` builtin ``name``, or None without triton.

    Several math builtins have moved between ``tl`` and ``tl.math`` across
    Triton versions, so the name is looked up in both and bound once here.
    """
    fn = getattr(tl, name, None)
    if fn is None:
        fn = getattr(getattr(tl, 'math', None), name, None)
    return fn


_tl_abs = _tl_builtin('abs')
_tl_floor = _tl_builtin('floor')
_tl_sqrt = _tl_builtin('sqrt')
# The tap loops want compile time trip counts, which tl.static_range gives.
# Plain range is the fallback for a toolchain without it.
_tap_range = _tl_builtin('static_range') or range

# The tile constants below were chosen on an H100.  Register pressure binds, since
# the inner loop holds the (BLOCK_P, BLOCK_L) accumulator and about six live tiles.
CONE_BACK_BLOCK_P = 16
CONE_BACK_BLOCK_L = 64
CONE_BACK_NUM_WARPS = 4
CONE_BACK_NUM_STAGES = 1
# A band or pixel subset smaller than this tile is padded, not shrunk.
CONE_BACK_MIN_TILE = 8
# The view batch this body asks for when the model's view_batch_size is None.
# The driver's transient budget may cap the realized batch below it.
CONE_BACK_VIEW_CHUNK = 128

CONE_FWD_BLOCK_P = 8
CONE_FWD_BLOCK_R = 128
CONE_FWD_NUM_WARPS = 8
CONE_FWD_NUM_STAGES = 1
CONE_FWD_MIN_TILE = 8
CONE_FWD_VIEW_CHUNK = 128

# Launch keys whose triton compilation has already finished in this process.  A key
# includes the runtime integers as well as the constexprs, because a false hit would
# race a compile.  All kernels share this set, so every key leads with its own name.
_COMPILED_LAUNCH_KEYS = set()


@_jit
def _cone_back_kernel(n_p_ptr, centers_ptr, w_p_c_ptr, weight_scale_ptr,
                      m0_ptr, w_p_r_ptr, pixel_mag_ptr, z_offset_ptr,
                      sino_ptr, out_ptr,
                      num_views, num_pixels, num_channels, num_rows, band_len,
                      slice_start, sino_view_stride,
                      delta_voxel_slice, slice_center, inv_sdd, num_rows_f,
                      PSF_RADIUS: tl.constexpr, COEFF_POWER: tl.constexpr,
                      BLOCK_P: tl.constexpr, BLOCK_L: tl.constexpr):
    """One program per (pixel block, slice chunk) of the output partial.

        out[p, l] = sum over views v, row taps tr, channel taps tc of
                    Wrow[v, p, l, tr] * Wchan[v, p, tc]
                    * sino[v, c(v, p) + tc, m(v, p, l) + tr]

    Both weight sets are formed in the kernel from the per (view, pixel)
    contract.  The pixel block is the fast grid axis, so programs that run
    together gather the same detector rows of the same view.

    Pixel lanes beyond ``num_pixels`` load zeroed contract values, which zeroes
    both weights, and their stores are masked.  ``band_len`` is the padded band
    the wrapper launches at.  Every sinogram address is clamped into the
    buffer, so a slice lane past the real band still reads inside the sinogram,
    and the wrapper returns only the real band columns.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    l_offs = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)     # (BLOCK_L,)
    p_mask = p_offs < num_pixels
    l_mask = l_offs < band_len
    tile_mask = p_mask[:, None] & l_mask[None, :]

    # k is the global slice index of each output column.  The affine row map and the
    # z chain are anchored at global slice 0, so a band only restricts the range of k.
    k = (slice_start + l_offs).to(tl.float32)                       # (BLOCK_L,)
    z_at_k = delta_voxel_slice * (k - slice_center)                 # (BLOCK_L,)

    acc = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
    for v in range(num_views):
        pix_base = v.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        w_p_c = tl.load(w_p_c_ptr + pix_base, mask=p_mask, other=0.0)
        weight_scale = tl.load(weight_scale_ptr + pix_base, mask=p_mask,
                               other=0.0)
        m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
        w_p_r = tl.load(w_p_r_ptr + pix_base, mask=p_mask, other=0.0)
        pixel_mag = tl.load(pixel_mag_ptr + pix_base, mask=p_mask, other=0.0)
        z_offset = tl.load(z_offset_ptr + v)

        # This is the direct form of the affine map, plus the cone angle
        # divisor from the z chain v = pixel_mag * z.
        m = m0[:, None] + w_p_r[:, None] * k[None, :]        # (BLOCK_P, BLOCK_L)
        m_center = _tl_floor(m + 0.5)                        # Ties carry zero weight.
        # The center is bounded before the integer conversions in the tap loop,
        # because float to int is undefined past the int32 range.
        m_center = tl.minimum(tl.maximum(m_center, -1.0 - PSF_RADIUS),
                              num_rows_f + PSF_RADIUS)
        v_det = pixel_mag[:, None] * (z_at_k[None, :] + z_offset)
        inv_cos_phi = _tl_sqrt(1.0 + (v_det * inv_sdd) * (v_det * inv_sdd))
        l_max_r = tl.minimum(w_p_r, 1.0)[:, None]
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tr in _tap_range(0, 2 * PSF_RADIUS + 1):
            m_tap = m_center + (tr - PSF_RADIUS)
            m_tap_i = m_tap.to(tl.int32)
            w_row = tl.maximum((w_p_r[:, None] + 1.0) / 2.0
                               - _tl_abs(m - m_tap), 0.0)
            # coeff_power is applied after the divisor.  Do not rearrange.
            w_row = tl.minimum(w_row, l_max_r) * inv_cos_phi
            w_row = tl.where((m_tap_i >= 0) & (m_tap_i < num_rows), w_row, 0.0)
            if COEFF_POWER == 2:
                w_row = w_row * w_row
            m_row = tl.minimum(tl.maximum(m_tap_i, 0), num_rows - 1)

            row_vals = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
            for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
                # The trapezoid rule zeroes the weight where the unclipped tap
                # left the detector, then clamps the index.
                n_tap = centers + (tc - PSF_RADIUS)
                w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, tl.minimum(w_p_c, 1.0)) * weight_scale
                w_chan = tl.where((n_tap >= 0) & (n_tap < num_channels),
                                  w_chan, 0.0)
                if COEFF_POWER == 2:
                    w_chan = w_chan * w_chan
                n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
                # The views are channel major, so the row axis is contiguous
                # and a tile's gather is a near unit stride read.
                chan_ptr = sino_view_ptr + n_chan.to(tl.int64) * num_rows
                vals = tl.load(chan_ptr[:, None] + m_row, mask=tile_mask,
                               other=0.0)
                row_vals = row_vals + w_chan[:, None] * vals
            acc = acc + w_row * row_vals

    out_ptrs = out_ptr + p_offs.to(tl.int64)[:, None] * band_len + l_offs[None, :]
    tl.store(out_ptrs, acc, mask=tile_mask)


def _next_pow2(n):
    return 1 << (max(int(n), 1) - 1).bit_length()


def _tile_size(cap, extent, min_tile):
    """Return the power of two tile size for an axis of length ``extent``.

    The result is the cap, shrunk when the axis is smaller but never below
    ``min_tile``, so a small axis does not launch a mostly padded tile.
    """
    return max(int(min_tile), min(int(cap), _next_pow2(extent)))


@torch.compiler.disable
def _cone_back_view_batch_triton(sino_batch, pixel_indices, view_params_batch,
                                 num_rows_r, num_channels, num_recon_rows,
                                 num_recon_cols, num_slices, delta_voxel,
                                 delta_voxel_row, delta_voxel_slice,
                                 delta_det_channel, delta_det_row,
                                 det_channel_offset, det_row_offset,
                                 recon_slice_offset, magnification,
                                 source_detector_dist, use_curved_detector,
                                 psf_radius, bp_psf_radius, coeff_power=1,
                                 slice_start=0, band_slices=None, plan=None):
    """Triton cone back body, replacing
    :func:`mbirtorch.cone_beam._cone_back_view_batch`.

    It has the same signature and the same (P, band) return, written fresh
    each call so the driver may accumulate into it in place.  The band is
    rounded up to a multiple of 16 for the launch, and the return is the real
    band slice, which is a strided view when the band is not such a multiple.
    This body must stay eager, which the ``torch.compiler.disable`` decorator
    and the ``_mbirtorch_no_compile`` marker below enforce for the two kinds of
    caller.  ``bp_psf_radius`` and ``plan`` are ignored.
    """
    if triton is None:
        raise RuntimeError('the Triton cone back body was called without '
                           'triton installed; the selection in '
                           'ConeBeamModel._view_batch_bodies should have kept '
                           'the torch body (see kernel_availability).')
    # The kernel has static branches for powers 1 and 2 only.
    if coeff_power not in (1, 2):
        return _cone_back_view_batch(
            sino_batch, pixel_indices, view_params_batch, num_rows_r,
            num_channels, num_recon_rows, num_recon_cols, num_slices,
            delta_voxel, delta_voxel_row, delta_voxel_slice, delta_det_channel,
            delta_det_row, det_channel_offset, det_row_offset,
            recon_slice_offset, magnification, source_detector_dist,
            use_curved_detector, psf_radius, bp_psf_radius,
            coeff_power=coeff_power, slice_start=slice_start,
            band_slices=band_slices, plan=plan)

    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, w_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    m0, w_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    num_views, num_pixels = n_p.shape
    band_len = int(num_slices if band_slices is None else band_slices)
    # The kernel is launched at a band rounded up to a multiple of 16.  Every
    # use of the band argument takes this value.
    launch_band = padded_kernel_width(band_len)
    # The views are made channel major, so the per tile gather is contiguous.  This
    # copy is not padded, because the kernel clamps every sinogram address it forms.
    sino_t = sino_batch.permute(0, 2, 1).contiguous()
    contract = [t.contiguous() for t in (n_p, centers, w_p_c, weight_scale,
                                         m0, w_p_r, pixel_mag, z_offset)]
    out = torch.empty((num_pixels, launch_band), dtype=_F32,
                      device=sino_batch.device)

    block_p = _tile_size(CONE_BACK_BLOCK_P, num_pixels, CONE_BACK_MIN_TILE)
    block_l = _tile_size(CONE_BACK_BLOCK_L, launch_band, CONE_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_band // block_l))
    inv_sdd = (0.0 if math.isinf(float(source_detector_dist))
               else 1.0 / float(source_detector_dist))
    launch_key = ('back', sino_batch.device.index, int(psf_radius),
                  int(coeff_power), block_p, block_l,
                  int(num_views), int(num_pixels), int(num_channels),
                  int(num_rows_r), launch_band, int(slice_start))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The device context is required for correctness.  A Triton launch targets the
    # launching thread's current CUDA device and stream, while every torch op targets
    # the tensor's device.  The drivers call this wrapper from worker threads whose
    # current device is 0, so without this bracket the launch lands on device 0.
    with torch.cuda.device(sino_batch.device), guard:
        _cone_back_kernel[grid](
            *contract, sino_t, out,
            int(num_views), int(num_pixels), int(num_channels),
            int(num_rows_r), launch_band, int(slice_start),
            int(num_channels) * int(num_rows_r),
            float(delta_voxel_slice), (int(num_slices) - 1) / 2.0, inv_sdd,
            float(num_rows_r),
            PSF_RADIUS=int(psf_radius), COEFF_POWER=int(coeff_power),
            BLOCK_P=block_p, BLOCK_L=block_l,
            num_warps=CONE_BACK_NUM_WARPS, num_stages=CONE_BACK_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_band == band_len:
        return out
    # The padded columns are sliced off.  The result is a strided view whose
    # rows are each still contiguous.
    return out[:, :band_len]


# The driver reads this marker in maybe_compile.
_cone_back_view_batch_triton._mbirtorch_no_compile = True


def _cone_back_view_batch_cost(num_pixels, num_band_rows, args):
    """Return the bytes resident per view in one back kernel batch, and this
    kernel's nominal view chunk.

    The driver reads this through the ``_view_batch_cost`` attribute.  One view
    holds the two precomputed geometry tables at 48 bytes per (view, pixel),
    plus the channel major copy of its sinogram plane.  Tensors whose size does
    not depend on the batch are not charged.
    """
    plane_bytes = 4 * int(args['num_channels']) * int(num_band_rows)
    return 48 * int(num_pixels) + plane_bytes, CONE_BACK_VIEW_CHUNK


_cone_back_view_batch_triton._view_batch_cost = _cone_back_view_batch_cost


@_jit
def _cone_forward_kernel(n_p_ptr, centers_ptr, w_p_c_ptr, weight_scale_ptr,
                         m0_ptr, w_p_r_ptr, pixel_mag_ptr, z_offset_ptr,
                         values_ptr, out_ptr,
                         num_pixels, num_channels, num_rows, band_len,
                         slice_start, out_view_stride,
                         delta_voxel_slice, slice_center, inv_sdd,
                         k_center_lo, k_center_hi,
                         PSF_RADIUS: tl.constexpr, BP_PSF_RADIUS: tl.constexpr,
                         BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr):
    """One program per (pixel block, row chunk, view) of the sinogram.

        out[v, c, m] += sum over pixels p, channel taps tc, slice taps tk of
                        Wchan[v, p, tc] * Wslice[v, p, m, tk] / cos_phi
                        * values[p, k(v, p, m) + tk]

    The detector side vertical fan runs first, then the horizontal fan
    scatter.  The vertical map is the inverse of the affine map,
    ``k = (m - m0) / W_p_r``.

    The view is a grid axis because each view writes its own output plane.
    The pixel block is the fast axis, so programs that run together read the
    same rows of ``values`` and hit neighboring channels.  The view axis is
    last, and it stays well under the CUDA limit of 65535 on grid dims 1 and 2.

    The output is channel major (Vb, C, R), which puts the row axis on the
    contiguous stride.  The wrapper transposes the view on return.

    Pixel lanes beyond ``num_pixels`` have their atomics masked off, so their
    contract values may be any finite number.  ``num_rows`` is the padded row
    count the wrapper launches at.  Row lanes past the real rows gather with a
    clamped and masked index, and their atomics land in extra output rows that
    the wrapper slices off.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)     # (BLOCK_R,)
    v = tl.program_id(2)
    p_mask = p_offs < num_pixels
    r_mask = r_offs < num_rows
    tile_mask = p_mask[:, None] & r_mask[None, :]

    pix_base = v.to(tl.int64) * num_pixels + p_offs
    n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
    centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
    w_p_c = tl.load(w_p_c_ptr + pix_base, mask=p_mask, other=0.0)
    weight_scale = tl.load(weight_scale_ptr + pix_base, mask=p_mask, other=0.0)
    m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
    # The filler is 1.0 rather than 0.0, because a padded lane divides by its slope
    # and 0/0 would give a NaN in the float to int conversion below.
    w_p_r = tl.load(w_p_r_ptr + pix_base, mask=p_mask, other=1.0)
    pixel_mag = tl.load(pixel_mag_ptr + pix_base, mask=p_mask, other=0.0)
    z_offset = tl.load(z_offset_ptr + v)

    # Vertical fan, detector side, from rows to fractional slice indices.
    slope = w_p_r[:, None]                                          # (BLOCK_P, 1)
    k_m = (r_offs.to(tl.float32)[None, :] - m0[:, None]) / slope
    k_center = _tl_floor(k_m + 0.5)                          # Ties carry zero weight.
    # The center is bounded before the integer conversions in the tap loop, because
    # float to int is undefined past the int32 range.
    k_center = tl.minimum(tl.maximum(k_center, k_center_lo), k_center_hi)
    # m_p is the offset of the center slice from the row, in row units.
    m_p = slope * (k_center - k_m)
    l_max_r = tl.minimum(w_p_r, 1.0)[:, None]

    det_col = tl.zeros((BLOCK_P, BLOCK_R), dtype=tl.float32)
    for tk in _tap_range(0, 2 * BP_PSF_RADIUS + 1):
        k_off = tk - BP_PSF_RADIUS
        k_ind = k_center + k_off
        k_ind_i = k_ind.to(tl.int32)
        w_slice = tl.maximum((slope + 1.0) / 2.0 - _tl_abs(m_p + slope * k_off),
                             0.0)
        w_slice = tl.minimum(w_slice, l_max_r)
        # The weight is zeroed where the tap left the band, and the index is then
        # clamped.  The z anchor stays on the full slice count.
        in_band = (k_ind_i >= slice_start) & (k_ind_i < slice_start + band_len)
        w_slice = tl.where(in_band, w_slice, 0.0)
        # The cone angle divisor belongs to the tapped slice, not to the row.
        # The z chain is v = pixel_mag * z, as in the back kernel.
        v_det = pixel_mag[:, None] * (delta_voxel_slice * (k_ind - slice_center)
                                      + z_offset)
        inv_cos_phi = _tl_sqrt(1.0 + (v_det * inv_sdd) * (v_det * inv_sdd))
        # The clamp keeps the index legal.  Carrying in_band into the load mask
        # saves the read of a tap that already carries zero weight.
        k_local = tl.minimum(tl.maximum(k_ind_i - slice_start, 0), band_len - 1)
        vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * band_len
                       + k_local, mask=tile_mask & in_band, other=0.0)
        det_col = det_col + w_slice * (vals * inv_cos_phi)

    # Horizontal fan scatter, one atomic add per tap.
    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
        # The trapezoid rule again.  Taps outside the detector drop out of the
        # atomic mask instead of being added as zeros.
        n_tap = centers + (tc - PSF_RADIUS)
        w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                            - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
        w_chan = tl.minimum(w_chan, tl.minimum(w_p_c, 1.0)) * weight_scale
        n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
        out_ptrs = (out_view_ptr + n_chan.to(tl.int64)[:, None] * num_rows
                    + r_offs[None, :])
        tl.atomic_add(out_ptrs, w_chan[:, None] * det_col,
                      mask=tile_mask & ((n_tap >= 0)
                                        & (n_tap < num_channels))[:, None])


@torch.compiler.disable
def _cone_forward_view_batch_triton(values, pixel_indices, view_params_batch,
                                    num_rows_r, num_channels, num_recon_rows,
                                    num_recon_cols, num_slices, delta_voxel,
                                    delta_voxel_row, delta_voxel_slice,
                                    delta_det_channel, delta_det_row,
                                    det_channel_offset, det_row_offset,
                                    recon_slice_offset, magnification,
                                    source_detector_dist, use_curved_detector,
                                    psf_radius, bp_psf_radius, slice_start=0,
                                    plan=None):
    """Triton cone forward body, replacing
    :func:`mbirtorch.cone_beam._cone_forward_view_batch`.

    It has the same signature and the same (Vb, R, C) return, zeroed each call
    because the kernel accumulates into it with atomics.  The detector row
    count is rounded up to a multiple of 16 for the launch, and the return is
    the real row slice.  ``values`` is (P, L).  When L is less than num_slices
    it is the slice band starting at ``slice_start``, and the z geometry stays
    anchored on the full num_slices center, so summing the outputs over a
    tiling of the slice axis reproduces the unbanded projection.  ``plan`` is
    ignored.
    """
    if triton is None:
        raise RuntimeError('the Triton cone forward body was called without '
                           'triton installed; the selection in '
                           'ConeBeamModel._view_batch_bodies should have kept '
                           'the torch body (see kernel_availability).')
    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, w_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    m0, w_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    num_views, num_pixels = n_p.shape
    band_len = int(values.shape[1])
    # The kernel is launched at a detector row count rounded up to a multiple of 16.
    # The geometry builders above keep the real row count.
    launch_rows = padded_kernel_width(int(num_rows_r))
    values = values.contiguous()
    contract = [t.contiguous() for t in (n_p, centers, w_p_c, weight_scale,
                                         m0, w_p_r, pixel_mag, z_offset)]
    # The output is channel major and zeroed, because the atomics accumulate
    # into it.  The return transposes each view.
    out = torch.zeros((num_views, num_channels, launch_rows), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(CONE_FWD_BLOCK_P, num_pixels, CONE_FWD_MIN_TILE)
    block_r = _tile_size(CONE_FWD_BLOCK_R, launch_rows, CONE_FWD_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_rows // block_r), num_views)
    inv_sdd = (0.0 if math.isinf(float(source_detector_dist))
               else 1.0 / float(source_detector_dist))
    # These are the bounds the kernel clamps the slice center to.  They are tight: at
    # the low bound the highest tap is slice_start - 1, and at the high bound the
    # lowest tap is slice_start + band_len.
    bp = int(bp_psf_radius)
    k_center_lo = float(int(slice_start) - bp - 1)
    k_center_hi = float(int(slice_start) + band_len + bp)
    launch_key = ('fwd', values.device.index, int(psf_radius), bp, block_p,
                  block_r, int(num_views),
                  int(num_pixels), int(num_channels), launch_rows,
                  band_len, int(slice_start))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch runs on the tensors' own device, and the device leads the
    # launch key.
    with torch.cuda.device(values.device), guard:
        _cone_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), launch_rows, band_len,
            int(slice_start), int(num_channels) * launch_rows,
            float(delta_voxel_slice), (int(num_slices) - 1) / 2.0, inv_sdd,
            k_center_lo, k_center_hi,
            PSF_RADIUS=int(psf_radius), BP_PSF_RADIUS=bp,
            BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=CONE_FWD_NUM_WARPS, num_stages=CONE_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_rows == int(num_rows_r):
        return out.permute(0, 2, 1)
    # The extra detector rows are sliced off before the transpose.
    return out[:, :, :int(num_rows_r)].permute(0, 2, 1)


# The driver reads this marker in maybe_compile.
_cone_forward_view_batch_triton._mbirtorch_no_compile = True


def _cone_forward_view_batch_cost(num_pixels, band_len, args):
    """Return the bytes resident per view in one forward kernel batch, and
    this kernel's nominal view chunk.

    One view holds the same 48 bytes per (view, pixel) of precomputed geometry
    as the back kernel, plus its zeroed channel major output plane.  That plane
    spans the full detector rows at the padded row count, whatever slice band
    the values carry.
    """
    plane_rows = padded_kernel_width(int(args['num_rows_r']))
    plane_bytes = 4 * int(args['num_channels']) * plane_rows
    return 48 * int(num_pixels) + plane_bytes, CONE_FWD_VIEW_CHUNK


_cone_forward_view_batch_triton._view_batch_cost = \
    _cone_forward_view_batch_cost
