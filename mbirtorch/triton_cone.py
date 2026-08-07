"""The hand-written Triton cone-beam kernels: increments K1 and K3 of the
Phase 5 kernel plan -- the cone BACK and FORWARD projections as alternative
view-batch bodies.

A kernel here is an ALTERNATIVE BODY, never a new driver.  Each wrapper below
has the same signature as the torch body it replaces
(:func:`mbirtorch.cone_beam._cone_back_view_batch`,
:func:`mbirtorch.cone_beam._cone_forward_view_batch`), so the driver's
view-range loop, its lazy assembly, the banded seams (``slice_start``,
``band_slices``, ``dev_index``) and the ``plan`` slot all pass through
unchanged, and the torch bodies stay compiled in everywhere as the value
reference and the fallback -- exactly as mbirjax keeps its XLA path beside
pallas.

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
the gradient path, 1e-4 at coeff_power=2, the mbirjax rounding carve-out):

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
        """Stand-in for ``triton.language`` so this module still IMPORTS with
        no triton installed.  Only ``constexpr`` is touched at import time (it
        appears in the kernel's annotations, which python evaluates eagerly);
        every other name is reached at kernel COMPILE time, which never
        happens on such a machine."""
        constexpr = None

    tl = _NoTritonLanguage()


def _jit(fn):
    """``triton.jit`` where triton is importable, else the undecorated
    function -- a marker, never callable as a kernel (the wrapper below raises
    a clear error first)."""
    return fn if triton is None else triton.jit(fn)


def _tl_builtin(name):
    """The ``triton.language`` builtin ``name``, looked up in ``tl`` and then
    in ``tl.math``: several math builtins have moved between the two across
    Triton versions, and a kernel body that names one directly would compile
    on some toolchains and not others.  Bound once here, the kernel calls the
    alias and stays version-agnostic (Triton resolves a global alias to a
    builtin exactly as it resolves the dotted name).  None without triton."""
    fn = getattr(tl, name, None)
    if fn is None:
        fn = getattr(getattr(tl, 'math', None), name, None)
    return fn


_tl_abs = _tl_builtin('abs')
_tl_floor = _tl_builtin('floor')
_tl_sqrt = _tl_builtin('sqrt')
# The tap loops want COMPILE-TIME trip counts (the design's static-shapes
# rule): tl.static_range unrolls them in python at trace time.  Plain range is
# the correct-but-unrolled fallback for a toolchain without it -- same values,
# a dynamic scf.for that LLVM will usually unroll anyway, since the psf radius
# is a constexpr and the bounds are therefore IR constants.
_tap_range = _tl_builtin('static_range') or range

# ── H100 starting constants (the K2 sweep axes) ──────────────────────────────
# The pallas cone back kernel ran one PIXEL per program with a 128-slice
# register tile (CONE_LC=128, num_warps=1).  Triton programs are tiles in both
# axes, so the same accumulator budget buys a (pixel, slice) rectangle instead,
# and the binding constraint is registers: the inner loop holds the (BLOCK_P,
# BLOCK_L) accumulator plus ~6 live tiles of the same shape (m, inv_cos_phi,
# the row centers, the row weight, the row partial, the gathered values), so
# 32x64 at 4 warps is ~96 registers per thread -- the largest tile that stays
# clear of spilling by inspection.  64x128 (the design's "BLOCK_P 64ish, LC
# 128" reading) is ~8x that and would spill hard; K2 measures the rectangle
# rather than reasoning about it.  Sweep axes for K2: BLOCK_P, BLOCK_L,
# NUM_WARPS, NUM_STAGES, and the driver's view batch.
CONE_BACK_BLOCK_P = 16
CONE_BACK_BLOCK_L = 64
CONE_BACK_NUM_WARPS = 4
# 1 stage = no software pipelining: the view loop is gather-bound, not
# dot-bound, and extra stages buy latency hiding only at more register
# pressure, which is already the limiter above.
CONE_BACK_NUM_STAGES = 1
# The smallest tile worth launching: a band or pixel subset below this pads
# rather than shrinking further.
CONE_BACK_MIN_TILE = 8

# ── H100 starting constants for the FORWARD kernel (the K4-era sweep axes) ───
# Deliberately conservative, and deliberately the back kernel's rectangle: the
# forward program holds the same class of live tiles at (BLOCK_P, BLOCK_R) --
# the detector-column accumulator plus ~5 companions (k_m, the slice center,
# the tap weight, the cone divisor, the gathered values) -- so the register
# argument that picked 16x64 there transfers here unchanged.  The forward has
# one extra pressure the back does not: its atomic phase holds the accumulator
# live across the channel tap loop.  Sweep axes: BLOCK_P, BLOCK_R, NUM_WARPS,
# NUM_STAGES, and the driver's view batch.  The one number the sweep should
# not assume transfers is the pallas forward's num_warps=2 win (measured for a
# segment-walk kernel with a different tile shape, not for this one).
CONE_FWD_BLOCK_P = 8
CONE_FWD_BLOCK_R = 128
CONE_FWD_NUM_WARPS = 8
# 1 stage = no software pipelining: the tap loop is gather- and atomic-bound
# rather than dot-bound, so extra stages buy latency hiding only at more
# register pressure.
CONE_FWD_NUM_STAGES = 1
CONE_FWD_MIN_TILE = 8

# Launch keys whose triton compilation has already completed in this process.
# The compile lock must cover the FIRST launch of each configuration (triton
# compiles at launch, outside torch.compile, and races the same launcher state
# the lock exists for) but must NOT cover the steady state, where taking it
# would serialize the per-device threads.  The key deliberately over-keys on
# the runtime ints as well as the constexprs: a missed key only costs one
# extra uncontended lock acquisition, while a false hit would race a compile.
# Both kernels share this set, so every key leads with the kernel's name --
# otherwise a forward key of matching ints would mark a back configuration
# compiled (a false hit is the one error mode the over-keying exists to avoid).
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
    """One program per (pixel block, slice chunk) of the output partial:

        out[p, l] = sum over views v, row taps tr, channel taps tc of
                    Wrow[v, p, l, tr] * Wchan[v, p, tc]
                    * sino[v, c(v, p) + tc, m(v, p, l) + tr]

    with both weight sets formed in-kernel from the per-(view, pixel) contract
    (the module docstring's two builders).  The pixel block is the FAST grid
    axis so that concurrently scheduled programs gather from the same detector
    rows of the same view -- the L2 residency the pallas grid ordering bought
    for these transaction-bound gathers.

    Pixels beyond ``num_pixels`` and slices beyond ``band_len`` ride as padded
    lanes: their loaded contract values are zeroed, which zeroes both tap
    weights, and their stores are masked (the poison-the-padding rule).
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    l_offs = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)     # (BLOCK_L,)
    p_mask = p_offs < num_pixels
    l_mask = l_offs < band_len
    tile_mask = p_mask[:, None] & l_mask[None, :]

    # GLOBAL slice index of each output column: the affine row map and the z
    # chain are both anchored at global slice 0, so a band only restricts the
    # range of k (never the geometry).
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

        # Vertical fan: the DIRECT form of the shared affine map, exactly
        # affine in the slice index, plus the cone-angle divisor from the same
        # z chain the torch body uses (v = pixel_mag * z).
        m = m0[:, None] + w_p_r[:, None] * k[None, :]        # (BLOCK_P, BLOCK_L)
        m_center = _tl_floor(m + 0.5)                        # ties are inert
        # Bound the center before the integer conversions in the tap loop: a
        # degenerate geometry (a voxel approaching the source plane) can send
        # m far outside the panel, and float-to-int is undefined past the
        # int32 range.  Provably inert: a tap can carry weight only where
        # |m - tap| < (W_p_r + 1)/2 <= psf_radius + 0.5 at a tap inside the
        # panel, which keeps every contributing center well within these
        # bounds; anything clamped was masked to zero either way.
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
            # coeff_power is applied AFTER the divisor (the mbirjax rule).
            w_row = tl.minimum(w_row, l_max_r) * inv_cos_phi
            w_row = tl.where((m_tap_i >= 0) & (m_tap_i < num_rows), w_row, 0.0)
            if COEFF_POWER == 2:
                w_row = w_row * w_row
            m_row = tl.minimum(tl.maximum(m_tap_i, 0), num_rows - 1)

            row_vals = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
            for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
                # The horizontal_fan trapezoid rule, in-kernel: zero the
                # weight where the unclipped tap left the detector, then clamp
                # the index (the zero-and-clamp convention).
                n_tap = centers + (tc - PSF_RADIUS)
                w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, tl.minimum(w_p_c, 1.0)) * weight_scale
                w_chan = tl.where((n_tap >= 0) & (n_tap < num_channels),
                                  w_chan, 0.0)
                if COEFF_POWER == 2:
                    w_chan = w_chan * w_chan
                n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
                # Channel-major views: the slice axis walks the CONTIGUOUS row
                # axis, so a tile's gather is a near-unit-stride read.
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
    """The power-of-two tile for an axis of length ``extent``: the pinned cap,
    shrunk (never below ``min_tile``) when the axis is smaller, so a small
    band, row count, or pixel subset does not launch a mostly-padded tile.
    Mirrors the pallas ``lc = min(CONE_LC, next_pow2(band))`` rule; the handful
    of resulting shape variants each compile once."""
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
    """The Triton cone back body: a drop-in replacement for
    :func:`mbirtorch.cone_beam._cone_back_view_batch` (same signature, same
    (P, band) return, freshly written each call so the driver may accumulate
    into it in place).

    Eager python by construction, declared twice over because the two
    mechanisms cover different callers: ``torch.compiler.disable`` keeps
    dynamo out when a compiled region CALLS this body, and the
    ``_mbirtorch_no_compile`` marker set below keeps the driver's
    ``maybe_compile`` from compiling it DIRECTLY (torch.compile unwraps the
    disable decorator and would trace the launch anyway).  The per-view
    builders run ONCE per call, outside every loop (the hoisted-builders rule:
    a per-chunk rebuild is the bench artifact that hid 3.54x-vs-9.07x in the
    pallas campaign).

    ``bp_psf_radius`` is accepted and unused, as in the torch body: the back
    path's row taps use ``psf_radius`` (the forward path is the one that
    gathers voxels per detector row).  ``plan`` is likewise accepted and
    ignored -- the sorted/CSR stream slot, not yet built.
    """
    if triton is None:
        raise RuntimeError('the Triton cone back body was called without '
                           'triton installed; the selection in '
                           'ConeBeamModel._view_batch_bodies should have kept '
                           'the torch body (see kernel_availability).')
    # Powers other than 1 and 2 are outside the kernel's static branch (and
    # outside every caller in the package): delegate rather than diverge.
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
    # Channel-major views, as in the torch body: the kernel's per-tile gather
    # walks the slice axis, whose row index is contiguous in this layout.
    sino_t = sino_batch.permute(0, 2, 1).contiguous()
    contract = [t.contiguous() for t in (n_p, centers, w_p_c, weight_scale,
                                         m0, w_p_r, pixel_mag, z_offset)]
    out = torch.empty((num_pixels, band_len), dtype=_F32,
                      device=sino_batch.device)

    block_p = _tile_size(CONE_BACK_BLOCK_P, num_pixels, CONE_BACK_MIN_TILE)
    block_l = _tile_size(CONE_BACK_BLOCK_L, band_len, CONE_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-band_len // block_l))
    inv_sdd = (0.0 if math.isinf(float(source_detector_dist))
               else 1.0 / float(source_detector_dist))
    launch_key = ('back', int(psf_radius), int(coeff_power), block_p, block_l,
                  int(num_views), int(num_pixels), int(num_channels),
                  int(num_rows_r), band_len, int(slice_start))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    with guard:
        _cone_back_kernel[grid](
            *contract, sino_t, out,
            int(num_views), int(num_pixels), int(num_channels),
            int(num_rows_r), band_len, int(slice_start),
            int(num_channels) * int(num_rows_r),
            float(delta_voxel_slice), (int(num_slices) - 1) / 2.0, inv_sdd,
            float(num_rows_r),
            PSF_RADIUS=int(psf_radius), COEFF_POWER=int(coeff_power),
            BLOCK_P=block_p, BLOCK_L=block_l,
            num_warps=CONE_BACK_NUM_WARPS, num_stages=CONE_BACK_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    return out


# See the wrapper's docstring: the driver reads this marker in maybe_compile.
_cone_back_view_batch_triton._mbirtorch_no_compile = True


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
    """One program per (pixel block, row chunk, view) of the sinogram:

        out[v, c, m] += sum over pixels p, channel taps tc, slice taps tk of
                        Wchan[v, p, tc] * Wslice[v, p, m, tk] / cos_phi
                        * values[p, k(v, p, m) + tk]

    -- the DETECTOR-side vertical fan (for each detector row, which voxels
    project onto it) followed by the horizontal fan scatter, exactly as in
    :func:`mbirtorch.cone_beam._cone_forward_view_batch`.  The vertical map is
    the INVERSE of the shared affine, ``k = (m - m0) / W_p_r``, so the two
    directions consume the two algebraic forms of the one geometry bridge.

    Grid choice (three axes, justified against the two the design offered):
    the VIEW axis has to be a grid axis rather than an in-program loop --
    unlike the back kernel, whose view loop reduces into one register
    accumulator, the forward writes a separate output plane per view, so a
    view loop would only serialize.  That leaves (pixel block, row chunk) for
    the other two, and both are grid axes rather than one being a loop so that
    a small pixel subset (the fine VCD tail) still fills the machine through
    the row axis.  The PIXEL block is the fast axis: concurrently scheduled
    programs then read the same rows of ``values`` and hit neighbouring
    channels of the same view with their atomics, since neighbouring pixels
    project to neighbouring channels -- the L2 residency argument the pallas
    grid ordering bought, applied to the scatter side.  The view axis is last
    because it is the only one that is small; it does inherit CUDA's 65535
    limit on grid dims 1 and 2, which the driver's view batch (default 64,
    capped further by the transient budget) is nowhere near.

    The output is CHANNEL-MAJOR (Vb, C, R), the layout fan_forward_batch also
    accumulates in: it puts the row axis -- the kernel's vector axis -- on the
    contiguous stride, so one tile row's atomics land on consecutive addresses
    instead of striding by C.  The wrapper transposes the view on return, as
    the torch body does.

    Pixels beyond ``num_pixels`` and rows beyond ``num_rows`` ride as padded
    lanes: their atomics are masked off entirely (the poison-the-padding rule),
    which is why a padded lane's contract values may be anything finite.
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
    # A padded lane DIVIDES by its slope, so its filler is 1.0 rather than the
    # 0.0 every other load uses: 0/0 is a NaN, and a NaN survives the clamp
    # below into an undefined float-to-int conversion.  1.0 keeps the lane's
    # arithmetic finite and ordinary; its atomics are masked off regardless.
    w_p_r = tl.load(w_p_r_ptr + pix_base, mask=p_mask, other=1.0)
    pixel_mag = tl.load(pixel_mag_ptr + pix_base, mask=p_mask, other=0.0)
    z_offset = tl.load(z_offset_ptr + v)

    # ── vertical fan, detector side: rows -> fractional slice indices ────────
    slope = w_p_r[:, None]                                          # (BLOCK_P, 1)
    k_m = (r_offs.to(tl.float32)[None, :] - m0[:, None]) / slope
    k_center = _tl_floor(k_m + 0.5)                          # ties are inert
    # Bound the center before the integer conversions in the tap loop: a
    # near-degenerate slope sends k_m far outside the volume, and
    # float-to-int is undefined past the int32 range.  Provably inert: a tap
    # can contribute only from inside the band, and every tap of a center
    # outside these bounds lands outside the band either way, so the clamp
    # only moves centers whose whole tap window was already zero-weighted.
    k_center = tl.minimum(tl.maximum(k_center, k_center_lo), k_center_hi)
    # The offset of the center slice from the row, in ROW units -- the torch
    # body's m_p, hoisted out of the tap loop in its own float order.
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
        # Zero the weight where the tap left the BAND, then clamp the index
        # (the zero-and-clamp convention): the z anchor stays on the full slice
        # count, so a band only restricts which taps contribute.
        in_band = (k_ind_i >= slice_start) & (k_ind_i < slice_start + band_len)
        w_slice = tl.where(in_band, w_slice, 0.0)
        # The cone-angle divisor belongs to the TAPPED slice, not to the row:
        # the torch body scales the whole cylinder by 1/cos(phi) before it
        # gathers.  Same z chain as the back kernel (v = pixel_mag * z).
        v_det = pixel_mag[:, None] * (delta_voxel_slice * (k_ind - slice_center)
                                      + z_offset)
        inv_cos_phi = _tl_sqrt(1.0 + (v_det * inv_sdd) * (v_det * inv_sdd))
        # The clamp keeps the index legal; carrying in_band into the load mask
        # as well is the optimization, not the contract -- an out-of-band tap
        # is already zero-weighted, and this only saves the read (a whole tile
        # of them, at the edges of a narrow shard band).
        k_local = tl.minimum(tl.maximum(k_ind_i - slice_start, 0), band_len - 1)
        vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * band_len
                       + k_local, mask=tile_mask & in_band, other=0.0)
        det_col = det_col + w_slice * (vals * inv_cos_phi)

    # ── horizontal fan scatter: the plain per-tap atomic form ────────────────
    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
        # The horizontal_fan trapezoid rule, in-kernel (as in the back kernel);
        # here the out-of-detector taps drop out of the atomic MASK rather than
        # being added as zeros, which is the same value and one less atomic.
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
    """The Triton cone forward body: a drop-in replacement for
    :func:`mbirtorch.cone_beam._cone_forward_view_batch` (same signature, same
    (Vb, R, C) return, freshly zeroed each call because the kernel accumulates
    into it with atomics).

    Eager python by construction and declared twice over, for the two reasons
    :func:`_cone_back_view_batch_triton` spells out, and with the same hoisted
    builders: the hfan contract and the vertical affine are built ONCE per
    call, outside every loop.

    ``values`` is (P, L); when L < num_slices it is the slice BAND starting at
    ``slice_start`` and the z geometry stays anchored on the full num_slices
    center, so summing the outputs over a tiling of the slice axis reproduces
    the unbanded projection.  ``plan`` is accepted and ignored -- the
    sorted/CSR stream slot, not yet built.
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
    values = values.contiguous()
    contract = [t.contiguous() for t in (n_p, centers, w_p_c, weight_scale,
                                         m0, w_p_r, pixel_mag, z_offset)]
    # Channel-major, zeroed: the atomics accumulate, and the return transposes
    # the view exactly as the torch body transposes fan_forward_batch's.
    out = torch.zeros((num_views, num_channels, num_rows_r), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(CONE_FWD_BLOCK_P, num_pixels, CONE_FWD_MIN_TILE)
    block_r = _tile_size(CONE_FWD_BLOCK_R, num_rows_r, CONE_FWD_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-num_rows_r // block_r), num_views)
    inv_sdd = (0.0 if math.isinf(float(source_detector_dist))
               else 1.0 / float(source_detector_dist))
    # The inert bounds on the slice center (see the kernel).  They are exactly
    # tight, which is what makes them inert: at the low bound the HIGHEST tap
    # is slice_start - 1, one short of the band, and at the high bound the
    # LOWEST tap is slice_start + band_len, one past it -- so a clamped center
    # contributes nothing, and so did the center it replaced.
    bp = int(bp_psf_radius)
    k_center_lo = float(int(slice_start) - bp - 1)
    k_center_hi = float(int(slice_start) + band_len + bp)
    launch_key = ('fwd', int(psf_radius), bp, block_p, block_r, int(num_views),
                  int(num_pixels), int(num_channels), int(num_rows_r),
                  band_len, int(slice_start))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    with guard:
        _cone_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), int(num_rows_r), band_len,
            int(slice_start), int(num_channels) * int(num_rows_r),
            float(delta_voxel_slice), (int(num_slices) - 1) / 2.0, inv_sdd,
            k_center_lo, k_center_hi,
            PSF_RADIUS=int(psf_radius), BP_PSF_RADIUS=bp,
            BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=CONE_FWD_NUM_WARPS, num_stages=CONE_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    return out.permute(0, 2, 1)


# See the back wrapper's docstring: the driver reads this marker in
# maybe_compile.
_cone_forward_view_batch_triton._mbirtorch_no_compile = True
