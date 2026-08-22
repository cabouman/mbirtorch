"""The hand-written Triton multiaxis-parallel kernels -- the multiaxis BACK and
FORWARD projections as alternative view-batch bodies.

A kernel here is an ALTERNATIVE BODY, never a new driver, exactly as in
triton_cone.py and triton_parallel.py.  Each wrapper below has the same
signature as the torch body it replaces
(:func:`mbirtorch.multiaxis_parallel._multiaxis_back_view_batch`,
:func:`mbirtorch.multiaxis_parallel._multiaxis_forward_view_batch`), so the
driver's view-range loop, its lazy assembly, the banded seams (``slice_start``,
``band_slices``) and the ``plan`` slot would all pass through unchanged, and
the torch bodies stay the value reference and the fallback.

MultiAxisParallelModel's ``_view_batch_bodies`` now selects these bodies
wherever their availability gates pass (``multiaxis_back_kernel_usable`` and
``multiaxis_forward_kernel_usable`` in kernel_availability.py), one direction
at a time, and keeps the torch body where a gate says no.  The gates state that
a kernel reproduces its torch body on the device that will run it; no composed
performance measurement has been made for this geometry, and the constants
below are still the adopted ones described further down.

Both kernels stand on the two eager builders the torch bodies already use, and
the wrappers call them rather than reimplementing the geometry:
:func:`mbirtorch.multiaxis_parallel._multiaxis_horizontal_data` (the hfan
contract n_p, centers, W_p_c, weight_scale of horizontal_fan.py, plus the
rotated in-plane depth y) and
:func:`mbirtorch.multiaxis_parallel._multiaxis_vertical_terms` (the affine
slice-to-row map m0 and slope, the vertical footprint W_p_r, its clip bound
L_max, and the mass-conserving amplitude scaling).  Both run ONCE per call,
outside every loop.  Every tap weight is then derived from their outputs
IN-KERNEL, so the tap axis is never materialized, and the kernels' tap centers
are the bodies' own centers -- which is what keeps the two directions adjoint.

Four things differ from the cone kernels, and they are why these are separate
kernels rather than cone variants:

  - Each view carries TWO angles.  ``view_params_batch[:, 0]`` is the azimuth
    and ``[:, 1]`` the elevation, and the detector row coordinate is
    v = z * cos(elevation) + y * sin(elevation).
  - The slope of the slice-to-row map is PER VIEW rather than per
    (view, pixel): m(v, p, k) = m0(v, p) + slope(v) * k, with
    slope = delta_voxel_slice * cos(elevation) / delta_det_row.  Only three
    arrays are therefore per (view, pixel) -- n_p, its integer center, and the
    row anchor m0 -- and everything else the kernel reads is one float per
    view.
  - The vertical FOOTPRINT W_p_r is a separate quantity from the slope: it is
    the largest of the voxel's three projected edges on the detector row axis,
    divided by delta_det_row.  The trapezoid weight uses the footprint, never
    the slope.  In cone the two coincide, so a kernel that carried the cone
    form over would be wrong at every nonzero elevation.
  - There is no cone-angle divisor.  Its place is taken by the per-view
    mass-conserving amplitude ``scaling``, which the back path applies AFTER
    raising the tap weight to coeff_power, as ``scaling ** coeff_power``, and
    which the forward path folds into the values before the vertical weight,
    as the torch forward body's ``scaled_values`` does.  Those are the torch
    bodies' own orders and the kernels reproduce them.

Back-kernel output columns for slices at or past ``num_slices`` are zeroed, as
the torch body zeroes them, and the forward kernel drops the same slices out of
its value gather for the same reason.  The test is on the GLOBAL slice index
``slice_start + l``, so a banded call whose band overhangs the volume behaves
exactly as the torch bodies do there.

The FORWARD kernel INVERTS the vertical fan where the torch forward body
scatters, and that is the one place the two directions are not mirror images.
The torch body walks the slices: for each recon slice it adds a weighted copy
into the detector rows around m_p(k).  The kernel walks the ROWS, as the cone
forward does -- each program owns one detector row m, enumerates the slices k
whose vertical footprint reaches m, sums their weighted values into a register
partial, and scatters that partial across the horizontal channel taps with
atomic adds.  The two forms reach the same sums in a different float order.

Inverting m_p(v, p, k) = m0(v, p) + slope(v) * k around row m gives
k_center = (m - m0) / slope, and the count of slices reaching one row is about
(W_p_r + 1) / slope + 1.  At a tilted view slope = delta_voxel_slice *
cos(elevation) / delta_det_row falls below 1, so that count EXCEEDS the
2 * psf_radius + 1 taps the back kernel's loops use: a slice-tap loop bounded
by psf_radius would silently drop contributions.  The forward's slice-tap
radius is therefore computed per call from the per-view arrays
(:func:`_multiaxis_slice_tap_radius`) and passed as a constexpr, and a
geometry whose bound exceeds ``MULTIAXIS_FWD_MAX_SLICE_RADIUS`` -- the
vanishing-slope case the module docstring of multiaxis_parallel.py names as
the reason its body scatters -- delegates to the torch body rather than
running an inversion it cannot cover.

One arithmetic deviation from the torch bodies, inherited from the cone
kernels: the back kernel's row centers use ``floor(m + 0.5)`` where the torch
body uses ``torch.round`` (half to even).  The two differ only at an exact .5
tie, and there the tap that enters or leaves the window carries weight exactly
zero.  That tap sits at |m - tap| = psf_radius + 0.5, so its trapezoid weight
is clip((W_p_r + 1) / 2 - (psf_radius + 0.5), 0), which is zero whenever
W_p_r <= 2 * psf_radius.  That bound holds by construction of
:meth:`mbirtorch.multiaxis_parallel.MultiAxisParallelModel.get_psf_radius`,
whose vertical radius is ceil(ceil(footprint / delta_det_row) / 2) taken over
an envelope of the three footprint edges at the model's own elevations, and
every view's W_p_r sits under that envelope.

The forward kernel has no such tie to absorb.  Its ``floor(k_center + 0.5)``
rounds a quantity the torch body never forms, so there is no torch value to
differ from: the rounding only chooses WHICH integer slices the loop
enumerates, and the coverage bound is built to hold for either side of a tie
(see :func:`_multiaxis_slice_tap_radius`).

The two directions do bound their windows differently, and it is the same
W_p_r <= 2 * psf_radius that makes the difference inert.  The torch body's
scatter reaches only the rows within psf_radius of round(m_p(k)), so it can
reach a (slice, row) pair only where |round(m_p) - m| <= psf_radius; the
kernel's gather carries the trapezoid window itself and no such extra test.
For an integer row m carrying a nonzero weight, |m_p - m| < (W_p_r + 1) / 2
forces the integer |round(m_p) - m| to be at most ceil(W_p_r / 2), which is at
most psf_radius under that same bound -- so every pair the kernel keeps is a
pair the body's scatter also reached, and every pair only the kernel enumerates
carries weight exactly zero from the identical
clip((W_p_r + 1) / 2 - |m_p(k) - m|, 0).  Everything else differs from the
torch bodies by float rounding alone, in two forms.  Summation order differs
wherever terms accumulate.  And the compiler may fuse the multiply-add forming
m_p = m0 + slope * k where torch rounds the product and the sum separately, so
the two round the row coordinate differently by about its own size times
float32 eps; the trapezoid weight subtracts two such coordinates, so at a
detector with many rows the weight carries an absolute perturbation of about
num_rows_r times eps, and the value gates at large row counts allow for it
(see the multi-row-chunk test).

Determinism: the BACK kernel gathers on both detector axes into a register
accumulator and stores each output element exactly once.  There are no atomic
adds, so repeated launches on the same inputs are bit-identical.  The FORWARD
kernel scatters with float ``tl.atomic_add``, which is commutative but not
associative, so its summation order over pixels and taps varies between
launches: identical inputs agree to float rounding rather than bit for bit, and
the repeat-consistency test in tests/test_triton_multiaxis.py measures that
spread directly.

No performance measurement has been made for either kernel.  The tile
constants below are the cone back and cone forward kernels', adopted as a
starting point; the sweep for this geometry has not run, and nothing here
should be read as a measured choice.

The module imports WITHOUT triton (a CPU/MPS install has none), so the test
suite can import it anywhere; only calling a wrapper needs a working triton.
"""

import contextlib
import math

import torch

from ._utils import padded_kernel_width
from .multiaxis_parallel import (_multiaxis_back_view_batch,
                                 _multiaxis_forward_view_batch,
                                 _multiaxis_horizontal_data,
                                 _multiaxis_vertical_terms)
from .projectors import compile_serialized
# The Triton language shims are IMPORTED from the cone module rather than
# duplicated, exactly as triton_parallel.py imports them.  They exist to absorb
# Triton API drift -- ``_tl_builtin``'s tl-vs-tl.math lookup and the
# static_range fallback -- which is a single moving target that should have a
# single home.  ``_COMPILED_LAUNCH_KEYS`` is deliberately shared too: every key
# leads with its kernel's name, so one set serves every kernel in the package
# and a cross-kernel false hit is impossible.  The import graph stays acyclic
# (triton_multiaxis -> triton_cone -> cone_beam, and triton_multiaxis ->
# multiaxis_parallel, which imports no triton).
from .triton_cone import (_COMPILED_LAUNCH_KEYS, _jit, _tap_range, _tile_size,
                          _tl_abs, _tl_floor, tl, triton)

_F32 = torch.float32

# ── starting constants, ADOPTED from the cone back kernel ────────────────────
# These are CONE_BACK_BLOCK_P, CONE_BACK_BLOCK_L, CONE_BACK_NUM_WARPS,
# CONE_BACK_NUM_STAGES, CONE_BACK_MIN_TILE and CONE_BACK_VIEW_CHUNK, copied
# rather than measured.  The two kernels hold a similar set of live tiles per
# program (the accumulator plus the row map, the row centers, the row weight,
# the row partial and the gathered values), which is the argument for starting
# here; it is an argument, not a reading.  No sweep has been run for this
# geometry, so nothing below is a measured value for it.  The sweep axes when
# it runs are the same four: BLOCK_P, BLOCK_L, NUM_WARPS, NUM_STAGES, plus the
# driver's view batch.
MULTIAXIS_BACK_BLOCK_P = 16
MULTIAXIS_BACK_BLOCK_L = 64
MULTIAXIS_BACK_NUM_WARPS = 4
# 1 stage = no software pipelining: the view loop is gather-bound rather than
# dot-bound, so extra stages would buy latency hiding only at more register
# pressure.
MULTIAXIS_BACK_NUM_STAGES = 1
# The smallest tile worth launching: a band or pixel subset below this pads
# rather than shrinking further.
MULTIAXIS_BACK_MIN_TILE = 8
# The driver's nominal view chunk for this kernel's batches: the batch this
# body asks for when the model's view_batch_size is None (automatic).  The
# batching rule rides on the body (see _multiaxis_back_view_batch_cost and
# Projectors._effective_view_batch), because the torch body's transient charge
# would force a small view batch at large cells for a kernel that holds no such
# transient.  Adopted from the cone back kernel with the tile constants above,
# and not yet swept here; the driver's transient budget may cap the realized
# batch below it.
MULTIAXIS_BACK_VIEW_CHUNK = 128

# ── starting constants, ADOPTED from the cone FORWARD kernel ─────────────────
# These are CONE_FWD_BLOCK_P, CONE_FWD_BLOCK_R, CONE_FWD_NUM_WARPS,
# CONE_FWD_NUM_STAGES, CONE_FWD_MIN_TILE and CONE_FWD_VIEW_CHUNK, copied rather
# than measured.  The two forward programs hold the same class of live tiles at
# (BLOCK_P, BLOCK_R) -- the detector-column accumulator plus the fractional
# slice index, the slice center, the tap weight and the gathered values -- and
# both hold that accumulator live across an atomic channel-tap loop, which is
# the argument for starting here; it is an argument, not a reading.  No sweep
# has been run for this geometry, so nothing below is a measured value for it.
# The sweep axes when it runs are BLOCK_P, BLOCK_R, NUM_WARPS, NUM_STAGES, plus
# the driver's view batch.
MULTIAXIS_FWD_BLOCK_P = 8
MULTIAXIS_FWD_BLOCK_R = 128
MULTIAXIS_FWD_NUM_WARPS = 8
# 1 stage = no software pipelining: the tap loops are gather- and atomic-bound
# rather than dot-bound, so extra stages would buy latency hiding only at more
# register pressure.
MULTIAXIS_FWD_NUM_STAGES = 1
MULTIAXIS_FWD_MIN_TILE = 8
# The forward's nominal view chunk (see MULTIAXIS_BACK_VIEW_CHUNK).
MULTIAXIS_FWD_VIEW_CHUNK = 128

# The largest slice-tap radius the forward wrapper will launch.  The slice tap
# loop is unrolled at compile time, so its trip count 2 * radius + 1 sets the
# kernel's code size; past this cap the wrapper delegates to the torch body
# instead.  Every geometry the model supports sits far below it -- the radius
# grows as 1 / slope, and the model warns above 45 degrees of elevation, where
# slope is still 0.7 * delta_voxel_slice / delta_det_row -- so the cap is
# reached only by a near-degenerate slope (a very thin slice pitch against a
# tall detector row, or an elevation approaching 90 degrees), which is exactly
# the case multiaxis_parallel.py's module docstring gives as the reason its
# forward body scatters instead of inverting.  Not a measured value: it is a
# code-size bound, chosen so the unrolled loop stays comparable to the widest
# tap loops the other kernels compile.
MULTIAXIS_FWD_MAX_SLICE_RADIUS = 32

# The divisor floor for the slice-to-row inversion.  The slope is strictly
# positive for every elevation under 90 degrees, so this never binds on a real
# model; it exists so that a degenerate view produces a finite (very large)
# radius that the cap above can catch, rather than an infinity or a NaN.
_SLOPE_FLOOR = 1e-6


def _multiaxis_slice_tap_radius(w_p_r, slope):
    """The forward kernel's slice-tap radius for one view batch: how far, in
    whole slices, the loop must reach on each side of ``round(k_center)`` to
    cover every slice whose vertical footprint touches the row a program owns.

    THE BOUND.  Write H = (W_p_r + 1) / (2 * slope) for one view.  Slice k
    contributes to row m only where |m_p(k) - m| < (W_p_r + 1) / 2, and
    m_p(k) - m = slope * (k - k_center) with k_center = (m - m0) / slope, so
    that window is exactly |k - k_center| < H.  The loop enumerates
    k = round(k_center) + t for |t| <= radius, and |round(k_center) - k_center|
    <= 0.5, so every contributing k has |t| < H + 0.5; an integer strictly
    below H + 0.5 is at most floor(H + 0.5).  Taking W_p_r at its largest over
    the batch's views and the slope at its smallest makes floor(H + 0.5) an
    upper bound for every view at once.

    THE MARGIN.  One more slice is added on each side.  The bound above is
    computed here from the built arrays while the kernel rounds its own
    ``k_center`` in float32, and the two can disagree by one step where
    ``k_center`` sits within a rounding of a .5 tie.  The extra ring makes that
    disagreement inert without an argument about float error; the taps it adds
    carry weight zero wherever the tight bound was already enough, so it costs
    loop iterations and changes no value.  A sweep may drop it once the tie
    behavior is measured -- it is margin, not a correctness requirement of the
    derivation above.

    The minimum is taken on the SIGNED slope, which is how a slope that is not
    strictly positive reports itself: the derivation above needs slope > 0 (it
    divides by it and keeps the inequality's direction), and a view whose slope
    is zero or negative -- an elevation at or past 90 degrees, which no
    supported model has -- drops the minimum onto ``_SLOPE_FLOOR`` and returns a
    radius far above the caller's cap, so it delegates instead of inverting.

    Args:
        w_p_r: the per-view vertical footprint in detector rows, (Vb, 1).
        slope: the per-view slice-to-row slope, (Vb, 1).

    Returns:
        int: the radius, before the caller compares it against
        ``MULTIAXIS_FWD_MAX_SLICE_RADIUS``.
    """
    # Two scalar reads, which means one device sync per forward call.  It buys
    # a COMPILE-TIME trip count for the slice tap loop, which a runtime bound
    # would give up; the angles are fixed at construction, so the radius is one
    # constant per model and a later step can memoize it through the ``plan``
    # slot rather than re-reducing it per call.
    w_max = float(w_p_r.max())
    slope_min = max(float(slope.min()), _SLOPE_FLOOR)
    return int(math.floor((w_max + 1.0) / (2.0 * slope_min) + 0.5)) + 1


@_jit
def _multiaxis_back_kernel(n_p_ptr, centers_ptr, m0_ptr,
                           w_p_c_ptr, weight_scale_ptr,
                           slope_ptr, w_p_r_ptr, l_max_ptr, scale_pow_ptr,
                           sino_ptr, out_ptr,
                           num_views, num_pixels, num_channels, num_rows,
                           band_len, slice_start, num_slices,
                           sino_view_stride, num_rows_f,
                           PSF_RADIUS: tl.constexpr, COEFF_POWER: tl.constexpr,
                           BLOCK_P: tl.constexpr, BLOCK_L: tl.constexpr):
    """One program per (pixel block, slice chunk) of the output partial:

        out[p, l] = sum over views v, row taps tr, channel taps tc of
                    Wrow[v, l, tr] * Wchan[v, p, tc]
                    * sino[v, c(v, p) + tc, m(v, p, l) + tr]

    zeroed wherever the global slice index slice_start + l is at or past
    num_slices.  Both weight sets are formed in-kernel from the per-(view,
    pixel) contract (n_p, centers, m0) and the per-view floats (W_p_c,
    weight_scale, slope, W_p_r, L_max, scaling ** coeff_power) that the module
    docstring's two builders produce.

    The pixel block is the FAST grid axis so that concurrently scheduled
    programs gather from the same detector rows of the same view, which keeps
    those rows resident in L2 for these transaction-bound gathers.  Views are
    an in-program loop because they reduce into the one register accumulator.

    Pixels beyond ``num_pixels`` ride as padded lanes: their loaded contract
    values are zeroed, which zeroes both tap weights, and their stores are
    masked.

    The slice axis works differently.  ``band_len`` is the band the WRAPPER
    launches, which is the real band rounded up to a multiple of 16 (see
    :func:`mbirtorch._utils.padded_kernel_width`).  Slice lanes between the
    real band and that rounded-up value are ordinary live lanes here: they
    load, they compute, and they store.  Two things make that safe.  Every
    sinogram address this kernel forms is clamped into the buffer, so a lane
    whose global slice index points past the volume still reads inside the
    sinogram.  And the wrapper returns only the first real-band columns of
    ``out``, so nothing reads what those lanes stored.
    """
    p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)     # (BLOCK_P,)
    l_offs = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)     # (BLOCK_L,)
    p_mask = p_offs < num_pixels
    l_mask = l_offs < band_len
    tile_mask = p_mask[:, None] & l_mask[None, :]

    # GLOBAL slice index of each output column: the slice-to-row map is
    # anchored at global slice 0, so a band restricts the range of k and
    # nothing else.  The same global index carries the validity test the torch
    # body applies to its output.
    k_global = slice_start + l_offs                                 # (BLOCK_L,)
    k = k_global.to(tl.float32)
    valid_k = k_global < num_slices

    acc = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
    for v in range(num_views):
        pix_base = v.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
        # Per-VIEW scalars: under this geometry the horizontal footprint
        # depends on the azimuth alone, and the vertical slope, footprint,
        # clip bound and amplitude depend on the elevation and azimuth alone.
        w_p_c = tl.load(w_p_c_ptr + v)
        weight_scale = tl.load(weight_scale_ptr + v)
        clip_c = tl.minimum(w_p_c, 1.0)
        slope = tl.load(slope_ptr + v)
        w_p_r = tl.load(w_p_r_ptr + v)
        l_max_r = tl.load(l_max_ptr + v)
        scale_pow = tl.load(scale_pow_ptr + v)

        # Vertical fan: the affine slice-to-row map, exactly as the torch body
        # forms it.  The SLOPE maps slices to rows; the FOOTPRINT below is a
        # different quantity and only it enters the trapezoid weight.
        m = m0[:, None] + slope * k[None, :]                 # (BLOCK_P, BLOCK_L)
        m_center = _tl_floor(m + 0.5)                        # ties are inert
        # Bound the center before the integer conversions in the tap loop: a
        # steep tilt with a tall volume can send m outside the panel, and
        # float-to-int is undefined past the int32 range.  Provably inert: a
        # tap can carry weight only where |m - tap| < (W_p_r + 1)/2 <=
        # psf_radius + 0.5 at a tap inside the panel, which keeps every
        # contributing center well within these bounds; anything clamped had
        # its whole tap window outside the panel and was masked to zero either
        # way.
        m_center = tl.minimum(tl.maximum(m_center, -1.0 - PSF_RADIUS),
                              num_rows_f + PSF_RADIUS)
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tr in _tap_range(0, 2 * PSF_RADIUS + 1):
            m_tap = m_center + (tr - PSF_RADIUS)
            m_tap_i = m_tap.to(tl.int32)
            # The vertical trapezoid rule of the torch body's back vertical
            # fan: the FOOTPRINT sets the width, the clip bound caps it, taps
            # that left the panel get weight zero, and the index is clamped
            # (the zero-and-clamp convention of horizontal_fan.py).
            w_row = tl.maximum((w_p_r + 1.0) / 2.0
                               - _tl_abs(m - m_tap), 0.0)
            w_row = tl.minimum(w_row, l_max_r)
            w_row = tl.where((m_tap_i >= 0) & (m_tap_i < num_rows), w_row, 0.0)
            if COEFF_POWER == 2:
                w_row = w_row * w_row
            # The amplitude is applied AFTER the power, already raised to the
            # same power by the wrapper.  This is the torch body's order (its
            # scale_pow term); do not fold it into the weight above.
            w_row = w_row * scale_pow
            m_row = tl.minimum(tl.maximum(m_tap_i, 0), num_rows - 1)

            row_vals = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
            for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
                # The horizontal_fan trapezoid rule, in-kernel: zero the
                # weight where the unclipped tap left the detector, then clamp
                # the index (the zero-and-clamp convention).
                n_tap = centers + (tc - PSF_RADIUS)
                w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, clip_c) * weight_scale
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

    # Slices at or past the real count contribute nothing, exactly as the torch
    # body's closing mask states it.  Applied to every lane, so a padded lane
    # that somehow reached the real range would be zeroed too (it cannot: the
    # wrapper slices the real band off before returning).
    acc = tl.where(valid_k[None, :], acc, 0.0)
    out_ptrs = out_ptr + p_offs.to(tl.int64)[:, None] * band_len + l_offs[None, :]
    tl.store(out_ptrs, acc, mask=tile_mask)


@torch.compiler.disable
def _multiaxis_back_view_batch_triton(sino_batch, pixel_indices,
                                      view_params_batch, num_rows_r,
                                      num_channels, num_recon_rows,
                                      num_recon_cols, num_slices, delta_voxel,
                                      delta_voxel_row, delta_voxel_slice,
                                      delta_det_channel, delta_det_row,
                                      det_channel_offset, det_row_offset,
                                      recon_slice_offset, psf_radius,
                                      coeff_power=1, slice_start=0,
                                      band_slices=None, plan=None):
    """The Triton multiaxis back body: a drop-in replacement for
    :func:`mbirtorch.multiaxis_parallel._multiaxis_back_view_batch` (same
    signature, same (P, band) return, freshly written each call so a driver may
    accumulate into it in place).

    The model binds this body in the torch body's place wherever the back
    kernel's availability gate passes; see the module docstring.

    The band argument is rounded up to a multiple of 16 before the launch,
    because Triton compiles a faster kernel for an integer argument it can
    prove divisible by 16.  The return is then the real-band slice of a
    slightly wider output, so a band that is not a multiple of 16 returns a
    strided view instead of a contiguous one.  A band that IS a multiple takes
    exactly the path it took before, with the same allocations.

    Eager python by construction, declared twice over because the two
    mechanisms cover different callers: ``torch.compiler.disable`` keeps dynamo
    out when a compiled region CALLS this body, and the
    ``_mbirtorch_no_compile`` marker set below keeps the driver's
    ``maybe_compile`` from compiling it DIRECTLY (torch.compile unwraps the
    disable decorator and would trace the launch anyway).  The two geometry
    builders run ONCE per call, outside every loop.

    ``plan`` is accepted and ignored -- the sorted/CSR stream slot, not yet
    built.
    """
    if triton is None:
        raise RuntimeError('the Triton multiaxis back body was called without '
                           'triton installed; a caller must keep the torch '
                           'body where no triton is available.')
    # Powers other than 1 and 2 are outside the kernel's static branch (and
    # outside every caller in the package): delegate rather than diverge.
    if coeff_power not in (1, 2):
        return _multiaxis_back_view_batch(
            sino_batch, pixel_indices, view_params_batch, num_rows_r,
            num_channels, num_recon_rows, num_recon_cols, num_slices,
            delta_voxel, delta_voxel_row, delta_voxel_slice, delta_det_channel,
            delta_det_row, det_channel_offset, det_row_offset,
            recon_slice_offset, psf_radius, coeff_power=coeff_power,
            slice_start=slice_start, band_slices=band_slices, plan=plan)

    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    # The body's OWN builders, so the kernel's tap centers and weights derive
    # from the same computation the torch pair uses.
    n_p, centers, w_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    m0, slope, w_p_r, l_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)
    # The torch body's own scale_pow expression, hoisted here so the kernel
    # applies exactly the value the body applies.
    scale_pow = scaling ** coeff_power

    num_views, num_pixels = n_p.shape
    band_len = int(num_slices if band_slices is None else band_slices)
    # The band the kernel is LAUNCHED at, rounded up to a multiple of 16 so
    # that Triton compiles the faster specialization of it.  Every use of the
    # band argument takes this value -- the grid, the tile mask, and the output
    # row stride -- because splitting them would leave a non-divisible integer
    # in the launch and lose the specialization again.  A band that is already
    # a multiple of 16 gets its own value back, so every allocation and every
    # argument below is exactly what it was before this padding existed.
    launch_band = padded_kernel_width(band_len)
    # Channel-major views, as in the torch body: the kernel's per-tile gather
    # walks the slice axis, whose row index is contiguous in this layout.  The
    # copy is NOT padded, following the cone back kernel: the kernel clamps
    # every sinogram address it forms, so a padded slice lane reads an existing
    # detector row.
    sino_t = sino_batch.permute(0, 2, 1).contiguous()
    # Per-(view, pixel): three arrays, 12 bytes a pixel a view.
    contract = [t.contiguous() for t in (n_p, centers, m0)]
    # Per-view scalars, and the reshape is the check as well as the
    # conversion: a term that ever became per-pixel would raise here rather
    # than broadcast silently.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale, slope, w_p_r, l_max,
                           scale_pow)]
    out = torch.empty((num_pixels, launch_band), dtype=_F32,
                      device=sino_batch.device)

    block_p = _tile_size(MULTIAXIS_BACK_BLOCK_P, num_pixels,
                         MULTIAXIS_BACK_MIN_TILE)
    block_l = _tile_size(MULTIAXIS_BACK_BLOCK_L, launch_band,
                         MULTIAXIS_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_band // block_l))
    # The padded band keys the launch, because it is the integer the
    # compilation is keyed on.
    launch_key = ('maback', sino_batch.device.index, int(psf_radius),
                  int(coeff_power), block_p, block_l,
                  int(num_views), int(num_pixels), int(num_channels),
                  int(num_rows_r), launch_band, int(slice_start),
                  int(num_slices))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch must be bracketed on the tensors' device, and the device leads
    # the launch key -- see _cone_back_view_batch_triton (triton_cone), whose
    # comment carries the measured basis.
    with torch.cuda.device(sino_batch.device), guard:
        _multiaxis_back_kernel[grid](
            *contract, sino_t, out,
            int(num_views), int(num_pixels), int(num_channels),
            int(num_rows_r), launch_band, int(slice_start), int(num_slices),
            int(num_channels) * int(num_rows_r), float(num_rows_r),
            PSF_RADIUS=int(psf_radius), COEFF_POWER=int(coeff_power),
            BLOCK_P=block_p, BLOCK_L=block_l,
            num_warps=MULTIAXIS_BACK_NUM_WARPS,
            num_stages=MULTIAXIS_BACK_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_band == band_len:
        return out
    # The padded columns hold values no caller reads, so they are sliced off.
    # The result is a strided view, which a driver's accumulation and the
    # cross-device reduce both handle: each row is still one contiguous run.
    return out[:, :band_len]


# See the wrapper's docstring: the driver reads this marker in maybe_compile.
_multiaxis_back_view_batch_triton._mbirtorch_no_compile = True


def _multiaxis_back_view_batch_cost(num_pixels, num_band_rows, args):
    """Charged bytes resident per view in one back-kernel batch, and this
    kernel's nominal view chunk -- the batching rule for this body, read
    through the ``_view_batch_cost`` attribute in
    ``Projectors._effective_view_batch``.

    One view of a batch holds the per-(view, pixel) contract at 12 bytes per
    (view, pixel): ``n_p`` and ``m0`` as f32 and ``centers`` as i32.  The rest
    of the contract -- W_p_c, weight_scale, slope, W_p_r, L_max and the
    amplitude -- is one float per view under this geometry, which rounds to
    nothing per pixel and is not charged.  A view also holds the channel-major
    copy of its sinogram plane (``sino_t`` in the wrapper); the second argument
    is the sinogram's row count at this call site, so the plane term follows
    the input directly.  That copy is not padded even when the band is: the
    kernel clamps every sinogram address it forms, so a padded slice lane reads
    a row this copy already holds.

    Call-fixed tensors -- the (P, band) output partial and the register
    accumulators inside the kernel -- exist at any batch size, so the batch
    choice cannot control them and they are not charged, exactly as the
    torch-body budget never charged its own fixed outputs.  The charge is a
    counted estimate that protects the budget boundary; the chunk constant is
    the performance chooser, and it has not been swept for this geometry (see
    MULTIAXIS_BACK_VIEW_CHUNK)."""
    plane_bytes = 4 * int(args['num_channels']) * int(num_band_rows)
    return 12 * int(num_pixels) + plane_bytes, MULTIAXIS_BACK_VIEW_CHUNK


_multiaxis_back_view_batch_triton._view_batch_cost = \
    _multiaxis_back_view_batch_cost


@_jit
def _multiaxis_forward_kernel(n_p_ptr, centers_ptr, m0_ptr,
                              w_p_c_ptr, weight_scale_ptr,
                              slope_ptr, w_p_r_ptr, l_max_ptr, scaling_ptr,
                              values_ptr, out_ptr,
                              num_pixels, num_channels, num_rows,
                              out_row_stride, out_view_stride,
                              band_len, slice_start, num_slices,
                              slope_floor, k_center_lo, k_center_hi,
                              PSF_RADIUS: tl.constexpr,
                              SLICE_RADIUS: tl.constexpr,
                              BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr):
    """One program per (pixel block, row chunk, view) of the sinogram:

        out[v, c, m] += sum over pixels p, channel taps tc, slices k of
                        Wchan[v, p, tc] * Wrow[v, p, k, m] * scaling[v]
                        * values[p, k - slice_start]

    -- the DETECTOR-side vertical fan (for each detector row, which recon
    slices project onto it) followed by the horizontal fan scatter.  The torch
    body computes the same sum from the other side, scattering each slice into
    the rows around m_p(k); this kernel gathers, so the slices reaching one row
    have to be ENUMERATED, and ``SLICE_RADIUS`` is what bounds that enumeration
    (see :func:`_multiaxis_slice_tap_radius` for the bound and its proof).

    Grid choice, inherited from the cone forward: the VIEW axis has to be a
    grid axis rather than an in-program loop, because the forward writes a
    separate output plane per view and a view loop would only serialize.  The
    PIXEL block is the fast axis so that concurrently scheduled programs read
    the same rows of ``values`` and hit neighbouring channels of the same view
    with their atomics, since neighbouring pixels project to neighbouring
    channels.  The view axis is last because it is the only small one; it
    inherits CUDA's 65535 limit on grid dims 1 and 2, which the driver's view
    batch is nowhere near.

    The output is CHANNEL-MAJOR (Vb, C, R), the layout fan_forward_batch also
    accumulates in: it puts the row axis -- the kernel's vector axis -- on the
    contiguous stride, so one tile row's atomics land on consecutive addresses
    instead of striding by C.  It must reach the kernel ZEROED, because these
    atomics accumulate into it.  The wrapper transposes the view on return, as
    the torch body does.

    Pixels beyond ``num_pixels`` ride as padded lanes: their atomics are masked
    off entirely, so a padded lane's contract values may be anything finite.
    Unlike the cone forward, no load here needs a nonzero filler to stay
    finite: that kernel divides by a per-(view, pixel) ``w_p_r`` and fills a
    padded lane with 1.0 to keep 0/0 out of its float-to-int conversions, while
    the only divisor here is the per-VIEW slope, which is one unmasked scalar
    load and cannot be zeroed by a padded pixel lane.  It is guarded by
    ``slope_floor`` instead, against a degenerate view rather than against
    padding.

    ``out_row_stride`` is the detector row count the WRAPPER allocated, which
    is the real count rounded up to a multiple of 16 (see
    :func:`mbirtorch._utils.padded_kernel_width`); ``num_rows`` is the real
    count.  Row lanes between the two are padded lanes whose atomics are masked
    off by ``r_mask``, so the extra output rows keep the zeros they were
    allocated with and the wrapper slices them away.  That mask is also where
    the torch body's ``(m >= 0) & (m < num_rows_r)`` row-range factor lives:
    every row this kernel writes is a real detector row, so the factor is 1 on
    every lane that stores and the mask carries the rest.
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
    m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
    # Per-VIEW scalars: under this geometry the horizontal footprint depends on
    # the azimuth alone, and the vertical slope, footprint, clip bound and
    # amplitude depend on the elevation and azimuth alone.
    w_p_c = tl.load(w_p_c_ptr + v)
    weight_scale = tl.load(weight_scale_ptr + v)
    clip_c = tl.minimum(w_p_c, 1.0)
    slope = tl.load(slope_ptr + v)
    w_p_r = tl.load(w_p_r_ptr + v)
    l_max_r = tl.load(l_max_ptr + v)
    scaling = tl.load(scaling_ptr + v)

    # ── vertical fan, detector side: rows -> fractional slice indices ────────
    # The SLOPE inverts the map; the FOOTPRINT below is a different quantity
    # and only it enters the trapezoid weight.  The floor on the divisor cannot
    # bind on a model the geometry class accepts (see _SLOPE_FLOOR); it is here
    # so that a degenerate view costs loop iterations, never a NaN reaching the
    # float-to-int conversions in the tap loop.
    m_f = r_offs.to(tl.float32)                                     # (BLOCK_R,)
    k_m = (m_f[None, :] - m0[:, None]) / tl.maximum(slope, slope_floor)
    k_center = _tl_floor(k_m + 0.5)
    # Bound the center before those conversions: a small slope sends k_m far
    # outside the volume, and float-to-int is undefined past the int32 range.
    # These bounds are exactly tight, which is what makes them inert: at the low
    # bound the HIGHEST tap is slice_start - 1, one short of the band, and at
    # the high bound the LOWEST tap is slice_start + band_len, one past it -- so
    # a clamped center contributes nothing, and neither did the center it
    # replaced.
    k_center = tl.minimum(tl.maximum(k_center, k_center_lo), k_center_hi)

    det_col = tl.zeros((BLOCK_P, BLOCK_R), dtype=tl.float32)
    for tk in _tap_range(0, 2 * SLICE_RADIUS + 1):
        k_ind = k_center + (tk - SLICE_RADIUS)
        k_ind_i = k_ind.to(tl.int32)
        # The torch body's own vertical weight for the pair (slice k, row m):
        # it forms |m_p - m_tap| with m_tap an integer row near round(m_p), and
        # for a FIXED integer row m the distance |m_p(k) - m| is that same
        # quantity.  The FOOTPRINT sets the width and the clip bound caps it.
        m_p = m0[:, None] + slope * k_ind
        w_row = tl.maximum((w_p_r + 1.0) / 2.0 - _tl_abs(m_p - m_f[None, :]),
                           0.0)
        w_row = tl.minimum(w_row, l_max_r)
        # Zero the weight where the tap left the BAND, then clamp the index
        # (the zero-and-clamp convention): the slice-to-row map is anchored on
        # the full slice count, so a band only restricts which slices
        # contribute.  The same test carries the torch body's validity mask on
        # the GLOBAL slice index, which that body folds into its scaled_values.
        live = ((k_ind_i >= slice_start) & (k_ind_i < slice_start + band_len)
                & (k_ind_i < num_slices))
        w_row = tl.where(live, w_row, 0.0)
        # The clamp keeps the address legal; carrying ``live`` into the load
        # mask as well is the optimization, not the contract -- an out-of-band
        # tap is already zero-weighted, and this only saves the read.
        k_local = tl.minimum(tl.maximum(k_ind_i - slice_start, 0), band_len - 1)
        vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * band_len
                       + k_local, mask=tile_mask & live, other=0.0)
        # The amplitude multiplies the VALUE, not the weight: that is where the
        # torch body folds it (its scaled_values), and the back kernel's
        # adjointness rests on the two agreeing.
        det_col = det_col + w_row * (vals * scaling)

    # ── horizontal fan scatter: the plain per-tap atomic form ────────────────
    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
        # The horizontal_fan trapezoid rule, in-kernel (as in the back kernel);
        # here the out-of-detector taps drop out of the atomic MASK rather than
        # being added as zeros, which is the same value and one less atomic.
        # No coefficient power: the forward path has none.
        n_tap = centers + (tc - PSF_RADIUS)
        w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                            - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
        w_chan = tl.minimum(w_chan, clip_c) * weight_scale
        n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
        out_ptrs = (out_view_ptr + n_chan.to(tl.int64)[:, None] * out_row_stride
                    + r_offs[None, :])
        tl.atomic_add(out_ptrs, w_chan[:, None] * det_col,
                      mask=tile_mask & ((n_tap >= 0)
                                        & (n_tap < num_channels))[:, None])


@torch.compiler.disable
def _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                         view_params_batch, num_rows_r,
                                         num_channels, num_recon_rows,
                                         num_recon_cols, num_slices,
                                         delta_voxel, delta_voxel_row,
                                         delta_voxel_slice, delta_det_channel,
                                         delta_det_row, det_channel_offset,
                                         det_row_offset, recon_slice_offset,
                                         psf_radius, slice_start=0, plan=None):
    """The Triton multiaxis forward body: a drop-in replacement for
    :func:`mbirtorch.multiaxis_parallel._multiaxis_forward_view_batch` (same
    signature, same (Vb, R, C) return, freshly ZEROED each call because the
    kernel accumulates into it with atomics).

    The model binds this body in the torch body's place wherever the forward
    kernel's availability gate passes; see the module docstring.  The signature
    is the torch body's -- ``slice_start`` and ``plan``, no ``coeff_power`` and
    no ``band_slices``, because the forward's band rides in the columns of
    ``values``.

    The detector row count is rounded up to a multiple of 16 before the launch,
    because Triton compiles a faster kernel for an integer argument it can
    prove divisible by 16, and that count is the output's row stride.  The
    return is then the real-row slice of a slightly taller output.  A row count
    that IS a multiple of 16 takes exactly the path it took before, with the
    same allocations.  ``values`` needs no such padding: the slice gather is
    clamped into the band it was handed and masked by that band, so no read
    leaves it.

    Eager python by construction, declared twice over because the two
    mechanisms cover different callers: ``torch.compiler.disable`` keeps dynamo
    out when a compiled region CALLS this body, and the
    ``_mbirtorch_no_compile`` marker set below keeps the driver's
    ``maybe_compile`` from compiling it DIRECTLY (torch.compile unwraps the
    disable decorator and would trace the launch anyway).  The two geometry
    builders run ONCE per call, outside every loop.

    ``values`` is (P, L); when L < num_slices it is the slice BAND starting at
    ``slice_start`` and the slice-to-row map stays anchored on the full
    num_slices center, so summing the outputs over a tiling of the slice axis
    reproduces the unbanded projection.  ``plan`` is accepted and ignored --
    the sorted/CSR stream slot, not yet built.
    """
    if triton is None:
        raise RuntimeError('the Triton multiaxis forward body was called '
                           'without triton installed; a caller must keep the '
                           'torch body where no triton is available.')
    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    # The body's OWN builders, so the kernel's tap centers and weights derive
    # from the same computation the torch pair uses.
    n_p, centers, w_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    m0, slope, w_p_r, l_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)

    # How far the slice tap loop must reach on each side of round(k_center).
    # A geometry whose bound exceeds the cap is the vanishing-slope case the
    # gather cannot cover at a bounded trip count: delegate rather than diverge,
    # exactly as the back wrapper delegates a coefficient power its static
    # branch does not hold.
    slice_radius = _multiaxis_slice_tap_radius(w_p_r, slope)
    if slice_radius > MULTIAXIS_FWD_MAX_SLICE_RADIUS:
        return _multiaxis_forward_view_batch(
            values, pixel_indices, view_params_batch, num_rows_r, num_channels,
            num_recon_rows, num_recon_cols, num_slices, delta_voxel,
            delta_voxel_row, delta_voxel_slice, delta_det_channel,
            delta_det_row, det_channel_offset, det_row_offset,
            recon_slice_offset, psf_radius, slice_start=slice_start, plan=plan)

    num_views, num_pixels = n_p.shape
    band_len = int(values.shape[1])
    # The detector row count the OUTPUT is allocated and addressed at, rounded
    # up to a multiple of 16 so that Triton compiles the faster specialization
    # of that stride.  A row count that is already a multiple gets its own value
    # back, so every allocation and every argument below is exactly what it was
    # before this padding existed.  The geometry builders above keep the REAL
    # row count, and so does the kernel's row mask; only the stride and the
    # allocation move.
    launch_rows = padded_kernel_width(int(num_rows_r))
    values = values.contiguous()
    # Per-(view, pixel): three arrays, 12 bytes a pixel a view.
    contract = [t.contiguous() for t in (n_p, centers, m0)]
    # Per-view scalars, and the reshape is the check as well as the conversion:
    # a term that ever became per-pixel would raise here rather than broadcast
    # silently.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale, slope, w_p_r, l_max, scaling)]
    # Channel-major, ZEROED: the atomics accumulate into this, and the return
    # transposes the view exactly as the torch body transposes
    # fan_forward_batch's.
    out = torch.zeros((num_views, num_channels, launch_rows), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(MULTIAXIS_FWD_BLOCK_P, num_pixels,
                         MULTIAXIS_FWD_MIN_TILE)
    block_r = _tile_size(MULTIAXIS_FWD_BLOCK_R, launch_rows,
                         MULTIAXIS_FWD_MIN_TILE)
    # The row grid covers the REAL rows: a program past them would be masked
    # off on every lane, so launching it would only cost a scheduling slot.
    grid = (-(-num_pixels // block_p), -(-int(num_rows_r) // block_r),
            num_views)
    # The inert bounds on the slice center (see the kernel).
    k_center_lo = float(int(slice_start) - slice_radius - 1)
    k_center_hi = float(int(slice_start) + band_len + slice_radius)
    # The padded row stride and the slice-tap radius key the launch, because
    # they are the integers the compilation is keyed on.
    launch_key = ('mafwd', values.device.index, int(psf_radius), slice_radius,
                  block_p, block_r, int(num_views), int(num_pixels),
                  int(num_channels), int(num_rows_r), launch_rows, band_len,
                  int(slice_start), int(num_slices))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch must be bracketed on the tensors' device, and the device leads
    # the launch key -- see _cone_back_view_batch_triton (triton_cone), whose
    # comment carries the measured basis.
    with torch.cuda.device(values.device), guard:
        _multiaxis_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), int(num_rows_r),
            launch_rows, int(num_channels) * launch_rows,
            band_len, int(slice_start), int(num_slices),
            _SLOPE_FLOOR, k_center_lo, k_center_hi,
            PSF_RADIUS=int(psf_radius), SLICE_RADIUS=slice_radius,
            BLOCK_P=block_p, BLOCK_R=block_r,
            num_warps=MULTIAXIS_FWD_NUM_WARPS,
            num_stages=MULTIAXIS_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_rows == int(num_rows_r):
        return out.permute(0, 2, 1)
    # The extra detector rows hold nothing any caller reads -- the kernel's row
    # mask left them at the zeros they were allocated with -- so they are sliced
    # off before the transpose.
    return out[:, :, :int(num_rows_r)].permute(0, 2, 1)


# See the back wrapper's docstring: the driver reads this marker in
# maybe_compile.
_multiaxis_forward_view_batch_triton._mbirtorch_no_compile = True


def _multiaxis_forward_view_batch_cost(num_pixels, band_len, args):
    """The forward twin of :func:`_multiaxis_back_view_batch_cost`: one view
    holds the same 12-byte-per-(view, pixel) contract (``n_p`` and ``m0`` as
    f32, ``centers`` as i32; the rest is one float per view under this geometry
    and rounds to nothing per pixel) and its zeroed channel-major output plane,
    which is what the atomics accumulate into.

    The output plane spans the FULL detector rows whatever slice band the values
    carry, so the plane term reads ``num_rows_r`` from the args rather than the
    band length, and it reads the PADDED row count because that is what the
    wrapper allocates.  ``values`` is call-fixed and not charged, as the
    call-fixed tensors never are.

    The charge is a counted estimate that protects the budget boundary; the
    chunk constant is the performance chooser, and it has not been swept for
    this geometry (see MULTIAXIS_FWD_VIEW_CHUNK)."""
    plane_rows = padded_kernel_width(int(args['num_rows_r']))
    plane_bytes = 4 * int(args['num_channels']) * plane_rows
    return 12 * int(num_pixels) + plane_bytes, MULTIAXIS_FWD_VIEW_CHUNK


_multiaxis_forward_view_batch_triton._view_batch_cost = \
    _multiaxis_forward_view_batch_cost
