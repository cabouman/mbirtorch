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
# The Triton language shims live in the cone module.  _COMPILED_LAUNCH_KEYS is
# shared by every kernel, and each key leads with its kernel's name.
from .triton_cone import (_COMPILED_LAUNCH_KEYS, _jit, _tap_range, _tile_size,
                          _tl_abs, _tl_floor, tl, triton)

_F32 = torch.float32

# The tile constants below are the cone kernels' constants, adopted as a
# starting point.  No sweep has been run for this geometry.
MULTIAXIS_BACK_BLOCK_P = 16
MULTIAXIS_BACK_BLOCK_L = 64
MULTIAXIS_BACK_NUM_WARPS = 4
MULTIAXIS_BACK_NUM_STAGES = 1
# A band or pixel subset smaller than this tile is padded, not shrunk.
MULTIAXIS_BACK_MIN_TILE = 8
# The view batch this body asks for when the model's view_batch_size is None.
# The driver's transient budget may cap the realized batch below it.
MULTIAXIS_BACK_VIEW_CHUNK = 128

MULTIAXIS_FWD_BLOCK_P = 8
MULTIAXIS_FWD_BLOCK_R = 128
MULTIAXIS_FWD_NUM_WARPS = 8
MULTIAXIS_FWD_NUM_STAGES = 1
MULTIAXIS_FWD_MIN_TILE = 8
MULTIAXIS_FWD_VIEW_CHUNK = 128

# The largest slice tap radius the forward wrapper will launch.  The tap loop is
# unrolled at compile time, so past this cap the wrapper uses the torch body.
MULTIAXIS_FWD_MAX_SLICE_RADIUS = 32

# The divisor floor for the slice to row inversion.  It keeps a degenerate view
# from producing an infinity or a NaN, and it never binds on a real model.
_SLOPE_FLOOR = 1e-6


def _multiaxis_slice_tap_radius(w_p_r, slope):
    """Return the forward kernel's slice tap radius for one view batch.

    The radius is how far in whole slices the loop must reach on each side of
    round(k_center) to cover every slice whose vertical footprint touches the
    row a program owns.  Slice k reaches row m only where
    |k - k_center| < (W_p_r + 1) / (2 * slope), so floor of that quantity plus
    0.5 bounds the reach.  The largest W_p_r and the smallest slope over the
    batch make the bound hold for every view at once, and one extra slice on
    each side absorbs a rounding disagreement with the kernel.

    A slope that is zero or negative never arises for an elevation under 90
    degrees.  Such a view falls onto ``_SLOPE_FLOOR`` and returns a radius far
    above the caller's cap, so the caller delegates to the torch body.

    Args:
        w_p_r: the per view vertical footprint in detector rows, (Vb, 1).
        slope: the per view slice to row slope, (Vb, 1).

    Returns:
        int: the radius, which the caller compares against
        ``MULTIAXIS_FWD_MAX_SLICE_RADIUS``.
    """
    # These two scalar reads cost one device sync per forward call.  They buy a
    # compile time trip count for the slice tap loop.
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
    """One program per (pixel block, slice chunk) of the output partial.

        out[p, l] = sum over views v, row taps tr, channel taps tc of
                    Wrow[v, l, tr] * Wchan[v, p, tc]
                    * sino[v, c(v, p) + tc, m(v, p, l) + tr]

    The result is zeroed wherever the global slice index slice_start + l is at
    or past num_slices.  Both weight sets are formed in the kernel from the per
    (view, pixel) arrays n_p, centers and m0, and the per view floats W_p_c,
    weight_scale, slope, W_p_r, L_max and scaling raised to coeff_power.

    The pixel block is the fast grid axis, so programs that run together gather
    the same detector rows of the same view.  Views are an in program loop,
    because they reduce into the one register accumulator.

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

    # k_global is the global slice index of each output column.  The slice to row
    # map is anchored at global slice 0, so a band only restricts the range of k.
    k_global = slice_start + l_offs                                 # (BLOCK_L,)
    k = k_global.to(tl.float32)
    valid_k = k_global < num_slices

    acc = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
    for v in range(num_views):
        pix_base = v.to(tl.int64) * num_pixels + p_offs
        n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
        centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
        m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
        # These are per view scalars.  The horizontal footprint depends on the
        # azimuth alone, and the vertical terms on the azimuth and elevation.
        w_p_c = tl.load(w_p_c_ptr + v)
        weight_scale = tl.load(weight_scale_ptr + v)
        clip_c = tl.minimum(w_p_c, 1.0)
        slope = tl.load(slope_ptr + v)
        w_p_r = tl.load(w_p_r_ptr + v)
        l_max_r = tl.load(l_max_ptr + v)
        scale_pow = tl.load(scale_pow_ptr + v)

        # This is the affine slice to row map, whose slope maps slices to rows.  The
        # footprint W_p_r is a different quantity, and only it enters the weight.
        m = m0[:, None] + slope * k[None, :]                 # (BLOCK_P, BLOCK_L)
        m_center = _tl_floor(m + 0.5)                        # Ties carry zero weight.
        # The center is bounded before the integer conversions in the tap loop,
        # because float to int is undefined past the int32 range.
        m_center = tl.minimum(tl.maximum(m_center, -1.0 - PSF_RADIUS),
                              num_rows_f + PSF_RADIUS)
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tr in _tap_range(0, 2 * PSF_RADIUS + 1):
            m_tap = m_center + (tr - PSF_RADIUS)
            m_tap_i = m_tap.to(tl.int32)
            # In the vertical trapezoid rule the footprint sets the width and the
            # clip bound caps it.  A tap off the panel gets weight zero, then clamping.
            w_row = tl.maximum((w_p_r + 1.0) / 2.0
                               - _tl_abs(m - m_tap), 0.0)
            w_row = tl.minimum(w_row, l_max_r)
            w_row = tl.where((m_tap_i >= 0) & (m_tap_i < num_rows), w_row, 0.0)
            if COEFF_POWER == 2:
                w_row = w_row * w_row
            # The amplitude is applied after the power, already raised to the
            # same power by the wrapper.  Do not fold it into the weight above.
            w_row = w_row * scale_pow
            m_row = tl.minimum(tl.maximum(m_tap_i, 0), num_rows - 1)

            row_vals = tl.zeros((BLOCK_P, BLOCK_L), dtype=tl.float32)
            for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
                # The trapezoid rule zeroes the weight where the unclipped tap
                # left the detector, then clamps the index.
                n_tap = centers + (tc - PSF_RADIUS)
                w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                    - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                w_chan = tl.minimum(w_chan, clip_c) * weight_scale
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

    # Slices at or past the real count contribute nothing.
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
    """Triton multiaxis back body, replacing
    :func:`mbirtorch.multiaxis_parallel._multiaxis_back_view_batch`.

    It has the same signature and the same (P, band) return, written fresh each
    call so a driver may accumulate into it in place.  The band is rounded up
    to a multiple of 16 for the launch, and the return is the real band slice,
    which is a strided view when the band is not such a multiple.  This body
    must stay eager, which the ``torch.compiler.disable`` decorator and the
    ``_mbirtorch_no_compile`` marker below enforce for the two kinds of caller.
    ``plan`` is ignored.
    """
    if triton is None:
        raise RuntimeError('the Triton multiaxis back body was called without '
                           'triton installed; a caller must keep the torch '
                           'body where no triton is available.')
    # The kernel has static branches for powers 1 and 2 only.
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
    # These are the torch body's own builders, so the kernel's tap centers and
    # weights come from the same computation.
    n_p, centers, w_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    m0, slope, w_p_r, l_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)
    scale_pow = scaling ** coeff_power

    num_views, num_pixels = n_p.shape
    band_len = int(num_slices if band_slices is None else band_slices)
    # The kernel is launched at a band rounded up to a multiple of 16.  Every
    # use of the band argument takes this value.
    launch_band = padded_kernel_width(band_len)
    # The views are made channel major, so the per tile gather is contiguous.  This
    # copy is not padded, because the kernel clamps every sinogram address it forms.
    sino_t = sino_batch.permute(0, 2, 1).contiguous()
    # These three arrays are per (view, pixel), at 12 bytes per pixel per view.
    contract = [t.contiguous() for t in (n_p, centers, m0)]
    # The reshape both converts to per view scalars and checks that they are
    # per view.
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
    launch_key = ('maback', sino_batch.device.index, int(psf_radius),
                  int(coeff_power), block_p, block_l,
                  int(num_views), int(num_pixels), int(num_channels),
                  int(num_rows_r), launch_band, int(slice_start),
                  int(num_slices))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch runs on the tensors' own device, and the device leads the
    # launch key.
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
    # The padded columns are sliced off.  The result is a strided view whose
    # rows are each still contiguous.
    return out[:, :band_len]


# The driver reads this marker in maybe_compile.
_multiaxis_back_view_batch_triton._mbirtorch_no_compile = True


def _multiaxis_back_view_batch_cost(num_pixels, num_band_rows, args):
    """Return the bytes resident per view in one back kernel batch, and this
    kernel's nominal view chunk.

    The driver reads this through the ``_view_batch_cost`` attribute.  One view
    holds three per (view, pixel) arrays at 12 bytes each pixel, plus the
    channel major copy of its sinogram plane.  The remaining contract terms are
    one float per view and are not charged.  Tensors whose size does not depend
    on the batch are not charged either.
    """
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
    """One program per (pixel block, row chunk, view) of the sinogram.

        out[v, c, m] += sum over pixels p, channel taps tc, slices k of
                        Wchan[v, p, tc] * Wrow[v, p, k, m] * scaling[v]
                        * values[p, k - slice_start]

    The detector side vertical fan runs first, then the horizontal fan scatter.
    The torch body computes the same sum from the other side by scattering each
    slice into the rows around m_p(k).  This kernel gathers, so the slices
    reaching one row must be enumerated, and ``SLICE_RADIUS`` bounds that
    enumeration.  See :func:`_multiaxis_slice_tap_radius` for the bound.

    The view is a grid axis because each view writes its own output plane.
    The pixel block is the fast axis, so programs that run together read the
    same rows of ``values`` and hit neighboring channels.  The view axis is
    last, and it stays well under the CUDA limit of 65535 on grid dims 1 and 2.

    The output is channel major (Vb, C, R), which puts the row axis on the
    contiguous stride.  It must reach the kernel zeroed, because the atomics
    accumulate into it.  The wrapper transposes the view on return.

    Pixel lanes beyond ``num_pixels`` have their atomics masked off, so their
    contract values may be any finite number.  The only divisor here is the per
    view slope, which no padded pixel lane can zero.  ``slope_floor`` guards it
    against a degenerate view.

    ``num_rows`` and ``out_row_stride`` are both the padded row count the
    wrapper launches at.  Row lanes past the real rows gather with a clamped
    and masked index, and their atomics land in extra output rows that the
    wrapper slices off.  Masking those lanes off instead would reach the same
    values, but it costs a factor of 3.1, because Triton compiles a slower
    kernel for a bound it cannot prove is a multiple of 16.
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
    # These are per view scalars.  The horizontal footprint depends on the
    # azimuth alone, and the vertical terms on the azimuth and elevation.
    w_p_c = tl.load(w_p_c_ptr + v)
    weight_scale = tl.load(weight_scale_ptr + v)
    clip_c = tl.minimum(w_p_c, 1.0)
    slope = tl.load(slope_ptr + v)
    w_p_r = tl.load(w_p_r_ptr + v)
    l_max_r = tl.load(l_max_ptr + v)
    scaling = tl.load(scaling_ptr + v)

    # Vertical fan, detector side, from rows to fractional slice indices.  The floor
    # on the divisor keeps a degenerate view from sending a NaN into the conversions.
    m_f = r_offs.to(tl.float32)                                     # (BLOCK_R,)
    k_m = (m_f[None, :] - m0[:, None]) / tl.maximum(slope, slope_floor)
    k_center = _tl_floor(k_m + 0.5)
    # The center is bounded before those conversions, because float to int is undefined
    # past the int32 range.  The bounds are tight: at the low bound the highest tap is
    # slice_start - 1, and at the high bound the lowest tap is slice_start + band_len.
    k_center = tl.minimum(tl.maximum(k_center, k_center_lo), k_center_hi)

    det_col = tl.zeros((BLOCK_P, BLOCK_R), dtype=tl.float32)
    for tk in _tap_range(0, 2 * SLICE_RADIUS + 1):
        k_ind = k_center + (tk - SLICE_RADIUS)
        k_ind_i = k_ind.to(tl.int32)
        # This is the same vertical weight the torch body forms.  The
        # footprint sets the width and the clip bound caps it.
        m_p = m0[:, None] + slope * k_ind
        w_row = tl.maximum((w_p_r + 1.0) / 2.0 - _tl_abs(m_p - m_f[None, :]),
                           0.0)
        w_row = tl.minimum(w_row, l_max_r)
        # The weight is zeroed where the tap left the band, and the index is then
        # clamped.  The same test drops slices at or past the real slice count.
        live = ((k_ind_i >= slice_start) & (k_ind_i < slice_start + band_len)
                & (k_ind_i < num_slices))
        w_row = tl.where(live, w_row, 0.0)
        # The clamp keeps the address legal.  Carrying ``live`` into the load
        # mask saves the read of a tap that already carries zero weight.
        k_local = tl.minimum(tl.maximum(k_ind_i - slice_start, 0), band_len - 1)
        vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * band_len
                       + k_local, mask=tile_mask & live, other=0.0)
        # The amplitude multiplies the value, not the weight.  The torch body does the
        # same, and the two must agree for the back kernel to remain the adjoint.
        det_col = det_col + w_row * (vals * scaling)

    # Horizontal fan scatter, one atomic add per tap.
    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
        # This applies the trapezoid rule in the channel direction.  Taps outside the
        # detector drop out of the atomic mask instead of being added as zeros.
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
    """Triton multiaxis forward body, replacing
    :func:`mbirtorch.multiaxis_parallel._multiaxis_forward_view_batch`.

    It has the same signature and the same (Vb, R, C) return, zeroed each call
    because the kernel accumulates into it with atomics.  The forward's band
    rides in the columns of ``values``, so there is no ``band_slices``
    argument.  The detector row count is rounded up to a multiple of 16 for the
    launch, and that padded count is the output's row stride, the grid's row
    extent, and the bound the kernel masks its row lanes against.  The return
    is the real row slice.  ``values`` needs no padding, because the slice
    gather is clamped and masked into the band it was handed.

    This body must stay eager, which the ``torch.compiler.disable`` decorator
    and the ``_mbirtorch_no_compile`` marker below enforce for the two kinds of
    caller.  ``values`` is (P, L).  When L is less than num_slices it is the
    slice band starting at ``slice_start``, and the slice to row map stays
    anchored on the full num_slices center, so summing the outputs over a
    tiling of the slice axis reproduces the unbanded projection.  ``plan`` is
    ignored.
    """
    if triton is None:
        raise RuntimeError('the Triton multiaxis forward body was called '
                           'without triton installed; a caller must keep the '
                           'torch body where no triton is available.')
    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    # These are the torch body's own builders, so the kernel's tap centers and
    # weights come from the same computation.
    n_p, centers, w_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    m0, slope, w_p_r, l_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)

    # A geometry whose slice tap radius exceeds the cap has a vanishing slope,
    # which the gather cannot cover at a bounded trip count.
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
    # The output is allocated and the kernel launched at a detector row count rounded
    # up to a multiple of 16.  The geometry builders above keep the real row count.
    launch_rows = padded_kernel_width(int(num_rows_r))
    values = values.contiguous()
    # These three arrays are per (view, pixel), at 12 bytes per pixel per view.
    contract = [t.contiguous() for t in (n_p, centers, m0)]
    # The reshape both converts to per view scalars and checks that they are
    # per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale, slope, w_p_r, l_max, scaling)]
    # The output is channel major and zeroed, because the atomics accumulate
    # into it.  The return transposes each view.
    out = torch.zeros((num_views, num_channels, launch_rows), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(MULTIAXIS_FWD_BLOCK_P, num_pixels,
                         MULTIAXIS_FWD_MIN_TILE)
    block_r = _tile_size(MULTIAXIS_FWD_BLOCK_R, launch_rows,
                         MULTIAXIS_FWD_MIN_TILE)
    # The row grid covers the padded rows, because those lanes are live.
    grid = (-(-num_pixels // block_p), -(-launch_rows // block_r), num_views)
    # These are the bounds the kernel clamps the slice center to.
    k_center_lo = float(int(slice_start) - slice_radius - 1)
    k_center_hi = float(int(slice_start) + band_len + slice_radius)
    launch_key = ('mafwd', values.device.index, int(psf_radius), slice_radius,
                  block_p, block_r, int(num_views), int(num_pixels),
                  int(num_channels), int(num_rows_r), launch_rows, band_len,
                  int(slice_start), int(num_slices))
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch runs on the tensors' own device, and the device leads the
    # launch key.
    with torch.cuda.device(values.device), guard:
        _multiaxis_forward_kernel[grid](
            *contract, values, out,
            int(num_pixels), int(num_channels), launch_rows,
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
    # The extra detector rows are sliced off before the transpose.
    return out[:, :, :int(num_rows_r)].permute(0, 2, 1)


# The driver reads this marker in maybe_compile.
_multiaxis_forward_view_batch_triton._mbirtorch_no_compile = True


def _multiaxis_forward_view_batch_cost(num_pixels, band_len, args):
    """Return the bytes resident per view in one forward kernel batch, and
    this kernel's nominal view chunk.

    One view holds three per (view, pixel) arrays at 12 bytes each pixel, plus
    its zeroed channel major output plane.  That plane spans the full detector
    rows at the padded row count, whatever slice band the values carry.
    """
    plane_rows = padded_kernel_width(int(args['num_rows_r']))
    plane_bytes = 4 * int(args['num_channels']) * plane_rows
    return 12 * int(num_pixels) + plane_bytes, MULTIAXIS_FWD_VIEW_CHUNK


_multiaxis_forward_view_batch_triton._view_batch_cost = \
    _multiaxis_forward_view_batch_cost
