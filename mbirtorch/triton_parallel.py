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
# The Triton language shims are IMPORTED from the cone module rather than
# duplicated or hoisted to a third module.  The import graph stays acyclic
# (triton_parallel -> triton_cone -> cone_beam; the geometry classes import
# their kernel modules lazily, inside _view_batch_bodies), and the shims exist
# to absorb Triton API drift -- ``_tl_builtin``'s tl-vs-tl.math lookup and the
# static_range fallback -- which is a single moving target that should have a
# single home.  ``_COMPILED_LAUNCH_KEYS`` is deliberately shared too: every key
# leads with its kernel's name, so one set serves all four kernels and a
# cross-kernel false hit is impossible.
from .triton_cone import (_COMPILED_LAUNCH_KEYS, _jit, _tap_range, _tile_size,
                          _tl_abs, tl, triton)

_F32 = torch.float32

# ── H100 PINNED constants (from the 60-config sweep at both gate cells) ──────
# The register-pressure prediction held: with only ~3 live tiles per program
# (the cone back holds ~6), the taller BLOCK_R=256 back rectangle runs
# without spilling (50 regs/thread, 0 spills) and wins the 1024 cell.  The
# back winners split by cell -- BLOCK_R=64 wins the 512 cell and BLOCK_R=256
# wins the 1024 cell, each by about 9 and 24 percent over the other -- and
# one config is pinned anyway: (8, 256, 4, 1) sits within 0.6 percent of
# best at 1024, where the back kernel dominates the composed time, and its
# 9 percent concession at 512 is invisible in composition (the whole back
# call is under 0.8 ms against a 10.6 ms compiled body).  Isolated speedups
# at the pin: 13x over the compiled torch body at 512 and 7.8x at 1024,
# bit-exact across repeat launches (no atomics on the back path).
PARALLEL_BACK_BLOCK_P = 8
PARALLEL_BACK_BLOCK_R = 256
PARALLEL_BACK_NUM_WARPS = 4
# 1 stage = no software pipelining: the view loop is gather-bound, not
# dot-bound, and extra stages buy latency hiding only at more register
# pressure (the stages=2 twin of the pin measured within noise of it).
PARALLEL_BACK_NUM_STAGES = 1
# The smallest tile worth launching: a row band or pixel subset below this pads
# rather than shrinking further.
PARALLEL_BACK_MIN_TILE = 8
# The driver's nominal view chunk for this kernel's batches: the batch this
# body asks for when the model's view_batch_size is None (automatic).  The
# batching rule rides on the body (see _parallel_back_view_batch_cost and
# Projectors._effective_view_batch), because the torch bodies' gather-slab
# charge would force view batch 1 at large cells for a kernel that holds no
# such slab.  Swept beside the tile constants; the driver's transient budget
# may cap the realized batch below it.
PARALLEL_BACK_VIEW_CHUNK = 128

# The forward pin is the cone-seeded config, confirmed by its own sweep: best
# at the 1024 cell outright and within 3.5 percent of best at 512, where the
# forward is a 5 ms term.  Isolated it is 1.21x over the compiled body at 512
# and 0.78x at 1024 -- and composition reverses the 1024 loss (see the
# selection hook in parallel_beam.py): the composed both-kernels arm beats
# the back-only arm by 19-24 percent.
PARALLEL_FWD_BLOCK_P = 8
PARALLEL_FWD_BLOCK_R = 128
PARALLEL_FWD_NUM_WARPS = 8
# 1 stage: the tap loop is atomic-bound rather than dot-bound.
PARALLEL_FWD_NUM_STAGES = 1
PARALLEL_FWD_MIN_TILE = 8
# The forward's nominal view chunk (see PARALLEL_BACK_VIEW_CHUNK).
PARALLEL_FWD_VIEW_CHUNK = 128

# The sorted-contraction forward's pins: the mg33 spike's winner (32 pixels,
# a 16-channel window, 128 columns, 8 warps, 16-view chunks), where it read
# 3.97x over the tap kernel at the full mask and 2.8x to 3.9x at the VCD
# subset sizes, with the atomic adds down 31.6x (findings 1.30 in the plans
# repository).  The window is a hard tl.dot minimum (16) and the sorted
# spans measured 2 to 3 channels, so there is no headroom question in the
# window choice itself.
PARALLEL_SORTED_BLOCK_P = 32
PARALLEL_SORTED_WINDOW = 16
PARALLEL_SORTED_BLOCK_R = 128
PARALLEL_SORTED_NUM_WARPS = 8
PARALLEL_SORTED_NUM_STAGES = 1
# tl.dot needs every dimension at 16 or more, so the column tile never
# shrinks below 16 (the tap kernel's floor is 8).
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
    """One program per (pixel block, row chunk) of the output partial:

        out[p, r] = sum over views v, channel taps tc of
                    Wchan[v, p, tc] ** coeff_power * sino[v, c(v, p) + tc, r]

    -- the cone back kernel with its vertical fan deleted.  The row axis is
    inert geometry here (row r is slice r), so it rides as the vector axis and
    the gathered row band IS the output's slice band; a banded call simply
    hands the kernel fewer rows.

    The pixel block is the FAST grid axis so that concurrently scheduled
    programs gather from the same detector rows of the same view -- the L2
    residency the pallas grid ordering bought for these transaction-bound
    gathers.

    ``coeff_power`` rides on the CHANNEL weight, the only weight there is (the
    cone kernel splits it across the row and channel weights).  It is a
    constexpr branch, so power 2 costs one multiply and no divergence.

    Pixels beyond ``num_pixels`` ride as padded lanes: their loaded contract
    values are zeroed, which zeroes the tap weight, and their stores are
    masked (the poison-the-padding rule).

    The row axis works differently.  ``num_band_rows`` is the row count the
    WRAPPER launches, which is the band's real row count rounded up to a
    multiple of 16 (see :func:`mbirtorch._utils.padded_kernel_width`).  Row
    lanes between the two are ordinary live lanes: they load, they compute,
    and they store.  Two things make that safe.  The wrapper hands this
    kernel a sinogram copy allocated at the padded row count whose extra rows
    are zero, so those lanes read zeros and accumulate exactly zero.  And
    their stores land in the extra output columns, which the wrapper slices
    off before it returns.
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
        # Per-VIEW scalars (see the module docstring): the projected footprint
        # of a voxel depends on the view angle alone under parallel beam.
        w_p_c = tl.load(w_p_c_ptr + v)
        weight_scale = tl.load(weight_scale_ptr + v)
        clip = tl.minimum(w_p_c, 1.0)
        sino_view_ptr = sino_ptr + v.to(tl.int64) * sino_view_stride

        for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
            # The horizontal_fan trapezoid rule, in-kernel: zero the weight
            # where the unclipped tap left the detector, then clamp the index
            # (the zero-and-clamp convention).
            n_tap = centers + (tc - PSF_RADIUS)
            w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
            w_chan = tl.minimum(w_chan, clip) * weight_scale
            w_chan = tl.where((n_tap >= 0) & (n_tap < num_channels),
                              w_chan, 0.0)
            if COEFF_POWER == 2:
                w_chan = w_chan * w_chan
            n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
            # Channel-major views: the row axis walks the CONTIGUOUS axis, so a
            # tile's gather is a unit-stride read.
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
    """The Triton parallel back body: a drop-in replacement for
    :func:`mbirtorch.parallel_beam._parallel_back_view_batch` (same signature,
    same (P, rows) return, freshly written each call so the driver may
    accumulate into it in place).

    The row count is rounded up to a multiple of 16 before the launch,
    because Triton compiles a faster kernel for an integer argument it can
    prove divisible by 16.  The channel-major sinogram copy is then made at
    the padded row count with its extra rows zeroed, and the return is the
    real-row slice of a slightly wider output.  A row count that IS a
    multiple of 16 takes exactly the path it took before, with the same
    allocations.

    Eager python by construction, declared twice over because the two
    mechanisms cover different callers: ``torch.compiler.disable`` keeps dynamo
    out when a compiled region CALLS this body, and the
    ``_mbirtorch_no_compile`` marker set below keeps the driver's
    ``maybe_compile`` from compiling it DIRECTLY (torch.compile unwraps the
    disable decorator and would trace the launch anyway).  The hfan builder
    runs ONCE per call, outside every loop (the hoisted-builders rule).

    The band asserts are the torch body's, inherited rather than relaxed: a
    row-aligned geometry carries its band in the SINOGRAM's row axis, so the
    two-fan band keywords must stay at their defaults and the returned column
    count is the input's row count.  ``plan`` is accepted and ignored -- the
    sorted/CSR stream slot, not yet built.
    """
    if triton is None:
        raise RuntimeError('the Triton parallel back body was called without '
                           'triton installed; the selection in '
                           'ParallelBeamModel._view_batch_bodies should have '
                           'kept the torch body (see kernel_availability).')
    assert slice_start == 0 and band_slices is None
    # Powers other than 1 and 2 are outside the kernel's static branch (and
    # outside every caller in the package): delegate rather than diverge.
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
    # The row count the kernel is LAUNCHED at, rounded up to a multiple of 16
    # so that Triton compiles the faster specialization of it.  Every use of
    # the row argument takes this value -- the grid, the tile mask, the
    # sinogram row stride and the output row stride.  A row count that is
    # already a multiple of 16 gets its own value back, so every allocation
    # and every argument below is exactly what it was before this padding
    # existed.
    launch_rows = padded_kernel_width(num_band_rows)
    # Channel-major views, as in the torch body: the kernel's per-tile gather
    # walks the row axis, which is contiguous in this layout.
    if launch_rows == num_band_rows:
        sino_t = sino_batch.permute(0, 2, 1).contiguous()
    else:
        # This kernel's sinogram is band-sized and its gather is bounded by
        # the row argument alone, so a padded row lane would read past the
        # last real row.  The copy the wrapper already makes is therefore
        # made at the padded row count instead, with the extra rows set to
        # zero: a zero sinogram row contributes exactly zero to the sum, and
        # the sliced return below discards the columns it lands in.  The
        # extra cost is the zero fill, not a second pass over the data.
        sino_t = torch.empty(
            (int(sino_batch.shape[0]), int(sino_batch.shape[2]), launch_rows),
            dtype=sino_batch.dtype, device=sino_batch.device)
        sino_t[:, :, :num_band_rows] = sino_batch.permute(0, 2, 1)
        sino_t[:, :, num_band_rows:] = 0.0
    contract = [t.contiguous() for t in (n_p, centers)]
    # Per-view scalars, and the shape check that they really are per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    out = torch.empty((num_pixels, launch_rows), dtype=_F32,
                      device=sino_batch.device)

    block_p = _tile_size(PARALLEL_BACK_BLOCK_P, num_pixels,
                         PARALLEL_BACK_MIN_TILE)
    block_r = _tile_size(PARALLEL_BACK_BLOCK_R, launch_rows,
                         PARALLEL_BACK_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_rows // block_r))
    # The padded row count keys the launch, because it is the integer the
    # compilation is keyed on.
    launch_key = ('pback', sino_batch.device.index, int(psf_radius),
                  int(coeff_power), block_p, block_r,
                  int(num_views), int(num_pixels), int(num_channels),
                  launch_rows)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch must be bracketed on the tensors' device, and the device
    # leads the launch key -- see _cone_back_view_batch_triton (triton_cone),
    # whose comment carries the measured basis.
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
    # The padded columns hold values no caller reads, so they are sliced off.
    # The result is a strided view, which the driver's accumulation and the
    # cross-device reduce both handle: each row is still one contiguous run.
    return out[:, :num_band_rows]


# See the wrapper's docstring: the driver reads this marker in maybe_compile.
_parallel_back_view_batch_triton._mbirtorch_no_compile = True


def _parallel_back_view_batch_cost(num_pixels, band_rows, args):
    """Charged bytes resident per view in one back-kernel batch, and this
    kernel's nominal view chunk -- the driver's batching rule for this body,
    read through the ``_view_batch_cost`` attribute in
    ``Projectors._effective_view_batch``.

    One view of a batch holds the hfan contract at 16 bytes per
    (view, pixel): ``n_p`` (f32) and ``centers`` (i32) held, plus the
    builder's live intermediate and an expression temporary of the same
    footprint (``W_p_c`` and ``weight_scale`` ride as per-view scalars).  It
    also holds the channel-major copy of its sinogram plane (``sino_t``
    above), which the wrapper allocates at the PADDED row count, so the plane
    term reads the padded value.  Call-fixed tensors -- the (P, rows) output
    partial -- exist at any batch size, so the batch choice cannot control
    them and they are not charged, exactly as the torch-body budget never
    charged its own fixed outputs.  The charge is a counted estimate that
    protects the budget boundary; the chunk constant is the swept performance
    chooser, and the composed gates re-measure the real peaks."""
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
    """One program per (pixel block, column chunk, view) of the sinogram:

        out[v, c, r] += sum over pixels p, channel taps tc of
                        Wchan[v, p, tc] * values[p, r]

    -- the cone forward kernel's horizontal scatter alone, with the vertical
    fan deleted: under parallel beam a voxel cylinder's column r lands on
    detector row r whatever the view, so the (BLOCK_P, BLOCK_R) tile of
    ``values`` that the cone kernel had to BUILD from a slice tap loop is here
    simply LOADED, once per program, before the tap loop starts.

    Grid choice, inherited from the cone forward: the VIEW axis has to be a
    grid axis rather than an in-program loop, because the forward writes a
    separate output plane per view and a view loop would only serialize.  The
    PIXEL block is the fast axis so concurrently scheduled programs read the
    same rows of ``values`` and hit neighbouring channels of the same view with
    their atomics (neighbouring pixels project to neighbouring channels).  The
    view axis is last because it is the only small one; it inherits CUDA's
    65535 limit on grid dims 1 and 2, which the driver's view batch is nowhere
    near.

    One specialization this grid leaves on the table, for the sweep to measure
    rather than for this increment to assume: because ``values`` does not
    depend on the view here (it does for cone), moving the view axis into an
    in-program LOOP would read each values tile once instead of once per view,
    at the cost of a grid smaller by the view batch.  Whether that trade wins
    depends on whether the kernel is atomic-bound or load-bound, which is a
    measurement, not an argument -- measure, then specialize.

    The output is CHANNEL-MAJOR (Vb, C, R), the layout fan_forward_batch also
    accumulates in: it puts the row axis -- the kernel's vector axis -- on the
    contiguous stride, so one tile row's atomics land on consecutive addresses
    instead of striding by C.  The wrapper transposes the view on return, as
    the torch body does.

    Pixels beyond ``num_pixels`` ride as padded lanes: their atomics are
    masked off entirely (the poison-the-padding rule).

    The column axis works differently.  ``num_cols`` is the column count the
    WRAPPER launches, which is the real count rounded up to a multiple of 16
    (see :func:`mbirtorch._utils.padded_kernel_width`).  Column lanes between
    the two are ordinary live lanes: they load and their atomics land.  Two
    things make that safe.  The wrapper hands this kernel a ``values`` copy
    allocated at the padded column count whose extra columns are zero, so
    those lanes add exactly 0.0 wherever they land.  And they land in the
    extra output columns, which the wrapper slices off before it returns.
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
    # Per-VIEW scalars (see the module docstring).
    w_p_c = tl.load(w_p_c_ptr + v)
    weight_scale = tl.load(weight_scale_ptr + v)
    clip = tl.minimum(w_p_c, 1.0)

    # The voxel cylinders, read ONCE and held in registers across every tap:
    # ``values`` does not depend on the view or the tap, so a per-tap reload
    # would be the same bytes psf_width times over.
    vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * num_cols
                   + r_offs[None, :], mask=tile_mask, other=0.0)

    out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
    for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
        # The horizontal_fan trapezoid rule, in-kernel (as in the back kernel);
        # here the out-of-detector taps drop out of the atomic MASK rather than
        # being added as zeros, which is the same value and one less atomic.
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
    """The sorted-contraction forward: one program per (pixel block, column
    chunk, view chunk), with the pixels PRE-SORTED per view by channel
    center and the values rows reached through that view's permutation.

    Sorted, a tile's taps land in a narrow channel window, so the scatter
    becomes a small dense contraction: the tile's trapezoid weights form a
    (BLOCK_P, WINDOW) matrix, transpose(W) @ values accumulates the whole
    tile (a segmented reduction where many pixels share a channel), and the
    window lands with one atomic add per (channel, column) instead of one
    per (pixel, tap, column).  The contraction runs in the full-precision
    input mode; the tensor-core default rounds inputs to a 10-bit mantissa
    and would fail the 1e-5 value gates.

    A tile whose sorted span still exceeds the window -- a sparse pixel
    set is the ordinary cause -- takes the original per-tap block below,
    so correctness never rests on the span.  A view chunk past the batch's
    end clamps its view index for every address and masks its stores, so a
    batch of any length is safe (the driver's tail batches are shorter
    than the chunk).
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
        # The clamp keeps every address in bounds for a tail chunk; the
        # store masks carry v_ok, so a clamped iteration writes nothing.
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
            # The window weights are the tap path's trapezoid formula
            # evaluated at every window channel: the psf radius is chosen
            # so the trapezoid's support sits inside the taps, so the
            # window holds the same nonzero weights plus true zeros.
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
            # The span mask keeps the power-of-two window from inflating
            # the adds: lanes past the tile's real span carry zero weight
            # AND issue no atomic.
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
    """The Triton parallel forward body: a drop-in replacement for
    :func:`mbirtorch.parallel_beam._parallel_forward_view_batch` (same
    signature, same (Vb, rows, C) return, freshly zeroed each call because the
    kernel accumulates into it with atomics).

    The column count is rounded up to a multiple of 16 before the launch, for
    the reason :func:`_parallel_back_view_batch_triton` gives for its rows.
    ``values`` is then copied into a zero-padded array of that width and the
    return is the real-column slice of a slightly wider output.  A column
    count that IS a multiple of 16 takes exactly the path it took before,
    with the same allocations, which covers the production slice counts.

    Eager python by construction and declared twice over, for the two reasons
    :func:`_parallel_back_view_batch_triton` spells out, and with the same
    hoisted builder.

    ``values`` is (P, cols), and cols is the detector ROW count of the block
    this call produces -- rows track slices, so a slice band is a row band and
    needs no z anchor, which is what the inherited ``slice_start == 0`` assert
    states.  ``plan`` is accepted and ignored -- the sorted/CSR stream slot,
    not yet built.
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
    # The column count the kernel is LAUNCHED at, rounded up to a multiple of
    # 16 so that Triton compiles the faster specialization of it.  Every use
    # of the column argument takes this value -- the grid, the tile mask, the
    # values row stride and the output row stride.  A column count that is
    # already a multiple of 16 gets its own value back, so every allocation
    # and every argument below is exactly what it was before this padding
    # existed; the production slice counts (1008, 2016) are such values.
    launch_cols = padded_kernel_width(num_value_cols)
    if launch_cols == num_value_cols:
        values = values.contiguous()
    else:
        # A padded column lane would read past the last real column of
        # ``values``, so the copy the wrapper already makes is made at the
        # padded width instead, with the extra columns set to zero.  Those
        # lanes then add exactly 0.0 through the atomics, and the sliced
        # return below discards the columns they land in.
        padded_values = torch.empty((int(values.shape[0]), launch_cols),
                                    dtype=values.dtype, device=values.device)
        padded_values[:, :num_value_cols] = values
        padded_values[:, num_value_cols:] = 0.0
        values = padded_values
    if sorted_forward_enabled():
        # THE SORTED ROUTE (the default; findings 1.30 in the plans
        # repository).  Per view, the pixels are sorted by channel center
        # and the contract is gathered into that order, so the sorted
        # kernel's tiles sit in narrow channel windows and its contraction
        # replaces almost all of the atomic scatter.  The permutation maps
        # each sorted position back to its values row; the kernel gathers
        # the rows per view.  The sort computes per call; the orderings
        # depend only on (pixel set, view batch), so a memoization through
        # the ``plan`` slot is the recorded follow-up if the per-call
        # milliseconds ever matter.
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
    # Per-view scalars, and the shape check that they really are per view.
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    # Channel-major, zeroed: the atomics accumulate, and the return transposes
    # the view exactly as the torch body transposes fan_forward_batch's.
    out = torch.zeros((num_views, num_channels, launch_cols), dtype=_F32,
                      device=values.device)

    block_p = _tile_size(PARALLEL_FWD_BLOCK_P, num_pixels,
                         PARALLEL_FWD_MIN_TILE)
    block_r = _tile_size(PARALLEL_FWD_BLOCK_R, launch_cols,
                         PARALLEL_FWD_MIN_TILE)
    grid = (-(-num_pixels // block_p), -(-launch_cols // block_r),
            num_views)
    # The padded column count keys the launch, because it is the integer the
    # compilation is keyed on.
    launch_key = ('pfwd', values.device.index, int(psf_radius), block_p,
                  block_r, int(num_views),
                  int(num_pixels), int(num_channels), launch_cols)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    # The launch must be bracketed on the tensors' device, and the device
    # leads the launch key -- see _cone_back_view_batch_triton (triton_cone),
    # whose comment carries the measured basis.
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
    # The extra columns hold values no caller reads, so they are sliced off
    # before the transpose.
    return out[:, :, :num_value_cols].permute(0, 2, 1)


# See the back wrapper's docstring: the driver reads this marker in
# maybe_compile.
_parallel_forward_view_batch_triton._mbirtorch_no_compile = True


def _parallel_forward_view_batch_cost(num_pixels, num_value_cols, args):
    """The forward twin of :func:`_parallel_back_view_batch_cost`: one view
    holds the same 16-byte-per-(view, pixel) hfan contract and its zeroed
    channel-major output plane (the atomics' target).  ``values`` is
    call-fixed and not charged.

    The plane is allocated at the PADDED column count, because that is what
    the wrapper launches at, so the charge reads the padded value too.

    The sorted route adds per-(view, pixel) residents beside the 16-byte
    contract: the argsort's int64 order (8), the int32 permutation (4), and
    the gathered float32 and int32 contract copies (8).  Twenty bytes per
    (view, pixel), charged only while the route is on, so the code and the
    charge cannot disagree."""
    plane_bytes = (4 * int(args['num_channels'])
                   * padded_kernel_width(num_value_cols))
    per_view = 16 * int(num_pixels) + plane_bytes
    if sorted_forward_enabled():
        per_view += 20 * int(num_pixels)
    return per_view, PARALLEL_FWD_VIEW_CHUNK


_parallel_forward_view_batch_triton._view_batch_cost = \
    _parallel_forward_view_batch_cost
