"""The per-device peak memory ledger and the reconstruction preflight.

The ledger is a closed-form model of how many bytes a reconstruction holds
on each device, phase by phase.  It exists because a reconstruction that
cannot fit should fail in seconds with a readable message, instead of
launching and dying later inside the allocator.

The ledger answers ONE production question: how many devices should this
reconstruction spread across?  A reconstruction spreads onto a device only
when that device can hold its share, so the automatic device-count choice
needs a per-device peak BEFORE any large allocation, and needs it without
running the compiles it would otherwise wait for.  That is why the model is
closed form rather than a compile query: the torch updater is eager python,
so no single compiled artifact ever sees the cross-call lineup of live
tensors.

Scope.  The ledger gates only the AUTOMATIC multi-device path.  A single
device, an explicitly configured layout, and the non-CUDA backends get no
preflight: torch's caching allocator raises a fast, readable error on a
single-device overflow, so the fail-fast job is already done there, and an
explicitly requested device count is the caller's to request.  The one
retained case is a machine where NO device count fits, including one: the
answer to "which count" is then "none", and raising with the dominant phase
named beats launching a known-doomed run.

Calibration is a separate mode.  With ``MBIRTORCH_MEMORY_CALIBRATION=1`` the
ledger is computed at ANY device count, including one, and compared against
``torch.cuda.max_memory_allocated`` at the end of the reconstruction.  That
mode owns the peak counter (it resets it), so it is never on by default.

Two consumers share ONE view batch.  The projection drivers use
``Projectors.view_batch_charge`` to choose a view batch; the ledger calls
the same function so it prices the batch the driver would actually run.  The
charge excludes the call-fixed outputs by contract, so the ledger adds those
itself, per phase.  It also reprices the batch when the body is a torch body,
because there the driver's number is a nominal slab used to bound the batch
and not a statement of what the batch holds; see TORCH_BODY_VIEW_SLABS.
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import _sharding, tomography_utils

_F32_BYTES = 4
_INT64_BYTES = 8

# The qGGMRF prior's co-live cylinder count, which is the largest single
# modelling uncertainty in the ledger.  Under torch.compile the kernel holds
# four arrays across its in-slice loop (the central cylinders, the running
# gradient, the running hessian, and b_tilde_2_delta -- which is built, read
# once, and then never rebound, so it survives to the return), and each loop
# iteration adds five (the neighbor gather, the difference, the b_tilde_2,
# and the incoming buffers of the two accumulator rebinds).  The neighbor
# gather cannot fuse away, so the count does not collapse under compilation.
# Eager holds the b_tilde_by_definition chain's temporaries as well; see
# qggmrf_cylinder_count below, which picks between the two.
QGGMRF_CYLINDERS_COMPILED = 9
QGGMRF_CYLINDERS_EAGER = 16
# The proximal-map prior is pointwise: recon rows, prox rows, the difference,
# and the scaled result.
PROX_CYLINDERS = 4
# direction_worker's simultaneous maximum is six (prior gradient and hessian,
# the back-projected error shard, the forward gradient, the forward hessian,
# and the update direction); the seventh covers the hessian row gather, live
# while the forward hessian is formed.
DIRECTION_CYLINDERS = 7
# apply_worker holds the direction and the scaled direction.
APPLY_CYLINDERS = 2
# How many gathered column cylinders a forward on the column-gather path
# holds at once (TomographyModel._sparse_forward_project_columns).  The driver
# issues each batch's gather one batch ahead of the projection that reads it,
# so at the widest instant -- inside the gather that runs ahead -- a device
# holds three: the cylinder the projection is about to read, the pieces
# arriving from the slice-owners for the batch after it, and the concatenation
# those pieces are assembled into.
#
# The last batch of a pass has nothing to gather ahead of it, and a pass that
# fits in one batch never gathers ahead at all, so both hold two rather than
# three.  The charge covers the widest instant, which is the rule the ledger
# keeps: it may charge more than a run needs but never less.
COLUMN_GATHER_RESIDENTS = 3

# ── the denoiser's own counts ────────────────────────────────────────────────
# QGGMRFDenoiser runs its own sweep, so its per-subset counts are read from
# denoising.py rather than shared with the reconstruction counts above.  Only
# the prior's count is shared, because the denoiser calls the same qGGMRF
# kernel the reconstruction calls.
#
# How many subset cylinders the denoiser holds once that kernel has returned.
# The single-device updater, vcd_subset_denoiser, holds eight at its widest
# instant, which is the residual update at the end.  Those eight are the
# prior gradient, the prior hessian, the gathered residual, the forward
# gradient, the unscaled update direction, the direction scaled by the step
# size, the second scaled copy the residual update forms, and the updated
# residual itself.
#
# The sharded worker, terms_worker, holds seven.  Its widest instant is the
# division that forms the direction, where it holds the prior gradient, the
# prior hessian, the gathered residual, the forward gradient, the two
# operands of that division, and its result.  Eight is charged on both paths,
# because the ledger may charge more than a run needs but never less.
DENOISE_DIRECTION_CYLINDERS = 8
# How many the state application holds: the direction the subset produced,
# the direction scaled by the step size, and one temporary -- the negated
# step the residual subtracts, or the absolute value the ell-1 sum reduces.
DENOISE_APPLY_CYLINDERS = 3
# How many qGGMRF boundary columns one device holds while the sharded sweep
# exchanges halos.  The exchange rebinds the pass's halos only after it
# returns, so a device holds the two columns the previous pass left, the one
# this exchange has already received, and the outgoing column the move copies
# from.  The two end devices hold one halo rather than two, and this charges
# them four as well.
DENOISE_HALO_COLUMNS = 4

# Library workspace that torch allocates through its own caching allocator,
# and that the ledger's array enumeration therefore cannot see.  Measured as
# a FLAT 32 to 33 MiB across problem sizes whose peaks span 2.26 GiB to 26.68
# GiB.  The residual does not scale over that twelvefold range, so it is a
# fixed per-process allocation rather than a missing array term.  The size and
# the architecture match the cuBLAS workspace, which is 32 MiB per stream on
# this class of device.  Charged at 64 MiB, which covers the measurement with
# headroom at a cost of 2.8 percent at the smallest size measured and 0.2
# percent at the largest.
FIXED_DEVICE_OVERHEAD_BYTES = 64 * 2 ** 20

CALIBRATION_ENV_VAR = 'MBIRTORCH_MEMORY_CALIBRATION'
# A process-wide pin on the automatic device count.  It exists so that a test
# suite, a nightly, or a measurement script can be deterministic on a machine
# whose GPU count it does not control.  Setting it is as explicit as calling
# configure_devices: the count is not searched and is never reduced, while the
# empty-shard validation and the preflight still apply.
DEVICE_COUNT_ENV_VAR = 'MBIRTORCH_NUM_DEVICES'
# How many per-view slabs one view batch of a TORCH BODY holds.  A torch body
# is a projection body written as general torch code, which is what a geometry
# with no hand-written kernel runs.  A hand-written kernel body declares what
# one of its views costs; a torch body declares nothing, so the driver prices
# it at ONE nominal slab -- (view batch, pixels, columns) floats -- and that
# single slab is what the ledger used to charge.
#
# A torch body holds a whole loop of those slabs at once.  It walks the
# interpolation kernel one offset at a time, and each offset materializes an
# integer index array, a weight array and a gathered array of the slab's
# shape, none of which fuse away; the running output and the mapped centers
# stay live across the whole loop beside them.
#
# Measured 2026-08-10 on four H100s (job mg8), over the two geometries with no
# hand-written kernels, four problem sizes, and one, two and four devices.
# The runs whose measured peak is set by the projection itself need 12.9
# slabs to cover it, and no run needs more.  Charged at 14, eight percent
# above the tightest of those readings.
#
# ONE count covers both projection directions and both geometries, because the
# ledger cannot tell which body it holds: it sees only that the body declares
# no cost.  The count is a measured multiplier and not a count of named
# arrays: the two geometries plainly do not hold the same number of slabs --
# the runs of one need at most 7.0 where the other needs 12.9 -- and nothing
# in the plan distinguishes them, so the larger has to be charged to both.
# TORCH_BODY_CALIBRATION_BAND says what that costs the smaller.
TORCH_BODY_VIEW_SLABS = 14

# The band the modeled peak must land in against the measured peak.  The
# lower bound is the one that matters: a ledger that under-predicts would let
# a doomed run start, which is the failure this module exists to prevent.
CALIBRATION_BAND = (1.00, 1.30)
# The same band for a reconstruction whose projection bodies are torch bodies.
# It is far wider than the one above for two reasons, both measured rather
# than assumed.  One slab count has to cover two geometries that hold
# different numbers of slabs, since nothing in the plan distinguishes them.
# And two of the measured two-device runs peaked twice as high on one device
# as on the other from identical shards, which a per-device model built from
# shapes alone cannot reproduce: it must cover the higher device, so it
# over-charges the lower one by that factor.  The widest over-charge measured
# is 5.74x, on the lower device of one of those two runs.
TORCH_BODY_CALIBRATION_BAND = (1.00, 5.80)


class MemoryPreflightError(RuntimeError):
    """Raised when no device layout can hold this reconstruction."""


@dataclass
class PhaseCharge:
    """One phase's per-device bytes, with the terms that make them up."""
    name: str
    per_device: list
    terms: list = field(default_factory=list)   # [(term_name, [bytes/device])]

    def dominant_terms(self, dev_index, count=3):
        """The largest terms on one device, largest first."""
        ranked = sorted(self.terms, key=lambda t: -t[1][dev_index])
        return [(name, vals[dev_index]) for name, vals in ranked[:count]
                if vals[dev_index] > 0]


@dataclass
class Ledger:
    """The modeled peak, per device, as a maximum over phases."""
    devices: list
    phases: list
    # The ROR-masked pixel count the subset phases were built from, so a
    # consumer can map a measured subset size back to its granularity.
    num_pixels_full: int = 0

    def peak_bytes(self, dev_index):
        return max(p.per_device[dev_index] for p in self.phases)

    def per_device_peaks(self):
        return [self.peak_bytes(i) for i in range(len(self.devices))]

    def dominant_phase(self, dev_index):
        return max(self.phases, key=lambda p: p.per_device[dev_index])

    def format_table(self):
        """The phase-by-device table, for the verbose printout and the error."""
        width = max(28, max(len(p.name) for p in self.phases) + 2)
        head = 'phase'.ljust(width) + ''.join(
            f'{str(d):>14}' for d in self.devices)
        lines = [head, '-' * len(head)]
        for phase in self.phases:
            lines.append(phase.name.ljust(width) + ''.join(
                f'{_gb(b):>14}' for b in phase.per_device))
        lines.append('-' * len(head))
        lines.append('PEAK'.ljust(width) + ''.join(
            f'{_gb(b):>14}' for b in self.per_device_peaks()))
        return '\n'.join(lines)


def _gb(num_bytes):
    return f'{num_bytes / 2 ** 30:.2f} GB'


@dataclass
class LedgerPlan:
    """Everything the ledger math needs, and nothing that needs a device.

    Built from a live model by :func:`plan_from_model`, and built by hand in
    the tests -- the ledger must be checkable on CPU with synthetic budgets,
    so no field here may require CUDA.
    """
    # ── shapes and layout ────────────────────────────────────────────────────
    sinogram_shape: tuple                 # (V, R, C), the problem's shape
    recon_shape: tuple                    # (Rr, Rc, S), the problem's shape
    devices: list                         # one entry per device, in order
    view_blocks: list                     # views held, per device
    slice_blocks: list                    # slices held, per device
    sino_rows: int                        # the detector row count
    rows_track_slices: bool
    # ── pixel counts ─────────────────────────────────────────────────────────
    num_pixels_full: int                  # the ROR-masked set
    num_pixels_grid: int                  # the unmasked grid (the hessian's)
    granularities: tuple                  # subset counts the sequence visits
    partition_granularities: tuple        # every subset count built up front
    # ── what this call runs and supplies ─────────────────────────────────────
    # Which call the plan prices: 'recon' for a full reconstruction, 'direct'
    # for a direct reconstruction alone, 'denoise' for one QGGMRFDenoiser
    # sweep.  Each holds a different set of arrays.  A direct reconstruction
    # builds no prior, no hessian, no partitions and no loop state, and a
    # denoise builds no projector, no view batch and no hessian at all.  They
    # are therefore separate plans rather than one plan with the extra terms
    # zeroed.
    workload: str = 'recon'
    weights_supplied: bool = False
    fm_hessian_supplied: bool = False
    init_recon_supplied: bool = False
    resume: bool = False
    prox: bool = False
    positivity: bool = False
    helical: bool = False
    # Whether the hessian back-projects at the ROR-masked index set rather
    # than the full grid.  vcd_recon does; a direct call to the public method
    # does not, and neither does an unmasked model.
    hessian_masked: bool = False
    # ── knobs and model choices ──────────────────────────────────────────────
    forward_band: int = None
    back_band: int = None
    # The pixel-column batch the forward's column gather assembles at once,
    # or None when the forward walks slice bands instead.  One field rather
    # than a flag and a width, so the two can never disagree, and resolved by
    # the model in plan_from_model rather than re-derived here.
    column_pixel_batch: int = None
    qggmrf_cylinders: int = QGGMRF_CYLINDERS_COMPILED
    # (direction, num_pixels, band_cols) -> (view_batch, bytes_per_view), with
    # direction in {'forward', 'back'}.  Defaults to a no-charge model so a
    # hand-built plan can exercise the state terms alone.
    view_charge: object = None
    # Which of 'forward' and 'back' bind a torch body -- a body that declares
    # no per-view cost of its own, so the ledger prices its views itself (see
    # TORCH_BODY_VIEW_SLABS).  The two directions are named separately because
    # a model may bind a hand-written kernel one way and a torch body the
    # other.  Empty means both directions declare their own cost, which is
    # what a hand-built plan gets: its charge reads exactly as before.
    torch_body_directions: tuple = ()

    @property
    def n_devices(self):
        return len(self.devices)

    def batch_bytes(self, direction, num_pixels, band_cols):
        if self.view_charge is None:
            return 0
        view_batch, bytes_per_view = self.view_charge(
            direction, int(num_pixels), int(band_cols))
        return int(view_batch) * int(bytes_per_view)

    def band_length(self, dev_index, direction):
        """The slice-band length one owner streams, matching
        ``TomographyModel._slice_band_length``: the whole shard by default."""
        local_slices = self.slice_blocks[dev_index]
        fixed = self.forward_band if direction == 'forward' else self.back_band
        return min(int(fixed), local_slices) if fixed else local_slices


def estimate_peak_device_bytes(plan):
    """The modeled per-device peak for one reconstruction.

    Pure: no device is queried, nothing is allocated, and the result depends
    only on ``plan``.  That is what lets the widening rule price a device
    count the model is not configured for, and what lets the tests run the
    whole model on CPU.

    Which phases are emitted follows ``plan.workload``: a full reconstruction
    by default, the filter and single back projection of a direct
    reconstruction under ``'direct'``, and one denoiser sweep under
    ``'denoise'``.  All three share the charges below, so the plans cannot
    drift apart.

    Returns:
        Ledger: the phases and their per-device bytes.
    """
    n = plan.n_devices
    num_rows_dev, num_channels = plan.sino_rows, int(plan.sinogram_shape[2])
    rows_recon, cols_recon = int(plan.recon_shape[0]), int(plan.recon_shape[1])

    def sino_dev(i):
        return plan.view_blocks[i] * num_rows_dev * num_channels * _F32_BYTES

    def recon_dev(i):
        return rows_recon * cols_recon * plan.slice_blocks[i] * _F32_BYTES

    def cyl(i, num_pixels):
        return int(num_pixels) * plan.slice_blocks[i] * _F32_BYTES

    def is_view_owner(i):
        return plan.view_blocks[i] > 0

    def is_slice_owner(i):
        return plan.slice_blocks[i] > 0

    def per_dev(fn):
        return [int(fn(i)) for i in range(n)]

    # ── the shared projection terms ──────────────────────────────────────────
    def back_cols(i):
        """The back call's band_cols: its local sinogram's row count."""
        if n == 1:
            return int(plan.sinogram_shape[1])
        return (plan.band_length(i, 'back') if plan.rows_track_slices
                else num_rows_dev)

    def column_gather_slices():
        """The slice extent one column-gather call is handed: the WHOLE
        slice axis, because the gathered cylinder spans every slice-owner at
        once."""
        return sum(int(block) for block in plan.slice_blocks)

    def forward_call_pixels(num_pixels):
        """How many pixel columns ONE forward call is handed: every pixel of
        the pass by default, and one column batch on the column-gather path,
        which is what makes that path's per-call terms fall."""
        if plan.column_pixel_batch:
            return min(int(num_pixels), int(plan.column_pixel_batch))
        return int(num_pixels)

    def forward_cols(i):
        """The forward call's band_cols: its voxel columns."""
        if n == 1:
            return int(plan.recon_shape[2])
        if plan.column_pixel_batch:
            return column_gather_slices()
        return plan.band_length(i, 'forward')

    def band_slices(i, direction):
        """The slice extent one projection call is handed: the whole slice
        axis at one device, this owner's slice band under sharding, and the
        whole device-form axis again on the column-gather path."""
        if n == 1:
            return int(plan.recon_shape[2])
        if direction == 'forward' and plan.column_pixel_batch:
            return column_gather_slices()
        return plan.band_length(i, direction)

    def torch_body_batch(i, direction, num_pixels):
        """What one view batch of a TORCH BODY holds.

        The body sweeps two axes -- the detector rows and the slice band it
        was handed -- and every array in its interpolation loop spans the
        view batch, the pixels, and whichever of those two axes is wider.
        It holds TORCH_BODY_VIEW_SLABS of them at once, where the driver's
        nominal charge prices one.

        The view batch itself stays the driver's own choice: only what that
        batch is charged changes here, so the ledger and the driver still
        agree on how many views one body call takes.
        """
        if plan.view_charge is None:
            return 0
        cols = back_cols(i) if direction == 'back' else forward_cols(i)
        view_batch = int(plan.view_charge(direction, int(num_pixels), cols)[0])
        width = max(int(plan.sino_rows), int(band_slices(i, direction)))
        return (TORCH_BODY_VIEW_SLABS * view_batch * int(num_pixels)
                * width * _F32_BYTES)

    def back_batch(i, num_pixels):
        if not is_view_owner(i):
            return 0
        if 'back' in plan.torch_body_directions:
            return torch_body_batch(i, 'back', num_pixels)
        return plan.batch_bytes('back', num_pixels, back_cols(i))

    def forward_batch(i, num_pixels):
        if not is_view_owner(i):
            return 0
        # A call's own pixel count, which is the pass's on the banded path
        # and one column batch on the column-gather path.
        call_pixels = forward_call_pixels(num_pixels)
        if 'forward' in plan.torch_body_directions:
            return torch_body_batch(i, 'forward', call_pixels)
        return plan.batch_bytes('forward', call_pixels, forward_cols(i))

    def band_reduce(i, num_pixels):
        """The back reduce's co-residency on a slice-owner.

        ``sum_band_to_owner`` streams: it forms the running total for a band
        once on the owner, then adds each arriving partial one row slab at a
        time and frees the slab before the next one arrives.  At the widest
        instant the owner holds

          * the bands of its shard it has already reduced this pass and is
            holding for the concatenation, at most ``shard - band`` slices,
          * the running total for the band it is on, one band,
          * the partial it produced itself, which the driver keeps alive
            across the reduce, one band,
          * one slab per arriving partial, each bounded by
            ``_sharding.REDUCE_SLAB_BYTES``.

        That is ``shard + band`` slices of cylinder plus a bounded slab term,
        which at the default band -- the whole shard -- is TWO
        cylinder-shards.  So it now falls as 1/n with the device count.  The
        old materialize-then-sum form held n whole bands plus the running
        totals, which is the same number of bytes at every device count: it
        measured 1.5x a full-volume cylinder set at both two and four
        devices, and adding devices did not move it.

        The slab term does not shrink with the device count, but it is a
        fixed number of bytes rather than a share of the volume.  When a band
        is smaller than one slab the whole band moves in one piece, which is
        what the reduce always did, and this reads as the n + 1 bands that
        then really are live.
        """
        if n == 1 or not is_slice_owner(i):
            return 0
        band = plan.band_length(i, 'back')
        shard = plan.slice_blocks[i]
        row_bytes = int(band) * _F32_BYTES
        slab_rows = _sharding.reduce_slab_rows(int(num_pixels), row_bytes)
        return (int(num_pixels) * (int(shard) + int(band)) * _F32_BYTES
                + (n - 1) * slab_rows * row_bytes)

    def back_view_batches(i, num_pixels):
        """How many batches one worker's view loop runs, or None when this
        plan prices no batch (a hand-built plan with no cost model)."""
        local_views = plan.view_blocks[i]
        if local_views <= 0 or plan.view_charge is None:
            return None
        view_batch = int(plan.view_charge(
            'back', int(num_pixels), back_cols(i))[0])
        return max(1, -(-int(local_views) // max(1, view_batch)))

    def back_fixed(i, num_pixels):
        """The back view loop's live cylinder-shards.

        ``Projectors.sparse_back_project_view_range`` is ``block =
        back_body(...)`` then ``out.add_(block)`` then ``block = None``.
        Python evaluates the call before it rebinds, so the loop holds the
        accumulator and the incoming block -- ``min(2, view_batches)``.  The
        release is what makes it two: without it the outgoing block survives
        the next kernel as well, for three.

        The count of two comes from measurement, not from reading the code
        alone.  Measured multi-device runs hold somewhat fewer live blocks
        than the reading predicts, because a block is often freed partway
        through the next kernel; the shortfall is absorbed by the ``back
        batch`` charge beside it, which the same runs show to be 30 to 45
        percent larger than what is actually held.  Two blocks here plus that
        batch charge covers every measured multi-device peak.

        n == 1 stays at THREE.  The release removes the same array there, but
        the single-device charge already sits within about a percent of the
        measured peak, and dropping a whole cylinder from a charge that thin
        risks landing below the true peak if the peak instant is not exactly
        where this reading puts it.  The ledger may over-charge; it may not
        under-charge.  The third cylinder comes off n=1 only when a
        single-device measurement confirms the drop.

        This is charged on every VIEW owner: the workers run wherever there
        are views to project, not only where the bands land.
        """
        if n == 1:
            return 3 * cyl(i, num_pixels) if is_slice_owner(i) else 0
        if not is_view_owner(i):
            return 0
        batches = back_view_batches(i, num_pixels)
        live = 2 if batches is None else min(2, batches)
        return live * cyl(i, num_pixels)

    def back_own_band(i, num_pixels):
        """The band this device already finished, live from its own pass on.

        Each slice-owner keeps its reduced band in ``recon_tensors`` for the
        rest of the loop, so from its own pass onward it carries one extra
        cylinder-shard through every later pass's projection.  Real and
        unavoidable: measurement shows the extra cylinder appearing on a
        device as soon as it has owned a pass.
        """
        if n == 1 or not is_slice_owner(i):
            return 0
        return cyl(i, num_pixels)

    # ── the forward terms ────────────────────────────────────────────────────
    # These terms charge only arrays the code can be seen to allocate: no
    # phase carries a safety margin, and none may be added back.  The two that
    # replaced the old margin are the loop's second live block (forward_block)
    # measured at the block's real detector-row extent (forward_block_rows).
    # Checked against measured peaks on 2026-08-10 (four H100s, at two and
    # four devices, both geometries, weighted and unweighted), where the
    # forward projection was the phase that set the modeled peak on nearly
    # every run.
    #
    # The constraint these terms have to keep: every modeled peak must sit at
    # or above the measured one, and the thinnest margin measured was a
    # fraction of a percent, so they may not be trimmed casually.
    # Over-charging is bounded too -- CALIBRATION_BAND asks the model to stay
    # within 1.30x of the measurement -- so an unneeded term is also a defect.
    def forward_fixed(i):
        """The forward's assembled output.  A multi-device owner holds the
        per-band pieces AND their concatenation (a row-aligned geometry, one
        whose detector row r comes from recon slice r), or the running partial
        AND the incoming one (a two-fan geometry such as cone, where one slice
        projects onto many detector rows), so it pays twice.

        The COLUMN-GATHER forward holds one rather than two: its batches add
        into the owner's block from inside the projector's view loop, so there
        is no separate incoming block to hold beside it.  The charge stays at
        two anyway.  It is shared with the banded path, which really does hold
        both, and the ledger's rule is that it may charge more than a run needs
        but never less -- so the column-gather path is deliberately over-charged
        by one block here rather than given a term of its own."""
        if not is_view_owner(i):
            return 0
        return sino_dev(i) if n == 1 else 2 * sino_dev(i)

    def forward_band_copy(i, num_pixels):
        """The broadcast band the forward leaves resident on every projector.

        ``_sharding.broadcast_band_to_views`` copies the current slice-owner's
        band onto every view-owner, and the copy stays live for the whole of
        that band's projection.  One band is the whole shard by default, so
        the copy is a full cylinder-shard on each device, on top of the
        device's own shard.  Without this term the model falls below the
        measured peak on a large cone reconstruction at four devices.

        The column-gather path broadcasts no band at all, so this term is
        zero there and ``forward_column_cylinder`` charges what it holds
        instead.
        """
        if n == 1 or not is_view_owner(i) or plan.column_pixel_batch:
            return 0
        return cyl(i, num_pixels)

    def forward_column_cylinder(i, num_pixels):
        """The gathered cylinder the column-gather forward assembles.

        ``_sharding.gather_column_band`` moves one batch of pixel columns
        from every slice-owner and concatenates them, so what a view-owner
        holds is that batch by the WHOLE device-form slice axis -- and,
        unlike the band copy it replaces, that does not grow with the shard,
        so it does not grow with the problem at a fixed batch.  Three are live
        at the widest instant, because the driver gathers one batch ahead of
        the projection that reads it; see COLUMN_GATHER_RESIDENTS for which
        three.

        Measured 2026-08-10 on four H100s, job mg10: ONE such cylinder read
        7.9, 15.8 and 31.5 MiB at batches 2048, 4096 and 8192 at 1008 slices,
        which is the closed form exactly.
        """
        if n == 1 or not is_view_owner(i) or not plan.column_pixel_batch:
            return 0
        return (COLUMN_GATHER_RESIDENTS * forward_call_pixels(num_pixels)
                * column_gather_slices() * _F32_BYTES)

    def forward_view_batches(i, num_pixels):
        """How many batches one owner's forward view loop runs, or None when
        this plan prices no batch (a hand-built plan with no cost model).
        The counterpart of ``back_view_batches`` above, using the forward's
        own cost model."""
        local_views = plan.view_blocks[i]
        if local_views <= 0 or plan.view_charge is None:
            return None
        view_batch = int(plan.view_charge(
            'forward', forward_call_pixels(num_pixels), forward_cols(i))[0])
        return max(1, -(-int(local_views) // max(1, view_batch)))

    def forward_block_rows(i):
        """The DETECTOR-ROW extent of one forward view block.

        A row-aligned geometry's body sizes its output by the value columns it
        was handed -- ``_parallel_forward_view_batch_triton`` allocates
        ``(views, channels, num_value_cols)`` -- so a slice band yields the
        matching row band and the block shrinks with the band.  A TWO-FAN
        body's output spans the whole detector whatever band the values carry:
        ``_cone_forward_view_batch_triton`` allocates ``(views, channels,
        num_rows_r)`` and reads ``num_rows_r`` from the params, because one
        slice band lights up every row it projects onto.  So the block does
        NOT shrink with the band there, and charging it at the band instead
        would under-charge the cone forward by ``(rows - band)`` per view.
        """
        return forward_cols(i) if plan.rows_track_slices else plan.sino_rows

    def forward_block(i, num_pixels):
        """The view block the loop holds BESIDES the one the batch prices.

        ``Projectors.sparse_forward_project_view_range`` is ``block =
        fwd_body(...)`` then ``out[...] = block`` (or ``out[...].add_(block)``
        when the caller accumulates), with no release: python evaluates the next
        call before it rebinds ``block``, so the loop holds the outgoing block
        and the incoming one -- ``min(2, view_batches)`` blocks.  Which of the
        two arms runs does not change that count.  The back loop would hold the
        same two if it did not release its block explicitly.

        ONE of those two is already inside ``forward batch`` when the body
        declares its own cost.  A forward kernel body's output plane scales
        with the view batch, so its ``_view_batch_cost`` charges it per view
        and says so; the back body's cost model does not, its output being
        call-fixed at any batch.  Against a declared cost this term is
        therefore the REMAINDER -- one block while the loop runs more than a
        single batch, and nothing when it runs one, which is the whole live
        set there.

        A TORCH BODY declares nothing, and what the ledger charges for it in
        its place is the body's INTERNAL slab set, which does not include the
        output plane.  Nothing is already paid for there, so both blocks are
        charged.

        The batch follows the pixel count of THIS call, so the subset phases
        must pass their own subset size rather than the full index count --
        and on the column-gather path a call's pixel count is one column
        batch, which raises the view batch and with it this block.
        """
        if not is_view_owner(i):
            return 0
        batches = forward_view_batches(i, num_pixels)
        live = 2 if batches is None else min(2, batches)
        already_paid = 0 if 'forward' in plan.torch_body_directions else 1
        view_batch = 1
        if plan.view_charge is not None:
            view_batch = plan.view_charge('forward',
                                          forward_call_pixels(num_pixels),
                                          forward_cols(i))[0]
        return ((live - already_paid) * int(view_batch)
                * forward_block_rows(i) * num_channels * _F32_BYTES)

    # ── the direct recon's filter ────────────────────────────────────────────
    # Charged only by the 'direct' plan: inside a full reconstruction the
    # filter runs between phases that hold more than it does.
    def filter_row_weights(i):
        """The FDK cosine pre-weight, one detector plane per device.

        ``fdk_filter`` builds it and ``_apply_direct_recon_filter`` copies it
        onto every device (``row_weight.to(d)``).  The FBP filters pass none,
        so this over-charges them by one detector plane, which is one view of
        the sinogram.
        """
        return int(plan.sino_rows) * num_channels * _F32_BYTES

    def filter_row_batch(i):
        """What one batch of the filter's row loop holds.

        ``tomography_utils.apply_row_filter`` walks the shard
        ROW_FILTER_BATCH detector rows at a time, convolving in frequency
        space.  At the widest instant -- inside the inverse transform -- one
        batch holds the pre-weighted window, the window's real FFT, its
        product with the filter's transform, and the inverse transform's
        output.  The two frequency arrays are complex over the zero-padded
        length, and the inverse is real over that same length; the filtered
        sinogram the batches write into is charged separately, as the array
        it is.

        The batch is a fixed row count, so this term does not fall with the
        device count.
        """
        rows_in_shard = plan.view_blocks[i] * int(plan.sino_rows)
        batch = min(tomography_utils.ROW_FILTER_BATCH, rows_in_shard)
        # channels + (2 * channels - 1) taps - 1, the linear convolution
        # length apply_row_filter transforms at.
        padded = 3 * num_channels - 2
        per_row = (num_channels * _F32_BYTES
                   + 2 * (padded // 2 + 1) * (2 * _F32_BYTES)
                   + padded * _F32_BYTES)
        return batch * per_row

    # ── the persistent set ───────────────────────────────────────────────────
    # One sinogram-shaped weights term, never two: when the caller supplies
    # weights the hessian's weight array is a bare ALIAS of them, and when it
    # does not, the internally built all-ones sinogram is the only one.  It is
    # charged whenever either exists.
    # WHEN the weights array exists differs from WHETHER it exists.  A
    # supplied weights array is placed at the top of vcd_recon, so it is
    # resident from the direct recon onward.  The internally built all-ones
    # array is created inside the hessian block, so on an unweighted run
    # nothing weights-shaped exists before that.  Measurement confirms both:
    # an unweighted direct recon starts with one sinogram-shaped array live,
    # a weighted one with two.
    weights_resident = plan.weights_supplied or not plan.fm_hessian_supplied

    def weights_term(i):
        """The weights array from the hessian phase onward."""
        return sino_dev(i) if weights_resident else 0

    def supplied_weights_term(i):
        """The weights array in the phases BEFORE the hessian builds one."""
        return sino_dev(i) if plan.weights_supplied else 0

    persistent = {
        'error sinogram': per_dev(sino_dev),
        'weights': per_dev(weights_term),
        'flat recon': per_dev(recon_dev),
        'hessian diagonal': per_dev(recon_dev),
    }
    if plan.prox:
        persistent['prox input'] = per_dev(recon_dev)
    # The partitions and the cached full index set live on the lead device.
    partition_bytes = sum(
        g * math.ceil(plan.num_pixels_full / g) * _INT64_BYTES
        for g in plan.partition_granularities)
    partition_bytes += plan.num_pixels_full * _INT64_BYTES
    persistent['partitions (lead device)'] = [
        partition_bytes if i == 0 else 0 for i in range(n)]

    # The partitions and the index cache are built before the reconstruction
    # starts and live on the lead device for its whole duration, so they are a
    # base under EVERY phase, not only the loop.  The workspace term is named
    # separately because it is the only one of the two a direct reconstruction
    # also carries.
    workspace_term = ('library workspace', [FIXED_DEVICE_OVERHEAD_BYTES] * n)
    constant_terms = [
        ('partitions (lead device)',
         persistent.pop('partitions (lead device)')),
        workspace_term,
    ]
    constant_base = [sum(vals[i] for _name, vals in constant_terms)
                     for i in range(n)]
    persistent_total = [constant_base[i]
                        + sum(vals[i] for vals in persistent.values())
                        for i in range(n)]

    phases = []

    def back_phases(name, resident_terms, num_pixels, base, base_terms):
        """One sharded back projection, as its TWO consecutive sub-steps.

        The workers project and the reduce gathers, and they never run at the
        same time: the workers' locals die on return, and the reduce's copies
        do not exist until they do.  Summing the two would charge a peak that
        is never live -- measurement puts the sum several cylinders per device
        above the larger of the two sub-steps.  Both sub-phases are emitted
        and the per-device maximum over phases picks between them, exactly as
        the loop/scatter split does.

        The sub-phase names keep the parent name as a prefix, so any consumer
        matching on the parent (a preflight message, a calibration row, a
        test) still finds it.  At n == 1 there is no reduce, so the phase is
        emitted whole under the parent name and nothing about the
        single-device ledger moves.
        """
        worker_terms = list(resident_terms) + [
            ('back output', per_dev(lambda i: back_fixed(i, num_pixels))),
            ('finished own band',
             per_dev(lambda i: back_own_band(i, num_pixels))),
            ('back batch', per_dev(lambda i: back_batch(i, num_pixels))),
        ]
        reduce_term = ('band reduce',
                       per_dev(lambda i: band_reduce(i, num_pixels)))
        if n == 1:
            return [_phase(name, worker_terms + [reduce_term], n,
                           base=base, base_terms=base_terms)]
        reduce_terms = list(resident_terms) + [reduce_term]
        return [_phase(f'{name} [back workers]', worker_terms, n,
                       base=base, base_terms=base_terms),
                _phase(f'{name} [band reduce]', reduce_terms, n,
                       base=base, base_terms=base_terms)]

    # ── the direct plan ──────────────────────────────────────────────────────
    # A direct reconstruction is the filter and one back projection, and this
    # is all of it.  It builds no prior, no hessian diagonal, no partition
    # sequence and no reconstruction loop, so the only thing under its phases
    # is the library's own workspace, and the phases themselves are the ones
    # the full plan gives the same code.
    #
    # The device count is still chosen for a full recon; this plan is what the
    # capacity check that can REFUSE is made against.  See
    # TomographyModel._apply_device_policy.
    if plan.workload == 'direct':
        p_full = plan.num_pixels_full
        base_terms = [workspace_term]
        base = list(workspace_term[1])
        # The sinogram is placed at entry (_shard_sinogram) and the filter
        # writes a second array of the same shape (apply_row_filter's `out`),
        # which is the input the back projection then reads.
        residents = [
            ('sinogram', per_dev(sino_dev)),
            ('filtered sinogram', per_dev(sino_dev)),
        ]
        filter_terms = residents + [
            ('filter row weights', per_dev(filter_row_weights)),
            ('filter row batch', per_dev(filter_row_batch)),
        ]
        scatter_terms = residents + [
            ('back cylinders', per_dev(lambda i: cyl(i, p_full))),
            ('scatter buffer', per_dev(recon_dev)),
        ]
        if plan.helical:
            scatter_terms.append(('helical z-weight', per_dev(recon_dev)))
        phases.append(_phase('direct recon (filter)', filter_terms, n,
                             base=base, base_terms=base_terms))
        phases.extend(back_phases('direct recon (back loop)', residents,
                                  p_full, base, base_terms))
        phases.append(_phase('direct recon (scatter)', scatter_terms, n,
                             base=base, base_terms=base_terms))
        return Ledger(devices=list(plan.devices), phases=phases,
                      num_pixels_full=int(plan.num_pixels_full))

    # ── the denoise plan ─────────────────────────────────────────────────────
    # One QGGMRFDenoiser sweep, and this is all of it.  The denoiser's forward
    # model is the identity, so it has no projectors at all; its
    # create_projectors is a no-op.  Nothing here therefore charges a view
    # batch, a projection body, a hessian diagonal or a weights array.  Its
    # sinogram shape IS its image shape, so every term below is image-shaped
    # and none is sinogram-shaped.  It fixes one partition rather than a
    # sequence, so it builds exactly the granularities this plan names.
    #
    # The arrays are split by SLICE, which is how _shard_recon places a
    # recon-shaped array.  Every term therefore follows slice_blocks and none
    # follows view_blocks.
    if plan.workload == 'denoise':
        base_terms = [workspace_term]
        base = list(workspace_term[1])

        def halo_columns(i):
            """The qGGMRF boundary columns one device holds across a pass.

            ``_sharding.exchange_qggmrf_halos`` gives each shard the image
            slice just beyond each of its boundaries, as a ``(num_pixels,)``
            column on the shard's own device.  A single device runs the
            compiled sweep instead and exchanges nothing, so the term is zero
            there.  See DENOISE_HALO_COLUMNS for which columns are counted.
            """
            if n == 1:
                return 0
            return (DENOISE_HALO_COLUMNS * int(plan.num_pixels_grid)
                    * _F32_BYTES)

        def partition_indices(i):
            """The subset partition, which EVERY device holds whole.

            The sharded sweep copies the whole partition onto each device
            once rather than splitting it, because a subset's indices address
            the in-slice pixel grid and every shard updates those same pixels
            in its own slices.  One partition of g subsets is
            ``g x ceil(P / g)`` int64 values.  Charged from the partitions
            this plan BUILDS, which for a denoiser is the one partition the
            sweep visits and no other.
            """
            return sum(
                int(g) * math.ceil(plan.num_pixels_full / max(1, int(g)))
                * _INT64_BYTES for g in plan.partition_granularities)

        # What the sweep holds from the moment its state exists until it
        # returns.  The denoiser places the input image, clones it into the
        # working image, and forms the residual between the two, so three
        # image-shaped arrays are live throughout.  A caller-supplied initial
        # image is a fourth.  By default that argument aliases the input image
        # and costs nothing.
        #
        # The reshape into flat (pixels, slices) form allocates nothing more.
        # A shard arrives from its cross-device copy contiguous, and a
        # single-device image is contiguous as placed, so the reshape is a
        # view on either path.
        residents = [
            ('input image', per_dev(recon_dev)),
            ('init image', per_dev(
                lambda i: recon_dev(i) if plan.init_recon_supplied else 0)),
            ('working image', per_dev(recon_dev)),
            ('residual', per_dev(recon_dev)),
            ('subset indices', per_dev(partition_indices)),
            ('qggmrf halos', per_dev(halo_columns)),
        ]
        phases.append(_phase('denoise state placement', residents, n,
                             base=base, base_terms=base_terms))
        for granularity in plan.granularities:
            p_sub = math.ceil(plan.num_pixels_full / max(1, int(granularity)))
            # The prior and the update direction are consecutive rather than
            # co-live.  The kernel's own working set is dead before the
            # direction is formed, and the two arrays that survive the call
            # are the prior gradient and hessian, which both counts include.
            # The per-device maximum over phases picks between them.
            sub_phases = (
                ('prior', [('prior cylinders', per_dev(
                    lambda i: plan.qggmrf_cylinders * cyl(i, p_sub)))]),
                ('update direction', [('direction cylinders', per_dev(
                    lambda i: DENOISE_DIRECTION_CYLINDERS * cyl(i, p_sub)))]),
                ('state application', [
                    ('direction and scaled direction', per_dev(
                        lambda i: DENOISE_APPLY_CYLINDERS * cyl(i, p_sub)))]),
            )
            for name, terms in sub_phases:
                phases.append(_phase(
                    f'denoise subset {name} (granularity {granularity})',
                    residents + terms, n, base=base, base_terms=base_terms))
        # The convergence test reads the working image's ell-1 norm once per
        # pass, and the absolute value it reduces is a whole image-shaped
        # array on every device.
        phases.append(_phase(
            'denoise per-pass statistics',
            residents + [('image magnitude', per_dev(recon_dev))], n,
            base=base, base_terms=base_terms))
        return Ledger(devices=list(plan.devices), phases=phases,
                      num_pixels_full=int(plan.num_pixels_full))

    # ── phase B: the direct reconstruction ───────────────────────────────────
    # Runs only when no initial reconstruction was supplied.  Its full-index
    # back projection is the largest single projection of the run.
    if not plan.init_recon_supplied and not plan.resume:
        p_full = plan.num_pixels_full
        # The back LOOP and the SCATTER are consecutive, not co-live: the
        # driver's accumulator is freed into the scatter's input.  Charging
        # both together over-counted the direct recon by a recon-shaped array.
        loop_residents = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('filtered sinogram', per_dev(sino_dev)),
        ]
        scatter_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('filtered sinogram', per_dev(sino_dev)),
            ('back cylinders', per_dev(lambda i: cyl(i, p_full))),
            ('scatter buffer', per_dev(recon_dev)),
        ]
        if plan.helical:
            scatter_terms.append(('helical z-weight', per_dev(recon_dev)))
        # BOTH sub-peaks are emitted, and the per-device maximum over phases
        # picks between them.  Picking one whole sub-phase by its cross-device
        # total would under-charge a device where the other sub-phase is the
        # larger one, which is the direction this module may not err in.
        phases.extend(back_phases('direct recon (back loop)', loop_residents,
                                  p_full, constant_base, constant_terms))
        phases.append(_phase('direct recon (scatter)', scatter_terms, n,
                             base=constant_base, base_terms=constant_terms))

    # ── phase C: the initial error state ─────────────────────────────────────
    # Two sub-peaks: the forward projection of the initial volume, then the
    # formation of the error sinogram.  They hold different arrays.
    if not plan.resume:
        p_full = plan.num_pixels_full
        forward_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('init recon', per_dev(recon_dev)),
            ('voxel gather', per_dev(lambda i: cyl(i, p_full))),
            ('broadcast band', per_dev(lambda i: forward_band_copy(i, p_full))),
            ('column cylinder', per_dev(
                lambda i: forward_column_cylinder(i, p_full))),
            ('forward output', per_dev(forward_fixed)),
            ('forward block', per_dev(lambda i: forward_block(i, p_full))),
            ('forward batch', per_dev(lambda i: forward_batch(i, p_full))),
        ]
        phases.append(_phase('initial forward projection', forward_terms,
                             n, base=constant_base,
                             base_terms=constant_terms))
        # The error sinogram is formed while the sinogram and the projection
        # are both live; the projection is then freed and the initial volume
        # is briefly doubled by its scaling.
        # The single-device branch binds `weighted_fwd = weights * fwd` for
        # its two dot products and now releases it before the error sinogram
        # is formed.  While it is alive it is co-live with the product
        # temporary of `torch.sum(weighted_fwd * fwd)`, which is its own
        # sub-peak and is modelled below.  The sharded branch has neither
        # array: it fuses the weights into per-shard dot products whose
        # locals die on worker return.
        weighted_fwd = per_dev(
            lambda i: sino_dev(i) if (n == 1 and plan.weights_supplied) else 0)
        dot_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('forward projection', per_dev(sino_dev)),
            ('weighted forward projection', weighted_fwd),
            ('dot product temporary', per_dev(sino_dev)),
            ('init recon', per_dev(recon_dev)),
        ]
        error_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('forward projection', per_dev(sino_dev)),
            # `error = sinogram - alpha * fwd` allocates the scaled projection
            # and then the difference, so both are live at the assignment.
            ('alpha-scaled projection', per_dev(sino_dev)),
            ('error sinogram', per_dev(sino_dev)),
            ('init recon', per_dev(recon_dev)),
        ]
        scale_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('error sinogram', per_dev(sino_dev)),
            ('init recon (x2, scaling)', per_dev(lambda i: 2 * recon_dev(i))),
        ]
        phases.append(_phase('initial dot products', dot_terms, n,
                             base=constant_base, base_terms=constant_terms))
        phases.append(_phase('error sinogram formation', error_terms, n,
                             base=constant_base, base_terms=constant_terms))
        phases.append(_phase('init recon scaling', scale_terms, n,
                             base=constant_base, base_terms=constant_terms))

    # ── phase D: the hessian diagonal ────────────────────────────────────────
    # Charged at the UNMASKED grid count, which is the one place the ledger
    # does not use the ROR-masked set.
    if not plan.fm_hessian_supplied:
        # The masked path back-projects the ROR set and scatters it into a
        # zero-filled volume; the dense path back-projects the whole grid and
        # reshapes.  The two differ in the pixel count AND in whether the
        # scatter's co-residency exists at all.
        p_hess = (plan.num_pixels_full if plan.hessian_masked
                  else plan.num_pixels_grid)
        hessian_residents = [
            ('error sinogram', per_dev(sino_dev)),
            ('hessian weights', per_dev(weights_term)),
            ('init recon', per_dev(recon_dev)),
        ]
        phases.extend(back_phases('hessian diagonal', hessian_residents,
                                  p_hess, constant_base, constant_terms))
        if plan.hessian_masked:
            # The scatter holds the masked cylinders and the zero-filled
            # volume at once.  It is a separate sub-peak from the back loop,
            # and it must stay below it: the loop's three cylinders at the
            # masked count exceed one cylinder plus one volume whenever the
            # mask keeps more than half the grid, which an inscribed ellipse
            # does.  The per-device maximum over phases enforces this rather
            # than assuming it.
            scatter_terms = [
                ('error sinogram', per_dev(sino_dev)),
                ('hessian weights', per_dev(weights_term)),
                ('init recon', per_dev(recon_dev)),
                ('hessian cylinders', per_dev(lambda i: cyl(i, p_hess))),
                ('hessian scatter volume', per_dev(recon_dev)),
            ]
            phases.append(_phase('hessian scatter', scatter_terms, n,
                                 base=constant_base,
                                 base_terms=constant_terms))

    # ── the per-iteration statistics ─────────────────────────────────────────
    # This phase has to be charged rather than assumed small: on an unweighted
    # run it is the peak.  Its transient measures EXACTLY two sinogram-shaped
    # arrays at the largest sizes tested, which is the two squared-error
    # products; the recon L1 fuses into its own reduction and materializes
    # nothing.
    phases.append(_phase(
        'per-iteration statistics',
        [('squared-error products', per_dev(lambda i: 2 * sino_dev(i)))],
        n, base=persistent_total,
        base_terms=constant_terms + list(persistent.items())))

    # ── phase E: the subset step, per granularity in the sequence ────────────
    prior_cylinders = PROX_CYLINDERS if plan.prox else plan.qggmrf_cylinders
    for granularity in plan.granularities:
        p_sub = math.ceil(plan.num_pixels_full / max(1, int(granularity)))
        index_bytes = per_dev(lambda i: p_sub * _INT64_BYTES)
        sub_phases = {
            'prior': [
                ('prior cylinders', per_dev(
                    lambda i: prior_cylinders * cyl(i, p_sub))),
            ],
            # The back projection carries only its RESIDENTS here: the two
            # sub-steps and their terms are built by back_phases below.
            'back projection': [
                ('prior gradient and hessian',
                 per_dev(lambda i: 2 * cyl(i, p_sub))),
                ('weighted error sinogram', per_dev(
                    lambda i: sino_dev(i) if plan.weights_supplied else 0)),
            ],
            'update direction': [
                ('direction cylinders', per_dev(
                    lambda i: DIRECTION_CYLINDERS * cyl(i, p_sub))),
            ],
            'delta forward projection': [
                ('update direction', per_dev(
                    lambda i: (2 if plan.positivity else 1) * cyl(i, p_sub))),
                ('delta sinogram', per_dev(sino_dev)),
                ('forward assembly', per_dev(
                    lambda i: sino_dev(i) if n > 1 and is_view_owner(i) else 0)),
                ('broadcast band', per_dev(
                    lambda i: forward_band_copy(i, p_sub))),
                ('column cylinder', per_dev(
                    lambda i: forward_column_cylinder(i, p_sub))),
                ('forward block', per_dev(lambda i: forward_block(i, p_sub))),
                ('forward batch', per_dev(lambda i: forward_batch(i, p_sub))),
            ],
            'state application': [
                ('direction and scaled direction',
                 per_dev(lambda i: APPLY_CYLINDERS * cyl(i, p_sub))),
                ('delta sinogram', per_dev(sino_dev)),
            ],
        }
        loop_base_terms = constant_terms + list(persistent.items())
        for name, terms in sub_phases.items():
            all_terms = terms + [('subset indices', index_bytes)]
            phase_name = f'subset {name} (granularity {granularity})'
            if name == 'back projection':
                phases.extend(back_phases(phase_name, all_terms, p_sub,
                                          persistent_total, loop_base_terms))
                continue
            phases.append(_phase(phase_name, all_terms, n,
                                 base=persistent_total,
                                 base_terms=loop_base_terms))

    # Every phase before the loop carries its own live set, which already
    # includes whatever part of the persistent set exists at that point.
    return Ledger(devices=list(plan.devices), phases=phases,
                  num_pixels_full=int(plan.num_pixels_full))


def _phase(name, terms, n, base=None, base_terms=None):
    totals = [sum(vals[i] for _, vals in terms) for i in range(n)]
    if base is not None:
        totals = [totals[i] + base[i] for i in range(n)]
        terms = list(base_terms or []) + list(terms)
    return PhaseCharge(name=name, per_device=totals, terms=list(terms))


# ── building a plan from a live model ────────────────────────────────────────
def qggmrf_cylinder_count(model):
    """The prior's charged cylinder count for THIS process.

    The compiled count is charged while compilation is working.  Once a
    qGGMRF compile has fallen back -- ``maybe_compile`` records every
    fallback and then rebinds permanently to eager -- the eager count is
    charged instead, because eager also materializes the surrogate chain's
    temporaries and under-charging is the one direction the ledger may not
    err in.
    """
    from . import projectors
    if not model.compile_enabled:
        return QGGMRF_CYLINDERS_EAGER
    for key in projectors._COMPILE_ERRORS:
        if 'qggmrf' in key:
            return QGGMRF_CYLINDERS_EAGER
    return QGGMRF_CYLINDERS_COMPILED


def plan_from_model(model, devices, workload='recon', partition_sequence=None,
                    weights=None, init_recon=None, fm_hessian=None,
                    prox_input=None, init_error_sinogram=None):
    """Build a :class:`LedgerPlan` for ``model`` over a CANDIDATE device list.

    The device list is an argument rather than a reading of the model's own
    placement, because the widening rule prices counts the model is not
    configured for.  The placements are rebuilt here from the current params,
    so a geometry change cannot leave a stale axis length behind.

    ``workload`` names the call the plan is for: ``'recon'`` (the default)
    prices a full reconstruction, ``'direct'`` prices a direct reconstruction
    -- the filter and one back projection, with none of the prior, hessian,
    partition and loop state a full reconstruction holds -- and ``'denoise'``
    prices one QGGMRFDenoiser sweep.

    A denoiser is read differently in three places, because it is built
    differently.  It has no projectors, so no per-view cost model is built and
    no projection body is asked what it costs.  Asking would raise, because a
    denoiser defines no bodies.  It also builds one partition rather than a
    sequence, so the plan names the single granularity the sweep visits.  And
    its sinogram shape is its image shape, which is checked here rather than
    assumed.  A plan built from a model where the two differ would price the
    wrong arrays.
    """
    sinogram_shape = tuple(int(s) for s in model.get_params('sinogram_shape'))
    recon_shape = tuple(int(s) for s in model.get_params('recon_shape'))
    devices = [torch.device(d) for d in devices]
    denoising = workload == 'denoise'
    if denoising and sinogram_shape != recon_shape:
        raise ValueError(
            "the 'denoise' workload prices a QGGMRFDenoiser, whose "
            'sinogram_shape is its image shape.  This model has '
            f'sinogram_shape {sinogram_shape} and recon_shape {recon_shape}.')

    sino_placement = _sharding.Placement(devices, axis=0,
                                         axis_len=sinogram_shape[0])
    recon_placement = _sharding.Placement(devices, axis=-1,
                                          axis_len=recon_shape[2])
    view_blocks = [end - start for _d, (start, end)
                   in sino_placement.shard_ranges()]
    slice_blocks = [end - start for _d, (start, end)
                    in recon_placement.shard_ranges()]
    rows_track_slices = bool(getattr(model, 'rows_track_slices', False))
    sino_rows = sinogram_shape[1]

    granularity = list(model.get_params('granularity'))
    if partition_sequence is None:
        partition_sequence = list(model.get_params('partition_sequence'))
    if denoising:
        # A denoise sweep builds and visits ONE partition: the one the FIRST
        # entry of the sequence names.  A reconstruction walks the whole
        # sequence and builds every granularity in the list, so reading the
        # sequence the reconstruction's way would charge partitions the
        # denoiser never builds and subset phases it never runs.
        index = int(partition_sequence[0]) if len(partition_sequence) else 0
        visited = [granularity[index]] if index < len(granularity) else []
        built = list(visited)
    else:
        visited = sorted({granularity[int(k)] for k in partition_sequence
                          if int(k) < len(granularity)})
        built = list(granularity)

    num_pixels_full = int(model.full_index_count())
    num_pixels_grid = recon_shape[0] * recon_shape[1]

    charge = None if denoising else _model_view_charge(model, len(devices))

    return LedgerPlan(
        sinogram_shape=sinogram_shape,
        recon_shape=recon_shape,
        devices=devices,
        workload=workload,
        view_blocks=view_blocks,
        slice_blocks=slice_blocks,
        sino_rows=int(sino_rows),
        rows_track_slices=rows_track_slices,
        num_pixels_full=num_pixels_full,
        num_pixels_grid=num_pixels_grid,
        granularities=tuple(visited) or (granularity[0],),
        partition_granularities=tuple(built) or (granularity[0],),
        weights_supplied=weights is not None,
        fm_hessian_supplied=fm_hessian is not None,
        init_recon_supplied=init_recon is not None,
        resume=init_error_sinogram is not None,
        prox=prox_input is not None,
        positivity=bool(model.get_params('positivity_flag')),
        helical=_is_helical(model),
        hessian_masked=model.get_params('use_ror_mask') is not False,
        forward_band=getattr(model, 'forward_project_slice_band', None),
        back_band=getattr(model, 'back_project_slice_band', None),
        # Both read from the model's own resolvers rather than re-derived
        # here: a charge that re-implements a driver rule is a charge that
        # can be left behind when the rule moves.  A denoiser has no forward
        # projection to resolve, so neither is asked.
        column_pixel_batch=(None if denoising else
                            (model._forward_pixel_batch()
                             if model._column_gather_forward() else None)),
        qggmrf_cylinders=qggmrf_cylinder_count(model),
        view_charge=charge,
        torch_body_directions=(() if denoising
                               else torch_body_directions(model)),
    )


def workload_covers(checked, incoming):
    """Whether a layout already checked for ``checked`` needs no fresh check
    before ``incoming`` runs on it.

    The recon plan charges every array the direct plan charges and a great
    deal besides, so a layout that passed a recon check passes a direct one.
    Nothing beyond that pair is claimed: a plan added later is unrelated to
    these until it has been priced against them.

    ``'denoise'`` is one such plan.  It covers nothing and is covered by
    nothing, because it holds arrays neither of the other two holds.  A
    denoise holds three image-shaped arrays at once, where a reconstruction
    holds one recon beside its sinogram-shaped set.  Equality below is what
    lets one denoise follow another with no fresh check.
    """
    return checked == incoming or (checked, incoming) == ('recon', 'direct')


def torch_body_directions(model):
    """Which projection directions this model runs as a torch body.

    A hand-written kernel body carries a ``_view_batch_cost`` attribute
    stating what one of its views holds; general torch code carries nothing,
    and the ledger prices those views itself (see TORCH_BODY_VIEW_SLABS).
    The two directions are asked separately, because a model may bind a
    kernel one way and a torch body the other, and because a kernel that is
    unavailable on this machine falls back to the torch body it replaced --
    the charge has to follow the body that will actually run.
    """
    fwd_body, back_body = model._view_batch_bodies()
    return tuple(name for name, body in (('forward', fwd_body),
                                         ('back', back_body))
                 if getattr(body, '_view_batch_cost', None) is None)


def _model_view_charge(model, n_devices):
    """A ``(direction, P, cols) -> (batch, bytes_per_view)`` closure over the
    bodies this model would actually bind."""
    projector_functions = model.projector_functions
    fwd_body, back_body = model._view_batch_bodies()
    args = model._view_batch_args()

    def charge(direction, num_pixels, band_cols):
        body = fwd_body if direction == 'forward' else back_body
        return projector_functions.view_batch_charge(
            body, num_pixels, band_cols, args, n_devices=n_devices)
    return charge


def _is_helical(model):
    params = model.params
    if 'view_params_array' not in params:
        return False
    try:
        import numpy as np
        shifts = np.asarray(model.get_params('view_params_array'))
        if shifts.ndim != 2 or shifts.shape[1] < 2:
            return False
        return float(np.max(shifts[:, 1]) - np.min(shifts[:, 1])) > 0
    except (TypeError, ValueError, IndexError):
        return False


# ── the device budget and the verdict ────────────────────────────────────────
def device_budget_bytes(device):
    """The bytes a NEW allocation can still obtain on one CUDA device.

    Two sources add up.  The driver's free memory is what CUDA will still
    hand out, and it already excludes memory held by other processes and
    memory this process has reserved.  The caching allocator's reserved but
    unused pool is the second, because torch releases cached segments before
    it reports an out-of-memory error.

    Memory held by other processes is therefore treated as unavailable, which
    is the right default: this check cannot evict a neighbor.  The reading is
    a snapshot, so a neighbor that grows afterwards can still exhaust the
    device, and fragmentation inside the reserved pool is invisible here.
    Both are what the margin is for.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    free, _total = torch.cuda.mem_get_info(device)
    reclaimable = (torch.cuda.memory_reserved(device)
                   - torch.cuda.memory_allocated(device))
    return int(free) + int(reclaimable)


def resident_credits(devices, arrays):
    """Bytes of ``arrays`` already resident on each device.

    An array the caller already placed on a device is counted in that
    device's allocated bytes, so it is already excluded from the budget
    above.  Charging it again in the demand would double-count it.
    """
    credits = [0] * len(devices)
    index = {}
    for position, device in enumerate(devices):
        index.setdefault(str(torch.device(device)), position)
    for array in arrays:
        for tensor in _iter_tensors(array):
            if tensor.device.type != 'cuda':
                continue
            position = index.get(str(tensor.device))
            if position is None:
                continue
            credits[position] += tensor.numel() * tensor.element_size()
    return credits


def _iter_tensors(array):
    if array is None:
        return
    if isinstance(array, _sharding.Shards):
        for tensor in array.tensors:
            yield tensor
    elif torch.is_tensor(array):
        yield array
    elif isinstance(array, (list, tuple)):
        for item in array:
            for tensor in _iter_tensors(item):
                yield tensor


def layout_fits(ledger, budgets, credits=None, margin=0.15):
    """Whether every device can hold its modeled share.

    Returns:
        (fits, rows): the verdict, and one (device, demand, budget) row per
        device for reporting.
    """
    peaks = ledger.per_device_peaks()
    credits = credits or [0] * len(peaks)
    rows, fits = [], True
    for i, device in enumerate(ledger.devices):
        demand = int((1.0 + margin) * max(0, peaks[i] - credits[i]))
        budget = budgets[i]
        ok = budget is None or demand <= budget
        fits = fits and ok
        rows.append((device, demand, budget))
    return fits, rows


def format_shortfall(ledger, rows, num_devices_tried, closest_count=None,
                     remedies=None):
    """The one readable error: what did not fit, which phase dominates it,
    and which knob moves that phase.

    ``rows`` describes the CLOSEST layout tried, not the last one, so the
    remedies are aimed at the shortfall a user can actually close.
    """
    lines = ['this reconstruction needs more memory than the available '
             'CUDA devices have free.', '']
    header = f'{"device":>10}{"modeled need":>16}{"available":>14}{"shortfall":>14}'
    lines.append(header)
    worst_index, worst_gap = 0, -1
    for i, (device, demand, budget) in enumerate(rows):
        gap = 0 if budget is None else demand - budget
        if gap > worst_gap:
            worst_index, worst_gap = i, gap
        shortfall = '-' if gap <= 0 else _gb(gap)
        available = '-' if budget is None else _gb(budget)
        lines.append(f'{str(device):>10}{_gb(demand):>16}'
                     f'{available:>14}{shortfall:>14}')

    phase = ledger.dominant_phase(worst_index)
    peak = ledger.peak_bytes(worst_index)
    lines += ['', f'The dominant phase on {ledger.devices[worst_index]} is '
                  f'"{phase.name}" at {_gb(phase.per_device[worst_index])} '
                  f'of the {_gb(peak)} peak.']
    top = phase.dominant_terms(worst_index)
    if top:
        named = ', '.join(f'{name} {_gb(value)}' for name, value in top)
        lines.append(f'Its largest terms are: {named}.')
    tried = (', '.join(str(c) for c in num_devices_tried)
             if isinstance(num_devices_tried, (list, tuple))
             else str(num_devices_tried))
    closest = ('' if closest_count is None
               else f'  The closest was {closest_count}, shown above.')
    lines += [
        '', f'Device counts tried, largest first: {tried}.  None fit.'
            + closest,
        '', 'Remedies, most effective first:',
        '  model.back_project_slice_band = <slices>   '
        '# shrinks every back projection transient',
        '                                             '
        '# that is sized by a band, on top of what',
        '                                             '
        '# more devices already save',
        '  model.view_batch_size = <views>            '
        '# caps the projector batch transient',
        '  model.set_params(granularity=[...])        '
        '# a finer coarsest granularity shrinks the',
        '                                             '
        '# per-subset transients',
        '  CUDA_VISIBLE_DEVICES=...                   '
        '# exclude a device another process is using',
    ]
    lines += list(remedies or [])
    lines += ['', 'To run anyway: model.skip_memory_preflight = True']
    return '\n'.join(lines)


# ── calibration ──────────────────────────────────────────────────────────────
def pinned_device_count():
    """The device count pinned by the environment, or None.

    A value that is not a positive integer is refused rather than ignored: a
    typo in a nightly's environment must not silently restore the automatic
    behavior the pin was set to prevent.
    """
    raw = os.environ.get(DEVICE_COUNT_ENV_VAR, '').strip()
    if not raw:
        return None
    try:
        count = int(raw)
    except ValueError:
        count = 0
    if count < 1:
        raise ValueError(
            f'{DEVICE_COUNT_ENV_VAR}={raw!r} is not a positive integer.  '
            'Unset it for the automatic device count, or set it to the '
            'number of devices to pin.')
    return count


def calibration_enabled():
    return os.environ.get(CALIBRATION_ENV_VAR, '') not in ('', '0', 'false')


def calibration_start(devices):
    """Reset the peak counters this mode is about to read.

    This mode OWNS ``max_memory_allocated`` while it runs, which is why it is
    behind an environment variable: resetting the counter would otherwise
    clobber a caller's own measurement.
    """
    for device in devices:
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)


def calibration_report(ledger, devices):
    """Modeled against measured, per device, as (device, modeled, measured,
    ratio) rows."""
    rows = []
    for i, device in enumerate(devices):
        device = torch.device(device)
        if device.type != 'cuda':
            continue
        measured = int(torch.cuda.max_memory_allocated(device))
        modeled = ledger.peak_bytes(i)
        ratio = (modeled / measured) if measured else float('inf')
        rows.append((device, modeled, measured, ratio))
    return rows


def format_calibration(rows, band=None):
    """The calibration table.  ``band`` defaults to CALIBRATION_BAND; a
    reconstruction whose projection bodies are torch bodies is judged against
    TORCH_BODY_CALIBRATION_BAND instead."""
    low, high = band or CALIBRATION_BAND
    lines = ['memory ledger calibration (this mode owns '
             'torch.cuda.max_memory_allocated)',
             f'{"device":>10}{"modeled":>14}{"measured":>14}'
             f'{"ratio":>10}{"verdict":>12}']
    for device, modeled, measured, ratio in rows:
        if ratio < low:
            verdict = 'UNDER'
        elif ratio > high:
            verdict = 'over'
        else:
            verdict = 'ok'
        lines.append(f'{str(device):>10}{_gb(modeled):>14}{_gb(measured):>14}'
                     f'{ratio:>10.3f}{verdict:>12}')
    return '\n'.join(lines)
