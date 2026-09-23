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
from ._utils import padded_kernel_width

_F32_BYTES = 4
_INT64_BYTES = 8

# The qGGMRF prior kernel holds this many cylinder arrays at once.
# qggmrf_cylinder_count chooses between the two counts.
QGGMRF_CYLINDERS_COMPILED = 9
QGGMRF_CYLINDERS_EAGER = 16
# The proximal-map prior is pointwise.  It holds the recon rows, the prox rows,
# the difference, and the scaled result.
PROX_CYLINDERS = 4
# direction_worker and apply_worker hold this many cylinder arrays at once.
DIRECTION_CYLINDERS = 7
APPLY_CYLINDERS = 2
# A multi-device forward projection holds this many transferred cylinder batches
# at once, because the driver transfers one batch ahead of the projection.
CYLINDER_TRANSFER_RESIDENTS = 3

# One QGGMRFDenoiser subset holds this many cylinder arrays at once, first while
# it forms the update direction and then while it applies the state.
DENOISE_DIRECTION_CYLINDERS = 8
DENOISE_APPLY_CYLINDERS = 3
# One device holds this many qGGMRF boundary columns while the sharded sweep
# exchanges halos.  The two end devices hold fewer and are charged four as well.
DENOISE_HALO_COLUMNS = 4
# The ell-1 and weighted dot product reductions run a chunk at a time, and these
# two values set the chunk for all of them.
ELL1_CHUNK_BYTES = 16 * 2 ** 20
ELL1_MAX_CHUNKS = 1024


def _chunk_count(n_bytes):
    """Return the number of chunks an array of ``n_bytes`` is reduced in."""
    return min(ELL1_MAX_CHUNKS, max(1, round(int(n_bytes) / ELL1_CHUNK_BYTES)))


def image_ell1(flat_image):
    """The ell-1 norm of a recon-shaped array, without a recon-shaped
    temporary.

    ``torch.sum(torch.abs(x))`` allocates a whole array of absolute values
    before it reduces.  Reducing a chunk at a time bounds the temporary to one
    chunk, so it stops scaling with the array.

    Considered but rejected: ``torch.linalg.vector_norm(x, ord=1)``, which
    allocates nothing at all.  On CPU it accumulates float32 sequentially
    where ``torch.sum`` sums pairwise, so its error grows with the element
    count.  On MPS it is accurate but runs the reduction about 34 times
    slower than ``sum(abs)``.

    Chunking keeps torch's pairwise summation inside each chunk and adds only
    the chunk totals, so it tracks the unchunked value to about 1e-7 at every
    size measured.

    An array below one chunk is reduced whole.
    """
    n_chunks = _chunk_count(flat_image.numel() * flat_image.element_size())
    if n_chunks == 1:
        return torch.sum(torch.abs(flat_image))
    return torch.stack([torch.sum(torch.abs(chunk)) for chunk
                        in torch.chunk(flat_image, n_chunks, dim=0)]).sum()


def stack_ell1(flat_stack):
    """The ell-1 norm of each volume in a stack of recon-shaped arrays,
    without a stack-shaped temporary.

    ``flat_stack`` has shape ``(num_volumes, num_pixels, num_slices)`` and
    the result has shape ``(num_volumes,)``.  This is :func:`image_ell1` with
    a leading volume axis.  The chunk count follows the same rule, applied to
    the whole stack, so the temporary is bounded by the same number of bytes,
    and the chunks are cut along the pixel axis, so every volume is reduced
    over the same pixel ranges.  A stack below one chunk is reduced whole.
    """
    n_bytes = flat_stack.numel() * flat_stack.element_size()
    n_chunks = min(ELL1_MAX_CHUNKS, max(1, round(n_bytes / ELL1_CHUNK_BYTES)))
    if n_chunks == 1:
        return torch.sum(torch.abs(flat_stack), dim=(1, 2))
    return torch.stack([torch.sum(torch.abs(chunk), dim=(1, 2)) for chunk
                        in torch.chunk(flat_stack, n_chunks, dim=1)]).sum(dim=0)


def _block_dot(a_block, b_block, weights):
    """Return the weighted dot product of one block."""
    if weights is None:
        return torch.sum(a_block * b_block)
    return torch.sum(a_block * b_block * weights)


def _paired_blocks(operand, reference, n_chunks, n_blocks):
    """Return the blocks of a second operand that pair with the reference's.

    An operand that spans the reference's view axis is split the same way.
    Anything else is handed to every block whole, so that it broadcasts.
    """
    if (torch.is_tensor(operand) and operand.ndim == reference.ndim
            and operand.shape[0] == reference.shape[0]):
        return torch.chunk(operand, n_chunks, dim=0)
    return [operand] * n_blocks


def weighted_dot(a, b, weights=None):
    """The weighted dot product of two sinograms, without a sinogram-shaped
    temporary.

    ``torch.sum(a * b * w)`` allocates a whole array of products and then a
    whole array of weighted products before it reduces.  Reducing a block of
    views at a time bounds both to one block, so they stop scaling with the
    sinogram.

    ``b`` is a sinogram of ``a``'s shape.  ``weights`` is None (the plain dot
    product), a scalar, or an array of that shape.  Either one is split into
    the same blocks as ``a`` when it spans the view axis, so each block meets
    the values that belong to it.

    Chunking keeps torch's pairwise summation inside each block and adds only
    the block totals, so it tracks the unchunked value the way image_ell1
    does.  A sinogram below one chunk is reduced whole.
    """
    n_chunks = _chunk_count(a.numel() * a.element_size())
    if n_chunks == 1:
        return _block_dot(a, b, weights)
    blocks = torch.chunk(a, n_chunks, dim=0)
    b_blocks = _paired_blocks(b, a, n_chunks, len(blocks))
    weight_blocks = _paired_blocks(weights, a, n_chunks, len(blocks))
    totals = [_block_dot(block, b_block, block_weights)
              for block, b_block, block_weights
              in zip(blocks, b_blocks, weight_blocks)]
    return torch.stack(totals).sum()


def weighted_square_sum(error_sinogram, weights=None):
    """The weighted sum of squares of a sinogram, without a sinogram-shaped
    temporary.

    The sum of squares is the dot product of a sinogram with itself, so this
    is weighted_dot on one array; routing it through that one routine is what
    keeps the two reductions on the same chunk rule and the same accumulation
    pattern.  See weighted_dot for what chunking costs in accuracy and for
    how weights are split.
    """
    return weighted_dot(error_sinogram, error_sinogram, weights)


def reduction_chunk_bytes(array_bytes):
    """What ONE chunk of a chunked reduction holds, for an array of
    ``array_bytes``.

    Shared by the ell-1 and the weighted dot products, which chunk by the
    same rule.  A phase that holds more than one array per chunk -- a weighted
    dot product holds the products and their weighted form -- multiplies this.

    An array small enough to want a single chunk is reduced whole, so the
    temporary is the array itself; that is the unchunked case and it is only
    reached below this module's chunk size, where one extra copy is small in
    absolute terms.
    """
    array_bytes = int(array_bytes)
    n_chunks = _chunk_count(array_bytes)
    if n_chunks == 1:
        return array_bytes, 1
    return math.ceil(array_bytes / n_chunks), n_chunks

# Torch allocates a library workspace the ledger cannot see by enumerating
# arrays.  It measures a flat 32 to 33 MiB, which matches the 32 MiB per-stream
# cuBLAS workspace, and it is charged at 64 MiB to leave headroom.
FIXED_DEVICE_OVERHEAD_BYTES = 64 * 2 ** 20

CALIBRATION_ENV_VAR = 'MBIRTORCH_MEMORY_CALIBRATION'
# This variable pins the automatic device count for the whole process.  The
# pinned count is not searched and never reduced, and the preflight still applies.
DEVICE_COUNT_ENV_VAR = 'MBIRTORCH_NUM_DEVICES'
# One view batch of a torch body holds this many per-view slabs.  A torch body is
# a projection body written as general torch code, and it declares no per-view
# cost of its own.  One count covers both geometries and both directions,
# because nothing in the plan distinguishes them.
TORCH_BODY_VIEW_SLABS = 14

# The modeled peak must land in this band against the measured peak.  The lower
# bound matters, because a model that predicts too little lets a run start that
# cannot finish.
CALIBRATION_BAND = (1.00, 1.30)
# Torch bodies are judged against this wider band, because one slab count covers
# two geometries that hold different numbers of slabs.
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
    # This is the ROR-masked pixel count the subset phases were built from.
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
    # The workload names the call the plan prices.  'recon' is a full
    # reconstruction, 'direct' is a direct reconstruction alone, and 'denoise'
    # is one QGGMRFDenoiser sweep.
    workload: str = 'recon'
    weights_supplied: bool = False
    fm_hessian_supplied: bool = False
    init_recon_supplied: bool = False
    resume: bool = False
    prox: bool = False
    positivity: bool = False
    helical: bool = False
    # True when the hessian back-projects at the ROR-masked index set rather than
    # the full grid.  _vcd_recon does so, and a direct public call does not.
    hessian_masked: bool = False
    # ── knobs and model choices ──────────────────────────────────────────────
    back_band: int = None
    # The number of pixels one transferred cylinder batch covers in the forward
    # projection.  A denoise plan has no forward projection and leaves it None.
    pixel_batch: int = None
    qggmrf_cylinders: int = QGGMRF_CYLINDERS_COMPILED
    # A callable taking (direction, num_pixels, band_cols) and returning
    # (view_batch, bytes_per_view), with direction 'forward' or 'back'.
    # None charges nothing.
    view_charge: object = None
    # The directions, out of 'forward' and 'back', that bind a torch body.  The
    # ledger prices those views itself.  See TORCH_BODY_VIEW_SLABS.
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

    def band_length(self, dev_index):
        """The slice-band length one owner streams in the BACK projection,
        matching ``TomographyModel._slice_band_length``: the whole shard by
        default.  The forward transfers whole cylinders and walks no bands."""
        local_slices = self.slice_blocks[dev_index]
        fixed = self.back_band
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

    def sino_reduction_block(i):
        """Return the bytes in one block of a chunked sinogram reduction.

        The reduction splits the view axis, so a block is a whole number of
        views.  A sinogram with few views and large detector planes therefore
        holds a block larger than the byte rule alone would give.
        """
        chunk, n_chunks = reduction_chunk_bytes(sino_dev(i))
        views = int(plan.view_blocks[i])
        if n_chunks == 1 or views <= 0:
            return chunk
        return (math.ceil(views / n_chunks) * num_rows_dev * num_channels
                * _F32_BYTES)

    def back_block(i, num_pixels):
        """Return the bytes in one live (pixels, band) back partial.

        A hand-written kernel wrapper rounds the band up to a multiple of 16
        before it allocates, then returns the real-band slice of that wider
        array.  The block therefore occupies the padded band.
        """
        slices = max(int(plan.slice_blocks[i]),
                     padded_kernel_width(plan.band_length(i)))
        return int(num_pixels) * slices * _F32_BYTES

    def is_view_owner(i):
        return plan.view_blocks[i] > 0

    def is_slice_owner(i):
        return plan.slice_blocks[i] > 0

    def per_dev(fn):
        return [int(fn(i)) for i in range(n)]

    # ── the shared projection terms ──────────────────────────────────────────
    def back_cols(i):
        """Return the back call's band_cols, its local sinogram's row count."""
        if n == 1:
            return int(plan.sinogram_shape[1])
        return (plan.band_length(i) if plan.rows_track_slices
                else num_rows_dev)

    def whole_slice_extent():
        """Return the slice extent one forward call is handed.  It is the
        whole slice axis, because a transferred cylinder spans every
        slice-owner at once."""
        return sum(int(block) for block in plan.slice_blocks)

    def forward_call_pixels(num_pixels):
        """Return the pixel count one forward call is handed.  It is one
        cylinder batch, capped by the pass.  A plan with no batch prices the
        whole pass."""
        if plan.pixel_batch:
            return min(int(num_pixels), int(plan.pixel_batch))
        return int(num_pixels)

    def forward_cols(i):
        """Return the forward call's band_cols, which is its slice extent.
        Every view-owner is handed the whole slice axis."""
        if n == 1:
            return int(plan.recon_shape[2])
        return whole_slice_extent()

    def band_slices(i, direction):
        """Return the slice extent one projection call is handed.  A sharded
        back call is handed this owner's slice band.  Every other case is
        handed the whole slice axis."""
        if n == 1:
            return int(plan.recon_shape[2])
        if direction == 'forward':
            return whole_slice_extent()
        return plan.band_length(i)

    def torch_body_batch(i, direction, num_pixels):
        """Return the bytes one view batch of a torch body holds.

        The body sweeps the detector rows and the slice band it was handed.
        Every array in its interpolation loop spans the view batch, the
        pixels, and whichever of those two axes is wider.  The body holds
        TORCH_BODY_VIEW_SLABS of them at once.
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
        call_pixels = forward_call_pixels(num_pixels)
        if 'forward' in plan.torch_body_directions:
            return torch_body_batch(i, 'forward', call_pixels)
        return plan.batch_bytes('forward', call_pixels, forward_cols(i))

    def band_reduce(i, num_pixels):
        """Return the bytes the back reduce holds on one slice-owner.

        ``sum_band_to_owner`` adds each arriving partial one row slab at a
        time, and frees the slab before the next one arrives.  At its widest
        the owner holds the bands it has already reduced this pass, the
        running total for the band it is on, the partial it produced itself,
        and one bounded slab per arriving partial.  Only the partial this
        device produced takes the length rounded up to a multiple of 16,
        because only it comes straight from a kernel wrapper.
        """
        if n == 1 or not is_slice_owner(i):
            return 0
        band = plan.band_length(i)
        shard = plan.slice_blocks[i]
        own_partial = padded_kernel_width(band)
        row_bytes = int(band) * _F32_BYTES
        slab_rows = _sharding.reduce_slab_rows(int(num_pixels), row_bytes)
        return (int(num_pixels) * (int(shard) + own_partial) * _F32_BYTES
                + (n - 1) * slab_rows * row_bytes)

    def back_view_batches(i, num_pixels):
        """Return the number of batches one worker's view loop runs.  Return
        None when this plan prices no batch."""
        local_views = plan.view_blocks[i]
        if local_views <= 0 or plan.view_charge is None:
            return None
        view_batch = int(plan.view_charge(
            'back', int(num_pixels), back_cols(i))[0])
        return max(1, -(-int(local_views) // max(1, view_batch)))

    def back_fixed(i, num_pixels):
        """Return the bytes the back view loop holds in live cylinder shards.

        ``Projectors.sparse_back_project_view_range`` releases each block
        after it adds it, so a multi-device loop holds the accumulator and
        the incoming block.  A single device is charged three blocks, because
        that charge sits within about one percent of the measured peak and
        the ledger must not charge less than a run holds.  Every view owner
        is charged, because the workers run wherever there are views.
        """
        if n == 1:
            return (3 * back_block(i, num_pixels) if is_slice_owner(i) else 0)
        if not is_view_owner(i):
            return 0
        batches = back_view_batches(i, num_pixels)
        live = 2 if batches is None else min(2, batches)
        return live * back_block(i, num_pixels)

    def back_own_band(i, num_pixels):
        """Return the bytes of the band this device has already finished.

        Each slice-owner keeps its reduced band in ``recon_tensors`` for the
        rest of the loop.  From its own pass onward it therefore carries one
        extra cylinder shard through every later projection.
        """
        if n == 1 or not is_slice_owner(i):
            return 0
        return cyl(i, num_pixels)

    # The forward terms charge only arrays the code allocates, with no safety
    # margin.  Every modeled peak must sit at or above the measured one, and the
    # thinnest measured margin was a fraction of a percent, so trimming a term
    # here can make a run start that cannot finish.
    def forward_fixed(i):
        """Return the bytes of the forward projection's assembled output.

        A multi-device owner holds one such block, because its batches add
        into that block inside the projector's view loop.  Two are charged
        anyway, which is a deliberate over-charge calibrated against measured
        peaks."""
        if not is_view_owner(i):
            return 0
        return sino_dev(i) if n == 1 else 2 * sino_dev(i)

    def forward_transferred_cylinders(i, num_pixels):
        """Return the bytes of the cylinder batches a multi-device forward
        holds.

        ``_sharding.transfer_cylinder_batch`` moves one batch of pixels from
        every slice-owner and concatenates them.  A view-owner therefore holds
        that batch by the whole device-form slice axis, which does not grow
        with the shard.  See CYLINDER_TRANSFER_RESIDENTS for how many are live
        at once.
        """
        if n == 1 or not is_view_owner(i) or not plan.pixel_batch:
            return 0
        return (CYLINDER_TRANSFER_RESIDENTS * forward_call_pixels(num_pixels)
                * whole_slice_extent() * _F32_BYTES)

    def forward_view_batches(i, num_pixels):
        """Return the number of batches one owner's forward view loop runs.
        Return None when this plan prices no batch."""
        local_views = plan.view_blocks[i]
        if local_views <= 0 or plan.view_charge is None:
            return None
        view_batch = int(plan.view_charge(
            'forward', forward_call_pixels(num_pixels), forward_cols(i))[0])
        return max(1, -(-int(local_views) // max(1, view_batch)))

    def forward_block_rows(i):
        """Return the detector-row extent of one forward view block.

        Both kernel bodies round that extent up to a multiple of 16 before
        they allocate, then return the real-width slice of the wider array.
        A kernel body's block is therefore charged at the padded extent.  A
        torch body rounds nothing up and keeps the real extent.
        """
        rows = forward_cols(i) if plan.rows_track_slices else plan.sino_rows
        if 'forward' in plan.torch_body_directions:
            return int(rows)
        return padded_kernel_width(rows)

    def forward_block(i, num_pixels):
        """Return the bytes of the view block the loop holds besides the one
        the batch prices.

        ``Projectors.sparse_forward_project_view_range`` does not release its
        block, so the loop holds the outgoing block and the incoming one.  A
        kernel body's declared cost already covers one of the two, because its
        output plane scales with the view batch.  A torch body declares
        nothing and the ledger charges it the body's internal slabs, which do
        not include the output plane, so both blocks are charged there.

        The batch follows the pixel count of this call, so a subset phase must
        pass its own subset size rather than the full index count.
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

    # Only the 'direct' plan charges the filter terms.  Inside a full
    # reconstruction the filter runs between phases that hold more than it does.
    def filter_row_weights(i):
        """Return the bytes of the FDK cosine pre-weight, which is one
        detector plane per device.

        The FBP filters pass no such weight, so this over-charges them by one
        detector plane.
        """
        return int(plan.sino_rows) * num_channels * _F32_BYTES

    def filter_row_batch(i):
        """Return the bytes one batch of the filter's row loop holds.

        ``tomography_utils.apply_row_filter`` walks the shard
        ROW_FILTER_BATCH detector rows at a time and convolves in frequency
        space.  At its widest one batch holds the pre-weighted window, the
        real FFT of that window, its product with the filter's transform, and
        the output of the inverse transform.  The batch is a fixed row count,
        so this term does not fall with the device count.
        """
        rows_in_shard = plan.view_blocks[i] * int(plan.sino_rows)
        batch = min(tomography_utils.ROW_FILTER_BATCH, rows_in_shard)
        # This is the linear convolution length apply_row_filter transforms at,
        # which is the channel count plus the filter tap count minus one.
        padded = 3 * num_channels - 2
        per_row = (num_channels * _F32_BYTES
                   + 2 * (padded // 2 + 1) * (2 * _F32_BYTES)
                   + padded * _F32_BYTES)
        return batch * per_row

    # There is one sinogram-shaped weights array and never two.  A supplied array
    # is resident from the direct recon onward, while an unweighted run builds an
    # all-ones array inside the hessian block and holds nothing before that.
    weights_resident = plan.weights_supplied or not plan.fm_hessian_supplied

    def weights_term(i):
        """Return the weights bytes from the hessian phase onward."""
        return sino_dev(i) if weights_resident else 0

    def supplied_weights_term(i):
        """Return the weights bytes in the phases before the hessian builds
        an array of its own."""
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

    # The partitions and the index cache live on the lead device for the whole
    # reconstruction, so they are a base under every phase.
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
        """Return one sharded back projection as its two consecutive steps.

        The workers project and the reduce gathers, and the two never run at
        the same time.  Both sub-phases are emitted, and the per-device
        maximum over phases picks between them.  The sub-phase names keep the
        parent name as a prefix, so a consumer matching on the parent still
        finds them.  A single device runs no reduce, so the phase is emitted
        whole under the parent name.
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

    # A direct reconstruction is the filter and one back projection.  It builds
    # no prior, hessian diagonal, partition sequence or reconstruction loop, so
    # the only term under its phases is the library workspace.
    if plan.workload == 'direct':
        p_full = plan.num_pixels_full
        base_terms = [workspace_term]
        base = list(workspace_term[1])
        # _shard_sinogram places the sinogram at entry, and the filter writes a
        # second array of the same shape that the back projection reads.
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

    # The denoise plan is one QGGMRFDenoiser sweep.  Its forward model is the
    # identity, so nothing here charges a view batch, a projection body, a
    # hessian diagonal or a weights array.  Every term is image shaped and
    # follows slice_blocks.
    if plan.workload == 'denoise':
        base_terms = [workspace_term]
        base = list(workspace_term[1])

        def halo_columns(i):
            """Return the bytes of the qGGMRF boundary columns one device
            holds across a pass.

            ``_sharding.exchange_qggmrf_halos`` gives each shard the image
            slice just beyond each of its boundaries.  A single device runs
            the compiled sweep and exchanges nothing, so the term is zero
            there.
            """
            if n == 1:
                return 0
            return (DENOISE_HALO_COLUMNS * int(plan.num_pixels_grid)
                    * _F32_BYTES)

        def partition_indices(i):
            """Return the bytes of the subset partition, which every device
            holds whole.

            The sharded sweep copies the whole partition onto each device
            rather than splitting it, because a subset's indices address the
            in-slice pixel grid and every shard updates those same pixels in
            its own slices.
            """
            return sum(
                int(g) * math.ceil(plan.num_pixels_full / max(1, int(g)))
                * _INT64_BYTES for g in plan.partition_granularities)

        # These arrays live from the moment the sweep's state exists until it
        # returns.  A caller-supplied initial image is a fourth array, and by
        # default that argument aliases the input image and costs nothing.
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
            # The prior and the update direction run one after the other, so the
            # per-device maximum over phases picks between them.
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
        # image_ell1 reduces the image a chunk at a time, so the absolute values
        # it forms are one chunk rather than a whole image.
        def ell1_chunk(i):
            return reduction_chunk_bytes(recon_dev(i))[0]

        phases.append(_phase(
            'denoise per-pass statistics',
            residents + [('ell-1 chunk', per_dev(ell1_chunk))], n,
            base=base, base_terms=base_terms))
        return Ledger(devices=list(plan.devices), phases=phases,
                      num_pixels_full=int(plan.num_pixels_full))

    # Phase B is the direct reconstruction.  It runs only when no initial
    # reconstruction was supplied, and its full-index back projection is the
    # largest single projection of the run.
    if not plan.init_recon_supplied and not plan.resume:
        p_full = plan.num_pixels_full
        # The back loop and the scatter run one after the other, and the driver's
        # accumulator is freed into the scatter's input.
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
        # Both sub-peaks are emitted, and the per-device maximum over phases picks
        # between them.  Picking one sub-phase by its cross-device total would
        # charge too little on a device where the other sub-phase is larger.
        phases.extend(back_phases('direct recon (back loop)', loop_residents,
                                  p_full, constant_base, constant_terms))
        phases.append(_phase('direct recon (scatter)', scatter_terms, n,
                             base=constant_base, base_terms=constant_terms))

    # Phase C is the initial error state.  It has two sub-peaks that hold
    # different arrays: the forward projection, then the error sinogram.
    if not plan.resume:
        p_full = plan.num_pixels_full
        forward_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('init recon', per_dev(recon_dev)),
            ('voxel gather', per_dev(lambda i: cyl(i, p_full))),
            ('transferred cylinders', per_dev(
                lambda i: forward_transferred_cylinders(i, p_full))),
            ('forward output', per_dev(forward_fixed)),
            ('forward block', per_dev(lambda i: forward_block(i, p_full))),
            ('forward batch', per_dev(lambda i: forward_batch(i, p_full))),
        ]
        phases.append(_phase('initial forward projection', forward_terms,
                             n, base=constant_base,
                             base_terms=constant_terms))
        # The error sinogram is formed in the projection's own buffer, so the two
        # are one array.  The dot products that set the scale are reduced a block
        # of views at a time, and a weighted block holds the products and their
        # weighted form, so two blocks are charged.
        dot_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('forward projection', per_dev(sino_dev)),
            ('dot product blocks', per_dev(
                lambda i: 2 * sino_reduction_block(i))),
            ('init recon', per_dev(recon_dev)),
        ]
        error_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            # The projection is scaled by -alpha in place and the sinogram is added
            # into it, so one sinogram-shaped array is charged for the pair.
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

    # Phase D is the hessian diagonal, the one phase charged at the unmasked grid
    # count rather than the ROR-masked set.
    if not plan.fm_hessian_supplied:
        # The masked path back-projects the ROR set and scatters it into a
        # zero-filled volume.  The dense path back-projects the whole grid.
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
            # The scatter holds the masked cylinders and the zero-filled volume at
            # once, so it is a separate sub-peak from the back loop.
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

    # The per-iteration statistics.  weighted_square_sum reduces a block of views
    # at a time, so the squared error holds two blocks.  The recon ell-1 runs
    # after those blocks are dead and is a separate sub-phase.
    stats_sub_phases = (
        ('squared error',
         ('squared-error products',
          per_dev(lambda i: 2 * sino_reduction_block(i)))),
        ('recon ell-1',
         ('recon ell-1 chunk',
          per_dev(lambda i: reduction_chunk_bytes(recon_dev(i))[0]))),
    )
    stats_base_terms = constant_terms + list(persistent.items())
    for name, term in stats_sub_phases:
        phases.append(_phase(f'per-iteration statistics ({name})', [term], n,
                             base=persistent_total,
                             base_terms=stats_base_terms))

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
            # The back projection carries only its resident arrays here.
            # back_phases below builds its two sub-steps and their terms.
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
                # The second sinogram-shaped block of a multi-device forward
                # projection, the over-charge forward_fixed describes.  A
                # positivity pass holds two blocks, because it projects a second
                # delta while the first is live.
                ('forward assembly', per_dev(
                    lambda i: sino_dev(i) if n > 1 and is_view_owner(i) else 0)),
                ('transferred cylinders', per_dev(
                    lambda i: forward_transferred_cylinders(i, p_sub))),
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
        # A denoise sweep builds and visits the one partition the first entry of
        # the sequence names.  A reconstruction walks the whole sequence.
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
        back_band=getattr(model, 'back_project_slice_band', None),
        # The model's own resolver supplies the pixel batch, so the charge cannot
        # drift from the driver.  A denoiser has no forward projection.
        pixel_batch=(None if denoising
                            else model._forward_pixel_batch()),
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
    """Return a closure over the bodies this model binds.  It takes
    (direction, num_pixels, band_cols) and returns (batch, bytes_per_view)."""
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


# QGGMRFDenoiser.denoise_stack sweeps several volumes of the same shape at once
# on one device, so every image-shaped term is held once per volume.  The terms
# named here are held once for the whole batch instead.
DENOISE_BATCH_FIXED_TERMS = frozenset({'library workspace', 'subset indices',
                                       'qggmrf halos'})


def _denoise_batch_phase_split(phase):
    """Return one denoise phase's bytes on its single device, split into the
    fixed bytes and the bytes per volume."""
    fixed = sum(vals[0] for name, vals in phase.terms
                if name in DENOISE_BATCH_FIXED_TERMS)
    per_volume = sum(vals[0] for name, vals in phase.terms
                     if name not in DENOISE_BATCH_FIXED_TERMS)
    return int(fixed), int(per_volume)


def denoise_batch_peak_bytes(ledger, batch_size):
    """The modeled peak of one batched denoise sweep holding ``batch_size``
    volumes, from the one-volume, one-device denoise ledger.

    Each phase is repriced as its fixed terms plus ``batch_size`` times its
    per-volume terms, and the peak is the largest phase.  The ell-1 chunk is
    scaled with the rest although the reduction bounds it at any size that
    chunks, so this charges more than the sweep holds there, never less.
    """
    peaks = []
    for phase in ledger.phases:
        fixed, per_volume = _denoise_batch_phase_split(phase)
        peaks.append(fixed + int(batch_size) * per_volume)
    return max(peaks)


def largest_denoise_batch(ledger, budget, margin=0.15):
    """The largest number of volumes whose batched sweep fits ``budget``
    bytes, judged as :func:`layout_fits` judges a layout: the modeled peak
    times ``1 + margin`` must not exceed the budget.

    Every phase grows linearly with the batch, so the answer is the smallest
    of the per-phase limits.  Returns 0 when one volume does not fit.
    """
    allowed = float(budget) / (1.0 + margin)
    limit = None
    for phase in ledger.phases:
        fixed, per_volume = _denoise_batch_phase_split(phase)
        if per_volume <= 0:
            room = 0 if fixed > allowed else None
        else:
            room = math.floor((allowed - fixed) / per_volume)
        if room is not None:
            limit = room if limit is None else min(limit, room)
    return max(0, limit if limit is not None else 0)


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
