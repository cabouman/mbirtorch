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

Two consumers share ONE per-view cost model.  The projection drivers use
``Projectors.view_batch_charge`` to choose a view batch; the ledger calls
the same function to price that batch's residency.  The charge excludes the
call-fixed outputs by contract, so the ledger adds those itself, per phase.
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import _sharding

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

# Library workspace that torch allocates through its own caching allocator,
# and that the ledger's array enumeration therefore cannot see.  Measured as
# a FLAT 32 to 33 MiB across gate cells whose peaks span 2.26 GiB to 26.68
# GiB.  The residual does not scale over that twelvefold range, so it is a
# fixed per-process allocation rather than a missing array term.  The size and
# the architecture match the cuBLAS workspace, which is 32 MiB per stream on
# this class of device.  Charged at 64 MiB, which covers the measurement with
# headroom at a cost of 2.8 percent at the smallest gate cell and 0.2 percent
# at the largest.
FIXED_DEVICE_OVERHEAD_BYTES = 64 * 2 ** 20

CALIBRATION_ENV_VAR = 'MBIRTORCH_MEMORY_CALIBRATION'
# The band the modeled peak must land in against the measured peak.  The
# lower bound is the one that matters: a ledger that under-predicts would let
# a doomed run start, which is the failure this module exists to prevent.
CALIBRATION_BAND = (1.00, 1.30)


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
    view_blocks: list                     # (padded_views, real_views) per device
    slice_blocks: list                    # (padded_slices, real_slices) per device
    sino_rows: int                        # the DEVICE-form detector row count
    rows_track_slices: bool
    # ── pixel counts ─────────────────────────────────────────────────────────
    num_pixels_full: int                  # the ROR-masked set
    num_pixels_grid: int                  # the unmasked grid (the hessian's)
    granularities: tuple                  # subset counts the sequence visits
    partition_granularities: tuple        # every subset count built up front
    # ── what this call supplies ──────────────────────────────────────────────
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
    qggmrf_cylinders: int = QGGMRF_CYLINDERS_COMPILED
    # (direction, num_pixels, band_cols) -> (view_batch, bytes_per_view), with
    # direction in {'forward', 'back'}.  Defaults to a no-charge model so a
    # hand-built plan can exercise the state terms alone.
    view_charge: object = None

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
        padded_slices = self.slice_blocks[dev_index][0]
        fixed = self.forward_band if direction == 'forward' else self.back_band
        return min(int(fixed), padded_slices) if fixed else padded_slices


def estimate_peak_device_bytes(plan):
    """The modeled per-device peak for one reconstruction.

    Pure: no device is queried, nothing is allocated, and the result depends
    only on ``plan``.  That is what lets the widening rule price a device
    count the model is not configured for, and what lets the tests run the
    whole model on CPU.

    Returns:
        Ledger: the phases and their per-device bytes.
    """
    n = plan.n_devices
    num_rows_dev, num_channels = plan.sino_rows, int(plan.sinogram_shape[2])
    rows_recon, cols_recon = int(plan.recon_shape[0]), int(plan.recon_shape[1])

    def sino_dev(i):
        return plan.view_blocks[i][0] * num_rows_dev * num_channels * _F32_BYTES

    def recon_dev(i):
        return rows_recon * cols_recon * plan.slice_blocks[i][0] * _F32_BYTES

    def cyl(i, num_pixels):
        return int(num_pixels) * plan.slice_blocks[i][0] * _F32_BYTES

    def is_view_owner(i):
        return plan.view_blocks[i][1] > 0

    def is_slice_owner(i):
        return plan.slice_blocks[i][1] > 0

    def per_dev(fn):
        return [int(fn(i)) for i in range(n)]

    # ── the shared projection terms ──────────────────────────────────────────
    def back_cols(i):
        """The back call's band_cols: its local sinogram's row count."""
        if n == 1:
            return int(plan.sinogram_shape[1])
        return (plan.band_length(i, 'back') if plan.rows_track_slices
                else num_rows_dev)

    def forward_cols(i):
        """The forward call's band_cols: its voxel columns."""
        return (int(plan.recon_shape[2]) if n == 1
                else plan.band_length(i, 'forward'))

    def back_batch(i, num_pixels):
        if not is_view_owner(i):
            return 0
        return plan.batch_bytes('back', num_pixels, back_cols(i))

    def forward_batch(i, num_pixels):
        if not is_view_owner(i):
            return 0
        return plan.batch_bytes('forward', num_pixels, forward_cols(i))

    def band_reduce(i, num_pixels):
        """The back reduce's co-residency on a slice-owner.

        ``sum_band_to_owner`` moves ALL n partials onto the owner before the
        summation loop begins, so the owner holds n arrays of one band plus
        the running total.  At three devices and above the old and the new
        total coexist during a rebind, so the count is n + 2 there.  Because
        one band is the whole shard by default, this term is very nearly
        INDEPENDENT of the device count: it reads 1.5x a full-volume cylinder
        set at both two and four devices.  Adding devices shrinks the
        persistent set and leaves this where it was, which is why the
        slice-band knob is the remedy the error message names for it.
        """
        if n == 1 or not is_slice_owner(i):
            return 0
        copies = n + 1 if n == 2 else n + 2
        return copies * int(num_pixels) * plan.band_length(i, 'back') * _F32_BYTES

    def back_fixed(i, num_pixels):
        """The back's call-fixed outputs.

        A single device holds THREE cylinder arrays, not two.  The driver's
        loop is ``block = back_body(...)`` followed by ``out.add_(block)``,
        and python evaluates the call before it rebinds the name, so the
        PREVIOUS block is still alive while the kernel produces the next one:
        the accumulator, the outgoing block, and the incoming block.  The
        phase probe confirmed the third array at both gate cells; charging
        two put the direct-recon and hessian phases 10 to 13 percent under
        their measured peaks.

        A multi-device slice-owner instead accumulates its per-band parts in
        a list and concatenates them, which is two.
        """
        if not is_slice_owner(i):
            return 0
        return (3 if n == 1 else 2) * cyl(i, num_pixels)

    def forward_fixed(i):
        """The forward's assembled output.  A multi-device owner holds the
        per-band pieces AND their concatenation (row-aligned), or the running
        partial AND the incoming one (two-fan), so it pays twice."""
        if not is_view_owner(i):
            return 0
        return sino_dev(i) if n == 1 else 2 * sino_dev(i)

    def forward_block(i, num_pixels, cols):
        """One assembled view block, which is not part of the batch charge.

        The block's size follows the view batch, and the batch follows the
        pixel count of THIS call, so the subset phases must pass their own
        subset size rather than the full index count.
        """
        if not is_view_owner(i):
            return 0
        view_batch = 1
        if plan.view_charge is not None:
            view_batch = plan.view_charge('forward', num_pixels, cols)[0]
        return view_batch * cols * num_channels * _F32_BYTES

    # ── the persistent set ───────────────────────────────────────────────────
    # One sinogram-shaped weights term, never two: when the caller supplies
    # weights the hessian's weight array is a bare ALIAS of them, and when it
    # does not, the internally built all-ones sinogram is the only one.  It is
    # charged whenever either exists.
    # WHEN the weights array exists differs from WHETHER it exists.  A
    # supplied weights array is placed at the top of vcd_recon, so it is
    # resident from the direct recon onward.  The internally built all-ones
    # array is created inside the hessian block, so on an unweighted run
    # nothing weights-shaped exists before that.  The phase probe's entry
    # column pinned both: the unweighted direct recon enters at one sinogram,
    # the weighted one at two.
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
    # base under EVERY phase, not only the loop.
    constant_terms = [
        ('partitions (lead device)',
         persistent.pop('partitions (lead device)')),
        ('library workspace',
         [FIXED_DEVICE_OVERHEAD_BYTES] * n),
    ]
    constant_base = [sum(vals[i] for _name, vals in constant_terms)
                     for i in range(n)]
    persistent_total = [constant_base[i]
                        + sum(vals[i] for vals in persistent.values())
                        for i in range(n)]

    phases = []

    # ── phase B: the direct reconstruction ───────────────────────────────────
    # Runs only when no initial reconstruction was supplied.  Its full-index
    # back projection is the largest single projection of the run.
    if not plan.init_recon_supplied and not plan.resume:
        p_full = plan.num_pixels_full
        # The back LOOP and the SCATTER are consecutive, not co-live: the
        # driver's accumulator is freed into the scatter's input.  Charging
        # both together over-counted the direct recon by a recon-shaped array.
        loop_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('filtered sinogram', per_dev(sino_dev)),
            ('back output', per_dev(lambda i: back_fixed(i, p_full))),
            ('band reduce', per_dev(lambda i: band_reduce(i, p_full))),
            ('back batch', per_dev(lambda i: back_batch(i, p_full))),
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
        phases.append(_phase('direct recon (back loop)', loop_terms, n,
                             base=constant_base, base_terms=constant_terms))
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
            ('forward output', per_dev(forward_fixed)),
            ('forward block',
             per_dev(lambda i: forward_block(i, p_full, forward_cols(i)))),
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
        terms = [
            ('error sinogram', per_dev(sino_dev)),
            ('hessian weights', per_dev(weights_term)),
            ('init recon', per_dev(recon_dev)),
            ('back output', per_dev(lambda i: back_fixed(i, p_hess))),
            ('band reduce', per_dev(lambda i: band_reduce(i, p_hess))),
            ('back batch', per_dev(lambda i: back_batch(i, p_hess))),
        ]
        phases.append(_phase('hessian diagonal', terms, n,
                             base=constant_base,
                             base_terms=constant_terms))
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
            'back projection': [
                ('prior gradient and hessian',
                 per_dev(lambda i: 2 * cyl(i, p_sub))),
                ('weighted error sinogram', per_dev(
                    lambda i: sino_dev(i) if plan.weights_supplied else 0)),
                ('back output', per_dev(lambda i: back_fixed(i, p_sub))),
                ('band reduce', per_dev(lambda i: band_reduce(i, p_sub))),
                ('back batch', per_dev(lambda i: back_batch(i, p_sub))),
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
                ('forward block', per_dev(
                    lambda i: forward_block(i, p_sub, forward_cols(i)))),
                ('forward batch', per_dev(lambda i: forward_batch(i, p_sub))),
            ],
            'state application': [
                ('direction and scaled direction',
                 per_dev(lambda i: APPLY_CYLINDERS * cyl(i, p_sub))),
                ('delta sinogram', per_dev(sino_dev)),
            ],
        }
        for name, terms in sub_phases.items():
            all_terms = terms + [('subset indices', index_bytes)]
            phases.append(_phase(
                f'subset {name} (granularity {granularity})',
                all_terms, n, base=persistent_total,
                base_terms=constant_terms + list(persistent.items())))

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


def plan_from_model(model, devices, partition_sequence=None, weights=None,
                    init_recon=None, fm_hessian=None, prox_input=None,
                    init_error_sinogram=None):
    """Build a :class:`LedgerPlan` for ``model`` over a CANDIDATE device list.

    The device list is an argument rather than a reading of the model's own
    placement, because the widening rule prices counts the model is not
    configured for.  The placements are rebuilt here from the current params,
    so a geometry change cannot leave a stale real size behind.
    """
    sinogram_shape = tuple(int(s) for s in model.get_params('sinogram_shape'))
    recon_shape = tuple(int(s) for s in model.get_params('recon_shape'))
    devices = [torch.device(d) for d in devices]

    sino_placement = _sharding.Placement(devices, axis=0,
                                         real_size=sinogram_shape[0])
    recon_placement = _sharding.Placement(devices, axis=-1,
                                          real_size=recon_shape[2])
    view_blocks = [(end - start, n_valid)
                   for _d, (start, end), n_valid
                   in sino_placement.padded_shard_ranges()]
    slice_blocks = [(end - start, n_valid)
                    for _d, (start, end), n_valid
                    in recon_placement.padded_shard_ranges()]
    # A row-aligned geometry presents the same padded length on the detector
    # row axis as on the sharded slice axis.
    rows_track_slices = bool(getattr(model, 'rows_track_slices', False))
    sino_rows = (recon_placement.padded_size
                 if rows_track_slices and recon_placement.is_padded
                 else sinogram_shape[1])

    granularity = list(model.get_params('granularity'))
    if partition_sequence is None:
        partition_sequence = list(model.get_params('partition_sequence'))
    visited = sorted({granularity[int(k)] for k in partition_sequence
                      if int(k) < len(granularity)})

    num_pixels_full = int(model.full_index_count())
    num_pixels_grid = recon_shape[0] * recon_shape[1]

    charge = _model_view_charge(model, len(devices))

    return LedgerPlan(
        sinogram_shape=sinogram_shape,
        recon_shape=recon_shape,
        devices=devices,
        view_blocks=view_blocks,
        slice_blocks=slice_blocks,
        sino_rows=int(sino_rows),
        rows_track_slices=rows_track_slices,
        num_pixels_full=num_pixels_full,
        num_pixels_grid=num_pixels_grid,
        granularities=tuple(visited) or (granularity[0],),
        partition_granularities=tuple(granularity),
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
        qggmrf_cylinders=qggmrf_cylinder_count(model),
        view_charge=charge,
    )


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


def format_shortfall(ledger, rows, num_devices_tried):
    """The one readable error: what did not fit, which phase dominates it,
    and which knob moves that phase."""
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
    lines += [
        '', f'Device counts tried, largest first, down to 1: '
            f'{num_devices_tried}.  None fit.',
        '', 'Remedies, most effective first:',
        '  model.back_project_slice_band = <slices>   '
        '# the band reduce barely shrinks with more',
        '                                             '
        '# devices; this is its only lever',
        '  model.view_batch_size = <views>            '
        '# caps the projector batch transient',
        '  model.set_params(granularity=[...])        '
        '# a finer coarsest granularity shrinks the',
        '                                             '
        '# per-subset transients',
        '  CUDA_VISIBLE_DEVICES=...                   '
        '# exclude a device another process is using',
        '',
        'To run anyway: model.skip_memory_preflight = True',
    ]
    return '\n'.join(lines)


# ── calibration ──────────────────────────────────────────────────────────────
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


def format_calibration(rows):
    low, high = CALIBRATION_BAND
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
