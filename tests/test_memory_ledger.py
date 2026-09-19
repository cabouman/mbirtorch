"""The per-device memory ledger and the preflight verdict.

Every test here runs on CPU with synthetic device budgets.  That is the
point of the ledger being a pure function of shapes, a placement, a call
plan, and the per-view cost models: the arithmetic that decides whether a
reconstruction can run must be checkable without a GPU.
"""

import math

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _memory_ledger, _sharding
from mbirtorch._memory_ledger import (Ledger, LedgerPlan, PhaseCharge,
                                      estimate_peak_device_bytes)
from mbirtorch._utils import padded_kernel_width

GB = 2 ** 30


def make_plan(n_devices=1, num_views=64, num_rows=32, num_channels=32,
              recon=(32, 32, 32), num_pixels_full=800, granularities=(4,),
              rows_track_slices=False, **kwargs):
    """A hand-built plan: no model, no device, no CUDA."""
    devices = ['cpu'] * n_devices
    sino_placement = _sharding.Placement(devices, axis=0, axis_len=num_views)
    recon_placement = _sharding.Placement(devices, axis=-1, axis_len=recon[2])
    return LedgerPlan(
        sinogram_shape=(num_views, num_rows, num_channels),
        recon_shape=recon,
        devices=devices,
        view_blocks=[e - s for _d, (s, e) in sino_placement.shard_ranges()],
        slice_blocks=[e - s for _d, (s, e) in recon_placement.shard_ranges()],
        sino_rows=num_rows,
        rows_track_slices=rows_track_slices,
        num_pixels_full=num_pixels_full,
        num_pixels_grid=recon[0] * recon[1],
        granularities=tuple(granularities),
        partition_granularities=tuple(granularities),
        **kwargs)


# ── the state terms ──────────────────────────────────────────────────────────
def test_the_charges_scale_with_shape_cylinders_granularity_and_devices():
    """The four proportionality rules the ledger's arithmetic rests on."""
    # Doubling the slice count doubles every recon-shaped term and leaves the
    # sinogram-shaped ones alone, so the peak grows by less than 2x.
    small = estimate_peak_device_bytes(make_plan(recon=(32, 32, 32)))
    large = estimate_peak_device_bytes(make_plan(recon=(32, 32, 64)))
    assert large.peak_bytes(0) > small.peak_bytes(0)
    assert large.peak_bytes(0) < 2 * small.peak_bytes(0)

    # The prior phase is proportional to the cylinder count it is priced at.
    nine = estimate_peak_device_bytes(make_plan(qggmrf_cylinders=9))
    sixteen = estimate_peak_device_bytes(make_plan(qggmrf_cylinders=16))
    prior_9 = dict(_named(nine, 'prior').terms)['prior cylinders'][0]
    prior_16 = dict(_named(sixteen, 'prior').terms)['prior cylinders'][0]
    assert prior_16 == pytest.approx(prior_9 * 16 / 9)

    # P_g = ceil(P_full / g), so the coarsest granularity holds the most.
    ledger = estimate_peak_device_bytes(make_plan(granularities=(4, 128)))
    coarse = _named(ledger, 'prior (granularity 4)').per_device[0]
    fine = _named(ledger, 'prior (granularity 128)').per_device[0]
    assert coarse > fine
    assert ledger.peak_bytes(0) >= coarse

    # Every persistent array is a share of the volume, so it shrinks exactly
    # with the device count.
    one = estimate_peak_device_bytes(make_plan(n_devices=1))
    four = estimate_peak_device_bytes(make_plan(n_devices=4))
    persistent_1 = dict(_named(one, 'prior').terms)['error sinogram'][0]
    persistent_4 = dict(_named(four, 'prior').terms)['error sinogram'][0]
    assert persistent_4 == persistent_1 // 4


def test_weights_are_charged_once_and_only_where_they_are_resident():
    """The hessian's weight array is an ALIAS of supplied weights.

    Charging both would over-count a full sinogram on every weighted run,
    which is the common case.  Supplied weights are placed at the top of
    _vcd_recon, so they are resident from there on; the unweighted run's
    all-ones array is built INSIDE the hessian block, so nothing
    weights-shaped is resident before it.
    """
    sino_bytes = 64 * 32 * 32 * 4
    unweighted = estimate_peak_device_bytes(make_plan(weights_supplied=False))
    weighted = estimate_peak_device_bytes(make_plan(weights_supplied=True))
    # The hessian phase holds the weights either way (as the ones array, or as
    # the caller's array), so it must read identically.
    assert (_named(unweighted, 'hessian diagonal').per_device[0]
            == _named(weighted, 'hessian diagonal').per_device[0])
    # The subset back projection materializes a weighted product only when
    # weights are supplied: exactly one sinogram-shaped array more.
    back_un = _named(unweighted, 'back projection').per_device[0]
    back_w = _named(weighted, 'back projection').per_device[0]
    assert back_w - back_un == sino_bytes
    # Resident through the pre-loop phases when supplied, absent when not.
    for fragment in ('direct recon', 'initial forward projection',
                     'error sinogram formation'):
        assert dict(_named(weighted, fragment).terms)['weights'][0] \
            == sino_bytes, fragment
        assert dict(_named(unweighted, fragment).terms)['weights'][0] == 0, \
            fragment
    # From the hessian onward the unweighted run has one too.
    assert dict(_named(unweighted, 'hessian diagonal').terms)[
        'hessian weights'][0] == sino_bytes
    assert dict(_named(unweighted, 'subset prior').terms)['weights'][0] \
        == sino_bytes


def test_supplied_state_drops_the_phases_it_replaces():
    """A supplied hessian, a supplied initial volume, and a resume each drop
    the phases that would have produced them."""
    # A grid much larger than the masked set makes the hessian dominate, so
    # dropping it has to move the peak and not merely the phase list.
    shape = dict(recon=(64, 64, 32), num_pixels_full=800)
    full = estimate_peak_device_bytes(make_plan(**shape))
    assert _has(full, 'direct recon') and _has(full, 'hessian diagonal')
    assert full.dominant_phase(0).name == 'hessian diagonal'
    supplied = estimate_peak_device_bytes(make_plan(
        init_recon_supplied=True, fm_hessian_supplied=True, **shape))
    assert not _has(supplied, 'direct recon')
    assert not _has(supplied, 'hessian diagonal')
    assert supplied.peak_bytes(0) < full.peak_bytes(0)

    resumed = estimate_peak_device_bytes(make_plan(
        resume=True, init_recon_supplied=True, fm_hessian_supplied=True))
    names = [p.name for p in resumed.phases]
    assert not any('direct recon' in n or 'initial forward' in n
                   or 'error sinogram formation' in n or 'hessian' in n
                   for n in names)
    # What survives is the loop: the subset steps and the per-iteration
    # statistics, which run on every iteration however the state was reached.
    assert all('subset' in n or n.startswith('per-iteration statistics')
               for n in names)
    assert any(n.startswith('per-iteration statistics') for n in names)


# ── the multi-device terms ───────────────────────────────────────────────────
def test_band_reduce_shrinks_with_the_device_count():
    """The signature that replaced the flat one, and the closed form it rests
    on.

    sum_band_to_owner used to move all n partials onto the owner before
    summing them, so the owner held n bands plus the running total.  One band
    is the whole shard by default, so that was about 1.5x a full-volume
    cylinder set at BOTH two and four devices: adding devices shrank the
    persistent set and left this where it was.  The reduce now streams each
    arriving partial in bounded row slabs, so the owner holds its running
    total, the partial it produced itself, and one slab per source --
    ``num_pixels x (shard + band)`` plus ``(n - 1)`` slabs, which is two
    cylinder-SHARDS at the default band and therefore halves when the device
    count doubles.

    Priced at a production-like size, where a band is far larger than one
    slab.  At the small sizes the other tests use, a whole band fits inside a
    single slab and moves in one piece, exactly as it always did.
    """
    pixels, slices = 800_000, 1024

    def reduce_bytes(n):
        ledger = estimate_peak_device_bytes(make_plan(
            n_devices=n, recon=(1024, 1024, slices),
            num_pixels_full=pixels, granularities=(1,)))
        return dict(_sub(ledger, 'subset back projection', n,
                         'band reduce').terms)['band reduce'][0]

    # The closed form, pinned exactly at both counts.
    for n in (2, 4):
        band = slices // n                       # one shard, the default band
        slab = _sharding.reduce_slab_rows(pixels, band * 4) * band * 4
        assert reduce_bytes(n) == 2 * pixels * band * 4 + (n - 1) * slab
    two, four = reduce_bytes(2), reduce_bytes(4)
    # It now falls with the device count instead of standing still.  Not
    # exactly a half, because the slab term is a fixed number of bytes and
    # there are three of them at four devices against one at two; at the
    # 256 MiB slab (measured best, 2026-08-17) that term is a larger share
    # than the old 64 MiB slab was, so the ratio sits near 0.69.
    assert 0.5 <= four / two <= 0.75
    # And it is well under what the old materialize-then-sum form charged:
    # n + 1 bands at two devices, n + 2 at four.
    assert two < 0.8 * 3 * pixels * (slices // 2) * 4
    assert four < 0.6 * 6 * pixels * (slices // 4) * 4
    assert reduce_bytes(1) == 0       # a single device never runs the reduce


def test_band_reduce_charges_the_whole_shard_and_the_padded_band():
    """A band smaller than the shard means several reduces per owner, and the
    owner holds the ones it has finished until it concatenates them.

    The old charge counted only the band in flight, so it fell toward zero as
    the band narrowed while the owner really was holding most of a shard.
    The ``shard + band`` form covers both: the bands already done, at most
    ``shard - band``, and the two live ones.

    The partial the owner produced itself came from a kernel wrapper, which
    allocates the band rounded up to a multiple of 16, so that one term is
    charged at the padded length.
    """
    p_sub, shard = math.ceil(800 / 4), 16

    def charge(band_length):
        led = estimate_peak_device_bytes(make_plan(n_devices=2,
                                                   back_band=band_length))
        return dict(_sub(led, 'subset back projection', 2,
                         'band reduce').terms)['band reduce'][0]

    band = 4
    own_partial = padded_kernel_width(band)
    assert own_partial == 16
    slab = _sharding.reduce_slab_rows(p_sub, band * 4) * band * 4
    assert charge(band) == p_sub * (shard + own_partial) * 4 + slab

    # The floor rule in the place it bites: however narrow the band, the owner
    # still ends the pass holding a whole shard, so the charge may not fall
    # under one cylinder-shard.  The old form did, which is what this
    # replaces: it charged only the band in flight.
    one_shard = p_sub * shard * 4
    for band_length in (1, 2, 4, 8, 16):
        assert charge(band_length) > one_shard, band_length
    # And it still falls as the band narrows, so the knob remains a lever.
    assert charge(1) < charge(4) < charge(16)

    def terms(slices):
        ledger = estimate_peak_device_bytes(
            make_plan(n_devices=2, recon=(32, 32, slices)))
        workers = dict(_sub(ledger, 'subset back projection', 2,
                            'back workers').terms)
        reduce_phase = dict(_sub(ledger, 'subset back projection', 2,
                                 'band reduce').terms)
        return workers['back output'][0], reduce_phase['band reduce'][0]

    # 40 slices over two devices is a shard and a band of 20, which rounds up
    # to 32.  The two live blocks are charged at 32 slices each, and the
    # reduce holds the owner's own partial at 32 beside its shard at 20.
    # 32 slices over two devices is a band of 16, which rounds up to itself,
    # so both terms read exactly as they did before the padding existed.
    assert padded_kernel_width(20) == 32
    assert padded_kernel_width(16) == 16
    for band in (20, 16):
        padded = padded_kernel_width(band)
        blocks, reduce_bytes = terms(2 * band)
        assert blocks == 2 * p_sub * padded * 4, band
        slab = _sharding.reduce_slab_rows(p_sub, band * 4) * band * 4
        assert reduce_bytes == p_sub * (band + padded) * 4 + slab, band


def test_each_term_lands_only_where_its_array_exists():
    """A device with no real views does no projection; one with no real slices
    holds no band; the partition sequence is the lead device's alone."""
    # 3 views over 4 devices: device 3 owns no real view.
    ledger = estimate_peak_device_bytes(
        make_plan(n_devices=4, num_views=3, recon=(32, 32, 32)))
    assert dict(_named(ledger, 'back projection').terms)['back batch'][3] == 0
    # And with 3 slices over 4 devices, device 3 owns no real slice.
    ledger = estimate_peak_device_bytes(
        make_plan(n_devices=4, num_views=64, recon=(32, 32, 3)))
    reduce_phase = _sub(ledger, 'subset back projection', 4, 'band reduce')
    assert dict(reduce_phase.terms)['band reduce'][3] == 0
    # The band it finished is charged only where a band lands, too.
    workers = _sub(ledger, 'subset back projection', 4, 'back workers')
    assert dict(workers.terms)['finished own band'][3] == 0
    # The partitions are charged to the lead device only.
    two = estimate_peak_device_bytes(make_plan(n_devices=2))
    partitions = dict(_named(two, 'prior').terms)['partitions (lead device)']
    assert partitions[0] > 0
    assert partitions[1] == 0


def test_the_back_output_holds_the_blocks_the_view_loop_realizes():
    """The driver evaluates `block = back_body(...)` before rebinding the
    name, so the previous block is still alive while the next is produced:
    accumulator, outgoing block, incoming block.  With a cost model the count
    follows the batches actually realized: min(2, view_batches).
    """
    p_sub = math.ceil(800 / 4)
    ledger = estimate_peak_device_bytes(make_plan(n_devices=1))
    back = dict(_named(ledger, 'subset back projection').terms)['back output'][0]
    assert back == 3 * p_sub * 32 * 4
    # A multi-device slice-owner instead accumulates band parts and
    # concatenates them, which is two.
    shared = estimate_peak_device_bytes(make_plan(n_devices=2))
    back2 = dict(_named(shared, 'subset back projection').terms)['back output'][0]
    assert back2 == 2 * p_sub * 16 * 4

    def blocks(batches_per_device):
        # A charge that yields exactly `batches_per_device` batches over the
        # 32 views each of two devices owns.
        def charge(direction, num_pixels, band_cols):
            return max(1, 32 // batches_per_device), 1
        led = estimate_peak_device_bytes(
            make_plan(n_devices=2, view_charge=charge))
        terms = dict(_sub(led, 'subset back projection', 2,
                          'back workers').terms)
        return terms['back output'][0] / (p_sub * 16 * 4)

    assert blocks(1) == 1                        # one batch, one block
    assert blocks(2) == 2
    assert blocks(8) == 2                        # capped at the accumulator + 1
    # No cost model at all: the ceiling, which is what the docstring promises.
    assert dict(_sub(shared, 'subset back projection', 2,
                     'back workers').terms)['back output'][0] \
        == 2 * p_sub * 16 * 4


def test_the_initial_error_state_charges_blocks_not_whole_sinograms():
    """The two dot products that set the initial scale reduce in blocks, and
    the error sinogram is formed in the projection's own buffer.

    Both branches used to bind the weights product `weights * fwd` and to
    reduce `sum(wf * fwd)` through a whole product temporary beside it.  That
    pair of sinogram-shaped arrays made this sub-phase the widest instant of
    a weighted initialization.  The reductions now walk a block of views at a
    time, so neither array exists and what is charged is two blocks.

    Two blocks are charged on the unweighted path as well, where only one is
    really live.  A ledger may over-charge and may not under-charge, and the
    single rule keeps the two paths from drifting apart.
    """
    sino_bytes = 64 * 32 * 32 * 4
    assert sino_bytes < _memory_ledger.ELL1_CHUNK_BYTES
    for n_devices, weights_supplied in ((1, True), (2, True), (1, False)):
        ledger = estimate_peak_device_bytes(
            make_plan(n_devices=n_devices, weights_supplied=weights_supplied))
        terms = dict(_named(ledger, 'initial dot products').terms)
        assert 'weighted forward projection' not in terms
        assert 'dot product temporary' not in terms
        assert terms['dot product blocks'][0] == 2 * sino_bytes // n_devices

    # `_initial_error_state` scales the projection by -alpha in place and adds
    # the sinogram into it, so the projection and the error sinogram are one
    # array and the scaled copy has no existence.  Written as a relation
    # between the two phases and their own terms rather than as byte counts,
    # so it holds at every size and device count.
    for n_devices in (1, 2):
        ledger = estimate_peak_device_bytes(
            make_plan(n_devices=n_devices, weights_supplied=True))
        dots = _named(ledger, 'initial dot products')
        error = _named(ledger, 'error sinogram formation')
        dot_terms, err_terms = dict(dots.terms), dict(error.terms)
        assert 'alpha-scaled projection' not in err_terms
        assert 'forward projection' not in err_terms
        for i in range(n_devices):
            state = (dot_terms['sinogram'][i] + dot_terms['weights'][i]
                     + dot_terms['forward projection'][i]
                     + dot_terms['init recon'][i])
            overhead = (dot_terms['library workspace'][i]
                        + dot_terms['partitions (lead device)'][i])
            blocks = dot_terms['dot product blocks'][i]
            assert dots.per_device[i] == state + blocks + overhead
            # The projection's buffer IS the error sinogram, so the same
            # state carries into the next phase under a different name.
            assert (err_terms['error sinogram'][i]
                    == dot_terms['forward projection'][i])
            assert error.per_device[i] == state + overhead
            assert dots.per_device[i] == error.per_device[i] + blocks

    # At a production-sized sinogram the blocks are capped by the reduction
    # chunk instead of following the sinogram.  That is what takes the initial
    # error state off the peak of a large weighted run.
    target = _memory_ledger.ELL1_CHUNK_BYTES
    big = estimate_peak_device_bytes(make_plan(
        num_views=2048, num_rows=256, num_channels=256, weights_supplied=True))
    charged = dict(_named(big, 'initial dot products')
                   .terms)['dot product blocks'][0]
    assert charged == 2 * target
    assert charged < 2048 * 256 * 256 * 4 / 8


def test_the_per_iteration_statistics_hold_capped_blocks():
    """The squared-error transient is two blocks -- the squares and their
    weighted form -- and the recon ell-1 is its own sub-phase beside it.

    The ell-1 phase used to charge the squared-error products alone, on the
    reading that the L1 "fuses into its own reduction and materializes
    nothing".  It does not: ``sum(abs(flat_recon))`` allocated a whole second
    recon.  Both reductions are now bounded to a chunk, so neither charge
    follows the array it reduces.
    """
    sino_bytes = 64 * 32 * 32 * 4
    target = _memory_ledger.ELL1_CHUNK_BYTES
    assert sino_bytes < target
    ledger = estimate_peak_device_bytes(make_plan())
    squared = _named(ledger, 'per-iteration statistics (squared error)')
    ell1 = _named(ledger, 'per-iteration statistics (recon ell-1)')
    # Below one chunk the sinogram is reduced whole and a block is the whole
    # sinogram -- the arithmetic the chunked form replaced.
    assert dict(squared.terms)['squared-error products'][0] == 2 * sino_bytes
    # Both carry the persistent set, as every in-loop phase does, and neither
    # holds the other's transient.
    assert dict(squared.terms)['error sinogram'][0] == sino_bytes
    assert dict(squared.terms)['flat recon'][0] > 0
    assert dict(ell1.terms)['recon ell-1 chunk'][0] > 0
    assert dict(ell1.terms)['error sinogram'][0] > 0
    assert 'squared-error products' not in dict(ell1.terms)
    assert 'recon ell-1 chunk' not in dict(squared.terms)

    # Few views, big recon: two sinograms are much smaller than one recon,
    # which is the geometry the old ell-1 charge missed.
    big_recon = estimate_peak_device_bytes(make_plan(
        recon=(512, 512, 512), num_views=8, num_rows=64, num_channels=64,
        num_pixels_full=512 * 512))
    terms = dict(_named(big_recon,
                        'per-iteration statistics (recon ell-1)').terms)
    products = dict(_named(big_recon, 'per-iteration statistics (squared '
                                      'error)').terms)['squared-error products']
    recon_bytes = terms['flat recon'][0]
    chunk = terms['recon ell-1 chunk'][0]
    assert recon_bytes > 100 * products[0]
    assert chunk <= recon_bytes / 16
    assert 0.5 * target <= chunk <= 2 * target

    # A production-sized sinogram: the charge is two chunks rather than two
    # sinograms, which is what takes this phase off the peak of a large run.
    big_sino = estimate_peak_device_bytes(make_plan(
        num_views=2048, num_rows=256, num_channels=256))
    charged = dict(_named(big_sino, 'per-iteration statistics (squared error)')
                   .terms)['squared-error products'][0]
    assert charged == 2 * target
    assert charged < 2048 * 256 * 256 * 4 / 8

    # The reduction splits the view axis, so a block cannot be finer than one
    # view: a sinogram with few views and large detector planes holds a block
    # larger than the byte rule alone gives.  Every phase that reduces a
    # sinogram in blocks prices its block by the same rule, so the initial dot
    # products are checked here beside the per-iteration statistics.
    few_views = estimate_peak_device_bytes(
        make_plan(num_views=4, num_rows=2048, num_channels=3000))
    view_bytes = 2048 * 3000 * 4
    assert view_bytes > target
    assert dict(_named(few_views, 'per-iteration statistics (squared error)')
                .terms)['squared-error products'][0] == 2 * view_bytes
    assert dict(_named(few_views, 'initial dot products').terms)[
        'dot product blocks'][0] == 2 * view_bytes


# ── the back projection's two sub-steps ──────────────────────────────────────
SPLIT_PARENTS = ('direct recon (back loop)', 'hessian diagonal',
                 'subset back projection')


def test_consecutive_sub_phases_contribute_their_max_never_their_sum():
    """Sub-steps that feed one another are consecutive sub-peaks, not one sum.

    The back accumulator feeds the scatter; the back workers project and the
    reduce gathers.  Both are emitted and the per-device maximum over phases
    picks between them.  Charging their sum priced a peak that is never live,
    which was the largest over-charge at n>1.  Picking one whole sub-phase by
    a cross-device total would under-charge a device where the other is
    larger, so the selection must stay per device.
    """
    ledger = estimate_peak_device_bytes(make_plan())
    loop = _named(ledger, 'direct recon (back loop)')
    scatter = _named(ledger, 'direct recon (scatter)')
    assert 'scatter buffer' not in dict(loop.terms)
    assert 'back output' not in dict(scatter.terms)
    combined = max(loop.per_device[0], scatter.per_device[0])
    assert combined < loop.per_device[0] + scatter.per_device[0]
    assert ledger.peak_bytes(0) >= combined

    two = estimate_peak_device_bytes(make_plan(n_devices=2))
    names = [p.name for p in two.phases]
    for i in (0, 1):
        assert two.peak_bytes(i) >= max(
            _named(two, 'direct recon (back loop)').per_device[i],
            _named(two, 'direct recon (scatter)').per_device[i])
    for parent in SPLIT_PARENTS:
        workers = _sub(two, parent, 2, 'back workers')
        reduce_phase = _sub(two, parent, 2, 'band reduce')
        for i in (0, 1):
            both = max(workers.per_device[i], reduce_phase.per_device[i])
            assert both < workers.per_device[i] + reduce_phase.per_device[i]
            assert two.peak_bytes(i) >= both
        # And the unsplit parent name is gone -- nothing charges the sum.
        assert parent not in names

    # The peak really sits below what summing the two would have charged.
    four = estimate_peak_device_bytes(make_plan(
        n_devices=4, recon=(64, 64, 32), num_pixels_full=3000))
    for parent in SPLIT_PARENTS:
        summed = (_sub(four, parent, 4, 'back workers').per_device[0]
                  + _sub(four, parent, 4, 'band reduce').per_device[0])
        assert four.peak_bytes(0) < summed


def test_the_forward_block_is_sized_by_its_geometry_and_batch_count():
    """A two-fan body's output plane spans the FULL detector rows, so the
    block follows the row count.  A row-aligned body sizes its output by the
    values it was handed, which under sharding is a transferred cylinder --
    the WHOLE slice axis, at every device count.  The two are separated here
    by a plan whose slice count differs from its detector row count.

    The count follows the realized view batches.  The view-range loop has no
    release, so it holds the outgoing block and the incoming one:
    min(2, view_batches).  For a body that declares its own per-view cost one
    of those is already inside the batch charge -- a forward kernel body's
    declaration prices its output plane per view -- so this term charges the
    remainder.  A torch body declares nothing, so both blocks are charged.
    """
    rows, channels, slices = 32, 32, 48
    for aligned, expected in ((True, slices), (False, rows)):
        for n in (1, 2, 4):
            ledger = estimate_peak_device_bytes(
                make_plan(n_devices=n, rows_track_slices=aligned,
                          num_rows=rows, recon=(32, 32, slices)))
            terms = dict(_named(ledger, 'initial forward projection').terms)
            assert terms['forward block'][0] == expected * channels * 4, \
                (aligned, n)

    # make_plan defaults to a two-fan geometry: one whose detector rows are
    # not tied 1:1 to recon slices, as in cone beam.
    def block(batches_per_device, directions=()):
        # A charge that yields exactly `batches_per_device` batches over the
        # 32 views each of two devices owns.
        view_batch = max(1, 32 // batches_per_device)

        def charge(direction, num_pixels, band_cols):
            return view_batch, 1

        ledger = estimate_peak_device_bytes(make_plan(
            n_devices=2, view_charge=charge,
            torch_body_directions=directions))
        return (dict(_named(ledger, 'initial forward projection')
                     .terms)['forward block'][0], view_batch)

    assert block(1)[0] == 0                  # one batch, one block, all priced
    for batches in (2, 8):
        charged, view_batch = block(batches)
        assert charged == view_batch * rows * channels * 4
    # No cost model at all: the ceiling of two, one of them charged here,
    # which is what the docstring promises.
    plain = estimate_peak_device_bytes(make_plan(n_devices=2))
    assert dict(_named(plain, 'initial forward projection')
                .terms)['forward block'][0] == 1 * rows * channels * 4
    # A torch body pays for both blocks; the back direction alone leaves the
    # forward's own term where it was.
    assert block(4, ('forward', 'back'))[0] == 2 * 8 * rows * channels * 4
    assert block(4, ('back',))[0] == 1 * 8 * rows * channels * 4


# ── the masked hessian ───────────────────────────────────────────────────────
def test_masked_hessian_agrees_with_the_full_grid_at_the_masked_indices():
    """The only places the engine ever reads the hessian.

    Back projection is independent per pixel, so a masked run must reproduce
    the dense run exactly at every masked index.  Outside the mask the masked
    run holds zeros instead of computed-but-never-read values.
    """
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    weights = np.abs(np.random.RandomState(0).randn(12, 8, 10)).astype(np.float32) + 0.5

    dense = model.compute_hessian_diagonal(weights=weights)
    indices = model.full_indices_device()
    masked = model.compute_hessian_diagonal(weights=weights, indices=indices)

    shape = tuple(model.get_params('recon_shape'))
    flat_dense = dense.reshape(-1, shape[2])
    flat_masked = masked.reshape(-1, shape[2])
    idx = indices.cpu().numpy()
    np.testing.assert_array_equal(flat_masked[idx], flat_dense[idx])
    # Outside the mask the masked form is exactly zero.
    outside = np.setdiff1d(np.arange(shape[0] * shape[1]), idx)
    if outside.size:
        assert np.all(flat_masked[outside] == 0)
        assert np.any(flat_dense[outside] != 0)


def test_every_index_the_loop_reads_is_inside_the_mask():
    """The precondition the masked hessian rests on, asserted directly.

    The masked hessian is zero outside the ROR set, so the change is
    value-preserving only if the loop never reads there.  The loop reads the
    hessian at partition indices and nowhere else, so this test pins that
    every partition index is inside the same mask that `full_indices_device`
    returns.
    """
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    sinogram = np.zeros((12, 8, 10), dtype=np.float32)
    sinogram[:, 4, 5] = 1.0
    (_s, _w, _i, partitions, _seq, _g,
     _r) = model.initialize_recon(sinogram, None, None, 3, 0)
    masked = set(model.full_indices_device().cpu().numpy().tolist())
    for partition in partitions:
        outside = set(partition.cpu().numpy().ravel().tolist()) - masked
        assert not outside, f'{len(outside)} partition indices outside the mask'


def test_recon_is_bitwise_identical_with_the_masked_hessian():
    """The whole-recon parity, in eager.

    The two runs differ in ONE variable: where the hessian came from.  The
    control supplies one computed the dense way, which bypasses the internal
    masked call; the comparison run lets _vcd_recon compute it at the masked
    indices.

    Compilation is OFF deliberately, and not to hide a difference.  The two
    runs necessarily back-project at different pixel counts, so they compile
    different shapes, and dynamo's shape specialization then perturbs the
    float realization of kernels that have nothing to do with the hessian.
    A compiled whole-recon comparison therefore measures the compiler rather
    than this change.  The change's own value claim is proved directly by
    the two tests above: the hessian is bitwise equal at every masked index,
    and the loop reads nowhere else.
    """
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles,
                                        compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    sinogram = np.zeros((12, 8, 10), dtype=np.float32)
    sinogram[:, 4, 5] = 1.0
    weights = np.full((12, 8, 10), 0.75, dtype=np.float32)

    # One set of partitions, shared, so the hessian is the only difference.
    (_s, _w, _i, partitions, sequence, _g,
     _r) = model.initialize_recon(sinogram, weights, None, 3, 0)
    dense_hessian = model.compute_hessian_diagonal(weights=weights,
                                                   output_sharded=True)

    np.random.seed(7)
    control, _losses = model._vcd_recon(sinogram.copy(), partitions, sequence,
                                        0.0, weights=weights,
                                        fm_hessian=dense_hessian.clone())
    np.random.seed(7)
    masked, _losses = model._vcd_recon(sinogram.copy(), partitions, sequence,
                                       0.0, weights=weights)
    np.testing.assert_array_equal(control.cpu().numpy(), masked.cpu().numpy())


def test_ledger_charges_the_masked_hessian_and_its_scatter():
    dense = estimate_peak_device_bytes(make_plan(
        recon=(64, 64, 32), num_pixels_full=3000, hessian_masked=False))
    masked = estimate_peak_device_bytes(make_plan(
        recon=(64, 64, 32), num_pixels_full=3000, hessian_masked=True))
    # The loop's cylinders follow the masked count, not the grid.
    assert dict(_named(masked, 'hessian diagonal').terms)['back output'][0] \
        == 3 * 3000 * 32 * 4
    assert dict(_named(dense, 'hessian diagonal').terms)['back output'][0] \
        == 3 * (64 * 64) * 32 * 4
    # The scatter exists only on the masked path, and stays below the loop.
    assert not _has(dense, 'hessian scatter')
    scatter = _named(masked, 'hessian scatter').per_device[0]
    assert scatter < _named(masked, 'hessian diagonal').per_device[0]
    assert masked.peak_bytes(0) < dense.peak_bytes(0)


# ── the projector cost model ─────────────────────────────────────────────────
def test_view_batch_charge_matches_the_driver_batch():
    """One cost model, two consumers: the number the ledger prices must be the
    number the driver would actually run.  It must also be answerable for a
    layout the model is not in, since that is what the ledger prices."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    fwd_body, back_body = model._view_batch_bodies()
    args = model._view_batch_args()
    for body in (fwd_body, back_body):
        batch, per_view = model.projector_functions.view_batch_charge(
            body, 40, 6, args)
        assert batch == model.projector_functions._effective_view_batch(
            body, 40, 6, args)
        assert per_view > 0

    big = mbirtorch.ParallelBeamModel(
        (512, 64, 64), np.linspace(0, np.pi, 512, endpoint=False))
    big.configure_devices(devices=['cpu'])
    big_fwd, _back = big._view_batch_bodies()
    big_args = big._view_batch_args()
    live = big.projector_functions.view_batch_charge(big_fwd, 400, 64, big_args)
    hypothetical = big.projector_functions.view_batch_charge(
        big_fwd, 400, 64, big_args, n_devices=4)
    assert big.sino_placement.n_devices == 1         # unchanged by the query
    assert live[1] == hypothetical[1]                # same per-view charge


# ── the verdict and the message ──────────────────────────────────────────────
def test_layout_fits_applies_the_margin_and_the_credits():
    ledger = _fixed_ledger(peak=10 * GB)
    fits, rows = _memory_ledger.layout_fits(ledger, [11 * GB], margin=0.15)
    assert not fits                                  # 11.5 GB demanded
    fits, _rows = _memory_ledger.layout_fits(ledger, [12 * GB], margin=0.15)
    assert fits
    # A caller array already on the device is already excluded from the
    # budget, so charging it again would double-count it.
    fits, rows = _memory_ledger.layout_fits(
        ledger, [11 * GB], credits=[2 * GB], margin=0.15)
    assert fits
    assert rows[0][1] == int(1.15 * 8 * GB)
    # And a credit is counted only for a matching CUDA device.
    cpu_tensor = torch.zeros(1024, dtype=torch.float32)
    assert _memory_ledger.resident_credits(['cpu'], [cpu_tensor]) == [0]
    assert _memory_ledger.resident_credits(['cpu'], [None]) == [0]


# ── the direct workload ──────────────────────────────────────────────────────
# A direct reconstruction is the filter and one back projection.  It is priced
# as its own plan because the policy checks capacity against the call in
# progress, while still choosing the device count for a full recon.
def test_the_direct_plan_is_the_filter_and_one_back_projection():
    """The phases a direct reconstruction really runs, and nothing else.

    The filter holds the placed sinogram, the copy it writes, and its own row
    batch; the back projection reads that copy and scatters into the volume.
    No prior, no hessian, no partitions and no loop exist while it runs, and
    dropping them has to move the peak and not merely the phase list.
    """
    sino_bytes = 64 * 32 * 32 * 4
    ledger = estimate_peak_device_bytes(make_plan(workload='direct'))
    assert [p.name for p in ledger.phases] == [
        'direct recon (filter)', 'direct recon (back loop)',
        'direct recon (scatter)']
    terms = dict(_named(ledger, 'direct recon (filter)').terms)
    assert terms['sinogram'][0] == sino_bytes
    assert terms['filtered sinogram'][0] == sino_bytes
    assert terms['filter row batch'][0] > 0
    assert terms['library workspace'][0] == \
        _memory_ledger.FIXED_DEVICE_OVERHEAD_BYTES
    # The partitions are the full plan's: a direct reconstruction builds no
    # partition sequence, so the lead device does not carry one.
    assert 'partitions (lead device)' not in terms
    scatter = dict(_named(ledger, 'direct recon (scatter)').terms)
    assert scatter['scatter buffer'][0] == 32 * 32 * 32 * 4

    # A grid much larger than the masked set makes the hessian dominate the
    # full plan, so the phases the direct plan drops move the peak.
    shape = dict(recon=(64, 64, 32), num_pixels_full=800)
    full = estimate_peak_device_bytes(make_plan(**shape))
    direct = estimate_peak_device_bytes(make_plan(workload='direct', **shape))
    assert direct.peak_bytes(0) < full.peak_bytes(0)
    for gone in ('hessian', 'subset', 'per-iteration statistics',
                 'initial forward projection'):
        assert not _has(direct, gone)


def test_workload_covers_is_the_table_it_claims_to_be():
    """The full plan charges everything the direct plan charges, so a layout
    checked for a recon needs no direct check.  A denoise holds arrays neither
    of the other plans holds, so no check substitutes for it and it
    substitutes for none.  Nothing else is claimed."""
    assert _memory_ledger.workload_covers('recon', 'direct')
    assert _memory_ledger.workload_covers('direct', 'direct')
    assert _memory_ledger.workload_covers('recon', 'recon')
    assert not _memory_ledger.workload_covers('direct', 'recon')
    assert not _memory_ledger.workload_covers(None, 'recon')
    assert _memory_ledger.workload_covers('denoise', 'denoise')
    for other in ('recon', 'direct', None):
        assert not _memory_ledger.workload_covers(other, 'denoise')
        assert not _memory_ledger.workload_covers('denoise', other)


# ── the denoise workload ─────────────────────────────────────────────────────
# One QGGMRFDenoiser sweep.  The denoiser has no projectors at all, its
# sinogram shape is its image shape, and it fixes one partition, so it is
# priced as its own plan rather than as a reconstruction with terms zeroed.
DENOISE_PHASES = [
    'denoise state placement',
    'denoise subset prior (granularity 16)',
    'denoise subset update direction (granularity 16)',
    'denoise subset state application (granularity 16)',
    'denoise per-pass statistics',
]


def make_denoise_plan(image=(32, 32, 32), num_pixels_full=None, **kwargs):
    """A hand-built denoiser plan.

    The sinogram shape IS the image shape, as QGGMRFDenoiser sets it, and the
    pixel set is the whole unmasked grid, which is what the denoiser's
    ``use_ror_mask=False`` default keeps.  The granularity is the denoiser's
    own fixed 16.
    """
    rows, cols, slices = image
    kwargs.setdefault('granularities', (16,))
    kwargs.setdefault('num_views', rows)
    return make_plan(workload='denoise', num_rows=cols, num_channels=slices,
                     recon=image,
                     num_pixels_full=(rows * cols if num_pixels_full is None
                                      else num_pixels_full),
                     **kwargs)


def test_the_denoise_plan_is_the_state_and_the_subset_sweep():
    """The phases a denoiser really runs, and nothing else.

    It places the image, clones it into the working image and forms the
    residual; then per subset it runs the qGGMRF prior, forms the update
    direction, and applies it; then once per pass it reads the working
    image's ell-1 norm.  Nothing else exists while it runs: its
    ``create_projectors`` is a no-op, so no view batch, no projection block
    and no assembled projection output exists; it builds no hessian diagonal
    and no weights array; and it builds one partition rather than a sequence.
    """
    image_bytes = 32 * 32 * 32 * 4
    ledger = estimate_peak_device_bytes(make_denoise_plan())
    assert [p.name for p in ledger.phases] == DENOISE_PHASES
    terms = dict(_named(ledger, 'denoise state placement').terms)
    assert terms['input image'][0] == image_bytes
    assert terms['working image'][0] == image_bytes
    assert terms['residual'][0] == image_bytes
    assert terms['library workspace'][0] == \
        _memory_ledger.FIXED_DEVICE_OVERHEAD_BYTES
    # The initial image aliases the input unless the caller supplies one.
    assert terms['init image'][0] == 0
    supplied = estimate_peak_device_bytes(
        make_denoise_plan(init_recon_supplied=True))
    assert dict(_named(supplied, 'denoise state placement').terms)[
        'init image'][0] == image_bytes
    assert supplied.peak_bytes(0) - ledger.peak_bytes(0) == image_bytes

    sharded = estimate_peak_device_bytes(make_denoise_plan(n_devices=2))
    charged = {name for phase in sharded.phases for name, _vals in phase.terms}
    for absent in ('batch', 'block', 'forward', 'back', 'hessian', 'weights',
                   'sinogram', 'scatter', 'band', 'partitions'):
        assert not any(absent in name for name in charged), (absent, charged)
    # And the view axis never enters: the same plan with every view on one
    # device reads identically, because only the slice split is used.
    lopsided = estimate_peak_device_bytes(
        make_denoise_plan(n_devices=2, num_views=1))
    assert lopsided.per_device_peaks() == sharded.per_device_peaks()


def test_the_denoise_charges_follow_the_subset_size_and_the_slice_split():
    """A coarser partition means a bigger subset, and a bigger subset means a
    bigger working set: every per-subset term is the subset's pixel count by
    the device's slices.  The denoiser divides its image by SLICE, so every
    term is the device's own slice block and no term follows the view axis.
    """
    coarse = estimate_peak_device_bytes(make_denoise_plan(granularities=(4,)))
    fine = estimate_peak_device_bytes(make_denoise_plan(granularities=(64,)))
    for fragment, term in (('prior', 'prior cylinders'),
                           ('update direction', 'direction cylinders'),
                           ('state application',
                            'direction and scaled direction')):
        big = dict(_named(coarse, fragment).terms)[term][0]
        small = dict(_named(fine, fragment).terms)[term][0]
        assert big > small, fragment
        # Exactly the subset ratio: ceil(1024/4) against ceil(1024/64).
        assert big == small * (1024 // 4) // (1024 // 64), fragment
    assert coarse.peak_bytes(0) > fine.peak_bytes(0)

    # The denoiser calls the same prior kernel a reconstruction calls, so it
    # is priced by the same cylinder count and not by one of its own.
    p_sub = math.ceil(1024 / 16)
    for cylinders in (_memory_ledger.QGGMRF_CYLINDERS_COMPILED,
                      _memory_ledger.QGGMRF_CYLINDERS_EAGER):
        ledger = estimate_peak_device_bytes(
            make_denoise_plan(qggmrf_cylinders=cylinders))
        assert dict(_named(ledger, 'denoise subset prior').terms)[
            'prior cylinders'][0] == cylinders * p_sub * 32 * 4

    # 30 slices over four devices: 8, 8, 7, 7.
    ledger = estimate_peak_device_bytes(
        make_denoise_plan(image=(32, 32, 30), n_devices=4))
    blocks = [8, 8, 7, 7]
    for phase in ledger.phases:
        terms = dict(phase.terms)
        assert terms['input image'] == [32 * 32 * b * 4 for b in blocks]
        assert terms['working image'] == terms['input image']
        assert terms['residual'] == terms['input image']
    prior = dict(_named(ledger, 'denoise subset prior').terms)[
        'prior cylinders']
    assert prior == [_memory_ledger.QGGMRF_CYLINDERS_COMPILED * p_sub * b * 4
                     for b in blocks]
    # And the peak falls with the device count, since every image-shaped term
    # is a share of the volume.
    one = estimate_peak_device_bytes(make_denoise_plan(image=(32, 32, 32)))
    four = estimate_peak_device_bytes(make_denoise_plan(image=(32, 32, 32),
                                                        n_devices=4))
    assert four.peak_bytes(0) < one.peak_bytes(0)


def test_the_denoise_statistics_hold_one_chunk_not_one_image():
    """The convergence test reduces the working image a chunk at a time, so at
    any size worth chunking it holds a chunk and not a fourth image.

    As ``sum(abs(working image))`` it held a whole image of absolute values and
    was the denoiser's widest instant.  Bounded to a chunk, the peak falls on
    the qGGMRF prior's working set instead, so this asserts which phase is the
    peak and not merely what the statistic costs.  Below one chunk the
    reduction runs unchunked and the whole image is charged -- the arithmetic
    the chunked form replaced, left alone because an extra image is small in
    absolute terms at those sizes.
    """
    target = _memory_ledger.ELL1_CHUNK_BYTES
    ledger = estimate_peak_device_bytes(
        make_denoise_plan(image=(512, 512, 512)))
    stats = _named(ledger, 'denoise per-pass statistics')
    placement = _named(ledger, 'denoise state placement')
    charged = dict(stats.terms)['ell-1 chunk'][0]
    assert charged == target
    # The chunk is the whole of what this phase adds to the state.
    assert stats.per_device[0] - placement.per_device[0] == charged
    # The peak moved off this phase and onto the prior.
    prior = _named(ledger, 'denoise subset prior')
    assert ledger.peak_bytes(0) == prior.per_device[0]
    assert prior.per_device[0] > stats.per_device[0]

    # The image grows about fifteenfold over this pair; the chunk does not
    # move off the target, because the chunk COUNT absorbs the growth, while
    # the prior, which does scale, really does grow.
    small = estimate_peak_device_bytes(
        make_denoise_plan(image=(256, 256, 256)))
    big = estimate_peak_device_bytes(make_denoise_plan(image=(640, 640, 640)))

    def chunk(led):
        return dict(_named(led, 'per-pass statistics').terms)['ell-1 chunk'][0]

    def prior_bytes(led):
        return dict(_named(led, 'denoise subset prior').terms)[
            'prior cylinders'][0]

    assert 256 ** 3 * 15 < 640 ** 3
    for led in (small, big):
        assert 0.5 * target <= chunk(led) <= 2 * target
    assert chunk(big) < 1.1 * chunk(small)
    assert prior_bytes(big) > 15 * prior_bytes(small)

    # Below one chunk the whole image is charged.
    image_bytes = 32 * 32 * 32 * 4
    assert image_bytes < target
    tiny = estimate_peak_device_bytes(make_denoise_plan())
    assert dict(_named(tiny, 'denoise per-pass statistics').terms)[
        'ell-1 chunk'][0] == image_bytes


def test_the_ledger_chunk_matches_what_the_reduction_really_allocates():
    """The anti-drift gate on the two constants.

    The reductions and ``reduction_chunk_bytes`` share ELL1_CHUNK_BYTES, but
    they are still two pieces of arithmetic that could disagree.  This runs
    the real reductions and reads the largest array each one actually
    allocates, so the charges are checked against the allocation rather than
    against a restatement of the same formula.  Every ell-1, dot-product and
    squared-error charge in the module prices its block with
    reduction_chunk_bytes, so this covers them all.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    from mbirtorch._memory_ledger import (image_ell1, weighted_dot,
                                          weighted_square_sum)

    class Biggest(TorchDispatchMode):
        def __init__(self):
            self.nbytes = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(t, torch.Tensor):
                    self.nbytes = max(self.nbytes,
                                      t.numel() * t.element_size())
            return out

    # One chunked size and one below the chunk, so both branches are checked.
    for num_pixels, num_slices in ((256 * 256, 256), (32 * 32, 32)):
        image = torch.zeros(num_pixels, num_slices, dtype=torch.float32)
        image_bytes = image.numel() * image.element_size()
        with Biggest() as seen:
            image_ell1(image)
        predicted, _n_chunks = _memory_ledger.reduction_chunk_bytes(image_bytes)
        assert seen.nbytes == predicted, (image_bytes, seen.nbytes, predicted)

    # The same gate on the sinogram reductions, whose phases charge two of
    # these blocks: the products and their weighted form.  The dot product of
    # two different sinograms is checked beside the sum of squares, because
    # the initial error state reduces both.
    for num_views, num_rows in ((256, 256), (8, 32)):
        error = torch.zeros(num_views, num_rows, 64, dtype=torch.float32)
        other = torch.zeros_like(error)
        weights = torch.ones_like(error)
        sino_bytes = error.numel() * error.element_size()
        predicted, _n_chunks = _memory_ledger.reduction_chunk_bytes(sino_bytes)
        for weight_arg in (None, weights):
            for call in (lambda w: weighted_square_sum(error, w),
                         lambda w: weighted_dot(error, other, w)):
                with Biggest() as seen:
                    call(weight_arg)
                assert seen.nbytes == predicted, (sino_bytes, seen.nbytes,
                                                  predicted)


def test_the_chunked_reductions_are_accurate_at_a_size_that_chunks():
    """The reductions behind the reported forward-model loss and the initial
    scale, checked where the goldens cannot check them.

    The golden sinograms are far below one chunk, so they only ever exercise
    the unchunked branch.  This runs sinograms large enough to chunk and
    scores each result against a float64 reference over the same float32
    values, so it measures the reductions' own arithmetic rather than the
    loss's.  Both the weighted and the plain forms are checked, because a
    weighted reconstruction uses one and an unweighted one the other.

    The dot product's two arrays are non-negative, as a projection and a
    sinogram are.  A relative gate on a sum whose terms cancel would measure
    the cancellation rather than the reduction.
    """
    from mbirtorch._memory_ledger import weighted_dot, weighted_square_sum
    torch.manual_seed(13)
    error = torch.randn(256, 128, 256)
    weights = torch.rand(256, 128, 256) + 0.5
    assert (error.numel() * error.element_size()
            > _memory_ledger.ELL1_CHUNK_BYTES)
    for w, reference in (
            (None, float((error.double() ** 2).sum())),
            (weights, float((error.double() ** 2 * weights.double()).sum()))):
        value = float(weighted_square_sum(error, w))
        rel = abs(value - reference) / abs(reference)
        assert rel < 1e-6, rel
    del error

    torch.manual_seed(17)
    a = torch.rand(256, 128, 256)
    b = torch.rand(256, 128, 256)
    for w, reference in (
            (None, float((a.double() * b.double()).sum())),
            (weights, float((a.double() * b.double()
                             * weights.double()).sum()))):
        value = float(weighted_dot(a, b, w))
        rel = abs(value - reference) / abs(reference)
        assert rel < 1e-6, rel


def test_the_unchunked_reductions_leave_a_small_sinogram_bit_for_bit():
    """Below one chunk both reductions are the sums they replaced, so small
    problems -- every golden among them -- cannot move at all."""
    torch.manual_seed(5)
    error = torch.randn(16, 8, 8)
    weights = torch.rand(16, 8, 8) + 0.5
    assert (error.numel() * error.element_size()
            < _memory_ledger.ELL1_CHUNK_BYTES)
    assert (float(_memory_ledger.weighted_square_sum(error, weights))
            == float(torch.sum(error * error * weights)))
    assert (float(_memory_ledger.weighted_square_sum(error))
            == float(torch.sum(error * error)))
    # A scalar weight broadcasts, which is the form the loss passes when the
    # caller supplies no weights array.
    assert (float(_memory_ledger.weighted_square_sum(error, 1))
            == float(torch.sum(error * error * 1)))

    torch.manual_seed(7)
    a = torch.randn(16, 8, 8)
    b = torch.randn(16, 8, 8)
    w = torch.rand(16, 8, 8) + 0.5
    assert (float(_memory_ledger.weighted_dot(a, b, w))
            == float(torch.sum(a * b * w)))
    assert (float(_memory_ledger.weighted_dot(a, b))
            == float(torch.sum(a * b)))
    # The sum of squares is this reduction on one array, and routing it
    # through here must not have moved it.
    assert (float(_memory_ledger.weighted_dot(a, a, w))
            == float(_memory_ledger.weighted_square_sum(a, w)))


def test_plan_from_model_prices_a_denoiser_without_its_projectors():
    """The round trip: a live denoiser to a plan to a ledger.

    The projector reads are the reason this needs a branch of its own.  A
    denoiser's ``create_projectors`` is a no-op, so it defines no per-view
    projection bodies, and the recon plan's cost-model read raises on it --
    which is asserted here, so the branch cannot be quietly removed.

    A denoise sweep also builds ONE partition, the one the first entry of the
    sequence names, so the plan may not read the sequence the way a
    reconstruction does: reading it that way would charge partitions the
    denoiser never builds and emit subset phases it never runs.
    """
    denoiser = mbirtorch.QGGMRFDenoiser((16, 16, 12), compile_mode='off')
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0)

    with pytest.raises(NotImplementedError, match='projection bodies'):
        _memory_ledger.plan_from_model(denoiser, ['cpu'])

    plan = _memory_ledger.plan_from_model(denoiser, ['cpu', 'cpu'],
                                          workload='denoise')
    assert plan.workload == 'denoise'
    assert plan.n_devices == 2
    assert denoiser.recon_placement.n_devices == 1    # the model is untouched
    # The denoiser's sinogram shape IS its image shape.
    assert plan.sinogram_shape == plan.recon_shape == (16, 16, 12)
    assert plan.num_pixels_full == denoiser.full_index_count() == 16 * 16
    assert plan.num_pixels_grid == 16 * 16
    # No projector was asked anything, so no cost model and no body reached
    # the plan.
    assert plan.view_charge is None
    assert plan.torch_body_directions == ()
    assert plan.pixel_batch is None
    # The one fixed partition the denoiser builds.
    assert plan.granularities == (16,)
    assert plan.partition_granularities == (16,)
    ledger = estimate_peak_device_bytes(plan)
    assert [p.name for p in ledger.phases] == DENOISE_PHASES
    assert ledger.peak_bytes(0) > 0
    assert ledger.peak_bytes(0) == ledger.peak_bytes(1)   # an even slice split

    # A granularity list of three and a sequence starting at 1 must still
    # yield the single partition granularity[sequence[0]].
    listed = mbirtorch.QGGMRFDenoiser((16, 16, 12), compile_mode='off')
    listed.configure_devices(devices=['cpu'])
    listed.set_params(no_warning=True, verbose=0,
                      granularity=[8, 16, 32], partition_sequence=[1, 2])
    one_device = _memory_ledger.plan_from_model(listed, ['cpu'],
                                               workload='denoise')
    assert one_device.granularities == (16,)
    assert one_device.partition_granularities == (16,)
    single = estimate_peak_device_bytes(one_device)
    assert [p.name for p in single.phases] == DENOISE_PHASES
    # The partition charge is that one partition, not the three the list names.
    assert dict(_named(single, 'denoise subset prior').terms)[
        'subset indices'][0] == 16 * math.ceil(16 * 16 / 16) * 8


# ── the model-facing plan ────────────────────────────────────────────────────
def test_plan_from_model_reads_the_current_params_and_a_candidate_layout():
    """The plan reads the model rather than re-deriving its rules, and asking
    about a candidate device count leaves the model where it is."""
    from mbirtorch.tomography_model import FORWARD_PIXEL_BATCH
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    plan = _memory_ledger.plan_from_model(model, ['cpu', 'cpu'])
    assert plan.n_devices == 2
    assert model.sino_placement.n_devices == 1       # the model is untouched
    assert plan.sinogram_shape == (8, 6, 8)
    assert plan.num_pixels_full == model.full_index_count()
    assert plan.num_pixels_grid == 6 * 6 or plan.num_pixels_grid > 0
    # The visited granularities come from the sequence, not the whole list.
    assert set(plan.granularities) <= set(model.get_params('granularity'))
    ledger = estimate_peak_device_bytes(plan)
    assert ledger.peak_bytes(0) > 0
    # A plan is for a full reconstruction unless the caller says otherwise.
    assert plan.workload == 'recon'
    assert _memory_ledger.plan_from_model(
        model, ['cpu', 'cpu'], workload='direct').workload == 'direct'

    # The pixel batch is the one the model would actually walk, so a changed
    # default or an override reaches the charge without a second edit here.
    # Every projection plan carries it, because the cylinder transfer is the
    # only multi-device forward.
    cell = (8, 8, 8)
    cone = mbirtorch.ConeBeamModel(
        cell, np.linspace(0, 2 * np.pi, cell[0], endpoint=False),
        source_detector_dist=32, source_iso_dist=16)
    cone.configure_devices(devices=['cpu'])
    cone.set_params(no_warning=True, verbose=0)
    devices = ['cpu', 'cpu']
    assert _memory_ledger.plan_from_model(
        cone, devices).pixel_batch == FORWARD_PIXEL_BATCH
    cone.forward_project_pixel_batch = 512
    assert _memory_ledger.plan_from_model(
        cone, devices).pixel_batch == 512
    # The row-aligned geometry takes the same path, so the same resolution has
    # to reach its charge.
    par = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0],
                                                        endpoint=False))
    par.configure_devices(devices=['cpu'])
    par.set_params(no_warning=True, verbose=0)
    assert _memory_ledger.plan_from_model(
        par, devices).pixel_batch == FORWARD_PIXEL_BATCH
    par.forward_project_pixel_batch = 256
    assert _memory_ledger.plan_from_model(
        par, devices).pixel_batch == 256


def test_the_device_layout_policy_consults_and_records_correctly():
    """The ledger's production job is choosing a CUDA device count, so a CPU
    or MPS model never consults one and never pays for it.  The MATH is
    device-agnostic and can be built for any backend, which is what lets these
    tests run; refusing to consult it is the policy's decision.

    Automatic means NO configure_devices call has been made.  The constructor
    amendment collapsed eligibility to that one bit: there is no device string
    to parse, so EVERY call is explicit, including an unindexed
    ``devices=['cuda']``.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    assert model._apply_device_policy() is None
    assert model._build_memory_ledger() is not None

    untouched = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    assert untouched.device_layout_is_automatic is True
    assert model.device_layout_is_automatic is False
    if torch.cuda.is_available():
        plain = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
        plain.configure_devices(devices=['cuda'])
        assert plain.device_layout_is_automatic is False
        indexed = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
        indexed.configure_devices(devices=['cuda:0'])
        assert indexed.device_layout_is_automatic is False

    # A later call takes the layout out of automatic mode again.
    model.device_layout_is_automatic = True          # as a CUDA model would be
    model.configure_devices(devices=['cpu', 'cpu'])
    assert model.device_layout_is_automatic is False


def test_recon_is_unaffected_on_a_cpu_model():
    """The n=1 path must be untouched: no ledger, no preflight, same result."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    sinogram = np.zeros((8, 6, 8), dtype=np.float32)
    sinogram[:, 3, 4] = 1.0
    np.random.seed(0)
    recon, _info = model.recon(sinogram, max_iterations=2)
    assert np.all(np.isfinite(recon))


# ── the torch-body projection charge ─────────────────────────────────────────
# A torch body is a projection body written as general torch code, which is
# what a geometry with no hand-written kernel runs.  It declares no per-view
# cost, so the ledger prices its views itself.
SLABS = _memory_ledger.TORCH_BODY_VIEW_SLABS


def test_torch_body_directions_follow_the_bound_bodies():
    """A body that declares its own per-view cost is priced by that
    declaration; one that declares nothing is a torch body.  The two
    directions are asked separately, because a model may bind a kernel one
    way and a torch body the other."""
    def kernel_body():
        pass
    kernel_body._view_batch_cost = lambda p, cols, args: (1, 1)

    def torch_body():
        pass

    class FakeModel:
        def __init__(self, fwd, back):
            self._bodies = (fwd, back)

        def _view_batch_bodies(self):
            return self._bodies

    directions = _memory_ledger.torch_body_directions
    assert directions(FakeModel(kernel_body, kernel_body)) == ()
    assert directions(FakeModel(torch_body, torch_body)) == ('forward', 'back')
    assert directions(FakeModel(torch_body, kernel_body)) == ('forward',)
    assert directions(FakeModel(kernel_body, torch_body)) == ('back',)


def test_a_declared_per_view_cost_is_charged_exactly_as_declared():
    """The kernel-declared path may not move: a body that states what one of
    its views holds is charged that and nothing more."""
    def charge(direction, num_pixels, band_cols):
        return 8, 1024                       # 8 views at 1 KiB each

    ledger = estimate_peak_device_bytes(make_plan(view_charge=charge))
    terms = dict(_named(ledger, 'back projection').terms)
    assert terms['back batch'][0] == 8 * 1024
    forward = dict(_named(ledger, 'initial forward projection').terms)
    assert forward['forward batch'][0] == 8 * 1024


def test_the_torch_body_slab_follows_the_wider_of_rows_and_slices():
    """A torch body holds a loop of slabs where the driver's nominal charge
    prices one, so the ledger charges the measured count of them.

    The slab is (view batch, pixels, width) floats, with width the wider of
    the detector rows and the slice band the call was handed -- the two axes
    the body sweeps.  The view batch stays the driver's own choice.

    The two directions are handed different slice extents.  The forward is
    handed transferred cylinders, which span the whole slice axis at every
    device count, so its slab does not shrink as devices are added.  The back
    is handed one owner's slice band, so a tall volume's back slab does shrink
    with the device count.
    """
    rows, channels, slices = 32, 32, 32
    p_sub = math.ceil(800 / 4)

    def declared_charge(direction, num_pixels, band_cols):
        return 8, 1024                       # the driver's batch and nominal

    plan_kwargs = dict(view_charge=declared_charge, num_pixels_full=800,
                       num_rows=rows, num_channels=channels,
                       recon=(32, 32, slices))
    declared = estimate_peak_device_bytes(make_plan(**plan_kwargs))
    torch_body = estimate_peak_device_bytes(make_plan(
        torch_body_directions=('forward', 'back'), **plan_kwargs))

    # One device: the band is the whole slice axis, so width is max(32, 32).
    width = max(rows, slices)
    back = dict(_named(torch_body, 'back projection').terms)['back batch'][0]
    assert back == SLABS * 8 * p_sub * width * 4
    assert dict(_named(declared, 'back projection').terms)['back batch'][0] \
        == 8 * 1024
    forward = dict(_named(torch_body, 'initial forward projection')
                   .terms)['forward batch'][0]
    assert forward == SLABS * 8 * 800 * width * 4
    # Every other term is untouched, so the peak moves only by the charge.
    assert torch_body.peak_bytes(0) > declared.peak_bytes(0)

    def charge(direction, num_pixels, band_cols):
        return 1, 1

    def ledger_for(rows, slices, n_devices):
        return estimate_peak_device_bytes(make_plan(
            n_devices=n_devices, view_charge=charge,
            torch_body_directions=('forward', 'back'),
            num_pixels_full=800, num_rows=rows, recon=(32, 32, slices)))

    def forward_batch(rows, slices, n_devices):
        return dict(_named(ledger_for(rows, slices, n_devices),
                           'initial forward projection')
                    .terms)['forward batch'][0]

    def back_batch(rows, slices, n_devices):
        ledger = ledger_for(rows, slices, n_devices)
        phase = _sub(ledger, 'direct recon (back loop)', n_devices,
                     'back workers')
        return dict(phase.terms)['back batch'][0]

    # Tall volume, narrow detector.  The forward keeps all 64 slices at four
    # devices, because the cylinders it transfers span them.
    assert forward_batch(32, 64, 1) == SLABS * 800 * 64 * 4
    assert forward_batch(32, 64, 4) == SLABS * 800 * 64 * 4
    # The back is handed the 16-slice shard there, which is below the 32 rows,
    # so the rows set its slab instead.
    assert back_batch(32, 64, 1) == SLABS * 800 * 64 * 4
    assert back_batch(32, 64, 4) == SLABS * 800 * 32 * 4
    # Wide detector: the rows set the slab both ways at every device count.
    assert forward_batch(128, 32, 1) == SLABS * 800 * 128 * 4
    assert forward_batch(128, 32, 4) == SLABS * 800 * 128 * 4
    assert back_batch(128, 32, 4) == SLABS * 800 * 128 * 4


# One row per measured arm: (sinogram shape, recon shape, masked pixel count,
# per-device measured peak bytes).  Every row is the two geometries with no
# hand-written kernels, weighted, from a supplied sinogram with no initial
# volume.  One builder prices them all, _measured_arm_ledger below.
#
# The MULTI-DEVICE rows were measured 2026-08-17 on four H100s of node h014,
# in the comparison job that ran beside slurm 15307729.  They are that job's
# composed-reconstruction arms at pixel batch 8192, the shipped default when
# the runs were taken (the default moved to 32768 on 2026-08-17) (rows file
# mg18_ab_h014_20260816_231137.jsonl in the plans repository).  They took the
# place of multi-device rows measured 2026-08-10 (job mg8), which were taken
# with the forward walking SLICE BANDS.  The banded forward was removed on
# 2026-08-17, so those older peaks can no longer be priced: the ledger has no
# banded forward left to charge.
#
# The SINGLE-DEVICE rows keep their mg8 measurements, unchanged.  They
# survived the removal because one device runs neither multi-device driver --
# the trivial path uses the plain projectors -- so the code they were measured
# against is the code that runs today.
#
# MEASURED_ARM_PIXEL_BATCH is the batch the runs used, and
# _measured_arm_ledger sets it on the plan.  Without it the plan would price
# forward calls at the whole pass and judge these peaks against a call shape
# the runs never made.
MEASURED_ARMS = {
    'ma1024_n1': ((1024, 1008, 992), (992, 992, 1148), 771240,
                  [37310451712]),
    'ma1024_n2': ((1024, 1008, 992), (992, 992, 1148), 771240,
                  [21420991488, 19492240896]),
    'ma1024_n4': ((1024, 1008, 992), (992, 992, 1148), 771240,
                  [29052623872, 28982226944, 28982226944, 28977516544]),
    'ma512_n1': ((512, 448, 384), (384, 384, 510), 115164,
                 [12253271552]),
    'tct2k_n1': ((256, 1900, 3000), (118, 360, 240), 42480,
                 [29262431744]),
    'tct2k_n2': ((256, 1900, 3000), (118, 360, 240), 42480,
                 [14972986368, 14962369024]),
    'tct2k_n4': ((256, 1900, 3000), (118, 360, 240), 42480,
                 [20367477760, 20363603968, 20363603968, 20363433984]),
    'tct1k_n1': ((256, 950, 1500), (59, 180, 120), 10620,
                 [8227791872]),
}
MEASURED_ARM_PIXEL_BATCH = 8192
# The granularity list those runs used, which is the library default.
MEASURED_GRANULARITY = (1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128)
MEASURED_VISITED = (4, 16, 64)


def _measured_view_charge(sinogram_shape, recon_shape, n_devices):
    """The view batch and nominal slab the DRIVER chose in those runs.

    Written out here rather than taken from a live model because the batch
    depends on the transient budget, and that budget is scaled by the
    per-device sinogram on CUDA and flat on CPU -- these tests run on CPU, so
    a CPU model would choose a different batch than the measured runs did and
    the comparison would be against the wrong arithmetic.
    """
    from mbirtorch.projectors import Projectors
    views, rows, channels = sinogram_shape
    cols = max(int(recon_shape[2]), int(rows))
    local_views = -(-int(views) // int(n_devices))
    budget = max(Projectors.VIEW_BATCH_TRANSIENT_FLOOR_BYTES,
                 min(Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES,
                     Projectors.VIEW_BATCH_SINO_MULTIPLE
                     * local_views * rows * channels * 4))

    def charge(direction, num_pixels, band_cols):
        bytes_per_view = int(num_pixels) * cols * 4
        return (max(1, min(Projectors.VIEW_BATCH_BODY_DEFAULT,
                           budget // max(1, bytes_per_view))),
                bytes_per_view)
    return charge


def _measured_arm_ledger(arm):
    sinogram_shape, recon_shape, num_pixels, measured = MEASURED_ARMS[arm]
    n_devices = len(measured)
    devices = ['cpu'] * n_devices
    sino = _sharding.Placement(devices, axis=0, axis_len=sinogram_shape[0])
    recon = _sharding.Placement(devices, axis=-1, axis_len=recon_shape[2])
    plan = LedgerPlan(
        sinogram_shape=sinogram_shape,
        recon_shape=recon_shape,
        devices=devices,
        view_blocks=[e - s for _d, (s, e) in sino.shard_ranges()],
        slice_blocks=[e - s for _d, (s, e) in recon.shard_ranges()],
        sino_rows=sinogram_shape[1],
        rows_track_slices=False,
        num_pixels_full=num_pixels,
        num_pixels_grid=recon_shape[0] * recon_shape[1],
        granularities=MEASURED_VISITED,
        partition_granularities=MEASURED_GRANULARITY,
        weights_supplied=True,
        # The translation arms carry no cylindrical mask, so their masked set
        # IS the whole grid and their hessian back-projects the grid directly.
        hessian_masked=num_pixels < recon_shape[0] * recon_shape[1],
        pixel_batch=MEASURED_ARM_PIXEL_BATCH,
        view_charge=_measured_view_charge(sinogram_shape, recon_shape,
                                          n_devices),
        torch_body_directions=('forward', 'back'))
    return estimate_peak_device_bytes(plan), measured


def test_the_torch_body_ledger_covers_every_measured_peak():
    """The floor, on the runs the slab count was calibrated from.

    A modeled peak below the measured one lets a doomed reconstruction start
    and die inside the allocator, which is the failure this module exists to
    prevent.  Every device of every measured arm must sit at or above 1.00.

    The other side of the floor: an over-charge spreads a reconstruction over
    more devices than it needs, so the band is asserted too.  The band is
    wider than CALIBRATION_BAND, and the module records the two measurements
    that set its width.  The widest over-charge among these arms is about
    4.2x, on tct1k_n1, against a bound of 5.8x.  The bound is not narrowed to
    match, because it was set by a measurement rather than by this set of
    arms.
    """
    low, high = _memory_ledger.TORCH_BODY_CALIBRATION_BAND
    worst = 0.0
    for arm in sorted(MEASURED_ARMS):
        ledger, measured = _measured_arm_ledger(arm)
        for i, peak in enumerate(measured):
            assert ledger.peak_bytes(i) >= peak, (
                f'{arm} device {i}: modeled {ledger.peak_bytes(i)} < '
                f'measured {peak}')
            worst = max(worst, ledger.peak_bytes(i) / peak)
    assert low <= worst <= high


# ── the forward's cylinder transfer ──────────────────────────────────────────
def test_the_cylinder_transfer_charges_its_cylinders():
    """The transfer leaves cylinders resident on every view-owner: one pixel
    batch wide and the whole slice axis tall.  Both forward phases carry them.

    Both GEOMETRIES are priced by the same arithmetic: what a transfer holds
    is set by the shape it assembles -- one pixel batch by the whole
    device-form slice axis -- and not by whether the geometry's detector rows
    track its slices.

    Several such batches are charged, not one: the driver transfers one batch
    ahead of the projection that reads it, so the widest instant holds the
    cylinder about to be projected, the pieces arriving for the batch after
    it, and their concatenation.
    """
    slices, batch = 32, 100                  # make_plan's slice axis
    residents = _memory_ledger.CYLINDER_TRANSFER_RESIDENTS
    for aligned in (False, True):
        transferred = estimate_peak_device_bytes(
            make_plan(n_devices=2, rows_track_slices=aligned,
                      pixel_batch=batch))
        for fragment in ('initial forward projection',
                         'subset delta forward projection'):
            terms = dict(_named(transferred, fragment).terms)
            assert (terms['transferred cylinders']
                    == [residents * batch * slices * 4] * 2), (aligned,
                                                               fragment)

    # A batch wider than the pixel set transfers the pixel set: the charge
    # follows what one call is actually handed, which is what keeps the term
    # honest at the small end without a separate rule.  Such a pass runs as a
    # single batch and so transfers nothing ahead, holding two cylinders where
    # the charge is three.  That over-charge is deliberate: the ledger's one
    # hard rule is that it may never charge less than a run needs.
    capped = estimate_peak_device_bytes(
        make_plan(n_devices=2, pixel_batch=10 ** 6))
    assert dict(_named(capped, 'initial forward projection')
                .terms)['transferred cylinders'] \
        == [residents * 800 * slices * 4] * 2

    # The term is the batch by the WHOLE slice axis on every view-owner, so
    # adding devices does not change it.
    charges = []
    for n in (2, 4):
        transferred = estimate_peak_device_bytes(
            make_plan(n_devices=n, pixel_batch=batch))
        charges.append(dict(_named(transferred, 'initial forward projection')
                            .terms)['transferred cylinders'][0])
    assert charges[0] == charges[1]
    # A single device never transfers: it holds the whole volume already.
    one = estimate_peak_device_bytes(
        make_plan(n_devices=1, pixel_batch=batch))
    assert dict(_named(one, 'initial forward projection')
                .terms)['transferred cylinders'] == [0]


def test_the_cylinder_transfer_prices_the_call_it_actually_makes():
    """The two numbers that set the forward's call shape.  One call is handed
    the WHOLE device-form slice axis, and one pixel batch rather than every
    pixel of the pass, so the per-view cost model must be asked those two."""
    asked = []

    def charge(direction, num_pixels, band_cols):
        asked.append((direction, num_pixels, band_cols))
        return 4, 1024

    batch = 100
    estimate_peak_device_bytes(
        make_plan(n_devices=2, view_charge=charge, pixel_batch=batch,
                  rows_track_slices=True))
    transferred = [(p, c) for d, p, c in asked if d == 'forward']
    assert {c for _p, c in transferred} == {32}     # the whole slice axis
    assert max(p for p, _c in transferred) == batch
    # The BACK call is handed a slice band instead, which is one shard here,
    # and every pixel of its pass rather than a cylinder batch.
    back = [(p, c) for d, p, c in asked if d == 'back']
    assert {c for _p, c in back} == {16}            # one slice shard of 32
    assert max(p for p, _c in back) > batch


# ── helpers ──────────────────────────────────────────────────────────────────
def _named(ledger, fragment):
    for phase in ledger.phases:
        if fragment in phase.name:
            return phase
    raise AssertionError(f'no phase matching {fragment!r} in '
                         f'{[p.name for p in ledger.phases]}')


def _sub(ledger, parent, n, step):
    """One sub-step of a split back-projection phase.

    At n == 1 there is no reduce, so the phase is emitted whole under the
    parent name and that is what comes back.
    """
    if n == 1:
        return _named(ledger, parent)
    for phase in ledger.phases:
        if parent in phase.name and f'[{step}]' in phase.name:
            return phase
    raise AssertionError(f'no {step!r} sub-phase of {parent!r} in '
                         f'{[p.name for p in ledger.phases]}')


def _has(ledger, fragment):
    return any(fragment in p.name for p in ledger.phases)


def _fixed_ledger(peak):
    return Ledger(devices=['cuda:0'],
                  phases=[PhaseCharge('synthetic', [peak], [('all', [peak])])])
