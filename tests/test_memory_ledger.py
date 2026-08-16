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
from mbirtorch._memory_ledger import (Ledger, LedgerPlan, MemoryPreflightError,
                                      PhaseCharge, estimate_peak_device_bytes)

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
def test_persistent_set_is_the_four_arrays_and_scales_with_shape():
    # Doubling the slice count doubles every recon-shaped term and leaves the
    # sinogram-shaped ones alone, so the peak grows by less than 2x.
    small = estimate_peak_device_bytes(make_plan(recon=(32, 32, 32)))
    large = estimate_peak_device_bytes(make_plan(recon=(32, 32, 64)))
    assert large.peak_bytes(0) > small.peak_bytes(0)
    assert large.peak_bytes(0) < 2 * small.peak_bytes(0)


def test_weights_are_charged_once_not_twice():
    """The hessian's weight array is an ALIAS of supplied weights.

    Charging both would over-count a full sinogram on every weighted run,
    which is the common case.  With weights supplied and the hessian computed
    internally, the peak must not exceed the unweighted case by more than one
    sinogram-shaped array on any phase that holds both.
    """
    sino_bytes = 64 * 32 * 32 * 4
    unweighted = estimate_peak_device_bytes(make_plan(weights_supplied=False))
    weighted = estimate_peak_device_bytes(make_plan(weights_supplied=True))
    # The hessian phase holds the weights either way (as the ones array, or as
    # the caller's array), so it must read identically.
    hess_un = _named(unweighted, 'hessian diagonal').per_device[0]
    hess_w = _named(weighted, 'hessian diagonal').per_device[0]
    assert hess_un == hess_w
    # The subset back projection materializes a weighted product only when
    # weights are supplied: exactly one sinogram-shaped array more.
    back_un = _named(unweighted, 'back projection').per_device[0]
    back_w = _named(weighted, 'back projection').per_device[0]
    assert back_w - back_un == sino_bytes


def test_supplied_hessian_and_init_recon_drop_their_phases():
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


def test_resume_drops_the_initialization_phases_entirely():
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


def test_hessian_phase_uses_the_unmasked_grid_count():
    """The one phase charged at the full grid rather than the ROR-masked set."""
    masked = make_plan(num_pixels_full=800, recon=(64, 64, 32))
    ledger = estimate_peak_device_bytes(masked)
    hessian = _named(ledger, 'hessian diagonal')
    grid_term = dict(hessian.terms)['back output'][0]
    # 3 x cyl(P_grid, slices), with P_grid = 64*64 and not the 800 masked.
    assert grid_term == 3 * (64 * 64) * 32 * 4


def test_prior_cylinder_count_drives_the_prior_phase():
    nine = estimate_peak_device_bytes(make_plan(qggmrf_cylinders=9))
    sixteen = estimate_peak_device_bytes(make_plan(qggmrf_cylinders=16))
    prior_9 = dict(_named(nine, 'prior').terms)['prior cylinders'][0]
    prior_16 = dict(_named(sixteen, 'prior').terms)['prior cylinders'][0]
    assert prior_16 == pytest.approx(prior_9 * 16 / 9)


def test_coarsest_granularity_dominates_the_subset_phases():
    """P_g = ceil(P_full / g), so the coarsest granularity holds the most."""
    ledger = estimate_peak_device_bytes(make_plan(granularities=(4, 128)))
    coarse = _named(ledger, 'prior (granularity 4)').per_device[0]
    fine = _named(ledger, 'prior (granularity 128)').per_device[0]
    assert coarse > fine
    assert ledger.peak_bytes(0) >= coarse


# ── the multi-device terms ───────────────────────────────────────────────────
def test_persistent_set_shrinks_with_the_device_count():
    one = estimate_peak_device_bytes(make_plan(n_devices=1))
    four = estimate_peak_device_bytes(make_plan(n_devices=4))
    persistent_1 = dict(_named(one, 'prior').terms)['error sinogram'][0]
    persistent_4 = dict(_named(four, 'prior').terms)['error sinogram'][0]
    assert persistent_4 == persistent_1 // 4


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
    # there is one more of them at four devices.
    assert 0.5 <= four / two <= 0.56
    # And it is well under what the old materialize-then-sum form charged:
    # n + 1 bands at two devices, n + 2 at four.
    assert two < 0.8 * 3 * pixels * (slices // 2) * 4
    assert four < 0.4 * 6 * pixels * (slices // 4) * 4
    assert reduce_bytes(1) == 0       # a single device never runs the reduce


def test_band_reduce_charges_the_bands_already_reduced_this_pass():
    """A band smaller than the shard means several reduces per owner, and the
    owner holds the ones it has finished until it concatenates them.

    The old charge counted only the band in flight, so it fell toward zero as
    the band narrowed while the owner really was holding most of a shard.
    The ``shard + band`` form covers both: the bands already done, at most
    ``shard - band``, and the two live ones.
    """
    plan = make_plan(n_devices=2, back_band=4)
    ledger = estimate_peak_device_bytes(plan)
    charged = dict(_sub(ledger, 'subset back projection', 2,
                        'band reduce').terms)['band reduce'][0]
    p_sub, shard, band = math.ceil(800 / 4), 16, 4
    slab = _sharding.reduce_slab_rows(p_sub, band * 4) * band * 4
    assert charged == p_sub * (shard + band) * 4 + slab

    # The floor rule in the place it bites: however narrow the band, the owner
    # still ends the pass holding a whole shard, so the charge may not fall
    # under one cylinder-shard.  The old form did, which is what this
    # replaces: it charged only the band in flight.
    def charge(band_length):
        led = estimate_peak_device_bytes(make_plan(n_devices=2,
                                                   back_band=band_length))
        return dict(_sub(led, 'subset back projection', 2,
                         'band reduce').terms)['band reduce'][0]

    one_shard = p_sub * shard * 4
    for band_length in (1, 2, 4, 8, 16):
        assert charge(band_length) > one_shard, band_length
    # And it still falls as the band narrows, so the knob remains a lever.
    assert charge(1) < charge(4) < charge(16)


def test_empty_shard_extensions_skip_their_role_terms():
    """A device with no real views does no projection; one with no real slices
    holds no band.  The ledger charges each role only where it exists."""
    # 3 views over 4 devices: device 3 owns no real view.
    plan = make_plan(n_devices=4, num_views=3, recon=(32, 32, 32))
    ledger = estimate_peak_device_bytes(plan)
    back = _named(ledger, 'back projection')
    batch = dict(back.terms)['back batch']
    assert batch[3] == 0
    # And with 3 slices over 4 devices, device 3 owns no real slice.
    plan = make_plan(n_devices=4, num_views=64, recon=(32, 32, 3))
    ledger = estimate_peak_device_bytes(plan)
    reduce_phase = _sub(ledger, 'subset back projection', 4, 'band reduce')
    assert dict(reduce_phase.terms)['band reduce'][3] == 0
    # The band it finished is charged only where a band lands, too.
    workers = _sub(ledger, 'subset back projection', 4, 'back workers')
    assert dict(workers.terms)['finished own band'][3] == 0


def test_partitions_are_charged_to_the_lead_device_only():
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    partitions = dict(_named(ledger, 'prior').terms)['partitions (lead device)']
    assert partitions[0] > 0
    assert partitions[1] == 0


def test_slice_band_knob_reduces_the_band_reduce():
    """The remedy the error message names first must actually work."""
    wide = estimate_peak_device_bytes(make_plan(n_devices=2))
    narrow = estimate_peak_device_bytes(make_plan(n_devices=2, back_band=4))
    wide_reduce = dict(_sub(wide, 'subset back projection', 2,
                            'band reduce').terms)['band reduce'][0]
    narrow_reduce = dict(_sub(narrow, 'subset back projection', 2,
                              'band reduce').terms)['band reduce'][0]
    assert narrow_reduce < wide_reduce


def test_back_projection_holds_three_cylinders_on_one_device():
    """The driver evaluates `block = back_body(...)` before rebinding the
    name, so the previous block is still alive while the next is produced:
    accumulator, outgoing block, incoming block."""
    ledger = estimate_peak_device_bytes(make_plan(n_devices=1))
    back = dict(_named(ledger, 'subset back projection').terms)['back output'][0]
    p_sub = math.ceil(800 / 4)
    assert back == 3 * p_sub * 32 * 4
    # A multi-device slice-owner instead accumulates band parts and
    # concatenates them, which is two.
    shared = estimate_peak_device_bytes(make_plan(n_devices=2))
    back2 = dict(_named(shared, 'subset back projection').terms)['back output'][0]
    assert back2 == 2 * p_sub * 16 * 4


def test_weights_are_live_in_every_pre_loop_phase_when_supplied():
    """Supplied weights are placed at the top of vcd_recon, so they are
    resident through the direct recon and the initial error state."""
    sino_bytes = 64 * 32 * 32 * 4
    ledger = estimate_peak_device_bytes(make_plan(weights_supplied=True))
    for fragment in ('direct recon', 'initial forward projection',
                     'error sinogram formation'):
        terms = dict(_named(ledger, fragment).terms)
        assert terms['weights'][0] == sino_bytes, fragment


def test_unweighted_run_has_no_weights_array_before_the_hessian():
    """The all-ones array is built INSIDE the hessian block, so nothing
    weights-shaped is resident before it."""
    sino_bytes = 64 * 32 * 32 * 4
    ledger = estimate_peak_device_bytes(make_plan(weights_supplied=False))
    for fragment in ('direct recon', 'initial forward projection',
                     'error sinogram formation'):
        assert dict(_named(ledger, fragment).terms)['weights'][0] == 0, fragment
    # From the hessian onward it exists.
    assert dict(_named(ledger, 'hessian diagonal').terms)[
        'hessian weights'][0] == sino_bytes
    assert dict(_named(ledger, 'subset prior').terms)['weights'][0] == sino_bytes


def test_weighted_projection_is_released_before_the_error_assignment():
    """`weighted_fwd` lives only across its two dot products.

    It is charged in the dot-product sub-phase and NOT in the error sinogram
    assignment, which is what the release in `_initial_error_state` buys.
    """
    one = estimate_peak_device_bytes(make_plan(n_devices=1, weights_supplied=True))
    assert dict(_named(one, 'initial dot products').terms)[
        'weighted forward projection'][0] > 0
    assert 'weighted forward projection' not in dict(
        _named(one, 'error sinogram formation').terms)
    # The sharded branch never had the array, because it fuses the weights
    # into per-shard dots whose locals die on worker return.
    two = estimate_peak_device_bytes(make_plan(n_devices=2, weights_supplied=True))
    assert dict(_named(two, 'initial dot products').terms)[
        'weighted forward projection'][0] == 0
    # Constant weights materialize no product at all.
    plain = estimate_peak_device_bytes(make_plan(n_devices=1))
    assert dict(_named(plain, 'initial dot products').terms)[
        'weighted forward projection'][0] == 0


def test_error_formation_holds_the_scaled_projection_and_its_result():
    """`error = sinogram - alpha * fwd` allocates the scaled projection and
    then the difference, so both are live at the assignment."""
    ledger = estimate_peak_device_bytes(make_plan())
    terms = dict(_named(ledger, 'error sinogram formation').terms)
    sino_bytes = 64 * 32 * 32 * 4
    assert terms['alpha-scaled projection'][0] == sino_bytes
    assert terms['error sinogram'][0] == sino_bytes
    assert terms['forward projection'][0] == sino_bytes


def test_direct_recon_loop_and_scatter_are_not_co_live():
    """The back accumulator feeds the scatter, so the two are consecutive
    sub-peaks rather than one sum.

    Both are emitted and the per-device maximum picks between them.  Picking
    one whole sub-phase by a cross-device total would under-charge a device
    where the other is larger, so the selection must stay per device.
    """
    ledger = estimate_peak_device_bytes(make_plan())
    loop = _named(ledger, 'direct recon (back loop)')
    scatter = _named(ledger, 'direct recon (scatter)')
    assert 'scatter buffer' not in dict(loop.terms)
    assert 'back output' not in dict(scatter.terms)
    # The contribution to the peak is the max of the two, never their sum.
    combined = max(loop.per_device[0], scatter.per_device[0])
    assert combined < loop.per_device[0] + scatter.per_device[0]
    assert ledger.peak_bytes(0) >= combined


def test_sub_phase_selection_is_per_device():
    """A two-device layout whose sub-peaks rank differently per device must
    charge each device its own larger sub-peak."""
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    for i in (0, 1):
        loop = _named(ledger, 'direct recon (back loop)').per_device[i]
        scatter = _named(ledger, 'direct recon (scatter)').per_device[i]
        assert ledger.peak_bytes(i) >= max(loop, scatter)


def test_per_iteration_statistics_are_charged():
    """Charged as zero by the first ledger, and measured as the peak of an
    unweighted run once the residency fixes shrank the other phases.

    The squared-error transient is two sinogram-shaped products.
    """
    sino_bytes = 64 * 32 * 32 * 4
    ledger = estimate_peak_device_bytes(make_plan())
    stats = _named(ledger, 'per-iteration statistics (squared error)')
    assert dict(stats.terms)['squared-error products'][0] == 2 * sino_bytes
    # It carries the persistent set, like every other in-loop phase.
    assert dict(stats.terms)['error sinogram'][0] == sino_bytes
    assert dict(stats.terms)['flat recon'][0] > 0


def test_the_recon_ell_1_is_charged_beside_the_squared_error_products():
    """The recon L1 does allocate, and is charged as its own sub-phase.

    This phase used to charge the squared-error products alone, on the reading
    that the L1 "fuses into its own reduction and materializes nothing".  It
    does not: ``sum(abs(flat_recon))`` allocated a whole second recon.  The
    sizes that were measured simply had two sinograms larger, so the miss did
    not show.  The two are consecutive rather than co-live, so both are
    emitted and the per-device maximum picks.
    """
    ledger = estimate_peak_device_bytes(make_plan())
    ell1 = _named(ledger, 'per-iteration statistics (recon ell-1)')
    squared = _named(ledger, 'per-iteration statistics (squared error)')
    charged = dict(ell1.terms)['recon ell-1 chunk'][0]
    assert charged > 0
    assert 'squared-error products' not in dict(ell1.terms)
    assert 'recon ell-1 chunk' not in dict(squared.terms)
    # Both carry the persistent set, as every in-loop phase does.
    assert dict(ell1.terms)['error sinogram'][0] > 0


def test_the_recon_ell_1_chunk_stops_following_the_recon():
    """A recon far larger than two sinograms is the geometry the old charge
    missed.  image_ell1 bounds the transient to a chunk, so the charge no
    longer scales with the recon."""
    target = _memory_ledger.ELL1_CHUNK_BYTES
    # Few views, big recon: two sinograms are much smaller than one recon.
    plan = make_plan(recon=(512, 512, 512), num_views=8,
                     num_rows=64, num_channels=64,
                     num_pixels_full=512 * 512)
    ledger = estimate_peak_device_bytes(plan)
    terms = dict(_named(ledger,
                        'per-iteration statistics (recon ell-1)').terms)
    squared = dict(_named(ledger,
                          'per-iteration statistics (squared error)').terms)
    recon_bytes = terms['flat recon'][0]
    chunk = terms['recon ell-1 chunk'][0]
    # The geometry the old charge missed: one recon dwarfs two sinograms, so
    # charging the sinogram products alone would have set the phase far too low.
    assert recon_bytes > 100 * squared['squared-error products'][0]
    # And the chunk does not follow the recon.
    assert chunk <= recon_bytes / 16
    assert 0.5 * target <= chunk <= 2 * target


# ── the back projection's two sub-steps ──────────────────────────────────────
SPLIT_PARENTS = ('direct recon (back loop)', 'hessian diagonal',
                 'subset back projection')


def test_every_back_phase_splits_into_workers_and_reduce_at_n_above_one():
    """The workers project and the reduce gathers, consecutively.

    Charging their sum priced a peak that is never live.  Both are emitted
    and the per-device maximum over phases picks between them, exactly as the
    direct recon's loop/scatter split does.
    """
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    names = [p.name for p in ledger.phases]
    for parent in SPLIT_PARENTS:
        workers = _sub(ledger, parent, 2, 'back workers')
        reduce_phase = _sub(ledger, parent, 2, 'band reduce')
        # The phase's contribution is the MAX of the two, never their sum.
        for i in (0, 1):
            both = max(workers.per_device[i], reduce_phase.per_device[i])
            assert both < workers.per_device[i] + reduce_phase.per_device[i]
            assert ledger.peak_bytes(i) >= both
        # And the unsplit parent name is gone -- nothing charges the sum.
        assert parent not in names


def test_the_split_sub_phases_keep_the_parent_name_visible():
    """The compatibility surface.

    Every consumer that matches a phase by its parent name -- the preflight
    message, a calibration row, these tests -- must still find it, so the
    sub-step is a SUFFIX and never a rewrite.
    """
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    for parent in SPLIT_PARENTS:
        for step in ('back workers', 'band reduce'):
            name = _sub(ledger, parent, 2, step).name
            assert name.startswith(parent), name
            assert name.endswith(f'[{step}]'), name
    # The message the user reads still names the parent it came from.
    _fits, rows = _memory_ledger.layout_fits(ledger, [1024, 1024])
    message = _memory_ledger.format_shortfall(ledger, rows,
                                              num_devices_tried=2)
    assert ledger.dominant_phase(0).name in message


def test_each_sub_phase_charges_only_its_own_terms():
    """The workers hold blocks, the band they already finished, and the
    batch; the reduce holds its gather and nothing of the loop."""
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    workers = dict(_sub(ledger, 'direct recon (back loop)', 2,
                        'back workers').terms)
    reduce_phase = dict(_sub(ledger, 'direct recon (back loop)', 2,
                             'band reduce').terms)
    for term in ('back output', 'finished own band', 'back batch'):
        assert term in workers and term not in reduce_phase, term
    assert 'band reduce' in reduce_phase and 'band reduce' not in workers
    # Both carry the same residents, which is why neither can be dropped.
    for term in ('sinogram', 'filtered sinogram', 'library workspace'):
        assert workers[term] == reduce_phase[term], term


def test_a_single_device_phase_is_not_split_and_does_not_move():
    """n == 1 has no reduce to split off, so the phase stays whole under the
    parent name and every single-device charge reads exactly as before."""
    ledger = estimate_peak_device_bytes(make_plan(n_devices=1))
    names = [p.name for p in ledger.phases]
    assert 'direct recon (back loop)' in names
    assert not any('[back workers]' in n or '[band reduce]' in n for n in names)
    terms = dict(_named(ledger, 'direct recon (back loop)').terms)
    assert terms['band reduce'][0] == 0          # never runs at one device
    assert terms['finished own band'][0] == 0    # nor does the band handoff
    assert terms['back output'][0] == 3 * 800 * 32 * 4   # still three


def test_the_worker_block_count_follows_the_realized_view_batches():
    """The calibrated term.

    The view loop releases each block after accumulating it, so it holds the
    accumulator plus the incoming block: min(2, view_batches).  A plan with
    no cost model cannot count batches and charges the ceiling of two.
    """
    p_sub = math.ceil(800 / 4)

    def blocks(batches_per_device):
        # A charge that yields exactly `batches_per_device` batches over the
        # 32 views each of two devices owns.
        def charge(direction, num_pixels, band_cols):
            return max(1, 32 // batches_per_device), 1
        ledger = estimate_peak_device_bytes(
            make_plan(n_devices=2, view_charge=charge))
        terms = dict(_sub(ledger, 'subset back projection', 2,
                          'back workers').terms)
        return terms['back output'][0] / (p_sub * 16 * 4)

    assert blocks(1) == 1                        # one batch, one block
    assert blocks(2) == 2
    assert blocks(8) == 2                        # capped at the accumulator + 1
    # No cost model at all: the ceiling, which is what the docstring promises.
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    assert dict(_sub(ledger, 'subset back projection', 2,
                     'back workers').terms)['back output'][0] == 2 * p_sub * 16 * 4


def test_the_finished_own_band_is_one_cylinder_on_every_slice_owner():
    """Each owner keeps its reduced band for the rest of the loop, so from
    its own pass onward it carries one extra cylinder through every later
    pass.  Real and unavoidable, so the ledger has to charge it."""
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    terms = dict(_sub(ledger, 'direct recon (back loop)', 2,
                      'back workers').terms)
    assert terms['finished own band'] == [800 * 16 * 4] * 2


def test_the_forward_charges_the_broadcast_band_and_no_margin():
    """The broadcast band is named from the code: the forward copies the
    current slice-owner's band onto every view-owner and it stays live for
    that band's projection.  It does not exist at one device.

    The ledger used to carry a safety margin beside it -- one per-device
    sinogram shard on every forward phase -- standing in for arrays nobody
    had enumerated yet.  Those arrays are now charged directly and checked
    against measurement, so NO phase may carry a margin term: every term must
    name an array the code allocates.
    """
    two = estimate_peak_device_bytes(make_plan(n_devices=2))
    for fragment in ('initial forward projection',
                     'subset delta forward projection'):
        terms = dict(_named(two, fragment).terms)
        assert terms['broadcast band'][0] > 0, fragment
    one = estimate_peak_device_bytes(make_plan(n_devices=1))
    assert dict(_named(one, 'initial forward projection')
                .terms)['broadcast band'][0] == 0
    for ledger in (one, two):
        for phase in ledger.phases:
            for name, _vals in phase.terms:
                assert 'margin' not in name, (phase.name, name)


def test_the_forward_block_count_follows_the_realized_view_batches():
    """The forward counterpart of the worker block count, with one
    difference.

    The view-range loop has no release, so it holds the outgoing block and the
    incoming one: min(2, view_batches).  One of those is already inside the
    batch charge -- a forward body's output plane scales with the view batch,
    so its ``_view_batch_cost`` prices it per view -- so this term charges the
    remainder: nothing at a single batch, one block above that.
    """
    # make_plan defaults to a two-fan geometry: one whose detector rows are
    # not tied 1:1 to recon slices, as in cone beam.
    rows, channels = 32, 32

    def block(batches_per_device):
        # A charge that yields exactly `batches_per_device` batches over the
        # 32 views each of two devices owns.
        view_batch = max(1, 32 // batches_per_device)

        def charge(direction, num_pixels, band_cols):
            return view_batch, 1

        ledger = estimate_peak_device_bytes(
            make_plan(n_devices=2, view_charge=charge))
        terms = dict(_named(ledger, 'initial forward projection').terms)
        return terms['forward block'][0], view_batch

    assert block(1)[0] == 0                  # one batch, one block, all priced
    for batches in (2, 8):
        charged, view_batch = block(batches)
        assert charged == view_batch * rows * channels * 4
    # No cost model at all: the ceiling of two, one of them charged here,
    # which is what the docstring promises.
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    assert dict(_named(ledger, 'initial forward projection')
                .terms)['forward block'][0] == 1 * rows * channels * 4


def test_the_forward_block_spans_the_detector_rows_on_a_two_fan_geometry():
    """A two-fan body's output plane spans the FULL detector rows whatever
    slice band the values carry, so the block does not shrink with the band;
    a row-aligned body's does, rows tracking slices one for one.  Charging
    the two-fan block at the band instead under-charges the cone forward.
    """
    band = 32 // 2                            # one slice shard of make_plan's
    rows, channels = 32, 32
    for aligned, expected in ((True, band), (False, rows)):
        ledger = estimate_peak_device_bytes(
            make_plan(n_devices=2, rows_track_slices=aligned))
        terms = dict(_named(ledger, 'initial forward projection').terms)
        assert terms['forward block'][0] == expected * channels * 4, aligned
    # At one device there is no band to differ from: the row-aligned form
    # hands the body every slice it owns.
    one = estimate_peak_device_bytes(make_plan(n_devices=1,
                                               rows_track_slices=True))
    assert dict(_named(one, 'initial forward projection')
                .terms)['forward block'][0] == 32 * channels * 4


def test_the_split_lowers_the_multi_device_peak_and_leaves_n1_alone():
    """The whole point: summing two sub-steps that are never live together
    was the largest over-charge at n>1, and removing it leaves the
    single-device ledger untouched."""
    plan_kwargs = dict(recon=(64, 64, 32), num_pixels_full=3000)
    four = estimate_peak_device_bytes(make_plan(n_devices=4, **plan_kwargs))
    for parent in SPLIT_PARENTS:
        workers = _sub(four, parent, 4, 'back workers').per_device[0]
        reduce_phase = _sub(four, parent, 4, 'band reduce').per_device[0]
        # Neither sub-step alone reaches what the sum would have charged.
        summed = workers + reduce_phase
        assert max(workers, reduce_phase) < summed
        assert four.peak_bytes(0) < summed


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


def test_masked_hessian_leaves_the_public_method_unchanged():
    """`indices=None` must keep today's behavior bit for bit."""
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    a = model.compute_hessian_diagonal()
    b = model.compute_hessian_diagonal(indices=None)
    np.testing.assert_array_equal(a, b)


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
    masked call; the comparison run lets vcd_recon compute it at the masked
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
    control, _losses = model.vcd_recon(sinogram.copy(), partitions, sequence,
                                       0.0, weights=weights,
                                       fm_hessian=dense_hessian.clone())
    np.random.seed(7)
    masked, _losses = model.vcd_recon(sinogram.copy(), partitions, sequence,
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
def test_view_charge_enters_the_projection_phases():
    calls = []

    def charge(direction, num_pixels, band_cols):
        calls.append((direction, num_pixels, band_cols))
        return 8, 1024                      # 8 views at 1 KiB each

    with_charge = estimate_peak_device_bytes(make_plan(view_charge=charge))
    without = estimate_peak_device_bytes(make_plan())
    # The charge lands on the PROJECTION phases; the peak may sit on a
    # state-only phase, so assert where the charge actually goes.
    charged = _named(with_charge, 'back projection').per_device[0]
    plain = _named(without, 'back projection').per_device[0]
    assert charged > plain
    directions = {d for d, _p, _c in calls}
    assert directions == {'forward', 'back'}


def test_view_batch_charge_matches_the_driver_batch():
    """One cost model, two consumers: the number the ledger prices must be the
    number the driver would actually run."""
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


def test_view_batch_charge_prices_a_hypothetical_device_count():
    """The ledger must be able to price a layout the model is not in."""
    angles = np.linspace(0, np.pi, 512, endpoint=False)
    model = mbirtorch.ParallelBeamModel((512, 64, 64), angles)
    model.configure_devices(devices=['cpu'])
    fwd_body, _back = model._view_batch_bodies()
    args = model._view_batch_args()
    live = model.projector_functions.view_batch_charge(fwd_body, 400, 64, args)
    hypothetical = model.projector_functions.view_batch_charge(
        fwd_body, 400, 64, args, n_devices=4)
    assert model.sino_placement.n_devices == 1      # unchanged by the query
    assert live[1] == hypothetical[1]               # same per-view charge


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


def test_resident_credits_counts_only_matching_cuda_devices():
    cpu_tensor = torch.zeros(1024, dtype=torch.float32)
    credits = _memory_ledger.resident_credits(['cpu'], [cpu_tensor])
    assert credits == [0]                            # CPU is never credited
    assert _memory_ledger.resident_credits(['cpu'], [None]) == [0]


def test_shortfall_message_names_the_phase_and_the_remedies():
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    _fits, rows = _memory_ledger.layout_fits(ledger, [1024, 1024])
    message = _memory_ledger.format_shortfall(ledger, rows, num_devices_tried=2)
    assert 'dominant phase' in message
    assert ledger.dominant_phase(0).name in message
    assert 'back_project_slice_band' in message      # the band-reduce lever
    assert 'view_batch_size' in message
    assert 'skip_memory_preflight' in message
    assert 'shortfall' in message


def test_preflight_error_is_raisable_and_readable():
    ledger = estimate_peak_device_bytes(make_plan())
    _fits, rows = _memory_ledger.layout_fits(ledger, [1024])
    with pytest.raises(MemoryPreflightError, match='more memory'):
        raise MemoryPreflightError(
            _memory_ledger.format_shortfall(ledger, rows, 1))


def test_calibration_band_verdicts():
    rows = [('cuda:0', 12 * GB, 12 * GB, 1.00),
            ('cuda:1', 9 * GB, 10 * GB, 0.90),
            ('cuda:2', 20 * GB, 10 * GB, 2.00)]
    text = _memory_ledger.format_calibration(rows)
    assert 'UNDER' in text                           # 0.90 under-predicts
    assert 'over' in text
    assert ' ok ' in text or text.count('ok') >= 1


# ── the direct workload ──────────────────────────────────────────────────────
# A direct reconstruction is the filter and one back projection.  It is priced
# as its own plan because the policy checks capacity against the call in
# progress, while still choosing the device count for a full recon.
def test_the_direct_plan_is_the_filter_and_one_back_projection():
    """The phases a direct reconstruction really runs, and nothing else.

    The filter holds the placed sinogram, the copy it writes, and its own row
    batch; the back projection reads that copy and scatters into the volume.
    No prior, no hessian, no partitions and no loop exist while it runs.
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


def test_the_direct_plan_costs_less_than_the_full_one_at_the_same_shapes():
    """The whole point of the narrowed check.  A grid much larger than the
    masked set makes the hessian dominate the full plan, so the phases the
    direct plan drops have to move the peak and not merely the phase list."""
    shape = dict(recon=(64, 64, 32), num_pixels_full=800)
    full = estimate_peak_device_bytes(make_plan(**shape))
    direct = estimate_peak_device_bytes(make_plan(workload='direct', **shape))
    assert direct.peak_bytes(0) < full.peak_bytes(0)
    for gone in ('hessian', 'subset', 'per-iteration statistics',
                 'initial forward projection'):
        assert not _has(direct, gone)


def test_the_direct_plan_splits_its_back_projection_like_the_full_one():
    """Same charges, so the same two consecutive sub-steps: the workers
    project and the reduce gathers, and the reduce's co-residency is charged
    where a direct reconstruction really pays it."""
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2,
                                                  workload='direct'))
    workers = _sub(ledger, 'direct recon (back loop)', 2, 'back workers')
    reduce_phase = _sub(ledger, 'direct recon (back loop)', 2, 'band reduce')
    assert dict(reduce_phase.terms)['band reduce'][0] > 0
    for i in (0, 1):
        both = max(workers.per_device[i], reduce_phase.per_device[i])
        assert both < workers.per_device[i] + reduce_phase.per_device[i]
        assert ledger.peak_bytes(i) >= both


def test_the_filter_batch_follows_the_rows_it_actually_walks():
    """apply_row_filter walks ROW_FILTER_BATCH detector rows at a time, so the
    charge is capped by that batch and not by the shard: a shard with fewer
    rows than one batch is charged for the rows it has."""
    from mbirtorch.tomography_utils import ROW_FILTER_BATCH

    small = estimate_peak_device_bytes(make_plan(
        workload='direct', num_views=4, num_rows=8))       # 32 rows in all
    large = estimate_peak_device_bytes(make_plan(
        workload='direct', num_views=1024, num_rows=8))    # far past the batch
    per_row_small = dict(_named(small, 'filter').terms)['filter row batch'][0]
    per_row_large = dict(_named(large, 'filter').terms)['filter row batch'][0]
    assert per_row_large == per_row_small * ROW_FILTER_BATCH // 32


def test_a_recon_check_covers_a_direct_one_but_not_the_reverse():
    """The full plan charges everything the direct plan charges, so a layout
    checked for a recon needs no direct check.  Nothing else is claimed."""
    assert _memory_ledger.workload_covers('recon', 'direct')
    assert _memory_ledger.workload_covers('direct', 'direct')
    assert _memory_ledger.workload_covers('recon', 'recon')
    assert not _memory_ledger.workload_covers('direct', 'recon')
    assert not _memory_ledger.workload_covers(None, 'recon')


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
    image's ell-1 norm.  Nothing else exists while it runs.
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


def test_the_denoise_plan_charges_no_projector_or_hessian_term():
    """The three sets of charges a denoiser must not carry.

    Its ``create_projectors`` is a no-op, so no view batch, no projection
    block and no assembled projection output exists.  It builds no hessian
    diagonal and no weights array.  And it builds one partition rather than a
    sequence, so no term follows a granularity it never visits.
    """
    ledger = estimate_peak_device_bytes(make_denoise_plan(n_devices=2))
    charged = {name for phase in ledger.phases for name, _vals in phase.terms}
    for absent in ('batch', 'block', 'forward', 'back', 'hessian', 'weights',
                   'sinogram', 'scatter', 'band', 'partitions'):
        assert not any(absent in name for name in charged), (absent, charged)
    # And the view axis never enters: the same plan with every view on one
    # device reads identically, because only the slice split is used.
    lopsided = estimate_peak_device_bytes(
        make_denoise_plan(n_devices=2, num_views=1))
    assert lopsided.per_device_peaks() == ledger.per_device_peaks()


def test_the_denoise_working_set_follows_the_subset_size():
    """A coarser partition means a bigger subset, and a bigger subset means a
    bigger working set: every per-subset term is the subset's pixel count by
    the device's slices."""
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


def test_the_denoise_prior_is_charged_at_the_shared_qggmrf_count():
    """The denoiser calls the same prior kernel a reconstruction calls, so it
    is priced by the same cylinder count and not by one of its own."""
    p_sub = math.ceil(1024 / 16)
    for cylinders in (_memory_ledger.QGGMRF_CYLINDERS_COMPILED,
                      _memory_ledger.QGGMRF_CYLINDERS_EAGER):
        ledger = estimate_peak_device_bytes(
            make_denoise_plan(qggmrf_cylinders=cylinders))
        charged = dict(_named(ledger, 'denoise subset prior').terms)[
            'prior cylinders'][0]
        assert charged == cylinders * p_sub * 32 * 4


def test_the_denoise_charges_follow_the_slice_split():
    """The denoiser divides its image by SLICE, so every term is the device's
    own slice block and no term follows the view axis."""
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
    p_sub = math.ceil(1024 / 16)
    assert prior == [_memory_ledger.QGGMRF_CYLINDERS_COMPILED * p_sub * b * 4
                     for b in blocks]
    # And the peak falls with the device count, since every image-shaped term
    # is a share of the volume.
    one = estimate_peak_device_bytes(make_denoise_plan(image=(32, 32, 32)))
    four = estimate_peak_device_bytes(make_denoise_plan(image=(32, 32, 32),
                                                        n_devices=4))
    assert four.peak_bytes(0) < one.peak_bytes(0)


def test_the_denoise_halos_exist_only_on_a_sharded_sweep():
    """The sharded sweep stages one boundary column per shard side; a single
    device runs the compiled sweep and exchanges nothing.

    The column count is written out here rather than read from the module, so
    that changing the constant alone cannot move the charge without this test
    noticing.
    """
    grid = 32 * 32
    one = estimate_peak_device_bytes(make_denoise_plan())
    two = estimate_peak_device_bytes(make_denoise_plan(n_devices=2))
    assert dict(_named(one, 'denoise subset prior').terms)['qggmrf halos'] \
        == [0]
    assert dict(_named(two, 'denoise subset prior').terms)['qggmrf halos'] \
        == [4 * grid * 4] * 2
    # A halo is a column of the in-slice grid, so it does not shrink with the
    # device count the way the image-shaped terms do.
    four = estimate_peak_device_bytes(make_denoise_plan(n_devices=4))
    assert dict(_named(four, 'denoise subset prior').terms)['qggmrf halos'] \
        == [4 * grid * 4] * 4


def test_the_denoise_partition_is_held_whole_on_every_device():
    """The sharded sweep copies the whole partition onto each device, because
    a subset's indices address the in-slice grid that every shard shares.

    A reconstruction charges its partitions to the lead device alone, so this
    is the one place the two plans differ in WHERE a term lands rather than
    in how large it is.
    """
    ledger = estimate_peak_device_bytes(make_denoise_plan(n_devices=2))
    charged = dict(_named(ledger, 'denoise subset prior').terms)[
        'subset indices']
    assert charged == [16 * math.ceil(1024 / 16) * 8] * 2
    recon = estimate_peak_device_bytes(make_plan(n_devices=2))
    assert dict(_named(recon, 'prior').terms)['partitions (lead device)'][1] \
        == 0


def test_the_denoise_statistics_phase_holds_one_chunk_not_one_image():
    """The convergence test reduces the working image a chunk at a time, so at
    any size worth chunking it holds a chunk and not a fourth image.

    As ``sum(abs(working image))`` it held a whole image of absolute values and
    was the denoiser's widest instant.  Bounded to a chunk, the peak falls on
    the qGGMRF prior's working set instead, so this asserts which phase is the
    peak and not merely what the statistic costs.
    """
    ledger = estimate_peak_device_bytes(
        make_denoise_plan(image=(512, 512, 512)))
    stats = _named(ledger, 'denoise per-pass statistics')
    placement = _named(ledger, 'denoise state placement')
    charged = dict(stats.terms)['ell-1 chunk'][0]
    assert charged == _memory_ledger.ELL1_CHUNK_BYTES
    # The chunk is the whole of what this phase adds to the state.
    assert stats.per_device[0] - placement.per_device[0] == charged
    # The peak moved off this phase and onto the prior.
    prior = _named(ledger, 'denoise subset prior')
    assert ledger.peak_bytes(0) == prior.per_device[0]
    assert prior.per_device[0] > stats.per_device[0]


def test_the_denoise_statistics_chunk_stops_following_the_image():
    """The point of the chunking: the statistic's transient stays near one
    chunk as the image grows, where every other denoise term scales with it.
    """
    target = _memory_ledger.ELL1_CHUNK_BYTES
    small = estimate_peak_device_bytes(
        make_denoise_plan(image=(256, 256, 256)))
    big = estimate_peak_device_bytes(make_denoise_plan(image=(640, 640, 640)))

    def chunk(led):
        return dict(_named(led, 'per-pass statistics').terms)['ell-1 chunk'][0]

    # The image grows about fifteenfold over this pair; the chunk does not
    # move off the target, because the chunk COUNT absorbs the growth.
    assert 256 ** 3 * 15 < 640 ** 3
    for led in (small, big):
        assert 0.5 * target <= chunk(led) <= 2 * target
    assert chunk(big) < 1.1 * chunk(small)
    # The prior, which does scale, really does grow over the same pair.
    def prior(led):
        return dict(_named(led, 'denoise subset prior').terms)[
            'prior cylinders'][0]

    assert prior(big) > 15 * prior(small)


def test_the_denoise_statistics_reduce_a_small_image_whole():
    """Below one chunk the reduction runs unchunked, so the ledger charges the
    whole image -- the arithmetic and the cost the chunked form replaced.  An
    extra image is small in absolute terms at these sizes, which is why the
    unchunked case is left alone."""
    image_bytes = 32 * 32 * 32 * 4
    assert image_bytes < _memory_ledger.ELL1_CHUNK_BYTES
    ledger = estimate_peak_device_bytes(make_denoise_plan())
    stats = _named(ledger, 'denoise per-pass statistics')
    assert dict(stats.terms)['ell-1 chunk'][0] == image_bytes


def test_the_ledger_chunk_matches_what_the_reduction_really_allocates():
    """The anti-drift gate on the two constants.

    ``image_ell1`` and ``ell1_chunk_bytes`` share ELL1_CHUNK_BYTES, but they
    are still two pieces of arithmetic that could disagree.  This runs the
    real reduction and reads the largest array it actually allocates, so the
    charge is checked against the allocation rather than against a restatement
    of the same formula.  Both the denoise and the recon branches price their
    ell-1 with ell1_chunk_bytes, so this covers both.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    from mbirtorch._memory_ledger import image_ell1

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
        predicted, _n_chunks = _memory_ledger.ell1_chunk_bytes(image_bytes)
        assert seen.nbytes == predicted, (image_bytes, seen.nbytes, predicted)


def test_a_denoise_check_covers_nothing_but_another_denoise():
    """A denoise holds arrays neither of the other plans holds, so no check
    substitutes for it and it substitutes for none."""
    assert _memory_ledger.workload_covers('denoise', 'denoise')
    for other in ('recon', 'direct', None):
        assert not _memory_ledger.workload_covers(other, 'denoise')
        assert not _memory_ledger.workload_covers('denoise', other)


def test_plan_from_model_prices_a_denoiser_without_its_projectors():
    """The round trip: a live denoiser to a plan to a ledger.

    The projector reads are the reason this needs a branch of its own.  A
    denoiser's ``create_projectors`` is a no-op, so it defines no per-view
    projection bodies, and the recon plan's cost-model read raises on it --
    which is asserted here, so the branch cannot be quietly removed.
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
    assert plan.column_pixel_batch is None
    # The one fixed partition the denoiser builds.
    assert plan.granularities == (16,)
    assert plan.partition_granularities == (16,)
    ledger = estimate_peak_device_bytes(plan)
    assert [p.name for p in ledger.phases] == DENOISE_PHASES
    assert ledger.peak_bytes(0) > 0
    assert ledger.peak_bytes(0) == ledger.peak_bytes(1)   # an even slice split


def test_plan_from_model_visits_only_the_partition_the_denoiser_builds():
    """A denoise sweep builds ONE partition, the one the first entry of the
    sequence names, so the plan may not read the sequence the way a
    reconstruction does.

    A reconstruction walks the whole sequence and builds every granularity in
    the list.  Reading it that way here would charge partitions the denoiser
    never builds and emit subset phases it never runs.
    """
    denoiser = mbirtorch.QGGMRFDenoiser((16, 16, 12), compile_mode='off')
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0,
                        granularity=[8, 16, 32], partition_sequence=[1, 2])
    plan = _memory_ledger.plan_from_model(denoiser, ['cpu'],
                                          workload='denoise')
    assert plan.granularities == (16,)          # granularity[sequence[0]]
    assert plan.partition_granularities == (16,)
    ledger = estimate_peak_device_bytes(plan)
    assert [p.name for p in ledger.phases] == DENOISE_PHASES
    # The partition charge is that one partition, not the three the list names.
    assert dict(_named(ledger, 'denoise subset prior').terms)[
        'subset indices'][0] == 16 * math.ceil(16 * 16 / 16) * 8


def test_a_denoise_plan_needs_a_model_whose_shapes_agree():
    """The assumption the plan rests on, checked rather than assumed.

    Every term in the denoise plan is image-shaped.  A model whose sinogram
    shape differs from its image shape is not a denoiser, and pricing one
    with this plan would charge the wrong arrays, so it is refused by name.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    assert tuple(model.get_params('sinogram_shape')) != \
        tuple(model.get_params('recon_shape'))
    with pytest.raises(ValueError, match='sinogram_shape is its image shape'):
        _memory_ledger.plan_from_model(model, ['cpu'], workload='denoise')


# ── the model-facing plan ────────────────────────────────────────────────────
def test_plan_from_model_reads_the_current_params_and_a_candidate_layout():
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


def test_full_index_count_matches_the_index_array_and_caches():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    assert model.full_index_count() == model._full_indices().shape[0]
    assert model.full_index_count() == model.full_index_count()


def test_the_policy_builds_no_ledger_for_a_cpu_model():
    """The ledger's production job is choosing a CUDA device count, so a CPU
    or MPS model never consults one and never pays for it.

    The MATH is device-agnostic and can be built for any backend, which is
    what lets these tests run.  Refusing to consult it is the policy's
    decision, so that is where the contract is asserted.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    assert model._apply_device_policy() is None
    assert model._build_memory_ledger() is not None


def test_only_an_unconfigured_model_is_eligible_for_the_automatic_count():
    """Automatic means NO configure_devices call has been made.

    The constructor amendment collapsed eligibility to that one bit: there
    is no device string to parse, so EVERY call is explicit, including an
    unindexed ``devices=['cuda']``.  This test's earlier form asserted the
    pre-amendment rule (unindexed cuda stays automatic) and was gated on
    CUDA, so it first ran, and failed, on the first full-suite H100 run
    after the amendment.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    untouched = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    assert untouched.device_layout_is_automatic is True
    cpu = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    cpu.configure_devices(devices=['cpu'])
    assert cpu.device_layout_is_automatic is False
    if torch.cuda.is_available():
        plain = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
        plain.configure_devices(devices=['cuda'])
        assert plain.device_layout_is_automatic is False
        indexed = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
        indexed.configure_devices(devices=['cuda:0'])
        assert indexed.device_layout_is_automatic is False


def test_configure_devices_takes_the_layout_out_of_automatic_mode():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    model.device_layout_is_automatic = True          # as a CUDA model would be
    model.configure_devices(devices=['cpu', 'cpu'])
    assert model.device_layout_is_automatic is False


def test_preflight_knobs_have_their_documented_defaults():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    assert model.skip_memory_preflight is False
    assert model.memory_preflight_margin == 0.15


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


def test_a_torch_body_view_batch_is_charged_at_the_measured_slab_count():
    """A torch body holds a loop of slabs where the driver's nominal charge
    prices one, so the ledger charges the measured count of them.

    The slab is (view batch, pixels, width) floats, with width the wider of
    the detector rows and the slice band the call was handed -- the two axes
    the body sweeps.  The view batch stays the driver's own choice.
    """
    rows, channels, slices = 32, 32, 32
    p_sub = math.ceil(800 / 4)

    def charge(direction, num_pixels, band_cols):
        return 8, 1024                       # the driver's batch and nominal

    plan_kwargs = dict(view_charge=charge, num_pixels_full=800,
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


def test_the_torch_body_slab_follows_the_wider_of_rows_and_band():
    """The body allocates arrays at the detector-row extent AND at the slice
    band; the wider of the two sets the slab.  Under sharding the band is one
    owner's shard, so a tall volume's slab shrinks with the device count and a
    wide detector's does not."""
    def charge(direction, num_pixels, band_cols):
        return 1, 1

    def batch(rows, slices, n_devices):
        ledger = estimate_peak_device_bytes(make_plan(
            n_devices=n_devices, view_charge=charge,
            torch_body_directions=('forward', 'back'),
            num_pixels_full=800, num_rows=rows, recon=(32, 32, slices)))
        return dict(_named(ledger, 'initial forward projection')
                    .terms)['forward batch'][0]

    # Tall volume, narrow detector: at one device the band is all 64 slices,
    # and at four devices it is the 16-slice shard -- below the 32 rows, which
    # then set the slab.
    assert batch(32, 64, 1) == SLABS * 800 * 64 * 4
    assert batch(32, 64, 4) == SLABS * 800 * 32 * 4
    # Wide detector: the rows set the slab at every device count.
    assert batch(128, 32, 1) == SLABS * 800 * 128 * 4
    assert batch(128, 32, 4) == SLABS * 800 * 128 * 4


def test_a_torch_body_pays_for_both_forward_blocks():
    """The forward loop holds the outgoing block and the incoming one.

    Against a body that declares its own cost, one of the two is already
    inside the batch charge, because a forward kernel body's declaration
    prices its output plane per view.  A torch body declares nothing, and
    what the ledger charges in its place is the body's internal slab set,
    which does not include the output plane -- so both blocks are charged.
    """
    rows, channels = 32, 32

    def charge(direction, num_pixels, band_cols):
        return 8, 1024                       # 4 batches over 32 views

    def block(directions):
        ledger = estimate_peak_device_bytes(make_plan(
            n_devices=2, view_charge=charge,
            torch_body_directions=directions))
        return dict(_named(ledger, 'initial forward projection')
                    .terms)['forward block'][0]

    assert block(()) == 1 * 8 * rows * channels * 4
    assert block(('forward', 'back')) == 2 * 8 * rows * channels * 4
    # The back direction alone leaves the forward's own term where it was.
    assert block(('back',)) == 1 * 8 * rows * channels * 4


# One row per measured arm: (sinogram shape, recon shape, masked pixel count,
# per-device measured peak bytes).  Measured 2026-08-10 on four H100s (job
# mg8) -- the two geometries with no hand-written kernels, at one, two and
# four devices, weighted, from a supplied sinogram with no initial volume.
MEASURED_ARMS = {
    'ma1024_n1': ((1024, 1008, 992), (992, 992, 1148), 771240,
                  [37310451712]),
    'ma1024_n2': ((1024, 1008, 992), (992, 992, 1148), 771240,
                  [26138702848, 23767433216]),
    'ma1024_n4': ((1024, 1008, 992), (992, 992, 1148), 771240,
                  [17888278016, 17820163072, 17820163072, 16934779392]),
    'ma512_n1': ((512, 448, 384), (384, 384, 510), 115164,
                 [12253271552]),
    'ma512_n2': ((512, 448, 384), (384, 384, 510), 115164,
                 [9492893184, 9452805632]),
    'ma512_n4': ((512, 448, 384), (384, 384, 510), 115164,
                 [3768753664, 3757754368, 3757754368, 3757981696]),
    'tct2k_n1': ((256, 1900, 3000), (118, 360, 240), 42480,
                 [29262431744]),
    'tct2k_n2': ((256, 1900, 3000), (118, 360, 240), 42480,
                 [40081962496, 17882421760]),
    'tct2k_n4': ((256, 1900, 3000), (118, 360, 240), 42480,
                 [34051462656, 34081272320, 34081272320, 34081102336]),
    'tct1k_n1': ((256, 950, 1500), (59, 180, 120), 10620,
                 [8227791872]),
    'tct1k_n2': ((256, 950, 1500), (59, 180, 120), 10620,
                 [11882288128, 5844949504]),
    'tct1k_n4': ((256, 950, 1500), (59, 180, 120), 10620,
                 [29162906112, 29161971200, 29161972224, 29161972224]),
}
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
        view_charge=_measured_view_charge(sinogram_shape, recon_shape,
                                          n_devices),
        torch_body_directions=('forward', 'back'))
    return estimate_peak_device_bytes(plan), measured


@pytest.mark.parametrize('arm', sorted(MEASURED_ARMS))
def test_the_torch_body_ledger_covers_every_measured_peak(arm):
    """The floor, on the runs the slab count was calibrated from.

    A modeled peak below the measured one lets a doomed reconstruction start
    and die inside the allocator, which is the failure this module exists to
    prevent.  Every device of every measured arm must sit at or above 1.00.
    """
    ledger, measured = _measured_arm_ledger(arm)
    for i, peak in enumerate(measured):
        assert ledger.peak_bytes(i) >= peak, (
            f'{arm} device {i}: modeled {ledger.peak_bytes(i)} < '
            f'measured {peak}')


def test_the_torch_body_over_charge_stays_inside_its_band():
    """The other side of the floor: an over-charge spreads a reconstruction
    over more devices than it needs, so the band is asserted too.  It is
    wider than CALIBRATION_BAND because one slab count covers two geometries
    that hold different numbers of slabs, and because two measured two-device
    runs peaked twice as high on one device as on the other from identical
    shards.
    """
    low, high = _memory_ledger.TORCH_BODY_CALIBRATION_BAND
    assert low == _memory_ledger.CALIBRATION_BAND[0]
    worst = 0.0
    for arm in MEASURED_ARMS:
        ledger, measured = _measured_arm_ledger(arm)
        for i, peak in enumerate(measured):
            worst = max(worst, ledger.peak_bytes(i) / peak)
    assert low <= worst <= high


def test_format_calibration_judges_against_the_band_it_is_given():
    rows = [('cuda:0', 40 * GB, 10 * GB, 4.00)]
    assert 'over' in _memory_ledger.format_calibration(rows)
    assert 'over' not in _memory_ledger.format_calibration(
        rows, band=_memory_ledger.TORCH_BODY_CALIBRATION_BAND)


# ── the forward's column gather ──────────────────────────────────────────────
@pytest.mark.parametrize('aligned', (False, True),
                         ids=('two-fan', 'row-aligned'))
def test_the_column_gather_swaps_the_band_copy_for_a_gathered_cylinder(aligned):
    """The two states of the same phase.  Walking slice bands leaves a
    broadcast band resident on every view-owner; gathering columns leaves a
    cylinder that is one pixel batch wide and the whole slice axis tall, and
    no band at all.  Both forward phases carry the swap.

    Both GEOMETRIES are priced by the same arithmetic, and the parametrization
    is the claim: what a gather holds is set by the shape it assembles -- one
    pixel batch by the whole device-form slice axis -- and not by whether the
    geometry's detector rows track its slices.  The two take the path for
    different reasons and pay the same term for it.

    THREE such cylinders are charged, not one: the driver gathers one batch
    ahead of the projection that reads it, so the widest instant holds the
    cylinder about to be projected, the pieces arriving for the batch after
    it, and their concatenation.  The count is written out here rather than
    read from the module, so that changing the constant alone cannot move the
    charge without this test noticing."""
    slices, batch = 32, 100                  # make_plan's slice axis
    banded = estimate_peak_device_bytes(
        make_plan(n_devices=2, rows_track_slices=aligned))
    gathered = estimate_peak_device_bytes(
        make_plan(n_devices=2, rows_track_slices=aligned,
                  column_pixel_batch=batch))
    for fragment in ('initial forward projection',
                     'subset delta forward projection'):
        walked = dict(_named(banded, fragment).terms)
        columns = dict(_named(gathered, fragment).terms)
        assert walked['broadcast band'][0] > 0, fragment
        assert walked['column cylinder'] == [0, 0], fragment
        assert columns['broadcast band'] == [0, 0], fragment
        assert columns['column cylinder'] == [3 * batch * slices * 4] * 2, \
            fragment


def test_the_gathered_cylinder_is_capped_by_the_pass_it_covers():
    """A batch wider than the pixel set gathers the pixel set: the charge
    follows what one call is actually handed, which is what keeps the term
    honest at the small end without a separate rule.

    Such a pass runs as a single batch and so gathers nothing ahead, holding
    two cylinders where the charge is three.  That over-charge is deliberate:
    the ledger's one hard rule is that it may never charge less than a run
    needs, and one term that covers the widest instant is simpler than a
    second rule for the passes that fall short of it."""
    slices, pixels = 32, 800
    ledger = estimate_peak_device_bytes(
        make_plan(n_devices=2, column_pixel_batch=10 ** 6))
    terms = dict(_named(ledger, 'initial forward projection').terms)
    assert terms['column cylinder'] == [3 * pixels * slices * 4] * 2


def test_the_gathered_cylinder_does_not_grow_with_the_device_count():
    """The property that dissolves the objection to assembling whole
    cylinders: the term is the batch by the WHOLE slice axis on every
    view-owner, so adding devices does not change it -- where the broadcast
    band it replaces is a shard and halves with the count."""
    charges, bands = [], []
    for n in (2, 4):
        gathered = estimate_peak_device_bytes(
            make_plan(n_devices=n, column_pixel_batch=100))
        walked = estimate_peak_device_bytes(make_plan(n_devices=n))
        charges.append(dict(_named(gathered, 'initial forward projection')
                            .terms)['column cylinder'][0])
        bands.append(dict(_named(walked, 'initial forward projection')
                          .terms)['broadcast band'][0])
    assert charges[0] == charges[1]
    assert bands[1] == bands[0] // 2
    # A single device never gathers: it holds the whole volume already.
    one = estimate_peak_device_bytes(
        make_plan(n_devices=1, column_pixel_batch=100))
    assert dict(_named(one, 'initial forward projection')
                .terms)['column cylinder'] == [0]


def test_the_column_gather_prices_the_call_it_actually_makes():
    """The two terms that move with the new call shape.  One call is handed
    the WHOLE device-form slice axis instead of a band, and one pixel batch
    instead of every pixel of the pass, so the per-view cost model must be
    asked those two numbers."""
    asked = []

    def charge(direction, num_pixels, band_cols):
        asked.append((direction, num_pixels, band_cols))
        return 4, 1024

    batch = 100
    estimate_peak_device_bytes(make_plan(n_devices=2, view_charge=charge))
    walked = [(p, c) for d, p, c in asked if d == 'forward']
    asked.clear()
    estimate_peak_device_bytes(
        make_plan(n_devices=2, view_charge=charge, column_pixel_batch=batch))
    gathered = [(p, c) for d, p, c in asked if d == 'forward']
    assert {c for _p, c in walked} == {16}          # one slice shard of 32
    assert {c for _p, c in gathered} == {32}        # the whole slice axis
    assert max(p for p, _c in walked) == 800        # the whole pass
    assert max(p for p, _c in gathered) == batch


def test_plan_from_model_reads_the_resolved_pixel_batch(monkeypatch):
    """The ledger must not re-derive the driver's rule.  It asks the model
    for the batch it would actually walk, so a changed default or an override
    reaches the charge without a second edit here.

    The environment knob is cleared first, because the first assertion reads
    the default and a suite run may be forcing the path on around it."""
    from mbirtorch.tomography_model import (COLUMN_GATHER_ENV_VAR,
                                            FORWARD_PIXEL_BATCH)
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    cell = (8, 8, 8)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=32,
                                    source_iso_dist=16)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    devices = ['cpu', 'cpu']
    # Unset means the gather (the shipped default), so the charge is present
    # at the shipped batch; refusing the gather removes it.
    assert _memory_ledger.plan_from_model(
        model, devices).column_pixel_batch == FORWARD_PIXEL_BATCH
    model.forward_column_gather = False
    assert _memory_ledger.plan_from_model(
        model, devices).column_pixel_batch is None
    model.forward_column_gather = True
    model.forward_project_pixel_batch = 512
    assert _memory_ledger.plan_from_model(
        model, devices).column_pixel_batch == 512
    # The row-aligned geometry takes the same path, so the same resolution has
    # to reach its charge -- present by default, absent when refused, exactly
    # as on cone.
    par = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0],
                                                        endpoint=False))
    par.configure_devices(devices=['cpu'])
    par.set_params(no_warning=True, verbose=0)
    assert _memory_ledger.plan_from_model(
        par, devices).column_pixel_batch == FORWARD_PIXEL_BATCH
    par.forward_column_gather = False
    assert _memory_ledger.plan_from_model(
        par, devices).column_pixel_batch is None


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
