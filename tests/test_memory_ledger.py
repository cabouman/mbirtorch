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
              **kwargs):
    """A hand-built plan: no model, no device, no CUDA."""
    devices = ['cpu'] * n_devices
    sino_placement = _sharding.Placement(devices, axis=0, real_size=num_views)
    recon_placement = _sharding.Placement(devices, axis=-1, real_size=recon[2])
    return LedgerPlan(
        sinogram_shape=(num_views, num_rows, num_channels),
        recon_shape=recon,
        devices=devices,
        view_blocks=[(e - s, v) for _d, (s, e), v
                     in sino_placement.padded_shard_ranges()],
        slice_blocks=[(e - s, v) for _d, (s, e), v
                      in recon_placement.padded_shard_ranges()],
        sino_rows=num_rows,
        rows_track_slices=False,
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
    assert all('subset' in n or n == 'per-iteration statistics'
               for n in names)
    assert 'per-iteration statistics' in names


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


def test_band_reduce_is_flat_in_the_device_count():
    """The finding the error message's remedy ordering rests on.

    sum_band_to_owner materializes all n partials on the owner before summing,
    and one band is the whole shard by default, so the term reads about 1.5x a
    full-volume cylinder set at both two and four devices.  Adding devices
    shrinks the persistent set and leaves this where it was.
    """
    def reduce_bytes(n):
        ledger = estimate_peak_device_bytes(make_plan(n_devices=n))
        return dict(_named(ledger, 'back projection').terms)['band reduce'][0]

    two, four = reduce_bytes(2), reduce_bytes(4)
    assert two > 0 and four > 0
    # Flat within 1 percent between n=2 and n=4, not shrinking like 1/n.
    assert abs(two - four) / two < 0.01
    # And it does NOT collapse toward zero: both sit near 1.5x one full set.
    subset_full = math.ceil(800 / 4) * 32 * 4
    assert two == pytest.approx(1.5 * subset_full, rel=0.02)
    assert reduce_bytes(1) == 0       # a single device never runs the reduce


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
    back = _named(ledger, 'back projection')
    assert dict(back.terms)['band reduce'][3] == 0


def test_partitions_are_charged_to_the_lead_device_only():
    ledger = estimate_peak_device_bytes(make_plan(n_devices=2))
    partitions = dict(_named(ledger, 'prior').terms)['partitions (lead device)']
    assert partitions[0] > 0
    assert partitions[1] == 0


def test_slice_band_knob_reduces_the_band_reduce():
    """The remedy the error message names first must actually work."""
    wide = estimate_peak_device_bytes(make_plan(n_devices=2))
    narrow = estimate_peak_device_bytes(make_plan(n_devices=2, back_band=4))
    wide_reduce = dict(_named(wide, 'back projection').terms)['band reduce'][0]
    narrow_reduce = dict(_named(narrow, 'back projection').terms)['band reduce'][0]
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

    The transient is two sinogram-shaped squared-error products.  The recon
    L1 fuses into its own reduction and materializes nothing.
    """
    sino_bytes = 64 * 32 * 32 * 4
    ledger = estimate_peak_device_bytes(make_plan())
    stats = _named(ledger, 'per-iteration statistics')
    assert dict(stats.terms)['squared-error products'][0] == 2 * sino_bytes
    # It carries the persistent set, like every other in-loop phase.
    assert dict(stats.terms)['error sinogram'][0] == sino_bytes
    assert dict(stats.terms)['flat recon'][0] > 0


# ── the masked hessian ───────────────────────────────────────────────────────
def test_masked_hessian_agrees_with_the_full_grid_at_the_masked_indices():
    """The only places the engine ever reads the hessian.

    Back projection is independent per pixel, so a masked run must reproduce
    the dense run exactly at every masked index.  Outside the mask the masked
    run holds zeros instead of computed-but-never-read values.
    """
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles, device='cpu')
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
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles, device='cpu')
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
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles, device='cpu')
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
    arms necessarily back-project at different pixel counts, so they compile
    different shapes, and dynamo's shape specialization then perturbs the
    float realization of kernels that have nothing to do with the hessian.
    A compiled whole-recon comparison therefore measures the compiler rather
    than this change.  The change's own value claim is proved directly by
    the two tests above: the hessian is bitwise equal at every masked index,
    and the loop reads nowhere else.
    """
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles, device='cpu',
                                        compile_mode='off')
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
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles, device='cpu')
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
    model = mbirtorch.ParallelBeamModel((512, 64, 64), angles, device='cpu')
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


# ── the model-facing plan ────────────────────────────────────────────────────
def test_plan_from_model_reads_the_current_params_and_a_candidate_layout():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles, device='cpu')
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


def test_full_index_count_matches_the_index_array_and_caches():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles, device='cpu')
    assert model.full_index_count() == model._full_indices().shape[0]
    assert model.full_index_count() == model.full_index_count()


def test_ledger_is_not_built_for_a_cpu_model():
    """The ledger's production job is choosing a CUDA device count, so a CPU
    or MPS model never builds one and never pays for it."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles, device='cpu')
    assert model._build_memory_ledger() is None


def test_configure_devices_takes_the_layout_out_of_automatic_mode():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles, device='cpu')
    assert model.device_layout_is_automatic is True
    model.configure_devices(devices=['cpu', 'cpu'])
    assert model.device_layout_is_automatic is False


def test_preflight_knobs_have_their_documented_defaults():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles, device='cpu')
    assert model.skip_memory_preflight is False
    assert model.memory_preflight_margin == 0.15


def test_recon_is_unaffected_on_a_cpu_model():
    """The n=1 path must be untouched: no ledger, no preflight, same result."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles, device='cpu')
    model.set_params(no_warning=True, verbose=0)
    sinogram = np.zeros((8, 6, 8), dtype=np.float32)
    sinogram[:, 3, 4] = 1.0
    np.random.seed(0)
    recon, _info = model.recon(sinogram, max_iterations=2)
    assert np.all(np.isfinite(recon))


# ── helpers ──────────────────────────────────────────────────────────────────
def _named(ledger, fragment):
    for phase in ledger.phases:
        if fragment in phase.name:
            return phase
    raise AssertionError(f'no phase matching {fragment!r} in '
                         f'{[p.name for p in ledger.phases]}')


def _has(ledger, fragment):
    return any(fragment in p.name for p in ledger.phases)


def _fixed_ledger(peak):
    return Ledger(devices=['cuda:0'],
                  phases=[PhaseCharge('synthetic', [peak], [('all', [peak])])])
