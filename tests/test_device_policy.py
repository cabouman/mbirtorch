"""The automatic device count and the preflight that gates it.

A CUDA model spreads a reconstruction across the devices that can hold their
share.  These tests pin the RULE rather than the hardware: the selection, the
validation, the pin, the fallback, and the readable failure all run on CPU,
by driving the policy with fabricated device lists and budgets.  The
multi-device VALUE gates live in test_sharding.py, which is where the seeded
n>1 parity patterns already are.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _memory_ledger, _sharding
from mbirtorch._memory_ledger import MemoryPreflightError

GB = 2 ** 30

# Sinogram shapes the speed floors are measured at, each named for its view
# count; the comment beside each is its sinogram element count.
CELL_512 = (512, 448, 384)          #    88,080,384
CELL_1024 = (1024, 1008, 992)       # 1,023,934,464
CELL_128 = (128, 112, 96)           #     1,376,256
SPARSE_VIEW_CELL = (64, 448, 384)   #    11,010,048


def make_model(shape=(8, 6, 8), device='cpu', **kwargs):
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(shape, angles, **kwargs)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


@pytest.fixture(autouse=True)
def kernel_declared_projection(monkeypatch):
    """Price the projection the way the CUDA model these tests stand in for
    would price it.

    A CUDA parallel or cone model binds the hand-written kernel bodies, and
    each of those declares what one of its views holds.  On CPU no kernel is
    available, so the same model binds the general torch bodies instead, and
    the ledger prices a torch body's views for itself at a much larger
    residency (``_memory_ledger.TORCH_BODY_VIEW_SLABS``).  These tests are
    about the device-count RULE, not about either residency, so they hold the
    projection charge at the kernel-declared one; otherwise the capacity
    arithmetic they drive would be a different model's.  The torch-body
    charge has its own tests in test_memory_ledger.py.
    """
    monkeypatch.setattr(_memory_ledger, 'torch_body_directions',
                        lambda model: ())


@pytest.fixture
def no_speed_guard(monkeypatch):
    """Turn off the widening speed floors.

    The floors hold small problems at one device, and the toy shapes these
    tests use are far below every one of them.  A test whose subject is the
    CAPACITY rule therefore has to opt out, exactly as a user would, or it
    would be measuring two rules at once.  The floors' own effect on the
    chosen count is tested below, at the sizes they were measured at.
    """
    monkeypatch.setenv('MBIRTORCH_WIDENING_GUARD', '0')


@pytest.fixture
def unpinned(monkeypatch):
    """Clear the suite's device-count pin.

    The conftest fixture pins every test to one device, which is exactly what
    keeps the suite deterministic on a multi-GPU host.  The tests that
    exercise the SEARCH have to opt out of it, and doing so explicitly keeps
    the pin's reach visible.
    """
    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES', raising=False)


def as_automatic(model, num_devices):
    """Make a CPU model behave like an eligible CUDA model with
    ``num_devices`` visible, so the rule can be exercised without a GPU.

    The fake devices carry INDICES, so a test can give different devices
    different budgets and exercise the heterogeneous case.
    """
    model.device_layout_is_automatic = True
    model._candidate_devices = lambda n: [torch.device('cpu', i)
                                          for i in range(n)]
    return num_devices


def recon_peak(model, num_devices):
    """The largest per-device peak of the model's full recon plan at
    ``num_devices``.

    ``layout_fits`` admits a layout only when EVERY device holds its share and
    refuses it when any one is short, so the number that decides a budget
    question is the largest per-device peak, not the lead device's.
    """
    devices = [torch.device('cpu', i) for i in range(num_devices)]
    return max(model._build_memory_ledger(devices=devices).per_device_peaks())


def budget_admitting_only(monkeypatch, model, fits, misses):
    """A per-device budget the recon plan clears at ``fits`` and misses at
    every entry of ``misses``, each given as a ``(device count, preflight
    margin)`` pair.

    Both ends of the window are modeled peaks scaled by the margin they are
    priced against, so the budget is placed at the MIDPOINT and recomputed on
    every run.  A charge that moves either peak is followed, rather than
    breaking a factor someone chose once against numbers that have since
    moved.  The window is as wide as widening's effect on the per-device peak,
    which is the property these tests exist to exercise, so a window that
    closes is reported as that and not as arithmetic that happened to invert.
    """
    count, margin = fits
    floor = int((1.0 + margin) * recon_peak(model, count))
    ceiling = min(int((1.0 + m) * recon_peak(model, n)) for n, m in misses)
    assert floor < ceiling, (
        f'no budget clears {fits} and misses {misses}: widening no longer '
        f'lowers the modeled per-device peak enough to tell them apart '
        f'({floor} is not below {ceiling})')
    budget = (floor + ceiling) // 2
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: budget)
    return budget


# ── the selection rule ───────────────────────────────────────────────────────
def test_widening_picks_the_largest_count_that_fits(monkeypatch, unpinned,
                                                    no_speed_guard):
    """Two budgets through the same selection call.

    With room on every device the largest valid count wins.  Per-device peaks
    SHRINK as the count grows, so no uniform budget can admit two devices and
    refuse four; what refuses four is one device without room.  Every device
    in a candidate set must pass its own budget, so a busy `cuda:2` sends the
    rule back to two devices rather than letting it start a run only three
    devices could hold.
    """
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)

    roomy = make_model((16, 8, 16))
    as_automatic(roomy, 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)
    roomy._apply_device_policy()
    assert roomy.sino_placement.n_devices == 4

    # Devices 2 and 3 are nearly full; 0 and 1 are free.
    crowded = make_model((16, 8, 16))
    as_automatic(crowded, 4)
    monkeypatch.setattr(
        _memory_ledger, 'device_budget_bytes',
        lambda d: 1024 if (d.index or 0) >= 2 else 64 * GB)
    crowded._apply_device_policy()
    assert crowded.sino_placement.n_devices == 2


def test_widening_refuses_a_layout_that_leaves_a_device_idle(monkeypatch, unpinned):
    """The empty-shard rules gate the count before the ledger does."""
    # 3 views and 3 slices over 4 devices leaves device 3 with neither.
    model = make_model((3, 6, 8))
    model.set_params(no_warning=True, recon_shape=(6, 6, 3))
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)
    assert not model._layout_is_valid(['cpu'] * 4)
    model._apply_device_policy()
    assert model.sino_placement.n_devices <= 3


def test_a_single_visible_device_gets_no_preflight_at_all(monkeypatch,
                                                          unpinned):
    """One visible device has no layout to choose, so the ledger does not run.

    Torch's caching allocator already raises a fast, readable error on a
    single-device overflow, which is the job the preflight does where the
    allocator cannot.  Skipping it keeps the n=1 path free of any new
    per-reconstruction cost, and it means a single-GPU user cannot be refused
    a run that would previously have started.
    """
    model = make_model()
    as_automatic(model, 1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    # A budget of zero would refuse everything, and must never be consulted.
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', lambda d: 0)
    called = []
    original = model._build_memory_ledger
    model._build_memory_ledger = lambda *a, **k: (called.append(1),
                                                  original(*a, **k))[1]
    before = model.projector_functions
    assert model._apply_device_policy() is None
    assert not called                               # the ledger never ran
    assert model.sino_placement.n_devices == 1
    assert model.projector_functions is before      # no needless rebuild


# ── what turns the rule off ──────────────────────────────────────────────────
def test_an_explicit_configure_devices_is_never_second_guessed(monkeypatch, unpinned):
    model = make_model()
    model.configure_devices(devices=['cpu', 'cpu'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    # A budget of zero would fail any preflight; an explicit layout does not
    # consult one at all.
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', lambda d: 0)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2


def test_a_machine_without_cuda_never_widens(monkeypatch, unpinned):
    """The other half of CUDA-only.

    Under the constructor amendment every model starts in automatic mode, so
    the backend check moved from construction into the policy.  A model on a
    machine with no CUDA must still resolve to its own device and stay there.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    assert model.device_layout_is_automatic is True
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    assert model._apply_device_policy() is None
    assert model.sino_placement.n_devices == 1
    assert model.device_layout_is_automatic is True


def test_the_environment_pin_fixes_the_count(monkeypatch):
    """The pin decides the count, it is capped by what is visible, and it
    holds through a whole reconstruction -- which is the suite's own
    determinism guarantee on a multi-GPU host."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '2')
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2      # not the 4 that would fit

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    capped = make_model((16, 8, 16))
    as_automatic(capped, 2)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '8')
    capped._apply_device_policy()
    assert capped.sino_placement.n_devices == 2

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '1')
    pinned = make_model()
    as_automatic(pinned, 4)
    sinogram = np.zeros((8, 6, 8), dtype=np.float32)
    sinogram[:, 3, 4] = 1.0
    np.random.seed(0)
    pinned.recon(sinogram, max_iterations=2)
    assert pinned.sino_placement.n_devices == 1


# ── the doomed run ───────────────────────────────────────────────────────────
def test_a_doomed_run_fails_before_allocating_anything(monkeypatch, unpinned):
    """No count fits, including one, so the answer to 'which count' is
    'none'.  That must arrive as a readable error rather than as a
    reconstruction that dies later inside the allocator, and it must arrive
    through the public entry as well as the internal one."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 1024)          # a kilobyte: nothing fits

    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    with pytest.raises(MemoryPreflightError):
        model._apply_device_policy()

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    through_recon = make_model((16, 8, 16))
    as_automatic(through_recon, 2)
    sinogram = np.zeros((16, 8, 16), dtype=np.float32)
    sinogram[:, 4, 8] = 1.0
    np.random.seed(0)
    with pytest.raises(MemoryPreflightError):
        through_recon.recon(sinogram, max_iterations=1)


def test_skip_memory_preflight_forces_a_doomed_run(monkeypatch, unpinned,
                                                   no_speed_guard):
    """The escape hatch: the layout is chosen by the empty-shard rules alone
    and the budget is not consulted."""
    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    model.skip_memory_preflight = True
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', lambda d: 1024)
    model._apply_device_policy()                 # does not raise
    assert model.sino_placement.n_devices == 4


# ── the constructor amendment ────────────────────────────────────────────────
def test_a_params_change_does_not_force_an_eager_projector_build():
    """The stale-bind protection rebuilds what EXISTS; it must not create
    projectors a caller has not asked for."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.set_params(no_warning=True, delta_voxel=1.5)
    assert model._projector_functions is None
    # Once built, a params change rebuilds them in place.
    built = model.projector_functions
    model.set_params(no_warning=True, delta_voxel=2.0)
    assert model.projector_functions is not built


# ── device inheritance through copy_ct_model ─────────────────────────────────
def test_copy_ct_model_inherits_an_explicit_device_choice():
    """A copy of a model whose devices the user set gets the same devices; a
    copy of a model with no explicit device choice chooses for itself."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu', 'cpu'])
    copy = mbirtorch.copy_ct_model(model, new_num_det_rows=4)
    assert copy.device_layout_is_automatic is False
    assert copy.sino_placement.devices == model.sino_placement.devices

    automatic = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    automatic_copy = mbirtorch.copy_ct_model(automatic, new_num_det_rows=4)
    assert automatic_copy.device_layout_is_automatic is True


# ── the widening speed floors ────────────────────────────────────────────────
# The guard is a SPEED rule laid over the capacity rule: below a measured
# floor, a device count is pushed behind every admitted count rather than
# removed, so capacity still wins when nothing admitted fits.  These tests use
# the sizes the floors were actually measured at.  They are also the guard's
# standing regression coverage: every nightly row is env-pinned, and a pin
# bypasses the guard, so nothing else exercises this ordering end to end.
def with_four_visible(monkeypatch, model, budget=64 * GB):
    """Four visible devices, every one of them with ample room, so the ONLY
    thing that can hold the count down is the speed guard."""
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: budget)
    return model


def test_a_small_parallel_problem_holds_at_one_device_however_free_the_gpus(
        monkeypatch, unpinned):
    """The 128-class shape, where widening to four was measured 13x slower,
    holds at one device however free the GPUs are -- and the guard does not
    cost the case widening was built for: at the 1024-class shape n=4 is
    admitted and capacity has room."""
    small = with_four_visible(monkeypatch, make_model(CELL_128))
    small._apply_device_policy()
    assert small.sino_placement.n_devices == 1

    large = with_four_visible(monkeypatch, make_model(CELL_1024))
    large._apply_device_policy()
    assert large.sino_placement.n_devices == 4


# ── what turns the guard off ─────────────────────────────────────────────────
def test_an_explicit_choice_a_pin_or_the_switch_ignores_the_speed_floors(
        monkeypatch, unpinned):
    """Three ways past the floors, at a size they would otherwise hold at one
    device.

    A count the caller named is not the library's to second-guess.  The
    environment pin reaches the policy by a different branch and must bypass
    the guard the same way.  The guard's own switch restores the pure
    capacity order.
    """
    explicit = make_model(CELL_128)
    explicit.configure_devices(devices=['cpu'] * 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    explicit._apply_device_policy()
    assert explicit.sino_placement.n_devices == 4

    pinned = with_four_visible(monkeypatch, make_model(CELL_128))
    order, held = pinned._speed_ordered_candidates(4)
    assert order == [1, 4, 3, 2] and set(held) == {2, 3, 4}
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '4')
    pinned._apply_device_policy()
    assert pinned.sino_placement.n_devices == 4
    assert pinned.device_choice_rejections == []

    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES')
    switched = with_four_visible(monkeypatch, make_model(CELL_128))
    monkeypatch.setenv('MBIRTORCH_WIDENING_GUARD', '0')
    assert switched._speed_ordered_candidates(4) == ([4, 3, 2, 1], {})
    switched._apply_device_policy()
    assert switched.sino_placement.n_devices == 4


# ── capacity still wins ──────────────────────────────────────────────────────
def test_capacity_falls_back_past_a_speed_floor_and_says_so(monkeypatch,
                                                            unpinned):
    """The reorder never removes a count.

    Below its floor, n=4 is tried only after n=1 has been refused for lack of
    memory -- and then it is taken, because a run that fits slowly beats a
    run that does not fit at all.  The log says which happened.
    """
    model = make_model((128, 64, 128))           # 1.0M elements: below every floor
    # This shape reconstructs 16,384 pixels a slice, which the shipped forward
    # batch covers whole, so that transient does not shrink as devices are
    # added and the one- and four-device peaks barely separate.  Pinning the
    # batch to half a slice leaves the sharded arrays to dominate the
    # difference, which is what the budget below is built from.
    model.forward_project_pixel_batch = 8192
    with_four_visible(monkeypatch, model)
    # Only the wide layout fits: n=1 is priced first, misses, and n=4 is then
    # taken past its speed floor.
    margin = 0.02
    budget_admitting_only(monkeypatch, model, fits=(4, margin),
                          misses=[(1, margin)])
    model.memory_preflight_margin = margin
    model._apply_device_policy()

    assert model.sino_placement.n_devices == 4
    reasons = dict(model.device_choice_rejections)
    assert 1 in reasons and 'short' in reasons[1]
    assert 'chosen past its speed floor' in reasons[4]
    assert 'no admitted count fits' in reasons[4]


# ── the ordering itself ──────────────────────────────────────────────────────
def test_the_guard_reorders_the_candidates_and_never_removes_one():
    """The invariant the whole design rests on: every visible count is still
    a candidate, admitted ones first, each group largest-first."""
    for shape in (CELL_128, SPARSE_VIEW_CELL, CELL_512, CELL_1024):
        model = make_model(shape)
        order, held = model._speed_ordered_candidates(4)
        assert sorted(order) == [1, 2, 3, 4], shape
        assert set(held) <= set(order), shape
        admitted = [n for n in order if n not in held]
        assert order == admitted + [n for n in order if n in held], shape
        assert admitted == sorted(admitted, reverse=True), shape
        assert 1 in admitted, shape       # n=1 is always admitted


# ── the settled layout ───────────────────────────────────────────────────────
# The automatic choice is made once per model and kept: settling records the
# (sinogram_shape, recon_shape) pair it decided from, later calls reuse the
# layout while those shapes hold, and only a shape change re-decides.  The
# poison budget below is the discriminator: a call that consults any budget
# raises, so a passing test proves the settled path ran no search.
def poison_budgets(monkeypatch):
    def no_budget(_device):
        raise AssertionError('a settled call consulted a device budget')
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', no_budget)


def test_a_settled_model_does_not_redecide_when_free_memory_moves(
        monkeypatch, unpinned, no_speed_guard):
    """The automatic choice is made once and kept.  The poisoned budget is
    the proof that no second search ran, either on a later policy call or
    after a recompile-flagged parameter that leaves the shapes alone."""
    model = make_model((16, 8, 16))
    with_four_visible(monkeypatch, model)
    model._apply_device_policy()
    placement = model.sino_placement
    assert placement.n_devices == 4
    poison_budgets(monkeypatch)
    model._apply_device_policy()
    # The same layout, and the SAME placement object: a re-decision would
    # reinstall the placements and invalidate every Shards a caller holds.
    assert model.sino_placement is placement
    # Re-deciding on every recompile would unsettle the layout on a
    # detector-offset edit, and on the sigma_noise the denoiser sets at every
    # call.
    model.set_params(no_warning=True, det_channel_offset=0.5)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4


def test_a_shape_change_redecides_at_the_new_shapes(monkeypatch, unpinned,
                                                    no_speed_guard):
    model = make_model((16, 8, 16))
    with_four_visible(monkeypatch, model)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4
    # Two of the four devices fill up between the calls.  A settled model
    # ignores that by design; a shape change must not, because the
    # decision's inputs are new.
    monkeypatch.setattr(
        _memory_ledger, 'device_budget_bytes',
        lambda d: 1024 if (d.index or 0) >= 2 else 64 * GB)
    model.set_params(no_warning=True, sinogram_shape=(16, 12, 16))
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2


# ── the direct reconstructions ───────────────────────────────────────────────
# recon_fbp and recon_fdk settle the layout themselves, so a direct
# reconstruction spreads across the devices instead of landing whole on the
# lead one.  All four geometries are covered here: cone has made the call
# since commit 72208bb and the other three gained it with this increment.
def _automatic_parallel(shape=(8, 6, 8)):
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    return mbirtorch.ParallelBeamModel(shape, angles)


def _multiaxis_model(shape):
    """Multiaxis angles are (azimuth, elevation) pairs, one row per view."""
    azimuth = np.linspace(0, np.pi, shape[0], endpoint=False)
    elevation = np.linspace(-0.4, 0.4, shape[0])
    return mbirtorch.MultiAxisParallelModel(shape, np.stack([azimuth, elevation], axis=1))


def _automatic_cone(shape=(8, 6, 8)):
    angles = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
    return mbirtorch.ConeBeamModel(shape, angles,
                                   source_detector_dist=4.0 * shape[2],
                                   source_iso_dist=2.0 * shape[2])


def _automatic_translation():
    """Translation geometry needs a source far enough from the object for the
    automatic recon shape to exist, so it carries its own size -- the one
    test_translation.py reconstructs at -- rather than the shared toy shape.
    A translation grid must multiply out to the view count, which no toy
    shape divides, so this factory carries its own."""
    vectors = mbirtorch.gen_translation_vectors(4, 4, x_spacing=3.0,
                                                z_spacing=2.0)
    return mbirtorch.TranslationModel((vectors.shape[0], 40, 32), vectors,
                                      source_detector_dist=128.0,
                                      source_iso_dist=32.0)


DIRECT_RECONS = [
    (_automatic_parallel, 'recon_fbp'),
    (_automatic_cone, 'recon_fdk'),
    (lambda: _multiaxis_model((8, 6, 8)), 'recon_fbp'),
    (_automatic_translation, 'recon_fdk'),
]
DIRECT_RECON_IDS = ['parallel', 'cone', 'multiaxis', 'translation']


def _automatic_on_cpu(make, verbose=0):
    """A model the caller has never placed, put on the CPU the way the
    automatic path itself does.

    ``_install_device_layout`` carries no policy, so the layout stays the
    library's to choose, while the fabricated CUDA visibility below cannot
    pull real allocations onto a device this host lacks.
    """
    model = make()
    model.set_params(no_warning=True, verbose=verbose)
    model._install_device_layout(['cpu'])
    assert model.device_layout_is_automatic is True
    return model


def _impulse_sinogram(model):
    shape = tuple(int(s) for s in model.get_params('sinogram_shape'))
    sinogram = np.zeros(shape, dtype=np.float32)
    sinogram[:, shape[1] // 2, shape[2] // 2] = 1.0
    return sinogram


@pytest.mark.parametrize("make,method", DIRECT_RECONS, ids=DIRECT_RECON_IDS)
def test_a_bare_recon_direct_settles_the_layout_and_spreads(
        monkeypatch, unpinned, no_speed_guard, make, method):
    """The A2 gap, closed for every geometry: a direct reconstruction on a
    model with no explicit layout uses the devices that fit, exactly as recon
    does, rather than running whole on the lead device."""
    model = _automatic_on_cpu(make)
    with_four_visible(monkeypatch, model)
    recon = getattr(model, method)(_impulse_sinogram(model))
    assert model.sino_placement.n_devices == 4
    assert recon.shape == tuple(model.get_params('recon_shape'))
    assert np.all(np.isfinite(recon))


# ── the check against the work in progress ───────────────────────────────────
# The device COUNT is chosen with the full recon plan, because the settled
# layout serves the model's whole life.  The capacity check that can REFUSE is
# made against the call in progress, so a direct reconstruction is not turned
# away for a recon it is not going to run.

# Sinogram shapes to look for that budget at, tried in this order.  The first
# is the one these tests have always run at and is expected to be the one they
# get.  The others are taller volumes, where the phases only a full recon
# builds -- the prior, the hessian diagonal, the error sinogram -- grow faster
# than the filter and single back projection of a direct reconstruction do, so
# the two peaks stand further apart.
BETWEEN_THE_PLANS_SHAPES = [(128, 64, 128), (128, 128, 128), (128, 256, 128)]

# How far above the preflight's own margin the budget's lower bound sits, so
# the direct plan clears the check with room instead of landing on it.  At the
# default 0.15 margin this is the 1.20 these tests have always used.
BETWEEN_THE_PLANS_HEADROOM = 0.05


def two_plan_peaks(model):
    """The direct plan's peak at four devices and the smallest full-recon peak
    over the counts the search tries, both the largest-per-device measure
    ``recon_peak`` explains.
    """
    direct = max(model._build_memory_ledger(
        devices=[torch.device('cpu', i) for i in range(4)],
        workload='direct').per_device_peaks())
    recon = min(recon_peak(model, n) for n in (1, 2, 3, 4))
    return direct, recon


def model_between_the_two_plans(monkeypatch):
    """A four-device model, and a per-device budget that its direct
    reconstruction fits at four devices while no full recon fits at any count.

    Both bounds come from the model's own ledger, so these tests fix the
    RELATION between the two plans rather than a number that would have to be
    rewritten whenever a charge moves.  The budget is placed at the MIDPOINT of
    the two, which is the point in the window furthest from both bounds, and it
    is recomputed on every run: a charge that moves either peak is followed,
    and only a charge set that closes the window outright can be a problem.
    Should one do that, the shapes above are tried in turn and the test skips
    saying which peaks it saw -- these tests are about the rule, and a window
    too narrow to express it is not a failure of the rule.
    """
    tried = []
    for shape in BETWEEN_THE_PLANS_SHAPES:
        model = _automatic_on_cpu(lambda: _automatic_parallel(shape))
        direct, recon = two_plan_peaks(model)
        floor = int((1.0 + model.memory_preflight_margin
                     + BETWEEN_THE_PLANS_HEADROOM) * direct)
        if floor < recon:
            with_four_visible(monkeypatch, model)
            budget = (floor + recon) // 2
            monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                                lambda d: budget)
            return model
        tried.append(f'{shape} needs {floor} for its direct plan but its '
                     f'smallest recon plan is {recon}')
    pytest.skip('no shape tried leaves a budget between the direct plan and '
                'the recon plan, so no budget can admit one and refuse the '
                'other: ' + '; '.join(tried))


def test_a_geometry_too_large_for_a_recon_still_runs_a_recon_direct(
        monkeypatch, unpinned, no_speed_guard):
    """The narrower calls run where a full recon is refused.

    Every device count is short for a full recon here.  A direct
    reconstruction, placing a sinogram, and the hessian diagonal each
    allocate far less than that, so each is checked against its own footprint
    and runs on the widest count that holds it.  The layout is then recorded
    as serving the narrower check, so the recon that follows is refused --
    HERE, rather than inside the allocator -- instead of running on a layout
    no recon check ever passed.
    """
    model = model_between_the_two_plans(monkeypatch)
    recon = model.recon_fbp(_impulse_sinogram(model))
    assert model.sino_placement.n_devices == 4
    assert model._settled_workload == 'direct'
    assert np.all(np.isfinite(recon))
    # The count in use is not reported as a rejection, and the line says why
    # it was taken.
    reasons = dict(model.device_choice_rejections)
    assert 'chosen for the direct reconstruction in progress' in reasons[4]
    assert '4 used, chosen for the direct reconstruction' in \
        model._device_report()
    with pytest.raises(MemoryPreflightError):
        model.recon(_impulse_sinogram(model), max_iterations=1)

    placing = model_between_the_two_plans(monkeypatch)
    prepared = placing.prepare_sino_for_devices(_impulse_sinogram(placing))
    assert placing.sino_placement.n_devices == 4
    assert placing._settled_workload == 'direct'
    assert prepared.placement == placing.sino_placement
    with pytest.raises(MemoryPreflightError):
        placing.recon(prepared, max_iterations=1)

    # The hessian diagonal is one masked back projection beside a weights
    # array, which the direct plan's charges cover.
    hessian_model = model_between_the_two_plans(monkeypatch)
    hessian = hessian_model.compute_hessian_diagonal()
    assert hessian_model.sino_placement.n_devices == 4
    assert hessian_model._settled_workload == 'direct'
    assert np.all(np.isfinite(hessian))
    with pytest.raises(MemoryPreflightError):
        hessian_model.recon(_impulse_sinogram(hessian_model),
                            max_iterations=1)


# ── the generation model ─────────────────────────────────────────────────────
# generate_demo_data builds a model of its own to project the phantom through,
# and that model settles like any other reconstruction entry.  It is the one
# model that settles with the capacity preflight skipped: it lives for a single
# projection and is deleted before the function returns, so there is no
# reconstruction lifetime to size it for.  The function's own name for it is
# gone by the time a test could read it, so these tests capture the instance as
# it is built.
DEMO_SHAPE = dict(num_views=8, num_det_rows=8, num_det_channels=12)


def capture_generation_model(monkeypatch, geometry, budget=64 * GB):
    """Record the model generate_demo_data builds, with the fabricated
    four-device visibility the rest of this file uses.

    The model is placed on the CPU as it is constructed, before the generation
    reaches its own set_params calls, so the fabricated visibility cannot pull
    a real allocation onto a device this host lacks.

    Returns:
        list: empty until the generation runs, then holding the one model.
    """
    built = []
    construct = getattr(mbirtorch, geometry)

    def build_and_record(*args, **kwargs):
        model = construct(*args, **kwargs)
        model.set_params(no_warning=True, verbose=0)
        model._install_device_layout(['cpu'])
        with_four_visible(monkeypatch, model, budget=budget)
        built.append(model)
        return model

    monkeypatch.setattr(mbirtorch, geometry, build_and_record)
    return built


def test_the_generation_model_projects_on_the_layout_it_settles(
        monkeypatch, unpinned, no_speed_guard):
    """The E1 gap, closed: with no devices= the phantom projection spreads over
    the devices that fit, rather than running whole on the lead one.  A
    devices= list stays an explicit pin, so the same generation runs on the
    two devices the caller named although four are free."""
    built = capture_generation_model(monkeypatch, 'ParallelBeamModel')
    _phantom, sinogram, _params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='cube', **DEMO_SHAPE)
    model, = built
    assert model.device_layout_is_automatic is True
    assert model.sino_placement.n_devices == 4
    assert sinogram.shape == (8, 8, 12)
    assert np.all(np.isfinite(sinogram)) and sinogram.max() > 0

    pinned = capture_generation_model(monkeypatch, 'ParallelBeamModel')
    _phantom, sinogram, _params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='cube',
        devices=['cpu', 'cpu'], **DEMO_SHAPE)
    pinned_model, = pinned
    assert pinned_model.device_layout_is_automatic is False
    assert pinned_model.sino_placement.n_devices == 2
    assert np.all(np.isfinite(sinogram)) and sinogram.max() > 0


# ── the full-array allocators ────────────────────────────────────────────────
# Three helpers allocate a whole sinogram or a whole volume before any
# reconstruction runs: compute_hessian_diagonal, prepare_sino_for_devices, and
# gen_weights_mar on the branch that forward projects.  Each settles the layout
# first, so its arrays land on the devices the later reconstructions use rather
# than whole on the lead one.
#
# Settling is what first hands a placed array to the entries below it, so those
# entries had to learn about the device form before the settle was added.
# gen_weights, recon_split_sino, and recon_plastic_metal all refuse one: each
# of the three would have to gather it before doing any work.
def _placed_cone_case(cell=(12, 24, 16), num_devices=2):
    """A cone model small enough to reconstruct in a test, placed on virtual
    CPU devices, with a phantom sinogram on the host.

    The devices are named explicitly, so no fabricated CUDA visibility is
    involved.
    """
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles,
                                    source_detector_dist=4.0 * cell[2],
                                    source_iso_dist=2.0 * cell[2])
    model.configure_devices(devices=['cpu'] * num_devices)
    model.set_params(no_warning=True, verbose=0)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        tuple(model.get_params('recon_shape')))
    sinogram = np.asarray(model.forward_project(phantom))
    return model, sinogram


def test_the_host_only_entries_refuse_a_placed_sinogram():
    """gen_weights, recon_split_sino and recon_plastic_metal all do their work
    on the host and would have to gather a placed sinogram before starting, so
    each refuses one.

    The silent wrong answer this replaces: a placed sinogram is neither numpy
    nor a tensor, so gen_weights resolved the array module to numpy and
    'unweighted' came back as a zero-dimensional object array.  Each half of
    the pair is checked, so a host sinogram carrying placed weights is refused
    too.
    """
    model, sinogram = _placed_cone_case()
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission')
    prepared, placed_weights = model.prepare_sino_for_devices(sinogram,
                                                              weights=weights)
    with pytest.raises(ValueError, match='placed on the devices'):
        mbirtorch.gen_weights(prepared, weight_type='unweighted')
    with pytest.raises(ValueError, match='sharded form'):
        model.recon_split_sino(prepared, weights=weights, half_overlap=3)
    with pytest.raises(ValueError, match='sharded form'):
        model.recon_split_sino(sinogram, weights=placed_weights,
                               half_overlap=3)
    with pytest.raises(ValueError, match='in sharded form'):
        model.recon_plastic_metal(prepared, weights, num_metal=0)
    with pytest.raises(ValueError, match='in sharded form'):
        model.recon_plastic_metal(sinogram, placed_weights, num_metal=0)


def test_the_supported_order_places_the_weights_with_the_sinogram():
    """What the rejection points the caller at: weights from the host
    sinogram, then one prepare call for the pair.  Placing copies, so the
    placed weights gather back to the plain computation exactly."""
    model, sinogram = _placed_cone_case()
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission')
    placed_sino, placed_weights = model.prepare_sino_for_devices(
        sinogram, weights=weights)
    assert placed_sino.placement.n_devices == 2
    assert placed_weights.placement.n_devices == 2
    assert np.array_equal(model._gather_sinogram(placed_weights), weights)
    assert np.array_equal(model._gather_sinogram(placed_sino), sinogram)


def test_compute_hessian_diagonal_settles_before_it_allocates(
        monkeypatch, unpinned, no_speed_guard):
    """A full sinogram of weights and a full volume, both sized by the model.
    On an unsettled model they landed whole on the lead device; after the
    settle they are spread, and the values are the single-device ones."""
    model = _automatic_on_cpu(lambda: _automatic_parallel((16, 8, 16)))
    with_four_visible(monkeypatch, model)
    spread = model.compute_hessian_diagonal(output_sharded=True)
    assert model.sino_placement.n_devices == 4
    assert len(spread.tensors) == 4
    # Spread means allocated per device, not built whole and then divided: the
    # slice axis is 8 long, so each of the four devices holds two slices.
    assert [int(t.shape[-1]) for t in spread.tensors] == [2, 2, 2, 2]

    reference = make_model((16, 8, 16)).compute_hessian_diagonal()
    gathered = model._gather_recon(spread)
    assert gathered.shape == reference.shape
    rel_max = float(np.max(np.abs(gathered - reference))
                    / max(float(np.max(np.abs(reference))), 1e-30))
    print(f"hessian diagonal, 4 devices vs 1: rel_max = {rel_max:.2e}")
    assert rel_max < 1e-5


def test_prepare_sino_for_devices_settles_before_it_places(
        monkeypatch, unpinned, no_speed_guard):
    """The whole sinogram, placed once.  Settling first is what makes the
    placement the final one, so a reconstruction on the same model reuses it
    instead of re-placing."""
    model = _automatic_on_cpu(lambda: _automatic_parallel((16, 8, 16)))
    with_four_visible(monkeypatch, model)
    sinogram = _impulse_sinogram(model)
    prepared = model.prepare_sino_for_devices(sinogram)
    assert model.sino_placement.n_devices == 4
    # 16 views over four devices, four views each.
    assert [int(t.shape[0]) for t in prepared.tensors] == [4, 4, 4, 4]
    assert np.array_equal(model._gather_sinogram(prepared), sinogram)
    # The placement the sinogram is on is the model's own, so a reconstruction
    # takes the prepared array as it stands.
    assert model._shard_sinogram(prepared) is prepared


def _mar_inputs(model, seed=0):
    """A sinogram and an initial reconstruction for gen_weights_mar, both on
    the host.  The values only have to span the metal threshold the tests
    pass, so they are drawn rather than reconstructed."""
    rng = np.random.default_rng(seed)
    sinogram = rng.random(tuple(model.get_params('sinogram_shape'))).astype(np.float32)
    init_recon = rng.random(tuple(model.get_params('recon_shape'))).astype(np.float32)
    return sinogram, init_recon


def test_gen_weights_mar_settles_on_the_branch_that_projects(
        monkeypatch, unpinned, no_speed_guard):
    """The init_recon branch forward projects a full metal mask, which is the
    allocation the settle protects.  The weights match the single-device
    ones."""
    model = _automatic_on_cpu(lambda: _automatic_parallel((16, 8, 16)))
    with_four_visible(monkeypatch, model)
    sinogram, init_recon = _mar_inputs(model)
    weights = mbirtorch.gen_weights_mar(model, sinogram, init_recon=init_recon,
                                        metal_threshold=0.8)
    assert model.sino_placement.n_devices == 4

    reference = mbirtorch.gen_weights_mar(make_model((16, 8, 16)), sinogram,
                                          init_recon=init_recon,
                                          metal_threshold=0.8)
    rel_max = float(np.max(np.abs(weights - reference))
                    / max(float(np.max(np.abs(reference))), 1e-30))
    print(f"gen_weights_mar, 4 devices vs 1: rel_max = {rel_max:.2e}")
    assert rel_max < 1e-6


# ── the denoiser under the policy ────────────────────────────────────────────
# denoise settles its layout through the same policy a reconstruction uses.
# What differs is the plan the candidates are priced with: a denoiser has no
# projectors and can never run a recon, so its own sweep is the largest
# workload it will ever hold, and pricing a recon plan on it raises instead.
#
# These tests RUN the sweep, so their candidate devices are plain cpu devices
# rather than the indexed fakes ``as_automatic`` builds: an indexed cpu device
# settles fine but cannot hold a tensor.
DENOISE_CELL = (8, 10, 13)   # 13 slices over 3 devices split 5/4/4


def automatic_denoiser(monkeypatch, image_shape=DENOISE_CELL, num_devices=3,
                       budget=64 * GB):
    """A denoiser the caller has never placed, with ``num_devices`` fabricated
    devices visible and room on every one of them.

    The CPU layout is installed the way the automatic path installs one, so
    the fabricated visibility cannot pull a real allocation onto a device this
    host lacks while the layout stays the library's to choose.
    """
    denoiser = mbirtorch.QGGMRFDenoiser(image_shape)
    denoiser.set_params(no_warning=True, verbose=0)
    denoiser._install_device_layout(['cpu'])
    assert denoiser.device_layout_is_automatic is True
    denoiser._candidate_devices = lambda n: [torch.device('cpu')] * n
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: num_devices)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: budget)
    return denoiser


def noisy_image(shape=DENOISE_CELL, seed=4):
    """A seeded noisy block, so a sharded run and a single-device run see the
    same problem."""
    clean = np.zeros(shape, dtype=np.float32)
    clean[1:-1, 2:-2, 2:-2] = 1.0
    noise = np.random.RandomState(seed).randn(*shape).astype(np.float32)
    return clean + 0.1 * noise


def run_denoise(denoiser, image, sigma_noise=0.1, max_iterations=2):
    np.random.seed(0)
    denoised, _ = denoiser.denoise(image, sigma_noise=sigma_noise,
                                   max_iterations=max_iterations,
                                   stop_threshold_change_pct=0.0,
                                   logfile_path=None, print_logs=False)
    return denoised


def test_denoise_settles_an_automatic_layout_and_matches_single_device(
        monkeypatch, unpinned, no_speed_guard):
    """A denoiser the caller has not placed spreads over the devices that fit,
    exactly as a reconstruction does, and the sharded sweep returns what one
    device returns (gated at the level test_denoiser.py uses for the same
    comparison)."""
    denoiser = automatic_denoiser(monkeypatch)
    image = noisy_image()
    out = run_denoise(denoiser, image)
    assert denoiser.recon_placement.n_devices == 3
    assert denoiser.device_layout_is_automatic is True

    single = mbirtorch.QGGMRFDenoiser(DENOISE_CELL)
    single.configure_devices(devices=['cpu'])
    single.set_params(no_warning=True, verbose=0)
    ref = run_denoise(single, image)
    rel = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    assert rel <= 1e-4


# ── two models sharing one device layout ─────────────────────────────────────
# A Plug-and-Play or ADMM loop alternates prox_map on a reconstruction model
# with denoise on a QGGMRFDenoiser over the same volume.  Both accept and
# return the device form, so the volume can stay on the devices for the whole
# loop -- but only if the two models place recon-like arrays the same way.
# configure_devices(like=...) is how they are made to agree, and the tests
# below drive the handoff on two virtual CPU devices, in both directions.
PAIR_CELL = (12, 8, 20)      # recon shape (20, 20, 8): 8 slices, split 4 + 4


def paired_ct_and_denoiser(sino_shape=PAIR_CELL, devices=('cpu', 'cpu')):
    """A parallel-beam model and a denoiser built at its recon shape, placed
    on the same (virtual) devices -- the two-line idiom the configure_devices
    docstring gives."""
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    ct_model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    ct_model.configure_devices(devices=list(devices))
    ct_model.set_params(no_warning=True, verbose=0)

    denoiser = mbirtorch.QGGMRFDenoiser(ct_model.get_params('recon_shape'))
    denoiser.configure_devices(like=ct_model)
    denoiser.set_params(no_warning=True, verbose=0)
    return ct_model, denoiser


def pair_phantom(ct_model):
    """A small block phantom in the model's recon shape."""
    recon_shape = tuple(ct_model.get_params('recon_shape'))
    phantom = np.zeros(recon_shape, dtype=np.float32)
    r0, c0, s0 = [max(1, n // 4) for n in recon_shape]
    phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    return phantom


def test_a_denoiser_placed_like_a_ct_model_takes_its_sharded_volume():
    """The handoff the whole seam exists for, on two virtual CPU devices.

    The two models are configured separately, so their recon placements are
    distinct OBJECTS naming the same devices, axis and slice count.  Equality
    rather than identity is what lets the denoiser accept a volume the
    reconstruction model placed, with no gather to the host in between.  The
    return leg closes the loop: what denoise hands back in the device form is
    a valid prox_input for the reconstruction model.
    """
    ct_model, denoiser = paired_ct_and_denoiser()
    assert denoiser.recon_placement == ct_model.recon_placement
    assert denoiser.recon_placement is not ct_model.recon_placement

    phantom = pair_phantom(ct_model)
    sinogram = ct_model.forward_project(phantom)
    placed = ct_model._shard_recon(phantom)
    assert isinstance(placed, _sharding.Shards)

    denoised, _ = denoiser.denoise(placed, sigma_noise=0.1, max_iterations=1,
                                   stop_threshold_change_pct=0.0,
                                   logfile_path=None, print_logs=False,
                                   output_sharded=True)
    assert isinstance(denoised, _sharding.Shards)
    assert denoised.placement == ct_model.recon_placement
    assert ([tuple(t.shape) for t in denoised.tensors]
            == [tuple(t.shape) for t in placed.tensors])
    assert np.isfinite(denoised.gather()).all()

    np.random.seed(0)
    recon, _ = ct_model.prox_map(denoised, sinogram, sigma_prox=0.5,
                                 init_recon=phantom, max_iterations=1,
                                 stop_threshold_change_pct=0.0,
                                 logfile_path=None, print_logs=False,
                                 output_sharded=True)
    assert isinstance(recon, _sharding.Shards)
    assert recon.placement == ct_model.recon_placement
    # The reconstruction comes back as a volume, one 3-D block per device,
    # split on the slice axis exactly as the input was.
    assert ([tuple(t.shape) for t in recon.tensors]
            == [tuple(t.shape) for t in denoised.tensors])
    assert np.isfinite(recon.gather()).all()


def test_like_refuses_a_denoiser_built_at_the_wrong_shape():
    """The realistic mistake: a denoiser built at the CT model's SINOGRAM
    shape instead of its recon shape.

    A volume from one model would then not fit the other, so the pairing is
    refused when it is configured -- with both shapes named -- rather than at
    some later array that fails to line up.  The check is on the whole recon
    shape, not the slice count alone: a mismatch in the rows or the columns
    divides into the same blocks and would pass a slice-count check, then
    fail deep inside a reconstruction as an unreadable tensor error.
    """
    sino_shape = PAIR_CELL          # recon shape (20, 20, 8): 8 slices, not 20
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    ct_model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    ct_model.configure_devices(devices=['cpu', 'cpu'])
    wrong = mbirtorch.QGGMRFDenoiser(sino_shape)

    with pytest.raises(ValueError, match='recon shape') as excinfo:
        wrong.configure_devices(like=ct_model)
    message = str(excinfo.value)
    assert 'QGGMRFDenoiser' in message and 'ParallelBeamModel' in message
    assert str(tuple(sino_shape)) in message
    assert str(tuple(ct_model.get_params('recon_shape'))) in message
    # And it names the way to share devices WITHOUT sharing arrays.
    assert 'devices=' in message

    # Two recon shapes that agree on the slice count but differ in the rows
    # and columns are refused as well.  Their placements would be equal -- the
    # slice axis divides identically -- so nothing downstream would notice
    # until a volume from one was reshaped for the other.
    recon_shape = tuple(ct_model.get_params('recon_shape'))
    wider = (recon_shape[0] + 4, recon_shape[1] + 4, recon_shape[2])
    with pytest.raises(ValueError, match='recon shape') as excinfo:
        mbirtorch.QGGMRFDenoiser(wider).configure_devices(like=ct_model)
    message = str(excinfo.value)
    assert str(wider) in message and str(recon_shape) in message


def test_a_sharded_prox_input_that_misses_the_volume_is_refused_clearly():
    """A sharded prox input has to cover the whole volume: the shards' leading
    dimensions have to span the pixel grid and their slice counts have to add
    up to the volume's.

    The container carries no shape of its own, so without this check a
    mis-shaped set of shards would surface as a torch reshape failure deep in
    the loop.  The message names what was expected and what arrived, in the
    same voice as the one a mis-shaped host array gets.
    """
    ct_model, _ = paired_ct_and_denoiser()
    phantom = pair_phantom(ct_model)
    sinogram = ct_model.forward_project(phantom)
    placement = ct_model.recon_placement
    blocks = [e - s for _d, (s, e) in placement.shard_ranges()]

    def run(shard_shapes):
        shards = _sharding.Shards(
            [torch.zeros(shape, dtype=torch.float32) for shape in shard_shapes],
            placement)
        np.random.seed(0)
        return ct_model.prox_map(shards, sinogram, sigma_prox=0.5,
                                 init_recon=phantom, max_iterations=1,
                                 stop_threshold_change_pct=0.0,
                                 logfile_path=None, print_logs=False,
                                 output_sharded=True)

    # Right slice split, wrong pixel grid.
    with pytest.raises(ValueError,
                       match='prox_input does not have the correct size'):
        run([(3, n) for n in blocks])
    # Right pixel grid, slice counts that do not add up to the volume's.
    recon_shape = tuple(ct_model.get_params('recon_shape'))
    with pytest.raises(ValueError,
                       match='prox_input does not have the correct size'):
        run([(recon_shape[0], recon_shape[1], n - 1) for n in blocks])


def test_a_shard_that_owns_no_slices_is_a_legal_prox_input():
    """More devices than slices leaves a trailing device with an empty block,
    which is a layout the library allows as long as that device still owns
    views.  Such a shard has no elements, so it must pass the prox input's
    whole-volume check rather than be read as a shard that covers nothing."""
    sino_shape = (12, 2, 20)        # recon shape (20, 20, 2): 2 slices
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    ct_model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    ct_model.configure_devices(devices=['cpu', 'cpu', 'cpu'])
    ct_model.set_params(no_warning=True, verbose=0)

    recon_shape = tuple(ct_model.get_params('recon_shape'))
    blocks = [e - s for _d, (s, e)
              in ct_model.recon_placement.shard_ranges()]
    assert blocks == [1, 1, 0]

    shards = _sharding.Shards(
        [torch.zeros((recon_shape[0], recon_shape[1], n), dtype=torch.float32)
         for n in blocks], ct_model.recon_placement)
    flat = ct_model._flatten_prox_shards(shards, recon_shape)
    assert isinstance(flat, _sharding.Shards)
    assert ([tuple(t.shape) for t in flat.tensors]
            == [(recon_shape[0] * recon_shape[1], n) for n in blocks])


# ── the prepared sinogram, straight into a reconstruction ────────────────────
# prepare_sino_for_devices divides a sinogram across the devices once, and the
# reconstruction entries take that divided form as it stands.  A Plug-and-Play
# loop can therefore prepare its sinogram once and pay the host-to-device
# transfer once, rather than on every prox_map call.  The tests below drive
# that on two virtual CPU devices, against the same runs on host arrays, at the
# small cell and phantom the section above already builds.


def model_on_devices(sino_shape, num_devices):
    """A parallel-beam model whose sinogram is divided across ``num_devices``
    virtual CPU devices."""
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=['cpu'] * num_devices)
    model.set_params(no_warning=True, verbose=0)
    return model


def view_shards(model, sinogram):
    """A sinogram in the divided device form, one tensor per device.

    This is what prepare_sino_for_devices returns, built here directly so that
    a single-device placement gives shards too: that call hands back a plain
    tensor when there is only one device, and the subsample arithmetic below
    is about the divided form at every device count.
    """
    return _sharding.Shards(
        [torch.as_tensor(sinogram[start:end]) for _d, (start, end)
         in model.sino_placement.shard_ranges()], model.sino_placement)


def max_relative_difference(out, ref):
    """The comparison the value tests in this file use: the largest pointwise
    difference as a fraction of the reference's largest magnitude."""
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def run_recon(model, sinogram, weights=None, max_iterations=2):
    """One short reconstruction, seeded so that two runs draw the same pixel
    partitions and differ only in the form their sinogram arrived in."""
    np.random.seed(0)
    recon, _ = model.recon(sinogram, weights=weights,
                           max_iterations=max_iterations,
                           stop_threshold_change_pct=0.0,
                           logfile_path=None, print_logs=False)
    return recon


@pytest.mark.parametrize('num_views,num_devices',
                         [(12, 2), (13, 3), (41, 2), (63, 3)])
def test_a_divided_sinogram_subsamples_exactly_like_a_whole_one(num_views,
                                                                num_devices):
    """The statistics that set the regularization parameters run on an
    evenly-spaced subsample of the views, and taking that subsample from the
    divided form is data movement rather than an approximation.  It therefore
    has to give the very same views in the very same order as subsampling the
    assembled sinogram.

    The parameters cover both things the arithmetic depends on.  12 and 13
    views are subsampled view by view (stride 1), 41 strides by 2 and 63 by
    3; and some of those counts divide evenly over their devices while others
    leave the first devices one view longer.
    """
    model = model_on_devices((num_views, 4, 6), num_devices)
    whole = np.random.RandomState(num_views).rand(
        num_views, 4, 6).astype(np.float32)
    divided = model.subsample_views(view_shards(model, whole))
    assert np.array_equal(divided, model.subsample_views(whole))


def test_a_prepared_sinogram_reconstructs_like_a_host_one():
    """The point of the whole seam: recon takes the divided form directly and
    returns what the same call on the host array returns, weights included --
    those are placed by the same call and travel the same path.

    Not bit-for-bit -- the per-device sums this model runs on CPU vary a
    little from run to run whatever their input was -- so the comparison is
    the relative one the value tests in this file use.
    """
    model = model_on_devices(PAIR_CELL, 2)
    phantom = pair_phantom(model)
    sinogram = model.forward_project(phantom)
    prepared = model.prepare_sino_for_devices(sinogram)
    assert isinstance(prepared, _sharding.Shards)

    from_host = run_recon(model, sinogram)
    from_prepared = run_recon(model, prepared)
    assert max_relative_difference(from_prepared, from_host) <= 1e-4

    weights = 0.5 + np.random.RandomState(5).rand(
        *PAIR_CELL).astype(np.float32)
    prepared, prepared_weights = model.prepare_sino_for_devices(
        sinogram, weights=weights)
    assert isinstance(prepared_weights, _sharding.Shards)

    from_host = run_recon(model, sinogram, weights=weights)
    from_prepared = run_recon(model, prepared, weights=prepared_weights)
    assert max_relative_difference(from_prepared, from_host) <= 1e-4


def test_a_prepared_sinogram_carries_a_plug_and_play_loop():
    """The shape a Plug-and-Play loop actually has: prepare the sinogram once,
    then call prox_map repeatedly, each pass feeding back the volume the last
    one returned in the device form and skipping the initialization the first
    pass did.  The whole loop stays on the devices and matches the same loop
    run on host arrays."""
    model = model_on_devices(PAIR_CELL, 2)
    phantom = pair_phantom(model)
    sinogram = model.forward_project(phantom)

    def two_passes(sino):
        np.random.seed(0)
        first, _ = model.prox_map(0.5 * phantom, sino, sigma_prox=0.5,
                                  init_recon=phantom, max_iterations=1,
                                  stop_threshold_change_pct=0.0,
                                  logfile_path=None, print_logs=False,
                                  output_sharded=True)
        np.random.seed(0)
        second, _ = model.prox_map(first, sino, sigma_prox=0.5,
                                   init_recon=phantom,
                                   do_initialization=False, max_iterations=1,
                                   stop_threshold_change_pct=0.0,
                                   logfile_path=None, print_logs=False,
                                   output_sharded=True)
        return second

    on_host = two_passes(sinogram)
    on_devices = two_passes(model.prepare_sino_for_devices(sinogram))
    assert isinstance(on_devices, _sharding.Shards)
    assert max_relative_difference(on_devices.gather(),
                                   on_host.gather()) <= 1e-4


def test_a_prepared_sinogram_is_refused_after_the_layout_changes():
    """A prepared sinogram belongs to the layout it was prepared on.  Once the
    model is configured differently the reconstruction says so, naming both
    placements, rather than reconstructing from a division that no longer
    matches."""
    model = model_on_devices(PAIR_CELL, 2)
    sinogram = model.forward_project(pair_phantom(model))
    prepared = model.prepare_sino_for_devices(sinogram)

    model.configure_devices(devices=['cpu'])
    with pytest.raises(ValueError,
                       match='different device configuration') as excinfo:
        run_recon(model, prepared, max_iterations=1)
    message = str(excinfo.value)
    assert repr(prepared.placement) in message
    assert repr(model.sino_placement) in message


