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
    arithmetic they drive would be a different model's.
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
