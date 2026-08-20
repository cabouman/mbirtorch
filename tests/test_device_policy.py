"""The automatic device count and the preflight that gates it.

A CUDA model spreads a reconstruction across the devices that can hold their
share.  These tests pin the RULE rather than the hardware: the selection, the
validation, the pin, the fallback, and the readable failure all run on CPU,
by driving the policy with fabricated device lists and budgets.  The
multi-device VALUE gates live in test_sharding.py, which is where the seeded
n>1 parity patterns already are.
"""

import os

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _memory_ledger, _sharding, _widening_floors
from mbirtorch._memory_ledger import MemoryPreflightError

GB = 2 ** 30

# Sinogram shapes the speed floors are measured at, each named for its view
# count; the comment beside each is its sinogram element count.
CELL_512 = (512, 448, 384)          #    88,080,384
CELL_768 = (768, 672, 576)          #   297,271,296
CELL_1024 = (1024, 1008, 992)       # 1,023,934,464
CELL_128 = (128, 112, 96)           #     1,376,256
SPARSE_VIEW_CELL = (64, 448, 384)   #    11,010,048
THIN_VOLUME_CELL = (1024, 32, 768)  #    25,165,824


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
    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    # Every device has room, so the largest valid count wins.
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4


def test_widening_falls_back_past_a_device_another_process_is_using(
        monkeypatch, unpinned, no_speed_guard):
    """The heterogeneous case the design calls out.

    Per-device peaks SHRINK as the count grows, so no uniform budget can
    admit two devices and refuse four.  What refuses four is one device
    without room.  Every device in a candidate set must pass its own budget,
    so a busy `cuda:2` sends the rule back to two devices rather than letting
    it start a run only three devices could hold.
    """
    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    # Devices 2 and 3 are nearly full; 0 and 1 are free.
    monkeypatch.setattr(
        _memory_ledger, 'device_budget_bytes',
        lambda d: 1024 if (d.index or 0) >= 2 else 64 * GB)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2


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


def test_a_configured_cpu_model_is_untouched_by_the_policy(monkeypatch):
    """An explicit layout is never revisited, whatever the machine has."""
    model = make_model()                         # configures ['cpu']
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    assert model.device_layout_is_automatic is False
    assert model._apply_device_policy() is None
    assert model.sino_placement.n_devices == 1


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
    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '2')
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2      # not the 4 that would fit


def test_the_environment_pin_is_capped_by_what_is_visible(monkeypatch):
    model = make_model((16, 8, 16))
    as_automatic(model, 2)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '8')
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2


def test_a_malformed_pin_is_refused_rather_than_ignored(monkeypatch):
    """A typo in a nightly's environment must not silently restore the
    automatic behavior the pin exists to prevent."""
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', 'two')
    with pytest.raises(ValueError, match='positive integer'):
        _memory_ledger.pinned_device_count()
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '0')
    with pytest.raises(ValueError, match='positive integer'):
        _memory_ledger.pinned_device_count()
    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES')
    assert _memory_ledger.pinned_device_count() is None


def test_the_suite_pin_is_active(monkeypatch):
    """The conftest fixture is the suite's determinism guarantee, so the
    suite asserts it rather than trusting it."""
    assert os.environ.get('MBIRTORCH_NUM_DEVICES') == '1'
    assert _memory_ledger.pinned_device_count() == 1


def test_a_pinned_suite_keeps_a_recon_on_one_device(monkeypatch):
    """The end-to-end form of the determinism guarantee."""
    model = make_model()
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)
    sinogram = np.zeros((8, 6, 8), dtype=np.float32)
    sinogram[:, 3, 4] = 1.0
    np.random.seed(0)
    model.recon(sinogram, max_iterations=2)
    assert model.sino_placement.n_devices == 1


# ── the doomed run ───────────────────────────────────────────────────────────
def test_a_doomed_run_fails_before_allocating_anything(monkeypatch, unpinned):
    """No count fits, including one, so the answer to 'which count' is
    'none'.  That must arrive as a readable error rather than as a
    reconstruction that dies later inside the allocator."""
    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 1024)          # a kilobyte: nothing fits
    with pytest.raises(MemoryPreflightError) as excinfo:
        model._apply_device_policy()
    message = str(excinfo.value)
    assert 'more memory' in message
    assert 'dominant phase' in message
    assert 'shortfall' in message
    assert 'Device counts tried' in message
    assert 'closest' in message
    # The remedies a user can act on, including the one the band-reduce
    # analysis says matters most.
    assert 'back_project_slice_band' in message
    assert 'view_batch_size' in message
    assert 'skip_memory_preflight' in message


def test_the_doomed_run_error_reaches_recon(monkeypatch, unpinned):
    """The failure must arrive through the public entry, not only through the
    internal one."""
    model = make_model((16, 8, 16))
    as_automatic(model, 2)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', lambda d: 1024)
    sinogram = np.zeros((16, 8, 16), dtype=np.float32)
    sinogram[:, 4, 8] = 1.0
    np.random.seed(0)
    with pytest.raises(MemoryPreflightError):
        model.recon(sinogram, max_iterations=1)


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


def test_the_margin_is_tunable_without_disabling_the_preflight(monkeypatch,
                                                               unpinned):
    """A user near the boundary can lower the margin instead of turning the
    preflight off entirely."""
    # Big enough that the arrays dominate the ledger's fixed workspace term;
    # at a tiny shape both peaks are just that constant and cannot separate.
    model = make_model((128, 64, 128))
    # This shape reconstructs 16,384 pixels a slice, which the shipped forward
    # batch covers whole, so that transient does not shrink as devices are
    # added and the one- and two-device peaks barely separate.  Pinning the
    # batch to half a slice leaves the sharded arrays to dominate the
    # difference, which is what the budget below is built from.
    model.forward_project_pixel_batch = 8192
    as_automatic(model, 2)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    # A budget the two-device layout misses at the default margin and clears at
    # the lowered one, so the verdict turns on the margin alone.  One device
    # must miss it at the lowered margin too, or the search would answer with
    # n=1 -- the floors are in force here and price that count first.
    default_margin, accepted_margin = 0.15, 0.02
    budget_admitting_only(monkeypatch, model, fits=(2, accepted_margin),
                          misses=[(2, default_margin), (1, accepted_margin)])
    model.memory_preflight_margin = default_margin
    with pytest.raises(MemoryPreflightError):
        model._apply_device_policy()
    model.memory_preflight_margin = accepted_margin  # user accepts less room
    model._apply_device_policy()                 # now it fits
    assert model.sino_placement.n_devices == 2


# ── the constructor amendment ────────────────────────────────────────────────
def test_the_constructors_take_no_device_argument():
    """One API for the layout.  Every explicit device choice goes through
    configure_devices, so the constructors carry no device string to parse
    and 'automatic' means exactly 'configure_devices was never called'."""
    import inspect

    angles = np.linspace(0, np.pi, 8, endpoint=False)
    for factory in (lambda **kw: mbirtorch.ParallelBeamModel((8, 6, 8),
                                                             angles, **kw),
                    lambda **kw: mbirtorch.ConeBeamModel(
                        (8, 6, 8), angles, source_detector_dist=40,
                        source_iso_dist=40, **kw),
                    lambda **kw: mbirtorch.QGGMRFDenoiser((4, 6, 8), **kw)):
        with pytest.raises(TypeError, match='device'):
            factory(device='cpu')
    for cls in (mbirtorch.ParallelBeamModel, mbirtorch.ConeBeamModel,
                mbirtorch.QGGMRFDenoiser):
        assert 'device' not in inspect.signature(cls.__init__).parameters


def test_construction_resolves_no_device_and_builds_no_projectors():
    """Laziness, asserted rather than assumed.

    A caller who only inspects a model, or who is about to call
    configure_devices, must not pay device resolution or a projector build
    that the next line would throw away.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    assert model._torch_device is None
    assert model._projector_functions is None
    assert model._sino_placement is None
    # An explicit layout is installed without ever resolving 'auto'.
    model.configure_devices(devices=['cpu'])
    assert model._torch_device == torch.device('cpu')
    # Reading the device resolves it; reading the projectors builds them.
    fresh = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    assert fresh.torch_device is not None
    assert fresh._projector_functions is None
    assert fresh.projector_functions is not None


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


def test_configure_devices_is_the_only_door_to_a_backend():
    """cpu, mps and an indexed CUDA device all arrive the same way."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu'])
    assert model.torch_device == torch.device('cpu')
    assert model.device_layout_is_automatic is False
    assert model.sino_placement.n_devices == 1


# ── the geometry-specific remedy ─────────────────────────────────────────────
def test_a_cone_model_names_split_sino_recon_as_a_remedy():
    """It nearly doubles the feasible size at a fixed device count, so the
    message names it where it exists."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    cone = mbirtorch.ConeBeamModel((8, 6, 8), angles,
                                   source_detector_dist=40,
                                   source_iso_dist=40)
    parallel = make_model()
    assert any('split_sino_recon' in line
               for line in cone._memory_remedies())
    assert not parallel._memory_remedies()


# ── device inheritance through copy_ct_model ─────────────────────────────────
def test_copy_ct_model_inherits_an_explicit_device_choice():
    """A copy of a model whose devices the user set gets the same devices."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    model.configure_devices(devices=['cpu', 'cpu'])
    copy = mbirtorch.copy_ct_model(model, new_num_det_rows=4)
    assert copy.device_layout_is_automatic is False
    assert copy.sino_placement.devices == model.sino_placement.devices


def test_copy_ct_model_leaves_an_automatic_model_automatic():
    """A copy of a model with no explicit device choice chooses for itself."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    copy = mbirtorch.copy_ct_model(model, new_num_det_rows=4)
    assert copy.device_layout_is_automatic is True


# ── the widening speed floors ────────────────────────────────────────────────
# The guard is a SPEED rule laid over the capacity rule: below a measured
# floor, a device count is pushed behind every admitted count rather than
# removed, so capacity still wins when nothing admitted fits.  These tests use
# the sizes the floors were actually measured at.  They are also the guard's
# standing regression coverage: every nightly row is env-pinned, and a pin
# bypasses the guard, so nothing else exercises this ordering end to end.
def make_cone_model(shape, device='cpu'):
    angles = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(shape, angles,
                                    source_detector_dist=4.0 * shape[2],
                                    source_iso_dist=2.0 * shape[2])
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


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
    """The 128-class shape, where widening to four was measured 13x
    slower."""
    model = with_four_visible(monkeypatch, make_model(CELL_128))
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 1


def test_a_large_parallel_problem_still_takes_all_four_devices(monkeypatch,
                                                               unpinned):
    """The guard must not cost the case widening was built for: at the
    1024-class shape n=4 is admitted and capacity has room."""
    model = with_four_visible(monkeypatch, make_model(CELL_1024))
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4


def test_a_sparse_view_shape_chooses_one_device_despite_its_large_volume(
        monkeypatch, unpinned):
    """The shape that picked the metric.

    (64, 448, 384) has 11.0M sinogram elements but 66M recon voxels, so the
    two candidate size metrics point at sizes a full step apart.  Widening it
    to two devices was measured as a 1.87x REGRESSION, which is the verdict
    this encodes: the floors index on sinogram elements, and this shape is
    below the n=2 floor.
    """
    model = with_four_visible(monkeypatch, make_model(SPARSE_VIEW_CELL))
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 1


def test_a_thin_volume_shape_holds_at_one_device_too(monkeypatch, unpinned):
    """(1024, 32, 768): many views, few slices -- the shape where view-axis
    work was expected to make widening pay early.  It did not (n=2 measured
    1.16x slower), so the single sinogram-element metric is not too
    conservative here either."""
    model = with_four_visible(monkeypatch, make_model(THIN_VOLUME_CELL))
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 1


def test_a_large_cone_problem_admits_four_devices_and_two(monkeypatch,
                                                          unpinned):
    """At the 1024-class shape every cone count clears its floor.

    n=2 used to be a sentinel here -- a row with no measured admission size
    -- but the 2026-08-10 refresh found one and gave it a floor at the
    512-class shape, so nothing is held back at this size and n=3 rides in on
    the n=4 floor it inherits.
    """
    model = with_four_visible(monkeypatch, make_cone_model(CELL_1024))
    order, held = model._speed_ordered_candidates(4)
    assert order == [4, 3, 2, 1] and held == {}
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4

    # BETWEEN the two cone floors only the narrower one is admitted: the
    # 512-class shape is exactly at the n=2 floor and far below the n=4 one,
    # which n=3 inherits.
    between = with_four_visible(monkeypatch, make_cone_model(CELL_512))
    order, held = between._speed_ordered_candidates(4)
    assert order == [2, 1, 4, 3]
    assert set(held) == {3, 4}


def test_a_cone_problem_at_its_n2_floor_chooses_two_devices(monkeypatch,
                                                            unpinned):
    """The 512-class shape, where the refresh put cone's first finite floor.

    It is also the first case where the guard prefers a MIDDLE count: n=4 and
    n=3 sit below their floor while every device has ample room, so capacity
    alone would have taken four.  The count that wins is neither the widest
    nor one.
    """
    cone = with_four_visible(monkeypatch, make_cone_model(CELL_512))
    cone._apply_device_policy()
    assert cone.sino_placement.n_devices == 2

    # Parallel now holds at ONE device here: the channel-sorted forward
    # kernel (mg40, 2026-08-19) made the one-device forward fast enough
    # that two devices stopped paying at this size, so the parallel n=2
    # floor sits a class above cone's and the two geometries part ways at
    # this shape.
    parallel = with_four_visible(monkeypatch, make_model(CELL_512))
    parallel._apply_device_policy()
    assert parallel.sino_placement.n_devices == 1


# ── what turns the guard off ─────────────────────────────────────────────────
def test_an_explicit_configure_devices_ignores_the_speed_floors(monkeypatch,
                                                                 unpinned):
    """A count the caller named is not the library's to second-guess, at any
    size."""
    model = make_model(CELL_128)
    model.configure_devices(devices=['cpu'] * 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4


def test_the_environment_pin_ignores_the_speed_floors(monkeypatch):
    """The other pin mechanism, which reaches the policy by a different
    branch and must bypass the guard the same way."""
    model = with_four_visible(monkeypatch, make_model(CELL_128))
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '4')
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4
    assert model.device_choice_rejections == []


def test_the_guard_env_switch_restores_the_pure_capacity_order(monkeypatch,
                                                               unpinned):
    """The escape hatch, at a size the guard would otherwise hold down."""
    model = with_four_visible(monkeypatch, make_model(CELL_128))
    order, held = model._speed_ordered_candidates(4)
    assert order == [1, 4, 3, 2] and set(held) == {2, 3, 4}
    monkeypatch.setenv('MBIRTORCH_WIDENING_GUARD', '0')
    assert model._speed_ordered_candidates(4) == ([4, 3, 2, 1], {})
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4


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


def test_every_wider_count_the_floors_held_back_is_named_in_the_log(
        monkeypatch, unpinned):
    """The guard's commonest action reaches none of the counts it excluded.

    Holding a 128-class problem at one device leaves three GPUs idle without
    the search ever pricing them, so the counts are named once the choice is
    made.  Idle hardware is never silent, and each entry carries the override
    the user can reach for.
    """
    model = with_four_visible(monkeypatch, make_model(CELL_128))
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 1

    reasons = dict(model.device_choice_rejections)
    assert sorted(reasons) == [2, 3, 4]
    assert [count for count, _why in model.device_choice_rejections] == \
        [4, 3, 2]                                # widest first, as elsewhere
    for count in (2, 3, 4):
        assert 'held by the speed floor' in reasons[count]
        assert '1.4M sinogram elements <' in reasons[count]
        assert 'configure_devices(num_devices={}) overrides'.format(count) \
            in reasons[count]
    assert 'the parallel n=2 floor' in reasons[2]
    assert 'the parallel n=4 floor, which n=3 inherits' in reasons[3]
    assert 'the parallel n=4 floor' in reasons[4]

    line = model._device_report()
    assert 'using 1 of 4 CPU devices' in line
    assert '4 rejected, held by the speed floor' in line


def with_a_sentinel_below_a_reachable_floor(monkeypatch, model):
    """Install a synthetic floor table where n=2 is a SENTINEL -- a row with
    no measured admission size -- while n=4 has a floor the problem clears.

    No SHIPPED family can produce that shape: since the 2026-08-10 refresh
    every measured family's floors rise with the count, so a held count is
    always wider than an admitted one.  The reporting rule outlives the data,
    which is why it is exercised against a table built here.
    """
    wide = _widening_floors.FLOORS[('cone', 4)]
    monkeypatch.setattr(_widening_floors, 'FLOORS', {
        ('synthetic', 2): wide._replace(
            family='synthetic', count=2, elements=None, cell=None, against=1,
            bracket=_widening_floors.Bracket(
                losing_cell=CELL_1024, losing_speedup=0.92,
                winning_cell=None, winning_speedup=None),
            note='synthetic sentinel'),
        ('synthetic', 4): wide._replace(family='synthetic'),
    })
    model._floor_family = 'synthetic'
    return model


def test_a_held_count_narrower_than_the_chosen_one_is_not_reported(
        monkeypatch, unpinned):
    """A held count BELOW the count in use was outranked, not excluded.

    The synthetic table holds n=2 as a sentinel while admitting n=4 at this
    size.  The run takes four devices, so reporting n=2 as turned down would
    explain an idleness that never happened.
    """
    model = with_a_sentinel_below_a_reachable_floor(
        monkeypatch, with_four_visible(monkeypatch,
                                       make_cone_model(CELL_1024)))
    order, held = model._speed_ordered_candidates(4)
    assert order == [4, 3, 1, 2] and set(held) == {2}
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4
    assert model.device_choice_rejections == []


def test_skipping_the_memory_preflight_leaves_the_speed_floors_in_force(
        monkeypatch, unpinned):
    """The flag forces past the CAPACITY check; the guard is a speed rule.

    With the reorder this is automatic rather than special-cased: the flag
    settles on the first candidate, and the first candidate is now the first
    ADMITTED count rather than the widest one.
    """
    model = with_four_visible(monkeypatch, make_model(CELL_128), budget=1024)
    model.skip_memory_preflight = True
    model._apply_device_policy()                 # does not raise
    assert model.sino_placement.n_devices == 1


# ── geometries the floors have never met ─────────────────────────────────────
class _UnlistedGeometry(mbirtorch.ParallelBeamModel):
    """A stand-in that declares no floor family.

    The two real classes below are the standing coverage for this path, and
    this one is kept beside them for the same reason
    test_widening_floors.py keeps a synthetic table: the RULE has to outlive
    the data.  Once a refresh measures multiaxis and translation they will
    declare families of their own and stop exercising the fallback, and the
    fallback still has to work for whatever geometry arrives next.
    """

    _floor_family = None


def _synthetic_no_family(shape):
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    return _UnlistedGeometry(shape, angles)


def _multiaxis_model(shape):
    """Multiaxis angles are (azimuth, elevation) pairs, one row per view.
    Since 2026-08-17 this class declares its own floor family, so it no
    longer belongs in the borrowed-floors lists below."""
    azimuth = np.linspace(0, np.pi, shape[0], endpoint=False)
    elevation = np.linspace(-0.4, 0.4, shape[0])
    return mbirtorch.MultiAxisParallelModel(shape, np.stack([azimuth, elevation], axis=1))


# Translation left this list on 2026-08-17, when the mg22 refresh gave it a
# measured family of its own; the synthetic class is now the one resident of
# the newly ported, no-family state.
UNMEASURED_GEOMETRIES = [
    (_synthetic_no_family, '_UnlistedGeometry'),
]


def _built(make, shape, verbose=0):
    model = make(shape)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=verbose)
    return model


@pytest.mark.parametrize("make,class_name", UNMEASURED_GEOMETRIES,
                         ids=[name for _make, name in UNMEASURED_GEOMETRIES])
def test_a_model_with_no_floor_family_gets_the_parallel_floors(
        monkeypatch, unpinned, make, class_name):
    """Every class that declares no floor family is governed by the parallel
    floors.  Since 2026-08-17 every shipped geometry declares a measured
    family, so the synthetic stand-in is what arrives in that state."""
    model = make(CELL_128)
    assert type(model).__name__ == class_name
    assert model._floor_family is None

    small = with_four_visible(monkeypatch, _built(make, CELL_128))
    small._apply_device_policy()
    assert small.sino_placement.n_devices == 1

    # The parallel floors, not a refusal: at the parallel n=2 floor (the
    # 768-class since the sorted forward kernel, mg40) it widens.
    at_the_floor = with_four_visible(monkeypatch, _built(make, CELL_768))
    at_the_floor._apply_device_policy()
    assert at_the_floor.sino_placement.n_devices == 2


@pytest.mark.parametrize("make,class_name", UNMEASURED_GEOMETRIES,
                         ids=[name for _make, name in UNMEASURED_GEOMETRIES])
def test_the_substituted_family_is_named_in_the_log(monkeypatch, unpinned,
                                                    caplog, make, class_name):
    """A geometry that was never measured must not have that fact hidden from
    it, so the selection path says which class borrowed which floors."""
    model = with_four_visible(monkeypatch, _built(make, CELL_128, verbose=2))
    with caplog.at_level('DEBUG', logger=model.logger.name):
        model._apply_device_policy()
    assert 'names no _floor_family' in caplog.text
    assert 'parallel widening speed floors' in caplog.text
    # The line names the class, so a log read months later says which
    # geometry was running on borrowed numbers.
    assert class_name in caplog.text


# ── the split_sino_recon halves ──────────────────────────────────────────────
def test_a_split_sino_half_model_is_governed_by_its_own_half_size(monkeypatch,
                                                                  unpinned):
    """split_sino_recon builds each half with copy_ct_model, and since the
    2026-08 prerelease change a half inherits no explicit layout: it lands on
    the automatic branch and chooses for itself.

    So the floors see the HALF's sinogram, not the parent's, which is the
    right question -- the half is the reconstruction that actually runs.
    """
    def automatic_cone_half(parent_shape, half_rows):
        # No configure_devices anywhere: a parent that placed itself by hand
        # would hand the half an EXPLICIT layout, which bypasses the guard.
        angles = np.linspace(0, 2 * np.pi, parent_shape[0], endpoint=False)
        parent = mbirtorch.ConeBeamModel(
            parent_shape, angles, source_detector_dist=4.0 * parent_shape[2],
            source_iso_dist=2.0 * parent_shape[2])
        parent.set_params(no_warning=True, verbose=0)
        half = mbirtorch.copy_ct_model(parent, new_num_det_rows=half_rows)
        assert half.device_layout_is_automatic is True
        # Place it on the CPU the way the automatic path itself does, which
        # by design does NOT set device_layout_is_automatic: the half stays
        # the library's to choose for, while the fabricated CUDA visibility
        # below cannot pull real allocations onto a device this host lacks.
        half._install_device_layout(['cpu'])
        assert half.device_layout_is_automatic is True
        return half

    half = automatic_cone_half((128, 224, 96), 112)
    assert tuple(half.get_params('sinogram_shape')) == CELL_128
    with_four_visible(monkeypatch, half)
    half._apply_device_policy()
    assert half.sino_placement.n_devices == 1

    big_half = automatic_cone_half((1024, 2016, 992), 1008)
    assert tuple(big_half.get_params('sinogram_shape')) == CELL_1024
    with_four_visible(monkeypatch, big_half)
    big_half._apply_device_policy()
    assert big_half.sino_placement.n_devices == 4

    # The size the guard was asked about is the HALF's, not the parent's:
    # the parent's sinogram is twice as large, and a floor read off it would
    # be answering a question no reconstruction asks.
    assert _widening_floors.sinogram_elements(
        big_half.get_params('sinogram_shape')) == 1_023_934_464


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


def test_the_device_line_does_not_call_the_count_it_is_using_rejected(
        monkeypatch, unpinned):
    """The fallback note names the count actually in use, and that count
    appears on a line whose other entries are refusals."""
    model = make_model((128, 64, 128))
    # This shape reconstructs 16,384 pixels a slice, which the shipped forward
    # batch covers whole, so that transient does not shrink as devices are
    # added and the one- and four-device peaks barely separate.  Pinning the
    # batch to half a slice leaves the sharded arrays to dominate the
    # difference, which is what the budget below is built from.
    model.forward_project_pixel_batch = 8192
    with_four_visible(monkeypatch, model)
    margin = 0.02
    budget_admitting_only(monkeypatch, model, fits=(4, margin),
                          misses=[(1, margin)])
    model.memory_preflight_margin = margin
    model._apply_device_policy()

    line = model._device_report()
    assert '4 used, chosen past its speed floor' in line
    assert '1 rejected,' in line
    assert '4 rejected' not in line


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


def test_a_nonshape_recompile_param_keeps_the_settled_layout(
        monkeypatch, unpinned, no_speed_guard):
    """A recompile-flagged parameter that leaves the shapes alone must not
    re-decide.  Re-deciding on every recompile would unsettle the layout on
    a detector-offset edit, and on the sigma_noise the denoiser sets at
    every call."""
    model = make_model((16, 8, 16))
    with_four_visible(monkeypatch, model)
    model._apply_device_policy()
    poison_budgets(monkeypatch)
    model.set_params(no_warning=True, det_channel_offset=0.5)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4


def test_configure_devices_overrides_a_settled_layout(monkeypatch, unpinned,
                                                      no_speed_guard):
    model = make_model((16, 8, 16))
    with_four_visible(monkeypatch, model)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4
    model.configure_devices(devices=['cpu'])
    assert model._settled_shapes is None
    poison_budgets(monkeypatch)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 1


# ── the direct reconstructions ───────────────────────────────────────────────
# fbp_recon and fdk_recon settle the layout themselves, so a direct
# reconstruction spreads across the devices instead of landing whole on the
# lead one.  All four geometries are covered here: cone has made the call
# since commit 72208bb and the other three gained it with this increment.
def _automatic_parallel(shape=(8, 6, 8)):
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    return mbirtorch.ParallelBeamModel(shape, angles)


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
    (_automatic_parallel, 'fbp_recon'),
    (_automatic_cone, 'fdk_recon'),
    # The multiaxis constructor is the floors section's, at a toy shape.
    (lambda: _multiaxis_model((8, 6, 8)), 'fbp_recon'),
    (_automatic_translation, 'fdk_recon'),
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
def test_a_bare_direct_recon_settles_the_layout_and_spreads(
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


@pytest.mark.parametrize("make,method", DIRECT_RECONS[3:],
                         ids=DIRECT_RECON_IDS[3:])
def test_translation_takes_its_own_floors_into_its_direct_recon(
        monkeypatch, unpinned, caplog, make, method):
    """Translation declares its own measured family (2026-08-17, mg22), so
    its rows govern the count its direct reconstruction settles on.  Since
    the mg48 refresh (2026-08-20) both rows are finite floors -- n=2
    admits at 364.8 million sinogram elements and n=4 at the production
    cell -- and this test's sinogram sits far below both, so the
    reconstruction holds at one device although four are free, and the
    log names the family's own floor rather than a borrowed one.
    """
    model = _automatic_on_cpu(make, verbose=2)
    with_four_visible(monkeypatch, model)
    assert model._floor_family == 'translation'
    with caplog.at_level('DEBUG', logger=model.logger.name):
        getattr(model, method)(_impulse_sinogram(model))
    assert model.sino_placement.n_devices == 1
    assert 'translation' in caplog.text
    assert 'held by the speed floor' in caplog.text
    assert 'names no _floor_family' not in caplog.text


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


def test_a_geometry_too_large_for_a_recon_still_runs_a_direct_recon(
        monkeypatch, unpinned, no_speed_guard):
    """The cost §2.3 records, removed.  Every device count is short for a full
    recon here, and the direct reconstruction that was going to run is not
    refused for it -- it runs, on the widest count that holds it."""
    model = model_between_the_two_plans(monkeypatch)
    recon = model.fbp_recon(_impulse_sinogram(model))
    assert model.sino_placement.n_devices == 4
    assert np.all(np.isfinite(recon))
    # The count in use is not reported as a rejection, and the line says why
    # it was taken.
    reasons = dict(model.device_choice_rejections)
    assert 'chosen for the direct reconstruction in progress' in reasons[4]
    assert '4 used, chosen for the direct reconstruction' in \
        model._device_report()


def test_a_recon_on_that_same_model_is_refused_by_the_preflight(
        monkeypatch, unpinned, no_speed_guard):
    """The other half of the rule.  The layout settled under the narrower
    check, so the recon that does not fit it must be refused HERE, with the
    message and the remedies, rather than reaching the allocator."""
    model = model_between_the_two_plans(monkeypatch)
    model._apply_device_policy(workload='direct')     # what fbp_recon does
    assert model.sino_placement.n_devices == 4
    assert model._settled_workload == 'direct'

    with pytest.raises(MemoryPreflightError) as excinfo:
        model.recon(_impulse_sinogram(model), max_iterations=1)
    message = str(excinfo.value)
    assert 'more memory' in message
    assert 'dominant phase' in message
    assert 'skip_memory_preflight' in message


def test_a_direct_recon_on_a_recon_settled_model_repeats_no_check(
        monkeypatch, unpinned, no_speed_guard):
    """The nested direct reconstruction inside vcd_recon arrives here on every
    run.  The recon plan charges everything the direct plan charges, so the
    settled layout has already been checked for it and the poisoned budget
    must not be consulted."""
    model = make_model((16, 8, 16))
    with_four_visible(monkeypatch, model)
    model._apply_device_policy()
    assert model._settled_workload == 'recon'
    poison_budgets(monkeypatch)
    model._apply_device_policy(workload='direct')
    assert model.sino_placement.n_devices == 4


def test_a_recon_after_a_direct_settle_rechecks_once_and_then_stops(
        monkeypatch, unpinned, no_speed_guard):
    """The narrowed check costs one extra preflight, not one per call.

    A layout settled under a direct reconstruction was checked against what
    that reconstruction allocates, so the first recon re-runs the check on it.
    The layout does not move -- the same placement object survives, so shards
    a caller holds stay valid -- and passing records the recon as the workload
    the layout is known to hold, which the poisoned budget then proves.
    """
    model = model_between_the_two_plans(monkeypatch)
    model._apply_device_policy(workload='direct')
    placement = model.sino_placement
    assert model._settled_workload == 'direct'

    # Room appears (the neighbor that was using the GPUs has exited).
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: 64 * GB)
    model._apply_device_policy()
    assert model.sino_placement is placement
    assert model._settled_workload == 'recon'
    poison_budgets(monkeypatch)
    model._apply_device_policy()


def test_configure_devices_clears_the_settled_workload_too(
        monkeypatch, unpinned, no_speed_guard):
    """A model the caller has placed carries no settled record of either kind.
    The explicit branch reads neither, so a record left behind would only be a
    state that can disagree with the layout in use."""
    model = make_model((16, 8, 16))
    with_four_visible(monkeypatch, model)
    model._apply_device_policy()
    assert model._settled_workload == 'recon'
    model.configure_devices(devices=['cpu'])
    assert model._settled_shapes is None
    assert model._settled_workload is None


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
    the devices that fit, rather than running whole on the lead one."""
    built = capture_generation_model(monkeypatch, 'ParallelBeamModel')
    _phantom, sinogram, _params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='cube', **DEMO_SHAPE)
    model, = built
    assert model.device_layout_is_automatic is True
    assert model.sino_placement.n_devices == 4
    assert sinogram.shape == (8, 8, 12)
    assert np.all(np.isfinite(sinogram)) and sinogram.max() > 0


def test_a_requested_device_list_still_pins_the_generation(monkeypatch,
                                                           unpinned,
                                                           no_speed_guard):
    """devices= stays an explicit pin.  Four devices are visible with ample
    room, so the automatic path would have taken all four; the projection runs
    on the two the caller named."""
    built = capture_generation_model(monkeypatch, 'ParallelBeamModel')
    _phantom, sinogram, _params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='cube',
        devices=['cpu', 'cpu'], **DEMO_SHAPE)
    model, = built
    assert model.device_layout_is_automatic is False
    assert model.sino_placement.n_devices == 2
    assert np.all(np.isfinite(sinogram)) and sinogram.max() > 0


def test_the_generation_settles_with_the_capacity_check_skipped(
        monkeypatch, unpinned, no_speed_guard):
    """The skip itself.  Every device has a kilobyte, which no plan fits, and
    the generation runs anyway on the first count the candidate order offers.
    """
    built = capture_generation_model(monkeypatch, 'ParallelBeamModel',
                                     budget=1024)
    _phantom, sinogram, _params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='cube', **DEMO_SHAPE)
    model, = built
    assert model.skip_memory_preflight is True
    assert model.sino_placement.n_devices == 4
    assert np.all(np.isfinite(sinogram))
    # The check was skipped, not passed: with the skip taken away and the
    # settled record cleared, the same model at the same budget is refused.
    model.skip_memory_preflight = False
    model._settled_shapes = None
    with pytest.raises(MemoryPreflightError):
        model._apply_device_policy()


# ── the full-array allocators ────────────────────────────────────────────────
# Three helpers allocate a whole sinogram or a whole volume before any
# reconstruction runs: compute_hessian_diagonal, prepare_sino_for_devices, and
# gen_weights_mar on the branch that forward projects.  Each settles the layout
# first, so its arrays land on the devices the later reconstructions use rather
# than whole on the lead one.
#
# Settling is what first hands a placed array to the entries below it, so those
# entries had to learn about the device form before the settle was added.
# gen_weights, split_sino_recon, and recon_plastic_metal all refuse one: each
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


def test_gen_weights_refuses_a_sinogram_that_is_already_placed():
    """The silent wrong answer this replaces: a placed sinogram is neither
    numpy nor a tensor, so the array module resolved to numpy and 'unweighted'
    came back as a zero-dimensional object array.  The message names the order
    that works instead."""
    model, sinogram = _placed_cone_case()
    prepared = model.prepare_sino_for_devices(sinogram)
    with pytest.raises(ValueError) as excinfo:
        mbirtorch.gen_weights(prepared, weight_type='unweighted')
    message = str(excinfo.value)
    assert 'placed on the devices' in message
    assert 'host sinogram first' in message
    assert 'prepare_sino_for_devices(sinogram, weights)' in message


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


def test_split_sino_recon_refuses_a_placed_sinogram():
    """Each half builds its own model and settles its own device layout, so
    this method has no use for the parent's device form.  Gathering the input
    would leave the caller's placed copy resident for the whole call, which is
    the memory the split exists to save, so the input is refused instead.  A
    host-array call is covered in test_split_sino.py."""
    model, sinogram = _placed_cone_case()
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission')
    prepared, placed_weights = model.prepare_sino_for_devices(sinogram,
                                                              weights=weights)
    with pytest.raises(ValueError) as excinfo:
        model.split_sino_recon(prepared, weights=weights, half_overlap=3)
    message = str(excinfo.value)
    assert 'does not accept' in message
    assert 'sharded form' in message
    # The weights are checked with the sinogram, so a host sinogram carrying
    # placed weights is refused too.
    with pytest.raises(ValueError, match='sharded form'):
        model.split_sino_recon(sinogram, weights=placed_weights, half_overlap=3)


def test_recon_plastic_metal_refuses_a_placed_sinogram(monkeypatch):
    """This driver applies its corrections on the host and hands a host
    sinogram to every reconstruction pass, so it has no use for the device
    form either.  The check goes in front of np.asarray, which would build an
    OBJECT array from the device form rather than fail.  The tensor coercion
    it replaces is unchanged, which the last block here checks."""
    model, sinogram = _placed_cone_case()
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission')
    prepared, placed_weights = model.prepare_sino_for_devices(sinogram,
                                                              weights=weights)
    with pytest.raises(ValueError) as excinfo:
        model.recon_plastic_metal(prepared, weights, num_metal=0)
    message = str(excinfo.value)
    assert 'in sharded form' in message
    assert 'Pass the host (numpy or tensor) sinogram and the host weights' \
        in message
    with pytest.raises(ValueError, match='in sharded form'):
        model.recon_plastic_metal(sinogram, placed_weights, num_metal=0)

    # A plain tensor is still converted to host numpy at entry.  The
    # reconstruction pass is stubbed out, so what it was handed is the whole
    # assertion and no reconstruction runs.
    seen = {}

    def record_and_return_zeros(sino, weights=None, **kwargs):
        seen['sino'], seen['weights'] = sino, weights
        return np.zeros(tuple(model.get_params('recon_shape')),
                        dtype=np.float32), {}

    monkeypatch.setattr(model, 'split_sino_recon', record_and_return_zeros)
    recon, _recon_dict = model.recon_plastic_metal(torch.as_tensor(sinogram),
                                                   torch.as_tensor(weights),
                                                   num_metal=0)
    assert isinstance(seen['sino'], np.ndarray)
    assert isinstance(seen['weights'], np.ndarray)
    assert np.array_equal(seen['sino'], sinogram)
    assert np.array_equal(seen['weights'], weights)
    assert isinstance(recon, np.ndarray)


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


def test_a_pin_after_a_settle_invalidates_the_prepared_sinogram(
        monkeypatch, unpinned, no_speed_guard):
    """The configure_devices docstring's new sentence, checked on the helper
    that returns the array most likely to be held across such a call.  The
    placement identity check is what reports it."""
    model = _automatic_on_cpu(lambda: _automatic_parallel((16, 8, 16)))
    with_four_visible(monkeypatch, model)
    prepared = model.prepare_sino_for_devices(_impulse_sinogram(model))
    assert model.sino_placement.n_devices == 4
    model.configure_devices(devices=['cpu'])
    with pytest.raises(ValueError, match='different device configuration'):
        model._shard_sinogram(prepared)


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


def test_the_otsu_branch_of_gen_weights_mar_does_not_settle(
        monkeypatch, unpinned, no_speed_guard):
    """Without init_recon the function thresholds the sinogram on the host and
    never projects, so there is no allocation to settle for.  The poisoned
    budget is the proof: a settle here would consult one and raise."""
    model = _automatic_on_cpu(lambda: _automatic_parallel((16, 8, 16)))
    with_four_visible(monkeypatch, model)
    sinogram, _init_recon = _mar_inputs(model)
    poison_budgets(monkeypatch)
    weights = mbirtorch.gen_weights_mar(model, sinogram)
    assert model.sino_placement.n_devices == 1
    assert model._settled_shapes is None
    assert weights.shape == sinogram.shape
    assert np.all(np.isfinite(weights))


# ── the calibration scope ────────────────────────────────────────────────────
def test_calibration_resets_once_per_reconstruction(monkeypatch):
    """The counters reset where the report that reads them lives, so the
    nested direct_recon inside a cone reconstruction cannot clear the peak
    mid-run."""
    resets = []
    monkeypatch.setenv('MBIRTORCH_MEMORY_CALIBRATION', '1')
    monkeypatch.setattr(_memory_ledger, 'calibration_start',
                        lambda devices: resets.append(list(devices)))
    model = make_cone_model((8, 6, 8))
    sinogram = np.zeros((8, 6, 8), dtype=np.float32)
    sinogram[:, 3, 4] = 1.0
    np.random.seed(0)
    model.recon(sinogram, max_iterations=2)
    assert len(resets) == 1


def test_a_standalone_direct_recon_does_not_open_a_calibration_scope(
        monkeypatch):
    """A direct reconstruction before the recon must neither reset the
    counters itself nor suppress the recon's own reset."""
    resets = []
    monkeypatch.setenv('MBIRTORCH_MEMORY_CALIBRATION', '1')
    monkeypatch.setattr(_memory_ledger, 'calibration_start',
                        lambda devices: resets.append(list(devices)))
    model = make_cone_model((8, 6, 8))
    sinogram = np.zeros((8, 6, 8), dtype=np.float32)
    sinogram[:, 3, 4] = 1.0
    model.direct_recon(sinogram)
    assert resets == []
    np.random.seed(0)
    model.recon(sinogram, max_iterations=2)
    assert len(resets) == 1


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


def test_denoise_prices_the_denoiser_plan_not_a_recon(monkeypatch, unpinned,
                                                      no_speed_guard):
    """The plan the search priced is the denoiser's own.  A recon plan on a
    model with no projection bodies raises before it prices anything, so a
    call that completes at all is the claim; the settled record and the phase
    names name the plan it completed with."""
    denoiser = automatic_denoiser(monkeypatch)
    run_denoise(denoiser, noisy_image())
    assert denoiser._settled_workload == 'denoise'
    assert denoiser._settled_shapes == (DENOISE_CELL, DENOISE_CELL)
    assert denoiser.last_memory_ledger is not None
    assert all(phase.name.startswith('denoise')
               for phase in denoiser.last_memory_ledger.phases)


def test_a_sigma_change_does_not_unsettle_the_denoiser(monkeypatch, unpinned,
                                                       no_speed_guard):
    """sigma_noise is recompile-flagged and set on every denoise call, and it
    leaves the shapes the settled decision came from alone.  The budgets below
    fit nothing at any count, so a second search would refuse the run rather
    than complete it on three devices."""
    denoiser = automatic_denoiser(monkeypatch)
    image = noisy_image()
    run_denoise(denoiser, image)
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', lambda d: 1024)
    run_denoise(denoiser, image, sigma_noise=0.2)
    assert denoiser.recon_placement.n_devices == 3


def test_configure_devices_still_wins_after_a_denoise_settle(
        monkeypatch, unpinned, no_speed_guard):
    """An explicit layout overrides a settled automatic one on a denoiser as
    it does on a reconstruction: three devices were free and the call runs on
    the one the caller named."""
    denoiser = automatic_denoiser(monkeypatch)
    image = noisy_image()
    run_denoise(denoiser, image)
    assert denoiser.recon_placement.n_devices == 3
    denoiser.configure_devices(devices=['cpu'])
    run_denoise(denoiser, image)
    assert denoiser.recon_placement.n_devices == 1
    assert denoiser.device_layout_is_automatic is False


def test_calibration_mode_prices_a_denoiser_with_its_own_plan(monkeypatch):
    """The calibration mode builds a ledger where the policy would otherwise
    build none, and on an explicitly placed denoiser that build has to be the
    denoise plan: a recon plan raises for want of a projection body."""
    monkeypatch.setenv('MBIRTORCH_MEMORY_CALIBRATION', '1')
    shape = (6, 8, 10)
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=['cpu', 'cpu'])
    denoiser.set_params(no_warning=True, verbose=0)
    run_denoise(denoiser, noisy_image(shape), max_iterations=1)
    assert denoiser.last_memory_ledger is not None
    assert all(phase.name.startswith('denoise')
               for phase in denoiser.last_memory_ledger.phases)


def test_the_denoiser_sentinel_floors_hold_an_automatic_denoiser_at_one_device(
        monkeypatch, unpinned):
    """The denoiser's measured rows are both sentinels: sharded denoising
    lost at every size probed up to a billion image voxels, so however free
    the devices are, the automatic path keeps a denoiser on one device and
    the run log says why.  Capacity is the one thing that overrides a
    sentinel, and that rule's own tests drive it with a synthetic table."""
    refused, why = _widening_floors.admitted('denoiser', 2, 88_080_384)[0], \
        _widening_floors.admitted('denoiser', 2, 88_080_384)[1]
    assert refused is False and 'sentinel' in why

    denoiser = mbirtorch.QGGMRFDenoiser(CELL_512)
    denoiser.set_params(no_warning=True, verbose=0)
    with_four_visible(monkeypatch, denoiser)
    denoiser._apply_device_policy(workload='denoise')
    assert denoiser.recon_placement.n_devices == 1
    reasons = dict(denoiser.device_choice_rejections)
    assert 'sentinel' in reasons[4] and 'sentinel' in reasons[2]


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
    reconstruction model placed, with no gather to the host in between.
    """
    ct_model, denoiser = paired_ct_and_denoiser()
    assert denoiser.recon_placement == ct_model.recon_placement
    assert denoiser.recon_placement is not ct_model.recon_placement

    recon_shape = tuple(ct_model.get_params('recon_shape'))
    volume = np.random.RandomState(2).rand(*recon_shape).astype(np.float32)
    placed = ct_model._shard_recon(volume)
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


def test_a_sharded_denoiser_output_goes_back_into_prox_map():
    """The return leg: what denoise hands back in the device form is a valid
    prox_input for the reconstruction model, so a loop closes without either
    half of it touching host memory."""
    ct_model, denoiser = paired_ct_and_denoiser()
    phantom = pair_phantom(ct_model)
    sinogram = ct_model.forward_project(phantom)

    denoised, _ = denoiser.denoise(ct_model._shard_recon(phantom),
                                   sigma_noise=0.1, max_iterations=1,
                                   stop_threshold_change_pct=0.0,
                                   logfile_path=None, print_logs=False,
                                   output_sharded=True)
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


def test_a_flat_sharded_prox_input_is_accepted_too():
    """A prox input already in the loop's flat (num_pixels, local_slices)
    form is accepted alongside the 3-D form denoise returns, so a caller who
    kept the flat shards does not have to reshape them back first."""
    ct_model, _ = paired_ct_and_denoiser()
    phantom = pair_phantom(ct_model)
    sinogram = ct_model.forward_project(phantom)
    recon_shape = tuple(ct_model.get_params('recon_shape'))

    volume = ct_model._shard_recon(0.5 * phantom)
    flat = _sharding.Shards(
        [t.reshape(recon_shape[0] * recon_shape[1], t.shape[-1])
         for t in volume.tensors], volume.placement)
    np.random.seed(0)
    recon, _ = ct_model.prox_map(flat, sinogram, sigma_prox=0.5,
                                 init_recon=phantom, max_iterations=1,
                                 stop_threshold_change_pct=0.0,
                                 logfile_path=None, print_logs=False,
                                 output_sharded=True)
    assert isinstance(recon, _sharding.Shards)
    assert np.isfinite(recon.gather()).all()


def test_like_copies_the_device_list_and_pins_the_layout():
    """like= takes the other model's devices verbatim, and it is an explicit
    configuration like any other: the automatic choice is off afterwards."""
    ct_model, denoiser = paired_ct_and_denoiser(devices=('cpu', 'cpu'))
    assert ([str(d) for d in denoiser.recon_placement.devices]
            == [str(d) for d in ct_model.recon_placement.devices])
    assert denoiser.device_layout_is_automatic is False
    # The denoiser's own arrays are divided the same way, 8 slices as 4 + 4.
    assert [e - s for _d, (s, e) in denoiser.recon_placement.shard_ranges()] \
        == [e - s for _d, (s, e) in ct_model.recon_placement.shard_ranges()]


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


def test_like_and_devices_together_are_refused():
    """The two arguments express different intents, so giving both is a
    question about which one was meant rather than a precedence rule."""
    ct_model, denoiser = paired_ct_and_denoiser()
    with pytest.raises(ValueError, match='not both'):
        denoiser.configure_devices(like=ct_model, devices=['cpu'])
    # The layout the refused call would have changed is left alone.
    assert denoiser.recon_placement == ct_model.recon_placement


def test_like_needs_a_model_to_copy_from():
    """An object that is not a placed model is reported as that, rather than
    as an attribute error from inside the configuration."""
    _ct_model, denoiser = paired_ct_and_denoiser()
    with pytest.raises(ValueError, match='no device placement to copy'):
        denoiser.configure_devices(like=object())


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


def test_shards_from_a_differently_placed_model_are_still_refused():
    """Value equality widens what is accepted; it does not remove the check.
    A volume from a model on a different device layout is refused, and the
    message shows both placements so the mismatch is readable."""
    ct_model, _ = paired_ct_and_denoiser()
    placed = ct_model._shard_recon(pair_phantom(ct_model))

    other = mbirtorch.QGGMRFDenoiser(ct_model.get_params('recon_shape'))
    other.configure_devices(devices=['cpu'])
    with pytest.raises(ValueError, match='different device configuration') as e:
        other._shard_recon(placed)
    message = str(e.value)
    assert repr(ct_model.recon_placement) in message
    assert repr(other.recon_placement) in message


def test_like_refuses_a_transverse_shape_mismatch():
    """Two models whose recon shapes agree on the slice count but differ in
    the rows and columns are refused as well.

    Their placements would be equal -- the slice axis divides identically --
    so nothing downstream would notice until a volume from one was reshaped
    for the other.  The check is on the whole recon shape for that reason.
    """
    angles = np.linspace(0, np.pi, PAIR_CELL[0], endpoint=False)
    ct_model = mbirtorch.ParallelBeamModel(PAIR_CELL, angles)
    ct_model.configure_devices(devices=['cpu', 'cpu'])
    recon_shape = tuple(ct_model.get_params('recon_shape'))
    wider = (recon_shape[0] + 4, recon_shape[1] + 4, recon_shape[2])
    denoiser = mbirtorch.QGGMRFDenoiser(wider)

    with pytest.raises(ValueError, match='recon shape') as excinfo:
        denoiser.configure_devices(like=ct_model)
    message = str(excinfo.value)
    assert str(wider) in message and str(recon_shape) in message
