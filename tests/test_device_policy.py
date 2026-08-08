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
from mbirtorch import _memory_ledger
from mbirtorch._memory_ledger import MemoryPreflightError

GB = 2 ** 30


def make_model(shape=(8, 6, 8), device='cpu', **kwargs):
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(shape, angles, **kwargs)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


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


# ── the selection rule ───────────────────────────────────────────────────────
def test_widening_picks_the_largest_count_that_fits(monkeypatch, unpinned):
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
        monkeypatch, unpinned):
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


def test_skip_memory_preflight_forces_a_doomed_run(monkeypatch, unpinned):
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
    as_automatic(model, 2)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    # A budget just over the two-device peak and under the one-device peak,
    # so the verdict turns on the margin alone.
    peak2 = model._build_memory_ledger(
        devices=[torch.device('cpu', i) for i in range(2)]).peak_bytes(0)
    peak1 = model._build_memory_ledger(devices=['cpu']).peak_bytes(0)
    budget = int(1.05 * peak2)
    assert budget < peak1                        # one device cannot rescue it
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes',
                        lambda d: budget)
    model.memory_preflight_margin = 0.15
    with pytest.raises(MemoryPreflightError):
        model._apply_device_policy()
    model.memory_preflight_margin = 0.02         # the user accepts less room
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
