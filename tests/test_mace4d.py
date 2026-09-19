"""Tests for the 4D reconstruction model and the frame-axis filter.

The filter matrix is checked against the transform it replaces.  The model
is checked on a 24-view cone-beam scan with a smooth sinogram.  A run with
three frames on one device gives finite values, and a four-frame run with
the filter on differs from the same run with it off.  A run with two
workers on the CPU equals the one-worker run and shows tasks from both
workers.  The compile check runs a reconstruction in a fresh process, because
torch keeps its variant counters per function for the whole process; it
asserts that no compiled body falls back and no function exceeds the
recompile budget.  Two worker threads on one MPS device crash inside Metal,
so the two-worker runs are on the CPU only.  The smooth sinogram is kept
because a random one reconstructs to a volume with extreme values, on which
the qGGMRF line search computes zero over zero.
"""

import csv
import math
import os

import numpy as np
import pytest
import torch
from scipy.fft import dct, idct

import mbirtorch
from mbirtorch import mace as mace_module
from mbirtorch.mace import MACE
from mbirtorch.mace4d import (MACE4DModel, _DataFitAgent, apply_temporal_filter,
                              temporal_filter_matrix)

NUM_VIEWS = 24          # 24 views over 360 degrees, 15 degrees per view
DET_ROWS = 8
DET_COLS = 10


def _rel_max(out, ref):
    out = np.asarray(out.detach().cpu() if torch.is_tensor(out) else out, dtype=np.float64)
    ref = np.asarray(ref.detach().cpu() if torch.is_tensor(ref) else ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def _small_model():
    """A small cone-beam model with evenly spaced angles over 360 degrees."""
    angles = np.radians(360.0 / NUM_VIEWS) * np.arange(NUM_VIEWS)
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, DET_ROWS, DET_COLS), angles,
                                    source_detector_dist=100.0, source_iso_dist=50.0)
    model.set_params(no_warning=True, verbose=0)
    return model


def _smooth_sino(num_views=NUM_VIEWS, rows=DET_ROWS, cols=DET_COLS):
    """A smooth, positive sinogram: a Gaussian bump modulated slowly over the views."""
    v = np.linspace(-1.0, 1.0, rows)
    c = np.linspace(-1.0, 1.0, cols)
    base = np.exp(-(v[:, None] ** 2 + c[None, :] ** 2))
    return np.stack([base * (1.0 + 0.1 * np.sin(0.5 * a))
                     for a in range(num_views)]).astype(np.float32)


# ── the frame-axis filter matrix ─────────────────────────────────────────────
def _dct_filter(x, period, harmonics=True, band_width=1):
    """The frame-axis filter written as transforms: the reference the matrix
    must reproduce.  Zeroes the DCT-I modes at the period and its harmonics,
    with band_width modes on each side, along axis 0."""
    x = np.asarray(x, dtype=np.float32)
    num_frames = x.shape[0]
    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        harmonic_list = list(range(1, int(np.floor(period / 2)) + 1))
    else:
        harmonic_list = list(harmonics)
    coefficients = dct(x, type=1, norm='ortho', axis=0)
    for harmonic in harmonic_list:
        center = int(round(2 * (num_frames - 1) / (period / harmonic)))
        low, high = max(0, center - band_width), min(num_frames, center + band_width + 1)
        if low < high:
            coefficients[low:high, ...] = 0
    return idct(coefficients, type=1, norm='ortho', axis=0).astype(np.float32)


@pytest.mark.parametrize('num_frames', [12, 30])
def test_filter_matrix_equals_the_transform_filter(num_frames):
    """The matrix reproduces the transform filter on random inputs and on
    unit impulses at the first and last frames, where the end behavior of
    the transform shows.  The tolerance is float32 rounding relative to the
    largest input value.  At period 6 the filter removes every mode of a
    three-frame axis, one frame cannot be filtered and is refused, and
    applying the matrix along another axis equals moving that axis to the
    front and filtering there."""
    matrix = temporal_filter_matrix(num_frames, period=6)
    assert matrix.shape == (num_frames, num_frames) and matrix.dtype == torch.float32
    rng = np.random.default_rng(num_frames)
    inputs = [rng.standard_normal((num_frames, 4, 5, 3)).astype(np.float32)]
    for frame in (0, num_frames - 1):
        impulse = np.zeros((num_frames, 1, 1, 1), dtype=np.float32)
        impulse[frame] = 1.0
        inputs.append(impulse)
    worst = 0.0
    for x in inputs:
        reference = _dct_filter(x, period=6)
        filtered = apply_temporal_filter(torch.as_tensor(x), matrix, axis=0).numpy()
        worst = max(worst, float(np.max(np.abs(filtered - reference)) / np.max(np.abs(x))))
    print(f"filter matrix vs transform filter, {num_frames} frames: {worst:.2e}")
    assert worst < 1e-5
    # A symmetric projection, to the same rounding.
    m = matrix.numpy().astype(np.float64)
    symmetry, idempotence = np.max(np.abs(m - m.T)), np.max(np.abs(m @ m - m))
    print(f"symmetry {symmetry:.2e}, idempotence {idempotence:.2e}")
    assert symmetry < 1e-5 and idempotence < 1e-5

    # Three frames at period 6: every mode removed.  The zero is exact, since
    # the transform of a zeroed coefficient array is zero.
    assert torch.equal(temporal_filter_matrix(3, period=6), torch.zeros(3, 3))
    with pytest.raises(ValueError, match='at least 2'):
        temporal_filter_matrix(1, period=6)

    # Along another axis: data movement, so the check is exact.
    torch.manual_seed(1)
    axis_matrix = temporal_filter_matrix(12, period=6)
    x = torch.randn(5, 12, 4, 3)
    along_axis_1 = apply_temporal_filter(x, axis_matrix, axis=1)
    moved = apply_temporal_filter(x.movedim(1, 0), axis_matrix, axis=0).movedim(0, 1)
    assert torch.equal(along_axis_1, moved)


# ── the exit checks of the model ─────────────────────────────────────────────
def _read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ── the unit groups: no reconstruction ───────────────────────────────────────
def test_construction_builds_the_frames_of_the_scan():
    """The 24-view scan at the defaults gives 5 frames of 8 views with a
    stride of 4; num_frames keeps the first frames, refuses zero, and is
    truncated silently above the count."""
    mace = MACE4DModel(_small_model())
    assert mace.num_frames == 5 and len(mace.model_list) == 5 and len(mace.view_slices) == 5
    assert mace.view_slices[1] == slice(4, 12)
    assert mace.recon_shape == tuple(mace.model_list[0].get_params('recon_shape'))
    assert mace.sinogram_shape == (NUM_VIEWS, DET_ROWS, DET_COLS)
    assert MACE4DModel(_small_model(), num_frames=2).num_frames == 2
    assert MACE4DModel(_small_model(), num_frames=99).num_frames == 5
    with pytest.raises(ValueError, match='at least 1'):
        MACE4DModel(_small_model(), num_frames=0)


def test_reconstructions_run_end_to_end_with_and_without_the_filter(device):
    """One iteration on three frames and one device gives finite, nonzero
    values of the 4D shape and an iteration count of 1.  Three frames are
    fewer than the default filter period of 6, so the filter turns itself
    off with a warning.  A four-frame run at a period the frame count can
    carry runs the filter path: two iterations give finite values of the 4D
    shape that differ from the filter-off run, and the denoiser strength,
    estimated from the filtered initial image, differs from the filter-off
    value.  Four frames per rotation with no overlap give four frames from
    the 24 views, and at four frames the period-4 filter keeps one mode, the
    zeroth cosine mode, so the filtered stack is not zero."""
    np.random.seed(0)
    mace = MACE4DModel(_small_model(), num_frames=3)
    mace.set_params(verbose=0)
    mace.set_device_pool([device])
    with pytest.warns(UserWarning, match='fewer than the filter period'):
        recon, recon_dict = mace.recon(_smooth_sino(), max_iterations=1, stop_threshold_change_pct=0)
    assert recon.shape == (3,) + mace.recon_shape
    assert np.all(np.isfinite(recon)) and np.max(np.abs(recon)) > 0
    assert len(recon_dict['timing']) == 1
    assert recon_dict['recon_params']['iterations completed'] == 1

    np.random.seed(0)
    filtered = MACE4DModel(_small_model(), frames_per_rotation=4, frame_overlap_factor=1.0)
    assert filtered.num_frames == 4
    filtered.set_params(verbose=0)
    filtered.set_device_pool(['cpu'])
    shape = (4,) + filtered.recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    on, on_dict = filtered.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                 stop_threshold_change_pct=0)
    assert on.shape == shape and np.all(np.isfinite(on))
    assert on_dict['recon_params']['iterations completed'] == 2
    filtered.set_params(dejitter=False)
    np.random.seed(0)
    off, off_dict = filtered.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                   stop_threshold_change_pct=0)
    rel = _rel_max(on, off)
    print(f"filter on vs off: rel_max = {rel:.2e}")
    assert rel > 1e-2                        # the filter is applied, not merely recorded
    assert on_dict['recon_params']['denoiser sigma_x'] != off_dict['recon_params']['denoiser sigma_x']


def test_two_workers_equal_one_worker_and_both_take_tasks(tmp_path):
    """The scan model is set to one pixel subset, so the worker threads draw
    nothing from the random generator that can change the result.  Under
    that setting a run on two CPU workers equals the run on one to within
    1e-5, which is float32 rounding over two iterations, and the task log
    shows tasks from both workers.  At the model's default partitions the
    two runs differ, because the threads draw the partitions in schedule
    order; that is recorded as an open decision."""
    shape = (4,) + MACE4DModel(_small_model(), num_frames=4).recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)

    def run(pool, log_dir):
        ct_model = _small_model()
        ct_model.set_params(no_warning=True, granularity=[1], partition_sequence=[0])
        mace = MACE4DModel(ct_model, num_frames=4)
        mace.set_params(dejitter=False, verbose=0)
        mace.set_device_pool(pool)
        np.random.seed(0)
        recon, _ = mace.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                              stop_threshold_change_pct=0, log_dir=log_dir)
        return recon

    init_before = init.copy()
    one = run(['cpu'], str(tmp_path / 'one'))
    two = run(['cpu', 'cpu'], str(tmp_path / 'two'))
    assert np.array_equal(init, init_before)     # the caller's initial image is not written
    rel = _rel_max(two, one)
    print(f"cpu: two workers vs one: rel_max = {rel:.2e}")
    assert np.all(np.isfinite(two))
    assert rel < 1e-5
    rows = _read_rows(str(tmp_path / 'two' / 'task_log.csv'))
    assert {row['worker'] for row in rows} == {'0', '1'}


_COMPILE_CHECK_SCRIPT = r'''
import json, sys
import numpy as np
import torch
import mbirtorch
from mbirtorch import projectors
from mbirtorch.mace4d import MACE4DModel
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

device, workers = sys.argv[1], int(sys.argv[2])
angles = np.radians(360.0 / 24) * np.arange(24)
model = mbirtorch.ConeBeamModel((24, 8, 10), angles, source_detector_dist=100.0, source_iso_dist=50.0)
model.set_params(no_warning=True, verbose=0)
v, c = np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 10)
base = np.exp(-(v[:, None] ** 2 + c[None, :] ** 2))
sinogram = np.stack([base * (1.0 + 0.1 * np.sin(0.5 * a)) for a in range(24)]).astype(np.float32)
mace = MACE4DModel(model, num_frames=4)
mace.set_params(dejitter=False, verbose=0)
mace.set_device_pool([device] * workers)
shape = (4,) + mace.recon_shape
init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)
np.random.seed(0)
mace.recon(sinogram, init_recon=init, max_iterations=2, stop_threshold_change_pct=0)

entries = {}
for key in projectors._COMPILE_CACHE:
    fn = key[0] if isinstance(key, tuple) else key
    count = len(_debug_get_cache_entry_list(fn.__code__))
    entries[fn.__name__] = max(entries.get(fn.__name__, 0), count)
try:
    torch.compile(lambda x: (x * 2 + 1).sum())(torch.ones(8, device=device))
    compiles = True
except Exception:
    compiles = False
print(json.dumps(dict(compiles=compiles, errors={k: v[:40] for k, v in projectors._COMPILE_ERRORS.items()},
                      entries=entries, floor=projectors._RECOMPILE_LIMIT_FLOOR)))
'''


def test_compiled_bodies_stay_within_the_recompile_budget(device):
    """After a reconstruction in a fresh process, two CPU workers or one
    worker on another device, no compiled body fell back to eager where the
    machine can compile for the device, every fallback is the toolchain's
    where it cannot, and no function holds as many compiled variants as the
    budget the compile module sets.  The run is a separate process so that
    the counters, which torch keeps per function for the whole process,
    belong to this run alone."""
    import json
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(mbirtorch.__file__)))
    workers = 2 if device == 'cpu' else 1
    result = subprocess.run([sys.executable, '-c', _COMPILE_CHECK_SCRIPT, device, str(workers)],
                            cwd=root, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if report['compiles']:
        assert report['errors'] == {}, f"compiled bodies fell back to eager: {sorted(report['errors'])}"
    else:
        print(f"{device}: this machine cannot build inductor kernels for the device; "
              f"{len(report['errors'])} compiled bodies ran eagerly")
        assert all(v.startswith('InductorError') for v in report['errors'].values()), report['errors']
    floor = report['floor']
    worst = max(report['entries'].values(), default=0)
    print(f"{device}, {workers} worker(s): largest compiled variant count {worst} against the budget "
          f"floor of {floor}; per function {report['entries']}")
    for name, count in report['entries'].items():
        assert count < floor, f'{name} holds {count} variants against a budget of {floor}'


def test_slabbed_denoising_equals_the_whole_orientation_run():
    """With a batch size the hyperplane counts do not divide, the result
    equals the run that sweeps each orientation whole, to the rounding of a
    batched sweep.  The slab budget is given in bytes so that every
    orientation gets three hyperplanes per slab: the test volumes are 1200
    and 960 bytes, and 3700 bytes holds three of either and not four."""
    shape = (3,) + MACE4DModel(_small_model(), num_frames=3).recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)

    def run(slab_gb=2.0):
        mace = MACE4DModel(_small_model(), num_frames=3)
        mace.set_params(dejitter=False, verbose=0, denoise_slab_gb=slab_gb)
        mace.set_device_pool(['cpu'])
        np.random.seed(0)
        recon, _ = mace.recon(_smooth_sino(), init_recon=init, max_iterations=1,
                              stop_threshold_change_pct=0)
        return recon

    whole = run()
    batched = run(slab_gb=3700 / 2 ** 30)
    rel = _rel_max(batched, whole)
    print(f"slabs of 3 vs whole orientations: rel_max = {rel:.2e}")
    assert rel < 1e-6


# ── the data-fit agent's filter path ─────────────────────────────────────────
class _StubFrameAgent:
    """A frame agent that doubles its input and adds the warm start it was
    handed, and records that warm start."""

    use_warm_start = True

    def __init__(self):
        self._previous_output = None
        self.warm_starts = []

    def load_state_dict(self, state):
        self._previous_output = state['previous_output']

    def state_dict(self):
        return {'previous_output': None}

    def __call__(self, w, iteration=0):
        self.warm_starts.append(self._previous_output.clone())
        return 2.0 * w + self._previous_output


def test_data_fit_agent_folds_the_filtered_stack_and_checkpoints():
    """With the filter on, the agent's contribution to the consensus is the
    filter applied to the stack of frame outputs, and the warm start each
    frame receives on the next step is its own unfiltered output.  The frame
    agents are stubs, so the check is float32 rounding of the filter.  The
    agent's state holds its stack, and loading a saved state restores the
    stack exactly.  The initial stack is the caller's array only when the
    agent is told to adopt it, so a caller's array is never written.  With
    no stack the agent folds each frame's output as it arrives, and the
    loop's average after one step is the stack of the frame outputs."""
    torch.manual_seed(3)
    num_frames = 12
    x0 = torch.randn(num_frames, 3, 4, 5)
    matrix = temporal_filter_matrix(num_frames, period=6)
    stubs = [_StubFrameAgent() for _ in range(num_frames)]
    agent = _DataFitAgent(stubs, ['cpu'] * num_frames, x0, matrix, keep_stack=True)
    assert agent.fold_after_all
    with MACE([agent], x0, mu=[1.0], rho=0.5, devices=['cpu', 'cpu']) as loop:
        loop.step()
        outputs = torch.stack([2.0 * x0[t] + x0[t] for t in range(num_frames)])
        rel = _rel_max(loop.x_bar, apply_temporal_filter(outputs, matrix, axis=0))
        print(f"data-fit agent pieces vs the filtered outputs: rel_max = {rel:.2e}")
        assert rel < 1e-6
        assert all(torch.equal(stub.warm_starts[0], x0[t]) for t, stub in enumerate(stubs))
        loop.step()
    for t, stub in enumerate(stubs):
        assert torch.equal(stub.warm_starts[1], outputs[t])
        assert stub._previous_output is None

    # The saved state holds a copy of the stack, and loading it restores the
    # stack exactly.
    torch.manual_seed(5)
    x0 = torch.randn(3, 2, 2, 2)
    stubs = [_StubFrameAgent() for _ in range(3)]
    agent = _DataFitAgent(stubs, ['cpu'] * 3, x0, None, keep_stack=True)
    with MACE([agent], x0, mu=[1.0], rho=0.5) as loop:
        loop.step()
        saved = agent.state_dict()
        stack_after_one = agent._stack.clone()
        loop.step()
    assert not torch.equal(agent._stack, stack_after_one)
    agent.load_state_dict(saved)
    assert torch.equal(agent._stack, stack_after_one)
    assert saved['stack'].data_ptr() != agent._stack.data_ptr()   # the saved state is a copy

    # The initial stack is the caller's array only when adopted; otherwise it
    # is a copy, so the caller's array is never written.
    caller = torch.zeros(2, 3, 4, 5)
    pair = [_StubFrameAgent(), _StubFrameAgent()]
    adopted = _DataFitAgent(pair, ['cpu', 'cpu'], caller, None, keep_stack=True, adopt_stack=True)
    copied = _DataFitAgent(pair, ['cpu', 'cpu'], caller, None, keep_stack=True, adopt_stack=False)
    none = _DataFitAgent(pair, ['cpu', 'cpu'], caller, None, keep_stack=False, adopt_stack=True)
    assert adopted._stack.data_ptr() == caller.data_ptr()
    assert copied._stack.data_ptr() != caller.data_ptr() and torch.equal(copied._stack, caller)
    assert none._stack is None

    # With the filter off and the prox warm start off, the agent keeps no
    # stack and folds each frame's output as it arrives.
    class _Doubler:
        use_warm_start = False

        def __call__(self, w, iteration=0):
            return 2.0 * w

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

    torch.manual_seed(6)
    x0 = torch.randn(3, 2, 2, 2)
    agent = _DataFitAgent([_Doubler() for _ in range(3)], ['cpu'] * 3, x0, None, keep_stack=False)
    assert agent._stack is None and not agent.fold_after_all
    with MACE([agent], x0, mu=[1.0], rho=0.5) as loop:
        loop.step()
    assert torch.equal(loop.x_bar, 2.0 * x0)


# ── the one-frame equality gate ──────────────────────────────────────────────
def test_one_frame_consensus_reproduces_the_standard_reconstruction(monkeypatch):
    """With one frame the three priors denoise the same volume along three
    axis pairs.  At the agent weights [1/2, 1/6, 1/6, 1/6], with the denoiser
    sigma at sigma_prox times the square root of 3/2 and sigma_x pinned to
    the value the standard reconstruction used, the consensus must reproduce
    that reconstruction on the frame's views within 1 percent NRMSE after 40
    iterations.  The reference is a 200-iteration recon; its own convergence
    is measured against the 100-iteration recon and must be below 0.5
    percent.  The start is the 30-iteration recon, more than 1 percent away,
    so the loop is seen to move.  The problem is the 64-view Shepp-Logan
    cone-beam scan at 32 channels and 4 rows; one frame at one frame per
    rotation holds every view, which converges where a limited-angle frame
    does not.  With one frame each hyperplane volume has 32 pixels, so the
    denoisers run one subset, the regime the class is verified in.  The
    denoiser sigma and sigma_x are pinned through the public parameters
    sigma_noise and sigma_x."""
    num_views, det = 64, 32
    phantom, sinogram, params = mbirtorch.generate_demo_data(
        model_type='cone', object_type='shepp-logan', num_views=num_views,
        num_det_rows=det, num_det_channels=det, target_max_attenuation=6.0)
    rng = np.random.default_rng(0)
    sinogram = (sinogram + np.sqrt(np.exp(sinogram) / 500.0) * rng.standard_normal(sinogram.shape))
    sinogram = np.ascontiguousarray(sinogram[:, det // 2 - 2:det // 2 + 2].astype(np.float32))
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')

    def scan_model():
        model = mbirtorch.ConeBeamModel(sinogram.shape, params['angles'],
                                        source_detector_dist=params['source_detector_dist'],
                                        source_iso_dist=params['source_iso_dist'])
        model.set_params(no_warning=True, sharpness=1.0, verbose=0)
        return model

    def nrmse(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    frames = dict(frames_per_rotation=1, frame_overlap_factor=1.0)
    plain = MACE4DModel(scan_model(), num_frames=1, **frames)
    assert plain.num_frames == 1 and plain.view_slices[0] == slice(0, num_views)
    frame_model = plain.model_list[0]
    frame_model.configure_devices(devices=['cpu'])

    def reference(iterations):
        np.random.seed(0)
        return frame_model.recon(sinogram, weights=weights, max_iterations=iterations,
                                 stop_threshold_change_pct=0.0, logfile_path=None, print_logs=False)

    recon_200, _ = reference(200)
    recon_100, recon_100_dict = reference(100)
    start, _ = reference(30)
    reference_error = nrmse(recon_100, recon_200)
    start_distance = nrmse(start, recon_200)
    regularization = recon_100_dict['recon_params']['regularization_params']
    sigma_prox, sigma_x = float(regularization['sigma_prox']), float(regularization['sigma_x'])
    print(f"reference: recon(100) vs recon(200) NRMSE {reference_error:.2e}; start recon(30) is "
          f"{start_distance:.4f} away; sigma_prox {sigma_prox:.5f}, sigma_x {sigma_x:.5f}")
    assert reference_error < 5e-3
    assert start_distance > 0.01

    trace = []
    original_step = mace_module.MACE.step

    def recording_step(self):
        change = original_step(self)
        trace.append(nrmse(self.x_bar.numpy(), recon_200))
        return change
    monkeypatch.setattr(mace_module.MACE, 'step', recording_step)

    gate = MACE4DModel(scan_model(), num_frames=1, **frames)
    gate.set_params(dejitter=False, verbose=0, mace_prior_weight=0.5, rho_mann=0.5,
                    sigma_noise=sigma_prox * math.sqrt(1.5))
    # Setting sigma_x pins the denoisers' strength, and disables the estimate
    # with the warning every model gives for a directly-set regularization
    # parameter.
    with pytest.warns(UserWarning, match='auto-regularization'):
        gate.set_params(sigma_x=sigma_x)
    gate.set_device_pool(['cpu'])
    np.random.seed(0)
    x, recon_dict = gate.recon(sinogram, weights=weights, init_recon=start[None],
                               max_iterations=40, stop_threshold_change_pct=0.0)
    settings = recon_dict['recon_params']
    assert settings['denoiser sigma (global)'] == pytest.approx(sigma_prox * math.sqrt(1.5))
    assert settings['denoiser sigma_x'] == pytest.approx(sigma_x)
    assert settings['denoiser sigma_x source'] == 'set by sigma_x'
    assert settings['denoiser sigma source'] == 'set by sigma_noise'
    assert settings['beta [fwd, xyt, yzt, xzt]'] == [0.5, 0.1667, 0.1667, 0.1667]
    print(f"one-frame gate: NRMSE to recon(200) after 10/20/30/40 iterations = "
          f"{trace[9]:.4f} / {trace[19]:.4f} / {trace[29]:.4f} / {trace[39]:.4f}")
    assert len(trace) == 40
    assert trace[39] < 0.01
    assert trace[39] < trace[9]                # the loop moved toward the reference
