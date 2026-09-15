"""Tests for the 4D reconstruction model and the frame-axis filter.

The filter matrix is checked against the transform it replaces.  The model
is checked on a 24-view cone-beam scan with a smooth sinogram: a run with
three frames on one device gives finite values and the three log files, a
run with two workers on the CPU equals the one-worker run and shows tasks
from both workers, no compiled body falls back and no function exceeds
torch's recompile budget, and the initial image is cached and reused.  Two
worker threads on one MPS device crash inside Metal, so the two-worker run
is on the CPU only.  The smooth sinogram is kept because a
random one reconstructs to a volume with extreme values, on which the
qGGMRF line search computes zero over zero.
"""

import csv
import os

import numpy as np
import pytest
import torch
from scipy.fft import dct, idct

import mbirtorch
from mbirtorch import projectors
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
    largest input value."""
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


def test_filter_matrix_for_three_frames_is_zero():
    """At period 6 the filter removes every mode of a three-frame axis.  The
    zero is exact: the transform of a zeroed coefficient array is zero.  One
    frame cannot be filtered and is refused."""
    assert torch.equal(temporal_filter_matrix(3, period=6), torch.zeros(3, 3))
    with pytest.raises(ValueError, match='at least 2'):
        temporal_filter_matrix(1, period=6)


def test_filter_applies_along_any_axis():
    """Applying the matrix along axis 1 of a stack equals moving that axis to
    the front, filtering, and moving it back.  This is data movement, so the
    check is exact."""
    torch.manual_seed(1)
    matrix = temporal_filter_matrix(12, period=6)
    x = torch.randn(5, 12, 4, 3)
    along_axis_1 = apply_temporal_filter(x, matrix, axis=1)
    moved = apply_temporal_filter(x.movedim(1, 0), matrix, axis=0).movedim(0, 1)
    assert torch.equal(along_axis_1, moved)


# ── the exit checks of the model ─────────────────────────────────────────────
def _read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _inductor_compiles_on(device):
    """Whether torch.compile can build a kernel for the device on this machine.
    A machine whose C++ toolchain cannot build inductor's CPU kernels makes
    every compiled body fall back to eager, which is not a defect of the code
    under test."""
    try:
        torch.compile(lambda x: (x * 2 + 1).sum())(torch.ones(8, device=device))
        return True
    except Exception:   # noqa: BLE001 - any failure means the backend is unusable here
        return False


def _check_compiled_bodies(device, errors_before):
    """After a run: no compiled body fell back to eager, where the machine can
    compile for the device at all, and no function holds as many compiled
    variants as torch's recompile budget allows."""
    import torch._dynamo.config as dynamo_config
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    new_errors = {k: v for k, v in projectors._COMPILE_ERRORS.items() if k not in errors_before}
    if _inductor_compiles_on(device):
        assert new_errors == {}, f'compiled bodies fell back to eager: {sorted(new_errors)}'
    else:
        # Every fallback must be the toolchain's compile error, and nothing else.
        print(f"{device}: this machine cannot build inductor kernels for the device; "
              f"{len(new_errors)} compiled bodies ran eagerly")
        assert all(v.startswith('InductorError') for v in new_errors.values()), new_errors
    limit = int(dynamo_config.recompile_limit)
    worst = 0
    for key in projectors._COMPILE_CACHE:
        fn = key[0] if isinstance(key, tuple) else key
        entries = len(_debug_get_cache_entry_list(fn.__code__))
        worst = max(worst, entries)
        assert entries < limit, f'{fn.__name__} holds {entries} variants against a budget of {limit}'
    print(f"{device}: largest compiled variant count {worst} against a budget of {limit}")


def test_a_three_frame_reconstruction_runs_and_logs(device, tmp_path):
    """One iteration on three frames and one device gives finite values of
    the 4D shape, the three log files, the four result keys, and an
    iteration count of 1.  The initial image is written to the cache on the
    first run and read from it on the second."""
    np.random.seed(0)
    errors_before = set(projectors._COMPILE_ERRORS)
    mace = MACE4DModel(_small_model(), num_frames=3)
    mace.set_params(dejitter=False, verbose=0)
    mace.set_device_pool([device])
    init_dir, log_dir = str(tmp_path / 'init'), str(tmp_path / 'logs')

    recon, recon_dict = mace.recon(_smooth_sino(), max_iterations=1, stop_threshold_change_pct=0,
                                   init_dir=init_dir, log_dir=log_dir)
    assert recon.shape == (3,) + mace.recon_shape
    assert np.all(np.isfinite(recon))
    for name in ('run_info.txt', 'timing_log.csv', 'task_log.csv'):
        assert os.path.isfile(os.path.join(log_dir, name))
    assert sorted(recon_dict) == ['model_params', 'notes', 'recon_params', 'timing']
    assert len(recon_dict['timing']) == 1
    assert recon_dict['recon_params']['iterations completed'] == 1
    assert recon_dict['recon_params']['weights'] == 'unit (weights=None)'
    assert recon_dict['recon_params']['init source'].startswith('computed')
    rows = _read_rows(os.path.join(log_dir, 'task_log.csv'))
    assert sorted(row['kind'] for row in rows) == ['denoise'] * 3 + ['prox'] * 3
    assert all(row['part'] == '' for row in rows if row['kind'] == 'prox')
    timing = _read_rows(os.path.join(log_dir, 'timing_log.csv'))[0]
    print(f"{device}: change {float(timing['consensus_change_pct']):.3f}%, "
          f"mean denoiser iterations {float(timing['denoise_mean_iterations']):.1f}, "
          f"sigma_x {recon_dict['recon_params']['denoiser sigma_x [xyt, yzt, xzt]']}")
    assert float(timing['denoise_mean_iterations']) >= 1

    assert os.path.isfile(os.path.join(init_dir, 'init_recon.npy'))
    np.random.seed(0)
    again, again_dict = mace.recon(_smooth_sino(), max_iterations=1, stop_threshold_change_pct=0,
                                   init_dir=init_dir)
    assert again_dict['recon_params']['init source'].startswith('cached')
    assert again.shape == recon.shape
    _check_compiled_bodies(device, errors_before)


def test_two_workers_equal_one_worker_and_both_take_tasks(tmp_path):
    """With the pixel partitions made trivial, so that the order in which
    the worker threads draw from the random generator cannot matter, a run
    on two CPU workers equals the run on one to float32 rounding, and the
    task log shows tasks from both workers.  The compiled bodies are checked
    after the run as in the one-worker test."""
    device = 'cpu'
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

    errors_before = set(projectors._COMPILE_ERRORS)
    one = run([device], str(tmp_path / 'one'))
    two = run([device, device], str(tmp_path / 'two'))
    rel = _rel_max(two, one)
    print(f"{device}: two workers vs one: rel_max = {rel:.2e}")
    assert np.all(np.isfinite(two))
    assert rel < 1e-5
    rows = _read_rows(str(tmp_path / 'two' / 'task_log.csv'))
    assert {row['device'] for row in rows} == {'0', '1'}

    _check_compiled_bodies(device, errors_before)


def test_wrong_shapes_are_refused_before_any_computation():
    """A sinogram, weights, or initial image of the wrong shape raises at once."""
    mace = MACE4DModel(_small_model(), num_frames=3)
    mace.set_params(verbose=0)
    sinogram = _smooth_sino()
    with pytest.raises(ValueError, match='sinogram shape'):
        mace.recon(sinogram[:-1])
    with pytest.raises(ValueError, match='weights shape'):
        mace.recon(sinogram, weights=np.ones((NUM_VIEWS, DET_ROWS, DET_COLS + 1), dtype=np.float32))
    with pytest.raises(ValueError, match='init_recon shape'):
        mace.recon(sinogram, init_recon=np.zeros((1, 2, 3, 4), dtype=np.float32))


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


def test_data_fit_agent_folds_the_filtered_stack_and_warm_starts_from_the_unfiltered_one():
    """With the filter on, the agent's contribution to the consensus is the
    filter applied to the stack of frame outputs, and the warm start each
    frame receives on the next step is its own unfiltered output.  The frame
    agents are stubs, so the check is float32 rounding of the filter."""
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


def test_the_filter_path_runs_end_to_end(tmp_path):
    """A run with the filter on, at a period the frame count can carry,
    gives finite values and records the filter in the run settings.  Four
    frames per rotation with no overlap give four frames from the 24 views,
    and the period-4 filter keeps the frame mean of each voxel."""
    np.random.seed(0)
    mace = MACE4DModel(_small_model(), frames_per_rotation=4, frame_overlap_factor=1.0)
    assert mace.num_frames == 4
    mace.set_params(verbose=0)
    mace.set_device_pool(['cpu'])
    shape = (4,) + mace.recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    log_dir = str(tmp_path / 'logs')
    recon, recon_dict = mace.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                   stop_threshold_change_pct=0, log_dir=log_dir)
    assert recon.shape == shape and np.all(np.isfinite(recon))
    assert recon_dict['recon_params']['dejitter'] is True
    assert recon_dict['recon_params']['iterations completed'] == 2
    print(f"filter path: change {[round(r['consensus_change_pct'], 3) for r in recon_dict['timing']]}%")
