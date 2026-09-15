"""Tests for the 4D reconstruction model and the frame-axis filter.

The filter matrix is checked against the transform it replaces.  The model
is checked on a 24-view cone-beam scan with a smooth sinogram.  A run with
three frames on one device gives finite values and the three log files.  A
run with two workers on the CPU equals the one-worker run and shows tasks
from both workers.  The initial image is written to the cache and read
back.  The compile check runs a reconstruction in a fresh process, because
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
import warnings

import numpy as np
import pytest
import torch
from scipy.fft import dct, idct

import mbirtorch
from mbirtorch import mace as mace_module
from mbirtorch.mace import MACE
from mbirtorch.mace4d import (MACE4DModel, _DataFitAgent, _normalize_prior_weights,
                              apply_temporal_filter, temporal_filter_matrix)

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


# ── the unit groups: no reconstruction ───────────────────────────────────────
def test_prior_weights_normalize_and_are_checked():
    """A scalar prior weight w gives the agent weights [1 - w, w/3, w/3, w/3]
    and a list of three gives [1 - sum, w1, w2, w3]; a weight above one,
    below zero, a list summing above one, or a list of two is refused."""
    assert np.allclose(_normalize_prior_weights(0.5), [0.5, 1 / 6, 1 / 6, 1 / 6])
    assert np.allclose(_normalize_prior_weights([0.1, 0.2, 0.3]), [0.4, 0.1, 0.2, 0.3])
    for bad in (1.5, -0.1, [0.5, 0.5, 0.5], [0.1, 0.2]):
        with pytest.raises(ValueError):
            _normalize_prior_weights(bad)


def test_the_model_holds_the_device_pool_it_is_given():
    """set_device_pool stores the resolved pool and devices reads it back;
    before any call, devices is the pool for None.  A count above the pool
    and an unknown platform string are refused."""
    mace = MACE4DModel(_small_model(), num_frames=2)
    mace.set_params(verbose=0)
    assert mace.devices == mbirtorch.resolve_device_pool(None)
    mace.set_device_pool(1)
    assert len(mace.devices) == 1
    mace.set_device_pool([0])
    assert mace.devices == mbirtorch.resolve_device_pool(None)[:1]
    mace.set_device_pool('cpu')
    assert mace.devices == [torch.device('cpu')]
    mace.set_device_pool(['cpu', 'cpu'])
    assert mace.devices == [torch.device('cpu'), torch.device('cpu')]
    with pytest.raises(ValueError):
        mace.set_device_pool(len(mbirtorch.resolve_device_pool(None)) + 1)
    with pytest.raises(ValueError):
        mace.set_device_pool('tpu')


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


def test_parameters_default_and_read_back():
    """The reconstruction parameters hold the plan's defaults, read back what
    they are set to, and an invalid prior weight or an unknown name is
    refused when set."""
    mace = MACE4DModel(_small_model(), num_frames=2)
    defaults = dict(mace_prior_weight=0.5, rho_mann=0.5, prox_num_iterations=3, prox_stop_threshold=0.02,
                    prox_partition_advance=1.0, prox_warm_start=True, denoiser_warm_start=False,
                    sigma_prox=None, dejitter=True, dejitter_verbose=0, verbose=1)
    for name, value in defaults.items():
        assert mace.get_params(name) == value, name
    mace.set_params(rho_mann=0.25, dejitter=False, sigma_prox=0.1)
    assert mace.get_params('rho_mann') == 0.25 and mace.get_params('dejitter') is False
    assert mace.get_params('sigma_prox') == 0.1
    with pytest.raises(ValueError):
        mace.set_params(mace_prior_weight=1.5)
    with pytest.raises(ValueError):
        mace.set_params(not_a_parameter=1)


def test_the_initialization_cache_is_read_when_valid(tmp_path):
    """An absent cache file gives None silently; a file of the wrong shape
    gives None with one warning naming it invalid; a valid file loads as
    float32."""
    mace = MACE4DModel(_small_model())
    mace.set_params(verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert mace._load_cached_init(str(tmp_path)) is None
    np.save(tmp_path / 'init_recon.npy', np.zeros((1, 2, 3, 4), dtype=np.float32))
    with pytest.warns(UserWarning, match='invalid') as caught:
        assert mace._load_cached_init(str(tmp_path)) is None
    assert len(caught) == 1
    good = np.zeros((mace.num_frames,) + mace.recon_shape, dtype=np.float64)
    np.save(tmp_path / 'init_recon.npy', good)
    loaded = mace._load_cached_init(str(tmp_path))
    assert loaded.dtype == np.float32 and loaded.shape == good.shape


def test_a_three_frame_reconstruction_runs_and_logs(device, tmp_path):
    """One iteration on three frames and one device gives finite, nonzero
    values of the 4D shape, the three log files, the four result keys, and
    an iteration count of 1.  Three frames are fewer than the default filter
    period of 6, so the filter turns itself off with a warning and the run
    settings say so.  The initial image is written to the cache on the first
    run and read from it on the second."""
    np.random.seed(0)
    mace = MACE4DModel(_small_model(), num_frames=3)
    mace.set_params(verbose=0)
    mace.set_device_pool([device])
    init_dir, log_dir = str(tmp_path / 'init'), str(tmp_path / 'logs')

    with pytest.warns(UserWarning, match='fewer than the filter period'):
        recon, recon_dict = mace.recon(_smooth_sino(), max_iterations=1, stop_threshold_change_pct=0,
                                       init_dir=init_dir, log_dir=log_dir)
    assert recon.shape == (3,) + mace.recon_shape
    assert np.all(np.isfinite(recon)) and np.max(np.abs(recon)) > 0
    assert recon_dict['recon_params']['dejitter'] is False
    assert 'fewer than' in recon_dict['recon_params']['temporal filter']
    assert 'temporal filter' in open(os.path.join(log_dir, 'run_info.txt')).read()
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
    # One task per orientation here, since no memory budget can be read.
    assert all(row['part'] == '0' for row in rows if row['kind'] == 'denoise')
    timing = _read_rows(os.path.join(log_dir, 'timing_log.csv'))[0]
    print(f"{device}: change {float(timing['consensus_change_pct']):.3f}%, "
          f"mean denoiser iterations {float(timing['denoise_mean_iterations']):.1f}, "
          f"sigma_x {recon_dict['recon_params']['denoiser sigma_x [xyt, yzt, xzt]']}")
    # From a cold start the sweeps never reach the 0.2 percent threshold on
    # this problem, so every volume runs the cap of 15 iterations.
    assert float(timing['denoise_mean_iterations']) == 15.0

    assert os.path.isfile(os.path.join(init_dir, 'init_recon.npy'))
    mace.set_params(dejitter=False)
    np.random.seed(0)
    again, again_dict = mace.recon(_smooth_sino(), max_iterations=1, stop_threshold_change_pct=0,
                                   init_dir=init_dir)
    assert again_dict['recon_params']['init source'].startswith('cached')
    assert again.shape == recon.shape

    # A huge stop threshold ends the loop after one iteration of five; a
    # supplied initial image, supplied weights, and a tensor sinogram are
    # taken and recorded.
    np.random.seed(0)
    _, early = mace.recon(torch.as_tensor(_smooth_sino()), weights=np.ones_like(_smooth_sino()),
                          init_recon=again, max_iterations=5, stop_threshold_change_pct=1e9)
    assert len(early['timing']) == 1 and early['recon_params']['iterations completed'] == 1
    assert early['recon_params']['init source'] == 'provided by caller'
    assert early['recon_params']['weights'] == 'supplied by caller'


def test_the_other_settings_run_and_are_recorded(tmp_path):
    """Both warm starts the other way round, a given sigma_prox, a list prior
    weight, and verbose logging run to finite values, and the run settings
    record every one of them; the given sigma_prox reaches the frame agents."""
    mace = MACE4DModel(_small_model(), num_frames=3)
    mace.set_params(dejitter=False, verbose=1, prox_warm_start=False, denoiser_warm_start=True,
                    sigma_prox=0.05, mace_prior_weight=[0.1, 0.2, 0.3], rho_mann=0.4)
    mace.set_device_pool(['cpu'])
    shape = (3,) + mace.recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    log_dir = str(tmp_path / 'logs')
    np.random.seed(0)
    recon, recon_dict = mace.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                   stop_threshold_change_pct=0, log_dir=log_dir)
    assert np.all(np.isfinite(recon))
    settings = recon_dict['recon_params']
    assert settings['prox_warm_start'] is False and settings['denoiser_warm_start'] is True
    assert settings['sigma_prox'] == 0.05 and settings['rho_mann'] == 0.4
    assert settings['beta [fwd, xyt, yzt, xzt]'] == [0.4, 0.1, 0.2, 0.3]
    text = open(os.path.join(log_dir, 'run_info.txt')).read()
    assert 'sigma_prox' in text and '0.05' in text
    # The denoiser warm start shows in the second iteration's sweep count,
    # which is below the cap the cold start always reaches.
    assert recon_dict['timing'][1]['denoise_mean_iterations'] < 15.0


def test_data_fit_agent_without_a_stack_folds_each_frame_as_it_completes():
    """With the filter off and the prox warm start off, the agent keeps no
    stack, folds each frame's output as it arrives, and the loop's average
    after one step is the stack of the frame outputs."""

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


def test_a_batch_size_that_does_not_divide_the_hyperplanes_sweeps_one_shape(monkeypatch):
    """With a batch size the hyperplane counts do not divide, every sweep
    still sees a stack of exactly that many volumes, the short last slab
    padded, so one compiled shape serves all of them; and the result equals
    the run that sweeps each orientation whole, to the rounding of a batched
    sweep.  The batch size is forced, because no memory budget can be read
    here."""
    shape = (3,) + MACE4DModel(_small_model(), num_frames=3).recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)

    def run():
        mace = MACE4DModel(_small_model(), num_frames=3)
        mace.set_params(dejitter=False, verbose=0)
        mace.set_device_pool(['cpu'])
        np.random.seed(0)
        recon, _ = mace.recon(_smooth_sino(), init_recon=init, max_iterations=1,
                              stop_threshold_change_pct=0)
        return recon

    whole = run()

    seen = []
    original = mbirtorch.QGGMRFDenoiser.denoise_stack

    def recording(self, stack, *args, **kwargs):
        seen.append((int(stack.shape[0]), kwargs.get('batch_size')))
        return original(self, stack, *args, **kwargs)
    monkeypatch.setattr(mbirtorch.QGGMRFDenoiser, 'denoise_stack', recording)
    monkeypatch.setattr(mbirtorch.QGGMRFDenoiser, 'auto_batch_size', lambda self, **kwargs: 3)
    batched = run()

    # 8 and 10 hyperplanes in slabs of 3: 3 + 4 + 4 = 11 sweeps, every one at 3 volumes.
    assert len(seen) == 11
    assert all(volumes == 3 and batch == 3 for volumes, batch in seen), seen
    rel = _rel_max(batched, whole)
    print(f"slabs of 3 vs whole orientations: rel_max = {rel:.2e}")
    assert rel < 1e-6


def test_wrong_inputs_are_refused():
    """A sinogram, weights, or initial image of the wrong shape raises
    ValueError, and so does a constant initial image, from which no denoiser
    noise level can be estimated."""
    mace = MACE4DModel(_small_model(), num_frames=3)
    mace.set_params(verbose=0)
    sinogram = _smooth_sino()
    with pytest.raises(ValueError, match='sinogram shape'):
        mace.recon(sinogram[:-1])
    with pytest.raises(ValueError, match='weights shape'):
        mace.recon(sinogram, weights=np.ones((NUM_VIEWS, DET_ROWS, DET_COLS + 1), dtype=np.float32))
    with pytest.raises(ValueError, match='init_recon shape'):
        mace.recon(sinogram, init_recon=np.zeros((1, 2, 3, 4), dtype=np.float32))
    mace.set_params(dejitter=False)
    with pytest.raises(ValueError, match='constant initial image'):
        mace.recon(sinogram, init_recon=np.zeros((3,) + mace.recon_shape, dtype=np.float32))


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


def test_data_fit_agent_filters_in_several_slabs(monkeypatch):
    """With the slab size forced small, the agent's pieces cover the last
    axis in several regions whose concatenation equals the filter applied to
    the whole stack.  This is data movement, so the check is exact."""
    import mbirtorch.mace4d as mace4d
    monkeypatch.setattr(mace4d, '_FILTER_SLAB_BYTES', 12 * 3 * 4 * 4 * 2)   # two slices per slab
    torch.manual_seed(4)
    x0 = torch.randn(12, 3, 4, 5)
    matrix = temporal_filter_matrix(12, period=6)
    agent = _DataFitAgent([_StubFrameAgent() for _ in range(12)], ['cpu'] * 12, x0, matrix,
                          keep_stack=True, adopt_stack=False)
    pieces = list(agent.pieces())
    assert len(pieces) == 3                  # slices 0:2, 2:4, 4:5
    assembled = torch.empty_like(x0)
    for region, piece in pieces:
        assembled[region] = piece
    assert torch.equal(assembled, apply_temporal_filter(x0, matrix, axis=0))


def test_data_fit_agent_checkpoint_round_trip():
    """The agent's state holds its stack; loading a saved state restores the
    stack exactly and hands each frame agent its own saved state."""
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


def test_data_fit_agent_adopts_or_copies_its_initial_stack():
    """The agent's stack is the initial image itself when told to adopt it,
    and a copy otherwise, so a caller's array is never written.  This is
    data movement, so the check is on storage."""
    x0 = torch.zeros(2, 3, 4, 5)
    stubs = [_StubFrameAgent(), _StubFrameAgent()]
    adopted = _DataFitAgent(stubs, ['cpu', 'cpu'], x0, None, keep_stack=True, adopt_stack=True)
    copied = _DataFitAgent(stubs, ['cpu', 'cpu'], x0, None, keep_stack=True, adopt_stack=False)
    none = _DataFitAgent(stubs, ['cpu', 'cpu'], x0, None, keep_stack=False, adopt_stack=True)
    assert adopted._stack.data_ptr() == x0.data_ptr()
    assert copied._stack.data_ptr() != x0.data_ptr() and torch.equal(copied._stack, x0)
    assert none._stack is None


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
    gives finite values that differ from the filter-off run, and records the
    filter in the run settings.  Four frames per rotation with no overlap
    give four frames from the 24 views, and at four frames the period-4
    filter keeps one mode, the zeroth cosine mode, so the filtered stack is
    not zero."""
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
    mace.set_params(dejitter=False)
    np.random.seed(0)
    unfiltered, _ = mace.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                               stop_threshold_change_pct=0)
    rel = _rel_max(recon, unfiltered)
    print(f"filter on vs off: rel_max = {rel:.2e}")
    assert rel > 1e-2                        # the filter is applied, not merely recorded
    assert recon_dict['recon_params']['dejitter'] is True
    # Period 4 over 4 frames: harmonics 1 and 2, periods 4 and 2; three modes removed, one kept.
    assert recon_dict['recon_params']['temporal filter'] == \
        'removes periods of 4, 2 frames; 3 of 4 modes removed, 1 kept'
    assert 'temporal filter' in open(os.path.join(log_dir, 'run_info.txt')).read()
    assert recon_dict['recon_params']['iterations completed'] == 2
    print(f"filter path: change {[round(r['consensus_change_pct'], 3) for r in recon_dict['timing']]}%")


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
    denoiser sigma and sigma_x are pinned through a test-only subclass,
    because no public parameter reaches them."""
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

    class _Pinned(MACE4DModel):
        """The model with the denoiser sigma and sigma_x pinned to the values
        the gate needs, which no public parameter reaches."""

        @staticmethod
        def _estimate_global_sigma(init_recon, device):
            return sigma_prox * math.sqrt(1.5)

        def _configure_orientation(self, axis, x0, sigma, device, init_supplied):
            shape, denoiser_params, batch_size = super()._configure_orientation(
                axis, x0, sigma, device, init_supplied)
            return shape, dict(denoiser_params, sigma_x=sigma_x), batch_size

    trace = []
    original_step = mace_module.MACE.step

    def recording_step(self):
        change = original_step(self)
        trace.append(nrmse(self.x_bar.numpy(), recon_200))
        return change
    monkeypatch.setattr(mace_module.MACE, 'step', recording_step)

    gate = _Pinned(scan_model(), num_frames=1, **frames)
    gate.set_params(dejitter=False, verbose=0, mace_prior_weight=0.5, rho_mann=0.5)
    gate.set_device_pool(['cpu'])
    np.random.seed(0)
    x, recon_dict = gate.recon(sinogram, weights=weights, init_recon=start[None],
                               max_iterations=40, stop_threshold_change_pct=0.0)
    settings = recon_dict['recon_params']
    assert settings['denoiser sigma (global)'] == pytest.approx(sigma_prox * math.sqrt(1.5))
    assert settings['denoiser sigma_x [xyt, yzt, xzt]'] == [sigma_x] * 3
    assert settings['beta [fwd, xyt, yzt, xzt]'] == [0.5, 0.1667, 0.1667, 0.1667]
    print(f"one-frame gate: NRMSE to recon(200) after 10/20/30/40 iterations = "
          f"{trace[9]:.4f} / {trace[19]:.4f} / {trace[29]:.4f} / {trace[39]:.4f}")
    assert len(trace) == 40
    assert trace[39] < 0.01
    assert trace[39] < trace[9]                # the loop moved toward the reference
