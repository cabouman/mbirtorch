"""QGGMRFDenoiser gates: golden parity vs mbirjax, a denoising smoke on every
backend, two shards against one device, the sigma_noise knob with the
automatic regularization off, an all-zero input, the stack denoiser against a
loop of single-volume calls, and one initialization reused across calls."""

import glob
import os

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import denoising
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "golden_*.npz")))


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


@pytest.mark.goldens
@pytest.mark.skipif(not _paths or "den_out" not in np.load(_paths[0]).files,
                    reason="no denoiser goldens: rerun tests/generate_goldens.py")
def test_denoiser_matches_golden():
    golden = np.load(_paths[0])
    shape = tuple(int(x) for x in golden["recon_shape"])
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=["cpu"])
    denoiser.set_params(no_warning=True, verbose=0)

    sigma_est = float(denoiser.estimate_image_noise_std(golden["den_noisy"]))
    est_rel = abs(sigma_est - float(golden["den_sigma_est"])) / float(golden["den_sigma_est"])
    print(f"sigma estimate: torch {sigma_est:.6g} vs jax "
          f"{float(golden['den_sigma_est']):.6g} (rel {est_rel:.2e})")
    assert est_rel < 1e-5

    np.random.seed(7)     # the golden's RECON_SEED (partition determinism)
    denoised, den_dict = denoiser.denoise(golden["den_noisy"], sigma_noise=0.1,
                                          max_iterations=5,
                                          stop_threshold_change_pct=0.0)
    rp = den_dict["recon_params"]
    alpha_rel = np.max(np.abs(np.array(rp["alpha_values"]) - golden["den_alpha"])
                       / np.abs(golden["den_alpha"]))
    nmae_rel = np.max(np.abs(np.array(rp["stop_threshold_change_pct"]) - golden["den_nmae_pct"])
                      / np.abs(golden["den_nmae_pct"]))
    out_rel = _rel_max(denoised, golden["den_out"])
    print(f"denoiser alpha rel = {alpha_rel:.2e}, nmae rel = {nmae_rel:.2e}, "
          f"output rel_max = {out_rel:.2e}")
    assert alpha_rel < 1e-2
    assert nmae_rel < 1e-3
    assert out_rel < 1e-3


def test_denoise_reduces_noise(device):
    shape = (32, 32, 32)
    clean = np.zeros(shape, dtype=np.float32)
    clean[8:-8, 8:-8, 8:-8] = 1.0
    noisy = clean + 0.1 * np.random.RandomState(2).randn(*shape).astype(np.float32)
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    denoised, _ = denoiser.denoise(noisy, sigma_noise=0.1, max_iterations=5,
                                   stop_threshold_change_pct=0.0)
    err_noisy = np.linalg.norm(noisy - clean)
    err_den = np.linalg.norm(denoised - clean)
    assert err_den < 0.6 * err_noisy, (err_den, err_noisy)


def test_sharded_denoise_matches_single_device():
    """Two CPU shards vs one device on the same seeded problem: the denoised
    volume and the automatically set regularization parameters must both agree
    with the single-device run.

    The sharded path stages halos once per pass and combines the step-size sums
    on the lead device, so agreement is at float level, not bitwise (gate per
    the measured iterated-comparison floor).  The regularization parameters are
    gated relatively for the same reason: the two runs reach numpy by different
    routes -- a strided view of the caller's own array on one, a concatenation
    of per-shard copies on the other -- and float32 reductions need not
    accumulate in the same order over different memory layouts.

    sigma_noise is left unset so the statistics run, and the two-device model
    is handed its device form, which is what a caller that keeps its volume on
    the devices does and what puts the statistics on the sharded path."""
    shape = (24, 24, 21)   # 2 shards pad the slice axis 21 -> 22
    clean = np.zeros(shape, dtype=np.float32)
    clean[6:-6, 6:-6, 5:-5] = 1.0
    noisy = clean + 0.1 * np.random.RandomState(4).randn(*shape).astype(np.float32)

    ref_den = mbirtorch.QGGMRFDenoiser(shape)
    ref_den.configure_devices(devices=['cpu'])
    ref_den.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    ref, ref_dict = ref_den.denoise(noisy, max_iterations=5,
                                    stop_threshold_change_pct=0.0, logfile_path=None)

    sh_den = mbirtorch.QGGMRFDenoiser(shape)
    sh_den.configure_devices(devices=['cpu', 'cpu'])
    sh_den.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    out, out_dict = sh_den.denoise(sh_den._shard_recon(noisy), max_iterations=5,
                                   stop_threshold_change_pct=0.0, logfile_path=None,
                                   output_sharded=True)
    out = np.asarray(out.gather())

    assert out.shape == ref.shape
    rel = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    print(f"sharded vs single denoise rel_max = {rel:.2e}")
    assert rel < 1e-4
    # The denoiser dict carries the run log and notes, like recon's.
    # (verbose=0 logs no iteration lines, so only the keys are checked.)
    assert 'recon_log' in out_dict and 'notes' in out_dict

    ref_params = ref_dict['recon_params']['regularization_params']
    out_params = out_dict['recon_params']['regularization_params']
    for name in ('sigma_x', 'sigma_prox', 'sigma_y'):
        rel = abs(out_params[name] - ref_params[name]) / abs(ref_params[name])
        print(f"{name}: sharded {out_params[name]:.8g} vs single "
              f"{ref_params[name]:.8g} (rel {rel:.2e})")
        assert rel < 1e-5, name


def test_denoise_pinned_params_keep_sigma_noise_knob(device):
    """With auto-regularization pinned off (the Plug-and-Play agent
    configuration: sigma_x fixed so the denoiser is the same operator every
    call), sigma_noise still sets the denoising strength.  For the identity
    forward model sigma_y IS sigma_noise, so denoise must keep them equal on
    the pinned path too -- before that sync, a pinned denoiser silently ran
    at a stale sigma_y and this knob was dead."""
    shape = (32, 32, 1)
    rng = np.random.default_rng(0)
    clean = np.zeros(shape, dtype=np.float32)
    clean[8:24, 8:24, :] = 1.0
    noisy = clean + 0.1 * rng.standard_normal(shape).astype(np.float32)

    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0, sigma_x=0.05,
                        auto_regularize_flag=False)

    np.random.seed(0)
    weak, _ = denoiser.denoise(noisy, sigma_noise=0.01, max_iterations=4,
                               stop_threshold_change_pct=0.0)
    assert float(denoiser.get_params('sigma_y')) == pytest.approx(0.01)

    np.random.seed(0)
    strong, _ = denoiser.denoise(noisy, sigma_noise=0.5, max_iterations=4,
                                 stop_threshold_change_pct=0.0)
    assert float(denoiser.get_params('sigma_y')) == pytest.approx(0.5)

    # Small sigma_noise hugs the input; large sigma_noise smooths it hard.
    dist_weak = float(np.linalg.norm(weak - noisy))
    dist_strong = float(np.linalg.norm(strong - noisy))
    assert dist_weak < 0.5 * dist_strong, (dist_weak, dist_strong)


def test_zero_input_comes_back_unchanged(device):
    """An all-zero image is a legitimate input (a Plug-and-Play loop
    initialized at zero feeds one in): the denoiser must return it unchanged
    instead of dividing its NMAE statistic by the zero image norm.

    The same holds for a stack: a stack of zeros has no neighbor differences,
    so the estimate is zero and sigma_x takes the floor, and denoise_stack
    returns the zeros with no NaN."""
    shape = (16, 16, 1)
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0, sigma_x=0.05,
                        auto_regularize_flag=False)
    np.random.seed(0)
    out, _ = denoiser.denoise(np.zeros(shape, dtype=np.float32),
                              sigma_noise=0.1, max_iterations=2,
                              stop_threshold_change_pct=0.0)
    assert np.array_equal(np.asarray(out), np.zeros(shape, dtype=np.float32))

    stack_shape = (8, 10, 12)
    zeros = np.zeros((3,) + stack_shape, dtype=np.float32)
    params = _auto_denoiser(stack_shape, device).auto_set_regularization_params_from_stack(zeros)
    assert params['sigma_x'] == denoising._SIGMA_X_FLOOR

    stack_denoiser = mbirtorch.QGGMRFDenoiser(stack_shape)
    stack_denoiser.configure_devices(devices=[device])
    stack_denoiser.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    stack_out, info = stack_denoiser.denoise_stack(zeros, sigma_noise=0.1, max_iterations=2,
                                                   stop_threshold_change_pct=0.0)
    assert info['regularization_params']['sigma_x'] == denoising._SIGMA_X_FLOOR
    assert np.all(np.isfinite(stack_out))
    assert np.array_equal(stack_out, zeros)


# ── denoise_stack: a stack of volumes against a loop of single volumes ───────

def _ramp_stack(num_volumes, shape, seed=3, noise_lo=0.02, noise_hi=0.08):
    """num_volumes noisy copies of one ramp volume.  The noise amplitude rises
    with the volume index, so the volumes reach the stopping threshold at
    different iterations."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, int(np.prod(shape))).reshape(shape)
    amplitudes = np.linspace(noise_lo, noise_hi, num_volumes)
    return np.stack([base + a * rng.normal(size=shape)
                     for a in amplitudes]).astype(np.float32)


def _pinned_denoiser(shape, device, sigma_x=0.02):
    """A denoiser with its prior parameters pinned, so that every call is the
    same operator and the only statistic left to compute is none."""
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0, sigma_x=sigma_x,
                        auto_regularize_flag=False)
    return denoiser


def test_denoise_stack_equals_a_loop_of_denoise_calls(device):
    """The stack sweep gives each volume its own step size and its own
    stopping test, so it must reproduce a loop of single-volume denoise calls
    with the same parameters and the same seeded partition, volume by volume,
    including the iteration count at which each volume stops.  The noise
    amplitude differs per volume so that the counts differ.

    Auto-regularization is on, so the sweep also sets sigma_x from the stack.
    That value is checked against the method that computes it, and the loop is
    pinned to it: the parameter moved and the sweep did not.

    The gate is a relative maximum difference of 1e-6, not equality: the
    line-search sums are reductions, and a reduction over one axis of a 3D
    tensor need not accumulate in the order of a full reduction of a 2D one.
    The difference seen is printed."""
    shape = (8, 10, 12)
    num_volumes = 6
    sigma_noise = 0.1
    stack = _ramp_stack(num_volumes, shape)

    auto = mbirtorch.QGGMRFDenoiser(shape)
    auto.configure_devices(devices=[device])
    auto.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    denoised, info = auto.denoise_stack(stack, sigma_noise=sigma_noise,
                                        max_iterations=15,
                                        stop_threshold_change_pct=0.2)

    # The parameter moved: the reported sigma_x is the method's own value.
    reported = info['regularization_params']['sigma_x']
    from_method = _auto_denoiser(shape, device, sigma_noise) \
        .auto_set_regularization_params_from_stack(stack)['sigma_x']
    rel_param = _rel(reported, from_method)
    assert rel_param < 1e-6

    # And the sweep did not: it is the loop of single-volume calls with
    # sigma_x pinned to that value on the same seeded partition.
    reference = np.empty_like(stack)
    reference_counts = []
    single = _pinned_denoiser(shape, device, sigma_x=reported)
    for volume in range(num_volumes):
        np.random.seed(0)     # the same partition for every volume
        out, out_dict = single.denoise(stack[volume], sigma_noise=sigma_noise,
                                       max_iterations=15,
                                       stop_threshold_change_pct=0.2,
                                       logfile_path=None, print_logs=False)
        reference[volume] = out
        reference_counts.append(int(out_dict['recon_params']['num_iterations']))

    rel = _rel_max(denoised, reference)
    counts = [int(n) for n in info['num_iterations']]
    print(f"denoise_stack vs loop of denoise on {device}: sigma_x {reported:.8g} "
          f"(rel {rel_param:.2e}), rel_max = {rel:.2e}, iteration counts {counts}")
    assert denoised.shape == stack.shape and denoised.dtype == np.float32
    assert rel < 1e-6
    assert counts == reference_counts
    assert len(set(counts)) >= 2, 'the volumes must stop at different iterations'
    assert [len(h) for h in info['nmae_pct']] == counts
    assert info['batch_size'] == num_volumes
    assert set(info['regularization_params']) == {'sigma_y', 'sigma_x', 'sigma_prox'}
    # The volumes changed: this is a denoise, not a copy.
    assert _rel_max(denoised, stack) > 1e-3


def test_denoise_stack_never_writes_the_caller_s_arrays(device):
    """The sweep writes its image in place, starting from the initial stack
    when one is given.  An initial stack that arrives as a copy, from the
    host or from another device, becomes that image directly; a caller's
    tensor already on the sweep device is cloned first.  Either way the call
    leaves both of the caller's arrays as it found them."""
    shape = (8, 10, 12)
    denoiser = _pinned_denoiser(shape, device)
    torch_device = denoiser.torch_device

    for kind in ('numpy', 'tensor on the sweep device'):
        stack = _ramp_stack(3, shape)
        init = np.zeros_like(stack) + 0.25
        if kind != 'numpy':
            stack = torch.as_tensor(stack).to(torch_device)
            init = torch.as_tensor(init).to(torch_device)
        before_stack = stack.clone() if torch.is_tensor(stack) else stack.copy()
        before_init = init.clone() if torch.is_tensor(init) else init.copy()

        out, _ = denoiser.denoise_stack(stack, sigma_noise=0.1, init_stack=init,
                                        max_iterations=2, stop_threshold_change_pct=0.0)
        moved = (float(torch.max(torch.abs(out - before_init))) if torch.is_tensor(out)
                 else float(np.max(np.abs(out - before_init))))
        print(f"{kind}: the sweep moved the image by {moved:.3e}")
        assert moved > 0                        # the sweep did run
        if torch.is_tensor(stack):
            assert torch.equal(stack, before_stack) and torch.equal(init, before_init)
        else:
            assert np.array_equal(stack, before_stack) and np.array_equal(init, before_init)


def test_one_initialization_fixes_the_stack_sweep(device):
    """The pixel grouping is a random draw, and at 16 subsets it moves the
    result by far more than the stopping threshold of a consensus loop.  A
    denoiser initialized once must therefore reuse that grouping: two sweeps
    with do_initialization=False agree to a relative maximum difference of
    1e-6, while two sweeps that draw again differ by more than 1e-3.  A False
    call that passes a new noise level uses that level and keeps the cache,
    since the noise level is one scalar and the grouping is not.  And a
    partition handed to initialize_denoiser is the one the cache holds and the
    sweep uses."""
    shape = (8, 10, 12)
    stack = _ramp_stack(4, shape)
    sweep = dict(sigma_noise=0.1, max_iterations=4, stop_threshold_change_pct=0.0)

    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0)
    settled = denoiser.initialize_denoiser(image=stack, sigma_noise=0.1)
    assert int(settled['partition'].shape[0]) == 16

    fixed_first, _ = denoiser.denoise_stack(stack, do_initialization=False, **sweep)
    fixed_again, _ = denoiser.denoise_stack(stack, do_initialization=False, **sweep)
    rel_fixed = _rel_max(fixed_again, fixed_first)

    drawn_first, _ = denoiser.denoise_stack(stack, **sweep)
    drawn_again, _ = denoiser.denoise_stack(stack, **sweep)
    rel_drawn = _rel_max(drawn_again, drawn_first)
    print(f"stack sweeps on {device}: two reused initializations differ by "
          f"{rel_fixed:.2e}, two fresh draws by {rel_drawn:.2e}")
    assert rel_fixed < 1e-6
    assert rel_drawn > 1e-3

    # A new noise level on a reused initialization: the level is used and the
    # grouping is kept.
    held = denoiser.denoise_data['partition']
    _out, info = denoiser.denoise_stack(stack, sigma_noise=0.25, max_iterations=1,
                                        stop_threshold_change_pct=0.0,
                                        do_initialization=False)
    assert denoiser.denoise_data['partition'] is held
    assert info['regularization_params']['sigma_y'] == pytest.approx(0.25)
    assert float(denoiser.get_params('sigma_y')) == pytest.approx(0.25)

    # A supplied partition.
    supplied = mbirtorch.gen_set_of_pixel_partitions(shape, [16], use_ror_mask=False)[0]
    cached = denoiser.initialize_denoiser(image=stack, sigma_noise=0.1,
                                          partition=supplied)['partition']
    assert torch.equal(cached.cpu(), supplied)
    denoiser.denoise_stack(stack, do_initialization=False, **sweep)
    assert denoiser.denoise_data['partition'] is cached


# ── the regularization parameters of a stack ─────────────────────────────────

def _auto_denoiser(shape, device, sigma_noise=0.1):
    """A denoiser with auto-regularization on and its noise level set, which
    is the state denoise_stack leaves before it sets the parameters."""
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0, sigma_noise=sigma_noise,
                        sigma_y=sigma_noise)
    return denoiser


def _rel(a, b):
    return abs(float(a) - float(b)) / abs(float(b))
