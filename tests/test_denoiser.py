"""QGGMRFDenoiser gates: scale equivariance on every backend, with the
regularization automatic and pinned; the stopping rule's distance to the MAP
estimate, with and without an offset; the error for the old name of the stop
parameter; two shards against one device; an all-zero input; the stack
denoiser against a loop of single-volume calls; the caller's arrays left
unwritten; and one initialization reused across calls."""

import numpy as np
import pytest
import torch
from torch._dynamo.utils import counters as dynamo_counters

import mbirtorch
from mbirtorch import denoising


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize('prior', ['automatic', 'pinned'])
def test_denoise_is_scale_equivariant(device, prior):
    """Scaling the noisy image by c must scale the denoised image by c.

    The cost that denoise minimizes and the automatic estimates of
    sigma_noise and sigma_x all scale with the image, so the result must not
    depend on the units of the image.  An update whose step does not scale
    with the image fails this test, because it barely moves an image in large
    units.  The scales are powers of 2, so the scaling adds no rounding.

    The 'automatic' case estimates sigma_noise and sigma_x from the image.
    The 'pinned' case fixes sigma_x, as the Plug-and-Play agent does, and
    passes sigma_noise, which must then set sigma_y.  In both cases the
    denoised image must be at least 30% closer to the clean image than the
    noisy image is, because returning the input unchanged is also scale
    equivariant.

    Each scale gives new values of sigma_noise and sigma_x.  These values
    enter the compiled update as tensors, so the calls after the first must
    not compile it again."""
    shape = (32, 32, 8)
    clean = np.zeros(shape, dtype=np.float32)
    clean[8:-8, 8:-8, 2:-2] = 1.0
    noisy = clean + 0.1 * np.random.RandomState(2).randn(*shape).astype(np.float32)

    def denoise_scaled(scale):
        """Return the denoised image of scale * noisy, divided by scale."""
        denoiser = mbirtorch.QGGMRFDenoiser(shape)
        denoiser.configure_devices(devices=[device])
        denoiser.set_params(no_warning=True, verbose=0)
        sigma_noise = None
        if prior == 'pinned':
            denoiser.set_params(no_warning=True, sigma_x=0.05 * scale,
                                auto_regularize_flag=False)
            sigma_noise = 0.1 * scale
        np.random.seed(0)     # the same partition at every scale
        out, _ = denoiser.denoise(scale * noisy, sigma_noise=sigma_noise,
                                  max_iterations=5, stop_threshold=0.0,
                                  logfile_path=None, print_logs=False)
        if prior == 'pinned':
            assert float(denoiser.get_params('sigma_y')) == pytest.approx(sigma_noise)
        return np.asarray(out, dtype=np.float64) / scale

    reference = denoise_scaled(1.0)
    err_noisy = np.linalg.norm(noisy - clean)
    err_den = np.linalg.norm(reference - clean)
    assert err_den < 0.7 * err_noisy, (err_den, err_noisy)
    # A recompilation is counted rather than made an error, because the
    # compiled wrappers catch an error and fall back to eager.
    graphs_before = dynamo_counters['stats']['unique_graphs']
    for scale in (2.0 ** 10, 2.0 ** -10):
        rel = _rel_max(denoise_scaled(scale), reference)
        print(f"{prior} on {device}: scale {scale:g} differs from scale 1 by {rel:.2e}")
        assert rel < 1e-5, (scale, rel)
    new_graphs = dynamo_counters['stats']['unique_graphs'] - graphs_before
    assert new_graphs == 0, f'{new_graphs} new compiled graphs for new noise levels'


def test_gradient_rule_stops_within_its_threshold_of_the_map_estimate(device):
    """The sweep stops when sigma_noise times the rms gradient of the cost
    falls below stop_threshold.  At any image, sigma_noise times the rms
    gradient bounds the rms distance to the MAP estimate in units of
    sigma_noise.  The sweep takes each gradient during the sweep rather than
    at the image it ends with, so its statistic only approximates that
    bound.  Each sweep must still end within its threshold of the MAP
    estimate, which is the same sweep run to a threshold of 1e-4.

    The prior is strong, with sigma_x / sigma_noise = 0.1, so the sweeps need
    many iterations.  The same volume is also denoised in HU-like units,
    scaled by 1000 with 1000 subtracted, so that air lies at -1000.  The old
    rule, a percent change of the image, stopped early on such an image,
    because the offset inflates the image's norm.  The rule on the gradient
    does not depend on an offset."""
    shape = (24, 24, 8)
    clean = np.zeros(shape, dtype=np.float32)
    clean[6:-6, 6:-6, 2:-2] = 1.0
    noisy = clean + 0.1 * np.random.RandomState(5).randn(*shape).astype(np.float32)

    def run(scale, offset, threshold, max_iterations=200):
        """Return the denoised image of scale * noisy + offset, mapped back by
        the inverse transform, and the gradient statistic of each iteration."""
        denoiser = mbirtorch.QGGMRFDenoiser(shape)
        denoiser.configure_devices(devices=[device])
        denoiser.set_params(no_warning=True, verbose=0, sigma_x=0.01 * scale,
                            auto_regularize_flag=False)
        np.random.seed(0)
        out, out_dict = denoiser.denoise(scale * noisy + offset, sigma_noise=0.1 * scale,
                                         max_iterations=max_iterations,
                                         stop_threshold=threshold,
                                         logfile_path=None, print_logs=False)
        out = (np.asarray(out, dtype=np.float64) - offset) / scale
        return out, out_dict['recon_params']['gradient_statistic']

    map_estimate, statistics = run(1.0, 0.0, 1e-4, max_iterations=2000)
    assert statistics[-1] < 1e-4
    for scale, offset in ((1.0, 0.0), (1000.0, -1000.0)):
        for threshold in (0.1, 0.03, 0.01):
            out, statistics = run(scale, offset, threshold)
            distance = float(np.sqrt(np.mean((out - map_estimate) ** 2))) / 0.1
            print(f"scale {scale:g}, offset {offset:g}, threshold {threshold}: "
                  f"{len(statistics)} iterations, last statistic {statistics[-1]:.4g}, "
                  f"distance to the MAP estimate {distance:.4g}")
            assert len(statistics) < 200 and statistics[-1] < threshold
            assert distance < threshold


def test_the_old_stop_parameter_raises_an_error_that_names_the_new_one():
    """stop_threshold replaced stop_threshold_change_pct, and its unit changed
    from percent to sigma_noise.  A call that passes the old name must fail
    with a message that names the new parameter."""
    shape = (8, 10, 12)
    denoiser = _pinned_denoiser(shape, 'cpu')
    stack = _ramp_stack(2, shape)
    with pytest.raises(TypeError, match='Use stop_threshold instead'):
        denoiser.denoise(stack[0], sigma_noise=0.1, stop_threshold_change_pct=0.2)
    with pytest.raises(TypeError, match='Use stop_threshold instead'):
        denoiser.denoise_stack(stack, sigma_noise=0.1, stop_threshold_change_pct=0.2)


def test_sharded_denoise_matches_single_device():
    """Two CPU shards vs one device on the same seeded problem: the denoised
    volume, the iteration at which the default stopping rule ends each sweep,
    the gradient statistic at each iteration, and the automatically set
    regularization parameters must all agree with the single-device run.

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
    ref, ref_dict = ref_den.denoise(noisy, logfile_path=None)

    sh_den = mbirtorch.QGGMRFDenoiser(shape)
    sh_den.configure_devices(devices=['cpu', 'cpu'])
    sh_den.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    out, out_dict = sh_den.denoise(sh_den._shard_recon(noisy), logfile_path=None,
                                   output_sharded=True)
    out = np.asarray(out.gather())

    assert out.shape == ref.shape
    rel = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    ref_statistics = np.array(ref_dict['recon_params']['gradient_statistic'])
    out_statistics = np.array(out_dict['recon_params']['gradient_statistic'])
    print(f"sharded vs single denoise rel_max = {rel:.2e}, iterations "
          f"{len(out_statistics)} vs {len(ref_statistics)}")
    assert rel < 1e-4
    assert len(out_statistics) == len(ref_statistics)
    assert _rel_max(out_statistics, ref_statistics) < 1e-5
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
                              stop_threshold=0.0)
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
                                                   stop_threshold=0.0)
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
    threshold = 0.03
    stack = _ramp_stack(num_volumes, shape)

    auto = mbirtorch.QGGMRFDenoiser(shape)
    auto.configure_devices(devices=[device])
    auto.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    denoised, info = auto.denoise_stack(stack, sigma_noise=sigma_noise,
                                        stop_threshold=threshold)

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
    reference_statistics = []
    single = _pinned_denoiser(shape, device, sigma_x=reported)
    for volume in range(num_volumes):
        np.random.seed(0)     # the same partition for every volume
        out, out_dict = single.denoise(stack[volume], sigma_noise=sigma_noise,
                                       stop_threshold=threshold,
                                       logfile_path=None, print_logs=False)
        reference[volume] = out
        reference_counts.append(int(out_dict['recon_params']['num_iterations']))
        reference_statistics.append(out_dict['recon_params']['gradient_statistic'])

    rel = _rel_max(denoised, reference)
    counts = [int(n) for n in info['num_iterations']]
    print(f"denoise_stack vs loop of denoise on {device}: sigma_x {reported:.8g} "
          f"(rel {rel_param:.2e}), rel_max = {rel:.2e}, iteration counts {counts}")
    assert denoised.shape == stack.shape and denoised.dtype == np.float32
    assert rel < 1e-6
    assert counts == reference_counts
    assert len(set(counts)) >= 2, 'the volumes must stop at different iterations'
    assert [len(h) for h in info['nmae_pct']] == counts
    # Each volume stopped by the rule, and its gradient statistic at each
    # iteration is that of its own denoise call.
    for statistics, reference_values in zip(info['gradient_statistic'], reference_statistics):
        assert statistics[-1] < threshold
        assert _rel_max(statistics, reference_values) < 1e-5
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
                                        max_iterations=2, stop_threshold=0.0)
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
    sweep = dict(sigma_noise=0.1, max_iterations=4, stop_threshold=0.0)

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
                                        stop_threshold=0.0,
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


# ── the noise estimate ───────────────────────────────────────────────────────

def test_noise_estimate_returns_the_level_of_white_and_correlated_noise():
    """The noise estimate is the median absolute difference of voxels 5
    apart, divided by sqrt(2) times 0.6745.  On a volume with no edges it must
    return the noise level to within 2%, both for white noise and for noise
    correlated between neighbors.  The correlated noise is white noise
    blurred by a Gaussian of 1 voxel, which gives neighbors a correlation of
    0.78 and makes adjacent voxels differ less.  An offset of -1000, as an
    image in HU has, must not change the estimate, and a region of exact
    zeros, as a mask leaves, must not lower it."""
    from scipy import ndimage

    shape = (64, 64, 32)
    sigma = 0.1
    white = np.random.default_rng(7).standard_normal(shape)
    correlated = ndimage.gaussian_filter(white, 1.0)
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    for name, noise in (('white', white), ('correlated', correlated)):
        volume = (1.0 + sigma * noise / noise.std()).astype(np.float32)
        estimate = denoiser.estimate_image_noise_std(volume)
        shifted = denoiser.estimate_image_noise_std(volume - 1000)
        masked = volume.copy()
        masked[:16] = 0
        in_mask = denoiser.estimate_image_noise_std(masked)
        print(f"{name} noise: estimate / sigma {estimate / sigma:.4f}, with an offset "
              f"{shifted / sigma:.4f}, with a masked region {in_mask / sigma:.4f}")
        assert abs(estimate / sigma - 1) < 0.02
        assert abs(shifted / estimate - 1) < 1e-3
        assert abs(in_mask / sigma - 1) < 0.02
