"""QGGMRFDenoiser gates: golden parity vs mbirjax, a denoising smoke on
every backend, and the stack denoiser against a loop of single-volume
calls."""

import glob
import os

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _memory_ledger, _sharding, denoising
from mbirtorch import qggmrf as _qggmrf
from mbirtorch._memory_ledger import image_ell1, stack_ell1

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


def test_ell1_reductions_match_a_float64_reference(device):
    """The reductions behind the reported nmae, checked where the goldens
    cannot check them.

    The golden image is far below one chunk, so the goldens only ever exercise
    the unchunked branch.  Below one chunk the reduction is the sum(abs) it
    replaced, so it must match bit for bit.  Above one chunk the result is
    scored against a float64 reference over the same float32 values, which
    measures the reduction's own arithmetic rather than the denoiser's: a
    reduction that accumulates float32 sequentially instead of pairwise drifts
    as the element count grows, which is why torch.linalg.vector_norm is not
    used here.  The per-volume form used by the stack sweep's stopping test is
    checked against the single-image form on each volume."""
    torch.manual_seed(11)

    small = torch.randn(32 * 32, 32)
    assert (small.numel() * small.element_size()
            < _memory_ledger.ELL1_CHUNK_BYTES)
    assert float(image_ell1(small)) == float(torch.sum(torch.abs(small)))

    shape = (256, 256, 256)
    flat = torch.randn(shape[0] * shape[1], shape[2])
    assert flat.numel() * flat.element_size() > _memory_ledger.ELL1_CHUNK_BYTES
    reference = float(flat.double().abs().sum())
    value = float(image_ell1(flat.to(device)))
    rel = abs(value - reference) / abs(reference)
    print(f"image_ell1 on {device}: rel vs float64 = {rel:.2e}")
    assert rel < 1e-5

    small_stack = torch.randn(3, 32 * 32, 32, device=device)
    assert small_stack.numel() * small_stack.element_size() < _memory_ledger.ELL1_CHUNK_BYTES
    per_volume = torch.stack([image_ell1(small_stack[b]) for b in range(3)])
    rel = float((stack_ell1(small_stack) - per_volume).abs().max() / per_volume.abs().max())
    assert rel < 1e-6

    large_stack = torch.randn(4, 512 * 512, 8, device=device)
    assert large_stack.numel() * large_stack.element_size() > _memory_ledger.ELL1_CHUNK_BYTES
    # The float64 reference is formed on the host: MPS has no float64.
    reference = large_stack.cpu().double().abs().sum(dim=(1, 2))
    rel = float((stack_ell1(large_stack).cpu().double() - reference).abs().max()
                / reference.abs().max())
    print(f"stack_ell1 on {device} at a size that chunks: rel vs float64 = {rel:.2e}")
    assert rel < 1e-5


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


def _as_shards(volume, n_shards):
    """Split a volume's slice axis over n CPU shards, as the denoiser does."""
    placement = _sharding.Placement(['cpu'] * n_shards, axis=-1,
                                    axis_len=volume.shape[2])
    tensors = [torch.as_tensor(volume[:, :, s0:s1].copy())
               for _dev, (s0, s1) in placement.shard_ranges()]
    return _sharding.Shards(tensors, placement)


def test_noise_estimate_is_the_same_sharded_or_not():
    """The noise estimate reads a strided subsample, and the subsample is the
    same numbers whether it comes from one array or from several shards, so
    the estimate is unchanged by the device layout.

    Exact equality is the right gate here: the subsample is assembled by
    moving data, and both paths hand numpy the same contiguous float32
    values, so the reductions have nothing to disagree about."""
    shape = (24, 24, 21)
    rng = np.random.default_rng(7)
    clean = np.zeros(shape, dtype=np.float32)
    clean[6:-6, 6:-6, 5:-5] = 1.0
    noisy = clean + 0.1 * rng.standard_normal(shape).astype(np.float32)

    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=['cpu'])
    denoiser.set_params(no_warning=True, verbose=0)

    reference = float(denoiser.estimate_image_noise_std(noisy))
    assert float(denoiser.estimate_image_noise_std(torch.as_tensor(noisy))) == reference
    for n_shards in (1, 2, 3):
        sharded = float(denoiser.estimate_image_noise_std(_as_shards(noisy, n_shards)))
        assert sharded == reference, (n_shards, sharded, reference)


def test_sharded_denoise_makes_no_whole_volume_host_transfer(monkeypatch):
    """A denoise on shards must not copy the volume to the host.

    Pinned two ways, because either alone could pass by accident.
    ``Shards.gather`` is the whole-volume host exit (it is what ``_to_host``
    calls), so requiring that it never runs catches a gather put back
    anywhere in the call.  And the elements the statistics do bring over are
    counted and compared against what the two subsampling rules ask for,
    which pins the AMOUNT rather than the route: a subsample quietly widened
    to fetch more would fail even though no gather ran.  Counting elements
    keeps the assertion on the data movement itself rather than on a timing
    or a peak-memory number that varies by machine.

    The image goes in sharded and ``output_sharded`` keeps the result there,
    which is the plug-and-play case this is about.  A caller who asks for a
    host array still gets one full-volume transfer at the end, by request.
    """
    shape = (700, 8, 21)      # tall enough that 20 sampled rows is a few percent
    rng = np.random.default_rng(9)
    clean = np.zeros(shape, dtype=np.float32)
    clean[100:-100, 2:-2, 5:-5] = 1.0
    noisy = clean + 0.1 * rng.standard_normal(shape).astype(np.float32)

    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=['cpu', 'cpu'])
    denoiser.set_params(no_warning=True, verbose=0)
    shards = denoiser._shard_recon(noisy)

    moved, gathered = [], []
    real_subsample = denoising._subsample_to_host
    real_gather = _sharding.Shards.gather

    def counting_subsample(image, *args, **kwargs):
        result = real_subsample(image, *args, **kwargs)
        moved.append(int(result.size))
        return result

    def counting_gather(self):
        result = real_gather(self)
        gathered.append(int(result.size))
        return result

    monkeypatch.setattr(denoising, '_subsample_to_host', counting_subsample)
    monkeypatch.setattr(_sharding.Shards, 'gather', counting_gather)

    volume_elements = shape[0] * shape[1] * shape[2]
    row_step = max(shape[0] // min(20, shape[0]), 1)
    row_elements = len(range(0, shape[0], row_step)) * shape[1] * shape[2]

    # With sigma_noise given -- as a plug-and-play loop gives it -- the only
    # statistic left is the auto-regularization one, which reads about 20 rows.
    np.random.seed(0)
    denoiser.denoise(shards, sigma_noise=0.1, max_iterations=1,
                     stop_threshold_change_pct=0.0, logfile_path=None,
                     output_sharded=True)
    print(f"sigma given: moved {sum(moved)} of {volume_elements} elements "
          f"({100 * sum(moved) / volume_elements:.2f} percent)")
    assert gathered == []
    assert moved == [row_elements]
    assert sum(moved) < 0.05 * volume_elements

    # With sigma_noise unset the noise estimate runs as well.  Its stride rule
    # is unchanged, and at this size it asks for a stride of 1, so it reads
    # the whole volume here; the claim is that it takes exactly the strided
    # grid it asks for and nothing more.  The same rule gives stride 6 on a
    # 1024-cubed volume, which is well under one percent of it.
    del moved[:]
    stride = round((volume_elements / min(5_000_000, volume_elements)) ** (1 / 3))
    noise_elements = int(np.prod([len(range(0, n, stride)) for n in shape]))
    np.random.seed(0)
    denoiser.denoise(shards, max_iterations=1, stop_threshold_change_pct=0.0,
                     logfile_path=None, output_sharded=True)
    assert gathered == []
    assert moved == [noise_elements, row_elements]


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


def test_batched_gradient_and_hessian_equal_the_single_image_function(device):
    """The batched prior repeats the single-image formulas with a leading
    volume axis, so on each volume of a stack it must give the single-image
    values.  Both run eagerly, on subset 0 of a seeded partition.  The gate is
    a relative maximum difference rather than equality, because a gather and
    an elementwise chain on a 3D tensor need not round exactly as on a 2D
    one; the difference seen is printed."""
    shape = (6, 24, 24)
    num_volumes = 4
    stack = _ramp_stack(num_volumes, shape)
    denoiser = _pinned_denoiser(shape, device, sigma_x=0.05)
    nbr_wts, sigma_x, p, q, T = denoiser.get_params(
        ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
    qggmrf_params = (_qggmrf.get_b_from_nbr_wts(nbr_wts), sigma_x, p, q, T)
    np.random.seed(0)
    partition = mbirtorch.gen_set_of_pixel_partitions(
        shape, [4], device=device, use_ror_mask=False)[0]
    flat = torch.as_tensor(stack, device=device).reshape(
        num_volumes, shape[0] * shape[1], shape[2])

    with torch.no_grad():
        grad_b, hess_b = _qggmrf.qggmrf_gradient_and_hessian_batched(
            flat, shape, partition[0], qggmrf_params)
        worst = 0.0
        for volume in range(num_volumes):
            grad, hess = _qggmrf.qggmrf_gradient_and_hessian_at_indices(
                flat[volume], shape, partition[0], qggmrf_params)
            worst = max(worst, _rel_max(grad_b[volume].cpu(), grad.cpu()),
                        _rel_max(hess_b[volume].cpu(), hess.cpu()))
    print(f"batched gradient and hessian vs single image on {device}: "
          f"rel_max = {worst:.2e}")
    assert grad_b.shape == (num_volumes, partition.shape[1], shape[2])
    assert worst < 1e-7


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


def test_denoise_stack_padded_last_batch_matches_one_batch(device):
    """Five volumes in batches of two leave a last batch of one, which is
    padded to two by repeating its volume.  The padding is discarded, so the
    result must match one batch of five.  The volumes are independent, so the
    gate is float rounding, and the difference seen is printed."""
    shape = (8, 10, 12)
    stack = _ramp_stack(5, shape)
    denoiser = _pinned_denoiser(shape, device)

    np.random.seed(0)
    whole, whole_info = denoiser.denoise_stack(stack, sigma_noise=0.1)
    np.random.seed(0)
    batched, batched_info = denoiser.denoise_stack(stack, sigma_noise=0.1,
                                                   batch_size=2)
    rel = _rel_max(batched, whole)
    print(f"batches of 2 vs one batch of 5 on {device}: rel_max = {rel:.2e}")
    assert whole_info['batch_size'] == 5 and batched_info['batch_size'] == 2
    assert rel < 1e-6
    assert list(batched_info['num_iterations']) == list(whole_info['num_iterations'])


def test_denoise_stack_takes_an_init_stack_and_checks_arguments(device):
    """An initial stack starts the sweep where it says; a wrong shape in
    either argument, or a batch size below one, is refused before any
    computation.  A denoiser configured with two devices is refused as well:
    the stack sweep runs on one device rather than silently using the first.
    Two 'virtual' CPU devices build that layout, so it runs everywhere."""
    shape = (8, 10, 12)
    stack = _ramp_stack(3, shape)
    denoiser = _pinned_denoiser(shape, device)

    # Starting every volume at zero moves the result away from the default
    # start at the input, so the argument is read.
    np.random.seed(0)
    from_input, _ = denoiser.denoise_stack(stack, sigma_noise=0.1, max_iterations=1,
                                           stop_threshold_change_pct=0.0)
    np.random.seed(0)
    from_zero, _ = denoiser.denoise_stack(stack, sigma_noise=0.1, max_iterations=1,
                                          stop_threshold_change_pct=0.0,
                                          init_stack=np.zeros_like(stack))
    assert _rel_max(from_zero, from_input) > 1e-3

    with pytest.raises(ValueError):
        denoiser.denoise_stack(stack[0], sigma_noise=0.1)          # a single volume
    with pytest.raises(ValueError):
        denoiser.denoise_stack(stack[:, :, :, :-1], sigma_noise=0.1)
    with pytest.raises(ValueError):
        denoiser.denoise_stack(stack, sigma_noise=0.1, init_stack=stack[:2])
    with pytest.raises(ValueError):
        denoiser.denoise_stack(stack, sigma_noise=0.1, batch_size=0)

    two_devices = mbirtorch.QGGMRFDenoiser(shape)
    two_devices.configure_devices(devices=['cpu', 'cpu'])
    two_devices.set_params(no_warning=True, verbose=0)
    with pytest.raises(ValueError, match='one device'):
        two_devices.denoise_stack(stack, sigma_noise=0.1)


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


def test_overwrite_input_takes_the_caller_s_tensors_over(device):
    """With ``overwrite_input`` a float32 tensor already on the sweep device
    becomes the sweep's own: the input is the working image when no initial
    stack is given, and the residual's buffer when one is, while the initial
    stack becomes the image.  The result is the one the default gives, and a
    numpy array is left alone whatever the flag says."""
    shape = (8, 10, 12)
    denoiser = _pinned_denoiser(shape, device)
    torch_device = denoiser.torch_device
    kwargs = dict(sigma_noise=0.1, max_iterations=2, stop_threshold_change_pct=0.0)

    # No initial stack: the input becomes the image, and the result is it.
    stack = torch.as_tensor(_ramp_stack(3, shape)).to(torch_device)
    np.random.seed(0)
    reference, _ = denoiser.denoise_stack(stack.clone(), **kwargs)
    np.random.seed(0)
    out, _ = denoiser.denoise_stack(stack, overwrite_input=True, **kwargs)
    rel = _rel_max(out.cpu().numpy(), reference.cpu().numpy())
    print(f"overwrite_input on {device}: rel_max = {rel:.2e}")
    assert rel < 1e-6
    assert out.data_ptr() == stack.data_ptr() and torch.equal(out, stack)

    # An initial stack: the image is the initial stack, and the input holds
    # the final residual, the input minus the image.
    stack = torch.as_tensor(_ramp_stack(3, shape)).to(torch_device)
    init = torch.zeros_like(stack) + 0.25
    before = stack.clone()
    np.random.seed(0)
    reference, _ = denoiser.denoise_stack(stack.clone(), init_stack=init.clone(), **kwargs)
    np.random.seed(0)
    out, _ = denoiser.denoise_stack(stack, init_stack=init, overwrite_input=True, **kwargs)
    assert _rel_max(out.cpu().numpy(), reference.cpu().numpy()) < 1e-6
    assert out.data_ptr() == init.data_ptr()
    assert torch.allclose(stack, before - out, atol=1e-5)

    # A numpy array is never written.
    stack = _ramp_stack(3, shape)
    before = stack.copy()
    np.random.seed(0)
    out, _ = denoiser.denoise_stack(stack, overwrite_input=True, **kwargs)
    assert isinstance(out, np.ndarray) and np.array_equal(stack, before)
    assert _rel_max(out, before) > 1e-3


# ── the regularization parameters of a stack ─────────────────────────────────

def _auto_denoiser(shape, device, sigma_noise=0.1):
    """A denoiser with auto-regularization on and its noise level set, which
    is the state denoise_stack leaves before it sets the parameters."""
    denoiser = mbirtorch.QGGMRFDenoiser(shape)
    denoiser.configure_devices(devices=[device])
    denoiser.set_params(no_warning=True, verbose=0, sigma_noise=sigma_noise,
                        sigma_y=sigma_noise)
    return denoiser


def _whole_stack_sigma_x(denoiser, volumes):
    """The reference statistic: the estimator on the given volumes merged into
    one 3D array with no subsampling, followed by the sigma_x rule."""
    merged = np.asarray(volumes).reshape(-1, volumes.shape[2], volumes.shape[3])
    indicator = denoiser._get_sino_indicator(merged)
    recon_std = denoiser._get_estimate_of_recon_std(merged, indicator)
    return 0.2 * (2 ** denoiser.get_params('sharpness')) * recon_std


def _rel(a, b):
    return abs(float(a) - float(b)) / abs(float(b))


def test_stack_regularization_uses_the_chosen_volumes(device, monkeypatch):
    """sigma_x is the estimator on the chosen volumes, merged into one 3D
    array with its neighbor differences between adjacent frames.

    Up to 39 volumes every volume is chosen, so sigma_x is the whole-stack
    statistic.  Above 39 about 20 are chosen, evenly spaced, by the rule
    subsample_views applies to views, and only those volumes cross to the
    host: a stack on the device is indexed there and nothing gathers the whole
    of it.  Which volumes move is data movement and is checked exactly; the
    statistics are computed floats stored as float32, so they are gated
    relatively.  With auto-regularization off the method changes nothing and
    returns the current values."""
    shape = (8, 10, 12)
    stack = _ramp_stack(6, shape)
    denoiser = _auto_denoiser(shape, device)
    params = denoiser.auto_set_regularization_params_from_stack(stack)
    expected = _whole_stack_sigma_x(denoiser, stack)
    rel = _rel(params['sigma_x'], expected)
    print(f"sigma_x from the method {params['sigma_x']:.8g} vs whole stack "
          f"{expected:.8g} (rel {rel:.2e})")
    assert set(params) == {'sigma_y', 'sigma_x', 'sigma_prox'}
    assert rel < 1e-6
    assert params['sigma_y'] == pytest.approx(0.1)
    assert _rel(params['sigma_prox'], expected) < 1e-6
    assert denoiser.get_params('sigma_x') == params['sigma_x']

    denoiser.set_params(no_warning=True, sigma_x=0.5, auto_regularize_flag=False)
    unchanged = denoiser.auto_set_regularization_params_from_stack(stack)
    assert unchanged['sigma_x'] == 0.5
    assert denoiser.get_params('sigma_x') == 0.5

    # A stack of 60 volumes: 20 are chosen, every third one.
    large_shape = (6, 16, 16)
    num_volumes = 60
    large = _ramp_stack(num_volumes, large_shape)
    on_host = _auto_denoiser(large_shape, device)
    chosen = on_host.subsample_views(np.arange(num_volumes))
    assert len(chosen) == 20 and chosen[1] - chosen[0] == 3

    moved, gathered = [], []
    real_subsample = denoising._subsample_to_host
    real_gather = _sharding.Shards.gather

    def recording_subsample(image, *args, **kwargs):
        result = real_subsample(image, *args, **kwargs)
        moved.append(result)
        return result

    def counting_gather(self):
        result = real_gather(self)
        gathered.append(int(result.size))
        return result

    monkeypatch.setattr(denoising, '_subsample_to_host', recording_subsample)
    monkeypatch.setattr(_sharding.Shards, 'gather', counting_gather)

    from_numpy = on_host.auto_set_regularization_params_from_stack(large)
    assert len(moved) == 1
    assert np.array_equal(moved[0].reshape((len(chosen),) + large_shape), large[chosen])
    assert gathered == []

    expected = _whole_stack_sigma_x(on_host, large[chosen])
    rel = _rel(from_numpy['sigma_x'], expected)
    print(f"sigma_x from the method {from_numpy['sigma_x']:.8g} vs the chosen volumes "
          f"{expected:.8g} (rel {rel:.2e})")
    assert rel < 1e-6

    # The same stack as a tensor on the device gives the same parameters, and
    # moves the same elements.
    del moved[:]
    on_device = torch.as_tensor(large).to(device)
    from_tensor = _auto_denoiser(large_shape, device) \
        .auto_set_regularization_params_from_stack(on_device)
    assert gathered == []
    assert [int(m.size) for m in moved] == [len(chosen) * int(np.prod(large_shape))]
    for name in from_numpy:
        rel = _rel(from_tensor[name], from_numpy[name])
        print(f"{name}: tensor {from_tensor[name]:.8g} vs numpy {from_numpy[name]:.8g} "
              f"(rel {rel:.2e})")
        assert rel < 1e-6, name


def test_the_recon_std_estimate_matches_the_form_it_replaced(device):
    """The estimate is read from shifted views of the image instead of from
    an index array, which holds about a quarter of the memory.  The value is
    unchanged: the two forms agree on a random image, on an object with a
    background around it, and on a ramp, over the whole image and over the
    thresholded support."""
    def gathered(image, support):
        """The form this replaced: gather the voxel and its three backward
        neighbors through np.where, then take the spread of the four."""
        inds = np.where(support)
        values = np.stack([image[inds[0], inds[1], inds[2]],
                           image[inds[0] - 1, inds[1], inds[2]],
                           image[inds[0], inds[1] - 1, inds[2]],
                           image[inds[0], inds[1], inds[2] - 1]], axis=0)
        return np.mean(np.std(values, axis=0))

    shape = (20, 24, 18)
    rng = np.random.default_rng(0)
    images = {'random': rng.standard_normal(shape).astype(np.float32)}
    block = np.zeros(shape, np.float32)
    block[5:15, 6:18, 4:14] = 1.0
    images['object'] = block + 0.05 * rng.standard_normal(shape).astype(np.float32)
    axes = np.meshgrid(*[np.linspace(0, 1, n) for n in shape], indexing='ij')
    images['ramp'] = (axes[0] + 2 * axes[1] - axes[2]).astype(np.float32)

    denoiser = _auto_denoiser(shape, device)
    for name, image in images.items():
        for label, support in (('whole', np.ones(shape, np.int8)),
                               ('support', denoiser._get_sino_indicator(image, sigma_noise=0.0))):
            new_value = denoiser._get_estimate_of_recon_std(image, support)
            old_value = gathered(image, support)
            rel = _rel(float(new_value), float(old_value))
            print(f"{name} over the {label}: {float(new_value):.9g} vs {float(old_value):.9g} "
                  f"(rel {rel:.2e})")
            assert rel < 1e-6


def test_the_whole_volume_statistics_keep_a_point_budget():
    """The statistics read at most five million voxels.  Above that the chosen
    volumes are sampled as a grid of contiguous tiles in the two trailing
    axes, so that a neighbor difference inside a tile is still between
    adjacent voxels and the sample still covers the field of view.  A stack
    under the budget is not sampled at all.

    The tiles are centered on equal shares of each axis rather than placed
    against its edges: at the production budget a tile is nine voxels wide,
    and edge placement would read the four corners of a 512 by 512 field,
    where a reconstruction holds only air.

    Entry 0 of each tiled axis is dropped from the support, because the
    estimate reads its neighbor with a wrap and would compare that entry with
    the far side of the tile.  Over a whole volume that is one plane in
    hundreds; over a tile it is one column in the tile's width, which moves
    the estimate by up to 23 percent on a structured image.
    """
    rows, cols = denoising._sample_tiles(512, 512, 30 * 512,
                                         denoising._STATISTICS_POINT_BUDGET)
    edge = rows[0].stop - rows[0].start
    assert len(rows) * len(cols) * edge * (cols[0].stop - cols[0].start) * 30 * 512 \
        <= denoising._STATISTICS_POINT_BUDGET
    # No tile against an edge, and the tiles span the axis.
    assert rows[0].start > 0 and rows[-1].stop < 512
    assert rows[-1].start - rows[0].start > 512 // 4
    # Under the budget the whole axis is kept, as one tile.
    assert denoising._sample_tiles(40, 40, 8, denoising._STATISTICS_POINT_BUDGET) == \
        ([slice(0, 40)], [slice(0, 40)])

    def sigma_x_at(stack, budget, monkey):
        monkey.setattr(denoising, '_STATISTICS_POINT_BUDGET', budget)
        denoiser = _auto_denoiser(tuple(int(n) for n in stack.shape[1:]), 'cpu')
        return denoiser.auto_set_regularization_params_from_stack(stack)['sigma_x']

    rng = np.random.default_rng(3)
    shape = (40, 8, 300, 300)                      # 20 chosen volumes: 14.4M voxels
    # A positive level well above sigma_noise, so the support is the whole
    # volume rather than the bright tail of a zero-mean image.
    uniform = (1.0 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    with pytest.MonkeyPatch.context() as monkey:
        sampled = sigma_x_at(uniform, denoising._STATISTICS_POINT_BUDGET, monkey)
        whole = sigma_x_at(uniform, 10 ** 12, monkey)
    print(f"uniform stack: sampled {sampled:.9g}, whole {whole:.9g} "
          f"(rel {_rel(sampled, whole):.2e})")
    assert _rel(sampled, whole) < 1e-3

    # An object with a boundary: the tiles read a sample of it, so the two
    # differ.  Measured across five positions of the boundary and several
    # stack shapes, the difference ran from 2e-3 to 7e-2, against 2e-2 to
    # 4e-2 for a single block of the same area.  How much depends on how the
    # tiles fall against the boundary, so the gate is set where a gross
    # regression shows and no tighter.  Even 1e-1 is well below the factor
    # of two per unit that sharpness moves sigma_x by.
    structured = uniform.copy()
    structured[:, :, 60:240, 60:240] += 1.0
    with pytest.MonkeyPatch.context() as monkey:
        sampled = sigma_x_at(structured, denoising._STATISTICS_POINT_BUDGET, monkey)
        whole = sigma_x_at(structured, 10 ** 12, monkey)
    print(f"object stack: sampled {sampled:.9g}, whole {whole:.9g} "
          f"(rel {_rel(sampled, whole):.2e})")
    assert _rel(sampled, whole) < 1e-1

    small = uniform[:6, :, :40, :40].copy()        # 6 * 8 * 40 * 40 = 76800 voxels
    with pytest.MonkeyPatch.context() as monkey:
        under = sigma_x_at(small, denoising._STATISTICS_POINT_BUDGET, monkey)
        whole_small = sigma_x_at(small, 10 ** 12, monkey)
    assert under == whole_small
