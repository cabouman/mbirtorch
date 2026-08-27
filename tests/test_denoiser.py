"""QGGMRFDenoiser gates: golden parity vs mbirjax and a denoising smoke on
every backend."""

import glob
import os

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _memory_ledger, _sharding, denoising
from mbirtorch._memory_ledger import image_ell1

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


def test_image_ell1_is_accurate_at_a_size_that_chunks(device):
    """The reduction behind the reported nmae, checked where the goldens
    cannot check it.

    The golden image is far below one chunk, so the goldens only ever exercise
    the unchunked branch.  This runs an image large enough to chunk and scores
    the result against a float64 reference over the same float32 values, so it
    measures the reduction's own arithmetic rather than the denoiser's.  The
    statistic is gated at 1e-3 relative, and this must hold with room to
    spare: a reduction that accumulates float32 sequentially instead of
    pairwise drifts past that gate as the element count grows, which is why
    torch.linalg.vector_norm is not used here.
    """
    shape = (256, 256, 256)
    torch.manual_seed(11)
    flat = torch.randn(shape[0] * shape[1], shape[2])
    assert flat.numel() * flat.element_size() > _memory_ledger.ELL1_CHUNK_BYTES
    reference = float(flat.double().abs().sum())

    value = float(image_ell1(flat.to(device)))
    rel = abs(value - reference) / abs(reference)
    print(f"image_ell1 on {device}: rel vs float64 = {rel:.2e}")
    assert rel < 1e-5


def test_image_ell1_leaves_a_small_image_bit_for_bit():
    """Below one chunk the reduction is the sum(abs) it replaced, so small
    problems -- every golden among them -- cannot move at all."""
    small = torch.randn(32 * 32, 32)
    assert (small.numel() * small.element_size()
            < _memory_ledger.ELL1_CHUNK_BYTES)
    assert float(image_ell1(small)) == float(torch.sum(torch.abs(small)))


def test_sharded_denoise_matches_single_device():
    """Two CPU shards vs one device on the same seeded problem.  The sharded
    path stages halos once per pass and combines the step-size sums on the
    lead device, so agreement is at float level, not bitwise (gate per the
    measured iterated-comparison floor)."""
    shape = (24, 24, 21)   # 2 shards pad the slice axis 21 -> 22
    clean = np.zeros(shape, dtype=np.float32)
    clean[6:-6, 6:-6, 5:-5] = 1.0
    noisy = clean + 0.1 * np.random.RandomState(4).randn(*shape).astype(np.float32)

    ref_den = mbirtorch.QGGMRFDenoiser(shape)
    ref_den.configure_devices(devices=['cpu'])
    ref_den.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    ref, ref_dict = ref_den.denoise(noisy, sigma_noise=0.1, max_iterations=5,
                                    stop_threshold_change_pct=0.0, logfile_path=None)

    sh_den = mbirtorch.QGGMRFDenoiser(shape)
    sh_den.configure_devices(devices=['cpu', 'cpu'])
    sh_den.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    out, out_dict = sh_den.denoise(noisy, sigma_noise=0.1, max_iterations=5,
                                   stop_threshold_change_pct=0.0, logfile_path=None)

    assert out.shape == ref.shape
    rel = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    print(f"sharded vs single denoise rel_max = {rel:.2e}")
    assert rel < 1e-4
    # The denoiser dict now carries the run log and notes, like recon's.
    # (verbose=0 logs no iteration lines, so only the keys are checked.)
    assert 'recon_log' in out_dict and 'notes' in out_dict


def _as_shards(volume, n_shards):
    """Split a volume's slice axis over n CPU shards, as the denoiser does."""
    placement = _sharding.Placement(['cpu'] * n_shards, axis=-1,
                                    axis_len=volume.shape[2])
    tensors = [torch.as_tensor(volume[:, :, s0:s1].copy())
               for _dev, (s0, s1) in placement.shard_ranges()]
    return _sharding.Shards(tensors, placement)


@pytest.mark.parametrize('n_shards', [1, 2, 3])
def test_subsample_to_host_reproduces_striding_the_volume(n_shards):
    """Assembling a strided subsample from the shards is data movement, so
    the gate is exact equality with striding the whole volume, not a
    tolerance.

    Three shards matter: 23 slices split 8/8/7, so the shard boundaries fall
    at 8 and 16 and most of the strides below do not divide them.  That is
    what the per-shard starting offset exists for, and a subsample that
    restarted at each shard instead would disagree here while still matching
    on an evenly divided split."""
    rng = np.random.default_rng(5)
    volume = rng.standard_normal((9, 7, 23)).astype(np.float32)
    shards = _as_shards(volume, n_shards)
    assert denoising._volume_shape(shards) == volume.shape

    for row_step in (1, 2, 4):
        for col_step in (1, 3):
            for slice_step in (1, 2, 3, 5, 7, 8, 23, 30):
                assembled = denoising._subsample_to_host(
                    shards, row_step, col_step, slice_step)
                expected = volume[::row_step, ::col_step, ::slice_step]
                assert np.array_equal(assembled, expected), (
                    n_shards, row_step, col_step, slice_step)

    # The tensor and numpy forms take the same three strides.
    assert np.array_equal(denoising._subsample_to_host(torch.as_tensor(volume), 2, 3, 5),
                          volume[::2, ::3, ::5])
    assert np.array_equal(denoising._subsample_to_host(volume, 2, 3, 5),
                          volume[::2, ::3, ::5])


def test_subsample_to_host_handles_a_shard_with_no_slices():
    """More devices than slices leaves a trailing shard empty; it must
    contribute nothing rather than break the assembly."""
    rng = np.random.default_rng(6)
    volume = rng.standard_normal((4, 3, 2)).astype(np.float32)
    shards = _as_shards(volume, 3)
    assert [int(t.shape[-1]) for t in shards.tensors] == [1, 1, 0]
    assert denoising._volume_shape(shards) == volume.shape

    for slice_step in (1, 2, 3):
        assert np.array_equal(denoising._subsample_to_host(shards, 1, 1, slice_step),
                              volume[:, :, ::slice_step])


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


def test_sharded_denoise_sets_the_same_regularization_params():
    """Two shards with sigma_noise unset must reach the same regularization
    parameters as one device, since both statistics see the same subsample.

    The gate is relative, not exact: the two runs reach numpy by different
    routes -- a strided view of the caller's own array on one, a
    concatenation of per-shard copies on the other -- and float32 reductions
    need not accumulate in the same order over different memory layouts.  The
    claim being tested is about the statistics, not about the layout, so a
    float32-sized tolerance is what states it."""
    shape = (24, 24, 21)
    rng = np.random.default_rng(8)
    clean = np.zeros(shape, dtype=np.float32)
    clean[6:-6, 6:-6, 5:-5] = 1.0
    noisy = clean + 0.1 * rng.standard_normal(shape).astype(np.float32)

    ref_den = mbirtorch.QGGMRFDenoiser(shape)
    ref_den.configure_devices(devices=['cpu'])
    ref_den.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    _, ref_dict = ref_den.denoise(noisy, max_iterations=2,
                                  stop_threshold_change_pct=0.0, logfile_path=None)

    sh_den = mbirtorch.QGGMRFDenoiser(shape)
    sh_den.configure_devices(devices=['cpu', 'cpu'])
    sh_den.set_params(no_warning=True, verbose=0)
    np.random.seed(0)
    # Hand the two-device model its device form, which is what a caller that
    # keeps its volume on the devices does, and what puts the statistics on
    # the sharded path.
    _, out_dict = sh_den.denoise(sh_den._shard_recon(noisy), max_iterations=2,
                                 stop_threshold_change_pct=0.0, logfile_path=None,
                                 output_sharded=True)

    ref_params = ref_dict['recon_params']['regularization_params']
    out_params = out_dict['recon_params']['regularization_params']
    for name in ('sigma_x', 'sigma_prox', 'sigma_y'):
        rel = abs(out_params[name] - ref_params[name]) / abs(ref_params[name])
        print(f"{name}: sharded {out_params[name]:.8g} vs single "
              f"{ref_params[name]:.8g} (rel {rel:.2e})")
        assert rel < 1e-5, name


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


def test_median_filter3d_refuses_the_divided_form():
    """Each output voxel needs the 26 voxels around it, so a volume divided
    across devices on its slice axis would need its neighboring slices
    exchanged between them.  The filter refuses it by name instead of taking it
    for numpy, which fails on a torch dtype message that says nothing about
    where the array actually is.  Two 'virtual' CPU devices build the divided
    form, so this runs everywhere."""
    volume = np.random.RandomState(0).rand(6, 6, 8).astype(np.float32)
    placement = _sharding.Placement(['cpu', 'cpu'], axis=-1,
                                    axis_len=volume.shape[-1])
    divided = _sharding.Shards(
        [torch.as_tensor(volume[..., start:end].copy())
         for _, (start, end) in placement.shard_ranges()], placement)

    with pytest.raises(TypeError) as refusal:
        mbirtorch.median_filter3d(divided)
    message = str(refusal.value)
    assert 'median_filter3d' in message
    assert 'form: x.' in message          # the argument that was wrong
    assert 'divided device form' in message
    assert 'shards.gather()' in message

    # The whole-array forms still go through.
    assert mbirtorch.median_filter3d(volume).shape == volume.shape


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


def test_denoise_accepts_zero_image(device):
    """An all-zero image is a legitimate input (a Plug-and-Play loop
    initialized at zero feeds one in): the denoiser must return it unchanged
    instead of dividing its NMAE statistic by the zero image norm."""
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
