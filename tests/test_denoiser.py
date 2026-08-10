"""QGGMRFDenoiser gates: golden parity vs mbirjax and a denoising smoke on
every backend."""

import glob
import os

import numpy as np
import pytest

import mbirtorch

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
