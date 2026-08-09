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
