"""Sandbox: standard recon vs DRUNet postprocessing vs DRUNet-MACE.

Edit the parameters just below, run, and compare three reconstructions of
the 2D noisy cone-beam problem (from cone_beam_2d.py) side by side:

  1. the STANDARD recon -- recon() with the qGGMRF prior ("direct recon" is
     reserved for the FDK-style initialization recon_direct produces);
  2. DRUNet POSTPROCESSING -- one application of the denoiser to the
     standard recon;
  3. the DRUNet MACE recon -- the denoiser as the prior agent, equilibrated
     with the data-fit proximal map.

The standard recon is cached in ./output/ and reused while the problem is
unchanged, so parameter tweaks to the denoiser and the MACE loop rerun in
less time.  Run from this directory with the repo root on PYTHONPATH:

    PYTHONPATH=../.. python sandbox.py
"""

import os

import numpy as np
import torch

import mbirtorch

from agents import DRUNetAgent, ForwardProxAgent, load_drunet
from cone_beam_2d import get_data
from mace import mace

# ------------------------------------------------------------- parameters --

# DENOISER STRENGTH, in the network's own scale.  DRUNet is a Gaussian
# denoiser conditioned on the noise standard deviation of images valued in
# [0, 1].  The recon is mapped into that range by a fixed intensity scale c
# (chosen once: the standard recon's 99.9th-percentile value goes to 0.9),
# the network runs at noise level SIGMA_SCALED, and the result is mapped
# back, so in recon units the strength is SIGMA_SCALED / c.  For reference,
# a natural-image noise level of 25/255 is about 0.1.  The useful range here
# is roughly 0.02 (light) to 0.2 (heavy); in the 2026-08 sweep the best MACE
# recon used 0.075 and the best postprocessing 0.10.  In the MACE recon this
# is the one regularization-strength knob (the data-prox strength sigma_prox
# stays at the model's auto value).
SIGMA_SCALED = 0.075

# MACE outer iterations.  Each runs INNER_PROX_ITERATIONS of the data prox
# plus one DRUNet application.  30 gets close; 60 was converged (consensus
# spread below 1e-3) in the 2026-08 sweep.
NUM_ITERATIONS = 20

# Mann averaging parameter in (0, 1): 0.5 is the ADMM / Douglas-Rachford
# value; smaller damps the iteration (try 0.2-0.4 if it oscillates).
RHO = 0.5

# VCD iterations inside each data-prox call.
INNER_PROX_ITERATIONS = 3

# Sharpness of the standard recon (the usual image-quality control; changing
# it triggers a recompute of the cached standard recon).
SHARPNESS = 1.0

# Reuse the cached standard recon when the problem is unchanged (the cache
# is invalidated automatically if the sinogram or SHARPNESS changes).  Set
# False to force recomputation.
REUSE_STANDARD_RECON = True

# Show the interactive slice viewer at the end.
SHOW_VIEWER = True

CACHE_PATH = './output/sandbox_standard.npz'

# ----------------------------------------------------------------------------


def nrmse(a, b):
    """||a - b|| / ||b|| for same-device tensors."""
    return float(torch.sqrt(torch.sum((a - b) ** 2))
                 / torch.sqrt(torch.sum(b ** 2)))


def pick_device():
    """The same preference order the models use: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def standard_recon(ct_model, sinogram, weights):
    """The standard qGGMRF reconstruction and its auto sigma_prox, cached
    under CACHE_PATH and reused while the sinogram and SHARPNESS match."""
    fingerprint = float(np.sum(sinogram, dtype=np.float64))
    if REUSE_STANDARD_RECON and os.path.exists(CACHE_PATH):
        cache = np.load(CACHE_PATH)
        if (tuple(cache['sinogram_shape']) == tuple(sinogram.shape)
                and abs(float(cache['fingerprint']) - fingerprint)
                <= 1e-4 * abs(fingerprint)
                and float(cache['sharpness']) == SHARPNESS):
            print(f'Standard recon loaded from {CACHE_PATH} '
                  '(set REUSE_STANDARD_RECON = False to recompute).')
            return cache['standard'], float(cache['sigma_prox'])
    print('Computing the standard recon (cached for later runs).')
    np.random.seed(0)
    recon, recon_dict = ct_model.recon(
        sinogram, weights=weights, max_iterations=30,
        stop_threshold_change_pct=0.0, print_logs=False)
    sigma_prox = float(
        recon_dict['recon_params']['regularization_params']['sigma_prox'])
    os.makedirs('./output', exist_ok=True)
    np.savez(CACHE_PATH, standard=recon, sigma_prox=sigma_prox,
             sinogram_shape=np.array(sinogram.shape),
             fingerprint=fingerprint, sharpness=SHARPNESS)
    return recon, sigma_prox


def main():
    np.random.seed(0)
    phantom, sinogram, params = get_data()
    ct_model = mbirtorch.ConeBeamModel(
        sinogram.shape, params['angles'],
        source_detector_dist=params['source_detector_dist'],
        source_iso_dist=params['source_iso_dist'])
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')
    ct_model.set_params(sharpness=SHARPNESS)
    device = pick_device()
    ct_model.configure_devices(devices=[device])

    standard_np, sigma_prox = standard_recon(ct_model, sinogram, weights)
    standard = torch.as_tensor(standard_np, device=device)
    phantom_t = torch.as_tensor(phantom, device=device)

    # The fixed intensity scale mapping the recon into the network's [0, 1]
    # range; see the SIGMA_SCALED comment.
    robust_max = float(torch.quantile(standard.flatten(), 0.999))
    intensity_scale = 0.9 / robust_max
    sigma_noise = SIGMA_SCALED / intensity_scale

    if ct_model.get_params('use_ror_mask'):
        mask = mbirtorch.get_2d_ror_mask(tuple(standard.shape))
        ror_mask = torch.as_tensor(mask.astype(np.float32),
                                   device=device).unsqueeze(-1)
    else:
        ror_mask = None

    net = load_drunet(device)
    denoiser = DRUNetAgent(net, sigma_noise, intensity_scale,
                           ror_mask=ror_mask)

    print(f'Problem: {tuple(standard.shape)} recon on {device}, '
          f'sigma_scaled {SIGMA_SCALED:g} '
          f'({sigma_noise:.5f} in recon units), sigma_prox {sigma_prox:.5f}')

    with torch.no_grad():
        postproc = denoiser(standard)

    print(f'Running MACE: {NUM_ITERATIONS} outer iterations, rho {RHO:g}, '
          f'{INNER_PROX_ITERATIONS} prox inner iterations')
    np.random.seed(0)
    forward = ForwardProxAgent(
        ct_model, sinogram, weights=weights, sigma_prox=sigma_prox,
        inner_iterations=INNER_PROX_ITERATIONS,
        init_recon=standard.clone())

    def progress(iteration, x_bar):
        if (iteration + 1) % 5 == 0 or iteration == NUM_ITERATIONS - 1:
            print(f'  iteration {iteration + 1:3d}: '
                  f'NRMSE vs phantom {nrmse(x_bar, phantom_t):.5f}')

    mace_recon, info = mace([forward, denoiser], standard.clone(), rho=RHO,
                            num_iterations=NUM_ITERATIONS, callback=progress)

    results = [('Standard recon', standard),
               ('DRUNet postprocessing', postproc),
               ('DRUNet MACE recon', mace_recon)]
    print('NRMSE vs phantom:')
    for name, volume in results:
        print(f'  {name:22s} {nrmse(volume, phantom_t):.5f}')
    print(f'MACE consensus spread at exit: '
          f'{info["consensus_spread"][-1]:.2e}')

    if SHOW_VIEWER:
        labels = ['Phantom'] + [
            f'{name} (NRMSE {nrmse(volume, phantom_t):.3f})'
            for name, volume in results]
        mbirtorch.slice_viewer(
            phantom, standard_np, postproc.cpu().numpy(),
            mace_recon.cpu().numpy(), vmin=0.0, slice_label=labels,
            title=f'sigma_scaled {SIGMA_SCALED:g}, '
                  f'{NUM_ITERATIONS} MACE iterations')


if __name__ == '__main__':
    main()
