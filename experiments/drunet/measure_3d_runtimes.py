"""Increment-1 runtime measurement for the multi-slice fusion plan: time the
pieces a MACE iteration is made of at the 3D demo scale, so the gate and the
comparison runs are sized to the machine before they are launched.

Pieces timed: the standard recon at 2 and 4 iterations (their difference
gives a per-iteration cost free of setup), one 3-iteration prox_map call,
one 8-iteration qGGMRF denoise, and one DRUNet pass over each of the three
slice orientations.  Run from this directory with the repo root on
PYTHONPATH; results print to stdout.
"""

import time

import numpy as np
import torch

import mbirtorch

from agents import load_drunet
from cone_beam_3d import get_data


def timed(label, fn):
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    print(f'  {label}: {elapsed:.1f} s')
    return result, elapsed


def main():
    phantom, sinogram, params = get_data()
    ct_model = mbirtorch.ConeBeamModel(
        sinogram.shape, params['angles'],
        source_detector_dist=params['source_detector_dist'],
        source_iso_dist=params['source_iso_dist'])
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')
    ct_model.set_params(sharpness=1.0)
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    ct_model.configure_devices(devices=[device])
    print(f'Problem: sinogram {sinogram.shape}, device {device}')

    np.random.seed(0)
    (recon2, _), t2 = timed('standard recon, 2 iterations (includes setup)',
                            lambda: ct_model.recon(
                                sinogram, weights=weights, max_iterations=2,
                                stop_threshold_change_pct=0.0,
                                print_logs=False, output_sharded=True))
    np.random.seed(0)
    (recon4, d4), t4 = timed('standard recon, 4 iterations',
                             lambda: ct_model.recon(
                                 sinogram, weights=weights, max_iterations=4,
                                 stop_threshold_change_pct=0.0,
                                 print_logs=False, output_sharded=True))
    print(f'  -> per recon iteration: {(t4 - t2) / 2:.1f} s')
    recon_shape = tuple(recon4.shape)
    print(f'  recon shape: {recon_shape}')

    sigma_prox = float(
        d4['recon_params']['regularization_params']['sigma_prox'])
    np.random.seed(0)
    timed('prox_map, 3 iterations (includes its initialization)',
          lambda: ct_model.prox_map(
              recon4, sinogram, sigma_prox=sigma_prox, weights=weights,
              init_recon=recon4, max_iterations=3,
              stop_threshold_change_pct=0.0, print_logs=False,
              output_sharded=True))

    denoiser = mbirtorch.QGGMRFDenoiser(recon_shape)
    denoiser.configure_devices(like=ct_model)
    denoiser.set_params(no_warning=True, verbose=0, sigma_x=0.05,
                        auto_regularize_flag=False)
    np.random.seed(0)
    timed('qGGMRF denoise, 8 iterations',
          lambda: denoiser.denoise(recon4, sigma_noise=sigma_prox,
                                   max_iterations=8,
                                   stop_threshold_change_pct=0.0,
                                   print_logs=False, output_sharded=True))

    net = load_drunet(device)
    with torch.no_grad():
        for axis in (0, 1, 2):
            stack = torch.moveaxis(recon4, axis, 0).unsqueeze(1)
            timed(f'DRUNet over the axis-{axis} slice stack '
                  f'({stack.shape[0]} slices)',
                  lambda: torch.cat([net(batch, 0.075)
                                     for batch in stack.split(8)]))


if __name__ == '__main__':
    main()
