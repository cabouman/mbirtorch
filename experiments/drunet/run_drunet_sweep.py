"""DRUNet as the MACE prior agent on the 2D noisy cone-beam problem.

For each denoiser strength in the sweep (given in the network's scaled
units, where the recon is mapped into roughly [0, 1]), this reports
  - DRUNet as POSTPROCESSING: one denoiser application to the standard
    qGGMRF reconstruction (the baseline use of the same component), and
  - DRUNet as a PRIOR: MACE with the data-fit proximal map and the DRUNet
    agent, initialized at the standard reconstruction.
The standard qGGMRF reconstruction itself is the incumbent baseline.  (The
STANDARD recon is recon() with the qGGMRF prior; "direct recon" is reserved
for the FDK-style initialization recon_direct produces.)  The
forward agent's sigma_prox stays at the model's auto value throughout;
sigma_noise is the one knob swept.

Outputs: a printed table, plus volumes (best strength per method) and the
sweep arrays in ./output/drunet_sweep.npz.  Run from this directory with the
repo root on PYTHONPATH.
"""

import argparse
import os

import numpy as np
import torch

import mbirtorch

from agents import DRUNetAgent, ForwardProxAgent, load_drunet
from cone_beam_2d import get_data
from mace import mace


def nrmse(a, b):
    """||a - b|| / ||b|| for same-device tensors."""
    return float(torch.sqrt(torch.sum((a - b) ** 2))
                 / torch.sqrt(torch.sum(b ** 2)))


def main():
    parser = argparse.ArgumentParser(
        description='DRUNet postprocessing vs MACE prior on the 2D problem.')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--inner-prox', type=int, default=3)
    parser.add_argument('--sigma-scaled', type=float, nargs='*',
                        default=[0.02, 0.05, 0.10, 0.15, 0.20],
                        help='denoiser strengths in the network scale '
                             '(sigma of 25/255 on natural images is ~0.1)')
    parser.add_argument('--view', action='store_true',
                        help='show phantom / direct / best MACE-DRUNet')
    args = parser.parse_args()

    phantom, sinogram, params = get_data()
    ct_model = mbirtorch.ConeBeamModel(
        sinogram.shape, params['angles'],
        source_detector_dist=params['source_detector_dist'],
        source_iso_dist=params['source_iso_dist'])
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')
    ct_model.set_params(sharpness=1.0)

    np.random.seed(0)
    direct, direct_dict = ct_model.recon(
        sinogram, weights=weights, max_iterations=30,
        stop_threshold_change_pct=0.0, print_logs=False, output_sharded=True)
    if not torch.is_tensor(direct):
        raise RuntimeError('This proof of concept assumes a single-device '
                           'model (got the multi-device form).')
    device = direct.device
    phantom_t = torch.as_tensor(phantom, device=device)
    direct_nrmse = nrmse(direct, phantom_t)

    # The fixed intensity scale: the recon's robust maximum maps to 0.9, so
    # the volume lands in the [0, 1] range the network was trained on.  The
    # scale is chosen once, here, and shared by every agent in the sweep.
    robust_max = float(torch.quantile(direct.flatten(), 0.999))
    intensity_scale = 0.9 / robust_max

    # The region-of-reconstruction convention matches the recon model's.
    recon_shape = tuple(direct.shape)
    if ct_model.get_params('use_ror_mask'):
        mask = mbirtorch.get_2d_ror_mask(recon_shape).astype(np.float32)
        ror_mask = torch.as_tensor(mask, device=device).unsqueeze(-1)
    else:
        ror_mask = None

    net = load_drunet(device)
    sigma_prox = float(
        direct_dict['recon_params']['regularization_params']['sigma_prox'])
    print(f'Problem: {recon_shape} recon on {device}, '
          f'auto sigma_prox {sigma_prox:.5f}, '
          f'intensity scale {intensity_scale:.2f} '
          f'(robust max {robust_max:.5f})')
    print(f'Standard qGGMRF recon: NRMSE vs phantom {direct_nrmse:.5f}')
    print(f'MACE: {args.iterations} outer iterations, rho {args.rho}, '
          f'{args.inner_prox} prox inner iterations')

    rows = []
    volumes = {}
    for sigma_scaled in args.sigma_scaled:
        sigma_noise = sigma_scaled / intensity_scale

        denoiser = DRUNetAgent(net, sigma_noise, intensity_scale,
                               ror_mask=ror_mask)
        with torch.no_grad():
            postproc = denoiser(direct)
        postproc_nrmse = nrmse(postproc, phantom_t)

        np.random.seed(0)
        forward = ForwardProxAgent(
            ct_model, sinogram, weights=weights, sigma_prox=sigma_prox,
            inner_iterations=args.inner_prox, init_recon=direct.clone())
        mace_recon, info = mace([forward, denoiser], direct.clone(),
                                rho=args.rho,
                                num_iterations=args.iterations)
        mace_nrmse = nrmse(mace_recon, phantom_t)
        spread = info['consensus_spread'][-1]
        rows.append((sigma_scaled, sigma_noise, postproc_nrmse, mace_nrmse,
                     spread))
        volumes[sigma_scaled] = (postproc, mace_recon)
        print(f'  sigma_scaled {sigma_scaled:.3f} '
              f'(recon units {sigma_noise:.5f}): '
              f'postproc NRMSE {postproc_nrmse:.5f}, '
              f'MACE NRMSE {mace_nrmse:.5f}, spread {spread:.2e}')

    best_postproc = min(rows, key=lambda r: r[2])
    best_mace = min(rows, key=lambda r: r[3])
    print(f'Standard recon      NRMSE {direct_nrmse:.5f}')
    print(f'Best postprocessing NRMSE {best_postproc[2]:.5f} '
          f'at sigma_scaled {best_postproc[0]:g}')
    print(f'Best MACE prior     NRMSE {best_mace[3]:.5f} '
          f'at sigma_scaled {best_mace[0]:g}')

    os.makedirs('./output', exist_ok=True)
    table = np.array([r[:4] for r in rows])
    out_path = './output/drunet_sweep.npz'
    np.savez(out_path,
             phantom=phantom, direct=direct.cpu().numpy(),
             direct_nrmse=direct_nrmse,
             sweep_table=table,
             sweep_columns=np.array(['sigma_scaled', 'sigma_recon_units',
                                     'postproc_nrmse', 'mace_nrmse']),
             intensity_scale=intensity_scale, sigma_prox=sigma_prox,
             best_postproc=volumes[best_postproc[0]][0].cpu().numpy(),
             best_mace=volumes[best_mace[0]][1].cpu().numpy())
    print(f'Volumes and sweep table saved to {out_path}')

    if args.view:
        mbirtorch.slice_viewer(
            phantom, direct.cpu().numpy(),
            volumes[best_mace[0]][1].cpu().numpy(), vmin=0.0,
            title='Phantom / direct qGGMRF / MACE with DRUNet prior')


if __name__ == '__main__':
    main()
