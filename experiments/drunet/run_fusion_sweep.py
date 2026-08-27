"""Strength sweep for the 3D multi-slice fusion comparison.

Completes increments 2 and 3 of multi_slice_fusion.md: every method runs
over the same strength grid, so each is judged at ITS best strength.  The
methods, all initialized at the standard recon: post-1 (axis-2
postprocessing), post-3 (three-orientation-average postprocessing), mace-1
(single-orientation prior), and mace-3 (fusion at weights
(1/2, 1/6, 1/6, 1/6)).  After the grid, the best mace-1 and mace-3
strengths re-run at double the iterations to confirm the ordering at
convergence (the 2D work showed 30 iterations lands close but not settled).

Outputs: printed tables plus ./output/fusion_sweep.npz with the metric
grids and the best volumes.  Setup helpers are shared with
run_fusion_initial.py.  Run from this directory with the repo root on
PYTHONPATH.
"""

import argparse
import os

import numpy as np
import torch

import mbirtorch

from agents import DRUNetAgent, ForwardProxAgent, load_drunet
from cone_beam_3d import get_data
from mace import mace
from run_fusion_initial import (SHARPNESS, check_slice_axis, nrmse,
                                pick_device, standard_recon)


def main():
    parser = argparse.ArgumentParser(
        description='Strength sweep for the 3D fusion comparison.')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--confirm-iterations', type=int, default=60,
                        help='iterations for the best-strength re-runs')
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--inner-prox', type=int, default=3)
    parser.add_argument('--sigma-scaled', type=float, nargs='*',
                        default=[0.05, 0.075, 0.10, 0.125, 0.15])
    args = parser.parse_args()

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
    sinogram_t = torch.as_tensor(sinogram, device=device)
    weights_t = torch.as_tensor(np.asarray(weights), device=device)

    def data_consistency(volume):
        residual = sinogram_t - torch.as_tensor(
            ct_model.forward_project(volume), device=device)
        return float(torch.sqrt(torch.sum(weights_t * residual ** 2)
                                / torch.sum(weights_t)))

    robust_max = float(torch.quantile(standard.flatten(), 0.999))
    intensity_scale = 0.9 / robust_max
    if ct_model.get_params('use_ror_mask'):
        mask = mbirtorch.get_2d_ror_mask(tuple(standard.shape))
        ror_mask = torch.as_tensor(mask.astype(np.float32),
                                   device=device).unsqueeze(-1)
    else:
        ror_mask = None
    net = load_drunet(device)
    check_slice_axis(net, device)

    print(f'Problem: {tuple(standard.shape)} on {device}, sigma_prox '
          f'{sigma_prox:.5f}, intensity scale {intensity_scale:.2f}, '
          f'grid {args.sigma_scaled}, {args.iterations} iterations '
          f'(+{args.confirm_iterations} confirmations)', flush=True)
    print(f'standard: NRMSE {nrmse(standard, phantom_t):.5f}, '
          f'rms_w {data_consistency(standard):.5f}; noise floor rms_w '
          f'{data_consistency(phantom_t):.5f}', flush=True)

    def make_denoiser(sigma_scaled, axis):
        return DRUNetAgent(net, sigma_scaled / intensity_scale,
                           intensity_scale, ror_mask=ror_mask,
                           slice_axis=axis)

    def run_mace(sigma_scaled, axes, mu, iterations):
        np.random.seed(0)
        forward = ForwardProxAgent(
            ct_model, sinogram, weights=weights, sigma_prox=sigma_prox,
            inner_iterations=args.inner_prox, init_recon=standard.clone())
        agents = [forward] + [make_denoiser(sigma_scaled, axis)
                              for axis in axes]
        result, info = mace(agents, standard.clone(), mu=mu, rho=args.rho,
                            num_iterations=iterations)
        return result, info['consensus_spread'][-1]

    methods = ['post-1', 'post-3', 'mace-1', 'mace-3']
    metric_grid = {m: [] for m in methods}      # (nrmse, rms_w, spread)
    best = {m: None for m in methods}           # (nrmse, sigma, volume cpu)

    def record(method, sigma_scaled, volume, spread=0.0):
        entry = (nrmse(volume, phantom_t), data_consistency(volume), spread)
        metric_grid[method].append(entry)
        if best[method] is None or entry[0] < best[method][0]:
            best[method] = (entry[0], sigma_scaled, volume.cpu().numpy())
        return entry

    for sigma_scaled in args.sigma_scaled:
        with torch.no_grad():
            e1 = record('post-1', sigma_scaled,
                        make_denoiser(sigma_scaled, 2)(standard))
            e3 = record('post-3', sigma_scaled,
                        sum(make_denoiser(sigma_scaled, axis)(standard)
                            for axis in (0, 1, 2)) / 3.0)
        m1, spread1 = run_mace(sigma_scaled, [2], [0.5, 0.5],
                               args.iterations)
        em1 = record('mace-1', sigma_scaled, m1, spread1)
        m3, spread3 = run_mace(sigma_scaled, [0, 1, 2],
                               [0.5, 1 / 6, 1 / 6, 1 / 6], args.iterations)
        em3 = record('mace-3', sigma_scaled, m3, spread3)
        print(f'sigma {sigma_scaled:5g}: post-1 {e1[0]:.5f}  '
              f'post-3 {e3[0]:.5f}  mace-1 {em1[0]:.5f} (sp {em1[2]:.1e})  '
              f'mace-3 {em3[0]:.5f} (sp {em3[2]:.1e})', flush=True)

    print('Best per method over the grid:', flush=True)
    for method in methods:
        print(f'  {method:7s} NRMSE {best[method][0]:.5f} at sigma_scaled '
              f'{best[method][1]:g}', flush=True)

    confirmed = {}
    for method, axes, mu in [('mace-1', [2], [0.5, 0.5]),
                             ('mace-3', [0, 1, 2],
                              [0.5, 1 / 6, 1 / 6, 1 / 6])]:
        sigma_scaled = best[method][1]
        volume, spread = run_mace(sigma_scaled, axes, mu,
                                  args.confirm_iterations)
        confirmed[method] = (nrmse(volume, phantom_t),
                            data_consistency(volume), spread,
                            sigma_scaled, volume.cpu().numpy())
        print(f'confirmed {method} at sigma_scaled {sigma_scaled:g}, '
              f'{args.confirm_iterations} iterations: NRMSE '
              f'{confirmed[method][0]:.5f}, rms_w {confirmed[method][1]:.5f}'
              f', spread {confirmed[method][2]:.1e}', flush=True)

    os.makedirs('./output', exist_ok=True)
    saved = {'sigma_grid': np.asarray(args.sigma_scaled),
             'iterations': args.iterations,
             'confirm_iterations': args.confirm_iterations,
             'rho': args.rho, 'sigma_prox': sigma_prox,
             'intensity_scale': intensity_scale,
             'standard': standard_np,
             'standard_nrmse': nrmse(standard, phantom_t),
             'noise_floor_rms': data_consistency(phantom_t)}
    for method in methods:
        key = method.replace('-', '_')
        saved[key + '_grid'] = np.asarray(metric_grid[method])
        saved[key + '_best_sigma'] = best[method][1]
        saved[key + '_best'] = best[method][2]
    for method, values in confirmed.items():
        key = method.replace('-', '_') + '_confirmed'
        saved[key] = values[4]
        saved[key + '_metrics'] = np.asarray(values[:4])
    np.savez('./output/fusion_sweep.npz', **saved)
    print('Grids and best volumes saved to ./output/fusion_sweep.npz',
          flush=True)


if __name__ == '__main__':
    main()
