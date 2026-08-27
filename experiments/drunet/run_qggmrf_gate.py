"""Gate: MACE with the qGGMRF denoiser agent must reproduce the standard recon.

Two agents on the 2D noisy cone-beam problem from cone_beam_2d.py: the
proximal map of the data-fit term, and the qGGMRF denoiser with its prior
parameters pinned to the standard recon's values.  (The STANDARD recon is
recon() with the qGGMRF prior; "direct recon" is reserved for the FDK-style
initialization recon_direct produces.)  With matched strengths
(sigma_noise = sigma_prox = the model's auto sigma_prox), the consensus
equilibrium solves the same objective as recon(), so the whole Plug-and-Play
plumbing is validated by one number: NRMSE(consensus, standard recon) <
GATE_TOL.

The sigma sweep reruns MACE at scaled matched strengths.  The exact
equilibrium does not depend on sigma (only the convergence path does), so the
drift across the sweep measures the damage from the inexact fixed-iteration
inner solves.  --from-zero adds a run initialized at zero instead of at the
standard recon, as a convergence-from-far robustness check (its corners differ
harmlessly: pixels outside the region-of-reconstruction mask are never
updated, so they keep their initial values).

Outputs: a printed report, plus the volumes and per-iteration traces in
./output/qggmrf_gate.npz.  Run from this directory with the repo root on
PYTHONPATH.
"""

import argparse
import os

import numpy as np
import torch

import mbirtorch

from agents import ForwardProxAgent, QGGMRFDenoiserAgent
from cone_beam_2d import get_data
from mace import mace

GATE_TOL = 0.01


def nrmse(a, b):
    """||a - b|| / ||b|| for same-device tensors."""
    return float(torch.sqrt(torch.sum((a - b) ** 2))
                 / torch.sqrt(torch.sum(b ** 2)))


def run_case(label, ct_model, sinogram, weights, x0, sigma, sigma_x,
             references, args):
    """One MACE run.  Fresh agents each time; the shared ct_model re-runs its
    prox initialization on the agent's first call."""
    np.random.seed(0)
    forward = ForwardProxAgent(
        ct_model, sinogram, weights=weights, sigma_prox=sigma,
        inner_iterations=args.inner_prox, init_recon=x0)
    denoiser = QGGMRFDenoiserAgent(
        tuple(x0.shape), sigma_noise=sigma,
        pinned_params={'sigma_x': sigma_x},
        inner_iterations=args.inner_denoise, like_model=ct_model,
        use_ror_mask=ct_model.get_params('use_ror_mask'))

    traces = {name: [] for name in references}

    def record(iteration, x_bar):
        for name, reference in references.items():
            traces[name].append(nrmse(x_bar, reference))

    x_bar, info = mace([forward, denoiser], x0, rho=args.rho,
                       num_iterations=args.iterations, callback=record)
    info.update(traces)
    print(f'  {label}: NRMSE vs standard {traces["direct"][-1]:.5f}, '
          f'vs phantom {traces["phantom"][-1]:.5f}, '
          f'consensus spread {info["consensus_spread"][-1]:.2e}')
    return x_bar, info


def main():
    parser = argparse.ArgumentParser(
        description='MACE/qGGMRF equality gate on the 2D cone-beam problem.')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--inner-prox', type=int, default=3)
    parser.add_argument('--inner-denoise', type=int, default=8)
    parser.add_argument('--sigma-scales', type=float, nargs='*',
                        default=[0.5, 2.0],
                        help='extra matched-sigma runs at these multiples of '
                             'the auto sigma_prox')
    parser.add_argument('--from-zero', action='store_true',
                        help='add a run initialized at zero')
    parser.add_argument('--view', action='store_true',
                        help='show phantom / direct / MACE in the viewer')
    args = parser.parse_args()

    phantom, sinogram, params = get_data()
    ct_model = mbirtorch.ConeBeamModel(
        sinogram.shape, params['angles'],
        source_detector_dist=params['source_detector_dist'],
        source_iso_dist=params['source_iso_dist'])
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')
    ct_model.set_params(sharpness=1.0)

    # The gate target: a well-converged standard qGGMRF reconstruction.
    np.random.seed(0)
    direct, direct_dict = ct_model.recon(
        sinogram, weights=weights, max_iterations=30,
        stop_threshold_change_pct=0.0, print_logs=False, output_sharded=True)
    if not torch.is_tensor(direct):
        raise RuntimeError('This proof of concept assumes a single-device '
                           'model (got the multi-device form).')
    regularization = direct_dict['recon_params']['regularization_params']
    sigma = float(regularization['sigma_prox'])
    sigma_x = float(regularization['sigma_x'])

    phantom_t = torch.as_tensor(phantom, device=direct.device)
    direct_nrmse = nrmse(direct, phantom_t)
    print(f'Problem: {tuple(direct.shape)} recon on {direct.device}, '
          f'auto sigma_prox {sigma:.5f}, sigma_x {sigma_x:.5f}')
    print(f'Standard qGGMRF recon: NRMSE vs phantom {direct_nrmse:.5f}')
    print(f'MACE: {args.iterations} outer iterations, rho {args.rho}, '
          f'{args.inner_prox} prox + {args.inner_denoise} denoise inner '
          f'iterations')

    references = {'direct': direct, 'phantom': phantom_t}
    results = {}

    print('Matched-sigma run (the gate):')
    x_matched, info_matched = run_case(
        'sigma x 1.0', ct_model, sinogram, weights, direct.clone(), sigma,
        sigma_x, references, args)
    results['matched'] = (x_matched, info_matched)
    gate_value = info_matched['direct'][-1]

    print('Sigma sweep (equilibrium should not move; drift = inexactness):')
    for scale in args.sigma_scales:
        x_s, info_s = run_case(
            f'sigma x {scale:g}', ct_model, sinogram, weights, direct.clone(),
            scale * sigma, sigma_x, references, args)
        results[f'scale_{scale:g}'] = (x_s, info_s)

    if args.from_zero:
        print('From-zero initialization (robustness):')
        x_z, info_z = run_case(
            'from zero', ct_model, sinogram, weights,
            torch.zeros_like(direct), sigma, sigma_x, references, args)
        results['from_zero'] = (x_z, info_z)

    passed = gate_value < GATE_TOL
    print(f'GATE {"PASS" if passed else "FAIL"}: NRMSE(MACE, standard recon) = '
          f'{gate_value:.5f} (tolerance {GATE_TOL})')

    os.makedirs('./output', exist_ok=True)
    saved = {'phantom': phantom, 'direct': direct.cpu().numpy(),
             'sigma': sigma, 'sigma_x': sigma_x,
             'rho': args.rho, 'iterations': args.iterations,
             'gate_value': gate_value, 'gate_tol': GATE_TOL}
    for name, (volume, info) in results.items():
        saved[f'{name}_recon'] = volume.cpu().numpy()
        for trace_name, values in info.items():
            saved[f'{name}_{trace_name}'] = np.asarray(values)
    out_path = './output/qggmrf_gate.npz'
    np.savez(out_path, **saved)
    print(f'Volumes and traces saved to {out_path}')

    if args.view:
        mbirtorch.slice_viewer(
            phantom, direct.cpu().numpy(), x_matched.cpu().numpy(),
            vmin=0.0, title='Phantom / direct qGGMRF / MACE consensus')


if __name__ == '__main__':
    main()
