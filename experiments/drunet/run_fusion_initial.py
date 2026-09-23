"""Initial multi-slice fusion comparison on the 3D noisy cone-beam problem.

One run at fixed, stated parameters -- no sweeps yet.  Methods compared, all
initialized at the standard recon:

  standard   the standard qGGMRF recon (the incumbent)
  post-1     DRUNet postprocessing, slices along axis 2 only
  post-3     DRUNet postprocessing, average of the three orientations
  mace-1     MACE with a single-orientation DRUNet prior (axis 2)
  mace-3     MACE multi-slice fusion: the forward prox plus three
             orientation agents, weights (1/2, 1/6, 1/6, 1/6)

Initial parameter choices, to be swept later: sigma_scaled 0.10 for
postprocessing and 0.075 for the MACE priors (the best values from the 2D
sweep in mace_poc_findings.md), 30 outer iterations, rho 0.5, 3 inner prox
iterations.  Metrics: NRMSE vs phantom, and the data-consistency residual
rms_w(y - Ax) = sqrt(sum(w (y - Ax)^2) / sum(w)), with the noisy phantom's
own residual as the noise-floor reference.

Outputs: a printed table plus volumes and metrics in
./output/fusion_initial.npz.  --view opens the slice viewer (transpose axes
there to inspect through-plane behavior).  The standard recon is cached in
./output/ and reused while the problem is unchanged.  Run from this
directory with the repo root on PYTHONPATH.
"""

import argparse
import os

import numpy as np
import torch

import mbirtorch

from agents import DRUNetAgent, ForwardProxAgent, load_drunet
from cone_beam_3d import get_data
from mace import mace

CACHE_PATH = './output/fusion_standard_3d.npz'
SHARPNESS = 1.0


def nrmse(a, b):
    """||a - b|| / ||b|| for same-device tensors."""
    return float(torch.sqrt(torch.sum((a - b) ** 2))
                 / torch.sqrt(torch.sum(b ** 2)))


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def check_slice_axis(net, device):
    """Denoising along axis 0 must equal denoising the moved-axis volume
    along axis 2, moved back -- exactly, since both orders feed the network
    identical batches.  Odd sizes exercise the padding."""
    volume = torch.rand(12, 20, 8, device=device)
    agent0 = DRUNetAgent(net, 0.05, 1.0, slice_axis=0)
    agent2 = DRUNetAgent(net, 0.05, 1.0, slice_axis=2)
    direct_path = agent0(volume)
    moved_path = torch.moveaxis(agent2(torch.moveaxis(volume, 0, 2)), 2, 0)
    assert torch.equal(direct_path, moved_path), 'slice_axis check failed'


def standard_recon(ct_model, sinogram, weights):
    """The standard qGGMRF reconstruction and its auto sigma_prox, cached
    across runs while the sinogram and SHARPNESS match."""
    fingerprint = float(np.sum(sinogram, dtype=np.float64))
    if os.path.exists(CACHE_PATH):
        cache = np.load(CACHE_PATH)
        if (tuple(cache['sinogram_shape']) == tuple(sinogram.shape)
                and abs(float(cache['fingerprint']) - fingerprint)
                <= 1e-4 * abs(fingerprint)
                and float(cache['sharpness']) == SHARPNESS):
            print(f'Standard recon loaded from {CACHE_PATH}.')
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
    parser = argparse.ArgumentParser(
        description='Initial fusion comparison at fixed parameters.')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--inner-prox', type=int, default=3)
    parser.add_argument('--sigma-mace', type=float, default=0.075,
                        help='MACE denoiser strength, network scale')
    parser.add_argument('--sigma-post', type=float, default=0.10,
                        help='postprocessing strength, network scale')
    parser.add_argument('--view', action='store_true')
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
    print(f'Problem: {tuple(standard.shape)} recon on {device}, '
          f'sigma_prox {sigma_prox:.5f}, intensity scale '
          f'{intensity_scale:.2f}, sigma_scaled {args.sigma_mace:g} (MACE) '
          f'/ {args.sigma_post:g} (postprocessing)')

    def make_denoiser(sigma_scaled, axis):
        return DRUNetAgent(net, sigma_scaled / intensity_scale,
                           intensity_scale, ror_mask=ror_mask,
                           slice_axis=axis)

    with torch.no_grad():
        post_1 = make_denoiser(args.sigma_post, 2)(standard)
        post_3 = sum(make_denoiser(args.sigma_post, axis)(standard)
                     for axis in (0, 1, 2)) / 3.0

    def run_mace(label, axes, mu):
        np.random.seed(0)
        forward = ForwardProxAgent(
            ct_model, sinogram, weights=weights, sigma_prox=sigma_prox,
            inner_iterations=args.inner_prox, init_recon=standard.clone())
        agents = [forward] + [make_denoiser(args.sigma_mace, axis)
                              for axis in axes]

        def progress(iteration, x_bar):
            if (iteration + 1) % 5 == 0:
                print(f'  {label} iteration {iteration + 1:3d}: '
                      f'NRMSE vs phantom {nrmse(x_bar, phantom_t):.5f}')

        result, info = mace(agents, standard.clone(), mu=mu, rho=args.rho,
                            num_iterations=args.iterations,
                            callback=progress)
        print(f'  {label} consensus spread at exit: '
              f'{info["consensus_spread"][-1]:.2e}')
        return result, info

    mace_1, info_1 = run_mace('mace-1', [2], [0.5, 0.5])
    mace_3, info_3 = run_mace('mace-3', [0, 1, 2],
                              [0.5, 1 / 6, 1 / 6, 1 / 6])

    results = [('standard', standard), ('post-1', post_1),
               ('post-3', post_3), ('mace-1', mace_1), ('mace-3', mace_3)]
    noise_floor = data_consistency(phantom_t)
    print(f'{"method":10s} {"NRMSE":>8s} {"data rms_w":>11s}')
    print(f'{"(phantom)":10s} {"":8s} {noise_floor:11.5f}   <- noise floor')
    metrics = {}
    for name, volume in results:
        metrics[name] = (nrmse(volume, phantom_t), data_consistency(volume))
        print(f'{name:10s} {metrics[name][0]:8.5f} {metrics[name][1]:11.5f}')

    os.makedirs('./output', exist_ok=True)
    saved = {'phantom': phantom, 'noise_floor_rms': noise_floor,
             'sigma_mace': args.sigma_mace, 'sigma_post': args.sigma_post,
             'rho': args.rho, 'iterations': args.iterations,
             'sigma_prox': sigma_prox, 'intensity_scale': intensity_scale,
             'mace1_spread': np.asarray(info_1['consensus_spread']),
             'mace3_spread': np.asarray(info_3['consensus_spread'])}
    for name, volume in results:
        saved[name.replace('-', '_')] = volume.cpu().numpy()
        saved[name.replace('-', '_') + '_metrics'] = np.array(metrics[name])
    np.savez('./output/fusion_initial.npz', **saved)
    print('Volumes and metrics saved to ./output/fusion_initial.npz')

    if args.view:
        labels = ['Phantom'] + [f'{name} (NRMSE {metrics[name][0]:.3f})'
                                for name, _ in results]
        mbirtorch.slice_viewer(
            phantom, *[volume.cpu().numpy() for _, volume in results],
            vmin=0.0, slice_label=labels,
            title=f'Initial fusion comparison, sigma {args.sigma_mace:g} '
                  f'(MACE) / {args.sigma_post:g} (post)')


if __name__ == '__main__':
    main()
