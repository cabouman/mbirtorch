"""prox_map smoke: the proximal map pulls the recon toward prox_input, and the
prox_data caching path (do_initialization=False) works."""

import numpy as np

import mbirtorch


def test_prox_map_pulls_toward_input(device):
    sino_shape = (40, 32, 32)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    recon_shape = model.get_params('recon_shape')

    phantom = np.zeros(tuple(recon_shape), dtype=np.float32)
    r0, c0, s0 = [n // 4 for n in recon_shape]
    phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    sinogram = model.forward_project(phantom)

    # A prox input distinct from the data-only solution.
    prox_input = 0.5 * phantom

    np.random.seed(0)
    recon_small, _ = model.prox_map(prox_input, sinogram, sigma_prox=1e-4,
                                    init_recon=phantom, max_iterations=3,
                                    stop_threshold_change_pct=0.0)
    # Tiny sigma_prox: the prior dominates, so the result hugs prox_input.
    dist_small = float(np.linalg.norm(recon_small - prox_input))

    np.random.seed(0)
    recon_large, _ = model.prox_map(prox_input, sinogram, sigma_prox=1e3,
                                    init_recon=phantom, max_iterations=3,
                                    stop_threshold_change_pct=0.0,
                                    do_initialization=False)
    # Huge sigma_prox: the data dominates, so the result stays near the phantom.
    dist_large = float(np.linalg.norm(recon_large - phantom))

    scale = float(np.linalg.norm(phantom))
    assert dist_small / scale < 0.05, dist_small / scale
    assert dist_large / scale < 0.05, dist_large / scale
    # And the two regimes genuinely differ.
    assert float(np.linalg.norm(recon_small - recon_large)) / scale > 0.1
