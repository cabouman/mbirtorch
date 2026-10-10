"""prox_map gates: the proximal map pulls the reconstruction toward prox_input
at a small sigma and toward the data at a large one, and a supplied generator
fixes every draw the call makes."""

import numpy as np

import mbirtorch


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


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


def test_prox_map_draws_follow_a_supplied_generator(device):
    """Every random draw a prox_map call makes -- the pixel partitions and the
    order the subsets are visited in -- comes from the generator it is given,
    so two calls handed a fresh generator of one seed agree to a relative
    maximum difference of 1e-6, and a call handed another seed differs by more
    than that.  Nothing here touches the global np.random state, which is what
    makes a call reproducible on whichever thread runs it."""
    sino_shape = (24, 16, 16)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    recon_shape = model.get_params('recon_shape')

    phantom = np.zeros(tuple(recon_shape), dtype=np.float32)
    r0, c0, s0 = [n // 4 for n in recon_shape]
    phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    sinogram = model.forward_project(phantom)
    prox_input = 0.5 * phantom

    def run(seed):
        out, _ = model.prox_map(prox_input, sinogram, init_recon=phantom,
                                max_iterations=3, stop_threshold_change_pct=0.0,
                                logfile_path=None, print_logs=False,
                                rng=np.random.default_rng(seed))
        return out

    first, again, other = run(1), run(1), run(2)
    same = _rel_max(again, first)
    different = _rel_max(other, first)
    print(f"prox_map on {device}: one seed twice {same:.2e}, another seed {different:.2e}")
    assert same < 1e-6
    assert different > 1e-6
