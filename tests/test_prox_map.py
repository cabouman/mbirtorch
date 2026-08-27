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


def test_prox_map_resume_advances_partition_sequence(device):
    """A resume call (do_initialization=False) honors first_iteration: the
    partition sequence picks up where the previous call stopped and, past the
    end of the model's sequence, repeats the last (finest) entry.  A
    Plug-and-Play loop relies on this to walk coarse partitions first and the
    finest ones for the rest of the run."""
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
    prox_input = 0.5 * phantom
    seq_param = list(model.get_params('partition_sequence'))

    def used_sequence(recon_dict):
        return list(recon_dict['recon_params']['partition_sequence'])

    def expected_sequence(first_iteration, max_iterations):
        full = mbirtorch.gen_partition_sequence(seq_param, max_iterations)
        return [int(v) for v in full[first_iteration:]]

    np.random.seed(0)
    _, d0 = model.prox_map(prox_input, sinogram, max_iterations=2,
                           stop_threshold_change_pct=0.0)
    assert used_sequence(d0) == expected_sequence(0, 2)

    # Resume where the first call stopped: entries 2 and 3 of the sequence.
    _, d1 = model.prox_map(prox_input, sinogram, do_initialization=False,
                           first_iteration=2, max_iterations=4,
                           stop_threshold_change_pct=0.0)
    assert used_sequence(d1) == expected_sequence(2, 4)

    # Far past the end of the model's sequence: every entry is the last one.
    far = len(seq_param) + 5
    _, d2 = model.prox_map(prox_input, sinogram, do_initialization=False,
                           first_iteration=far, max_iterations=far + 2,
                           stop_threshold_change_pct=0.0)
    assert used_sequence(d2) == [seq_param[-1]] * 2
