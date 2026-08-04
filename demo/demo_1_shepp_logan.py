"""Demo 1: 3D Shepp-Logan reconstruction with mbirtorch (the mbirjax demo_1
equivalent, Phase 1 form).

Generates a Shepp-Logan phantom, forward projects it to a sinogram, and runs
the VCD reconstruction; prints the per-iteration traces and the final NRMSE
against the phantom.  Run parameters sit at the top (no CLI arguments).  Set
SHOW_SLICES = True to display center slices with matplotlib.
"""

import time

import numpy as np

import mbirtorch

# ── run parameters ────────────────────────────────────────────────────────────
SINOGRAM_SHAPE = (128, 128, 128)     # (num_views, num_det_rows, num_det_channels)
MAX_ITERATIONS = 10
SHARPNESS = 0.0
DEVICE = "auto"                      # 'auto' -> cuda > mps > cpu
SEED = 0
SHOW_SLICES = False
# ──────────────────────────────────────────────────────────────────────────────


def main():
    n_views = SINOGRAM_SHAPE[0]
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirtorch.ParallelBeamModel(SINOGRAM_SHAPE, angles, device=DEVICE)
    model.set_params(no_warning=True, sharpness=SHARPNESS)
    recon_shape = model.get_params("recon_shape")
    print(f"device = {model.torch_device}, recon_shape = {recon_shape}")

    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = model.forward_project(phantom)
    weights = mbirtorch.gen_weights(sinogram / np.max(sinogram),
                                    weight_type="transmission_root")

    np.random.seed(SEED)
    t0 = time.time()
    recon, recon_dict = model.recon(sinogram, weights=weights,
                                    max_iterations=MAX_ITERATIONS)
    elapsed = time.time() - t0

    nrmse = float(np.linalg.norm(recon - phantom) / np.linalg.norm(phantom))
    rp = recon_dict["recon_params"]
    print(f"\nElapsed: {elapsed:.2f} s for {rp['num_iterations']} iterations")
    print(f"Final forward loss: {rp['fm_rmse'][-1]:.4f}")
    print(f"NRMSE vs phantom: {nrmse:.4f}")

    if SHOW_SLICES:
        import matplotlib.pyplot as plt
        mid = recon_shape[2] // 2
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(phantom[:, :, mid], cmap="gray")
        axes[0].set_title("phantom (center slice)")
        axes[1].imshow(recon[:, :, mid], cmap="gray")
        axes[1].set_title("mbirtorch recon")
        plt.show()


if __name__ == "__main__":
    main()
