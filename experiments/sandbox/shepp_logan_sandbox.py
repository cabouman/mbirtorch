"""Demo 1: 3D Shepp-Logan reconstruction with mbirtorch (the mbirjax demo_1
equivalent).

Generates a Shepp-Logan phantom, forward projects it to a sinogram, and runs
the VCD reconstruction; prints the per-iteration traces and the final NRMSE
against the phantom.  Run parameters sit at the top (no CLI arguments);
MODEL_TYPE selects the geometry ('parallel' or 'cone', as in the mbirjax
demo).  Set SHOW_SLICES = True to explore the ground truth phantom and the
reconstruction in the slice viewer (the recon's data dict rides along).
"""

import time

import numpy as np

import mbirtorch

# ── run parameters ────────────────────────────────────────────────────────────
MODEL_TYPE = "parallel"              # 'parallel' or 'cone'
SINOGRAM_SHAPE = (128, 128, 128)     # (num_views, num_det_rows, num_det_channels)
MAX_ITERATIONS = 15
SHARPNESS = 1.0
# Devices are not named here.  The model resolves cuda > mps > cpu on its
# own, and on a machine with several CUDA devices it spreads the
# reconstruction across the ones that can hold their share.  To pin it, call
# model.configure_devices(num_devices=1) or configure_devices(devices=['cpu']).
SEED = 0
SHOW_SLICES = True
# ──────────────────────────────────────────────────────────────────────────────

def build_model():
    n_views, _, num_channels = SINOGRAM_SHAPE
    if MODEL_TYPE == "cone":
        # Cone beam: full-circle angles, and source-detector / source-iso
        # distances in the goldens' convention (magnification 2).  The auto
        # recon geometry sets the recon shape, including the axial padding.
        angles = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
        source_detector_dist = 4 * num_channels
        source_iso_dist = 2 * num_channels
        _model = mbirtorch.ConeBeamModel(
            SINOGRAM_SHAPE, angles,
            source_detector_dist=source_detector_dist,
            source_iso_dist=source_iso_dist)
        _params = {'angles': angles, 'source_detector_dist': source_detector_dist, 'source_iso_dist': source_iso_dist}
    elif MODEL_TYPE == "parallel":
        angles = np.linspace(0, np.pi, n_views, endpoint=False)
        _model = mbirtorch.ParallelBeamModel(SINOGRAM_SHAPE, angles)
        _params = {'angles': angles}
    else:
        raise ValueError(f"MODEL_TYPE must be 'parallel' or 'cone', got {MODEL_TYPE!r}")

    return _model, _params


def get_data():
    model, params = build_model()
    model.set_params(sharpness=SHARPNESS)
    recon_shape = model.get_params("recon_shape")
    print(f"model = {MODEL_TYPE}, device = {model.torch_device}, "
          f"recon_shape = {recon_shape}")

    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = model.forward_project(phantom)
    weights = mbirtorch.gen_weights(sinogram / np.max(sinogram),
                                    weight_type="transmission_root")
    return sinogram, weights, params, phantom


def main():
    sinogram, weights, params, phantom = get_data()

    # if MODEL_TYPE == "parallel":
    #     angles = params["angles"]
    #     recon, recon_dict = simple_parallel(sinogram, angles, weights=weights,
    #                                         max_iterations=MAX_ITERATIONS)
    #     mbirtorch.slice_viewer(recon, data_dicts=recon_dict)
    #     return

    model, params = build_model()

    # model.configure_devices(devices=['cpu']*4)
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
    mbirtorch.get_memory_stats()

    if SHOW_SLICES:
        mbirtorch.slice_viewer(
            phantom, recon,
            slice_label=["ground truth phantom", "mbirtorch recon"],
            data_dicts=[None, recon_dict],
            title=f"Shepp-Logan {MODEL_TYPE} demo (NRMSE {nrmse:.4f})")


if __name__ == "__main__":
    main()
