import os
import time
import numpy as np
import mbirtorch
from mbirtorch.preprocess import volumax

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------
# The scan folder: the folder that contains AcquisitionParameters.json.
scan_dir = '/depot/bouman/data/ORNL/volumax/Hexagonal_volumax/proj/M0685 - Hexagonal Part (Scanning Stragety Optimization)_2026-08-31T10-07-09'
# scan_dir = '/depot/bouman/data/ORNL/volumax/HIP_Can_Proj/HIP Can - Vanshika 92.8um Res_2025-09-12T14-46-58'
downsample_factor = (2, 2)        # detector (rows, channels) block averaging
subsample_view_factor = 1         # keep every n-th view
sharpness = 1.0
snr_db = 35.0
max_iterations = 15
weight_type = None                # None, or a weight type for mbirtorch.gen_weights, e.g. 'transmission'
sinogram_path = None              # optional precomputed -log sinogram (.npy)
output_path = None                # e.g. './output/volumax_recon.h5'; None skips saving
show_viewer = True
compare_uncalibrated = False      # also reconstruct with the uncalibrated channel offset from the metadata, to compare

# ----------------------------------------------------------------------------------------------
# 1. Sinogram and calibrated cone-beam model
# ----------------------------------------------------------------------------------------------
sino, ct_model = volumax.get_sino_and_model(scan_dir, downsample_factor=downsample_factor,
                                            subsample_view_factor=subsample_view_factor, sinogram_path=sinogram_path)

# ----------------------------------------------------------------------------------------------
# 2. Weights, regularization, and reconstruction
# ----------------------------------------------------------------------------------------------
weights = None if weight_type is None else mbirtorch.gen_weights(sino, weight_type=weight_type)
ct_model.set_params(sharpness=sharpness, snr_db=snr_db)

t0 = time.time()
recon, recon_dict = ct_model.recon(sino, weights=weights, max_iterations=max_iterations)
recon = np.asarray(recon)
print(f'Reconstruction {recon.shape} in {time.time() - t0:.1f} s; min {recon.min():.4f}, max {recon.max():.4f}')

# ----------------------------------------------------------------------------------------------
# 3. Optional: save the reconstruction
# ----------------------------------------------------------------------------------------------
if output_path is not None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ct_model.save_recon_hdf5(output_path, recon, recon_dict)
    print(f'Reconstruction saved to {output_path}')

# ----------------------------------------------------------------------------------------------
# 4. Optional: the same reconstruction with the uncalibrated channel offset from the metadata
# ----------------------------------------------------------------------------------------------
if compare_uncalibrated:
    # The channel offset from the metadata alone, in the frame of the model; the calibration starts from it.
    _, volumax_params = volumax.load_scans_and_params(scan_dir, subsample_view_factor=subsample_view_factor,
                                                      downsample_factor=downsample_factor, verbose=0, load_scans=False)
    geometry = volumax.compute_geometry(volumax_params, verbose=0)
    _, optional_params = volumax.convert_volumax_to_mbirtorch_params(volumax_params, geometry,
                                                                     downsample_factor=downsample_factor, verbose=0)
    meta_offset = optional_params['det_channel_offset']
    ct_model_meta = mbirtorch.copy_ct_model(ct_model, no_warning=True)
    ct_model_meta.set_params(det_channel_offset=meta_offset)
    t0 = time.time()
    recon_meta, _ = ct_model_meta.recon(sino, weights=weights, max_iterations=max_iterations)
    recon_meta = np.asarray(recon_meta)
    print(f'Uncalibrated reconstruction (det_channel_offset {meta_offset:+.4f} mm instead of '
          f'{ct_model.get_params("det_channel_offset"):+.4f} mm) in {time.time() - t0:.1f} s')

# ----------------------------------------------------------------------------------------------
# 5. View the reconstruction
# ----------------------------------------------------------------------------------------------
if show_viewer:
    if compare_uncalibrated:
        mbirtorch.slice_viewer(recon_meta, recon,
                               slice_label=[f'uncalibrated (metadata offset {meta_offset:+.3f} mm)',
                                            f'calibrated (offset {ct_model.get_params("det_channel_offset"):+.3f} mm)'],
                               title='VoluMax reconstruction: uncalibrated vs calibrated channel offset')
    else:
        mbirtorch.slice_viewer(recon, data_dicts=[recon_dict], title='VoluMax cone-beam MBIR reconstruction')
