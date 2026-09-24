"""Reconstruct a ZEISS VoluMax scan with mbirtorch using mbirtorch.preprocess.volumax."""
import os
import time

import numpy as np
import mbirtorch

from mbirtorch.preprocess import volumax

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------
dataset_dir = '/depot/bouman/data/ORNL/volumax/Hexagonal_volumax'
downsample_factor = (2, 2)        # detector (rows, channels) binning -> 1512 x 1512
subsample_view_factor = 1         # keep every k-th view (1 = all 2000 views)
sharpness = 1.0
snr_db = 35.0
max_iterations = 15
weight_type = None                # or 'transmission'
sinogram_path = None              # optional precomputed -log sinogram (e.g. LEAP BH corrected), same binning / views
output_path = None                # e.g. './output/volumax_recon.h5' to save; None = save nothing
show_viewer = True
compare_uncalibrated = False      # also reconstruct with the metadata-only channel offset (objectPosition taken as the
                                  # rotation axis) and show it next to the calibrated result; see demo_volumax_calibration.py

# ----------------------------------------------------------------------------------------------
# 1. Sinogram and ready-to-use cone-beam model (geometry table is printed)
# ----------------------------------------------------------------------------------------------
sino, ct_model, geom = volumax.get_sino_and_model(dataset_dir, downsample_factor=downsample_factor,
                                                  subsample_view_factor=subsample_view_factor,
                                                  sinogram_path=sinogram_path, return_geometry=True)

# ----------------------------------------------------------------------------------------------
# 2. Weights, sharpness, reconstruction
# ----------------------------------------------------------------------------------------------
weights = None if weight_type is None else mbirtorch.gen_weights(sino, weight_type=weight_type)
ct_model.set_params(sharpness=sharpness,snr_db=snr_db)

t0 = time.time()
recon, recon_dict = ct_model.recon(sino, weights=weights, max_iterations=max_iterations)
recon = np.asarray(recon)
print(f'Reconstruction {recon.shape} in {time.time() - t0:.1f} s; min {recon.min():.4f}, max {recon.max():.4f}')

# ----------------------------------------------------------------------------------------------
# 3. Optional save, then view the reconstruction
# ----------------------------------------------------------------------------------------------
if output_path is not None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ct_model.save_recon_hdf5(output_path, recon, recon_dict)
    print(f'Reconstruction saved to {output_path}')

# ----------------------------------------------------------------------------------------------
# 4. Optional: the same reconstruction with the uncalibrated (metadata-only) channel offset
# ----------------------------------------------------------------------------------------------
if compare_uncalibrated:
    meta_offset, _ = volumax.rotate_offsets_for_det_rotation(geom['det_channel_offset_metadata'], geom['det_row_offset'],
                                                             geom['det_rotation'])
    ct_model_meta = mbirtorch.copy_ct_model(ct_model, no_warning=True)
    ct_model_meta.set_params(det_channel_offset=meta_offset, sharpness=sharpness)
    ct_model_meta.set_params(recon_slice_offset=ct_model.get_params('recon_slice_offset'), no_warning=True)
    t0 = time.time()
    recon_meta, _ = ct_model_meta.recon(sino, weights=weights, max_iterations=max_iterations)
    recon_meta = np.asarray(recon_meta)
    print(f'Uncalibrated reconstruction (det_channel_offset {meta_offset:+.4f} mm instead of '
          f'{ct_model.get_params("det_channel_offset"):+.4f} mm) in {time.time() - t0:.1f} s')

if show_viewer:
    if compare_uncalibrated:
        mbirtorch.slice_viewer(recon_meta, recon, vmin=0.0, vmax=0.1,
                               slice_label=[f'uncalibrated (metadata offset {meta_offset:+.3f} mm)',
                                            f'calibrated (data offset {ct_model.get_params("det_channel_offset"):+.3f} mm)'],
                               title='VoluMax MBIR: uncalibrated vs calibrated channel offset')
    else:
        mbirtorch.slice_viewer(recon, data_dicts=[recon_dict], vmin=0.0, vmax=0.1,
                               title='VoluMax cone-beam MBIR reconstruction (mbirtorch)')
