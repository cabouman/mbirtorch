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
compare_calibrated = False        # also reconstruct with the channel offset estimated from the sinogram, to compare
                                  # (the estimate compares opposite views, so it needs a full rotation)

# ----------------------------------------------------------------------------------------------
# 1. Sinogram and cone-beam model with the geometry from the metadata
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
# 4. Optional: the same reconstruction with the channel offset estimated from the sinogram
# ----------------------------------------------------------------------------------------------
if compare_calibrated:
    # The calibration needs the detector rotation from the metadata, which was already removed from the sinogram.
    _, volumax_params = volumax.load_scans_and_params(scan_dir, subsample_view_factor=subsample_view_factor,
                                                      downsample_factor=downsample_factor, verbose=0, load_scans=False)
    det_rotation = volumax.compute_geometry(volumax_params, verbose=0)['det_rotation']
    ct_model_cal = mbirtorch.copy_ct_model(ct_model, no_warning=True)
    volumax.calibrate_volumax_geometry(ct_model_cal, sino, det_rotation=det_rotation)
    meta_offset = ct_model.get_params('det_channel_offset')
    cal_offset = ct_model_cal.get_params('det_channel_offset')
    t0 = time.time()
    recon_cal, _ = ct_model_cal.recon(sino, weights=weights, max_iterations=max_iterations)
    recon_cal = np.asarray(recon_cal)
    print(f'Calibrated reconstruction (det_channel_offset {cal_offset:+.4f} mm instead of {meta_offset:+.4f} mm) in '
          f'{time.time() - t0:.1f} s')

# ----------------------------------------------------------------------------------------------
# 5. View the reconstruction
# ----------------------------------------------------------------------------------------------
if show_viewer:
    if compare_calibrated:
        mbirtorch.slice_viewer(recon, recon_cal, vmin=0.0, vmax=0.1,
                               slice_label=[f'metadata offset {meta_offset:+.3f} mm',
                                            f'calibrated offset {cal_offset:+.3f} mm'],
                               title='VoluMax reconstruction: metadata vs calibrated channel offset')
    else:
        mbirtorch.slice_viewer(recon, data_dicts=[recon_dict], vmin=0.0, vmax=0.1,
                               title='VoluMax cone-beam MBIR reconstruction')
