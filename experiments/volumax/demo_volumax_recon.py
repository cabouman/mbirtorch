import os
import time
import numpy as np
import torch
import mbirtorch
import mbirtorch.preprocess as mtp
from mbirtorch.preprocess import volumax

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------
# The scan folder: the folder that contains AcquisitionParameters.json.
scan_dir = '/depot/bouman/data/ORNL/volumax/Hexagonal_volumax/proj/M0685 - Hexagonal Part (Scanning Stragety Optimization)_2026-08-31T10-07-09'
# scan_dir = '/depot/bouman/data/ORNL/volumax/HIP_Can_Proj/HIP Can - Vanshika 92.8um Res_2025-09-12T14-46-58'
# Full resolution, downsample_factor = (1, 1), needs 8 GPUs of 80 GB.  The script reconstructs in
# parts with recon_split_sino when the full reconstruction does not fit on the GPUs it runs on.
downsample_factor = (2, 2)        # detector (rows, channels) block averaging
subsample_view_factor = 1         # keep every n-th view
fit_alignment = True              # refine the detector offsets and remove per-view jitter and rotation
sharpness = 1.0
snr_db = 35.0
max_iterations = 15
gpu_model = 'H100'                # the GPUs the job runs on, for the memory and time estimate
weight_type = None                # None, or a weight type for mbirtorch.gen_weights, e.g. 'transmission'
sinogram_path = None              # optional precomputed -log sinogram (.npy)
output_path = None                # e.g. './output/volumax_recon.h5'; None skips saving
show_viewer = True

# ----------------------------------------------------------------------------------------------
# 1. Sinogram and cone-beam model with the geometry from the metadata
# ----------------------------------------------------------------------------------------------
sino, ct_model, metadata = volumax.get_sino_and_model(scan_dir, downsample_factor=downsample_factor,
                                                      subsample_view_factor=subsample_view_factor,
                                                      sinogram_path=sinogram_path)
tube = metadata['acquisition']['tubeParameters']
print(f'Tube: {tube["accelerationVoltageInKV"]} kV, {tube["sourceCurrentInMicroA"]} uA')

# ----------------------------------------------------------------------------------------------
# 2. Channel offset estimated from the sinogram
# ----------------------------------------------------------------------------------------------
ct_model.set_params(det_channel_offset=mtp.estimate_det_channel_offset(ct_model, sino))

# ----------------------------------------------------------------------------------------------
# 3. Offsets and rotation refined from the reprojection of a direct reconstruction
# ----------------------------------------------------------------------------------------------
# fit_det_alignment compares each view with the reprojection of a first reconstruction.  The
# median over views of the fitted offsets goes into the model; each view's deviation from the
# median and its rotation are resampled out of the data by correct_det_alignment.
if fit_alignment:
    recon_direct = ct_model.recon_direct(sino)
    model_params, view_params = mtp.fit_det_alignment(ct_model, sino, recon_direct, rotation=True)
    ct_model.set_params(**model_params)
    sino = mtp.correct_det_alignment(ct_model, sino, view_params)
    del recon_direct

# ----------------------------------------------------------------------------------------------
# 4. Weights, regularization, and reconstruction
# ----------------------------------------------------------------------------------------------
weights = None if weight_type is None else mbirtorch.gen_weights(sino, weight_type=weight_type)
ct_model.set_params(sharpness=sharpness, snr_db=snr_db)

t0 = time.time()
num_gpus = torch.cuda.device_count()
fits = True
if num_gpus > 0:
    estimate = mbirtorch.estimate_resources(ct_model, gpu_model=gpu_model, num_gpus=num_gpus,
                                            max_iterations=max_iterations)
    print(estimate)
    fits = estimate.recon.fits is not False      # an unknown answer runs the full reconstruction
if fits:
    recon, recon_dict = ct_model.recon(sino, weights=weights, max_iterations=max_iterations)
else:
    recon, recon_dict = ct_model.recon_split_sino(sino, weights=weights, max_iterations=max_iterations)
recon = np.asarray(recon)
print(f'Reconstruction {recon.shape} in {time.time() - t0:.1f} s; min {recon.min():.4f}, max {recon.max():.4f}')

# ----------------------------------------------------------------------------------------------
# 5. Optional: save the reconstruction
# ----------------------------------------------------------------------------------------------
if output_path is not None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ct_model.save_recon_hdf5(output_path, recon, recon_dict)
    print(f'Reconstruction saved to {output_path}')

# ----------------------------------------------------------------------------------------------
# 6. View the reconstruction
# ----------------------------------------------------------------------------------------------
if show_viewer:
    mbirtorch.slice_viewer(recon, data_dicts=[recon_dict], vmin=0.0, vmax=0.1,
                           title='VoluMax cone-beam MBIR reconstruction')
