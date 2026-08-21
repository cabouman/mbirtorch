"""Demo 2: cone-beam reconstruction, with the practices real data needs.

This demo adds four things to the basic pipeline of demo 1:

1. Cone-beam geometry, which needs two distances: source to detector, and
   source to the rotation axis.
2. Simulated measurement noise with the physically correct structure:
   rays through dense material are noisier.
3. Noise weighting: the weights tell the reconstruction to trust the
   noisier measurements less.
4. Saving the reconstruction to a file.
"""

import numpy as np
import mbirtorch

# Problem size.
num_views = 128
num_det_rows = 128
num_det_channels = 128

# Make a phantom and its cone-beam sinogram.  target_max_attenuation scales
# the phantom so the sinogram is in attenuation units (the units of real
# -log(I/I0) data), roughly in the range [0, 6].
phantom, sinogram, params = mbirtorch.generate_demo_data(
    model_type='cone', object_type='shepp-logan',
    num_views=num_views, num_det_rows=num_det_rows,
    num_det_channels=num_det_channels, target_max_attenuation=6.0)

# Add measurement noise.  For a transmission scan with a dosage of
# lambda_0 input photons per measurement, the attenuation measurements are
# approximately
#     y = ybar + sqrt(exp(ybar) / lambda_0) * W,    W ~ N(0, 1),
# so the noise standard deviation grows with attenuation.  (Bouman and
# Sauer, "A Unified Approach to Statistical Tomography Using Coordinate
# Descent Optimization," IEEE Trans. on Image Processing, 1996.)
dosage = 10000.0
noise_std = np.sqrt(np.exp(sinogram) / dosage)
rng = np.random.default_rng(0)
sinogram = sinogram + noise_std * rng.standard_normal(sinogram.shape).astype(np.float32)

# The generator also returns the geometry it used.
angles = params['angles']
source_detector_dist = params['source_detector_dist']
source_iso_dist = params['source_iso_dist']

# Build the cone-beam model.  The two distances set the cone geometry.
ct_model = mbirtorch.ConeBeamModel(sinogram.shape, angles,
                                   source_detector_dist=source_detector_dist,
                                   source_iso_dist=source_iso_dist)

# Noise weights.  The noise model above has variance exp(y) / lambda_0, so
# down-weighting by the transmission gives the noisier measurements less
# influence.  For a first look at any new data set, weights=None is also fine.
weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')

# Sharpness is the main image-quality control: higher gives crisper edges,
# lower gives smoother images.  Typical useful range is about -1 to 2.
ct_model.set_params(sharpness=1.0)

# Reconstruct.
recon, recon_dict = ct_model.recon(sinogram, weights=weights)

nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
print(f'Normalized RMS error between reconstruction and phantom: {nrmse:.3f}')

# Save the reconstruction and its settings to one file.  The file can be
# reloaded later for viewing, or to continue from this result:
#     recon, recon_dict = mbirtorch.TomographyModel.load_recon_hdf5(filepath)
filepath = './output/demo2_recon.h5'
ct_model.save_recon_hdf5(filepath, recon, recon_dict)
print(f'Reconstruction saved to {filepath}')

# View the phantom and the reconstruction side by side.
mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict], vmin=0.0,
                       title='Phantom (left) and cone-beam MBIR reconstruction (right)', block=False)
mbirtorch.slice_viewer(sinogram, title='Sinogram', slice_axis=0)
