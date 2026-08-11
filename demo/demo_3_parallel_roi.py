"""Demo 3: a region-of-interest scan (object extends outside the field of view).

In many real parallel-beam applications the object is wider than the
detector, so only a region of interest is scanned.  Voxels outside the
field of view still contribute to some measurements, and if the
reconstruction ignores them, their contributions get pushed into the image
as artifacts.

The fix is to enlarge the reconstruction region a little (about 1.3 times
the field of view), giving those outside contributions somewhere to go.
This demo reconstructs without and with the enlargement so you can see the
artifacts and their fix.
"""

import numpy as np
import mbirtorch

# Problem size.
num_views = 128
num_det_rows = 128
num_det_channels = 128

sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = np.linspace(0, np.pi, num_views, endpoint=False)

# Make a phantom 1.5 times wider than the field of view in both lateral
# directions, and project it.  The generation model is told the phantom's
# true size; only its projection onto the detector is kept.
gen_model = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
phantom_shape = (int(1.5 * num_det_channels), int(1.5 * num_det_channels),
                 num_det_rows)
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)
gen_model.set_params(recon_shape=phantom_shape)
sinogram = gen_model.forward_project(phantom)

# Reconstruction 1: the default region (exactly the field of view).
# The outside contributions have nowhere to go, so artifacts appear.
model_default = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
recon_default, dict_default = model_default.recon(sinogram)

# Reconstruction 2: enlarge the region by 1.3 in both lateral directions.
model_padded = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
model_padded.scale_recon_shape(row_scale=1.3, col_scale=1.3)
recon_padded, dict_padded = model_padded.recon(sinogram)

# Compare both to the phantom over the same central region of interest.
def center_crop(volume, rows, cols):
    r0 = (volume.shape[0] - rows) // 2
    c0 = (volume.shape[1] - cols) // 2
    return volume[r0:r0 + rows, c0:c0 + cols, :]

rows, cols = recon_default.shape[0], recon_default.shape[1]
phantom_roi = center_crop(phantom, rows, cols)
padded_roi = center_crop(recon_padded, rows, cols)

nrmse_default = (np.linalg.norm(recon_default - phantom_roi)
                 / np.linalg.norm(phantom_roi))
nrmse_padded = (np.linalg.norm(padded_roi - phantom_roi)
                / np.linalg.norm(phantom_roi))
print(f'Region-of-interest error without enlargement: {nrmse_default:.3f}')
print(f'Region-of-interest error with enlargement:    {nrmse_padded:.3f}')

# View: phantom region, the artifacted reconstruction, and the fixed one.
mbirtorch.slice_viewer(
    phantom_roi, recon_default, padded_roi,
    data_dicts=[None, dict_default, dict_padded], vmin=0.0, vmax=1.0,
    title='Phantom region (left), default recon with artifacts (center),\n'
          'enlarged-region recon (right)')
