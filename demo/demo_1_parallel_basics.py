"""Demo 1: the basic MBIRTorch pipeline.

Make a simple 3D phantom, forward project it to get a sinogram, and
reconstruct it with model-based iterative reconstruction (MBIR).

In a real application you would skip the phantom and load your measured
sinogram as a numpy array with axes in the order
(views, detector rows, detector channels).
"""

import numpy as np
import mbirtorch

# Problem size: small enough to run on a laptop CPU in about a minute.
num_views = 128
num_det_rows = 128
num_det_channels = 128

# Make a phantom and project it to get a synthetic sinogram.
phantom, sinogram, params = mbirtorch.generate_demo_data(
    model_type='parallel', object_type='shepp-logan',
    num_views=num_views, num_det_rows=num_det_rows,
    num_det_channels=num_det_channels)

# The generator also returns the projection angles it used.
angles = params['angles']

# Build the reconstruction model from the sinogram shape and the angles.
ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)

# Reconstruct.  Everything is at its default value.  The one parameter worth
# trying first is sharpness (default 1.0): higher gives crisper edges, lower
# gives smoother images.  To change it:  ct_model.set_params(sharpness=1.5)
recon, recon_dict = ct_model.recon(sinogram)

# Compare the reconstruction to the phantom.
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
print(f'Normalized RMS error between reconstruction and phantom: {nrmse:.3f}')

# View them side by side.  Use the sliders to change slice and intensity.
mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict], vmin=0.0,
                       title='Phantom (left) and MBIR reconstruction (right)', block=False)
mbirtorch.slice_viewer(sinogram, title='Sinogram', slice_axis=0)
