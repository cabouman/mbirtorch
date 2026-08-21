"""Demo 7: the multiaxis parallel geometry (laminography).

In this geometry each view has two angles: the usual rotation about the
vertical axis (the azimuth), plus a tilt of the beam out of the horizontal
plane (the elevation).  A constant tilt is laminography, which is useful
for flat objects such as circuit boards.
"""

import numpy as np
import mbirtorch

# Problem size and tilt.
num_views = 120
num_det_rows = 96
num_det_channels = 128
elevation_degrees = 30.0

# Make a phantom and its tilted sinogram.
phantom, sinogram, params = mbirtorch.generate_demo_data(
    model_type='multiaxis', elevation_degrees=elevation_degrees,
    num_views=num_views, num_det_rows=num_det_rows,
    num_det_channels=num_det_channels)

# View the sinogram.  Each view looks at the object from 30 degrees above
# the horizontal, so the projections show the object at an angle.
mbirtorch.slice_viewer(sinogram, slice_axis=0, slice_label='View', vmin=0.0,
                       title=f'Sinogram at {elevation_degrees:.0f} degree tilt')

# The generator also returns the (azimuth, elevation) angle pairs it used.
angles = params['angles']

# Build the model and reconstruct.
ct_model = mbirtorch.MultiAxisParallelModel(sinogram.shape, angles)
recon, recon_dict = ct_model.recon(sinogram)

nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
print(f'Normalized RMS error between reconstruction and phantom: {nrmse:.3f}')

mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict], vmin=0.0,
                       title='Phantom (left) and laminography reconstruction (right)', block=False)
mbirtorch.slice_viewer(sinogram, title='Sinogram', slice_axis=0)
