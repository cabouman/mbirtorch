"""Demo 6: helical cone-beam reconstruction.

In a helical scan the object moves steadily along the rotation axis while
the source rotates, so a short detector can cover a long object.  Each view
therefore has an axial shift as well as an angle.  The pitch is the travel
per rotation divided by the detector height, so a pitch of 1.0 means the
object moves one detector height per rotation.
"""

import numpy as np
import mbirtorch

# Problem size.  The detector is short (few rows); the helix covers a
# longer object.
num_views = 180
num_det_rows = 32
num_det_channels = 128

helical_pitch = 1.0     # travel per rotation, in detector heights
helical_z_range = 40.0  # total travel over the scan, in ALU

# Make a phantom and its helical sinogram.
phantom, sinogram, params = mbirtorch.generate_demo_data(
    model_type='cone', object_type='shepp-logan',
    num_views=num_views, num_det_rows=num_det_rows,
    num_det_channels=num_det_channels,
    use_helical=True, helical_pitch=helical_pitch,
    helical_z_range=helical_z_range)

# The per-view axial shifts are what make the scan helical.
angles = params['angles']
helical_z_shifts = params['helical_z_shifts']
source_detector_dist = params['source_detector_dist']
source_iso_dist = params['source_iso_dist']

# Build the model.  A helical scan uses the ordinary cone-beam model with
# one addition: the per-view axial shifts.
ct_model = mbirtorch.ConeBeamModel(sinogram.shape, angles,
                                   source_detector_dist=source_detector_dist,
                                   source_iso_dist=source_iso_dist,
                                   helical_z_shifts=helical_z_shifts)

# Reconstruct.  Note that the reconstruction covers the full helical
# travel, so it has many more slices than the detector has rows.
recon, recon_dict = ct_model.recon(sinogram)
print(f'Detector rows: {num_det_rows}; reconstruction slices: {recon.shape[2]}')

nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
print(f'Normalized RMS error between reconstruction and phantom: {nrmse:.3f}')

mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict], vmin=0.0,
                       title='Phantom (left) and helical reconstruction (right)', block=False)
mbirtorch.slice_viewer(sinogram, title='Sinogram', slice_axis=0)
