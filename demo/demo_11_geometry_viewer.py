"""Demo 11: the geometry viewer.

The geometry viewer draws what a model describes: where the source, the
detector, and the reconstruction volume sit, which way the gantry turns, and
whether the volume projects inside the detector.  It is the first thing to
check when a reconstruction of real data looks wrong.  A detector offset with
the wrong sign, a detector too small for the object, or a helical travel that
leaves slices uncovered is visible in the drawing within seconds.

This demo builds a cone-beam scan with two detector offsets, projects the cube
phantom through it, and opens the viewer with the sinogram painted on the
detector face and the phantom drawn in the volume box.  A second geometry, with
the detector moved by ten channels, is drawn dashed over the first.  That is how
an estimated calibration is compared with a vendor geometry.

For a real scan, the model that a scanner reader returns is passed to
``mbirtorch.geometry_viewer`` in the same way.
"""

import numpy as np
import mbirtorch

# Problem size and the two source distances.
num_views = 180
num_det_rows = 96
num_det_channels = 128
source_detector_dist = 4.0 * num_det_channels
source_iso_dist = source_detector_dist / 2.0

# The detector offsets in ALU.  Change one and watch the detector iso marker
# move on the detector face and the offset label move in the top and side views.
det_channel_offset = 3.0
det_row_offset = -2.0

# The comparison: the same scan with the detector moved by this many channels.
comparison_channel_shift = 10.0

# One full turn of the gantry.
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

sinogram_shape = (num_views, num_det_rows, num_det_channels)
ct_model = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                   source_detector_dist=source_detector_dist,
                                   source_iso_dist=source_iso_dist)
ct_model.set_params(det_channel_offset=det_channel_offset,
                    det_row_offset=det_row_offset)

# The cube phantom and its sinogram.  Both are drawn by the viewer below.
phantom = mbirtorch.gen_cube_phantom(ct_model.get_params('recon_shape'))
sinogram = ct_model.forward_project(phantom)

# Where the center of the volume lands on the detector.  project_points is the
# model's own map from object points to fractional detector indices, and it is
# the map the drawing uses.
row, channel = ct_model.project_points([[0.0, 0.0, 0.0]], 0)
print(f"The volume's center lands at row {row[0]:.2f}, "
      f'channel {channel[0]:.2f} of view 0')

# Open the viewer.  A slider steps through the views.  The top row of toggles
# turns the source's path, the 3D zoom to the volume, and the angle-0 reference
# on and off, and the bottom row turns the sinogram, the phantom, and the
# comparison on and off.  The 3D panel turns with the mouse.  The comparison
# opens a second window that tables every difference between the two
# geometries.
delta_det_channel = ct_model.get_params('delta_det_channel')
mbirtorch.geometry_viewer(
    ct_model, show_trajectory=True, sinogram=sinogram, recon=phantom,
    compare=dict(det_channel_offset=det_channel_offset
                 + comparison_channel_shift * delta_det_channel),
    title='Cone-beam scan geometry with the cube phantom')
