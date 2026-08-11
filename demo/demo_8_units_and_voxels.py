"""Demo 8: units, detector spacing, and voxel shape.

MBIRTorch measures every length in ALUs (arbitrary length units): you pick
one physical unit — say millimeters — and use it for every distance and
spacing.  By default the detector spacing is 1 ALU, and the reconstruction
uses cubic voxels sized to match the detector.

The one rule this demo drives home: after changing any geometry parameter,
call auto_set_recon_geometry() so the reconstruction geometry is recomputed.
Setting the parameter alone is not enough.
"""

import numpy as np
import mbirtorch

# A small cone-beam model.  All lengths below are in ALUs; if your detector
# spacing is 0.2 mm and you work in mm, you would enter 0.2.
num_views = 128
num_det_rows = 128
num_det_channels = 128
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

ct_model = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                   source_detector_dist=4 * num_det_channels,
                                   source_iso_dist=2 * num_det_channels)

print('Default geometry (detector spacing 1 ALU):')
print(f"  recon_shape = {tuple(ct_model.get_params('recon_shape'))}, "
      f"delta_voxel = {ct_model.get_params('delta_voxel'):.3f} ALU")

# Change the detector spacing.  THE RULE: this alone does not update the
# reconstruction geometry -- the voxel size below is now stale.
ct_model.set_params(delta_det_channel=0.5, delta_det_row=0.5)
print('After set_params(delta_det_channel=0.5) alone (STALE):')
print(f"  delta_voxel = {ct_model.get_params('delta_voxel'):.3f} ALU  "
      '<- unchanged, wrong')

# Recompute the geometry.  Now the voxel size follows the new spacing.
ct_model.auto_set_recon_geometry()
print('After auto_set_recon_geometry():')
print(f"  recon_shape = {tuple(ct_model.get_params('recon_shape'))}, "
      f"delta_voxel = {ct_model.get_params('delta_voxel'):.3f} ALU")

# Restore the default spacing.  The rule applies to every change.
ct_model.set_params(delta_det_channel=1.0, delta_det_row=1.0)
ct_model.auto_set_recon_geometry()

# Voxel shape.  Voxels need not be cubes: voxel_slice_aspect = 2.0 makes
# each slice twice as thick as the in-plane voxel size, halving the number
# of slices (useful when axial resolution matters less than memory).
ct_model.set_params(voxel_slice_aspect=2.0)
ct_model.auto_set_recon_geometry()          # the same rule again
recon_shape = tuple(ct_model.get_params('recon_shape'))
print(f'With voxel_slice_aspect = 2.0: recon_shape = {recon_shape}')

# Reconstruct with the thick slices and view the result.
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
sinogram = ct_model.forward_project(phantom)
recon, recon_dict = ct_model.recon(sinogram)

nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
print(f'Thick-slice reconstruction error vs its phantom: {nrmse:.3f}')

mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict], vmin=0.0,
                       title='Thick-slice (voxel_slice_aspect = 2) phantom '
                             'and reconstruction')
