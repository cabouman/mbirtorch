"""Demo 4: cone-beam artifacts when the object extends past the top and
bottom of the field of view.

In a cone-beam scan the X-rays diverge, so material just above and below
the field of view is still measured in many views.  If the reconstruction
stops exactly at the field of view, those measurements corrupt the top and
bottom slices.

The fix is the axial_pad_fraction parameter, which extends the
reconstruction axially so the outside material has somewhere to go.  This
demo reconstructs without and with the padding.
"""

import numpy as np
import mbirtorch

# Problem size.
num_views = 128
num_det_rows = 128
num_det_channels = 128

sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
source_detector_dist = 4 * num_det_channels
source_iso_dist = 2 * num_det_channels

# Make a phantom 1.5 times taller than the field of view, and project it.
gen_model = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                    source_detector_dist=source_detector_dist,
                                    source_iso_dist=source_iso_dist)
fov_shape = tuple(gen_model.get_params('recon_shape'))
phantom_shape = (fov_shape[0], fov_shape[1], int(1.5 * fov_shape[2]))
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)
gen_model.set_params(recon_shape=phantom_shape)
sinogram = gen_model.forward_project(phantom)

# Reconstruction 1: the default region.  The slices near the top and
# bottom are corrupted by the material outside the field of view.
model_default = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                        source_detector_dist=source_detector_dist,
                                        source_iso_dist=source_iso_dist)
recon_default, dict_default = model_default.recon(sinogram)

# Reconstruction 2: extend the region axially.  axial_pad_fraction=1.0
# pads each end far enough to cover every measured ray.  After changing a
# geometry parameter, call auto_set_recon_geometry() to recompute the
# reconstruction region.
model_padded = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                       source_detector_dist=source_detector_dist,
                                       source_iso_dist=source_iso_dist)
model_padded.set_params(axial_pad_fraction=1.0)
model_padded.auto_set_recon_geometry()
recon_padded, dict_padded = model_padded.recon(sinogram)

# Compare both to the phantom over the slices of the field of view.
def center_slices(volume, num_slices):
    s0 = (volume.shape[2] - num_slices) // 2
    return volume[:, :, s0:s0 + num_slices]

num_fov_slices = recon_default.shape[2]
phantom_fov = center_slices(phantom, num_fov_slices)
padded_fov = center_slices(recon_padded, num_fov_slices)

# The artifacts concentrate in the slices near the ends, so measure there:
# the top and bottom eighth of the field of view.
n_end = max(1, num_fov_slices // 8)
ends = list(range(n_end)) + list(range(num_fov_slices - n_end, num_fov_slices))

def end_error(recon):
    diff = recon[:, :, ends] - phantom_fov[:, :, ends]
    return np.linalg.norm(diff) / np.linalg.norm(phantom_fov[:, :, ends])

print(f'End-slice error without axial padding: {end_error(recon_default):.3f}')
print(f'End-slice error with axial padding:    {end_error(padded_fov):.3f}')

# View all three.  Look at the top and bottom slices, where the difference
# is largest.
mbirtorch.slice_viewer(
    phantom_fov, recon_default, padded_fov,
    data_dicts=[None, dict_default, dict_padded], vmin=0.0, slice_axis=1,
    title='Phantom (left), default recon with axial artifacts (center),\n'
          'axially padded recon (right)', block=False)
mbirtorch.slice_viewer(sinogram, title='Sinogram', slice_axis=0)
