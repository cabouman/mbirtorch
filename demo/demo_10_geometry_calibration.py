"""Demo 10: calibrating the scan geometry from the sinogram.

A scan's metadata sometimes gets a geometry parameter wrong, and sometimes it
leaves the parameter out.  This demo makes a cone-beam scan whose true geometry
differs from the model's in two ways, and then recovers both differences from
the sinogram.

It shows four things:

1. What a wrong center of rotation and a wrong detector rotation do to a
   reconstruction.
2. The rotation-direction check, which decides from the data whether the view
   angles run the right way.
3. The automatic workflow: estimate the channel offset, then the detector
   rotation at that offset, then the channel offset again at that rotation.
4. The manual workflow: reconstruct one slice per candidate offset and choose
   the value by eye.
"""

import numpy as np
import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc
from mbirtorch.preprocess.utilities import correct_det_rotation

# Problem size.
num_views = 128
num_det_rows = 32
num_det_channels = 128

# These are the two geometry errors the demo recovers.  The offset is in
# detector channels, and the rotation is about the optical axis.
true_offset_channels = 1.7
true_rotation_degrees = 1.5

# Make the phantom and the cone-beam geometry.  The sinogram this call returns
# is not used, because the scan below is made with the true geometry instead.
phantom, _, params = mbirtorch.generate_demo_data(
    model_type='cone', object_type='shepp-logan', num_views=num_views,
    num_det_rows=num_det_rows, num_det_channels=num_det_channels)

# This is the model the user has.  Its center of rotation sits at the detector
# center and it carries no detector rotation, which is what the metadata said.
# Every model here uses compile_mode='off'.  This problem is small, and
# compiling the projectors would take longer than the rest of the demo.
ct_model = mbirtorch.ConeBeamModel((num_views, num_det_rows, num_det_channels),
                                   params['angles'],
                                   source_detector_dist=params['source_detector_dist'],
                                   source_iso_dist=params['source_iso_dist'],
                                   compile_mode='off')
delta_det_channel = float(ct_model.get_params('delta_det_channel'))

# The scanner's true geometry goes on a copy of that model.  Detector offsets
# are in ALU, so the offset in channels is multiplied by the channel pitch.
true_model = mbirtorch.copy_ct_model(ct_model)
true_model.compile_mode = ct_model.compile_mode   # copy_ct_model does not carry this over
true_model.set_params(det_channel_offset=true_offset_channels * delta_det_channel)

# The measured sinogram is projected at the true offset and then rotated.  The
# rotation is applied with the negative angle, so that applying the positive
# estimate later takes it back out.
sino = np.asarray(true_model.forward_project(phantom), dtype=np.float32)
sino = correct_det_rotation(sino, -np.radians(true_rotation_degrees))


def nrmse(recon):
    """The normalized root mean squared error against the phantom."""
    return float(np.linalg.norm(np.asarray(recon) - phantom) / np.linalg.norm(phantom))


# Reconstruct with the uncalibrated model, to see what the two errors cost.
recon_before = ct_model.recon_direct(sino)
print(f'Uncalibrated direct reconstruction: NRMSE {nrmse(recon_before):.3f}')

# Check that the view angles run in the right direction.  The check scores the
# angles as given and negated, and the lower score is the answer.  The two
# scores are close here, so the check also warns that its margin is small.  The
# margin grows with the fan angle, which is 14 degrees in this geometry.  It
# also shrinks while the geometry errors are still in the data.
direction = gc.check_rotation_direction(ct_model, sino)
print(f'Rotation direction: {direction.value:+.0f} (scores {direction.scores[0]:.4f} as given, '
      f'{direction.scores[1]:.4f} negated)')

# The automatic workflow runs three estimates.  The two quantities are coupled,
# so the offset is estimated a second time with the rotation applied.
offset_first = gc.estimate_det_channel_offset(ct_model, sino)
rotation = gc.estimate_det_rotation(ct_model, sino, det_channel_offset=offset_first.value)
offset = gc.estimate_det_channel_offset(ct_model, sino, det_rotation=rotation.value)
print(f'det_channel_offset: true {true_offset_channels:.3f} channels, '
      f'first estimate {offset_first.value / delta_det_channel:.3f}, '
      f'second estimate {offset.value / delta_det_channel:.3f}')
print(f'det_rotation: true {true_rotation_degrees:.3f} degrees, '
      f'estimate {np.degrees(rotation.value):.3f}')

# Apply both results.  The offset is set on the model, and the rotation is
# applied to the sinogram, which apply_calibration rotates in place.
ct_model, sino = gc.apply_calibration(ct_model, sino, [rotation, offset])
recon_after = ct_model.recon_direct(sino)
print(f'Calibrated direct reconstruction: NRMSE {nrmse(recon_after):.3f}')

# The manual workflow reconstructs one slice per candidate offset.  The
# candidates here run from two channels below the estimate to two channels
# above it.  A user pages through the stack and picks the sharpest slice.
sweep_offsets = offset.value + delta_det_channel * np.linspace(-2.0, 2.0, 5)
sweep = gc.parameter_sweep(ct_model, sino, 'det_channel_offset', sweep_offsets)

# View the phantom and the two reconstructions, then the sweep.
mbirtorch.slice_viewer(phantom, recon_before, recon_after, vmin=0.0,
                       title='Phantom (left), uncalibrated (center), calibrated (right)',
                       block=False)
mbirtorch.slice_viewer(sweep, title='det_channel_offset sweep: -2, -1, 0, +1, +2 channels '
                                    'around the estimate')
