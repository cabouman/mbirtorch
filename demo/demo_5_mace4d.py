"""Demo 5: 4D reconstruction of a moving object.

One continuous scan of an object that moves while the scanner turns is
reconstructed as one volume per time frame.  The views are divided into
overlapping angular windows, one per frame, and the MACE algorithm
reconstructs every frame at once, with a prior along time as well as space.
"""

import numpy as np
import mbirtorch

# The scan: two rotations of a parallel-beam scan of a rack and pinion.  The
# bar runs along the rotation axis, the long axis of the volume; the wheel
# turns one tooth over the scan and drives the bar one tooth pitch.  The
# sinogram is one normal scan: each view sees the object where it was at that
# moment.  Small enough for a laptop CPU in a few minutes.
num_views = 240
phantom_4d, sinogram, params = mbirtorch.generate_demo_data_4d(
    object_type='rack-and-pinion', model_type='parallel', num_views=num_views,
    num_rotations=2, num_det_rows=64, num_det_channels=64, num_steps=24)
angles = params['angles']

# The model of the whole scan, as for any reconstruction.
ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)

# The 4D model on top of it.  With 8 frames per rotation the frames are 45
# degrees apart, and with an overlap factor of 2 each spans 90 degrees of
# views and shares half of them with the next frame.
mace4d = mbirtorch.MACE4DModel(ct_model, frames_per_rotation=8, frame_overlap_factor=2.0)
print(f'{mace4d.num_frames} frames of shape {mace4d.recon_shape}')

# Reconstruct every frame.
recon_4d, recon_dict = mace4d.recon(sinogram, max_iterations=10)

# The baseline: each frame's window of views reconstructed on its own with
# the direct method, which is what a scan of a moving object gets without a
# 4D method.
direct_4d = []
for s in mace4d.view_slices:
    frame_model = mbirtorch.ParallelBeamModel((s.stop - s.start,) + sinogram.shape[1:], angles[s])
    direct_4d.append(frame_model.recon_fbp(sinogram[s]))
direct_4d = np.stack(direct_4d)

# The ground truth for each frame: the object at the step nearest the middle
# of the frame's window of views.
step_of_view = params['step_of_view']
truth_4d = np.stack([phantom_4d[step_of_view[(s.start + s.stop) // 2]] for s in mace4d.view_slices])
for name, volumes in (('FBP per window', direct_4d), ('4D MACE', recon_4d)):
    nrmse = np.linalg.norm(volumes - truth_4d) / np.linalg.norm(truth_4d)
    print(f'{name}: normalized RMS error over all frames {nrmse:.3f}')

# View them side by side, in the plane that holds the rotation axis and cuts
# through the gear, so the teeth and their motion are in view.  The frame
# slider and Play step through time, and a t-z plane shows the bar slide.
mbirtorch.slice_viewer4d(truth_4d, direct_4d, recon_4d, data_dicts=[None, None, recon_dict],
                         vmin=0.0, vmax=1.0, slice_axis=1,
                         slice_label=['GROUND TRUTH', 'FBP PER WINDOW', '4D MACE'],
                         title='Ground truth (left), FBP per window (middle), 4D MACE (right)')
