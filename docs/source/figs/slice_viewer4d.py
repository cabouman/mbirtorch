"""The 4D viewer on a synthetic 4D volume: a Shepp-Logan phantom that shifts by up to two
pixels, in a direction that turns 60 degrees per frame, next to the unshifted phantom.
The first figure shows the x-y plane with an ROI on an edge and its mean against frame.
The second shows the t-y plane, where the shift appears as a zigzag of the edges."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

import mbirtorch
from mbirtorch.viewers.slice_figure4d import PLANE_LABELS, SliceViewer4D

num_frames = 24
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range((96, 96, 8))
frames = []
for t in range(num_frames):
    angle = np.deg2rad(60 * t)
    shift = (int(round(2 * np.cos(angle))), int(round(2 * np.sin(angle))))
    frames.append(np.roll(phantom, shift, axis=(0, 1)))
moving = np.stack(frames)
labels = ['shifting phantom', 'static phantom']

# The x-y plane at frame 4, with an ROI on the left edge of the phantom, on the middle row.
# The frame is set first, so the ROI statistics describe the frame that is shown.
viewer = SliceViewer4D(moving, phantom, slice_label=labels, show_instructions=False,
                       movie_fn=mbirtorch.save_volume_as_gif,
                       title='x-y plane: an ROI on an edge and its mean against frame')
viewer.fig.canvas.draw()
viewer.frame_slider.set_val(4)
middle_row = phantom[48, :, 4]
edge = int(np.argmax(middle_row > 0.5 * middle_row.max()))
ax = viewer.axes[0]
for name, (x, y) in (('button_press_event', (edge, 48)),
                     ('motion_notify_event', (edge + 4, 48)),
                     ('button_release_event', (edge + 4, 48))):
    px, py = ax.transData.transform((x, y))
    viewer.fig.canvas.callbacks.process(
        name, MouseEvent(name, viewer.fig.canvas, px, py, button=1))

# The t-y plane at x = 48: each row is one frame.
viewer_ty = SliceViewer4D(moving, phantom, slice_label=labels, show_instructions=False,
                          movie_fn=mbirtorch.save_volume_as_gif,
                          title='t-y plane: one line of the volume in every frame')
viewer_ty.axis_radios[0].set_active(PLANE_LABELS.index('t-y'))
viewer_ty.frame_slider.set_val(48)
