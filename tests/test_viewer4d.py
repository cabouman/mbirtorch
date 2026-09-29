"""Tests for the 4D viewer (mbirtorch/viewers/slice_figure4d.py).

The model tests check the frame rules (a 3D volume stays fixed in time, and a
shorter 4D volume holds its last frame), how slice_axis counts each array's axes,
differences between 4D and 3D volumes, Load of a 4D array, the ROI mean against
frame, and the default display range.  The controller tests run headlessly on Agg:
they render a window, draw an ROI with mouse events, step and play frames, and open
the viewer through the mbirtorch wrapper with a tensor.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

import mbirtorch
from mbirtorch.viewers.slice_figure4d import SliceViewer4D, VolumeStack4D


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


def shifting_square(num_frames=12, n=32, depth=4):
    """A square that shifts along y by up to 2 pixels, repeating every 6 frames."""
    volume = np.zeros((num_frames, n, n, depth), dtype=np.float32)
    for t in range(num_frames):
        shift = int(round(2 * np.cos(2 * np.pi * t / 6)))
        volume[t, 10:22, 10 + shift:22 + shift, :] = 1.0
    return volume


# ---------------------------------------------------------------------------
# The data model (VolumeStack4D)
# ---------------------------------------------------------------------------

class TestFrames:
    def test_3d_volume_stays_fixed_while_4d_volume_steps(self):
        vol4d, vol3d = make_volume((5, 4, 6, 3), 1), make_volume((4, 6, 3), 2)
        stack = VolumeStack4D([vol4d, vol3d])
        assert stack.frame_counts == [5, 1]
        assert stack.set_frame(3) == [0]
        np.testing.assert_array_equal(stack.slice_image(0), vol4d[3][:, :, 1])
        np.testing.assert_array_equal(stack.slice_image(1), vol3d[:, :, 1])
        assert stack.has_time(0) and not stack.has_time(1)

    def test_shorter_volume_holds_its_last_frame(self):
        stack = VolumeStack4D([make_volume((5, 4, 4, 2)), make_volume((3, 4, 4, 2))])
        stack.set_frame(4)
        assert stack.cur_frames == [4, 2]
        assert stack.holds_last_frame(1) and not stack.holds_last_frame(0)
        # A frame index past the longest volume is clipped.
        stack.set_frame(10)
        assert stack.master_frame == 4

    def test_slice_axis_counts_each_arrays_own_axes(self):
        vol4d, vol3d = make_volume((2, 3, 4, 5)), make_volume((3, 4, 5))
        # Axis 1 of a 4D array and axis 0 of a 3D array both mean x.
        stack = VolumeStack4D([vol4d, vol3d], slice_axis=[1, 0])
        assert stack.axes_perms == [[1, 2, 0], [1, 2, 0]]
        assert stack.slice_image(0).shape == stack.slice_image(1).shape == (4, 5)
        # The default is z for both.
        assert VolumeStack4D([vol4d, vol3d]).axes_perms == [[0, 1, 2], [0, 1, 2]]
        with pytest.raises(ValueError, match='4D array must be 1, 2, or 3'):
            VolumeStack4D([vol4d], slice_axis=0)


class TestDifference:
    def test_3d_volume_is_subtracted_from_every_frame(self):
        vol4d, vol3d = make_volume((4, 5, 6, 3), 1), make_volume((5, 6, 3), 2)
        stack = VolumeStack4D([vol4d, vol3d])
        assert stack.can_difference(0, 1)
        stack.apply_difference(0, 1)  # panel 0 now shows panel 1 minus panel 0
        np.testing.assert_allclose(stack.frames[0], vol3d[np.newaxis] - vol4d)
        stack.set_frame(2)
        np.testing.assert_allclose(stack.slice_image(0), (vol3d - vol4d[2])[:, :, 1])
        assert stack.labels[0].startswith('Image 1 minus current')
        stack.restore(0)
        np.testing.assert_array_equal(stack.frames[0], vol4d)

    def test_3d_panel_minus_4d_volume_changes_with_time(self):
        vol4d, vol3d = make_volume((4, 5, 6, 3), 1), make_volume((5, 6, 3), 2)
        stack = VolumeStack4D([vol4d, vol3d])
        stack.apply_difference(1, 0)
        assert stack.frame_counts == [4, 4] and stack.has_time(1)

    def test_volumes_of_different_xyz_shape_are_refused(self):
        stack = VolumeStack4D([make_volume((4, 5, 6, 3)), make_volume((5, 6, 4))])
        assert not stack.can_difference(0, 1)


class TestLoadRoiAndRange:
    def test_loaded_4d_array_becomes_one_volume_in_its_panel(self):
        vol3d = make_volume((5, 6, 3))
        stack = VolumeStack4D([vol3d, vol3d])
        vol4d = make_volume((7, 5, 6, 3), 3)
        assert stack.load_array(1, vol4d) == [1]
        assert stack.original_data[1] is vol4d
        assert stack.original_data[0] is vol3d
        assert stack.frame_counts == [1, 7]

    def test_roi_frame_means_match_a_manual_circle(self):
        vol4d = make_volume((6, 16, 16, 3), 4)
        stack = VolumeStack4D([vol4d])
        x, y, radius = 7.3, 5.1, 3.2
        yv, xv = np.ogrid[:16, :16]
        mask = (xv - x) ** 2 + (yv - y) ** 2 <= radius ** 2
        expected = [vol4d[t][:, :, 1][mask].mean() for t in range(6)]
        np.testing.assert_allclose(stack.roi_frame_means(0, x, y, radius), expected,
                                   rtol=1e-6)

    def test_default_range_covers_every_frame(self):
        vol4d = np.zeros((5, 4, 4, 2), dtype=np.float32)
        vol4d[3, 1, 1, 1] = 9.0  # a hot voxel away from the first frame
        stack = VolumeStack4D([vol4d])
        assert (stack.vmin, stack.vmax) == (0.0, 9.0)


# ---------------------------------------------------------------------------
# The window (SliceViewer4D), headless on Agg
# ---------------------------------------------------------------------------

def _process(fig, name, x, y, button=None):
    event = MouseEvent(name, fig.canvas, x, y, button=button)
    fig.canvas.callbacks.process(name, event)


def _press_key(fig, key):
    fig.canvas.callbacks.process('key_press_event',
                                 KeyEvent('key_press_event', fig.canvas, key))


@pytest.fixture
def make_viewer():
    created = []

    def _make(*datasets, **kwargs):
        viewer = SliceViewer4D(*datasets, **kwargs)
        viewer.fig.canvas.draw()
        created.append(viewer)
        return viewer

    yield _make
    for viewer in created:
        plt.close(viewer.fig)


class TestWindow:
    def test_window_renders_with_frame_titles(self, make_viewer, tmp_path):
        viewer = make_viewer(shifting_square(), make_volume((32, 32, 4)),
                             slice_label=['moving', 'static'])
        assert viewer.axes[0].get_title().splitlines()[0] == 'moving: t = 0, z = 2'
        assert viewer.axes[1].get_title().splitlines()[0] == 'static: z = 2'
        path = str(tmp_path / 'viewer4d.png')
        viewer.fig.savefig(path)
        # The panel region must not be blank: a row through the square has edges.
        from matplotlib.image import imread
        pixels = imread(path)
        ax_bbox = viewer.axes[0].bbox
        height = viewer.fig.canvas.get_width_height()[1]
        row = int(height - (ax_bbox.y0 + ax_bbox.y1) / 2)
        panel_row = pixels[row, int(ax_bbox.x0):int(ax_bbox.x1), :3]
        assert panel_row.std() > 0.05

    def test_roi_plot_repeats_with_the_period_of_the_shift(self, make_viewer):
        viewer = make_viewer(shifting_square())
        ax = viewer.axes[0]
        # A circle on the left edge of the square (column 10, row 16), radius 3.
        x0, y0 = ax.transData.transform((10, 16))
        x1, y1 = ax.transData.transform((13, 16))
        _process(viewer.fig, 'button_press_event', x0, y0, 1)
        _process(viewer.fig, 'motion_notify_event', x1, y1)
        _process(viewer.fig, 'button_release_event', x1, y1, 1)
        means = np.asarray(viewer._roi_lines[0].get_ydata())
        assert means.size == 12
        np.testing.assert_allclose(means[:6], means[6:])
        assert means.std() > 0

    def test_menu_opens_at_the_cursor_on_a_retina_screen(self, make_viewer):
        # A Retina screen has two physical pixels per logical pixel, and mouse events
        # give the position in physical pixels.
        viewer = make_viewer(shifting_square())
        viewer.fig.canvas._set_device_pixel_ratio(2)
        viewer.fig.canvas.draw()
        bbox = viewer.axes[0].bbox
        x, y = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
        _process(viewer.fig, 'button_press_event', x, y, 3)
        menu = viewer._dialog['panel_ax'].get_position()
        assert menu.x0 == pytest.approx(x / viewer.fig.bbox.width, abs=0.005)
        assert menu.y1 == pytest.approx(y / viewer.fig.bbox.height, abs=0.005)

    def test_frame_keys_and_playback(self, make_viewer):
        viewer = make_viewer(shifting_square(), slice_label='moving')
        _press_key(viewer.fig, '.')
        assert viewer.stack.master_frame == 1
        _press_key(viewer.fig, ',')
        assert viewer.stack.master_frame == 0

        _press_key(viewer.fig, ' ')
        assert viewer.playing
        # The timer does not fire on Agg, so one playback step is taken by hand.
        viewer._play_step()
        assert viewer.stack.master_frame == 1
        np.testing.assert_array_equal(np.asarray(viewer.images[0].get_array()),
                                      viewer.stack.slice_image(0))
        _press_key(viewer.fig, ' ')
        assert not viewer.playing
        assert viewer.frame_slider.val == 1
        assert viewer.axes[0].get_title().splitlines()[0] == 'moving: t = 1, z = 2'

        # Playback loops at the end.
        viewer.frame_slider.set_val(11)
        _press_key(viewer.fig, ' ')
        viewer._play_step()
        assert viewer.stack.master_frame == 0
        _press_key(viewer.fig, ' ')
        assert not viewer.playing


class TestWrapper:
    def test_4d_tensor_is_converted(self):
        import torch
        import mbirtorch.viewers.slice_figure as viewer_module

        volume = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        with pytest.warns(UserWarning, match='non-interactive'):
            viewer = mbirtorch.slice_viewer4d(volume, block=False)
        try:
            np.testing.assert_array_equal(viewer.stack.original_data[0], volume.numpy())
            assert viewer.stack.frame_counts == [2]
        finally:
            plt.close(viewer.fig)
            if viewer in viewer_module._NONBLOCKING_VIEWERS:
                viewer_module._NONBLOCKING_VIEWERS.remove(viewer)
