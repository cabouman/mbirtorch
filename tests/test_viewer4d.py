"""Tests for the 4D viewer (mbirtorch/viewers/slice_figure4d.py).

The model tests check the frame rules (a 3D volume stays fixed in time, and a
shorter 4D volume holds its last frame), how slice_axis counts each array's axes,
differences between 4D and 3D volumes, Load of a 4D array, the ROI mean against
frame, and the default display range.  The controller tests run headlessly on Agg:
they render a window, draw an ROI with mouse events, step frames, play either slider,
save GIFs, and open the viewer through the mbirtorch wrapper with a tensor.
"""

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

import mbirtorch
from mbirtorch.viewers.slice_figure import SliceViewer
from mbirtorch.viewers.slice_figure4d import PLANE_LABELS, SliceViewer4D, VolumeStack4D


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
        # Axis 1 of a 4D array and axis 0 of a 3D array both mean x.  The display axes
        # are rows, columns, slice axis, and second axis, in (t, x, y, z) numbering.
        stack = VolumeStack4D([vol4d, vol3d], slice_axis=[1, 0])
        assert stack.display_axes == [[2, 3, 1, 0], [2, 3, 1, 0]]
        assert stack.slice_image(0).shape == stack.slice_image(1).shape == (4, 5)
        # The default is z for both.
        assert VolumeStack4D([vol4d, vol3d]).display_axes == [[1, 2, 3, 0], [1, 2, 3, 0]]
        with pytest.raises(ValueError, match='4D array must be 1, 2, or 3'):
            VolumeStack4D([vol4d], slice_axis=0)


class TestSpaceTimePlanes:
    def test_t_x_plane_shows_one_line_in_every_frame(self):
        vol4d = make_volume((5, 4, 6, 3), 1)
        stack = VolumeStack4D([vol4d])
        stack.set_plane([0], (0, 1))
        # The slice slider keeps z, and the frame-row slider moves y.
        assert stack.display_axes[0] == [0, 1, 3, 2]
        assert stack.second_axis == 2
        y, z = stack.cur_seconds[0], stack.cur_slices[0]
        np.testing.assert_array_equal(stack.slice_image(0), vol4d[:, :, y, z])
        stack.set_second(5)
        np.testing.assert_array_equal(stack.slice_image(0), vol4d[:, :, 5, z])

    def test_3d_volume_is_constant_and_shorter_volume_shows_its_own_frames(self):
        stack = VolumeStack4D([make_volume((5, 4, 6, 3)), make_volume((3, 4, 6, 3), 1),
                               make_volume((4, 6, 3), 2)])
        stack.set_plane([0, 1, 2], (0, 1))
        assert [stack.slice_image(i).shape[0] for i in range(3)] == [5, 3, 5]
        image = stack.slice_image(2)
        assert np.all(image == image[0])

    def test_frame_index_returns_with_a_spatial_plane(self):
        vol4d = make_volume((5, 4, 6, 3), 1)
        stack = VolumeStack4D([vol4d])
        stack.set_frame(3)
        stack.set_plane([0], (0, 2))
        assert stack.display_axes[0] == [0, 2, 3, 1]
        stack.set_plane([0], (1, 2))
        assert stack.display_axes[0] == [1, 2, 3, 0] and stack.master_frame == 3
        np.testing.assert_array_equal(stack.slice_image(0),
                                      vol4d[3][:, :, stack.cur_slices[0]])

    def test_t_z_plane_puts_y_on_the_slice_slider(self):
        stack = VolumeStack4D([make_volume((5, 4, 6, 3))])
        stack.set_plane([0], (0, 3))
        assert stack.display_axes[0] == [0, 3, 2, 1]
        stack.transpose(0)
        assert stack.display_axes[0] == [3, 0, 2, 1]
        assert stack.slice_image(0).shape == (3, 5)


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

    def test_roi_plot_returns_after_a_space_time_plane(self, make_viewer):
        viewer = make_viewer(shifting_square(), make_volume((32, 32, 4)))
        ax = viewer.axes[0]
        x0, y0 = ax.transData.transform((10, 16))
        x1, y1 = ax.transData.transform((13, 16))
        _process(viewer.fig, 'button_press_event', x0, y0, 1)
        _process(viewer.fig, 'motion_notify_event', x1, y1)
        _process(viewer.fig, 'button_release_event', x1, y1, 1)
        means = np.asarray(viewer._roi_lines[0].get_ydata())
        colors = [line.get_color() for line in viewer._roi_lines]
        radio = viewer.axis_radios[0]
        radio.set_active(PLANE_LABELS.index('t-y'))
        assert viewer.stack.display_axes[0] == [0, 2, 3, 1]
        assert not viewer.roi_plot_ax.get_visible()
        radio.set_active(PLANE_LABELS.index('x-y'))
        assert viewer.roi_plot_ax.get_visible()
        np.testing.assert_allclose(viewer._roi_lines[0].get_ydata(), means)
        # Each volume keeps its color when the plot is recomputed.
        assert [line.get_color() for line in viewer._roi_lines] == colors

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

    def test_dialog_layout_matches_the_slice_viewer_in_inches(self, make_viewer):
        # The inherited dialogs are laid out for the slice viewer's 8-inch figure, and
        # the 4D figure is taller.  The distances inside a dialog must match in inches.
        def range_dialog_layout(viewer):
            viewer._open_range_dialog_infigure()
            viewer.fig.canvas.draw()
            dpi = viewer.fig.dpi
            panel = viewer._dialog['panel_ax'].bbox
            hint_y = (viewer._dialog['texts']['hint'].get_position()[1]
                      * viewer.fig.bbox.height)
            min_box = viewer._dialog['widgets']['min'].ax.bbox
            return [panel.height / dpi, (panel.y1 - hint_y) / dpi,
                    (panel.y1 - min_box.y1) / dpi]

        viewer_3d = SliceViewer(make_volume((32, 32, 4)))
        viewer_3d.fig.canvas.draw()
        try:
            expected = range_dialog_layout(viewer_3d)
        finally:
            plt.close(viewer_3d.fig)
        np.testing.assert_allclose(range_dialog_layout(make_viewer(shifting_square())),
                                   expected, atol=0.01)

    def test_space_time_plane_shows_the_shift_as_a_zigzag(self, make_viewer):
        viewer = make_viewer(shifting_square(), slice_label='moving')
        radio = viewer.axis_radios[0]
        radio.set_active(PLANE_LABELS.index('t-y'))
        assert viewer.stack.display_axes[0] == [0, 2, 3, 1]
        assert viewer.frame_slider.label.get_text() == 'x'
        assert viewer.slice_slider.label.get_text() == 'z'
        assert not viewer.roi_plot_ax.get_visible()
        assert viewer.axes[0].get_aspect() == 'auto'
        assert viewer.axes[0].get_title().splitlines()[0] == 'moving: x = 16, z = 2'
        # Each row is one frame.  The left edge of the square repeats every 6 frames.
        image = viewer.stack.slice_image(0)
        left_edges = np.argmax(image > 0.5, axis=1)
        np.testing.assert_array_equal(left_edges[:6], left_edges[6:])
        assert len(set(left_edges[:6])) > 1
        # In a spatial plane the frame row sets the frame, and the ROI plot returns.
        radio.set_active(PLANE_LABELS.index('x-y'))
        assert viewer.frame_slider.label.get_text() == 't'
        assert viewer.roi_plot_ax.get_visible()
        assert viewer.axes[0].get_aspect() == 1.0

    def test_position_sliders_sit_together_above_the_intensity_slider(self, make_viewer):
        viewer = make_viewer(shifting_square())
        boxes = [slider.ax.get_position() for slider in
                 (viewer.slice_slider, viewer.frame_slider, viewer.intensity_slider)]
        assert boxes[0].y0 > boxes[1].y0 > boxes[2].y0
        # The three sliders line up.
        for box in boxes[1:]:
            assert (box.x0, box.x1) == pytest.approx((boxes[0].x0, boxes[0].x1))

    def test_space_time_plane_couples_the_panels(self, make_viewer):
        viewer = make_viewer(shifting_square(), shifting_square())
        viewer._toggle_couple_axes()
        assert not viewer.sync_axes and len(viewer.axis_radios) == 2
        viewer.axis_radios[1].set_active(PLANE_LABELS.index('t-x'))
        assert viewer.sync_axes and len(viewer.axis_radios) == 1
        assert viewer.stack.display_axes == [[0, 1, 3, 2], [0, 1, 3, 2]]
        labels = [label for label, _callback in viewer._menu_items(0)]
        assert 'Decouple slice axes' not in labels

    def test_shorter_volume_lines_up_in_time(self, make_viewer):
        viewer = make_viewer(shifting_square(12), shifting_square(6))
        viewer.axis_radios[0].set_active(PLANE_LABELS.index('t-y'))
        assert viewer.axes[0].get_ylim() == viewer.axes[1].get_ylim() == (11.5, -0.5)

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

    def test_slice_row_plays_along_the_slice_axis(self, make_viewer):
        viewer = make_viewer(shifting_square(), slice_label='moving')
        assert all(viewer.play_buttons[row].ax.get_visible() for row in ('slice', 'frame'))
        viewer.frame_slider.set_val(3)
        viewer._toggle_play('slice')
        assert viewer.playing and viewer.play_buttons['slice'].label.get_text() == 'Pause'
        # The timer does not fire on Agg, so the playback steps are taken by hand.  The
        # slice row loops through z at the current frame.
        viewer._play_step()
        assert (viewer.stack.master_index, viewer.stack.master_frame) == (3, 3)
        viewer._play_step()
        assert viewer.stack.master_index == 0
        np.testing.assert_array_equal(np.asarray(viewer.images[0].get_array()),
                                      viewer.stack.slice_image(0))
        viewer._toggle_play('slice')
        assert not viewer.playing and viewer.slice_slider.val == 0
        assert viewer.axes[0].get_title().splitlines()[0] == 'moving: t = 3, z = 0'

    def test_slice_row_is_hidden_without_slices(self, make_viewer):
        viewer = make_viewer(shifting_square(depth=1))
        assert not viewer.play_buttons['slice'].ax.get_visible()
        assert viewer.play_buttons['frame'].ax.get_visible()

    def test_one_row_plays_at_a_time(self, make_viewer):
        viewer = make_viewer(shifting_square())
        viewer._toggle_play('frame')
        viewer._toggle_play('slice')
        assert viewer.playing and viewer._play_row == 'slice'
        assert viewer.play_buttons['frame'].label.get_text() == 'Play'
        assert viewer.play_buttons['slice'].label.get_text() == 'Pause'
        # Space pauses whichever row plays.
        _press_key(viewer.fig, ' ')
        assert not viewer.playing

    def test_space_time_playback_sweeps_the_hidden_axis(self, make_viewer):
        viewer = make_viewer(shifting_square(), slice_label='moving')
        viewer.axis_radios[0].set_active(PLANE_LABELS.index('t-y'))
        assert viewer.play_buttons['frame'].ax.get_visible()
        # In t-y the frame row plays x, so every step shows the t-y image at the next x.
        _press_key(viewer.fig, ' ')
        assert viewer.playing
        viewer._play_step()
        assert viewer.stack.second_position == 17
        np.testing.assert_array_equal(np.asarray(viewer.images[0].get_array()),
                                      viewer.stack.slice_image(0))
        _press_key(viewer.fig, ' ')
        assert not viewer.playing and viewer.frame_slider.val == 17
        assert viewer.axes[0].get_title().splitlines()[0] == 'moving: x = 17, z = 2'

    def test_playback_without_blitting_keeps_the_images(self, make_viewer):
        # The WebAgg and notebook canvases cannot blit, so playback redraws the whole
        # figure on each frame, and the images must stay in those draws.
        viewer = make_viewer(shifting_square())
        canvas = viewer.fig.canvas
        canvas.supports_blit = False
        _press_key(viewer.fig, ' ')
        viewer._play_step()
        assert viewer.playing and viewer.stack.master_frame == 1
        canvas.draw()
        # A row through the square has edges, and a blank panel has none.
        pixels = np.asarray(canvas.buffer_rgba())
        bbox = viewer.axes[0].bbox
        row = int(pixels.shape[0] - (bbox.y0 + bbox.y1) / 2)
        assert pixels[row, int(bbox.x0):int(bbox.x1), :3].std() / 255 > 0.05
        _press_key(viewer.fig, ' ')
        assert not viewer.playing


def gif_frames(volume, frame_axis, slice_axis, slice_index):
    """The frames that save_volume_as_gif writes for these arguments, as its docstring
    states them: slice_axis held at slice_index, frame_axis looping, and the other two
    axes in increasing order."""
    index = [slice(None)] * 4
    index[slice_axis] = slice_index
    return np.moveaxis(np.asarray(volume)[tuple(index)],
                       frame_axis - (frame_axis > slice_axis), 0)


class TestMovie:
    @pytest.mark.parametrize('plane, transpose', [((1, 2), False), ((1, 2), True),
                                                  ((0, 2), False), ((0, 3), True)])
    @pytest.mark.parametrize('row', ['slice', 'frame'])
    def test_movie_frames_match_the_panel(self, plane, transpose, row):
        # A movie along either hidden axis shows what the panel shows when that
        # axis's slider steps.
        stack = VolumeStack4D([make_volume((5, 4, 6, 3), 1)])
        stack.set_plane([0], plane)
        if transpose:
            stack.transpose(0)
        slice_axis, second_axis = stack.display_axes[0][2:]
        axis = slice_axis if row == 'slice' else second_axis
        frames = gif_frames(*stack.movie_view(0, axis))
        assert len(frames) == stack.movie_frame_count(0, axis)
        for k in range(len(frames)):
            if row == 'slice':
                stack.set_master_index(k)
            else:
                stack.set_second(k)
            np.testing.assert_array_equal(frames[k], stack.slice_image(0))

    def test_gif_button_writes_one_gif_per_panel_that_moves(self, make_viewer, tmp_path):
        calls = []
        viewer = make_viewer(shifting_square(), make_volume((32, 32, 4)),
                             slice_label=['moving', 'static'],
                             movie_fn=lambda volume, filename, **kwargs:
                                 calls.append((os.path.basename(filename), kwargs)))
        assert 'Save movie' not in [label for label, _callback in viewer._menu_items(0)]
        # On Agg the in-figure folder dialog opens.  A 3D volume does not change in
        # time, so the frame row writes a GIF of the 4D panel only.
        viewer._on_gif_button('frame')
        assert viewer._dialog['kind'] == 'gif'
        viewer._dialog['widgets']['path'].set_val(str(tmp_path))
        viewer._gif_dialog_accept('frame')
        assert viewer._dialog is None
        assert [name for name, _kwargs in calls] == ['moving_x-y_along-t_z2.gif']
        kwargs = calls[0][1]
        assert (kwargs['frame_axis'], kwargs['slice_axis'], kwargs['slice_index']) == (0, 3, 2)
        assert (kwargs['vmin'], kwargs['vmax']) == viewer.images[0].get_clim()
        assert kwargs['fps'] == viewer.fps
        # Along z both panels change.  A file that exists already is kept.
        calls.clear()
        (tmp_path / 'static_x-y_along-z.gif').write_bytes(b'')
        viewer._write_gifs('slice', str(tmp_path))
        assert [name for name, _kwargs in calls] == ['moving_x-y_along-z_t0.gif',
                                                     'static_x-y_along-z_2.gif']
        kwargs = calls[0][1]
        assert (kwargs['frame_axis'], kwargs['slice_axis'], kwargs['slice_index']) == (3, 0, 0)

    def test_gif_buttons_need_a_writer(self, make_viewer):
        viewer = make_viewer(shifting_square())
        assert [len(viewer._row_buttons[row]) for row in ('slice', 'frame')] == [1, 1]

    def test_wrapper_writes_one_gif_frame_per_position(self, tmp_path):
        from PIL import Image
        import mbirtorch.viewers.slice_figure as viewer_module

        with pytest.warns(UserWarning, match='non-interactive'):
            viewer = mbirtorch.slice_viewer4d(shifting_square(), block=False)
        try:
            assert viewer.movie_fn is mbirtorch.save_volume_as_gif
            viewer._write_gifs('frame', str(tmp_path))
            viewer._write_gifs('slice', str(tmp_path))
            with Image.open(tmp_path / 'volume_x-y_along-t_z2.gif') as gif:
                assert gif.n_frames == 12
            with Image.open(tmp_path / 'volume_x-y_along-z_t0.gif') as gif:
                assert gif.n_frames == 4
        finally:
            plt.close(viewer.fig)
            if viewer in viewer_module._NONBLOCKING_VIEWERS:
                viewer_module._NONBLOCKING_VIEWERS.remove(viewer)


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
