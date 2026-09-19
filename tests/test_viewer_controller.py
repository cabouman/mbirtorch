"""Controller tests for the slice viewer, run headlessly on Agg.

Interaction tests synthesize MouseEvent/KeyEvent objects and process them
through fig.canvas.callbacks, the same technique matplotlib's own widget
tests use.  Layout is verified by savefig snapshots.
"""

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from mbirtorch.viewers.slice_figure import (Mode, SliceViewer, VolumeStack,
                                            _save_data_hdf5, slice_viewer)
import mbirtorch.viewers.slice_figure as viewer_module


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def _process(fig, name, x, y, button=None):
    event = MouseEvent(name, fig.canvas, x, y, button=button)
    fig.canvas.callbacks.process(name, event)


def press_data(fig, ax, xdata, ydata, button=1):
    x, y = ax.transData.transform((xdata, ydata))
    _process(fig, 'button_press_event', x, y, button)


def motion_data(fig, ax, xdata, ydata):
    x, y = ax.transData.transform((xdata, ydata))
    _process(fig, 'motion_notify_event', x, y)


def release_data(fig, ax, xdata, ydata, button=1):
    x, y = ax.transData.transform((xdata, ydata))
    _process(fig, 'button_release_event', x, y, button)


def click_widget(fig, widget_ax, button=1):
    """Press and release at the center of a widget's axes."""
    x = (widget_ax.bbox.x0 + widget_ax.bbox.x1) / 2
    y = (widget_ax.bbox.y0 + widget_ax.bbox.y1) / 2
    _process(fig, 'button_press_event', x, y, button)
    _process(fig, 'button_release_event', x, y, button)


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


@pytest.fixture
def make_viewer():
    created = []

    def _make(*datasets, **kwargs):
        viewer = SliceViewer(*datasets, **kwargs)
        viewer.fig.canvas.draw()
        created.append(viewer)
        return viewer

    yield _make
    for viewer in created:
        plt.close(viewer.fig)


# ---------------------------------------------------------------------------
# Construction and layout
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_panels_match_model(self, make_viewer):
        a, b = make_volume((16, 16, 8), 1), make_volume((16, 16, 8), 2)
        viewer = make_viewer(a, b, slice_label=['A', 'B'])
        assert len(viewer.axes) == 2
        for i in range(2):
            np.testing.assert_array_equal(
                np.asarray(viewer.images[i].get_array()),
                viewer.stack.slice_image(i))
            assert viewer.caxes[i] is not None
        title = viewer.axes[0].get_title()
        assert 'A 4' in title and 'Shape: (16, 16, 8)' in title

    def test_slice_slider_matches_deepest_volume_or_is_hidden(self,
                                                              make_viewer):
        # Volumes with slices: the slider spans the deepest one, steps by
        # whole slices, and starts at the model's master index.
        viewer = make_viewer(make_volume((8, 8, 24)), make_volume((8, 8, 12)))
        assert viewer.slice_slider.valstep == 1
        assert viewer.slice_slider.valmax == 23
        assert viewer.slice_slider.val == viewer.stack.master_index
        # Two-dimensional images have no slices, so there is no slider.
        flat = make_viewer(np.ones((8, 8)), np.zeros((6, 6)))
        assert flat.slice_slider is None
        assert not flat._slice_slider_ax.get_visible()


# ---------------------------------------------------------------------------
# ROI interaction through synthesized events
# ---------------------------------------------------------------------------

class TestRoi:
    def test_draw_roi(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6), 1),
                             make_volume((32, 32, 6), 2))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        assert viewer.mode is Mode.DRAW_ROI
        motion_data(viewer.fig, ax, 16, 10)
        release_data(viewer.fig, ax, 16, 10)
        assert viewer.mode is Mode.IDLE
        assert all(c is not None for c in viewer.circles)
        for circle in viewer.circles:
            assert circle.center == pytest.approx((10, 10))
            assert circle.get_radius() == pytest.approx(6.0)
        # Force-on-release means stats are current for every volume; expected
        # values use the circle's actual (transform round-tripped) geometry.
        for i, text in enumerate(viewer.stats_texts):
            circle = viewer.circles[i]
            stats = viewer.stack.roi_stats(i, *circle.center,
                                           circle.get_radius())
            assert f"{stats['mean']:.3g}" in text.get_text()

    def test_drag_moves_roi_and_edge_drag_resizes_it(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        # A press inside the circle, off center, moves it by the drag vector.
        press_data(viewer.fig, ax, 11, 10)
        assert viewer.mode is Mode.MOVE_ROI
        motion_data(viewer.fig, ax, 13, 12)
        release_data(viewer.fig, ax, 13, 12)
        assert viewer.circles[0].center == pytest.approx((12, 12))
        assert viewer.circles[0].get_radius() == pytest.approx(5.0)
        # A press on the edge ring resizes to the dragged radius instead.
        press_data(viewer.fig, ax, 17.2, 12)
        assert viewer.mode is Mode.RESIZE_ROI
        motion_data(viewer.fig, ax, 20, 12)
        release_data(viewer.fig, ax, 20, 12)
        assert viewer.circles[0].center == pytest.approx((12, 12))
        assert viewer.circles[0].get_radius() == pytest.approx(8.0)

    def test_stats_survive_none_circle(self, make_viewer):
        # The mbirjax _display_mean missing-continue bug crashed when one
        # circle was None while others were not; this must not.
        viewer = make_viewer(make_volume((32, 32, 6), 1),
                             make_volume((32, 32, 6), 2))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        viewer.circles[1].remove()
        viewer.circles[1] = None
        viewer._display_roi_stats(force=True)  # must not raise


# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------

class TestSliders:
    def test_slice_slider_updates_images(self, make_viewer):
        a = make_volume((16, 16, 24), 1)
        b = make_volume((16, 16, 12), 2)
        viewer = make_viewer(a, b)
        viewer.slice_slider.set_val(23)
        assert viewer.stack.cur_slices == [23, 11]
        np.testing.assert_array_equal(
            np.asarray(viewer.images[0].get_array()), a[:, :, 23])
        np.testing.assert_array_equal(
            np.asarray(viewer.images[1].get_array()), b[:, :, 11])
        assert 'Slice 23' in viewer.axes[0].get_title()


# ---------------------------------------------------------------------------
# Range dialog
# ---------------------------------------------------------------------------

class TestRangeDialog:
    def test_apply_sets_bounds_and_clim(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 4)))
        viewer._open_range_dialog()
        viewer._dialog['widgets']['min'].set_val('-0.25')
        viewer._dialog['widgets']['max'].set_val('0.5')
        click_widget(viewer.fig, viewer._dialog['widgets']['Apply'].ax)
        assert viewer._dialog is None
        assert (viewer.stack.vmin, viewer.stack.vmax) == (-0.25, 0.5)
        assert viewer.intensity_slider.valmin == -0.25
        assert viewer.images[0].get_clim() == (-0.25, 0.5)


# ---------------------------------------------------------------------------
# Zoom/pan sync
# ---------------------------------------------------------------------------

class TestZoomSync:
    def test_limits_propagate_only_while_coupled(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6), 1),
                             make_volume((32, 32, 6), 2))
        viewer.axes[0].set_xlim(5, 20)
        assert viewer.axes[1].get_xlim() == (5, 20)
        viewer.axes[1].set_ylim(25, 3)
        assert viewer.axes[0].get_ylim() == (25, 3)
        viewer.sync_limits = False
        before = viewer.axes[1].get_xlim()
        viewer.axes[0].set_xlim(8, 22)
        assert viewer.axes[1].get_xlim() == before


# ---------------------------------------------------------------------------
# Slice-axis radios and coupling
# ---------------------------------------------------------------------------

class TestAxisControls:
    def test_coupled_radio_changes_all(self, make_viewer):
        a, b = make_volume((8, 10, 12), 1), make_volume((8, 10, 12), 2)
        viewer = make_viewer(a, b)
        viewer.axis_radios[0].set_active(0)
        assert viewer.stack.axes_perms == [[1, 2, 0], [1, 2, 0]]
        np.testing.assert_array_equal(
            np.asarray(viewer.images[0].get_array()),
            viewer.stack.slice_image(0))
        assert viewer.slice_slider.valmax == 7
        # Decoupled, one radio changes only its own volume.
        viewer._toggle_couple_axes()
        assert not viewer.sync_axes and len(viewer.axis_radios) == 2
        viewer.axis_radios[1].set_active(1)
        assert viewer.stack.axes_perms == [[1, 2, 0], [0, 2, 1]]
        # Transposing swaps the two in-plane axes of one volume, and the
        # displayed pixels follow the new permutation.
        viewer._on_transpose_button(0)
        assert viewer.stack.axes_perms == [[2, 1, 0], [0, 2, 1]]
        np.testing.assert_array_equal(
            np.asarray(viewer.images[0].get_array()),
            np.transpose(a, (2, 1, 0))[:, :, viewer.stack.cur_slices[0]])


# ---------------------------------------------------------------------------
# Difference images
# ---------------------------------------------------------------------------

class TestDifference:
    def test_two_volume_difference_applies_immediately(self, make_viewer):
        a, b = make_volume((16, 16, 6), 1), make_volume((16, 16, 6), 2)
        viewer = make_viewer(a, b, slice_label=['A', 'B'])
        viewer._on_difference_button(0)
        assert viewer.stack.is_difference(0)
        assert viewer.stack.labels[0] == 'Image 1 minus current: A'
        np.testing.assert_allclose(viewer.stack.data[0], b - a, rtol=1e-6)
        # The menu now offers Restore for this volume.
        labels = [label for label, _cb in viewer._menu_items(0)]
        assert 'Restore original image' in labels
        assert 'Replace with difference image' not in labels
        viewer._on_restore(0)
        assert not viewer.stack.is_difference(0)
        assert viewer.stack.labels[0] == 'A'
        # The error image is the absolute difference.
        viewer._on_error_button(1)
        assert viewer.stack.labels[1].startswith('abs(Image 0 minus current)')
        np.testing.assert_allclose(viewer.stack.data[1], np.abs(a - b),
                                   rtol=1e-6)


# ---------------------------------------------------------------------------
# Toolbar interaction (zoom-to-rectangle, pan, and the menu)
# ---------------------------------------------------------------------------

class TestToolbarInteraction:
    @staticmethod
    def attach_toolbar(viewer):
        from matplotlib.backend_bases import NavigationToolbar2
        toolbar = NavigationToolbar2(viewer.fig.canvas)
        viewer.fig.canvas.toolbar = toolbar
        return toolbar

    @staticmethod
    def zoom_gesture(viewer, ax, d0, d1, buttons):
        from matplotlib.backend_bases import MouseEvent
        x0, y0 = ax.transData.transform(d0)
        x1, y1 = ax.transData.transform(d1)
        _process(viewer.fig, 'button_press_event', x0, y0, 1)
        for t in (0.3, 0.6, 1.0):
            event = MouseEvent('motion_notify_event', viewer.fig.canvas,
                               x0 + t * (x1 - x0), y0 + t * (y1 - y0),
                               buttons=buttons)
            viewer.fig.canvas.callbacks.process('motion_notify_event', event)
        _process(viewer.fig, 'button_release_event', x1, y1, 1)

    def test_zoom_survives_missing_buttons_state(self, make_viewer,
                                                 monkeypatch):
        # macosx motion events report live hardware state, so a fast drag
        # drains motions with an empty buttons set after release; the
        # viewer's drag_zoom patch must keep the gesture alive.
        viewer = make_viewer(make_volume((64, 64, 40)))
        toolbar = self.attach_toolbar(viewer)
        monkeypatch.setattr(matplotlib, 'get_backend', lambda: 'macosx')
        viewer._patch_toolbar()
        toolbar.zoom()
        before = viewer.axes[0].get_xlim()
        self.zoom_gesture(viewer, viewer.axes[0], (10, 10), (30, 30),
                          buttons=None)  # empty buttons on every motion
        assert viewer.axes[0].get_xlim() != before


# ---------------------------------------------------------------------------
# File load and save
# ---------------------------------------------------------------------------

class TestFileDialogs:
    def test_load_dialog_reads_npy_npz_and_h5(self, make_viewer, tmp_path):
        import h5py
        viewer = make_viewer(make_volume((8, 8, 4)))

        # A .npy path loads directly and resizes the slice slider.
        new = make_volume((6, 6, 10), 5)
        npy_path = str(tmp_path / 'new.npy')
        np.save(npy_path, new)
        viewer._on_load_button(0)
        assert viewer._dialog['kind'] == 'file'
        viewer._dialog['widgets']['path'].set_val(npy_path)
        click_widget(viewer.fig, viewer._dialog['widgets']['Load'].ax)
        assert viewer._dialog is None
        np.testing.assert_array_equal(viewer.stack.original_data[0], new)
        assert viewer.slice_slider.valmax == 9

        # A .npz path offers a chooser of the arrays it holds.
        first, second = make_volume((5, 5, 5), 1), make_volume((7, 7, 7), 2)
        npz_path = str(tmp_path / 'pair.npz')
        np.savez(npz_path, first=first, second=second)
        viewer._on_load_button(0)
        viewer._dialog['widgets']['path'].set_val(npz_path)
        viewer._load_dialog_accept(0)
        assert viewer._dialog['kind'] == 'choice'
        second_button = next(w for k, w in viewer._dialog['widgets'].items()
                             if k.startswith('second'))
        click_widget(viewer.fig, second_button.ax)
        assert viewer._dialog is None
        np.testing.assert_array_equal(viewer.stack.original_data[0], second)

        # An .h5 path brings the dataset's attributes along as the data dict.
        volume = make_volume((5, 6, 7), 3)
        h5_path = str(tmp_path / 'vol.h5')
        with h5py.File(h5_path, 'w') as f:
            dataset = f.create_dataset('volume', data=volume)
            dataset.attrs['notes'] = 'from file'
        viewer._on_load_button(0)
        viewer._dialog['widgets']['path'].set_val(h5_path)
        viewer._load_dialog_accept(0)
        assert viewer._dialog is None
        np.testing.assert_array_equal(viewer.stack.original_data[0], volume)
        assert viewer.stack.data_dicts[0] == {'notes': 'from file'}

    def test_default_save_fn_round_trips(self, tmp_path):
        volume = make_volume((5, 6, 7))
        path = str(tmp_path / 'round.h5')
        _save_data_hdf5(path, volume, 'volume', {'k': 'v'})
        array, data_dict = VolumeStack.read_file_array(path)
        np.testing.assert_allclose(array, volume)
        assert data_dict == {'k': 'v'}


class TestNativeFileDialogs:
    def test_agg_reports_native_unavailable(self, make_viewer, tmp_path):
        # Guarantees the test suite never opens a real OS dialog: on a
        # non-interactive backend the chain must bail out immediately.
        from mbirtorch.viewers.slice_figure import _NATIVE_UNAVAILABLE
        viewer = make_viewer(make_volume((8, 8, 4)))
        result = viewer._native_choose_file('load', str(tmp_path), 'v.h5')
        assert result is _NATIVE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Blitting and snapshots
# ---------------------------------------------------------------------------

class TestRendering:
    def test_partial_redraw_blits_and_falls_back_with_overlay(self,
                                                              make_viewer):
        viewer = make_viewer(make_volume((16, 16, 8), 1),
                             make_volume((16, 16, 8), 2))
        assert viewer.fig.canvas.supports_blit
        assert viewer._renderer_ready
        viewer._partial_redraw()
        viewer._partial_redraw([0], widgets=('slice', 'intensity'))
        viewer._show_message(True, message='overlay')
        viewer._partial_redraw()  # must take the draw_idle path, not crash

    @pytest.mark.parametrize('n_volumes', [1, 3])
    def test_snapshot_contains_image_pixels(self, make_viewer, tmp_path,
                                            n_volumes):
        volumes = [make_volume((32, 32, 8), s) for s in range(n_volumes)]
        viewer = make_viewer(*volumes, title='snapshot test')
        path = str(tmp_path / f'snapshot_{n_volumes}.png')
        viewer.fig.savefig(path)
        assert os.path.getsize(path) > 30_000
        # The panel region must not be blank: sample the image area.
        from matplotlib.image import imread
        pixels = imread(path)
        ax_bbox = viewer.axes[0].bbox
        height = viewer.fig.canvas.get_width_height()[1]
        row = int(height - (ax_bbox.y0 + ax_bbox.y1) / 2)
        col_range = slice(int(ax_bbox.x0), int(ax_bbox.x1))
        panel_row = pixels[row, col_range, :3]
        assert panel_row.std() > 0.05  # random noise, not a flat fill


# ---------------------------------------------------------------------------
# show() and the module-level slice_viewer
# ---------------------------------------------------------------------------

class TestShowAndWrapper:
    def test_nonblocking_viewers_registered_then_adopted(self):
        with pytest.warns(UserWarning, match='non-interactive'):
            first = slice_viewer(make_volume((8, 8, 4)), block=False)
        assert isinstance(first, SliceViewer)
        assert first in viewer_module._NONBLOCKING_VIEWERS
        with pytest.warns(UserWarning, match='non-interactive'):
            second = slice_viewer(make_volume((8, 8, 4)), block=True)
        try:
            assert isinstance(second, SliceViewer)
            assert viewer_module._NONBLOCKING_VIEWERS == []
        finally:
            plt.close(first.fig)
            plt.close(second.fig)
