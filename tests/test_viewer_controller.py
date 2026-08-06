"""Controller tests for the slice viewer, run headlessly on Agg.

Interaction tests synthesize MouseEvent/KeyEvent objects and process them
through fig.canvas.callbacks, the same technique matplotlib's own widget
tests use.  Layout is verified by savefig snapshots.
"""

import os
import time

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

from mbirtorch.viewer import (Mode, SliceViewer, VolumeStack, _save_data_hdf5,
                              slice_viewer)
import mbirtorch.viewer as viewer_module


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


def press_key(fig, key):
    event = KeyEvent('key_press_event', fig.canvas, key)
    fig.canvas.callbacks.process('key_press_event', event)


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

    def test_construction_does_not_show(self, make_viewer):
        # Construction must not enter an event loop or warn; show() is
        # separate (and warns on Agg).
        viewer = make_viewer(make_volume((8, 8, 4)))
        with pytest.warns(UserWarning, match='non-interactive'):
            viewer.show(block=False)

    def test_slice_slider_valstep_and_range(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 24)), make_volume((8, 8, 12)))
        assert viewer.slice_slider.valstep == 1
        assert viewer.slice_slider.valmax == 23
        assert viewer.slice_slider.val == viewer.stack.master_index

    def test_single_slice_slider_hidden(self, make_viewer):
        viewer = make_viewer(np.ones((8, 8)), np.zeros((6, 6)))
        assert viewer.slice_slider is None
        assert not viewer._slice_slider_ax.get_visible()

    def test_coupled_single_radio_when_axes_match(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 4)), make_volume((8, 8, 4)))
        assert viewer.sync_axes and len(viewer.axis_radios) == 1

    def test_decoupled_radios_when_axes_differ(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 4)), make_volume((8, 8, 4)),
                             slice_axis=[0, 2])
        assert not viewer.sync_axes and len(viewer.axis_radios) == 2

    def test_tooltips_constructed_exactly_once(self, make_viewer):
        # The mbirjax viewer built tooltips twice (_draw_images and
        # _connect_events), orphaning the first set; here the annotation
        # exists exactly once per panel.
        from mbirtorch.viewer import TOOLTIP_TEXT
        viewer = make_viewer(make_volume((8, 8, 4)), make_volume((8, 8, 4)))
        assert len(viewer.tooltips) == 2
        for ax in viewer.axes:
            matches = [t for t in ax.texts if t.get_text() == TOOLTIP_TEXT]
            assert len(matches) == 1

    def test_titles_show_per_volume_perms(self, make_viewer):
        # The mbirjax _update_axis wrote axes_perms[0] into every title;
        # each panel must report its own permutation.
        viewer = make_viewer(make_volume((8, 10, 12), 1),
                             make_volume((8, 10, 12), 2))
        viewer._toggle_couple_axes()  # decouple
        viewer.axis_radios[1].set_active(0)
        assert 'Axes: [0 1 2]' in viewer.axes[0].get_title()
        assert 'Axes: [1 2 0]' in viewer.axes[1].get_title()


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

    def test_move_roi(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        press_data(viewer.fig, ax, 11, 10)  # inside, off-center
        assert viewer.mode is Mode.MOVE_ROI
        motion_data(viewer.fig, ax, 13, 12)
        release_data(viewer.fig, ax, 13, 12)
        assert viewer.circles[0].center == pytest.approx((12, 12))
        assert viewer.circles[0].get_radius() == pytest.approx(5.0)

    def test_resize_roi(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        press_data(viewer.fig, ax, 15.2, 10)  # on the edge ring
        assert viewer.mode is Mode.RESIZE_ROI
        motion_data(viewer.fig, ax, 18, 10)
        release_data(viewer.fig, ax, 18, 10)
        assert viewer.circles[0].get_radius() == pytest.approx(8.0)

    def test_tooltip_hover(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        motion_data(viewer.fig, ax, 12, 10)  # inside -> visible
        assert viewer.tooltips[0].get_visible()
        motion_data(viewer.fig, ax, 25, 25)  # far away -> hidden
        assert not viewer.tooltips[0].get_visible()

    def test_escape_clears_roi(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        press_key(viewer.fig, 'escape')
        assert viewer.circles == [None]
        assert viewer.stats_texts == [None]
        assert viewer.mode is Mode.IDLE

    def test_trailing_stats_update_fires(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        # A throttled call schedules the trailing one-shot timer.
        viewer._last_stats_time = time.monotonic()
        viewer.circles[0].center = (12, 12)
        viewer._display_roi_stats()
        assert viewer._stats_timer is not None
        viewer._trailing_stats_fire()
        assert viewer._stats_timer is None
        circle = viewer.circles[0]
        stats = viewer.stack.roi_stats(0, *circle.center,
                                       circle.get_radius())
        assert f"{stats['mean']:.3g}" in viewer.stats_texts[0].get_text()

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

    def test_intensity_slider_sets_clim(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 4)))
        viewer.intensity_slider.set_val((-0.5, 0.75))
        assert viewer.images[0].get_clim() == (-0.5, 0.75)


# ---------------------------------------------------------------------------
# Range dialog
# ---------------------------------------------------------------------------

class TestRangeDialog:
    def test_right_click_intensity_slider_opens(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 4)))
        click_widget(viewer.fig, viewer.intensity_slider.ax, button=3)
        assert viewer._dialog is not None and viewer._dialog['kind'] == 'range'

    def test_range_button_opens(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 4)))
        click_widget(viewer.fig, viewer.range_button.ax)
        assert viewer._dialog is not None and viewer._dialog['kind'] == 'range'

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

    def test_blank_fields_keep_current_bounds(self, make_viewer):
        volume = np.zeros((4, 4, 4))
        volume[0, 0, 0] = 3.0
        viewer = make_viewer(volume, vmin=-1.0, vmax=1.0)
        viewer._open_range_dialog()
        viewer._dialog['widgets']['max'].set_val('2.0')  # min left blank
        viewer._apply_range_dialog()
        assert (viewer.stack.vmin, viewer.stack.vmax) == (-1.0, 2.0)

    def test_data_range_button(self, make_viewer):
        volume = np.zeros((4, 4, 4))
        volume[0, 0, 0] = 3.0
        viewer = make_viewer(volume, vmin=-1.0, vmax=1.0)
        viewer._open_range_dialog()
        click_widget(viewer.fig, viewer._dialog['widgets']['Data range'].ax)
        assert viewer._dialog is None
        assert (viewer.stack.vmin, viewer.stack.vmax) == (0.0, 3.0)
        assert viewer.intensity_slider.valmin == 0.0

    def test_invalid_range_shows_error(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 4)))
        viewer._open_range_dialog()
        viewer._dialog['widgets']['min'].set_val('2')
        viewer._dialog['widgets']['max'].set_val('1')
        viewer._apply_range_dialog()
        assert viewer._dialog is not None
        assert 'Minimum' in viewer._dialog['texts']['error'].get_text()

    def test_non_numeric_shows_error(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 4)))
        viewer._open_range_dialog()
        viewer._dialog['widgets']['min'].set_val('abc')
        viewer._apply_range_dialog()
        assert viewer._dialog is not None
        assert 'numbers' in viewer._dialog['texts']['error'].get_text()

    def test_dialog_blocks_canvas_interaction(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        viewer._open_range_dialog()
        press_data(viewer.fig, viewer.axes[0], 10, 10)
        assert viewer.mode is Mode.IDLE
        assert viewer.circles == [None]
        press_key(viewer.fig, 'escape')  # escape closes the dialog
        assert viewer._dialog is None


# ---------------------------------------------------------------------------
# Zoom/pan sync
# ---------------------------------------------------------------------------

class TestZoomSync:
    def test_limits_propagate_when_coupled(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6), 1),
                             make_volume((32, 32, 6), 2))
        viewer.axes[0].set_xlim(5, 20)
        assert viewer.axes[1].get_xlim() == (5, 20)
        viewer.axes[1].set_ylim(25, 3)
        assert viewer.axes[0].get_ylim() == (25, 3)

    def test_limits_do_not_propagate_when_decoupled(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6), 1),
                             make_volume((32, 32, 6), 2))
        viewer.sync_limits = False
        before = viewer.axes[1].get_xlim()
        viewer.axes[0].set_xlim(5, 20)
        assert viewer.axes[1].get_xlim() == before

    def test_couple_zoom_menu_toggle(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6), 1),
                             make_volume((32, 32, 6), 2))
        labels = [label for label, _cb in viewer._menu_items(0)]
        assert 'Decouple pan/zoom' in labels
        viewer._toggle_couple_zoom()
        assert viewer.sync_limits is False
        labels = [label for label, _cb in viewer._menu_items(0)]
        assert 'Couple pan/zoom' in labels
        viewer._toggle_couple_zoom()
        assert viewer.sync_limits is True


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

    def test_decouple_toggle_then_radio_changes_one(self, make_viewer):
        a, b = make_volume((8, 10, 12), 1), make_volume((8, 10, 12), 2)
        viewer = make_viewer(a, b, slice_axis=[2, 2])
        viewer._toggle_couple_axes()  # decouple
        assert not viewer.sync_axes and len(viewer.axis_radios) == 2
        viewer.axis_radios[1].set_active(1)
        assert viewer.stack.axes_perms == [[0, 1, 2], [0, 2, 1]]

    def test_recoupling_harmonizes_to_volume_zero(self, make_viewer):
        a, b = make_volume((8, 10, 12), 1), make_volume((8, 10, 12), 2)
        viewer = make_viewer(a, b, slice_axis=[2, 0])
        assert not viewer.sync_axes
        viewer._toggle_couple_axes()  # couple
        assert viewer.sync_axes
        assert viewer.stack.slice_axes == [2, 2]
        assert len(viewer.axis_radios) == 1

    def test_transpose_action(self, make_viewer):
        a = make_volume((8, 10, 12))
        viewer = make_viewer(a)
        viewer._on_transpose_button(0)
        assert viewer.stack.axes_perms[0] == [1, 0, 2]
        np.testing.assert_array_equal(
            np.asarray(viewer.images[0].get_array()),
            np.transpose(a, (1, 0, 2))[:, :, viewer.stack.cur_slices[0]])


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

    def test_error_action_uses_abs(self, make_viewer):
        a, b = make_volume((16, 16, 6), 1), make_volume((16, 16, 6), 2)
        viewer = make_viewer(a, b)
        viewer._on_error_button(1)
        assert viewer.stack.labels[1].startswith('abs(Image 0 minus current)')
        np.testing.assert_allclose(viewer.stack.data[1], np.abs(a - b),
                                   rtol=1e-6)

    def test_three_volume_selection_flow(self, make_viewer):
        volumes = [make_volume((16, 16, 6), s) for s in range(3)]
        viewer = make_viewer(*volumes)
        viewer._on_difference_button(0)
        assert viewer.mode is Mode.SELECT_COMPARISON
        assert viewer._message_artist is not None
        press_data(viewer.fig, viewer.axes[2], 8, 8)
        assert viewer.mode is Mode.IDLE
        assert viewer._message_artist is None
        assert viewer.stack.is_difference(0)
        np.testing.assert_allclose(viewer.stack.data[0],
                                   volumes[2] - volumes[0], rtol=1e-6)

    def test_invalid_comparison_keeps_selecting(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 6)),
                             make_volume((16, 16, 6), 1),
                             make_volume((12, 12, 6), 2))
        viewer._on_difference_button(0)
        press_data(viewer.fig, viewer.axes[2], 8, 8)  # wrong shape
        assert viewer.mode is Mode.SELECT_COMPARISON
        assert not viewer.stack.is_difference(0)
        press_key(viewer.fig, 'escape')
        assert viewer.mode is Mode.IDLE
        assert viewer._message_artist is None


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

    def test_zoom_applies_with_sane_motion_events(self, make_viewer):
        # The viewer's own handlers must not interfere with toolbar zoom.
        from matplotlib.backend_bases import MouseButton
        viewer = make_viewer(make_volume((64, 64, 40)))
        toolbar = self.attach_toolbar(viewer)
        toolbar.zoom()
        before = viewer.axes[0].get_xlim()
        self.zoom_gesture(viewer, viewer.axes[0], (10, 10), (30, 30),
                          buttons={MouseButton.LEFT})
        assert viewer.axes[0].get_xlim() != before

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

    def test_right_click_menu_with_toolbar_mode_active(self, make_viewer):
        viewer = make_viewer(make_volume((64, 64, 40)))
        toolbar = self.attach_toolbar(viewer)
        toolbar.zoom()
        press_data(viewer.fig, viewer.axes[0], 32, 32, button=3)
        assert viewer._dialog is not None
        assert viewer._dialog['kind'] == 'menu'

    def test_decoupled_axis_change_leaves_other_views(self, make_viewer):
        # Changing one volume's slice axis must not truncate the other
        # panels' views to the changed volume's extent.
        a = make_volume((64, 64, 40), 1)
        b = make_volume((64, 64, 40), 2)
        viewer = make_viewer(a, b)
        viewer._toggle_couple_axes()  # decouple
        viewer._on_axis_selected(1, 0)  # volume 1 -> depth 64, width 40
        assert viewer.axes[0].get_xlim() == (-0.5, 63.5)
        assert viewer.axes[0].get_ylim() == (63.5, -0.5)
        assert viewer.axes[1].get_xlim() == (-0.5, 39.5)

    def test_difference_selection_with_zoom_mode_armed(self, make_viewer):
        # With the zoom tool toggled on, the click that picks the comparison
        # image must still land.  The toolbar state itself is left alone:
        # toggling it programmatically desynchronizes the macosx backend's
        # native buttons.
        volumes = [make_volume((16, 16, 6), s) for s in range(3)]
        viewer = make_viewer(*volumes)
        toolbar = self.attach_toolbar(viewer)
        toolbar.zoom()
        assert toolbar.mode
        viewer._on_difference_button(0)
        assert toolbar.mode  # tool state untouched
        press_data(viewer.fig, viewer.axes[2], 8, 8)
        assert viewer.stack.is_difference(0)

    def test_home_with_empty_stack_resets_views(self, make_viewer):
        viewer = make_viewer(make_volume((64, 64, 40)))
        toolbar = self.attach_toolbar(viewer)
        viewer._patch_toolbar()
        viewer.axes[0].set_xlim(5, 20)  # zoomed state, nothing pushed
        toolbar.home()
        assert viewer.axes[0].get_xlim() == (-0.5, 63.5)

    def test_axis_change_clears_stale_navigation(self, make_viewer):
        from matplotlib.backend_bases import MouseButton
        viewer = make_viewer(make_volume((64, 64, 40)))
        toolbar = self.attach_toolbar(viewer)
        viewer._patch_toolbar()
        toolbar.zoom()
        self.zoom_gesture(viewer, viewer.axes[0], (10, 10), (30, 30),
                          buttons={MouseButton.LEFT})
        toolbar.zoom()
        viewer.axis_radios[0].set_active(0)  # displayed shape now 64 x 40
        toolbar.home()  # stale saved views were cleared; full new extent
        assert viewer.axes[0].get_xlim() == (-0.5, 39.5)

    def test_selection_click_wins_over_stale_toolbar_mode(self, make_viewer):
        viewer = make_viewer(*[make_volume((16, 16, 6), s) for s in range(3)])
        toolbar = self.attach_toolbar(viewer)
        viewer._on_difference_button(0)
        toolbar.mode = 'zoom rect'  # a tool re-armed mid-selection
        press_data(viewer.fig, viewer.axes[2], 8, 8)
        assert viewer.stack.is_difference(0)

    def test_home_restores_each_volumes_own_view(self, make_viewer):
        # Home restores every panel's own saved view; the sync callback
        # must not let the last panel restored stamp its view on the rest.
        from matplotlib.backend_bases import MouseButton
        a = make_volume((64, 64, 40), 1)
        b = make_volume((64, 64, 40), 2)
        viewer = make_viewer(a, b)
        toolbar = self.attach_toolbar(viewer)
        viewer._patch_toolbar()
        viewer._toggle_couple_axes()
        viewer._on_axis_selected(1, 0)  # panel 1 now 40 wide, others 64
        toolbar.zoom()
        self.zoom_gesture(viewer, viewer.axes[0], (10, 10), (30, 30),
                          buttons={MouseButton.LEFT})
        assert viewer.axes[0].get_xlim() != (-0.5, 63.5)  # zoom applied
        toolbar.home()
        assert viewer.axes[0].get_xlim() == (-0.5, 63.5)
        assert viewer.axes[1].get_xlim() == (-0.5, 39.5)


# ---------------------------------------------------------------------------
# Context menu
# ---------------------------------------------------------------------------

class TestContextMenu:
    def open_menu(self, viewer, i=0):
        h, w = viewer.stack.slice_image(i).shape
        press_data(viewer.fig, viewer.axes[i], w // 2, h // 2, button=3)

    def menu_button(self, viewer, label):
        for key, widget in viewer._dialog['widgets'].items():
            if key.startswith('item') and widget.label.get_text() == label:
                return widget
        raise AssertionError(f'no menu item {label!r}')

    def test_right_click_opens_in_figure_menu(self, make_viewer):
        # On Agg (not TkAgg) the menu is the in-figure popup.
        viewer = make_viewer(make_volume((16, 16, 6), 1),
                             make_volume((16, 16, 6), 2))
        self.open_menu(viewer)
        assert viewer._dialog is not None
        assert viewer._dialog['kind'] == 'menu'
        shown = [w.label.get_text() for k, w in
                 viewer._dialog['widgets'].items() if k.startswith('item')]
        assert shown == [label for label, _cb in viewer._menu_items(0)]

    def test_menu_items_single_volume(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 6)))
        labels = [label for label, _cb in viewer._menu_items(0)]
        assert labels == ['Show data dict', 'Transpose image', 'Load',
                          'Save data to h5', 'Reset', 'Cancel']

    def test_menu_items_multi_volume_labels(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 6), 1),
                             make_volume((16, 16, 6), 2))
        labels = [label for label, _cb in viewer._menu_items(0)]
        assert labels[:4] == ['Decouple slice axes', 'Decouple pan/zoom',
                              'Replace with difference image',
                              'Replace with error image']

    def test_menu_transpose_item_applies_and_closes(self, make_viewer):
        viewer = make_viewer(make_volume((8, 10, 12)))
        self.open_menu(viewer)
        click_widget(viewer.fig,
                     self.menu_button(viewer, 'Transpose image').ax)
        assert viewer._dialog is None
        assert viewer.stack.axes_perms[0] == [1, 0, 2]

    def test_menu_cancel_closes_without_action(self, make_viewer):
        viewer = make_viewer(make_volume((8, 10, 12)))
        self.open_menu(viewer)
        click_widget(viewer.fig, self.menu_button(viewer, 'Cancel').ax)
        assert viewer._dialog is None
        assert viewer.stack.axes_perms[0] == [0, 1, 2]

    def test_outside_click_dismisses_menu(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        self.open_menu(viewer)
        assert viewer._dialog is not None
        press_data(viewer.fig, viewer.axes[0], 30, 30)  # outside the popup
        assert viewer._dialog is None
        assert viewer.circles == [None]  # dismissed without starting an ROI

    def test_no_menu_during_comparison_selection(self, make_viewer):
        viewer = make_viewer(*[make_volume((16, 16, 6), s) for s in range(3)])
        viewer._on_difference_button(0)
        assert viewer.mode is Mode.SELECT_COMPARISON
        self.open_menu(viewer, 2)
        assert viewer._dialog is None

    def test_menu_decouple_axes_item(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 6), 1),
                             make_volume((16, 16, 6), 2))
        self.open_menu(viewer)
        click_widget(viewer.fig,
                     self.menu_button(viewer, 'Decouple slice axes').ax)
        assert viewer._dialog is None
        assert not viewer.sync_axes
        assert len(viewer.axis_radios) == 2

    def test_menu_difference_item_two_volumes(self, make_viewer):
        a, b = make_volume((16, 16, 6), 1), make_volume((16, 16, 6), 2)
        viewer = make_viewer(a, b, slice_label=['A', 'B'])
        self.open_menu(viewer)
        click_widget(viewer.fig, self.menu_button(
            viewer, 'Replace with difference image').ax)
        assert viewer.stack.is_difference(0)
        np.testing.assert_allclose(viewer.stack.data[0], b - a, rtol=1e-6)

    def test_tk_gate_closed_on_agg(self, make_viewer, monkeypatch):
        viewer = make_viewer(make_volume((8, 8, 4)))
        assert viewer._in_process_tk_ok() is False
        monkeypatch.setattr(matplotlib, 'get_backend', lambda: 'TkAgg')
        assert viewer._in_process_tk_ok() is True


# ---------------------------------------------------------------------------
# Data-dict dialogs
# ---------------------------------------------------------------------------

class TestDictDialogs:
    def test_no_dict_message(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._on_dict_button(0)
        assert viewer._dialog['kind'] == 'text'
        assert 'No data dict' in viewer._dialog['texts']['body'].get_text()

    def test_single_entry_goes_straight_to_text(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 4)),
                             data_dicts={'notes': 'hello world'})
        viewer._on_dict_button(0)
        assert viewer._dialog['kind'] == 'text'
        assert 'hello world' in viewer._dialog['texts']['body'].get_text()

    def test_chooser_then_entry_then_back(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 4)),
                             data_dicts={'a': 'alpha', 'b': 'beta'})
        viewer._on_dict_button(0)
        assert viewer._dialog['kind'] == 'choice'
        click_widget(viewer.fig, viewer._dialog['widgets']['b'].ax)
        assert viewer._dialog['kind'] == 'text'
        assert 'beta' in viewer._dialog['texts']['body'].get_text()
        click_widget(viewer.fig, viewer._dialog['widgets']['Back'].ax)
        assert viewer._dialog['kind'] == 'choice'
        click_widget(viewer.fig, viewer._dialog['widgets']['Cancel'].ax)
        assert viewer._dialog is None

    def test_long_text_pages(self, make_viewer):
        body = '\n'.join(f'line {k}' for k in range(75))
        viewer = make_viewer(make_volume((8, 8, 4)),
                             data_dicts={'log': body})
        viewer._on_dict_button(0)
        text = viewer._dialog['texts']['body'].get_text()
        assert 'line 0' in text and 'page 1 of 3' in text
        click_widget(viewer.fig, viewer._dialog['widgets']['Next'].ax)
        text = viewer._dialog['texts']['body'].get_text()
        assert 'line 30' in text and 'page 2 of 3' in text


# ---------------------------------------------------------------------------
# File load and save
# ---------------------------------------------------------------------------

class TestFileDialogs:
    def test_load_npy(self, make_viewer, tmp_path):
        viewer = make_viewer(make_volume((8, 8, 4)))
        new = make_volume((6, 6, 10), 5)
        path = str(tmp_path / 'new.npy')
        np.save(path, new)
        viewer._on_load_button(0)
        assert viewer._dialog['kind'] == 'file'
        viewer._dialog['widgets']['path'].set_val(path)
        click_widget(viewer.fig, viewer._dialog['widgets']['Load'].ax)
        assert viewer._dialog is None
        np.testing.assert_array_equal(viewer.stack.original_data[0], new)
        assert viewer.slice_slider.valmax == 9

    def test_load_missing_file_shows_error(self, make_viewer, tmp_path):
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._on_load_button(0)
        viewer._dialog['widgets']['path'].set_val(str(tmp_path / 'nope.npy'))
        viewer._load_dialog_accept(0)
        assert viewer._dialog is not None
        assert 'File not found' in viewer._dialog['texts']['error'].get_text()

    def test_load_npz_chooser(self, make_viewer, tmp_path):
        viewer = make_viewer(make_volume((8, 8, 4)))
        first, second = make_volume((5, 5, 5), 1), make_volume((7, 7, 7), 2)
        path = str(tmp_path / 'pair.npz')
        np.savez(path, first=first, second=second)
        viewer._on_load_button(0)
        viewer._dialog['widgets']['path'].set_val(path)
        viewer._load_dialog_accept(0)
        assert viewer._dialog['kind'] == 'choice'
        second_button = next(w for k, w in viewer._dialog['widgets'].items()
                             if k.startswith('second'))
        click_widget(viewer.fig, second_button.ax)
        assert viewer._dialog is None
        np.testing.assert_array_equal(viewer.stack.original_data[0], second)

    def test_load_h5_brings_data_dict(self, make_viewer, tmp_path):
        import h5py
        viewer = make_viewer(make_volume((8, 8, 4)))
        volume = make_volume((5, 6, 7), 3)
        path = str(tmp_path / 'vol.h5')
        with h5py.File(path, 'w') as f:
            dataset = f.create_dataset('volume', data=volume)
            dataset.attrs['notes'] = 'from file'
        viewer._on_load_button(0)
        viewer._dialog['widgets']['path'].set_val(path)
        viewer._load_dialog_accept(0)
        assert viewer._dialog is None
        assert viewer.stack.data_dicts[0] == {'notes': 'from file'}

    def test_save_uses_injected_fn_and_appends_suffix(self, make_viewer,
                                                      tmp_path):
        calls = []

        def recorder(path, array, name, attrs):
            calls.append((path, array, name, attrs))

        viewer = make_viewer(make_volume((8, 8, 4)),
                             data_dicts={'notes': 'n'}, save_fn=recorder)
        viewer._on_save_button(0)
        viewer._dialog['widgets']['path'].set_val(str(tmp_path / 'out'))
        click_widget(viewer.fig, viewer._dialog['widgets']['Save'].ax)
        assert viewer._dialog is None
        (path, array, name, attrs), = calls
        assert path.endswith('out.h5') and name == 'volume'
        np.testing.assert_array_equal(array, viewer.stack.original_data[0])
        assert attrs == {'notes': 'n'}
        assert 'Saved to' in viewer._message_artist.get_text()

    def test_default_save_fn_round_trips(self, tmp_path):
        volume = make_volume((5, 6, 7))
        path = str(tmp_path / 'round.h5')
        _save_data_hdf5(path, volume, 'volume', {'k': 'v'})
        array, data_dict = VolumeStack.read_file_array(path)
        np.testing.assert_allclose(array, volume)
        assert data_dict == {'k': 'v'}

    def test_save_error_shows_in_dialog(self, make_viewer, tmp_path):
        def failing(path, array, name, attrs):
            raise OSError('disk full')

        viewer = make_viewer(make_volume((8, 8, 4)), save_fn=failing)
        viewer._on_save_button(0)
        viewer._dialog['widgets']['path'].set_val(str(tmp_path / 'x.h5'))
        viewer._save_dialog_accept(0)
        assert viewer._dialog is not None
        assert 'disk full' in viewer._dialog['texts']['error'].get_text()


class TestFileBrowser:
    def _entry_button(self, viewer, label):
        shown = viewer._dialog['state']['entries_shown']
        for j, (entry_label, _full, _is_dir) in enumerate(shown):
            if entry_label == label:
                return viewer._dialog['widgets'][f'entry{j}']
        raise AssertionError(
            f'no entry {label!r} in {[e[0] for e in shown]}')

    def test_listing_shows_dirs_and_matching_files(self, make_viewer,
                                                   tmp_path):
        (tmp_path / 'sub').mkdir()
        np.save(str(tmp_path / 'vol.npy'), make_volume((4, 4, 4)))
        (tmp_path / 'noise.txt').write_text('not a volume')
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._open_file_dialog(0, 'load', directory=str(tmp_path))
        labels = [e[0] for e in viewer._dialog['state']['entries_shown']]
        assert '[..]' in labels
        assert 'sub' + os.sep in labels
        assert 'vol.npy' in labels
        assert 'noise.txt' not in labels

    def test_directory_click_navigates(self, make_viewer, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        np.save(str(sub / 'inner.npy'), make_volume((4, 4, 4)))
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._open_file_dialog(0, 'load', directory=str(tmp_path))
        click_widget(viewer.fig,
                     self._entry_button(viewer, 'sub' + os.sep).ax)
        state = viewer._dialog['state']
        assert state['directory'] == str(sub)
        assert viewer._dialog['widgets']['path'].text.startswith(str(sub))
        assert viewer._last_dir == str(sub)
        labels = [e[0] for e in state['entries_shown']]
        assert 'inner.npy' in labels

    def test_file_click_loads_immediately(self, make_viewer, tmp_path):
        new = make_volume((6, 6, 6), 7)
        np.save(str(tmp_path / 'pick.npy'), new)
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._open_file_dialog(0, 'load', directory=str(tmp_path))
        click_widget(viewer.fig, self._entry_button(viewer, 'pick.npy').ax)
        assert viewer._dialog is None
        np.testing.assert_array_equal(viewer.stack.original_data[0], new)

    def test_typed_directory_navigates_on_load(self, make_viewer, tmp_path):
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._on_load_button(0)
        viewer._dialog['widgets']['path'].set_val(str(tmp_path))
        viewer._load_dialog_accept(0)
        assert viewer._dialog is not None
        assert viewer._dialog['state']['directory'] == str(tmp_path)

    def test_pagination(self, make_viewer, tmp_path):
        for k in range(20):
            np.save(str(tmp_path / f'v{k:02d}.npy'), make_volume((2, 2, 2)))
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._open_file_dialog(0, 'load', directory=str(tmp_path))
        state = viewer._dialog['state']
        first_page = [e[0] for e in state['entries_shown']]
        assert 'page 1' in viewer._dialog['texts']['page'].get_text()
        click_widget(viewer.fig, viewer._dialog['widgets']['Next'].ax)
        second_page = [e[0] for e in viewer._dialog['state']['entries_shown']]
        assert second_page != first_page
        assert 'page 2' in viewer._dialog['texts']['page'].get_text()

    def test_save_navigation_keeps_filename(self, make_viewer, tmp_path):
        sub = tmp_path / 'results'
        sub.mkdir()
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._open_file_dialog(0, 'save', directory=str(tmp_path))
        viewer._dialog['widgets']['path'].set_val(
            os.path.join(str(tmp_path), 'myvol.h5'))
        click_widget(viewer.fig,
                     self._entry_button(viewer, 'results' + os.sep).ax)
        assert viewer._dialog['widgets']['path'].text == \
            os.path.join(str(sub), 'myvol.h5')

    def test_save_rejects_directory_path(self, make_viewer, tmp_path):
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._open_file_dialog(0, 'save', directory=str(tmp_path))
        viewer._dialog['widgets']['path'].set_val(str(tmp_path))
        viewer._save_dialog_accept(0)
        assert viewer._dialog is not None
        assert 'directory' in viewer._dialog['texts']['error'].get_text()

    def test_up_navigation_lands_on_page_of_child(self, make_viewer,
                                                  tmp_path):
        # Fifteen sibling directories push 'sub_last' onto page two; going
        # up from inside it must reopen page two, not page one.
        for k in range(15):
            (tmp_path / f'd{k:02d}').mkdir()
        sub = tmp_path / 'sub_last'
        sub.mkdir()
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._open_file_dialog(0, 'load', directory=str(sub))
        click_widget(viewer.fig, self._entry_button(viewer, '[..]').ax)
        state = viewer._dialog['state']
        assert state['directory'] == str(tmp_path)
        assert state['page'] == 1
        labels = [e[0] for e in state['entries_shown']]
        assert 'sub_last' + os.sep in labels


class TestNativeFileDialogs:
    def test_agg_reports_native_unavailable(self, make_viewer, tmp_path):
        # Guarantees the test suite never opens a real OS dialog: on a
        # non-interactive backend the chain must bail out immediately.
        from mbirtorch.viewer import _NATIVE_UNAVAILABLE
        viewer = make_viewer(make_volume((8, 8, 4)))
        result = viewer._native_choose_file('load', str(tmp_path), 'v.h5')
        assert result is _NATIVE_UNAVAILABLE

    def test_unavailable_falls_back_to_browser(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 4)))
        viewer._file_action(0, 'load')
        assert viewer._dialog is not None
        assert viewer._dialog['kind'] == 'file'

    def test_native_choice_loads_directly(self, make_viewer, tmp_path,
                                          monkeypatch):
        new = make_volume((6, 6, 6), 3)
        path = str(tmp_path / 'native.npy')
        np.save(path, new)
        viewer = make_viewer(make_volume((8, 8, 4)))
        monkeypatch.setattr(viewer, '_native_choose_file',
                            lambda mode, d, f: path)
        viewer._file_action(0, 'load')
        assert viewer._dialog is None
        np.testing.assert_array_equal(viewer.stack.original_data[0], new)
        assert viewer._last_dir == str(tmp_path)

    def test_native_cancel_does_nothing(self, make_viewer, monkeypatch):
        viewer = make_viewer(make_volume((8, 8, 4)))
        before = viewer.stack.original_data[0]
        monkeypatch.setattr(viewer, '_native_choose_file',
                            lambda mode, d, f: None)
        viewer._file_action(0, 'load')
        assert viewer._dialog is None
        assert viewer.stack.original_data[0] is before

    def test_native_save_appends_suffix_and_saves(self, make_viewer,
                                                  tmp_path, monkeypatch):
        calls = []
        viewer = make_viewer(
            make_volume((8, 8, 4)),
            save_fn=lambda *args: calls.append(args))
        monkeypatch.setattr(viewer, '_native_choose_file',
                            lambda mode, d, f: str(tmp_path / 'out'))
        viewer._file_action(0, 'save')
        (path, _array, name, _attrs), = calls
        assert path.endswith('out.h5') and name == 'volume'
        assert 'Saved to' in viewer._message_artist.get_text()

    def test_native_npz_still_uses_array_chooser(self, make_viewer,
                                                 tmp_path, monkeypatch):
        path = str(tmp_path / 'pair.npz')
        np.savez(path, one=make_volume((4, 4, 4), 1),
                 two=make_volume((5, 5, 5), 2))
        viewer = make_viewer(make_volume((8, 8, 4)))
        monkeypatch.setattr(viewer, '_native_choose_file',
                            lambda mode, d, f: path)
        viewer._file_action(0, 'load')
        assert viewer._dialog is not None
        assert viewer._dialog['kind'] == 'choice'


# ---------------------------------------------------------------------------
# Reset, help, escape
# ---------------------------------------------------------------------------

class TestResetAndHelp:
    def test_reset_restores_view_and_clears_roi(self, make_viewer):
        viewer = make_viewer(make_volume((32, 32, 6)))
        ax = viewer.axes[0]
        press_data(viewer.fig, ax, 10, 10)
        motion_data(viewer.fig, ax, 15, 10)
        release_data(viewer.fig, ax, 15, 10)
        ax.set_xlim(5, 12)
        viewer._on_reset_button(0)
        assert ax.get_xlim() == (-0.5, 31.5)
        assert ax.get_ylim() == (31.5, -0.5)
        assert viewer.circles == [None]

    def test_help_overlay_shows_and_escape_hides(self, make_viewer):
        viewer = make_viewer(make_volume((8, 8, 4)))
        press_key(viewer.fig, 'h')
        assert viewer._message_artist is not None
        assert 'ROI' in viewer._message_artist.get_text()
        press_key(viewer.fig, 'escape')
        assert viewer._message_artist is None


# ---------------------------------------------------------------------------
# Blitting and snapshots
# ---------------------------------------------------------------------------

class TestRendering:
    def test_partial_redraw_runs_on_agg(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 8), 1),
                             make_volume((16, 16, 8), 2))
        assert viewer.fig.canvas.supports_blit
        assert viewer._renderer_ready
        viewer._partial_redraw()
        viewer._partial_redraw([0], widgets=('slice', 'intensity'))

    def test_partial_redraw_falls_back_with_overlay(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 8)))
        viewer._show_message(True, message='overlay')
        viewer._partial_redraw()  # must take the draw_idle path, not crash

    def test_partial_redraw_tracks_last_regions(self, make_viewer):
        viewer = make_viewer(make_volume((16, 16, 8)))
        viewer._partial_redraw([0], widgets=('intensity',))
        assert ('volume', 0) in viewer._last_blit_regions
        assert ('widget', 'intensity') in viewer._last_blit_regions

    def test_non_whitelisted_backend_uses_draw_idle(self, make_viewer,
                                                    monkeypatch):
        # macosx reports blit support but mishandles partial regions on
        # HiDPI; the viewer must fall back to full redraws there.
        viewer = make_viewer(make_volume((16, 16, 8)))
        monkeypatch.setattr(matplotlib, 'get_backend', lambda: 'macosx')
        calls = []
        monkeypatch.setattr(viewer.fig.canvas, 'draw_idle',
                            lambda: calls.append(1))
        blits = []
        monkeypatch.setattr(viewer.fig.canvas, 'blit',
                            lambda *a, **k: blits.append(1))
        viewer._partial_redraw()
        assert calls and not blits

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
    def test_slice_viewer_returns_viewer_nonblocking(self):
        with pytest.warns(UserWarning, match='non-interactive'):
            viewer = slice_viewer(make_volume((8, 8, 4)), block=False)
        try:
            assert isinstance(viewer, SliceViewer)
            assert viewer in viewer_module._NONBLOCKING_VIEWERS
        finally:
            plt.close(viewer.fig)

    def test_blocking_call_adopts_nonblocking_viewers(self):
        with pytest.warns(UserWarning, match='non-interactive'):
            first = slice_viewer(make_volume((8, 8, 4)), block=False)
        with pytest.warns(UserWarning, match='non-interactive'):
            second = slice_viewer(make_volume((8, 8, 4)), block=True)
        try:
            assert isinstance(second, SliceViewer)
            assert viewer_module._NONBLOCKING_VIEWERS == []
        finally:
            plt.close(first.fig)
            plt.close(second.fig)
