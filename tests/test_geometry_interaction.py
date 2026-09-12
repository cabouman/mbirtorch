"""Headless tests for the widgets, the zoom, and the comparison overlay.

These are tests of `geometry_figure.py`.  The tests check five things.  The
view slider exists with the range the scan has, and moving it changes what is
drawn.  The trajectory toggle draws one polyline per panel and not one marker
per view.  The zoom toggle puts a cube around the volume on the 3D panel.  A
comparison whose channel offset is ten channels larger draws its projected
volume outline ten channels over from the primary's and lists the change in
the text panel.  Removing the comparison removes its artists.

A comparison also opens a second figure, which tables every difference between
the two geometries.  The tests check what that table holds, that removing the
comparison closes the window, and that saving the figure writes the window
beside it.  The text panel keeps the parameters that differ and counts the
derived quantities.

A later group covers the two data overlays.  Passing None removes an overlay
and leaves the view change working.  The sinogram's image is a moving artist,
animated exactly where the other moving artists are, and the silhouette's two
images are static.  The phantom's outline on the detector face is a moving
artist as well, and a view change puts it somewhere else.  All of them come
through a comparison being added and removed.  The last of them measures the
slider-step limit with a sinogram drawn: a step on the probe's 1800-view
timing model must stay under 100 ms.

A later group covers the three overlay toggles.  Each toggle hides the artists
of its own overlay and no others, and the state of each toggle follows its
``set_show_`` method and a click on the widget.  A toggle whose overlay is
absent is built all the same and does nothing.  Hiding the sinogram takes its
note out of the detector panel's title, and hiding the comparison leaves its
numbers in the text panel.  The last of them renders the figure: a sinogram
hidden and then stepped past twice must leave the detector panel's own
background on the screen, which is what the partial-redraw path does with an
artist that is animated and invisible at once.

Later tests cover the labels and the angle-0 reference.  The labels on the
source, the detector, and the pixel-0 marker exist in the panels that carry
them, they follow the source when the view changes, and they are animated
exactly when the other moving artists are.  The angle-0 reference is drawn in
the 3D view and the top view, it does not move with the view, it is never
animated, its toggle hides and shows it, and the panel limits hold it whether
or not it is drawn.

The tests run under the Agg backend and open no window.
"""

import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # the tests draw into a buffer and open no window

import geometry_probe as probe
from mbirtorch.geometry_scene import GeometryScene, required_parameter_names
import mbirtorch.geometry_figure as geometry_figure
from mbirtorch.geometry_figure import (COLORS, GeometryFigure,
                                       VOLUME_BOX_EDGES,
                                       ZOOM_VOLUME_WIDTH_FACTOR,
                                       TOP_PANEL_COLUMNS, SIDE_PANEL_COLUMNS)

#: The object coordinates the top and side panels put on their two axes, as
#: lists for indexing.  The top view is the xy plane seen from -z, so it draws
#: y across the screen and x down it; see the display convention in
#: `geometry_figure`.
TOP_COLUMNS = list(TOP_PANEL_COLUMNS)
SIDE_COLUMNS = list(SIDE_PANEL_COLUMNS)

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

#: How many channels the comparison's detector offset is moved by.  Ten
#: channels is the size the comparison test uses.
COMPARISON_CHANNEL_SHIFT = 10

#: The slider-step gate, in milliseconds, and how the step is timed: ten steps
#: spread over the scan so that each one changes every drawn position.  The
#: limit is measured with a sinogram drawn by
#: ``test_a_slider_step_with_a_sinogram_stays_under_the_gate``.
STEP_GATE_MS = 100.0
TIMED_STEPS = 10
TIMED_STEP_STRIDE = 89


def build_figure(name='cone flat', **kwargs):
    """Build the scene and a figure for one probe configuration."""
    cfg = CONFIGS_BY_NAME[name]
    scene = GeometryScene.from_model(probe.build_model(cfg))
    return scene, GeometryFigure(scene, **kwargs)


def close(figure):
    """Close a figure's matplotlib figures so that the tests do not pile up.

    A comparison opens a second figure, which is closed here as well.
    """
    import matplotlib.pyplot as plt
    if figure.compare_figure is not None:
        plt.close(figure.compare_figure)
    plt.close(figure.figure)


def lines_of_color(axes, color):
    """The visible lines of one color in a panel, in drawing order."""
    return [line for line in axes.get_lines()
            if line.get_color() == color and line.get_visible()]


def polylines_of_color(axes, color):
    """The visible polylines of one color, leaving out the point markers.

    A marker is a line artist too, so a panel that marks a point in the
    comparison color holds more lines of that color than it draws polylines.
    """
    return [line for line in lines_of_color(axes, color)
            if line.get_marker() in ('None', '', None)]


def finite_points(line):
    """One line's data as an (N, 2) array, with the NaN separators dropped."""
    points = np.stack([np.asarray(line.get_xdata(), dtype=np.float64),
                       np.asarray(line.get_ydata(), dtype=np.float64)],
                      axis=1)
    return points[np.isfinite(points).all(axis=1)]


# ── the view slider ─────────────────────────────────────────────────────────

def test_slider_has_the_range_of_the_scan():
    """The slider covers every view and steps by one whole view."""
    scene, figure = build_figure()
    try:
        slider = figure.view_slider
        assert slider is not None
        assert slider.valmin == 0
        assert slider.valmax == scene.num_views - 1
        assert slider.valstep == 1
        assert int(round(float(slider.val))) == figure.view_index
    finally:
        close(figure)


def test_moving_the_slider_changes_the_drawn_view():
    """Setting the slider's value redraws another view.

    The check reads the drawn data and not only the view index, because a
    slider that moved the index without redrawing would leave the figure
    showing the old view.
    """
    scene, figure = build_figure(view_index=0)
    try:
        before = finite_points(figure._top['detector']).copy()
        figure.view_slider.set_val(3)
        assert figure.view_index == 3
        after = finite_points(figure._top['detector'])
        assert not np.allclose(before, after)
        # The drawn outline is the scene's outline for the new view.
        expected = scene.view(3).detector_outline
        assert np.allclose(after, expected[:, TOP_COLUMNS])
        # The title names the view drawn.
        assert 'view 3' in figure.ax_detector.get_title()
    finally:
        close(figure)


def test_a_single_view_scan_hides_the_slider():
    """One view has nothing to slide, so no slider is built."""
    cfg = CONFIGS_BY_NAME['cone flat']
    scene = GeometryScene.from_model(probe.build_model(cfg))
    first_row = np.asarray(scene.params['view_params_array'],
                           dtype=np.float64)[:1]
    single = scene.with_parameters(
        dict(sinogram_shape=(1, scene.num_det_rows, scene.num_det_channels),
             view_params_array=first_row))
    figure = GeometryFigure(single)
    try:
        assert figure.view_slider is None
        assert figure._slider_axes.get_visible() is False
    finally:
        close(figure)


# ── the trajectory toggle ───────────────────────────────────────────────────

def test_trajectory_is_one_polyline_per_panel():
    """The source path is one line per panel, whatever the number of views.

    A path drawn as one marker per view would cost 1800 artists on a helical
    scan, which is what this test rules out.  The test therefore counts
    the lines and checks that they carry no marker.
    """
    scene, figure = build_figure('cone helical', show_trajectory=True)
    try:
        assert figure.show_trajectory is True
        for axes in (figure.ax_top, figure.ax_side):
            drawn = lines_of_color(axes, COLORS['trajectory'])
            assert len(drawn) == 1, axes.get_title()
            line = drawn[0]
            assert line.get_marker() in ('None', '', None)
            assert len(line.get_xdata()) == scene.num_views
        # The 3D panel draws it too, as one 3D line.
        path_3d = [line for line in figure.ax_3d.get_lines()
                   if line.get_color() == COLORS['trajectory']
                   and line.get_visible()]
        assert len(path_3d) == 1
        assert len(path_3d[0].get_data_3d()[0]) == scene.num_views
    finally:
        close(figure)


def test_the_trajectory_toggle_turns_the_path_on_and_off():
    """The toggle widget and set_show_trajectory agree on the state."""
    _, figure = build_figure('cone helical')
    try:
        assert figure.show_trajectory is False
        assert not lines_of_color(figure.ax_top, COLORS['trajectory'])
        assert figure.trajectory_check.get_status()[0] is False

        # Clicking the widget calls its callback with the label.
        figure.trajectory_check.set_active(0)
        assert figure.show_trajectory is True
        assert len(lines_of_color(figure.ax_top, COLORS['trajectory'])) == 1

        figure.set_show_trajectory(False)
        assert figure.trajectory_check.get_status()[0] is False
        assert not lines_of_color(figure.ax_top, COLORS['trajectory'])
    finally:
        close(figure)


# ── the 3D zoom ─────────────────────────────────────────────────────────────

def test_zoom_to_volume_puts_a_cube_around_the_volume():
    """The volume zoom is a cube on the volume, smaller than the scan cube."""
    scene, figure = build_figure()
    try:
        assert figure.zoom == 'scan'
        scan_width = float(np.ptp(figure.ax_3d.get_xlim()))

        figure.set_zoom('volume')
        assert figure.zoom == 'volume'
        limits = np.array([figure.ax_3d.get_xlim(), figure.ax_3d.get_ylim(),
                           figure.ax_3d.get_zlim()])
        widths = limits[:, 1] - limits[:, 0]
        # A cube: the three axes have one width.
        assert np.allclose(widths, widths[0])
        # About three volume extents wide.
        corners = scene.volume_corners()
        extent = float(np.max(corners.max(axis=0) - corners.min(axis=0)))
        assert widths[0] == pytest.approx(ZOOM_VOLUME_WIDTH_FACTOR * extent)
        # It holds the whole volume box, and it is closer in than the scan.
        for axis in range(3):
            assert limits[axis, 0] < corners[:, axis].min()
            assert limits[axis, 1] > corners[:, axis].max()
        assert widths[0] < scan_width

        figure.set_zoom('scan')
        assert float(np.ptp(figure.ax_3d.get_xlim())) == pytest.approx(
            scan_width)
    finally:
        close(figure)


def test_the_zoom_toggle_switches_the_two_states():
    """The toggle widget and set_zoom agree on the state."""
    _, figure = build_figure()
    try:
        assert figure.zoom_check.get_status()[0] is False
        figure.zoom_check.set_active(0)
        assert figure.zoom == 'volume'
        figure.set_zoom('scan')
        assert figure.zoom_check.get_status()[0] is False
        with pytest.raises(ValueError):
            figure.set_zoom('everything')
    finally:
        close(figure)


# ── the comparison overlay ──────────────────────────────────────────────────

def compare_overrides(scene, channels=COMPARISON_CHANNEL_SHIFT):
    """The override dictionary that moves the detector by whole channels."""
    return dict(det_channel_offset=(scene.det_channel_offset
                                    + channels * scene.delta_det_channel))


def test_comparison_outline_is_shifted_by_ten_channels():
    """Ten channels of offset move the comparison outline ten channels.

    The primary and the comparison differ in `det_channel_offset` alone, so
    their projected volume outlines have the same shape and the comparison's
    sits ten channels along the channel axis.  The check reads the two lines'
    data out of the detector-face panel.
    """
    scene, figure = build_figure(view_index=2)
    try:
        figure.set_compare(compare_overrides(scene))
        primary = polylines_of_color(figure.ax_detector, COLORS['volume'])
        comparison = polylines_of_color(figure.ax_detector,
                                        COLORS['compare'])
        assert len(primary) == 1 and len(comparison) == 1

        first = finite_points(primary[0])
        second = finite_points(comparison[0])
        assert first.shape == (2 * len(VOLUME_BOX_EDGES), 2)
        assert second.shape == first.shape
        # The panel's axes are (channel, row).
        assert np.allclose(second[:, 0] - first[:, 0],
                           COMPARISON_CHANNEL_SHIFT)
        assert np.allclose(second[:, 1], first[:, 1])
    finally:
        close(figure)


def test_comparison_is_listed_in_the_text_panel():
    """The text panel names the changed parameter with both values."""
    scene, figure = build_figure(view_index=2)
    try:
        overrides = compare_overrides(scene)
        figure.set_compare(overrides)
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'Comparison' in body
        assert 'det_channel_offset' in body
        line = [text for text in body.splitlines()
                if 'det_channel_offset' in text][0]
        assert '->' in line
        assert f'{scene.det_channel_offset:.3g}' in line
        assert f'{overrides["det_channel_offset"]:.3g}' in line
    finally:
        close(figure)


def test_comparison_follows_the_slider():
    """The comparison is drawn for the view the slider is on."""
    scene, figure = build_figure(view_index=0)
    try:
        figure.set_compare(compare_overrides(scene))
        figure.view_slider.set_val(4)
        assert figure.view_index == 4
        drawn = finite_points(figure._compare['detector_top'])
        expected = figure.compare_scene.view(4).detector_outline
        assert np.allclose(drawn, expected[:, TOP_COLUMNS])
    finally:
        close(figure)


def test_comparison_accepts_a_scene_and_a_model():
    """A comparison can be a scene, a model, or a dictionary of overrides."""
    scene, figure = build_figure()
    try:
        other = scene.with_parameters(compare_overrides(scene))
        figure.set_compare(other)
        assert figure.compare_scene is other

        model = probe.build_model(CONFIGS_BY_NAME['parallel'])
        figure.set_compare(model)
        assert figure.compare_scene.kind == 'parallel'

        figure.set_compare(compare_overrides(scene))
        assert figure.compare_scene.kind == scene.kind
    finally:
        close(figure)


def test_removing_the_comparison_removes_its_artists():
    """set_compare(None) leaves no comparison artist behind."""
    scene, figure = build_figure()
    try:
        figure.set_compare(compare_overrides(scene))
        assert figure.compare_scene is not None
        for axes in (figure.ax_3d, figure.ax_top, figure.ax_side,
                     figure.ax_detector):
            assert lines_of_color(axes, COLORS['compare'])

        figure.set_compare(None)
        assert figure.compare_scene is None
        for axes in (figure.ax_3d, figure.ax_top, figure.ax_side,
                     figure.ax_detector):
            assert not [line for line in axes.get_lines()
                        if line.get_color() == COLORS['compare']]
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'Comparison' not in body
        # The figure still redraws with no comparison.
        figure.set_view(1)
    finally:
        close(figure)


def test_a_comparison_at_construction_is_drawn():
    """The compare argument of the constructor draws the same overlay."""
    cfg = CONFIGS_BY_NAME['cone flat']
    scene = GeometryScene.from_model(probe.build_model(cfg))
    figure = GeometryFigure(scene, view_index=2,
                            compare=compare_overrides(scene))
    try:
        assert figure.compare_scene is not None
        assert lines_of_color(figure.ax_detector, COLORS['compare'])
    finally:
        close(figure)


# ── the comparison window ───────────────────────────────────────────────────

def compare_window_lines(figure):
    """The rows of the comparison window's table, in the order they are drawn.

    Each row is one text artist on the window's single axes, and the artists
    are created from the top down, so their order is the table's order.
    """
    return [artist.get_text()
            for artist in figure.compare_figure.axes[0].texts]


def test_the_comparison_window_tables_every_difference():
    """The window lists every difference, parameters before derived.

    The text panel can hold only a few lines, so the whole comparison lives in
    a window of its own.  This checks the three columns of the changed
    parameter and then checks the table against what the scene reports: one row
    per difference, in the order the scene gives them, with the parameters
    above the rule and the derived quantities below it.
    """
    scene, figure = build_figure(view_index=2)
    try:
        overrides = compare_overrides(scene)
        figure.set_compare(overrides)
        assert figure.compare_figure is not None
        lines = compare_window_lines(figure)

        assert 'primary (solid)' in lines[0]
        assert 'comparison (dashed)' in lines[0]
        changed = [line for line in lines if 'det_channel_offset' in line]
        assert len(changed) == 1
        assert f'{scene.det_channel_offset:.3g}' in changed[0]
        assert f'{overrides["det_channel_offset"]:.3g}' in changed[0]

        body = lines[1:]
        rule = [index for index, line in enumerate(body)
                if line.startswith('--')]
        assert len(rule) == 1, 'one rule separates the two kinds of entry'
        names = [line.split()[0] for line in body
                 if not line.startswith('--')]
        rows = scene.differences(figure.compare_scene)
        assert names == [name for name, _, _ in rows]
        parameters = set(required_parameter_names(scene.kind))
        assert all(line.split()[0] in parameters for line in body[:rule[0]])
        assert not any(line.split()[0] in parameters
                       for line in body[rule[0] + 1:])
    finally:
        close(figure)


def test_removing_the_comparison_closes_its_window():
    """set_compare(None) closes the window, and a new comparison opens one."""
    import matplotlib.pyplot as plt
    scene, figure = build_figure()
    try:
        figure.set_compare(compare_overrides(scene))
        first = figure.compare_figure
        assert plt.fignum_exists(first.number)

        figure.set_compare(None)
        assert figure.compare_figure is None
        assert not plt.fignum_exists(first.number)

        figure.set_compare(compare_overrides(scene, channels=4))
        assert figure.compare_figure is not None
        assert figure.compare_figure is not first
        assert plt.fignum_exists(figure.compare_figure.number)
    finally:
        close(figure)


def test_save_writes_the_comparison_window_beside_the_figure(tmp_path):
    """A save with a comparison drawn writes a second file.

    The second file carries the same name with ``_comparison`` before the
    extension, so the two images stay together.  A save with no comparison
    writes nothing beside the figure.
    """
    scene, figure = build_figure(view_index=1)
    try:
        path = tmp_path / 'scan.png'
        companion = tmp_path / 'scan_comparison.png'
        assert figure.save(str(path), dpi=80) == str(path)
        assert not companion.exists()

        figure.set_compare(compare_overrides(scene))
        assert figure.save(str(path), dpi=80) == str(path)
        assert companion.stat().st_size > 1024
    finally:
        close(figure)


def test_the_text_panel_keeps_the_parameters_and_counts_the_derived():
    """The panel lists the changed parameter and counts what it moved.

    The derived quantities are named in the window instead, so the panel names
    none of them.  This is what makes room for the parameter line at the
    smallest font the panel uses.
    """
    scene, figure = build_figure(view_index=2)
    try:
        figure.set_compare(compare_overrides(scene))
        block = ' '.join(figure._compare_text.get_text().split())
        parameters, derived = figure._difference_groups()
        assert [name for name, _, _ in parameters] == ['det_channel_offset']
        assert derived, 'a channel offset should move a derived quantity'
        assert 'det_channel_offset: ' in block
        assert f'{len(derived)} derived quantities differ' in block
        assert 'see the comparison window' in block
        for name, _, _ in derived:
            assert name not in block, f'{name} belongs in the window'
    finally:
        close(figure)


# ── construction never opens a window ───────────────────────────────────────

def test_construction_does_not_call_show(monkeypatch):
    """Building, redrawing, and saving a figure never calls show.

    The slice viewer keeps window handling out of construction for the same
    reason: a class that opened a window could not be used in a script or a
    test.
    """
    _, figure = build_figure()
    close(figure)
    import matplotlib.pyplot as plt
    calls = []
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: calls.append(1))

    scene, figure = build_figure(view_index=1)
    try:
        figure.set_view(2)
        figure.set_show_trajectory(True)
        figure.set_zoom('volume')
        figure.set_compare(compare_overrides(scene))
        assert calls == []
    finally:
        close(figure)


def test_show_prints_and_returns_under_a_backend_with_no_window(capsys):
    """Under Agg, show says there is no window and returns."""
    _, figure = build_figure()
    try:
        figure.show()
        printed = capsys.readouterr().out
        assert 'no window' in printed
    finally:
        close(figure)


# ── the partial redraw draws the same picture as a full repaint ─────────────

def test_the_partial_redraw_matches_a_full_repaint(tmp_path):
    """A view reached by stepping looks like the same view drawn from new.

    The partial redraw restores a cached background and draws the artists that
    moved.  A stale background or a missing artist would show up here as two
    different images.  The widget row is left out of the comparison, because a
    slider draws a mark at the value it was built with and the two figures were
    built at different views.
    """
    scene, stepped = build_figure(view_index=0)
    _, direct = build_figure(view_index=4)
    try:
        stepped.set_view(4)
        first = str(tmp_path / 'stepped.png')
        second = str(tmp_path / 'direct.png')
        stepped.save(first, dpi=80)
        direct.save(second, dpi=80)
        import matplotlib.image as mpimg
        left = mpimg.imread(first)
        right = mpimg.imread(second)
        assert left.shape == right.shape
        panels = slice(0, int(0.86 * left.shape[0]))
        assert float(np.abs(left[panels] - right[panels]).max()) == 0.0
    finally:
        close(stepped)
        close(direct)


def test_the_partial_redraw_leaves_the_six_toggles_painted():
    """A view change repaints the slider row and leaves the toggles alone.

    The partial redraw restores the cached background and paints an opaque
    rectangle over the slider before drawing it again, because the slider's bar
    moved and the background holds it where it was.  The six toggles do not
    move with the view, so they belong to that background and must lie outside
    the rectangle.  A toggle inside it would be painted over and not drawn
    again, which would leave a blank place in the widget row after one step.
    """
    _, figure = build_figure(view_index=0)
    try:
        canvas = figure.figure.canvas
        canvas.draw()
        before = np.asarray(canvas.buffer_rgba()).copy()
        figure.set_view(3)
        after = np.asarray(canvas.buffer_rgba()).copy()
        assert (before != after).any(), 'the step painted nothing'

        height = before.shape[0]
        renderer = canvas.get_renderer()
        for check in (figure.trajectory_check, figure.zoom_check,
                      figure.reference_check, figure.sinogram_check,
                      figure.recon_check, figure.compare_check):
            box = check.ax.get_window_extent(renderer)
            # The rendered array's first row is the top of the figure, and a
            # display box measures its height from the bottom.
            rows = slice(height - int(np.ceil(box.y1)), height - int(box.y0))
            columns = slice(int(box.x0), int(np.ceil(box.x1)))
            assert np.array_equal(before[rows, columns],
                                  after[rows, columns]), (
                f'the step repainted the {check.labels[0].get_text()!r} '
                'toggle')
    finally:
        close(figure)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_every_geometry_takes_every_control(name, tmp_path):
    """Each of the six geometries survives the slider, the toggles, and a
    comparison.

    The controls touch different code for different geometries: the
    translation geometry has no rotation arc, the parallel and multiaxis
    geometries have no source position, and a curved detector has no filled
    face.  The test walks every geometry through every control and saves the
    result, so an artist that one geometry does not create cannot break a
    redraw.
    """
    scene, figure = build_figure(name, view_index=1)
    try:
        figure.set_zoom('volume')
        figure.set_view(3)
        figure.set_show_trajectory(True)
        figure.set_compare(compare_overrides(scene))
        figure.set_view(4)
        figure.set_zoom('scan')
        figure.set_show_trajectory(False)
        figure.set_compare(None)
        path = str(tmp_path / f'{name.replace(" ", "_")}.png')
        figure.save(path, dpi=80)
        assert os.path.getsize(path) > 10 * 1024
    finally:
        close(figure)


def test_plain_draw_paints_the_moving_artists_without_blitting():
    """Without the fast path, a plain full draw must paint the source and the
    detector.

    A backend outside BLIT_BACKENDS repaints the whole figure on every view
    change.  A full draw skips animated artists, so the moving artists must
    not be animated there.  Before this rule the source, the detector, and
    everything the slider moves were invisible on the macosx backend, while a
    saved file, which unmarks the artists, looked right.  The test drives the
    figure through the same plain draw and checks that it paints exactly what
    a draw with every artist unmarked paints.
    """
    import numpy as np
    import mbirtorch
    from mbirtorch.geometry_figure import GeometryFigure
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    model = mbirtorch.ConeBeamModel((12, 16, 24), angles,
                                    source_detector_dist=200.0,
                                    source_iso_dist=100.0, compile_mode='off')
    figure = GeometryFigure(model, view_index=0, blit=False, widgets=True)
    assert all(not artist.get_animated() for _, artist in figure._moving)

    def plain_draw():
        figure.figure.canvas.draw()
        return np.asarray(figure.figure.canvas.buffer_rgba()).copy()

    painted = plain_draw()
    figure._set_animated(False)
    reference = plain_draw()
    # Both draws paint the same artists, so the two buffers are identical.
    assert np.array_equal(painted, reference)

    # And a view change through the plain path moves the source on screen.
    before = plain_draw()
    figure.set_view(3)
    after = plain_draw()
    assert (before != after).any(axis=2).sum() > 500


def test_moving_artists_are_animated_only_on_a_blit_backend():
    """Under Agg with blitting enabled the fast path runs, so the moving
    artists are animated; with blitting disabled they are not."""
    import numpy as np
    import mbirtorch
    from mbirtorch.geometry_figure import GeometryFigure
    angles = np.linspace(0.0, np.pi, 4, endpoint=False)
    model = mbirtorch.ParallelBeamModel((4, 6, 16), angles, compile_mode='off')
    with_blit = GeometryFigure(model, blit=True, widgets=False)
    without = GeometryFigure(model, blit=False, widgets=False)
    assert all(artist.get_animated() for _, artist in with_blit._moving)
    assert all(not artist.get_animated() for _, artist in without._moving)


# ── the labels of the source, the detector, and the pixel-0 marker ──────────

def moving_artists(figure):
    """The artists a view change updates, without their axes."""
    return [artist for _, artist in figure._moving]


def all_labels(figure):
    """The seven labels the three drawing panels carry, as (label, text)."""
    return [(figure._source_text_3d, 'source'),
            (figure._detector_text_3d, 'detector'),
            (figure._top['source_label'], 'source'),
            (figure._top['detector_label'], 'detector'),
            (figure._top['pixel0_label'], 'pixel (0,0)'),
            (figure._side['source_label'], 'source'),
            (figure._side['detector_label'], 'detector')]


def test_the_source_and_the_detector_are_labeled_in_three_panels():
    """Each of the three drawing panels names the source and the detector.

    Without labels nothing in the panels says which artist is the source and
    which the detector, so the elements carry their own labels.  The pixel-0
    marker is named in the top view alone.
    """
    scene, figure = build_figure(view_index=2)
    try:
        moving = moving_artists(figure)
        for label, text in all_labels(figure):
            assert label.get_text().strip() == text
            assert label in moving, text
        view = scene.view(2)
        # The labels sit where the things they name are.
        assert np.allclose(figure._source_text_3d.get_position_3d(),
                           view.source_draw)
        assert figure._top['source_label'].xy == pytest.approx(
            tuple(view.source_draw[TOP_COLUMNS]))
        assert figure._side['source_label'].xy == pytest.approx(
            tuple(view.source_draw[SIDE_COLUMNS]))
        assert figure._top['pixel0_label'].xy == pytest.approx(
            tuple(view.detector_pixel0[TOP_COLUMNS]))
        # The side view names no pixel-0 marker.
        assert 'pixel0_label' not in figure._side
        side_texts = [text.get_text() for text in figure.ax_side.texts]
        assert 'pixel (0,0)' not in side_texts
    finally:
        close(figure)


def test_the_labels_follow_the_source_and_the_detector():
    """A view change moves each label to the new position of its element."""
    scene, figure = build_figure(view_index=0)
    try:
        before = figure._top['source_label'].xy
        figure.set_view(3)
        view = scene.view(3)
        after = figure._top['source_label'].xy
        assert after != before
        assert after == pytest.approx(tuple(view.source_draw[TOP_COLUMNS]))
        assert np.allclose(figure._source_text_3d.get_position_3d(),
                           view.source_draw)
        # The detector's label sits on one of the detector's corners.
        for label, columns in ((figure._top['detector_label'], TOP_COLUMNS),
                               (figure._side['detector_label'],
                                SIDE_COLUMNS)):
            corners = view.detector_corners[:, columns]
            gap = np.linalg.norm(corners - np.asarray(label.xy)[None, :],
                                 axis=1)
            assert float(np.min(gap)) < 1e-9
        gap = np.linalg.norm(
            view.detector_corners
            - np.asarray(figure._detector_text_3d.get_position_3d())[None, :],
            axis=1)
        assert float(np.min(gap)) < 1e-9
    finally:
        close(figure)


def test_the_labels_are_animated_with_the_other_moving_artists():
    """A label follows the rule of every moving artist.

    The rule is `GeometryFigure._animate_moving`: a moving artist is animated
    only where the partial redraw runs.  A label that stayed animated on a
    backend without that path would be invisible there, which is the same
    failure the source and the detector showed before the rule existed.
    """
    _, with_blit = build_figure(blit=True)
    _, without = build_figure(blit=False)
    try:
        assert with_blit._animate_moving() is True
        for label, _ in all_labels(with_blit):
            assert label.get_animated() is True
        assert without._animate_moving() is False
        for label, _ in all_labels(without):
            assert label.get_animated() is False
        # And every moving artist agrees with its labels.
        assert all(artist.get_animated()
                   for artist in moving_artists(with_blit))
        assert not any(artist.get_animated()
                       for artist in moving_artists(without))
    finally:
        close(with_blit)
        close(without)


# ── the two data overlays ───────────────────────────────────────────────────

def overlay_arrays(scene):
    """A sinogram and a reconstruction of the shapes one scan asks for.

    The sinogram counts up over its whole array and the reconstruction is one
    block of ones, so neither is flat and both have a support to draw.
    """
    sinogram = np.arange(int(np.prod(scene.sinogram_shape)),
                         dtype=np.float32).reshape(scene.sinogram_shape)
    recon = np.zeros(scene.recon_shape, dtype=np.float32)
    recon[:3, :3, :3] = 1.0
    return sinogram, recon


def test_removing_the_overlays_leaves_the_view_change_working():
    """Passing None takes both overlays away and the slider still works.

    The sinogram's image is a moving artist, so removing it has to take it out
    of the list the partial redraw walks.  A view change after the removal
    would otherwise draw an artist that no longer belongs to any axes.
    """
    scene, figure = build_figure()
    try:
        sinogram, recon = overlay_arrays(scene)
        figure.set_sinogram(sinogram)
        figure.set_recon(recon)
        assert len(figure.ax_detector.images) == 1
        assert len(figure.ax_top.images) == 1
        assert len(figure.ax_side.images) == 1

        figure.set_sinogram(None)
        figure.set_recon(None)
        for axes in (figure.ax_detector, figure.ax_top, figure.ax_side):
            assert len(axes.images) == 0
        assert figure._recon_images == []
        # The image is gone from the list the partial redraw walks as well.
        from matplotlib.image import AxesImage
        assert figure._sinogram_image is None
        assert not any(isinstance(artist, AxesImage)
                       for _, artist in figure._moving)

        figure.set_view(3)
        assert figure.view_index == 3
        assert 'view 3' in figure.ax_detector.get_title()
    finally:
        close(figure)


def test_the_overlays_follow_the_rule_of_their_kind_of_artist():
    """The sinogram is a moving artist and the silhouette is a static one.

    The sinogram carries the view drawn, so it joins the artists the partial
    redraw updates and is animated exactly where they are.  The silhouette is
    the object, which this drawing holds fixed, so it belongs to the background
    and is never animated.
    """
    scene, with_blit = build_figure(blit=True)
    _, without = build_figure(blit=False)
    try:
        sinogram, recon = overlay_arrays(scene)
        for figure in (with_blit, without):
            figure.set_sinogram(sinogram)
            figure.set_recon(recon)

        assert with_blit._sinogram_image in moving_artists(with_blit)
        assert with_blit._sinogram_image.get_animated() is True
        assert all(artist.get_animated()
                   for artist in moving_artists(with_blit))
        assert without._sinogram_image.get_animated() is False

        for figure in (with_blit, without):
            for _, image in figure._recon_images:
                assert image not in moving_artists(figure)
                assert image.get_animated() is False
    finally:
        close(with_blit)
        close(without)


def test_a_view_change_moves_the_phantoms_projected_outline():
    """The phantom's outline on the detector face follows the view.

    The phantom does not move and the source and the detector do, so the
    phantom's shadow lands somewhere else in each view.  The line that outlines
    that shadow is therefore a moving artist: it is in the list the partial
    redraw walks, it is animated exactly where the other moving artists are,
    and a view change replaces its data.  The scan is the cone helical one,
    whose source turns and rises from one view to the next, so two views put
    the outline in two clearly different places.
    """
    scene, figure = build_figure('cone helical')
    try:
        _, recon = overlay_arrays(scene)
        figure.set_recon(recon)
        line = figure._recon_detector_line
        assert line is not None
        assert line in moving_artists(figure)
        assert line.get_animated() == figure._animate_moving()

        def drawn():
            return np.stack([np.asarray(line.get_xdata(), dtype=np.float64),
                             np.asarray(line.get_ydata(), dtype=np.float64)],
                            axis=1)

        figure.set_view(0)
        first = drawn()
        figure.set_view(scene.num_views // 2)
        second = drawn()
        assert first.shape == second.shape
        # The two arrays break their polylines at the same places, so the
        # finite points of one match the finite points of the other.
        finite = np.isfinite(first).all(axis=1)
        assert finite.any()
        assert np.array_equal(finite, np.isfinite(second).all(axis=1))
        moved = float(np.max(np.abs(first[finite] - second[finite])))
        assert moved > 1.0, ('the outline stayed where it was; the largest '
                             f'change between the two views was {moved} '
                             'detector pixels')
    finally:
        close(figure)


def test_the_overlays_survive_a_comparison_being_added_and_removed():
    """A comparison leaves both overlays drawn and the view change working.

    Installing a comparison rebuilds the legends, re-flows the text panel, and
    repaints the whole figure, and removing one closes a second window.  The
    overlays must come through all of that: the sinogram is still the moving
    artist the partial redraw draws, and the silhouette is still in its two
    panels.
    """
    scene, figure = build_figure()
    try:
        sinogram, recon = overlay_arrays(scene)
        figure.set_sinogram(sinogram)
        figure.set_recon(recon)

        figure.set_compare(compare_overrides(scene))
        assert figure.compare_scene is not None
        assert figure._sinogram_image in moving_artists(figure)
        assert len(figure._recon_images) == 2
        # The comparison gets no overlay of its own, so the counts do not grow.
        assert len(figure.ax_detector.images) == 1

        figure.set_view(3)
        drawn = np.asarray(figure._sinogram_image.get_array())
        assert np.array_equal(drawn, sinogram[3])

        figure.set_compare(None)
        figure.set_view(4)
        drawn = np.asarray(figure._sinogram_image.get_array())
        assert np.array_equal(drawn, sinogram[4])
        assert len(figure.ax_top.images) == 1
        assert len(figure.ax_side.images) == 1
    finally:
        close(figure)


def ball_phantom(scene, radius=8.0):
    """A ball of ones at the middle of a scan's reconstruction.

    A ball gives every section of the phantom a different outline, so the
    projected outline the timed step rebuilds is a real one and not four
    straight lines.
    """
    rows, cols, slices = scene.recon_shape
    i, j, k = np.meshgrid(np.arange(rows) - (rows - 1) / 2.0,
                          np.arange(cols) - (cols - 1) / 2.0,
                          np.arange(slices) - (slices - 1) / 2.0,
                          indexing='ij')
    inside = i ** 2 + j ** 2 + k ** 2 <= radius ** 2
    return np.where(inside, 1.0, 0.0).astype(np.float32)


def test_a_slider_step_with_a_sinogram_stays_under_the_gate():
    """A slider step on the 1800-view scan stays under 100 ms with a sinogram.

    The model is the probe's own timing model, with a sinogram painted on the
    detector face and a phantom drawn in the volume box.  A step replaces the
    image's data with one view of the array and projects the phantom's section
    outlines onto the detector face, and the partial redraw then draws both
    with the other moving artists.  The test prints the mean and the largest of
    ten steps, because the number is what the gate is about.
    """
    import time

    model = probe.build_timing_model()
    scene = GeometryScene.from_model(model)
    sinogram = np.zeros(scene.sinogram_shape, dtype=np.float32)
    # One bright pixel per view, walking across the detector, so that every
    # step really replaces the image's data.
    for view_index in range(scene.num_views):
        sinogram[view_index, view_index % scene.num_det_rows,
                 view_index % scene.num_det_channels] = 1.0
    figure = GeometryFigure(scene, sinogram=sinogram,
                            recon=ball_phantom(scene))
    try:
        # One step before the measurement, so that the background is cached
        # and the first step timed is like every later one.
        figure.set_view(1)
        times = []
        for step in range(TIMED_STEPS):
            started = time.perf_counter()
            figure.set_view((step + 1) * TIMED_STEP_STRIDE % scene.num_views)
            times.append(1000.0 * (time.perf_counter() - started))
        times = np.asarray(times)
        print(f'\nset_view step with a sinogram, {scene.num_views} views: '
              f'mean {times.mean():.1f} ms, max {times.max():.1f} ms')
        assert times.mean() < STEP_GATE_MS, (
            f'a slider step took {times.mean():.1f} ms on average, against '
            f'a gate of {STEP_GATE_MS:.0f} ms')
    finally:
        close(figure)


# ── the three overlay toggles ───────────────────────────────────────────────

#: How dark a pixel must be, in summed red, green, and blue out of 765, to
#: count as painted by the sinogram of
#: ``test_a_hidden_sinogram_is_not_drawn_after_a_view_change``.  That sinogram
#: is zeros but for one pixel, and zero is black in the gray colormap, so the
#: painted image covers the detector face in black.
DARK_SUM = 100

#: What fraction of the sinogram image's area is that dark with the sinogram
#: painted, and what fraction may be with the sinogram hidden.  The panel draws
#: its detector-iso marker in near black over the same area, which is the few
#: pixels the second number allows.
DARK_FRACTION_PAINTED = 0.5
DARK_FRACTION_HIDDEN = 0.01


def overlay_figure(name='cone flat', **kwargs):
    """A figure with all three overlays installed.

    The sinogram and the phantom are the arrays of ``overlay_arrays`` and the
    comparison is the ten-channel offset the other comparison tests use.
    """
    scene, figure = build_figure(name, **kwargs)
    sinogram, recon = overlay_arrays(scene)
    figure.set_sinogram(sinogram)
    figure.set_recon(recon)
    figure.set_compare(compare_overrides(scene))
    return scene, figure


def overlay_artists(figure):
    """The artists each overlay toggle governs, by the name of its toggle.

    The comparison's source path is left out, because it answers to the
    source-path toggle as well;
    ``test_the_comparison_path_follows_both_toggles`` covers it.
    """
    paths = [line for _, line in figure._compare_trajectory_lines]
    return {
        'sinogram': [figure._sinogram_image],
        'phantom': [artist for _, artist in
                    figure._recon_images + figure._recon_outlines],
        'comparison': [artist for _, artist in
                       figure._compare_moving + figure._compare_static
                       if artist not in paths],
    }


def test_each_overlay_toggle_hides_and_shows_its_artists():
    """Each toggle hides the artists of its own overlay and no others.

    Hiding removes nothing, so the counts do not change and showing the
    overlay again costs no rebuilding.  The phantom's toggle governs its two
    fills, its outline in each projected panel, its outline in the 3D panel,
    and its projected outline on the detector face, because those are one
    drawing of one array.
    """
    _, figure = overlay_figure()
    try:
        groups = overlay_artists(figure)
        assert len(groups['phantom']) == 6, (
            'two fills, two projected outlines, a 3D outline, a detector line')
        assert figure._recon_detector_line in groups['phantom']
        assert groups['comparison']

        for name, setter in (('sinogram', figure.set_show_sinogram),
                             ('phantom', figure.set_show_recon),
                             ('comparison', figure.set_show_compare)):
            setter(False)
            assert not any(artist.get_visible() for artist in groups[name])
            others = [artist for key, group in groups.items() if key != name
                      for artist in group]
            assert all(artist.get_visible() for artist in others), name
            setter(True)
            assert all(artist.get_visible() for artist in groups[name])
        # Nothing was removed, so the artists are the ones we started with.
        assert overlay_artists(figure) == groups
    finally:
        close(figure)


def test_the_overlay_toggles_and_the_set_show_methods_agree():
    """get_status of each toggle follows its set_show_ method, and a click on
    the toggle changes the figure's state.

    A click is made the way the widget makes one: ``set_active`` moves the
    button and calls the handler.  The handler is then called again by hand, so
    that the test covers the handler itself and not only the method it calls.
    """
    _, figure = overlay_figure()
    try:
        toggles = (('sinogram', figure.sinogram_check,
                    figure.set_show_sinogram, figure._on_sinogram_check,
                    lambda: figure.show_sinogram),
                   ('phantom', figure.recon_check, figure.set_show_recon,
                    figure._on_recon_check, lambda: figure.show_recon),
                   ('comparison', figure.compare_check,
                    figure.set_show_compare, figure._on_compare_check,
                    lambda: figure.show_compare))
        for label, check, setter, handler, state in toggles:
            assert state() is True, label
            assert check.get_status()[0] is True, label

            setter(False)
            assert state() is False
            assert check.get_status()[0] is False, label

            # Clicking the toggle turns the overlay back on.
            check.set_active(0)
            handler(label)
            assert state() is True, label
            assert check.get_status()[0] is True, label
    finally:
        close(figure)


def test_a_toggle_whose_overlay_is_absent_does_nothing():
    """A figure with no overlay has the three toggles all the same.

    The toggles are built whatever data the figure holds, so an array added
    later has its toggle ready and in the state the toggle is showing.
    """
    scene, figure = build_figure()
    try:
        assert figure.sinogram_check is not None
        assert figure.recon_check is not None
        assert figure.compare_check is not None
        assert figure._sinogram_image is None
        assert figure.compare_scene is None

        figure.set_show_sinogram(False)
        figure.set_show_recon(False)
        figure.set_show_compare(False)
        assert figure.show_sinogram is False
        assert figure.show_recon is False
        assert figure.show_compare is False
        # The figure still redraws, with nothing to hide.
        figure.set_view(2)
        assert figure.view_index == 2

        # An array given now is installed hidden, because the toggle says so.
        sinogram, recon = overlay_arrays(scene)
        figure.set_sinogram(sinogram)
        figure.set_recon(recon)
        assert figure._sinogram_image.get_visible() is False
        assert not any(artist.get_visible() for _, artist
                       in figure._recon_images + figure._recon_outlines)
        figure.set_show_sinogram(True)
        assert figure._sinogram_image.get_visible() is True
    finally:
        close(figure)


def test_hiding_an_overlay_leaves_its_numbers_in_the_text_panel():
    """The sinogram's title note follows its toggle and the comparison's
    numbers do not.

    The detector panel's title names a painted sinogram, so hiding the
    sinogram takes the note out of it, and the footer names only the overlays
    drawn.  The comparison is the other way: the drawing is hidden and the
    parameters stay, because those are the numbers a calibration user is
    reading.  The heading of that block says the drawing is hidden.
    """
    _, figure = overlay_figure()
    try:
        assert 'with sinogram' in figure.ax_detector.get_title()
        assert 'overlays : sinogram' in figure._footer_text.get_text()

        figure.set_show_sinogram(False)
        figure.set_show_recon(False)
        assert 'with sinogram' not in figure.ax_detector.get_title()
        assert 'overlays' not in figure._footer_text.get_text()

        block = figure._compare_text.get_text()
        assert 'det_channel_offset' in block
        assert 'hidden' not in block.splitlines()[0]
        figure.set_show_compare(False)
        block = figure._compare_text.get_text()
        assert 'hidden' in block.splitlines()[0]
        assert 'det_channel_offset' in block
        # The comparison window stays open, because it holds the numbers too.
        import matplotlib.pyplot as plt
        assert plt.fignum_exists(figure.compare_figure.number)
    finally:
        close(figure)


def test_the_comparison_path_follows_both_toggles():
    """The comparison's source path is drawn only with both toggles on.

    The path belongs to the comparison and to the source-path toggle, so
    either one hides it.
    """
    scene, figure = build_figure('cone helical', show_trajectory=True)
    try:
        figure.set_compare(compare_overrides(scene))
        lines = [line for _, line in figure._compare_trajectory_lines]
        assert lines and all(line.get_visible() for line in lines)

        figure.set_show_compare(False)
        assert not any(line.get_visible() for line in lines)
        figure.set_show_compare(True)
        assert all(line.get_visible() for line in lines)

        figure.set_show_trajectory(False)
        assert not any(line.get_visible() for line in lines)
        # With the path off, turning the comparison on leaves the path off.
        figure.set_show_compare(False)
        figure.set_show_compare(True)
        assert not any(line.get_visible() for line in lines)
    finally:
        close(figure)


def rendered_figure(figure):
    """The figure drawn, as an array of red, green, and blue values.

    A draw on a blitting backend paints the background and then the moving
    artists, so the buffer holds both by the time it is read.
    """
    canvas = figure.figure.canvas
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].astype(np.int16)


def dark_fraction(rendered, box):
    """What fraction of one display box is painted near black."""
    height, width = rendered.shape[:2]
    horizontal = np.arange(width)[None, :] + 0.5
    # The rendered array's first row is the top of the figure, and a display
    # box measures its height from the bottom.
    vertical = height - np.arange(height)[:, None] - 0.5
    inside = ((horizontal >= box.x0) & (horizontal <= box.x1)
              & (vertical >= min(box.y0, box.y1))
              & (vertical <= max(box.y0, box.y1)))
    dark = (rendered.sum(axis=2) < DARK_SUM) & inside
    return float(dark.sum()) / float(max(inside.sum(), 1))


def test_a_hidden_sinogram_is_not_drawn_after_a_view_change():
    """A hidden sinogram stays hidden through the partial-redraw path.

    The sinogram's image is a moving artist, and on a blitting backend a
    moving artist is marked animated, which keeps it out of a full draw and
    leaves it to ``_draw_moving_and_blit``.  That routine draws an artist
    through its axes, which is a path a full draw does not take, so hiding the
    image has to stop it there too.  It does, twice over: the routine draws
    only the artists that report themselves visible, and matplotlib's own draw
    returns at once for an invisible artist.

    The check is the rendered figure and not only the flag.  The sinogram is
    zeros but for one pixel, and zero is black in the gray colormap, so a
    painted sinogram covers the detector face in black and a hidden one leaves
    the panel's own light background.
    """
    scene, figure = build_figure()
    try:
        assert figure._blit_usable(), 'the fast path is what this test covers'
        sinogram = np.zeros(scene.sinogram_shape, dtype=np.float32)
        sinogram[0, 0, 0] = 1.0
        figure.set_sinogram(sinogram)
        assert figure._sinogram_image.get_animated() is True

        rendered = rendered_figure(figure)
        box = figure._sinogram_image.get_window_extent(
            figure.figure.canvas.get_renderer())
        assert dark_fraction(rendered, box) > DARK_FRACTION_PAINTED

        figure.set_show_sinogram(False)
        figure.set_view(1)
        figure.set_view(2)
        assert figure.view_index == 2
        assert figure._sinogram_image.get_visible() is False
        # The image still holds the view the slider is on, so the step did run.
        drawn = np.asarray(figure._sinogram_image.get_array())
        assert np.array_equal(drawn, sinogram[2])
        rendered = rendered_figure(figure)
        assert dark_fraction(rendered, box) < DARK_FRACTION_HIDDEN
    finally:
        close(figure)


# ── the angle-0 reference ───────────────────────────────────────────────────

def reference_positions(figure):
    """What each reference artist is drawn at, for the artists that can say.

    A line reports its data and a text its position.  The arrowheads, one
    ``quiver`` and one ``FancyArrowPatch``, report neither in a form worth
    comparing, so they are left out and only counted.
    """
    positions = []
    for _, artist in figure._reference_artists:
        if hasattr(artist, 'get_data_3d'):
            positions.append(np.concatenate(artist.get_data_3d()))
        elif hasattr(artist, 'get_position_3d'):
            positions.append(np.asarray(artist.get_position_3d()))
        elif isinstance(artist.get_visible(), bool) and hasattr(artist, 'xy'):
            positions.append(np.asarray(artist.xy, dtype=np.float64))
        elif hasattr(artist, 'get_data'):
            positions.append(np.concatenate(artist.get_data()))
    return positions


def test_the_reference_is_drawn_in_the_3d_and_top_panels():
    """The angle-0 reference is drawn where the projector's convention puts it.

    The 3D view and the top view get it.  The side view and the detector face
    do not: the side view is the yz plane, in which a rotation about z moves
    nothing.  The drawn source and detector are the scene's own
    ``reference_view``.
    """
    scene, figure = build_figure(view_index=2)
    try:
        assert figure.show_reference is True
        panels = [axes for axes, _ in figure._reference_artists]
        assert panels.count(figure.ax_3d) == 5
        assert panels.count(figure.ax_top) == 5
        assert figure.ax_side not in panels
        assert figure.ax_detector not in panels

        reference = scene.reference_view()
        stars = [artist for axes, artist in figure._reference_artists
                 if axes is figure.ax_top
                 and getattr(artist, 'get_marker', lambda: '')() == '*']
        assert len(stars) == 1
        assert np.allclose(np.concatenate(stars[0].get_data()),
                           reference.source_draw[TOP_COLUMNS])
        # The reference is not the view drawn, whose angle is not zero.
        assert not np.allclose(reference.source_draw,
                               scene.view(2).source_draw, atol=1e-3)
    finally:
        close(figure)


def test_the_reference_does_not_move_with_the_view():
    """The slider moves the geometry and leaves the reference where it is."""
    scene, figure = build_figure(view_index=0)
    try:
        before = reference_positions(figure)
        assert len(before) == 8
        figure.set_view(4)
        after = reference_positions(figure)
        assert len(after) == len(before)
        for first, second in zip(before, after):
            assert np.array_equal(first, second)
        # Meanwhile the source did move.
        assert not np.allclose(scene.view(0).source_draw,
                               scene.view(4).source_draw, atol=1e-3)
    finally:
        close(figure)


def test_the_reference_artists_are_not_animated():
    """The reference belongs to the background, so it is never animated.

    A view change does not redraw it, and a full repaint has to paint it.
    """
    _, figure = build_figure(blit=True)
    try:
        assert figure._animate_moving() is True
        for _, artist in figure._reference_artists:
            assert artist.get_animated() is False
        moving = moving_artists(figure)
        for _, artist in figure._reference_artists:
            assert artist not in moving
    finally:
        close(figure)


def test_the_reference_toggle_hides_and_shows_it():
    """The toggle widget and set_show_reference agree on the state."""
    _, figure = build_figure()
    try:
        assert figure.reference_check.get_status()[0] is True
        assert all(artist.get_visible()
                   for _, artist in figure._reference_artists)

        figure.set_show_reference(False)
        assert figure.show_reference is False
        assert figure.reference_check.get_status()[0] is False
        assert not any(artist.get_visible()
                       for _, artist in figure._reference_artists)

        # Clicking the widget turns it back on.
        figure.reference_check.set_active(0)
        assert figure.show_reference is True
        assert all(artist.get_visible()
                   for _, artist in figure._reference_artists)
    finally:
        close(figure)


def test_the_reference_is_off_when_the_constructor_says_so():
    """show_reference=False builds the artists and leaves them hidden."""
    _, figure = build_figure(show_reference=False)
    try:
        assert figure.show_reference is False
        assert figure.reference_check.get_status()[0] is False
        assert not any(artist.get_visible()
                       for _, artist in figure._reference_artists)
    finally:
        close(figure)


def test_the_panel_limits_hold_the_reference():
    """The reference is inside the panel limits, whether or not it is drawn.

    The limits are computed once, so a reference outside them would be cut off
    when the toggle turned it on.  The check is made with the reference off,
    because that is the case a limit computation could leave out.
    """
    scene, figure = build_figure(view_index=2, show_reference=False)
    try:
        reference = scene.reference_view()
        points = np.concatenate([reference.detector_outline,
                                 reference.source_draw.reshape(1, 3),
                                 reference.detector_origin.reshape(1, 3)])
        assert geometry_figure._within(points[:, TOP_COLUMNS],
                                       figure._limits['top'])
        assert geometry_figure._within(points, figure._limits['scan'])
    finally:
        close(figure)


def test_the_comparison_is_named_in_the_top_view():
    """A comparison's detector carries the word "comparison", once."""
    scene, figure = build_figure(view_index=2)
    try:
        figure.set_compare(compare_overrides(scene))
        label = figure._compare['label_top']
        assert label.get_text() == 'comparison'
        assert label.axes is figure.ax_top
        corners = figure.compare_scene.view(2).detector_corners[
            :, TOP_COLUMNS]
        gap = np.linalg.norm(corners - np.asarray(label.xy)[None, :], axis=1)
        assert float(np.min(gap)) < 1e-9
        for axes in (figure.ax_side, figure.ax_3d):
            assert 'comparison' not in [text.get_text()
                                        for text in axes.texts]
        figure.set_compare(None)
        assert 'comparison' not in [text.get_text()
                                    for text in figure.ax_top.texts]
    finally:
        close(figure)


def test_the_volume_zoom_hides_the_3d_labels_outside_its_cube():
    """A 3D label whose point leaves the cube is hidden, not drawn outside it.

    Matplotlib does not clip a 3D text artist or a ``quiver`` arrowhead to the
    axes limits, so in the volume zoom the source's label was drawn where its
    point projects, which is outside the panel and on top of the rest of the
    figure.
    """
    _, figure = build_figure(view_index=2)
    try:
        assert figure._source_text_3d.get_visible() is True
        assert figure._detector_text_3d.get_visible() is True
        assert figure._reference_text_3d.get_visible() is True

        figure.set_zoom('volume')
        cube = np.array([figure.ax_3d.get_xlim(), figure.ax_3d.get_ylim(),
                         figure.ax_3d.get_zlim()])
        source = figure.scene.view(2).source_draw
        # The source is outside the cube, so its label is not drawn.
        assert (source < cube[:, 0]).any() or (source > cube[:, 1]).any()
        assert figure._source_text_3d.get_visible() is False
        assert figure._detector_text_3d.get_visible() is False
        assert figure._reference_text_3d.get_visible() is False
        assert figure._reference_arrow_3d.get_visible() is False

        figure.set_zoom('scan')
        assert figure._source_text_3d.get_visible() is True
        assert figure._reference_text_3d.get_visible() is True
        assert figure._reference_arrow_3d.get_visible() is True
    finally:
        close(figure)


def test_geometry_viewer_mirrors_slice_viewer_nonblocking_registry():
    """geometry_viewer keeps a nonblocking figure alive in a module registry,
    and the next blocking call closes it, as mbirtorch.slice_viewer does."""
    import matplotlib.pyplot as plt
    import mbirtorch.geometry_figure as module
    cfg = CONFIGS_BY_NAME['parallel']
    model = probe.build_model(cfg)
    module._NONBLOCKING_FIGURES.clear()
    first = module.geometry_viewer(model, block=False)
    assert first in module._NONBLOCKING_FIGURES
    assert plt.fignum_exists(first.figure.number)
    # Under Agg, show prints a line and returns, so the blocking call returns
    # at once and runs the registry's closing step.
    second = module.geometry_viewer(model, view_index=2, block=True)
    assert module._NONBLOCKING_FIGURES == []
    assert not plt.fignum_exists(first.figure.number)
    assert second.view_index == 2
    plt.close(second.figure)


def test_geometry_viewer_takes_the_viewer_options_explicitly():
    """The entry point exposes the figure's options as named arguments."""
    import matplotlib.pyplot as plt
    import mbirtorch.geometry_figure as module
    cfg = CONFIGS_BY_NAME['cone flat']
    model = probe.build_model(cfg)
    figure = module.geometry_viewer(model, view_index=1, show_trajectory=True,
                                    compare=dict(det_channel_offset=5.0),
                                    show_reference=False, zoom='volume',
                                    title='named', block=False)
    assert figure.view_index == 1
    assert figure.show_trajectory is True
    assert figure.compare_scene is not None
    module._NONBLOCKING_FIGURES.clear()
    plt.close(figure.figure)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
