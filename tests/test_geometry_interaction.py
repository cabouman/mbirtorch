"""Headless tests for the widgets, the zoom, and the comparison overlay.

These are tests of `geometry_figure.py`.  The view slider covers every view of
the scan, and moving it redraws the geometry.  The source-path toggle draws one
polyline per panel and not one marker per view.  The volume zoom puts a cube
around the volume on the 3D panel.  A comparison whose channel offset is ten
channels larger draws its projected volume outline ten channels over from the
primary's, follows the slider, and leaves no artist behind when it is removed.

A later group covers the two data overlays.  Passing None removes an overlay
and leaves the view change working.  The phantom's outline on the detector face
is a moving artist, and a view change puts it somewhere else.  Both overlays
come through a comparison being added and removed.

A later group covers the three overlay toggles.  Each toggle hides the artists
of its own overlay and no others, and the state of each toggle follows its
``set_show_`` method and a click on the widget.  The last of them renders the
figure: a sinogram hidden and then stepped past twice must leave the detector
panel's own background on the screen, which is what the partial-redraw path
does with an artist that is animated and invisible at once.

The last tests cover the angle-0 reference: the panel limits hold it whether or
not it is drawn, its toggle hides and shows it, and it does not move with the
view.

The tests run under the Agg backend and open no window.
"""

import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # the tests draw into a buffer and open no window

import geometry_probe as probe
from mbirtorch.viewers.geometry_scene import GeometryScene
import mbirtorch.viewers.geometry_figure as geometry_figure
from mbirtorch.viewers.geometry_figure import (COLORS, GeometryFigure,
                                               VOLUME_BOX_EDGES,
                                               ZOOM_VOLUME_WIDTH_FACTOR,
                                               TOP_PANEL_COLUMNS)

#: The object coordinates the top panel puts on its two axes, as a list for
#: indexing.  The top view is the xy plane seen from -z, so it draws y across
#: the screen and x down it; see the display convention in `geometry_figure`.
TOP_COLUMNS = list(TOP_PANEL_COLUMNS)

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

#: How many channels the comparison's detector offset is moved by.  Ten
#: channels is the size the comparison test uses.
COMPARISON_CHANNEL_SHIFT = 10


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


def moving_artists(figure):
    """The artists a view change updates, without their axes."""
    return [artist for _, artist in figure._moving]


# ── the view slider ─────────────────────────────────────────────────────────

def test_the_slider_covers_the_scan_and_changes_the_drawn_view():
    """The slider covers every view, and setting its value redraws another one.

    The check reads the drawn data and not only the view index, because a
    slider that moved the index without redrawing would leave the figure
    showing the old view.
    """
    scene, figure = build_figure(view_index=0)
    try:
        slider = figure.view_slider
        assert slider is not None
        assert slider.valmin == 0
        assert slider.valmax == scene.num_views - 1
        assert slider.valstep == 1
        assert int(round(float(slider.val))) == figure.view_index

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


# ── the trajectory toggle ───────────────────────────────────────────────────

def test_the_source_path_toggles_and_is_one_polyline_per_panel():
    """The toggle turns the path on and off, and the path is one line per panel.

    A path drawn as one marker per view would cost 1800 artists on a helical
    scan, which is what this test rules out.  The test therefore counts the
    lines and checks that they carry no marker.  The widget and
    ``set_show_trajectory`` must agree on the state.
    """
    scene, figure = build_figure('cone helical')
    try:
        assert figure.show_trajectory is False
        assert not lines_of_color(figure.ax_top, COLORS['trajectory'])
        assert figure.trajectory_check.get_status()[0] is False

        # Clicking the widget calls its callback with the label.
        figure.trajectory_check.set_active(0)
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


def test_the_comparison_is_drawn_from_every_input_and_follows_the_slider():
    """A comparison given three ways is drawn, and it follows the slider.

    The comparison can be a scene, a model, or a dictionary of overrides, and
    the constructor's ``compare`` argument draws the same overlay as
    ``set_compare``.  The drawn outline is the comparison scene's own outline
    for the view the slider is on.
    """
    cfg = CONFIGS_BY_NAME['cone flat']
    scene = GeometryScene.from_model(probe.build_model(cfg))
    figure = GeometryFigure(scene, view_index=0,
                            compare=compare_overrides(scene))
    try:
        # The constructor's comparison is drawn.
        assert figure.compare_scene is not None
        assert lines_of_color(figure.ax_detector, COLORS['compare'])

        # A scene, a model, and a dictionary of overrides are all accepted.
        other = scene.with_parameters(compare_overrides(scene))
        figure.set_compare(other)
        assert figure.compare_scene is other

        model = probe.build_model(CONFIGS_BY_NAME['parallel'])
        figure.set_compare(model)
        assert figure.compare_scene.kind == 'parallel'

        figure.set_compare(compare_overrides(scene))
        assert figure.compare_scene.kind == scene.kind

        # The comparison is drawn for the view the slider is on.
        figure.view_slider.set_val(4)
        assert figure.view_index == 4
        drawn = finite_points(figure._compare['detector_top'])
        expected = figure.compare_scene.view(4).detector_outline
        assert np.allclose(drawn, expected[:, TOP_COLUMNS])
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
    from mbirtorch.viewers.geometry_figure import GeometryFigure
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
    source-path toggle as well.
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


def test_each_overlay_toggle_hides_and_shows_its_own_artists():
    """Each toggle hides the artists of its own overlay and no others.

    Hiding removes nothing, so showing the overlay again costs no rebuilding.
    The phantom's toggle governs its two fills, its outline in each projected
    panel, its outline in the 3D panel, and its projected outline on the
    detector face, because those are one drawing of one array.  The widget's
    ``get_status`` follows the ``set_show_`` method, and a click on the widget
    turns the overlay back on.  A click is made the way the widget makes one:
    ``set_active`` moves the button and calls the handler.  The handler is then
    called again by hand, so that the test covers the handler itself and not
    only the method it calls.
    """
    _, figure = overlay_figure()
    try:
        groups = overlay_artists(figure)
        assert figure._recon_detector_line in groups['phantom']
        assert groups['comparison']

        toggles = (('sinogram', figure.sinogram_check,
                    figure.set_show_sinogram, figure._on_sinogram_check,
                    lambda: figure.show_sinogram),
                   ('phantom', figure.recon_check, figure.set_show_recon,
                    figure._on_recon_check, lambda: figure.show_recon),
                   ('comparison', figure.compare_check,
                    figure.set_show_compare, figure._on_compare_check,
                    lambda: figure.show_compare))
        for name, check, setter, handler, state in toggles:
            assert state() is True, name
            assert check.get_status()[0] is True, name

            setter(False)
            assert state() is False, name
            assert check.get_status()[0] is False, name
            assert not any(artist.get_visible() for artist in groups[name])
            others = [artist for key, group in groups.items() if key != name
                      for artist in group]
            assert all(artist.get_visible() for artist in others), name

            # Clicking the toggle turns the overlay back on.
            check.set_active(0)
            handler(name)
            assert state() is True, name
            assert check.get_status()[0] is True, name
            assert all(artist.get_visible() for artist in groups[name])
        # Nothing was removed, so the artists are the ones we started with.
        assert overlay_artists(figure) == groups
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


def test_the_reference_stays_put_and_the_panel_limits_hold_it():
    """The panel limits hold the reference, its toggle shows and hides it, and
    the slider does not move it.

    The limits are computed once, so a reference outside them would be cut off
    when the toggle turned it on.  The check is made with the reference off,
    because that is the case a limit computation could leave out.  The slider
    then moves the geometry and leaves the reference where it is.
    """
    scene, figure = build_figure(view_index=0, show_reference=False)
    try:
        reference = scene.reference_view()
        points = np.concatenate([reference.detector_outline,
                                 reference.source_draw.reshape(1, 3),
                                 reference.detector_origin.reshape(1, 3)])
        assert geometry_figure._within(points[:, TOP_COLUMNS],
                                       figure._limits['top'])
        assert geometry_figure._within(points, figure._limits['scan'])

        assert figure.show_reference is False
        assert figure.reference_check.get_status()[0] is False
        assert not any(artist.get_visible()
                       for _, artist in figure._reference_artists)

        # Clicking the widget draws it.
        figure.reference_check.set_active(0)
        assert figure.show_reference is True
        assert all(artist.get_visible()
                   for _, artist in figure._reference_artists)

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

        figure.set_show_reference(False)
        assert figure.reference_check.get_status()[0] is False
        assert not any(artist.get_visible()
                       for _, artist in figure._reference_artists)
    finally:
        close(figure)


# ── the geometry_viewer entry point ─────────────────────────────────────────

def test_geometry_viewer_takes_the_options_and_keeps_a_nonblocking_figure():
    """The entry point exposes the figure's options as named arguments, keeps a
    nonblocking figure alive in a module registry, and closes it on the next
    blocking call, as mbirtorch.slice_viewer does."""
    import matplotlib.pyplot as plt
    import mbirtorch.viewers.geometry_figure as module
    cfg = CONFIGS_BY_NAME['cone flat']
    model = probe.build_model(cfg)
    module._NONBLOCKING_FIGURES.clear()
    first = module.geometry_viewer(model, view_index=1, show_trajectory=True,
                                   compare=dict(det_channel_offset=5.0),
                                   show_reference=False, zoom='volume',
                                   title='named', block=False)
    assert first.view_index == 1
    assert first.show_trajectory is True
    assert first.compare_scene is not None
    assert first in module._NONBLOCKING_FIGURES
    assert plt.fignum_exists(first.figure.number)
    # Under Agg, show prints a line and returns, so the blocking call returns
    # at once and runs the registry's closing step.
    second = module.geometry_viewer(model, view_index=2, block=True)
    assert module._NONBLOCKING_FIGURES == []
    assert not plt.fignum_exists(first.figure.number)
    assert second.view_index == 2
    close(second)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
