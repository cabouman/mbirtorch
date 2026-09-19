"""Headless tests for mbirtorch/viewers/geometry_figure.py.

The tests cover five things.  The first is the import discipline: importing
the figure module must not import ``matplotlib.pyplot``, which is checked in a
separate interpreter because this one has pyplot loaded already.  The second is
that a figure builds, redraws, and saves for each of the six scan geometries of
``geometry_probe.py``.  The third is that the detector-face panel draws
the scene's own projected volume outline and not a recomputed one.  The fourth
is the display convention: every panel draws negative z at the top, the top
and side views draw y to the left, and the detector face puts row 0 at the
top.  Those tests read display coordinates through ``ax.transData``, so they
check where a point is drawn on the screen and not only what the axes limits
say.

The fifth is the two data overlays: a sinogram painted on the detector face
and a reconstruction drawn as a silhouette in the volume box.  Those tests
render the figure and read the pixels back, because where an image lands on
the screen is what they are about.  One of them forward-projects a phantom
with the real projector and compares the painted sinogram with the projected
outline the panel draws over it.  Neither overlay is held whole: a detector
wider than the painted size is subsampled for display, and a reconstruction is
read a chunk of slices at a time, both on whatever device the array sits on,
which the last tests of each group check against the host's own answer.

The last group covers the phantom's outline.  The 3D panel draws the support in
a few planes across the direction it is thinnest along.  A phantom of one block
of voxels therefore gets one rectangle per plane it reaches, whose corners are
the scene's own voxel centers half a voxel outside that block.  A field of small
blobs is coarsened until its outlines fit the panel's point budget.  The same
sections are projected onto the detector face, where they have to follow the
shadow the real projector paints there.  The top view gets an outline that lies
on the block's projected rectangle.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # tests write files and open no window

import geometry_probe as probe
from mbirtorch.viewers.geometry_scene import GeometryScene
import mbirtorch.viewers.geometry_figure as geometry_figure
from mbirtorch.viewers.geometry_figure import (COLORS, GeometryFigure,
                                               VOLUME_BOX_EDGES,
                                               TOP_PANEL_COLUMNS, SIDE_PANEL_COLUMNS)

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

#: The five panels a figure builds: the 3D view, the top view, the side view,
#: the detector face, and the text panel.
EXPECTED_PANEL_COUNT = 5

#: The seven widget axes a figure builds beside the panels: the view slider and
#: the six toggles.  The figure's axes list holds these as well as the panels.
EXPECTED_WIDGET_AXES = 7

#: Smallest acceptable size of a saved figure, in bytes.  A PNG of an empty
#: figure of this size is a few kilobytes, so a file above this holds a
#: drawing.
MIN_PNG_BYTES = 10 * 1024


def build_scene(name):
    """Build the model and the scene of one configuration.

    A test that has to make an array of the scan's own shape needs the scene
    before it can build the figure, so the two steps are separate.
    """
    return GeometryScene.from_model(probe.build_model(CONFIGS_BY_NAME[name]))


def build_figure(name, **kwargs):
    """Build the model, the scene, and a figure for one configuration."""
    scene = build_scene(name)
    return scene, GeometryFigure(scene, **kwargs)


def close(figure):
    """Close a figure's matplotlib window so that the tests do not pile up."""
    import matplotlib.pyplot as plt
    plt.close(figure.figure)


# ── the import discipline ────────────────────────────────────────────────────

def test_import_does_not_load_pyplot():
    """Importing the viewer must not import pyplot.

    The check runs in a separate interpreter, because this test process has
    already imported pyplot itself.  A module that imported pyplot at import
    time would resolve a matplotlib backend, and on a machine with no display
    that can fail or open a window nobody asked for.  The slice viewer
    follows the same rule.
    """
    program = ('import sys; import mbirtorch.viewers.geometry_figure; '
               "print('matplotlib.pyplot' in sys.modules); "
               "print('mpl_toolkits.mplot3d' in sys.modules)")
    result = subprocess.run([sys.executable, '-c', program], cwd=HERE,
                            capture_output=True, text=True, check=True)
    assert result.stdout.split() == ['False', 'False'], result.stdout


# ── one figure per geometry ──────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_a_figure_builds_and_redraws_for_every_geometry(name):
    """A figure builds for every geometry, holds the five panels, and redraws.

    The panel count and the widget count are checked on the figure as built and
    again after a view change and a trajectory toggle, because a redraw that
    dropped or added an axes would change them.
    """
    _, figure = build_figure(name)
    try:
        assert len(figure.panel_axes) == EXPECTED_PANEL_COUNT
        assert (len(figure.figure.axes)
                == EXPECTED_PANEL_COUNT + EXPECTED_WIDGET_AXES)
        # The first panel is the 3D view, which counts as one axes.
        assert hasattr(figure.ax_3d, 'get_zlim')
        for axes in figure.panel_axes:
            assert axes.figure is figure.figure
            assert axes in figure.figure.axes

        figure.set_view(1)
        assert figure.view_index == 1
        figure.set_show_trajectory(True)
        assert figure.show_trajectory is True
        figure.set_view(0)
        figure.set_show_trajectory(False)
        assert (len(figure.figure.axes)
                == EXPECTED_PANEL_COUNT + EXPECTED_WIDGET_AXES)
    finally:
        close(figure)


def test_a_figure_built_from_a_model_saves_a_png(tmp_path):
    """A figure built from a model saves a PNG file that holds a drawing."""
    model = probe.build_model(CONFIGS_BY_NAME['cone flat'])
    figure = GeometryFigure.from_model(model, view_index=1)
    try:
        assert figure.scene.kind == 'cone'
        assert figure.view_index == 1
        path = str(tmp_path / 'cone_flat.png')
        figure.save(path, dpi=100)
        assert os.path.exists(path)
        assert os.path.getsize(path) > MIN_PNG_BYTES
    finally:
        close(figure)


# ── the detector-face panel uses the scene's numbers ─────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_the_detector_panel_draws_the_scenes_projected_volume_outline(name):
    """The projected volume edges are the scene's, corner for corner.

    The panel draws the twelve edges of the volume box in detector index
    coordinates.  Each edge's endpoints must be two entries of the scene's
    ``volume_outline_on_detector`` and nothing else, because a viewer that
    recomputed the projection could disagree with the projector.

    The line data is then read back out of the axes, so the drawing is checked
    and not only the array the figure kept.  The lines are in (channel, row)
    order, because that is the panel's horizontal and vertical axis, while the
    scene reports (row, channel).  The twelve edges are drawn as one polyline
    with a row of NaN between edges, so the non-finite separators are dropped
    before the comparison.
    """
    scene, figure = build_figure(name, view_index=2)
    try:
        outline = scene.view(2).volume_outline_on_detector
        drawn = figure.detector_volume_edges
        assert drawn.shape == (12, 2, 2)
        for index, (first, second) in enumerate(VOLUME_BOX_EDGES):
            assert np.array_equal(drawn[index, 0], outline[first])
            assert np.array_equal(drawn[index, 1], outline[second])

        drawn_points = []
        for line in figure.ax_detector.get_lines():
            if line.get_color() != COLORS['volume']:
                continue
            xdata, ydata = line.get_xdata(), line.get_ydata()
            drawn_points.extend(zip(np.asarray(ydata), np.asarray(xdata)))
        drawn_points = np.asarray(drawn_points, dtype=np.float64)
        drawn_points = drawn_points[np.isfinite(drawn_points).all(axis=1)]
        assert drawn_points.size > 0
        for corner in outline:
            distance = np.min(np.hypot(drawn_points[:, 0] - corner[0],
                                       drawn_points[:, 1] - corner[1]))
            assert distance < 1e-9, f'corner {corner} was not drawn'
    finally:
        close(figure)


def test_the_detector_face_draws_the_region_of_reconstruction():
    """The panel draws the shape the fit statement is about, and names it.

    The fit statement tests the region-of-reconstruction cylinder when the mask
    is on, so the detector face draws that cylinder's two projected rims.  A
    scan with no mask has no cylinder and gets no such line.
    """
    scene, figure = build_figure('cone flat', view_index=2)
    try:
        labels = [text.get_text()
                  for text in figure.ax_detector.get_legend().get_texts()]
        assert 'region of reconstruction' in labels
        drawn = [line for line in figure.ax_detector.get_lines()
                 if line.get_label() == 'region of reconstruction']
        assert len(drawn) == 1
        # The line holds both rims, joined by a row of NaN.
        points = np.stack([np.asarray(drawn[0].get_ydata(), dtype=np.float64),
                           np.asarray(drawn[0].get_xdata(),
                                      dtype=np.float64)], axis=1)
        points = points[np.isfinite(points).all(axis=1)]
        outline = scene.view(2).ror_outline_on_detector.reshape(-1, 2)
        for rim_point in outline:
            distance = np.min(np.hypot(points[:, 0] - rim_point[0],
                                       points[:, 1] - rim_point[1]))
            assert distance < 1e-9, f'rim point {rim_point} was not drawn'
    finally:
        close(figure)

    # The translation model turns the mask off, so it has no region to draw.
    _, figure = build_figure('translation', view_index=2)
    try:
        labels = [text.get_text()
                  for text in figure.ax_detector.get_legend().get_texts()]
        assert 'region of reconstruction' not in labels
    finally:
        close(figure)


# ── the display convention: negative z is up ─────────────────────────────────

def display_point(axes, horizontal, vertical):
    """Where a 2D panel draws one point, in display coordinates.

    Display coordinates run from the bottom left of the figure, so a larger
    second coordinate is higher on the screen.

    Args:
        axes: the panel.
        horizontal, vertical (float): the point, in the panel's own two
            coordinates.

    Returns:
        ndarray: the display position, (2,).
    """
    return np.asarray(axes.transData.transform((horizontal, vertical)),
                      dtype=np.float64)


def display_point_3d(axes, point):
    """Where the 3D panel draws one object-frame point, in display coordinates.

    The 3D panel projects a point with its own camera matrix, which the first
    draw builds, so the caller must have drawn the figure.
    """
    from mpl_toolkits.mplot3d import proj3d
    flat = proj3d.proj_transform(float(point[0]), float(point[1]),
                                 float(point[2]), axes.M)
    return np.asarray(axes.transData.transform(flat[:2]), dtype=np.float64)


def test_the_panels_draw_negative_z_up_y_left_and_row_0_at_the_top():
    """The panels draw the three directions the display convention asks for.

    Each check is made in display coordinates, so it is the drawn position of a
    point and not the order of the axes limits.  The limits are checked as well:
    an axis that increases to the left or downward holds its limits in
    decreasing order.

    Three directions are measured.  Larger z is drawn lower, in the side view
    and in the 3D view in both zoom states, because the zoom replaces the 3D
    panel's limits.  Larger y is drawn further left in the top and side views,
    which is what puts the source on the left of both panels and the detector on
    the right, as in the reference figure of Balke et al. (2018).  And a larger
    detector row index is drawn lower, which is how ``imshow`` shows one view of
    a sinogram.
    """
    scene, figure = build_figure('cone flat', view_index=2)
    try:
        top = figure.ax_top
        assert top.get_xlim()[0] > top.get_xlim()[1], 'y must run left'
        assert top.get_ylim()[0] > top.get_ylim()[1], 'x must run down'
        side = figure.ax_side
        assert side.get_xlim()[0] > side.get_xlim()[1], 'y must run left'
        assert side.get_ylim()[0] > side.get_ylim()[1], 'z must run down'
        detector = figure.ax_detector
        assert detector.get_xlim()[0] < detector.get_xlim()[1]
        assert detector.get_ylim()[0] > detector.get_ylim()[1], 'row 0 on top'

        low = display_point(figure.ax_side, 0.0, -10.0)
        high = display_point(figure.ax_side, 0.0, 10.0)
        assert high[1] < low[1], 'z must increase downward in the side view'

        figure.figure.canvas.draw()
        for zoom in ('scan', 'volume'):
            figure.set_zoom(zoom)
            figure.figure.canvas.draw()
            low = display_point_3d(figure.ax_3d, (0.0, 0.0, -3.0))
            high = display_point_3d(figure.ax_3d, (0.0, 0.0, 3.0))
            assert high[1] < low[1], f'z must increase downward, zoom {zoom}'

        for axes in (figure.ax_top, figure.ax_side):
            near = display_point(axes, 50.0, 0.0)
            far = display_point(axes, -50.0, 0.0)
            assert near[0] < far[0], 'y must increase to the left'
        # The top view's other axis is x, which increases downward.
        above = display_point(figure.ax_top, 0.0, -50.0)
        below = display_point(figure.ax_top, 0.0, 50.0)
        assert below[1] < above[1], 'x must increase downward'

        first = display_point(figure.ax_detector, 0.0, 0.0)
        last = display_point(figure.ax_detector, 0.0,
                             scene.num_det_rows - 1.0)
        assert last[1] < first[1], 'row 0 must be at the top'
        right = display_point(figure.ax_detector,
                              scene.num_det_channels - 1.0, 0.0)
        assert right[0] > first[0], 'the channel index must run right'
    finally:
        close(figure)


def test_the_source_travels_counterclockwise_in_the_top_view():
    """The source's travel arc turns counterclockwise on the screen.

    The object turns counterclockwise about z seen from +z, so in a drawing
    that holds the object fixed the source turns the other way, and the top
    view sees that from -z.  The two reversals cancel, so the source's arc
    reads counterclockwise on the screen and the object's own rotation reads
    clockwise.  The test measures the arc's signed area about the rotation
    axis in display coordinates, which is positive for a counterclockwise
    turn.
    """
    scene, figure = build_figure('cone flat', view_index=2)
    try:
        arc = scene.view(2).rotation_direction_arc
        columns = list(TOP_PANEL_COLUMNS)
        drawn = np.stack([display_point(figure.ax_top, *point[columns])
                          for point in arc])
        axis = display_point(figure.ax_top, 0.0, 0.0)
        spokes = drawn - axis[None, :]
        area = float(np.sum(spokes[:-1, 0] * spokes[1:, 1]
                            - spokes[1:, 0] * spokes[:-1, 1]))
        assert area > 0.0, 'the source travel must read counterclockwise'
    finally:
        close(figure)


# ── the data overlays ────────────────────────────────────────────────────────

#: The configuration the sinogram tests paint on, and the view, the row, and
#: the channel of the one bright pixel they paint.  The pixel sits away from
#: the middle of the detector in both directions, so a placement that mirrored
#: or transposed the array could not land on it by accident.
SINOGRAM_CONFIG = 'cone flat'
BRIGHT_VIEW = 5
BRIGHT_ROW = 7
BRIGHT_CHANNEL = 30

#: How close to the brightest pixel a pixel must be, in summed red, green, and
#: blue out of 765, to count as part of the bright block.  The block is drawn
#: in one flat color, so the tolerance only absorbs rounding.
BRIGHTNESS_TOLERANCE = 6

#: A pixel of a projected phantom counts as lit above this fraction of the
#: sinogram's largest value.  The bound separates a pixel the projector gave
#: some mass from a pixel it left at zero, so any small fraction serves.
LIT_FRACTION = 1e-3

#: How far the lit part of a painted sinogram may sit from the projected rims
#: of the region of reconstruction, in detector pixels.  The two numbers
#: differ, and ``test_the_painted_sinogram_lands_inside_the_projected_rims``
#: says why.
RIM_ROW_TOLERANCE = 1.0
RIM_CHANNEL_TOLERANCE = 2.0

#: The configuration the silhouette tests draw.  Its reconstruction has a
#: nonzero ``recon_slice_offset`` and three different voxel pitches, so a
#: drawing that ignored any of them would put the patch somewhere else.  It is
#: also a parallel-type geometry, whose panels span a few volume widths instead
#: of a whole source-detector distance, so a patch of a few voxels covers
#: several pixels on the screen.
SILHOUETTE_CONFIG = 'multiaxis'

#: How many voxels on a side the silhouette tests fill at a corner of the
#: volume.  One voxel is not enough to measure.  A corner voxel is drawn under
#: the volume box's own outline and, at voxel (0, 0, 0), under the marker that
#: names that voxel; in this configuration those two artists leave one pixel of
#: a single voxel showing in the top view, and in a cone geometry, whose panels
#: span the whole scan, one voxel is not a pixel wide to begin with.
SILHOUETTE_BLOCK = 2

#: How far apart two renderings of one pixel must be, in red, green, or blue,
#: for that pixel to count as changed.  Only rounding separates two renderings
#: of the same picture, so anything above a few counts is the overlay.
PIXEL_CHANGE_FLOOR = 4


def bright_sinogram(scene):
    """A sinogram of zeros with one bright pixel, for the placement tests."""
    values = np.zeros(scene.sinogram_shape, dtype=np.float32)
    values[BRIGHT_VIEW, BRIGHT_ROW, BRIGHT_CHANNEL] = 1.0
    return values


def rendered_rgb(figure):
    """The figure drawn, as an array of red, green, and blue values.

    Returns:
        ndarray: (height, width, 3) of int16.  Its first row is the top of the
        figure, which is the opposite of the display coordinates the axes use.
    """
    canvas = figure.figure.canvas
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].astype(np.int16)


def box_mask(shape, box):
    """The rendered pixels inside one display box, as a boolean array.

    The rendered array's first row is the top of the figure while a display box
    measures its height from the bottom, so the rows are flipped here.  A box
    whose limits come from an inverted axis holds them in decreasing order, so
    each pair is read as its smaller and its larger value.

    Args:
        shape (tuple): the rendered array's shape.
        box (Bbox): the display box.

    Returns:
        ndarray: (height, width) of bool.
    """
    height, width = shape[:2]
    horizontal = np.arange(width)[None, :] + 0.5
    vertical = height - np.arange(height)[:, None] - 0.5
    inside_x = ((horizontal >= min(box.x0, box.x1))
                & (horizontal <= max(box.x0, box.x1)))
    inside_y = ((vertical >= min(box.y0, box.y1))
                & (vertical <= max(box.y0, box.y1)))
    return inside_x & inside_y


def pixels_to_data(axes, height, rows, columns):
    """Where rendered pixels sit in one panel's own two coordinates.

    Half a pixel is added in each direction, so the position is the center of
    the pixel and not its corner.

    Args:
        axes: the panel.
        height (int): the rendered array's height.
        rows, columns (ndarray): the pixels' indices into that array.

    Returns:
        ndarray: (N, 2), each row one pixel's position in data coordinates.
    """
    display = np.stack([columns + 0.5, height - rows - 0.5], axis=1)
    return np.asarray(axes.transData.inverted().transform(display))


def changed_in_panel(figure, axes, before, after):
    """Where two renderings of the figure differ inside one panel.

    Returns:
        (ndarray, ndarray): the rows and the columns of the pixels that differ.
    """
    differs = np.abs(after - before).max(axis=2) > PIXEL_CHANGE_FLOOR
    window = axes.get_window_extent(figure.figure.canvas.get_renderer())
    inside = differs & box_mask(differs.shape, window)
    return np.nonzero(inside)


def test_the_bright_sinogram_pixel_is_drawn_at_its_row_and_channel():
    """The painted pixel lands on the screen where its row and channel are.

    This is the orientation gate.  The figure is rendered, the brightest block
    of pixels is found in the image's own area, and its display position is
    converted back to data coordinates through the panel's inverted
    ``transData``.  Those coordinates must be the channel and the row the array
    holds the bright value at, so the panel's inverted row axis is part of what
    is measured.

    Two areas are left out of the search.  The panel's background is white and
    so is the bright pixel, so only the image's own area is searched; the
    legend's box is white as well, so it is taken out of that area.

    The image itself is checked first.  Its extent puts array element (r, c) at
    data coordinates channel c and row r, which is what makes the panel's
    inverted row axis draw row 0 at the top.  A view change replaces the data,
    so after ``set_view`` the image holds the bright pixel at the row and
    channel it was painted at.  And the color scale is the whole array's, so it
    does not move from one view to the next.
    """
    scene = build_scene(SINOGRAM_CONFIG)
    values = bright_sinogram(scene)
    figure = GeometryFigure(scene, view_index=1, sinogram=values)
    try:
        image = figure._sinogram_image
        assert image.origin == 'upper'
        assert list(image.get_extent()) == [-0.5,
                                            scene.num_det_channels - 0.5,
                                            scene.num_det_rows - 0.5, -0.5]
        # The view built is not the bright one, so this view is all zeros.
        assert float(np.max(image.get_array())) == 0.0

        figure.set_view(BRIGHT_VIEW)
        drawn = np.asarray(image.get_array())
        assert drawn.shape == (scene.num_det_rows, scene.num_det_channels)
        assert (np.unravel_index(int(np.argmax(drawn)), drawn.shape)
                == (BRIGHT_ROW, BRIGHT_CHANNEL))
        assert image.get_clim() == (float(np.min(values)),
                                    float(np.max(values)))

        rendered = rendered_rgb(figure)
        renderer = figure.figure.canvas.get_renderer()
        image_box = figure._sinogram_image.get_window_extent(renderer)
        legend = figure.ax_detector.get_legend()
        searched = (box_mask(rendered.shape, image_box)
                    & ~box_mask(rendered.shape,
                                legend.get_window_extent(renderer)))
        brightness = np.where(searched, rendered.sum(axis=2), -1)
        rows, columns = np.nonzero(
            brightness >= brightness.max() - BRIGHTNESS_TOLERANCE)
        assert rows.size > 0, 'no bright pixel was drawn'
        data = pixels_to_data(figure.ax_detector, rendered.shape[0],
                              rows, columns)
        channel, row = data.mean(axis=0)
        assert round(float(channel)) == BRIGHT_CHANNEL
        assert round(float(row)) == BRIGHT_ROW
    finally:
        close(figure)


def test_the_painted_sinogram_lands_inside_the_projected_rims():
    """A forward-projected phantom lights the detector where the rims are.

    The phantom is one inside the region of reconstruction and zero outside,
    and ``model.forward_project`` turns it into the sinogram the panel paints.
    The panel draws the projected rims of that same region over the image, so
    the lit part of the image and the rims are two accounts of one shape: one
    from the projector and one from the scene.  The test compares their extents
    in rows and in channels, in every view.

    The extents are compared edge to edge.  A lit pixel covers the half pixel
    on each side of its index, so the lit extent runs from the lowest lit index
    less a half to the highest plus a half, and the rims are already continuous
    coordinates.

    The two tolerances differ, and the reason is where the mass sits relative
    to the rims.  The rims run through the centers of the outermost voxels,
    which is the ellipse mbirtorch's own mask uses, and the material of those
    voxels reaches half a voxel further, which is 0.91 of a detector channel in
    this scan.  The projector's footprint then spreads that mass by up to half
    a channel more, and any pixel with some mass counts as lit here, so the lit
    channel edge lies outside the rims by up to about 1.4 channels.  Along the
    rows the cylinder already spans the outer faces of the voxels, so only the
    footprint's spread remains.  At a threshold of a tenth of the largest value
    the lit channel edge and the rims agree to 0.1 channel on average, which is
    the measurement of the rims' semi-axes.
    """
    import torch
    import mbirtorch

    model = probe.build_model(CONFIGS_BY_NAME[SINOGRAM_CONFIG])
    scene = GeometryScene.from_model(model)
    phantom = np.zeros(scene.recon_shape, dtype=np.float32)
    phantom[mbirtorch.get_2d_ror_mask(scene.recon_shape)] = 1.0
    sinogram = np.asarray(model.forward_project(torch.tensor(phantom)))
    floor = LIT_FRACTION * float(np.max(sinogram))

    figure = GeometryFigure(scene, sinogram=sinogram)
    try:
        for view_index in range(scene.num_views):
            figure.set_view(view_index)
            drawn = np.asarray(figure._sinogram_image.get_array())
            lit = drawn > floor
            assert lit.any(), f'view {view_index} painted nothing'
            rims = np.asarray(scene.view(view_index).ror_outline_on_detector)
            rims = rims.reshape(-1, 2)
            for axis, tolerance, name in ((0, RIM_ROW_TOLERANCE, 'row'),
                                          (1, RIM_CHANNEL_TOLERANCE,
                                           'channel')):
                indices = np.flatnonzero(lit.any(axis=1 - axis))
                for edge, expected in ((indices.min() - 0.5,
                                        rims[:, axis].min()),
                                       (indices.max() + 0.5,
                                        rims[:, axis].max())):
                    assert abs(float(edge) - float(expected)) <= tolerance, (
                        f'view {view_index}: the lit {name} edge is at '
                        f'{edge}, and the rims reach {expected}')
    finally:
        close(figure)


#: The scan the subsampling tests paint on, and the block of ones they paint.
#: The detector is 2000 channels across, which is sixteen times the size the
#: painted sinogram keeps, and the block is the same size as that stride, so
#: one kept sample lands inside it.
LARGE_SINOGRAM_SHAPE = (4, 300, 2000)
LARGE_RECON_SHAPE = (20, 20, 10)
LARGE_BLOCK_VIEW = 2
LARGE_BLOCK_ROW = 160
LARGE_BLOCK_CHANNEL = 1600
LARGE_BLOCK_SIZE = 16


def large_sinogram_model():
    """A cone-beam scan whose detector is far wider than the painted size."""
    import mbirtorch
    angles = np.linspace(0.0, np.pi, LARGE_SINOGRAM_SHAPE[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(LARGE_SINOGRAM_SHAPE, angles,
                                    source_detector_dist=4000.0,
                                    source_iso_dist=2000.0,
                                    compile_mode='off')
    model.set_params(recon_shape=LARGE_RECON_SHAPE, no_warning=True)
    return model


def large_sinogram():
    """Zeros of the large scan's shape with one block of ones in one view."""
    values = np.zeros(LARGE_SINOGRAM_SHAPE, dtype=np.float32)
    values[LARGE_BLOCK_VIEW,
           LARGE_BLOCK_ROW:LARGE_BLOCK_ROW + LARGE_BLOCK_SIZE,
           LARGE_BLOCK_CHANNEL:LARGE_BLOCK_CHANNEL + LARGE_BLOCK_SIZE] = 1.0
    return values


def accelerator_device():
    """The name of a device that is not the host, or None when there is none.

    The tests that check that an array is read where it lives need somewhere
    other than the host to put it.
    """
    import torch
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return None


def test_a_large_sinogram_is_subsampled_for_display():
    """A detector wider than the painted size is sampled every s-th pixel.

    The stride is the same in both directions, so the kept pixels stay square,
    and it is the smallest one that brings the larger detector dimension under
    ``SINOGRAM_DISPLAY_PIXELS``.  A numpy array is viewed rather than copied,
    which is what keeps a large sinogram from being held twice.  The image's
    extent puts each kept sample at the detector row and channel it was taken
    from, so the bright block is drawn where the array holds it.
    """
    model = large_sinogram_model()
    values = large_sinogram()
    figure = GeometryFigure(model, view_index=0, sinogram=values,
                            widgets=False)
    try:
        stride = LARGE_SINOGRAM_SHAPE[2] // geometry_figure.SINOGRAM_DISPLAY_PIXELS + 1
        assert figure._sinogram_stride == stride == 16
        image = figure._sinogram_image
        assert np.asarray(image.get_array()).shape == (19, 125)
        assert np.shares_memory(figure._sinogram, values)
        assert list(image.get_extent()) == [-8.0, 1992.0, 296.0, -8.0]

        figure.set_view(LARGE_BLOCK_VIEW)
        drawn = np.asarray(image.get_array())
        sample = np.unravel_index(int(np.argmax(drawn)), drawn.shape)
        assert sample == (10, 100)
        # Where that sample's center sits in the panel's own coordinates,
        # read from the extent the image was given.
        left, right, bottom, top = image.get_extent()
        channel = left + (sample[1] + 0.5) * (right - left) / drawn.shape[1]
        row = top + (sample[0] + 0.5) * (bottom - top) / drawn.shape[0]
        assert channel == pytest.approx(LARGE_BLOCK_CHANNEL)
        assert row == pytest.approx(LARGE_BLOCK_ROW)
    finally:
        close(figure)


def test_a_sinogram_tensor_is_subsampled_where_it_lives():
    """A sinogram on a device is sliced there and paints the same picture.

    The subsample is taken before the array leaves the device, so only the
    kept pixels are brought to the host.  What the panel then paints has to be
    what the host array paints.
    """
    import torch

    device = accelerator_device()
    if device is None:
        pytest.skip('no device other than the host is available')
    values = large_sinogram()
    model = large_sinogram_model()
    on_host = GeometryFigure(model, view_index=LARGE_BLOCK_VIEW,
                             sinogram=values, widgets=False)
    on_device = GeometryFigure(model, view_index=LARGE_BLOCK_VIEW,
                               sinogram=torch.tensor(values, device=device),
                               widgets=False)
    try:
        assert on_device._sinogram_stride == on_host._sinogram_stride
        assert np.array_equal(np.asarray(on_device._sinogram_image.get_array()),
                              np.asarray(on_host._sinogram_image.get_array()))
        assert (list(on_device._sinogram_image.get_extent())
                == list(on_host._sinogram_image.get_extent()))
    finally:
        close(on_host)
        close(on_device)


@pytest.mark.parametrize('corner', ('low', 'high'))
def test_the_silhouette_is_drawn_where_its_voxels_are(corner):
    """The silhouette of a corner block lands on that corner of the box.

    A block of voxels is filled at one corner of the reconstruction, and the
    top view and the side view must each draw it where the scene puts that
    block.  The check is on the rendered figure, so it covers both panels'
    inverted axes, the three voxel pitches, and the volume's slice offset: the
    patch's position on the screen is converted back to data coordinates and
    compared with the block's center, which ``GeometryScene.voxel_centers``
    gives at the block's fractional index.  At the low corner it is also
    compared with the marker of voxel (0, 0, 0), which each panel already
    draws.

    The patch is found by rendering the figure twice, once without the
    silhouette and once with it, and taking the pixels that differ.  Matching
    the fill color instead does not work here: the fill is the volume's color
    over the panel's white background, and the antialiased edge of the volume
    box's own outline, which is that color drawn solid, produces the same
    pixels along every edge of the box.
    """
    scene = build_scene(SILHOUETTE_CONFIG)
    rows, cols, slices = scene.recon_shape
    block = SILHOUETTE_BLOCK
    recon = np.zeros(scene.recon_shape, dtype=np.float32)
    if corner == 'low':
        recon[:block, :block, :block] = 1.0
        center = [(block - 1) / 2.0] * 3
    else:
        recon[rows - block:, cols - block:, slices - block:] = 1.0
        center = [rows - (block + 1) / 2.0, cols - (block + 1) / 2.0,
                  slices - (block + 1) / 2.0]
    expected = scene.voxel_centers([center])[0]
    # The pitches are indexed the way a panel's columns are, as (x, y, z).
    pitches = (scene.delta_voxel, scene.delta_voxel_row,
               scene.delta_voxel_slice)

    figure = GeometryFigure(scene, view_index=2)
    try:
        before = rendered_rgb(figure)
        figure.set_recon(recon)
        after = rendered_rgb(figure)
        view = scene.view(figure.view_index)
        for axes, columns, name in ((figure.ax_top, TOP_PANEL_COLUMNS, 'top'),
                                    (figure.ax_side, SIDE_PANEL_COLUMNS,
                                     'side')):
            pixel_rows, pixel_columns = changed_in_panel(figure, axes,
                                                         before, after)
            assert pixel_rows.size > 0, f'nothing was drawn in the {name} view'
            data = pixels_to_data(axes, after.shape[0], pixel_rows,
                                  pixel_columns)
            drawn = data.mean(axis=0)
            allowed = np.asarray([pitches[column] for column in columns])
            wanted = expected[list(columns)]
            assert np.all(np.abs(drawn - wanted) <= allowed), (
                f'the {name} view drew the patch at {drawn}, and the block '
                f'sits at {wanted}')
            if corner == 'low':
                marker = np.asarray(view.voxel0_center)[list(columns)]
                assert np.all(np.abs(drawn - marker) <= allowed)
    finally:
        close(figure)


# ── the phantom's outline ────────────────────────────────────────────────────

#: The block of voxels the outline tests fill, as a half-open index range per
#: voxel index: the rows, the columns, and the slices.  The block sits away
#: from every face of the reconstruction, so its outline is a closed rectangle
#: in each projected panel and the array's own edge is not part of it.
OUTLINE_BLOCK = ((2, 5), (3, 7), (1, 4))


def block_phantom(scene, block=OUTLINE_BLOCK):
    """An array of the scan's recon shape with one block of ones in it."""
    values = np.zeros(scene.recon_shape, dtype=np.float32)
    (row0, row1), (col0, col1), (slice0, slice1) = block
    values[row0:row1, col0:col1, slice0:slice1] = 1.0
    return values


def block_corners(scene, block=OUTLINE_BLOCK):
    """The block's two outer corners in object coordinates, as (3,) arrays.

    The corners are the scene's own voxel centers half a voxel outside the
    first and the last voxel of the block in each direction, which is where the
    material of those voxels ends.
    """
    first = [float(low) - 0.5 for low, _ in block]
    last = [float(high) - 1 + 0.5 for _, high in block]
    corners = scene.voxel_centers([first, last])
    return corners[0], corners[1]


def outlines_in(figure, axes):
    """The phantom's outline artists in one panel."""
    return [artist for panel, artist in figure._recon_outlines
            if panel is axes]


def outline_line_3d(figure):
    """The one line artist the phantom's outline uses in the 3D panel."""
    lines = outlines_in(figure, figure.ax_3d)
    assert len(lines) == 1
    return lines[0]


def outline_parts_3d(figure):
    """The phantom's sections, one array per section, as the 3D panel draws
    them.

    The order is the one ``GeometryFigure._support_outline_parts`` builds: the
    sections in the order of their positions along the axis they lie across.
    A section is itself several segments with a row of NaN between them, and
    the sections are joined by one more such row, so the drawn line cannot be
    cut back into sections by its NaN rows alone.  The figure's own list is
    read instead, and the drawn line is checked against it here.
    """
    parts = figure._recon_outline_parts
    drawn = np.stack(outline_line_3d(figure).get_data_3d(), axis=1)
    joined = geometry_figure._joined(parts)
    assert drawn.shape == joined.shape
    assert np.array_equal(np.isnan(drawn), np.isnan(joined))
    assert np.allclose(finite_rows(drawn), finite_rows(joined))
    return parts


def finite_rows(points):
    """The rows of an array that are finite in every column."""
    points = np.asarray(points, dtype=np.float64)
    return points[np.isfinite(points).all(axis=1)]


def test_a_block_phantoms_outline_is_its_sections_and_its_rectangle():
    """A block phantom's outline is one rectangle per plane, in 3D and on top.

    The block spans three rows, four columns, and three slices, so the rows
    and the slices tie for the thinnest direction and z wins the tie.  Every
    one of the three slices the block reaches is drawn, because three is fewer
    than the count the panel would spread over a longer extent.  Each section
    is the block's rectangle at that slice's center, drawn through the outer
    faces of its voxels, so its four corners are the scene's own voxel centers
    half a voxel outside the block.  The three voxel pitches differ in this
    geometry and its volume is offset in z, so an outline built from anything
    but the scene's own voxel centers would land somewhere else.

    The top view draws the xy plane, so the same block projects to a rectangle
    whose sides are the block's outer faces in y and in x.  Every vertex of
    that outline has to be on the rectangle, which is checked to half a voxel
    pitch in each direction, and the outline has to reach every side of it.
    """
    scene = build_scene(SILHOUETTE_CONFIG)
    figure = GeometryFigure(scene, recon=block_phantom(scene))
    try:
        assert figure._recon_section_axis == 'z'
        assert figure._recon_section_indices == [1, 2, 3]
        parts = outline_parts_3d(figure)
        assert len(parts) == 3

        low, high = block_corners(scene)
        corners = np.array([[x, y] for x in (low[0], high[0])
                            for y in (low[1], high[1])])
        for part, index in zip(parts, figure._recon_section_indices):
            drawn = finite_rows(part)
            # Every point of a section sits in that slice's own plane.
            plane = scene.voxel_centers([[0.0, 0.0, float(index)]])[0]
            assert np.allclose(drawn[:, 2], plane[2])
            # The rectangle is four segments, so each corner is drawn twice,
            # once as the end of each segment that meets there.
            unique = np.unique(np.round(drawn[:, :2], 9), axis=0)
            assert unique.shape == (4, 2)
            for corner in corners:
                gap = np.linalg.norm(unique - corner[None, :], axis=1)
                assert float(np.min(gap)) < 1e-9, (
                    f'corner {corner} was not drawn')

        lines = outlines_in(figure, figure.ax_top)
        assert len(lines) == 1
        drawn = np.stack([np.asarray(lines[0].get_xdata(), dtype=np.float64),
                          np.asarray(lines[0].get_ydata(), dtype=np.float64)],
                         axis=1)
        drawn = drawn[np.isfinite(drawn).all(axis=1)]
        assert drawn.shape[0] >= 8, 'a rectangle is four segments'

        # The panel puts y across the screen and x down it, and the pitches are
        # indexed the same way, as (x, y, z).
        columns = list(TOP_PANEL_COLUMNS)
        pitches = np.array([scene.delta_voxel, scene.delta_voxel_row,
                            scene.delta_voxel_slice])
        edges = np.stack([low[columns], high[columns]])
        half = 0.5 * pitches[columns]
        # A vertex of the rectangle sits on one of the four sides and between
        # the other two, which is one condition per axis.
        on_a_side = np.abs(drawn[:, None, :] - edges[None, :, :]).min(axis=1)
        between = ((drawn >= edges[0][None, :] - half[None, :])
                   & (drawn <= edges[1][None, :] + half[None, :]))
        assert np.all(np.any(on_a_side <= half[None, :], axis=1))
        assert np.all(between)
        # And the outline reaches every side.
        assert np.all(np.abs(drawn.min(axis=0) - edges[0]) <= half)
        assert np.all(np.abs(drawn.max(axis=0) - edges[1]) <= half)
    finally:
        close(figure)


#: How far the phantom's projected outline may reach past the lit part of its
#: painted sinogram, and how far short of it the outline may fall, in detector
#: pixels.  ``test_the_projected_outline_follows_the_cube_phantoms_shadow``
#: says why the two numbers differ and why each is the right one.
SHADOW_OUTSIDE_TOLERANCE = 1.0
SHADOW_INSIDE_TOLERANCE = 2.0


def convex_hull(points):
    """The convex hull of a set of points in a plane, counterclockwise.

    This is Andrew's monotone chain.  The points are sorted, then walked once
    forward for the lower chain and once backward for the upper one, and a
    point is dropped whenever the last three turn the wrong way.

    Args:
        points (ndarray): (N, 2).

    Returns:
        ndarray: (M, 2), the hull's vertices in order.
    """
    ordered = np.asarray(points, dtype=np.float64)
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]

    def turn(first, second, third):
        return ((second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0]))

    def chain(sequence):
        kept = []
        for point in sequence:
            while len(kept) >= 2 and turn(kept[-2], kept[-1], point) <= 0.0:
                kept.pop()
            kept.append(point)
        # The last point of one chain is the first of the other.
        return kept[:-1]

    return np.array(chain(ordered) + chain(ordered[::-1]))


def inside_hull(points, hull, tolerance=1e-6):
    """Whether each point lies inside a counterclockwise convex hull.

    Args:
        points (ndarray): (N, 2).
        hull (ndarray): (M, 2), the hull's vertices counterclockwise.
        tolerance (float, optional): how far outside an edge a point may lie,
            in the units the points are in.  The default absorbs rounding.

    Returns:
        ndarray: (N,) of bool.
    """
    points = np.asarray(points, dtype=np.float64)
    flags = np.ones(points.shape[0], dtype=bool)
    for index in range(hull.shape[0]):
        first = hull[index]
        second = hull[(index + 1) % hull.shape[0]]
        edge = second - first
        # The cross product is the distance to the edge's line times the
        # edge's length, so dividing by that length gives a distance.
        side = (edge[0] * (points[:, 1] - first[1])
                - edge[1] * (points[:, 0] - first[0]))
        flags &= side / np.linalg.norm(edge) >= -tolerance
    return flags


def test_the_projected_outline_follows_the_cube_phantoms_shadow():
    """The outline on the detector face bounds the shadow the projector paints.

    The phantom is mbirtorch's cube phantom, and ``model.forward_project``
    turns it into the sinogram the panel paints.  The panel draws that
    phantom's own outline over the image, so the lit part of the image and the
    outline are two accounts of one shape: one from the projector and one from
    the scene.  The test compares their extents in rows and in channels, in
    every view.

    The extents are compared edge to edge.  A lit pixel covers the half pixel
    on each side of its index, so the lit extent runs from the lowest lit index
    less a half to the highest plus a half, and the outline is already in
    continuous index coordinates.

    The two tolerances differ, because the outline is not symmetric about the
    shadow.  Across the plane of a section the outline runs through the outer
    faces of the phantom's voxels, so the material it encloses ends where the
    outline does; the projector's footprint then spreads that material's mass
    by up to half a pixel, and any pixel with some mass counts as lit here,
    which puts the lit edge up to half a pixel further out again.  Those two
    half pixels are the one pixel the outline may reach past the lit region,
    and it does not use even that.

    Along the axis the sections lie across, a section sits at its plane's
    center and the phantom's material reaches half a voxel further on each
    side, so the section projects inside the shadow.  How far inside depends
    on the magnification, which the half voxel changes, and on the view, which
    turns that half voxel between the source direction and the detector's.  In
    this scan that is up to about one and a half pixels, so two pixels is the
    shortfall allowed.

    The outline also has to project inside the volume box's projected outline,
    because the phantom lies inside the volume.  That is what lets the panel
    draw this line without splitting it at the detector's edge: a part of it
    that leaves the grid leaves it inside the volume box's outline, which is
    split and already carries the overshoot color.
    """
    import torch
    import mbirtorch

    model = probe.build_model(CONFIGS_BY_NAME[SINOGRAM_CONFIG])
    scene = GeometryScene.from_model(model)
    phantom = np.asarray(mbirtorch.gen_cube_phantom(scene.recon_shape))
    sinogram = np.asarray(model.forward_project(torch.tensor(phantom)))
    floor = LIT_FRACTION * float(np.max(sinogram))

    figure = GeometryFigure(scene, sinogram=sinogram, recon=phantom)
    try:
        for view_index in range(scene.num_views):
            figure.set_view(view_index)
            drawn = np.asarray(figure._sinogram_image.get_array())
            lit = drawn > floor
            assert lit.any(), f'view {view_index} painted nothing'

            line = figure._recon_detector_line
            outline = np.stack(
                [np.asarray(line.get_xdata(), dtype=np.float64),
                 np.asarray(line.get_ydata(), dtype=np.float64)], axis=1)
            outline = outline[np.isfinite(outline).all(axis=1)]
            assert outline.shape[0] > 0
            # The panel's axes are (channel, row), and the lit extents below
            # are taken the way the array is indexed, as (row, channel).
            indexed = outline[:, ::-1]
            for axis, name in ((0, 'row'), (1, 'channel')):
                indices = np.flatnonzero(lit.any(axis=1 - axis))
                # ``outside`` is how far the outline reaches past the lit
                # region, and it is negative where the outline falls short.
                for edge, reached, sign in (
                        (indices.min() - 0.5, indexed[:, axis].min(), 1.0),
                        (indices.max() + 0.5, indexed[:, axis].max(), -1.0)):
                    outside = sign * (float(edge) - float(reached))
                    assert -SHADOW_INSIDE_TOLERANCE <= outside, (
                        f'view {view_index}: the lit {name} edge is at '
                        f'{edge}, and the outline reaches {reached}')
                    assert outside <= SHADOW_OUTSIDE_TOLERANCE, (
                        f'view {view_index}: the lit {name} edge is at '
                        f'{edge}, and the outline reaches {reached}')

            box = np.asarray(scene.view(view_index).volume_outline_on_detector,
                             dtype=np.float64)
            hull = convex_hull(box[:, ::-1])
            assert np.all(inside_hull(outline, hull)), (
                f'view {view_index}: the outline left the volume box')
    finally:
        close(figure)


# ── the sections at scale ────────────────────────────────────────────────────

#: How many blobs the point-budget test fills, how many voxels a blob is on a
#: side, and the reconstruction it fills them in.  The blobs are far too many
#: for their outlines to fit the budget at full resolution, so the sections
#: have to be coarsened, and four slices give four sections whatever the
#: coarsening.
BLOB_COUNT = 600
BLOB_SIZE = 3
BLOB_RECON_SHAPE = (300, 300, 4)


#: How many voxels across the ball of the chunked-pass tests is.  A ball has a
#: support that changes from one plane to the next in all three directions, so
#: a pass that mixed up an axis would give a different answer.
BALL_RADIUS = 3.0


def ball_phantom(scene):
    """A ball of ones at the middle of a scan's reconstruction."""
    rows, cols, slices = scene.recon_shape
    i, j, k = np.meshgrid(np.arange(rows) - (rows - 1) / 2.0,
                          np.arange(cols) - (cols - 1) / 2.0,
                          np.arange(slices) - (slices - 1) / 2.0,
                          indexing='ij')
    inside = i ** 2 + j ** 2 + k ** 2 <= BALL_RADIUS ** 2
    return np.where(inside, 1.0, 0.0).astype(np.float32)


def blob_phantom():
    """A field of small blobs at seeded positions, and those positions.

    Each blob is a square of voxels running through every slice, so every
    section holds the same field and the blobs are what the sections have to
    show.

    Returns:
        (ndarray, ndarray): the phantom, and the (row, column) index of each
        blob's first voxel.
    """
    generator = np.random.default_rng(0)
    values = np.zeros(BLOB_RECON_SHAPE, dtype=np.float32)
    corners = np.stack(
        [generator.integers(0, BLOB_RECON_SHAPE[0] - BLOB_SIZE, BLOB_COUNT),
         generator.integers(0, BLOB_RECON_SHAPE[1] - BLOB_SIZE, BLOB_COUNT)],
        axis=1)
    for row, column in corners:
        values[row:row + BLOB_SIZE, column:column + BLOB_SIZE, :] = 1.0
    return values, corners


def distance_to_outline(points, outline):
    """How far each point lies from the nearest point of a drawn outline.

    The outline is a set of segments, and what a reader sees is the whole
    segment and not only its ends, so the distance is to the segment.  Both
    are taken in the xy plane.

    Args:
        points (ndarray): (N, 3) or (N, 2).
        outline (ndarray): the outline, (M, 3), with a row of NaN between one
            segment and the next.

    Returns:
        ndarray: (N,) the distance from each point to the outline.
    """
    outline = np.asarray(outline, dtype=np.float64)
    finite = np.isfinite(outline).all(axis=1)
    pairs = finite[:-1] & finite[1:]
    starts = outline[:-1][pairs][:, :2]
    ends = outline[1:][pairs][:, :2]
    along = ends - starts
    length = np.maximum((along ** 2).sum(axis=1), 1e-12)
    gaps = []
    for point in np.asarray(points, dtype=np.float64)[:, :2]:
        fraction = np.clip(((point[None, :] - starts) * along).sum(axis=1)
                           / length, 0.0, 1.0)
        nearest = starts + fraction[:, None] * along
        gaps.append(float(np.min(np.linalg.norm(nearest - point[None, :],
                                                axis=1))))
    return np.asarray(gaps)


def test_a_field_of_small_blobs_stays_under_the_point_budget():
    """Many small blobs are coarsened until their outlines fit the budget.

    Six hundred blobs would draw far more points than the panel may hold, so
    every section is coarsened.  What the coarsening must keep is where the
    blobs are: the outline of a section still has to pass close to each of
    them, within about the width of one coarse block.  The footer says the
    outline was coarsened, because a coarsened outline covers the support
    rather than following it exactly.
    """
    import mbirtorch

    angles = np.linspace(0.0, np.pi, 8, endpoint=False)
    model = mbirtorch.ConeBeamModel((8, 32, 64), angles,
                                    source_detector_dist=800.0,
                                    source_iso_dist=400.0, compile_mode='off')
    model.set_params(recon_shape=BLOB_RECON_SHAPE, no_warning=True)
    values, corners = blob_phantom()
    figure = GeometryFigure(model, recon=values, widgets=False)
    try:
        scene = figure.scene
        assert figure._recon_section_axis == 'z'
        assert len(figure._recon_section_indices) == BLOB_RECON_SHAPE[2]
        parts = figure._recon_outline_parts
        drawn = sum(finite_rows(part).shape[0] for part in parts)
        assert drawn <= geometry_figure.PHANTOM_OUTLINE_POINT_BUDGET
        assert all(factor > 1 for factor in figure._recon_section_factors)

        pitch = max(scene.delta_voxel, scene.delta_voxel_row)
        centers = scene.voxel_centers(
            np.column_stack([corners[:, 0] + (BLOB_SIZE - 1) / 2.0,
                             corners[:, 1] + (BLOB_SIZE - 1) / 2.0,
                             np.zeros(BLOB_COUNT)]))
        for part, factor in zip(parts, figure._recon_section_factors):
            gaps = distance_to_outline(centers, part) / pitch
            assert float(np.max(gaps)) <= factor + 1, (
                f'a blob sits {np.max(gaps)} voxel pitches from the outline '
                f'of a section coarsened by {factor}')

        # The footer wraps its lines, so the words are joined up again here.
        body = ' '.join(artist.get_text() for artist in figure.ax_text.texts)
        coarsest = max(figure._recon_section_factors)
        assert f'outline coarsened by {coarsest}' in ' '.join(body.split())
    finally:
        close(figure)


def test_the_chunked_pass_matches_the_direct_computation(monkeypatch):
    """Reading the array a chunk at a time gives the whole array's answer.

    The chunk size is taken down to a few elements, so the pass runs in many
    chunks, and the three projections then have to be the ones a direct
    computation over the whole array gives.  The threshold is a tenth of the
    largest absolute value, which the chunked pass also finds.
    """
    monkeypatch.setattr(geometry_figure, 'OVERLAY_CHUNK_ELEMENTS', 50)
    scene = build_scene(SILHOUETTE_CONFIG)
    values = ball_phantom(scene)
    figure = GeometryFigure(scene, recon=values, widgets=False)
    try:
        level = figure._recon_threshold
        assert level == pytest.approx(0.1 * float(np.max(np.abs(values))))
        support = np.abs(values) > level
        assert np.array_equal(figure._recon_projections['xy'],
                              support.any(axis=2))
        assert np.array_equal(figure._recon_projections['yz'],
                              support.any(axis=1))
        assert np.array_equal(figure._recon_projections['xz'],
                              support.any(axis=0))
    finally:
        close(figure)


def test_a_phantom_tensor_is_thresholded_where_it_lives():
    """A reconstruction on a device is read there and draws the same outline.

    The array is thresholded and reduced where it lives, so only the
    projections and the section planes cross to the host.  What the panel
    draws from a device tensor has to be what it draws from the host array.
    """
    import torch

    device = accelerator_device()
    if device is None:
        pytest.skip('no device other than the host is available')
    scene = build_scene(SILHOUETTE_CONFIG)
    values = ball_phantom(scene)
    on_host = GeometryFigure(scene, recon=values, widgets=False)
    on_device = GeometryFigure(scene, recon=torch.tensor(values,
                                                         device=device),
                               widgets=False)
    try:
        assert on_device._recon_threshold == pytest.approx(
            on_host._recon_threshold)
        for name in ('xy', 'yz', 'xz'):
            assert np.array_equal(on_device._recon_projections[name],
                                  on_host._recon_projections[name])
        assert (on_device._recon_section_axis
                == on_host._recon_section_axis)
        assert (on_device._recon_section_indices
                == on_host._recon_section_indices)
        assert len(on_device._recon_outline_parts) == len(
            on_host._recon_outline_parts)
        for here, there in zip(on_host._recon_outline_parts,
                               on_device._recon_outline_parts):
            assert np.array_equal(np.isnan(here), np.isnan(there))
            assert np.allclose(finite_rows(here), finite_rows(there))
    finally:
        close(on_host)
        close(on_device)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
