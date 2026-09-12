"""Matplotlib drawing of an mbirtorch scan geometry, in five panels.

What this module is for.  ``geometry_scene.GeometryScene`` turns a tomography
model's numbers into points and polylines.  This module draws those primitives
so that a user can see the geometry: where the source and the detector sit,
where the reconstruction volume sits, which way the gantry turns, and whether
the volume projects inside the detector.  It owns no geometry of its own.
Every position it draws comes from the scene, and every number it prints comes
from ``GeometryScene.derived_quantities``.

The five panels.  ``GeometryFigure`` builds one figure with a 2 by 3 grid.  The
left column holds a 3D view for orientation.  The middle and right columns hold
a top view in the xy plane, a side view in the yz plane, a detector face in
index coordinates, and a text panel of derived numbers.  The four drawing
panels show one view of the scan at a time, and ``set_view`` moves to another.
The point where the central ray meets the detector is called the detector iso,
and the center of the detector grid is called the detector center.  The
detector face names its markers in a legend, and that legend sits in the band
under the panel rather than inside it.  Inside the panel it covered the
projected outlines and a painted sinogram.

The picture drawn.  The object stays fixed and the source and the detector
carry the view's motion, which is how a scanner user thinks of a scan.  The
projector code does the opposite, and the scene handles the conversion.

The display convention: negative z is the top of every panel.  Array indices
increase from top to bottom when an array is printed or shown, and the slice
index k runs along +z, so a drawing with -z at the top shows the volume the way
its array is indexed and the way ``imshow`` shows one slice.  The object frame
itself does not change.  It stays right-handed with +z along the increasing
slice index, and every primitive the scene reports is unchanged.  Only the
presentation changes: the three 2D panels invert the axes that the convention
turns around, and the 3D panel rolls its camera.  The module constant
:data:`Z_UP_SIGN` is the sign of the object-frame z that points up the screen,
and each panel reads it in one place, so setting it to +1 draws +z up again.

The screen orientation of each panel.  The orientation follows the figure
"Parallel Beam Geometry - top view" of Balke et al., Separable Models for
cone-beam MBIR Reconstruction, 2018.  In it the beam runs from left to
right, with the source on the left and the detector on the right.  Every panel
here follows that picture.  The top view is the xy plane seen from -z, with y
increasing to the left and x increasing downward.  The side view is the yz
plane seen from +x, with y increasing to the left and z increasing downward.
The detector face has the channel index increasing to the right and row 0 at
the top, which is the view from the source toward the detector.  In all three
the source of a view at angle 0 is on the left.

The 3D camera.  The default camera sits 25 degrees above the drawing's top,
which is the -z side, and 40 degrees off the +x axis toward -y, and it is
rolled by 180 degrees so that -z is up.  From there the 3D picture agrees with
the top view.  The y axis runs to the left, the x axis runs down the screen,
-z is up, and the source of a view at angle 0 sits at the back on the left.
The 40 degrees off the x axis matter.  Near an azimuth of zero the eye looks
along x, so the x axis and the z axis both run up and down the screen and the
detector stands edge on.  A detector that is upright is then hard to tell from
one lying flat.  The camera can be dragged with the mouse, and ``set_view``
keeps whatever camera the user has set.

Why that camera is rolled.  The roll is what inverts this panel's z, in place
of a reversed pair of z limits.  Reversed limits invert the z axis too, and
matplotlib draws them by flipping the z axis of the world it projects.  That
flip mirrors the picture.  The depth matplotlib sorts its artists by, and the
faces of the axes box it draws, then belong to an eye on the other side of the
object, and the x and y tick labels move over the panel's title.  The roll
turns the same view upside down instead.  The panel then stays a picture taken
from one viewpoint, and its z ticks read negative at the top.

The widgets.  A slider under the panels steps through the views, and six
toggles sit beside it in two rows of three.  The top row changes the drawing of
the geometry.  Its first toggle turns the source's path over all views on and
off.  Its second switches the 3D panel between the whole scan and a close view
of the volume, because a 12 ALU volume drawn to scale in a 200 ALU scan is
twenty pixels wide.  Its third turns the angle-0 reference on and off.  The
bottom row turns each of the three overlays on and off: the sinogram, the
phantom, and the comparison.  An overlay the figure does not hold has a toggle
all the same, so that an array added later has its toggle ready.  The widgets
follow the slice viewer of ``mbirtorch/viewer.py``: an integer-stepped
``Slider`` with ``drawon`` off, and ``CheckButtons`` for the toggles.

The labels.  The source and the detector carry a short text label in the 3D
view, the top view, and the side view.  The detector iso carries one in the 3D
view and the top view, and the detector's pixel-0 marker carries one in the top
view.  Each label names the element it sits beside, which one panel's legend
cannot do for the other panels.  A label moves with the element it names, so
it is a moving artist like that element.  The labels of one panel are placed
so that no two of them overlap and none reaches outside the panel, which a
test checks by measuring them.

The angle-0 reference.  The source and the detector are drawn a second time,
where they sit when the view action is the identity.  That position comes from
``GeometryScene.reference_view``.  These artists do not move with the slider,
so they show the current view against a fixed one, and the dotted arrow through
them is the direction the geometry projects at angle 0.  They are drawn dotted
and partly transparent in the 3D view and the top view, and the third toggle
turns them off.  The top view names them with a caption in its upper left
corner, because the ray they lie on crosses the middle of that panel, where the
labels of the source and the detector are.

How a redraw stays fast.  Every artist is created once and its data is replaced
in place, so a view change calls ``set_data`` and never ``clear``.  The panel
limits are computed from a sample of views and then held, so the ticks, the
grids, the legends, and the text block belong to a background that a view
change does not touch.  A view change restores that background, redraws only
the artists that moved, and blits, which is the same partial-redraw idea the
slice viewer uses.  Anything that changes the background instead, such as a
toggle or a new comparison, repaints the whole figure.  The timing test in
the test suite measures a slider step on an 1800-view helical scan against a
100 ms gate.

The comparison overlay.  A second geometry can be drawn over the first in one
color with dashed lines, from a second scene, a second model, or a dictionary
of parameter overrides.  This is the calibration use: a vendor geometry against
the same geometry with an estimated offset.  The text panel then lists the
parameters that differ, which are what the user changed, and counts the derived
quantities that differ.  A second window holds the whole table, one row per
difference, with the primary geometry's value beside the comparison's.  The
table has no cap of its own, because the window's height follows its row count.

The data overlays.  Two arrays can be drawn beside the geometry, and both are
optional.  A sinogram is painted on the detector face, one view at a time, in
gray with row 0 at the top, so that the object's shadow can be read against the
projected outlines drawn over it.  A reconstruction or a phantom is drawn as a
silhouette in the volume box of the top view and the side view: the voxels
whose absolute value is above a threshold, projected along the axis the panel
does not draw.  The sinogram moves with the slider and the silhouette does not,
because the object is the thing this drawing holds fixed.

Neither array is held whole.  A detector wider than
:data:`SINOGRAM_DISPLAY_PIXELS` is subsampled for display, on the device the
sinogram sits on, so a large sinogram transfers only the pixels the panel
paints.  The reconstruction is read a chunk of slices at a time, each chunk is
thresholded where it lives, and only the three projections the silhouettes need
and the planes the 3D outline draws reach the host.

The phantom's outline.  The silhouette's fill is light, so that the lines drawn
over it stay readable, and against a busy panel it can be hard to see.  Each of
the two projected panels therefore also draws the outline of the support, as a
solid line through the outer faces of the voxels in it.  The 3D panel draws
dashed sections of the support instead, which is the one thing that panel shows
of the phantom.  A section is the outline of the support in one plane, and the
sections lie across the axis the support is thinnest along, so that each of
them cuts the support where it is widest.  Sections say more than a bounding
box does, because a phantom need not be a box and need not be one connected
piece: mbirtorch's cube phantom is a rectangle that steps sideways from one
slice to the next, and a head phantom is a shell with structure inside it.  At
most :data:`PHANTOM_SECTION_COUNT` sections are drawn and the 3D panel's legend
says how many planes they are out of, because a reader has to know that the
panel shows a few planes and not the whole shape.  A section whose outline
would be very long is drawn from a coarsened mask, so that a field of small
blobs stays a drawing the panel can hold.  The detector face draws those same
sections projected onto it.  The projected outline says where the phantom's
shadow falls, against the outlines of the volume box and of the region of
reconstruction already drawn there.  All of these belong to the phantom's
toggle, and they are removed when the phantom is.

Import discipline.  This module imports numpy and the matplotlib base package
at import time, and nothing else.  ``pyplot``, the widgets, and the 3D toolkit
are imported inside :func:`_load_pyplot`, on the first figure construction.
Importing this module therefore never resolves a matplotlib backend and never
touches a GUI toolkit, which is the rule ``mbirtorch/viewer.py`` follows.

Colors.  One color per element, listed in :data:`COLORS`, is used in every
panel where the element appears.  The two index markers are the ones to look at
first: the marker at detector pixel (row 0, channel 0) and the marker at voxel
(0, 0, 0) show the index orientation of the data, and a mirrored channel order
or a wrong offset sign is visible from those two alone.
"""

import os
import textwrap

import numpy as np
import matplotlib  # base package only; no GUI toolkit is touched at import

from .geometry_scene import GeometryScene, required_parameter_names

__all__ = ['GeometryFigure', 'geometry_viewer', 'COLORS',
           'DEFAULT_ELEVATION_DEG', 'DEFAULT_AZIMUTH_DEG',
           'VOLUME_BOX_EDGES', 'ZOOM_MODES', 'BLIT_BACKENDS',
           'Z_UP_SIGN', 'TOP_PANEL_COLUMNS', 'SIDE_PANEL_COLUMNS',
           'ISO_NAME', 'CENTER_NAME', 'PHANTOM_NAME']


# --- names used in every panel ---

#: What the point where the central ray meets the detector is called.  Every
#: panel uses this name.
ISO_NAME = 'detector iso'

#: What the center of the detector grid is called.  The two detector offsets
#: move this point away from the detector iso.
CENTER_NAME = 'detector center'

#: What the array given as ``recon`` is called in the widget row and in the 3D
#: panel's legend.  The array is a reconstruction or a phantom, and one word
#: has to serve for both.  The word is "phantom" because the example and the
#: tests draw a phantom, and because a reader who has a reconstruction reads
#: "phantom" as the object it shows.  The methods keep the name ``recon``,
#: which is the name of the constructor argument and of mbirtorch's own shape.
PHANTOM_NAME = 'phantom'


# --- the display convention ---

#: The sign of the object-frame z that points up the screen.  The value -1
#: draws -z at the top of every panel, which is the convention this module
#: follows; see the module docstring.  The value +1 inverts no axis and draws
#: +z up.  Each panel reads this constant in one place: the three 2D panels
#: through :func:`_screen_pair`, which orders a pair of axes limits, and the 3D
#: panel through :func:`_camera_roll_deg`, which rolls its camera.  The
#: geometry itself carries no sign flip.
Z_UP_SIGN = -1


def _z_up_is_negative():
    """Whether -z points up the screen; see :data:`Z_UP_SIGN`."""
    return Z_UP_SIGN < 0


def _top_view_title():
    """The top panel's title, which names the side it is seen from."""
    side = 'left' if _z_up_is_negative() else 'right'
    return ('Top view, the xy plane, seen from -z (the top)\n'
            f'y increases to the {side}')


def _side_view_title():
    """The side panel's title, which names the side it is seen from."""
    return 'Side view, the yz plane, seen from +x'


def _detector_view_title():
    """The first line of the detector panel's title, naming the row order.

    Row 0 at the top is the view from the source toward the detector.  Row 0
    at the bottom, which +1 gives, is the same grid seen from behind the
    detector, so that title names the row order alone.
    """
    if _z_up_is_negative():
        return 'Detector face, seen from the source, row 0 at the top'
    return 'Detector face, row 0 at the bottom'


def _camera_roll_deg():
    """How far the 3D camera is rolled, so that -z is at the top of the panel.

    The 3D panel turns its z axis around with the camera and not with its axes
    limits.  Reversed limits would do it too, but matplotlib draws reversed
    limits by flipping the z axis of the world it projects, and that mirrors
    the picture: the depth it sorts artists by, and the faces of the axes box
    it draws, then belong to an eye on the other side.  A roll of 180 degrees
    turns the same physical view upside down instead, which leaves the picture
    a picture of the geometry and puts the negative z ticks at the top.
    """
    return CAMERA_ROLL_DEG if _z_up_is_negative() else 0.0


def _convention_note():
    """The text panel's words naming the display convention."""
    return '-z up.' if _z_up_is_negative() else '+z up.'


def _screen_step(delta):
    """Which way a step along a projected panel's axis points on the screen.

    Both projected panels turn both of their axes around under the display
    convention, so one rule serves the horizontal and the vertical axis of
    each.

    Args:
        delta (float): a step in the object coordinate the panel puts on that
            axis.

    Returns:
        int: 1 when the step points to the right or up the screen, and -1 when
        it points to the left or down.  The two answers swap with
        :data:`Z_UP_SIGN`.
    """
    forward = 1 if float(delta) >= 0.0 else -1
    return -forward if _z_up_is_negative() else forward


def _screen_bottom(values):
    """The end of a run of values that a panel draws at the bottom of the
    screen.

    A panel's vertical axis increases downward under the display convention, so
    the largest value is at the bottom.  A label that is to hang below a drawn
    object is placed at this value.
    """
    values = np.asarray(values, dtype=np.float64)
    return float(np.max(values) if _z_up_is_negative() else np.min(values))


def _screen_pair(low, high):
    """One axis's limits, ordered for the display convention.

    Under the convention a panel's axes increase downward or to the left, and
    matplotlib draws such an axis from a pair of limits in decreasing order.
    Which axes those are is stated per panel where the limits are applied.

    Args:
        low, high (float): the axis's limits, in increasing order.

    Returns:
        tuple: the pair, reversed when -z points up the screen.
    """
    if _z_up_is_negative():
        return (float(high), float(low))
    return (float(low), float(high))


#: Which object coordinates the top view puts on its horizontal and its
#: vertical axis, as indices into (x, y, z).  The top view is the xy plane seen
#: from -z, with y across the screen and x down it.
TOP_PANEL_COLUMNS = (1, 0)

#: The same for the side view, which is the yz plane seen from +x, with y
#: across the screen and z down it.
SIDE_PANEL_COLUMNS = (1, 2)

#: Which two of a view's four corner rays bound the plane the top view draws.
#: ``ViewScene.corner_rays`` runs over the detector's corners, and these two
#: sit at the extremes of the channel direction, which is the direction the top
#: view spreads the detector along.
TOP_EDGE_RAYS = (0, 1)

#: The same for the side view, whose plane the two corners at the extremes of
#: the row direction bound.
SIDE_EDGE_RAYS = (0, 3)


# --- appearance ---

#: One color per drawn element.  Every panel uses these, so an element keeps
#: its color across the figure.
COLORS = {
    'source': '#ff7f0e',        # orange
    'detector': '#1f77b4',      # blue
    'rays': '#c0c0c0',          # light gray
    'central_ray': '#111111',   # near black
    'volume': '#9467bd',        # purple
    'ror': '#17becf',           # cyan
    'axis': '#555555',          # gray
    'pixel0': '#e7298a',        # magenta
    'voxel0': '#2ca02c',        # green
    'trajectory': '#8c564b',    # brown
    'overshoot': '#d62728',     # red
    'compare': '#6b8e00',       # dark yellow-green
}

TITLE_FONT_SIZE = 9
LABEL_FONT_SIZE = 8
TICK_FONT_SIZE = 7
ANNOTATION_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6.5
TEXT_PANEL_FONT_SIZE = 7.5
WIDGET_FONT_SIZE = 8

#: Characters per line in the text panel's wrapped sentences.  The panel's
#: aligned "name : value" lines are shorter than this, so this width sets the
#: panel's overall text width.
TEXT_PANEL_WRAP_WIDTH = 52

#: How far to the side a label sits from the point it names, in points.  The
#: vertical gap is chosen per label instead, so that two labels near one point
#: sit on different lines.
LABEL_GAP_POINTS = 6

#: The vertical gap of the labels at the detector iso from that point, in
#: points, and the extra gap the second of them takes so that the two sit on
#: different lines.  One line of the annotation font is about eleven points
#: tall at the sizes this module uses.
ISO_LABEL_GAP_POINTS = 6.0
LABEL_LINE_POINTS = 11.0

#: The vertical gap of the top view's pixel-0 label from its marker, in points.
#: It is larger than the detector label's gap, so that the two sit on different
#: lines when the detector is drawn short against the panel.
PIXEL0_LABEL_GAP_POINTS = 14.0

#: The vertical gap of the top view's "source travel" label from the end of the
#: rotation arc, in points.
ARC_LABEL_GAP_POINTS = 8.0

#: What the two detector offsets are called in the projected panels.  These are
#: short forms of the parameter names ``det_channel_offset`` and
#: ``det_row_offset``.  The full names are twenty-odd characters wide, and in
#: the multiaxis geometry, whose top view spans a few tens of ALU, the channel
#: one reached across the panel and onto the source's marker.  The full names
#: are still printed in the text panel, which is where a reader looks for a
#: parameter by its name.
CHANNEL_OFFSET_LABEL = 'chan offset'
ROW_OFFSET_LABEL = 'row offset'

SOURCE_MARKER_SIZE = 13
INDEX_MARKER_SIZE = 8
INDEX_MARKER_WIDTH = 1.8

#: Color of the two corner rays that bound a 2D panel's plane.  They are drawn
#: darker than the other two so that the fan and the cone read as outlines.
EMPHASIZED_RAY_COLOR = '#8a8a8a'

#: The dash pattern and line width of every comparison line.  The comparison
#: is drawn dashed so that it can be told from the primary geometry in a black
#: and white print as well as by color.
COMPARE_DASHES = (0, (5.0, 2.0))
COMPARE_LINEWIDTH = 1.4

#: The alpha, the dash pattern, and the line width of the angle-0 reference.
#: The reference sits behind the view drawn and must not be read as part of it,
#: so it is dotted and partly transparent.
REFERENCE_ALPHA = 0.45
REFERENCE_DOTS = (0, (1.0, 2.0))
REFERENCE_LINEWIDTH = 1.2

#: The fraction of the reference central ray that its arrowhead spans.  The
#: head is drawn from a short segment at the detector end, which is how the
#: rotation arc's head is drawn.
REFERENCE_ARROW_FRACTION = 0.06

#: Where the angle-0 reference's label sits in the top view, in that panel's
#: axes coordinates.  It is a caption in the upper left corner rather than a
#: label on the ray; see :meth:`GeometryFigure._create_reference_artists`.
REFERENCE_LABEL_CORNER = (0.015, 0.985)

# --- the two data overlays ---

#: The colormap the sinogram is painted with on the detector face.  Gray is
#: the map a sinogram is usually shown in, and it leaves every color in
#: :data:`COLORS` free for the lines drawn over it.
SINOGRAM_COLORMAP = 'gray'

#: The largest number of detector pixels the painted sinogram keeps along
#: either detector axis.  A larger detector is subsampled for display: every
#: s-th row and every s-th channel is kept, with the same stride s in both
#: directions so that the kept pixels stay square, and s the smallest integer
#: that brings the larger detector dimension to this size or under.  The
#: subsample is taken before the array leaves the device that holds it, so a
#: sinogram on a GPU transfers only the kept pixels, and a numpy array is
#: viewed rather than copied.  A feature narrower than the stride, such as a
#: single bright pixel, may fall between the kept samples and is then not
#: shown.
SINOGRAM_DISPLAY_PIXELS = 128

#: The alpha of the reconstruction silhouette's fill.  The fill has to be light
#: enough that the volume box and the rays drawn over it stay readable, and
#: dark enough to be seen against the panel's white background.
RECON_FILL_ALPHA = 0.35

#: The line width of the phantom's outline in each panel that draws it.  In the
#: two projected panels and in the 3D panel the outline is drawn a little
#: thicker than the volume box's own line, which is 1.2 in those panels and 1.0
#: in the 3D panel, so that the two can be told apart where they run close
#: together.  On the detector face it is drawn thinner than the volume box's
#: line, which is 1.2 there, because the phantom's outline always projects
#: inside the volume box's and a thin line is easier to follow over a painted
#: sinogram.
RECON_OUTLINE_LINEWIDTH = 1.6
RECON_OUTLINE_3D_LINEWIDTH = 1.4
RECON_OUTLINE_DETECTOR_LINEWIDTH = 0.9

#: Where the phantom's outline sits in the drawing order of a projected panel.
#: It is above the silhouette's fill, which is at :data:`SILHOUETTE_ZORDER`,
#: and below the lines and the labels of the panel, which matplotlib draws at 2
#: and 3.
RECON_OUTLINE_ZORDER = 1.8

#: The fraction of the largest absolute value a voxel must exceed to belong to
#: the reconstruction's support, when the caller gives no threshold of its own.
DEFAULT_RECON_THRESHOLD_FRACTION = 0.1

#: How many sections of the phantom the 3D panel draws.  The sections lie
#: across the axis along which the support is thinnest, spread evenly from
#: the first plane with support to the last; when that extent is no larger
#: than this count, every plane is drawn.
PHANTOM_SECTION_COUNT = 9

#: What one section is.  'section' outlines the support in one plane, so the
#: outline lies on the object and its projection lies inside the object's
#: shadow.  'slab' outlines everything in the range of planes around the
#: section's plane, so every support voxel appears in some outline, at the
#: price of an outline that can lie outside the object where the object
#: tapers within the slab.
PHANTOM_SECTION_KIND = 'section'
PHANTOM_SECTION_KINDS = ('section', 'slab')

#: Which axis the sections lie across when two axes tie for the thinnest
#: extent, in order of preference.
PHANTOM_SECTION_AXIS_PREFERENCE = ('z', 'y', 'x')

#: The largest number of points the phantom's section outlines may hold in
#: total.  A section whose outline would exceed its share is coarsened: the
#: plane's mask is reduced by blocks of 2, 3, ... voxels, taking a block as
#: in the support when any of its voxels is, until the outline fits.  A
#: small object then becomes one block and stays visible.
PHANTOM_OUTLINE_POINT_BUDGET = 8000

#: How many elements of the reconstruction one chunk of the support
#: computation holds.  The support is never materialized as a whole: the
#: array is read a chunk of slices at a time, each chunk is thresholded where
#: it lives, and only the three projections and the section planes reach the
#: host.
OVERLAY_CHUNK_ELEMENTS = 2 ** 24

#: Where the two overlays sit in the drawing order of their panels.  A patch,
#: such as the detector face's translucent rectangle, is drawn at 1 and a line
#: at 2, so the sinogram covers the rectangle and every line and marker is
#: drawn over the sinogram.  The silhouette sits below the lines of its panel
#: for the same reason and above the panel's background.
SINOGRAM_ZORDER = 1.5
SILHOUETTE_ZORDER = 1.0

#: The keyword that asks matplotlib to hide the parts of a 3D artist that lie
#: outside the axes box.  matplotlib added ``axlim_clip`` in release 3.10, and
#: mbirtorch sets no lower bound on its matplotlib version, so an older release
#: draws the same figure without the clipping.  Every 3D drawing call spreads
#: this dictionary into its keywords.
AXLIM_CLIP = ({'axlim_clip': True}
              if matplotlib.__version_info__ >= (3, 10) else {})

#: The 3D camera, in degrees.  The eye sits 25 degrees above the drawing's top,
#: which is the -z side, and 40 degrees off the +x axis toward -y.  From there
#: y runs to the left and x runs down the screen, as in the top view, and the
#: source of a view at angle 0 sits at the back on the left.  The 40 degrees
#: off the x axis matter: near an azimuth of 0 the x axis and the z axis both
#: run up and down the screen, and the detector stands edge on, so a panel
#: that is upright is hard to tell from one lying flat.  See the module
#: docstring.
DEFAULT_ELEVATION_DEG = -25.0
DEFAULT_AZIMUTH_DEG = -40.0

#: How far the 3D camera is rolled about its own axis, in degrees, to put -z at
#: the top of the panel.  This is the 3D panel's one reading of Z_UP_SIGN; see
#: :func:`_camera_roll_deg`.
CAMERA_ROLL_DEG = 180.0

#: Points sampled around an ellipse when drawing the region of reconstruction.
ELLIPSE_SAMPLES = 65

#: Points sampled along each ray the 3D panel draws.  A ray is a straight
#: segment and two points would draw it, but the 3D panel clips a line point by
#: point at its axes limits, so a segment whose two ends are both outside the
#: limits disappears.  In the volume zoom every ray has both ends outside, and
#: sampling the segment keeps the part that crosses the panel.
RAY_SAMPLES = 33

#: Points sampled along one projected volume edge on the detector face, used
#: only when the edge crosses the detector's boundary and has to be split into
#: an inside part and an overshooting part.
EDGE_CLIP_SAMPLES = 64

#: Fractional margin added around the data of a 2D panel.
PANEL_MARGIN = 0.10

#: The two states of the 3D panel's zoom control.  ``'scan'`` puts the source,
#: the detector, and the volume in one cube.  ``'volume'`` puts a cube around
#: the volume box alone and lets the axes limits clip the rays and the axis.
ZOOM_MODES = ('scan', 'volume')
DEFAULT_ZOOM = 'scan'

#: Width of the ``'volume'`` zoom cube, as a multiple of the volume's largest
#: extent.  Three extents leave the volume filling the middle third of the
#: panel with room for the rotation axis and the nearest rays around it.
ZOOM_VOLUME_WIDTH_FACTOR = 3.0

#: How many views are sampled when the panel limits are computed.  The limits
#: are then held, so that a view change does not move the ticks and the
#: background can be reused.  A view whose content falls outside the limits
#: makes them grow and forces one full redraw.
LIMIT_SAMPLE_VIEWS = 16

#: At most this many differing parameters are listed in the text panel's
#: comparison section.  The panel holds about thirty-six lines of its own font
#: size, and the derived quantities and the footer take twenty-seven of them,
#: so a comparison between two unrelated geometries has to be capped or it
#: would run off the panel.  The cap falls further when the panel turns out to
#: be too short for it; see ``_place_text_blocks``.  The comparison window
#: lists every difference and is capped only by its own height.
MAX_COMPARISON_ENTRIES = 6

#: The comparison window's width in inches, the height of one table row, the
#: room the title takes above the table, the margin on its other three sides,
#: and the largest height the window may have.  The height follows the row
#: count up to that largest height, after which the rows that are left out are
#: counted in a last row.  Ten inches is about the height of a laptop screen.
COMPARE_WINDOW_WIDTH_IN = 7.5
COMPARE_ROW_HEIGHT_IN = 0.28
COMPARE_WINDOW_TITLE_IN = 0.55
COMPARE_WINDOW_MARGIN_IN = 0.25
COMPARE_WINDOW_MAX_HEIGHT_IN = 10.0

#: The comparison window's font size, and the width of one character of a
#: monospace font as a fraction of that size.  matplotlib's default monospace
#: face, DejaVu Sans Mono, advances 0.602 of its size per character, and every
#: character advances the same amount, so the width of a row of the table can
#: be computed from its length.  The window uses that to take the font size
#: down when a row is too long to fit the window's width; a face with a wider
#: advance than this would then reach a little past the margin.
COMPARE_WINDOW_FONT_SIZE = 9.5
MONOSPACE_ADVANCE = 0.602

#: The comparison window's title, and the name its window carries on a desktop
#: so that the user can tell the two windows apart.
COMPARE_WINDOW_TITLE = ('Comparison: primary (solid) against '
                        'comparison (dashed)')
COMPARE_WINDOW_NAME = 'Geometry comparison'

#: The three column headings of the comparison window's table.
COMPARE_TABLE_HEADING = ('name', 'primary (solid)', 'comparison (dashed)')

#: The gap between two columns of the comparison window's table, in characters.
COMPARE_COLUMN_GAP = 2

#: The text panel's font size while a comparison is drawn.  The derived
#: quantities alone nearly fill the panel at ``TEXT_PANEL_FONT_SIZE``, so the
#: whole panel is set smaller to make room for the comparison section.
COMPARING_FONT_SIZE = 6.0

#: The fraction of the text panel's height its three blocks may fill.  The
#: last line's descenders sit below the line the measurement counts, so a
#: block set to the full height reaches a few pixels past the panel.
TEXT_PANEL_FILL = 0.97

#: The smallest font the text panel will shrink to when its blocks are taller
#: than the panel; see ``GeometryFigure._fit_text_font``.  A saved figure's
#: numbers stop being readable below this.  Only the two parallel-type
#: geometries with a comparison drawn reach it: their drawing note is six lines
#: where a cone scan's is two.
TEXT_PANEL_FONT_MINIMUM = 5.0

#: Widget rectangles in figure coordinates, as (left, bottom, width, height).
#: The slider is shorter than the figure is wide, because the six toggles sit
#: beside it.  The toggles are laid out as two rows of three in the right half
#: of the widget row: the three that change the drawing of the geometry on the
#: top row, and the three that turn an overlay on and off below them.  Each
#: toggle draws its label at the middle of its own rectangle, so the two rows
#: are half the height the three toggles used to have and their labels are a
#: row apart.  The top row ends below :data:`DETECTOR_LEGEND_BOTTOM`, which is
#: where the detector face's legend band starts, and every toggle is to the
#: right of the slider, whose place did not change.
SLIDER_RECT = (0.07, 0.045, 0.38, 0.025)
TRAJECTORY_CHECK_RECT = (0.50, 0.052, 0.12, 0.038)
ZOOM_CHECK_RECT = (0.645, 0.052, 0.15, 0.038)
REFERENCE_CHECK_RECT = (0.815, 0.052, 0.15, 0.038)
SINOGRAM_CHECK_RECT = (0.50, 0.010, 0.12, 0.038)
RECON_CHECK_RECT = (0.645, 0.010, 0.15, 0.038)
COMPARE_CHECK_RECT = (0.815, 0.010, 0.15, 0.038)

#: The top of the panel grid is unchanged; its bottom leaves room for the
#: widget row.
GRID_BOTTOM = 0.115

#: Where the detector-face panel's box stops, as a fraction of the figure's
#: height.  The panel keeps the width of its grid cell and stops above the
#: cell's own bottom, which is :data:`GRID_BOTTOM`.  The band that leaves under
#: the panel holds the panel's x label and, below that, the panel's legend.
#: The panel holds an equal aspect by reshaping its box, so a shorter cell
#: costs the panel width only where its box is taller than the cell.  Of the
#: six probe geometries that is the curved cone scan alone, whose detector has
#: 48 rows against 64 channels.  Every other detector is wide enough that its
#: box was already shorter than its cell.
DETECTOR_PANEL_BOTTOM = 0.19

#: Where the detector-face panel's legend sits, as a fraction of the figure's
#: height, and how many columns it takes.  The legend's bottom edge goes here,
#: centered under the panel.  That is above the widget row and below the
#: panel's x label.  Three columns make the legend two rows tall, with or
#: without the entry a comparison adds.  Two rows fit the band and the five or
#: six rows of a single column do not.
DETECTOR_LEGEND_BOTTOM = 0.098
DETECTOR_LEGEND_COLUMNS = 3


def _corner_edges():
    """The twelve edges of the volume box, as index pairs into its corners.

    ``GeometryScene.volume_corners`` returns the eight corners in the order of
    three nested sign loops over x, then y, then z, so corner
    ``4 * ix + 2 * iy + iz`` carries sign ``ix`` in x, ``iy`` in y, and
    ``iz`` in z.  Two corners are joined by an edge when exactly one of those
    three signs differs, which is when their indices differ in one bit.
    """
    edges = []
    for first in range(8):
        for second in range(first + 1, 8):
            if bin(first ^ second).count('1') == 1:
                edges.append((first, second))
    return tuple(edges)


#: The twelve edges of the volume box; see :func:`_corner_edges`.
VOLUME_BOX_EDGES = _corner_edges()

#: A walk around the volume box's footprint in the xy plane, as corner
#: indices.  These four corners share the low z sign, so their x and y values
#: are the four combinations of the box's x and y extents.
_XY_FOOTPRINT_WALK = (0, 2, 6, 4, 0)

#: A walk around the volume box's face in the yz plane, as corner indices.
#: These four corners share the low x sign.
_YZ_FACE_WALK = (0, 1, 3, 2, 0)

#: Backends where the partial-redraw (blit) fast path is used.  These are the
#: two ``mbirtorch/viewer.py`` verifies: Agg for headless runs and TkAgg for
#: the interactive sessions the fast path exists for.  Everywhere else a view
#: change repaints the whole figure.
BLIT_BACKENDS = {'agg', 'tkagg'}

# Backends that have no interactive window; GeometryFigure.show prints a
# line and returns under these.
NONINTERACTIVE_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template', 'cairo'}


# Filled in by _load_pyplot on the first figure construction, so that importing
# this module never resolves a matplotlib backend.
plt = None
Poly3DCollection = None
Rectangle = None
FancyArrowPatch = None
Slider = None
CheckButtons = None
IdentityTransform = None
Bbox = None
ListedColormap = None


def _load_pyplot():
    """Import pyplot, the widgets, and the drawing classes on first use."""
    global plt, Poly3DCollection, Rectangle, FancyArrowPatch
    global Slider, CheckButtons, IdentityTransform, Bbox, ListedColormap
    if plt is not None:
        return
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d.art3d import (
        Poly3DCollection as _Poly3DCollection)
    from matplotlib.patches import (Rectangle as _Rectangle,
                                    FancyArrowPatch as _FancyArrowPatch)
    from matplotlib.widgets import (Slider as _Slider,
                                    CheckButtons as _CheckButtons)
    from matplotlib.transforms import (IdentityTransform as _IdentityTransform,
                                       Bbox as _Bbox)
    from matplotlib.colors import ListedColormap as _ListedColormap
    plt = _plt
    Poly3DCollection = _Poly3DCollection
    Rectangle = _Rectangle
    FancyArrowPatch = _FancyArrowPatch
    Slider = _Slider
    CheckButtons = _CheckButtons
    IdentityTransform = _IdentityTransform
    Bbox = _Bbox
    ListedColormap = _ListedColormap


# --- small formatting and geometry-free drawing helpers ---

def _three_figures(value):
    """A number formatted to three significant figures.

    Zero is added to the value so that a negative zero, which several offsets
    produce, prints as ``0`` rather than as ``-0``.
    """
    return f'{float(value) + 0.0:.3g}'


def _is_tensor(values):
    """Whether an array-like is a torch tensor.

    This module must not import torch, so a tensor is recognized by its
    ``detach`` method rather than by its class.
    """
    return hasattr(values, 'detach')


def _to_host(values):
    """One array-like as a numpy array, whatever kind of array it is.

    The two data overlays are given to the viewer as arrays, and a caller who
    has just reconstructed or projected something holds a torch tensor, which
    may live on a GPU.  A tensor is brought to the host and converted;
    anything else goes through numpy's own conversion, which views a numpy
    array rather than copying it.
    """
    if _is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _absolute(values):
    """The absolute value of an array, computed where the array lives."""
    if _is_tensor(values):
        return values.abs()
    return np.abs(values)


def _any_along(mask, axis):
    """Whether any entry of a boolean array is True along one axis.

    The reduction runs where the array lives, so a mask built on a GPU is
    reduced there and only the result crosses to the host.
    """
    if _is_tensor(mask):
        return mask.any(dim=axis)
    return mask.any(axis=axis)


def _shape_of(values):
    """The shape of an array-like, read without moving it.

    A tensor reports its shape through the attribute, so a tensor on a GPU is
    checked against the scan's shape without a transfer.
    """
    return tuple(np.shape(values))


def _reject_divided(name, values):
    """Refuse an array in the divided device form.

    mbirtorch's multi-device reconstructions can return a ``Shards``
    container, one tensor per device, which has no shape of its own and
    cannot be indexed.  The check is by attributes rather than by class, so
    this module needs no import of the sharding module or of torch.
    """
    if hasattr(values, 'tensors') and hasattr(values, 'placement'):
        raise TypeError(
            f'geometry_viewer does not accept a {name} in the divided device '
            'form.  Gather it to the host first with shards.gather(), then '
            'pass the host array.')


def _as_scene(model_or_scene, **scene_kwargs):
    """A ``GeometryScene``, built from a model when one is given."""
    if isinstance(model_or_scene, GeometryScene):
        if scene_kwargs:
            raise TypeError('Scene options apply only when a model is given; '
                            'a GeometryScene is already built.')
        return model_or_scene
    return GeometryScene.from_model(model_or_scene, **scene_kwargs)


def _ellipse_points(center, semi_axis_x, semi_axis_y, height):
    """Points around an axis-aligned ellipse at one height, (N, 3).

    The region of reconstruction is an elliptic cylinder about the rotation
    axis, and the scene reports it as a center, two semi-axes, and a pair of
    heights.  This turns that description into a polyline.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLES)
    x = center[0] + semi_axis_x * np.cos(angle)
    y = center[1] + semi_axis_y * np.sin(angle)
    z = np.full_like(angle, float(height))
    return np.stack([x, y, z], axis=1)


def _point_3d(point):
    """One point as three one-element arrays, ready for ``set_data_3d``.

    The 3D line artist reads the shape of its own data when any coordinate is
    not finite, so its data has to be arrays and not lists.
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return point[0:1], point[1:2], point[2:3]


def _sampled_segment(start, end, count=RAY_SAMPLES):
    """A straight segment as ``count`` points from ``start`` to ``end``.

    See :data:`RAY_SAMPLES` for why a segment is drawn with more than its two
    ends.
    """
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    fraction = np.linspace(0.0, 1.0, count)[:, None]
    return start[None, :] + fraction * (end - start)[None, :]


def _joined(parts, width=3):
    """Several polylines joined into one array, separated by rows of NaN.

    One line artist per polyline costs one artist to update and one artist to
    draw for every segment of a drawing.  A single artist whose data carries a
    NaN row between polylines draws the same picture, because matplotlib breaks
    a line at a non-finite point.  The volume box, the four corner rays, and
    the twelve projected edges on the detector face are each drawn this way.

    Args:
        parts (sequence): the polylines, each (N, width).
        width (int): the number of columns, 3 for object-frame points and 2
            for detector index pairs.

    Returns:
        ndarray: the joined polyline, (M, width), empty when ``parts`` is.
    """
    pieces = []
    separator = np.full((1, width), np.nan)
    for index, part in enumerate(parts):
        if index:
            pieces.append(separator)
        pieces.append(np.asarray(part, dtype=np.float64).reshape(-1, width))
    if not pieces:
        return np.zeros((0, width))
    return np.concatenate(pieces, axis=0)


def _runs_of_equal_flags(flags):
    """Index slices of consecutive equal entries, overlapping by one entry.

    The overlap makes neighboring runs share a point, so polylines drawn from
    the runs join up instead of leaving a gap.

    Args:
        flags (ndarray): a boolean array.

    Returns:
        list of (slice, bool): each run's slice and the value it holds.
    """
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return []
    breaks = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [flags.size]])
    runs = []
    for start, stop in zip(starts, stops):
        # Extend by one so that this run reaches the next run's first point.
        runs.append((slice(start, min(stop + 1, flags.size)),
                     bool(flags[start])))
    return runs


def _true_runs(flags):
    """The (start, stop) index pairs of the runs of True in a boolean array.

    Each pair is a half-open range, so a run of one entry at index 3 is
    ``(3, 4)``.

    Args:
        flags (ndarray): a boolean array.

    Returns:
        list of (int, int): one pair per run, in order.
    """
    flags = np.asarray(flags, dtype=bool)
    # A False on each end turns every run into one rise and one fall, so the
    # changes come in pairs and each pair is one run.
    padded = np.concatenate([[False], flags, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop))
            for start, stop in zip(changes[::2], changes[1::2])]


def _mask_outline(mask, across, down):
    """The boundary of a mask, as one polyline through its cell edges.

    The boundary is where a cell of the mask meets a cell that is not in the
    mask, and the edge of the array counts as outside.  The polyline is drawn
    through the cell edges themselves, so it encloses every cell of the mask
    rather than running through the cells' centers.  Neighboring edges along
    one boundary line are joined into a single segment, which keeps the
    polyline short for a mask whose boundary is long.

    Args:
        mask (ndarray): (R, C) of bool, indexed first by the panel's vertical
            cell index and then by its horizontal one.
        across (ndarray): the C + 1 cell edges along the horizontal axis, in
            increasing order.
        down (ndarray): the R + 1 cell edges along the vertical axis.

    Returns:
        ndarray: the boundary, (N, 2) as (horizontal, vertical) pairs, with a
        row of NaN between one segment and the next.
    """
    mask = np.asarray(mask, dtype=bool)
    # One ring of False around the mask, so that a cell at the array's edge
    # has a neighbor to differ from.
    padded = np.pad(mask, 1)
    parts = []
    # Boundary column c separates the cells in column c - 1 from those in
    # column c.  The outline runs down that column wherever the two differ.
    for column in range(mask.shape[1] + 1):
        differs = padded[1:-1, column] != padded[1:-1, column + 1]
        for start, stop in _true_runs(differs):
            parts.append(np.array([[across[column], down[start]],
                                   [across[column], down[stop]]]))
    # The same for the boundary rows, across the panel instead of down it.
    for row in range(mask.shape[0] + 1):
        differs = padded[row, 1:-1] != padded[row + 1, 1:-1]
        for start, stop in _true_runs(differs):
            parts.append(np.array([[across[start], down[row]],
                                   [across[stop], down[row]]]))
    return _joined(parts, 2)


# --- reading the reconstruction's support without materializing it ---

def _slice_chunks(shape):
    """The (first, last) slice ranges one chunked pass reads, half open.

    A reconstruction at production size does not fit in memory a second time
    as a boolean support, so the array is read a chunk of slices at a time.
    One chunk holds about :data:`OVERLAY_CHUNK_ELEMENTS` elements, and at
    least one slice whatever the slice's size.

    Args:
        shape (tuple): the reconstruction's shape, as (rows, cols, slices).

    Yields:
        (int, int): each chunk's first slice and the slice after its last.
    """
    per_slice = max(int(shape[0]) * int(shape[1]), 1)
    step = max(1, OVERLAY_CHUNK_ELEMENTS // per_slice)
    for start in range(0, int(shape[2]), step):
        yield start, min(start + step, int(shape[2]))


def _largest_magnitude(values):
    """The largest absolute value of an array, as a float.

    The pass is chunked, so nothing the size of the array is built, and each
    chunk's maximum is taken where the chunk lives.  An empty array has no
    values, and its largest magnitude is zero.
    """
    shape = _shape_of(values)
    if 0 in shape:
        return 0.0
    largest = 0.0
    for first, last in _slice_chunks(shape):
        largest = max(largest,
                      float(_absolute(values[:, :, first:last]).max()))
    return largest


def _support_projections(values, level):
    """The support seen along each of the three axes, in one chunked pass.

    A voxel is in the support when its absolute value is above ``level``.
    Each chunk is thresholded where it lives and reduced there, so only the
    three projections cross to the host.

    Args:
        values (array_like): the reconstruction, as (rows, cols, slices).
        level (float): the absolute value a voxel must exceed.

    Returns:
        dict: ``'xy'`` of shape (rows, cols), ``'yz'`` of shape
        (rows, slices), and ``'xz'`` of shape (cols, slices), each True where
        any voxel along the axis the array does not keep is in the support.
    """
    rows, cols, slices = _shape_of(values)
    projections = dict(xy=np.zeros((rows, cols), dtype=bool),
                       yz=np.zeros((rows, slices), dtype=bool),
                       xz=np.zeros((cols, slices), dtype=bool))
    for first, last in _slice_chunks((rows, cols, slices)):
        mask = _absolute(values[:, :, first:last]) > level
        projections['xy'] |= _to_host(_any_along(mask, 2))
        projections['yz'][:, first:last] = _to_host(_any_along(mask, 1))
        projections['xz'][:, first:last] = _to_host(_any_along(mask, 0))
    return projections


def _support_extent(flags):
    """The first and the last True entry of a 1-D boolean array.

    Returns:
        tuple or None: ``(lo, hi)``, or None when no entry is True.
    """
    indices = np.flatnonzero(np.asarray(flags, dtype=bool))
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1])


def _section_extents(projections):
    """The support's first and last plane along each of the three axes.

    The row index is y, the column index is x, and the slice index is z.

    Args:
        projections (dict): the three projections of
            :func:`_support_projections`.

    Returns:
        dict or None: one ``(lo, hi)`` pair per axis name, or None when the
        support is empty.
    """
    extents = dict(y=_support_extent(projections['xy'].any(axis=1)),
                   x=_support_extent(projections['xy'].any(axis=0)),
                   z=_support_extent(projections['yz'].any(axis=0)))
    if any(extent is None for extent in extents.values()):
        return None
    return extents


def _section_axis(projections):
    """Which axis the sections lie across, given the three projections.

    The sections lie across the axis the support is thinnest along, so that
    each section cuts the support where it is widest and few sections cover
    it.  A tie is broken by :data:`PHANTOM_SECTION_AXIS_PREFERENCE`.

    Args:
        projections (dict): the three projections of
            :func:`_support_projections`.

    Returns:
        str or None: ``'x'``, ``'y'``, or ``'z'``, or None when the support is
        empty.
    """
    extents = _section_extents(projections)
    if extents is None:
        return None
    counts = {axis: high - low + 1 for axis, (low, high) in extents.items()}
    smallest = min(counts.values())
    for axis in PHANTOM_SECTION_AXIS_PREFERENCE:
        if counts[axis] == smallest:
            return axis
    return None


def _section_positions(lo, hi, count, kind):
    """Where the sections sit in the support's extent along one axis.

    Args:
        lo, hi (int): the first and the last plane the support reaches.
        count (int): the largest number of sections to place.
        kind (str): one of :data:`PHANTOM_SECTION_KINDS`.

    Returns:
        list: one ``(position, (first_plane, last_plane))`` pair per section,
        in order.  A section's position is its plane index, and a slab's is
        the middle of its range of planes, which can fall between two planes.
    """
    if kind == 'slab':
        count = min(int(count), hi - lo + 1)
        bounds = np.round(np.linspace(lo, hi + 1, count + 1)).astype(int)
        return [(0.5 * (float(bounds[index]) + float(bounds[index + 1]) - 1.0),
                 (int(bounds[index]), int(bounds[index + 1]) - 1))
                for index in range(count)]
    if hi - lo + 1 <= count:
        planes = np.arange(lo, hi + 1)
    else:
        planes = np.unique(np.round(np.linspace(lo, hi, count)).astype(int))
    return [(int(plane), (int(plane), int(plane))) for plane in planes]


def _section_mask(values, axis, planes, level):
    """One section's support, as a boolean mask on the host.

    Only the planes the section covers are read from the array, and they are
    thresholded and reduced where they live, so a section of a reconstruction
    on a GPU costs one plane-sized transfer.

    The mask's first index runs down the section and its second runs across
    it.  Which voxel index each of those is depends on the axis: for ``'z'``
    they are the row and the column, for ``'y'`` the column and the slice,
    and for ``'x'`` the row and the slice.

    Args:
        values (array_like): the reconstruction, as (rows, cols, slices).
        axis (str): ``'x'``, ``'y'``, or ``'z'``.
        planes (tuple): the first and the last plane of the section,
            inclusive.
        level (float): the absolute value a voxel must exceed.

    Returns:
        ndarray: the mask, (down, across) of bool.
    """
    first, last = int(planes[0]), int(planes[1])
    if axis == 'z':
        block, reduced = values[:, :, first:last + 1], 2
    elif axis == 'y':
        block, reduced = values[first:last + 1, :, :], 0
    else:
        block, reduced = values[:, first:last + 1, :], 1
    return _to_host(_any_along(_absolute(block) > level, reduced))


def _coarsen(mask, factor):
    """A mask reduced by blocks, one entry per block of ``factor`` cells.

    A block is in the reduced mask when any of its cells is in the mask, so
    the reduced mask covers the mask.  The mask is padded with False to a
    whole number of blocks in each direction first.  A factor of 1 changes
    nothing.

    Args:
        mask (ndarray): (R, C) of bool.
        factor (int): the block's width, in cells, in both directions.

    Returns:
        ndarray: the reduced mask, of bool.
    """
    factor = int(factor)
    if factor <= 1:
        return mask
    down = -(-mask.shape[0] // factor)
    across = -(-mask.shape[1] // factor)
    padded = np.zeros((down * factor, across * factor), dtype=bool)
    padded[:mask.shape[0], :mask.shape[1]] = mask
    return padded.reshape(down, factor, across, factor).any(axis=(1, 3))


def _section_outline(mask, factor):
    """The outline of one section's support, in fractional voxel indices.

    The outline runs along the edges of the cells of the coarsened mask, so it
    encloses every cell in the support.  A coarse cell at the far edge covers
    fewer than ``factor`` voxels when the mask does not divide evenly, and the
    outline is held back to the array's own edge there rather than reaching
    past it.

    Args:
        mask (ndarray): the section's mask, (down, across) of bool.
        factor (int): the coarsening factor.

    Returns:
        ndarray: (N, 2) as (across, down), with a row of NaN between one
        segment and the next.
    """
    coarse = _coarsen(mask, factor)
    across = np.minimum(-0.5 + factor * np.arange(coarse.shape[1] + 1),
                        mask.shape[1] - 0.5)
    down = np.minimum(-0.5 + factor * np.arange(coarse.shape[0] + 1),
                      mask.shape[0] - 0.5)
    return _mask_outline(coarse, across, down)


def _budgeted_outline(mask, share):
    """One section's outline, coarsened until it fits the points it may have.

    Factors 1, 2, 3, ... are tried in turn and the first outline that fits is
    taken.  A mask coarsened down to a single cell is as short as an outline
    gets, so that one is taken whether it fits or not, which keeps a small
    object visible.

    Args:
        mask (ndarray): the section's mask, (down, across) of bool.
        share (int): the largest number of points this section may draw.

    Returns:
        (ndarray, int): the outline and the factor it was drawn at.
    """
    factor = 1
    while True:
        outline = _section_outline(mask, factor)
        finite = int(np.isfinite(outline).all(axis=1).sum())
        coarse_down = -(-mask.shape[0] // factor)
        coarse_across = -(-mask.shape[1] // factor)
        if finite <= share or (coarse_down <= 1 and coarse_across <= 1):
            return outline, factor
        factor += 1


def _bounds(points, margin=PANEL_MARGIN):
    """The (low, high) pair per column of ``points``, with a fractional margin.

    Args:
        points (ndarray): (N, 2) or (N, 3).
        margin (float): fraction of the largest extent added on each side.

    Returns:
        tuple: one (low, high) pair per column.
    """
    points = np.asarray(points, dtype=np.float64)
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    pad = margin * float(np.max(high - low))
    if pad <= 0.0:
        pad = 1.0
    return tuple((float(low[index] - pad), float(high[index] + pad))
                 for index in range(points.shape[1]))


def _cube_bounds(points, margin=PANEL_MARGIN):
    """One cubic (low, high) triple that holds ``points``.

    The three axes get the same extent, which is the largest of the three data
    extents.  One ALU is then the same length along x, y, and z, so an angle in
    the drawing is the angle in the geometry.  The cost is empty space along
    the short axes.  The alternative, an axis box shaped like the data, gives a
    long thin tunnel for a cone geometry whose source-detector distance is many
    times the volume's size, and that tunnel is unreadable when the camera
    looks along it.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    center = 0.5 * (low + high)
    extent = float(np.max(high - low)) * (1.0 + 2.0 * margin)
    if extent <= 0.0:
        extent = 1.0
    return tuple((float(center[index] - 0.5 * extent),
                  float(center[index] + 0.5 * extent))
                 for index in range(3))


def _far_corner(view, columns):
    """The detector corner a panel puts farthest from the pixel-0 marker.

    A label on the detector goes here.  The pixel-0 marker sits at one end of
    the detector, so the far end holds a label that neither runs into the
    pixel-0 label nor lands in the middle of the panel, where the rays are.

    Args:
        view (ViewScene): the view's primitives.
        columns (sequence of int): which object coordinates the panel draws,
            as indices into (x, y, z).  The 3D panel draws all three.

    Returns:
        ndarray: the corner, (3,).
    """
    corners = np.asarray(view.detector_corners, dtype=np.float64)
    columns = list(columns)
    pixel0 = np.asarray(view.detector_pixel0, dtype=np.float64)[columns]
    distance = np.linalg.norm(corners[:, columns] - pixel0[None, :], axis=1)
    return corners[int(np.argmax(distance))]


def _place_beside(label, point, side=None, vertical=None):
    """Move a label beside a point in a 2D panel.

    By default the horizontal offset takes the sign that points toward the
    middle of the panel, so that a label on a point at the panel's edge reads
    into the panel and not off it.

    Args:
        label: an annotation whose ``textcoords`` is ``'offset points'``.
        point (sequence): the panel-plane position of the thing it names.
        side (int, optional): 1 to put the label to the right of the point and
            -1 to put it to the left.  The default chooses the side that reads
            into the panel.
        vertical (float, optional): the vertical offset from the point, in
            points.  The default keeps the offset the label was created with.
    """
    if side is None:
        low, high = label.axes.get_xlim()
        past_middle = float(point[0]) > 0.5 * (low + high)
        # A panel whose horizontal axis increases to the left draws a point
        # past the middle of that axis on the left half of the screen, where
        # its label has to read to the right to stay in the panel.
        if label.axes.xaxis_inverted():
            past_middle = not past_middle
        side = -1 if past_middle else 1
    label.set_ha('right' if side < 0 else 'left')
    if vertical is None:
        vertical = float(label.xyann[1])
    else:
        vertical = float(vertical)
        label.set_va('bottom' if vertical >= 0.0 else 'top')
    label.xyann = (side * LABEL_GAP_POINTS, vertical)
    label.xy = (float(point[0]), float(point[1]))


def _within(points, limits):
    """Whether every point lies inside the given per-column limits.

    A pair of limits may come in either order, because a panel drawn with the
    display convention holds the limits of an inverted axis in decreasing
    order.  Each pair is therefore read as its smaller and its larger value
    and not as a low and a high.
    """
    points = np.asarray(points, dtype=np.float64)
    for index, pair in enumerate(limits):
        low, high = min(pair), max(pair)
        column = points[:, index]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        if float(np.min(column)) < low or float(np.max(column)) > high:
            return False
    return True


def _lateral_fit_text(quantities):
    """The lateral half of the fit statement: whether the shape tested stays
    inside the detector's channel range in every view, and by how much it
    misses when it does not.  The shape is named, because the answer depends
    on it: a scan with the region-of-reconstruction mask on is asked about the
    cylinder and a scan without the mask about the box."""
    shape = quantities['fit_shape']
    if quantities['fits_laterally']:
        return f'yes ({shape})'
    over = _three_figures(quantities['worst_channel_overshoot_pixels'])
    return f'no ({shape}, {over} px over)'


def _axial_fit_text(quantities):
    """The axial half of the fit statement.  A scan that does not travel is
    asked whether the shape stays inside the detector's row range in every
    view.  A helical scan is asked whether the detector's coverage, swept over
    the scan, contains the volume's z extent, and the swept range is printed;
    see ``GeometryScene.fit_report``."""
    if quantities['helical_fit_rule']:
        swept = (f"{_three_figures(quantities['swept_z_min'])} to "
                 f"{_three_figures(quantities['swept_z_max'])} ALU swept")
        if quantities['fits_axially']:
            return f'yes ({swept})'
        return f'no (z extent not in {swept})'
    if quantities['fits_axially']:
        return 'yes'
    over = _three_figures(quantities['worst_row_overshoot_pixels'])
    return f'no ({over} px over)'


#: Numbers in a comparison line are printed to this many significant figures,
#: and to ``COMPARISON_LONG_FIGURES`` when three figures make the two values
#: read as equal.  A difference the panel lists has to be a difference the
#: reader can see.
COMPARISON_FIGURES = 3
COMPARISON_LONG_FIGURES = 7


def _value_text(value, figures=COMPARISON_FIGURES):
    """One parameter or derived value, in a form a text line can carry.

    A per-view array is named by its shape rather than printed, because a
    comparison line has room for a number and not for 1800 of them.  A short
    array, such as a shape or one translation vector, is printed in full.
    """
    if value is None:
        return 'not set'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, str):
        return value if len(value) <= 28 else value[:25] + '...'
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return f'{float(array.reshape(-1)[0]) + 0.0:.{figures}g}'
    if array.ndim == 1 and array.size <= 4:
        return ('(' + ', '.join(f'{float(entry) + 0.0:.{figures}g}'
                                for entry in array) + ')')
    return f'array{tuple(int(n) for n in array.shape)}'


def _difference_values(mine, theirs):
    """The two values of one difference, as text.

    The two values are printed to more figures when three make them read as
    the same number, which happens for a quantity that a changed parameter
    moves only a little.  A difference that is shown has to be a difference the
    reader can see.

    Args:
        mine, theirs: the primary geometry's value and the comparison's.

    Returns:
        tuple of str: the two values, in that order.
    """
    left, right = _value_text(mine), _value_text(theirs)
    if left == right:
        left = _value_text(mine, COMPARISON_LONG_FIGURES)
        right = _value_text(theirs, COMPARISON_LONG_FIGURES)
    return left, right


def _difference_text(name, mine, theirs):
    """One comparison line, as ``name: primary -> comparison``."""
    left, right = _difference_values(mine, theirs)
    return f'{name}: {left} -> {right}'


def _derived_difference_sentence(count):
    """The text panel's sentence about the derived quantities that differ.

    The text panel lists the parameters that differ and counts the derived
    quantities, because the window is where the whole table is.

    Args:
        count (int): how many derived quantities differ.

    Returns:
        str: one sentence, which names the window when there is something to
        see there.
    """
    if count == 0:
        return 'no derived quantity differs'
    noun = 'quantity differs' if count == 1 else 'quantities differ'
    return f'{count} derived {noun}; see the comparison window'


def _companion_path(path):
    """Where the comparison window is written beside a saved figure.

    The name gets ``_comparison`` before its extension, so ``scan.png`` gives
    ``scan_comparison.png`` in the same directory.
    """
    stem, suffix = os.path.splitext(os.fspath(path))
    return stem + '_comparison' + suffix


class GeometryFigure:
    """A five-panel matplotlib figure of one scan geometry.

    The figure shows one view of the scan.  :meth:`set_view` moves to another
    view, :meth:`set_show_trajectory` turns the path of the source over all
    views on and off, :meth:`set_zoom` switches the 3D panel between the whole
    scan and the volume, :meth:`set_show_reference` turns the angle-0 reference
    on and off, and :meth:`set_compare` overlays a second geometry.  The slider
    and the six toggles under the panels call the same methods.
    :meth:`set_sinogram` paints a sinogram on the detector face and
    :meth:`set_recon` draws a reconstruction's silhouette in the volume box.

    Each of the three overlays has a toggle of its own as well:
    :meth:`set_show_sinogram`, :meth:`set_show_recon`, and
    :meth:`set_show_compare` hide the artists of one overlay and show them
    again.  Hiding removes nothing, so showing an overlay again costs no
    rebuilding, and an overlay that is hidden keeps whatever array or second
    geometry it was given.

    This class calls no matplotlib window function.  Use :meth:`save` to write
    a file, :meth:`show` to open a window, or
    :func:`~mbirtorch.view_utils.geometry_viewer` to do both steps at once.

    Args:
        model_or_scene: a ``GeometryScene``, or a ``TomographyModel`` from
            which one is built with ``GeometryScene.from_model``.
        view_index (int, optional): the view to draw.  Defaults to 0.
        show_trajectory (bool, optional): whether to draw the source's path
            over all views.  Defaults to False.
        figsize (tuple, optional): the figure size in inches.
        title (str, optional): the figure's suptitle.  Defaults to a line
            naming the geometry kind and the model's shapes.
        elevation_deg, azimuth_deg (float, optional): the initial 3D camera.
        compare (optional): a second geometry to draw over the first.  A
            ``GeometryScene``, a ``TomographyModel``, or a dictionary of
            parameter overrides applied to a copy of this scene's parameters,
            for example ``dict(det_channel_offset=12.5)``.  None draws no
            comparison.
        zoom (str, optional): the 3D panel's zoom, one of ``ZOOM_MODES``.
        show_reference (bool, optional): whether to draw the source and the
            detector at their angle-0 position as a fixed reference.  Defaults
            to True.
        widgets (bool, optional): whether to build the slider and the three
            toggles.  Defaults to True.
        blit (bool, optional): whether a view change may use the partial-redraw
            fast path.  Defaults to True.  False forces a full repaint per
            change, which is slower and is the reference the timing script
            compares against.
        sinogram (array_like, optional): a sinogram of shape
            ``(num_views, num_det_rows, num_det_channels)``, painted on the
            detector face one view at a time.  None (default) paints none.  A
            detector larger than ``SINOGRAM_DISPLAY_PIXELS`` pixels across is
            subsampled for display, on the device that holds the array.
        vmin, vmax (float, optional): the gray scale's ends.  None (default)
            takes the smallest and the largest value of the sinogram as
            displayed.
        recon (array_like, optional): a reconstruction or a phantom of shape
            ``recon_shape``, drawn as a silhouette in the volume box of the top
            view and the side view.  None (default) draws none.  The array is
            read a chunk at a time where it lives, so a tensor on a GPU is
            thresholded there, and the 3D panel draws up to
            ``PHANTOM_SECTION_COUNT`` outlines of the support in planes across
            its thinnest direction, with the legend naming the count.
        recon_threshold (float, optional): the absolute value above which a
            voxel belongs to that silhouette.  None (default) uses
            ``DEFAULT_RECON_THRESHOLD_FRACTION`` of the largest absolute
            value in ``recon``.
        show_sinogram (bool, optional): whether the painted sinogram is drawn.
            Defaults to True.
        show_recon (bool, optional): whether the phantom's silhouette and
            outlines are drawn.  Defaults to True.
        show_compare (bool, optional): whether the comparison's artists are
            drawn in the four drawing panels.  Defaults to True.

    Attributes:
        scene: the ``GeometryScene`` drawn.
        compare_scene: the second ``GeometryScene`` drawn, or None when no
            comparison is drawn.
        figure: the matplotlib ``Figure``.
        compare_figure: the second matplotlib ``Figure``, which tables every
            difference between the two geometries, or None when no comparison
            is drawn.  It is created when a comparison is installed and closed
            when the comparison is removed.
        panel_axes (tuple): the five axes, in the order 3D view, top view, side
            view, detector face, text panel.
        view_slider: the view ``Slider``, or None when the scan has one view or
            ``widgets`` is False.
        trajectory_check, zoom_check, reference_check, sinogram_check,
            recon_check, compare_check: the six ``CheckButtons``, or None when
            ``widgets`` is False.
        detector_volume_edges (numpy.ndarray): the twelve projected volume edges the
            detector-face panel drew, (12, 2, 2), as (row, channel) pairs taken
            from ``ViewScene.volume_outline_on_detector``.
    """

    def __init__(self, model_or_scene, view_index=0, show_trajectory=False,
                 figsize=(15.0, 9.0), title=None,
                 elevation_deg=DEFAULT_ELEVATION_DEG,
                 azimuth_deg=DEFAULT_AZIMUTH_DEG,
                 compare=None, zoom=DEFAULT_ZOOM, show_reference=True,
                 widgets=True, blit=True, sinogram=None, vmin=None, vmax=None,
                 recon=None, recon_threshold=None, show_sinogram=True,
                 show_recon=True, show_compare=True):
        _load_pyplot()
        self.scene = _as_scene(model_or_scene)
        self.compare_scene = None
        self.compare_figure = None
        self._view_index = self._checked_view_index(view_index)
        self._show_trajectory = bool(show_trajectory)
        self._show_reference = bool(show_reference)
        # The three overlay toggles are read while the overlays are installed,
        # so they are set before anything is built.
        self._show_sinogram = bool(show_sinogram)
        self._show_recon = bool(show_recon)
        self._show_compare = bool(show_compare)
        self._zoom = self._checked_zoom(zoom)
        self._elevation_deg = float(elevation_deg)
        self._azimuth_deg = float(azimuth_deg)

        self.enable_blit = bool(blit)
        self._background = None
        self._suspend_blit = False
        self._syncing_widgets = False
        self._slider_region = None

        self._trajectory = None
        self._compare_trajectory = None
        self._reference = None
        self._limits = {}
        self._limit_views = set()
        self.detector_volume_edges = None

        self._quantities = self.scene.derived_quantities()
        self._compare_quantities = None
        self._compare_differences = None
        self._compare_line_limit = None
        self._text_font_size = TEXT_PANEL_FONT_SIZE

        # Every artist, in four groups.  _moving holds the artists a view
        # change updates, as (axes, artist) pairs, because a partial redraw
        # draws an artist through its axes.  _compare_moving and
        # _compare_static hold the comparison overlay, which set_compare
        # creates and removes.  _reference_artists holds the angle-0
        # reference, which never moves and which its toggle hides.
        self._moving = []
        self._compare_moving = []
        self._compare_static = []
        self._reference_artists = []
        self._arrow_3d = None

        # The two data overlays, which _install_sinogram and _install_recon
        # create after the geometry's own artists exist.  _recon_outlines holds
        # every line of the phantom: the outline in each of the two projected
        # panels, the outline in the 3D panel, and the projected outline on the
        # detector face.  The first three are static artists like the
        # silhouette itself, and the last one moves with the view, so it is
        # also in _moving and is named on its own in _recon_detector_line.
        # _recon_outline_parts holds one polyline per section of the 3D
        # outline, which the detector face projects for each view, and the
        # other _recon_section_ entries say which planes those sections are.
        self._sinogram = None
        self._sinogram_stride = 1
        self._sinogram_image = None
        self._recon_projections = None
        self._recon_section_axis = None
        self._recon_section_indices = []
        self._recon_section_total = 0
        self._recon_section_factors = []
        self._recon_images = []
        self._recon_outlines = []
        self._recon_outline_parts = []
        self._recon_detector_line = None
        self._recon_threshold = None
        self._recon_threshold_given = None

        self._build_panels(figsize, title)
        self._create_widgets(widgets)
        self._create_artists()
        self._install_sinogram(sinogram, vmin=vmin, vmax=vmax)
        self._install_recon(recon, recon_threshold)
        if compare is not None:
            self._install_compare(compare)
        self._create_legends()
        self.figure.canvas.mpl_connect('draw_event', self._on_draw_event)
        self._refresh(rebuild_limits=True)

    @classmethod
    def from_model(cls, model, scene_kwargs=None, **kwargs):
        """Build a figure from an mbirtorch model.

        Args:
            model: a ``TomographyModel`` subclass instance.
            scene_kwargs (dict, optional): passed to ``GeometryScene``.
            **kwargs: passed to the constructor.

        Returns:
            GeometryFigure
        """
        scene = GeometryScene.from_model(model, **(scene_kwargs or {}))
        return cls(scene, **kwargs)

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def view_index(self):
        """The view currently drawn."""
        return self._view_index

    @property
    def show_trajectory(self):
        """Whether the source's path over all views is drawn."""
        return self._show_trajectory

    @property
    def zoom(self):
        """The 3D panel's zoom, one of :data:`ZOOM_MODES`."""
        return self._zoom

    @property
    def show_reference(self):
        """Whether the angle-0 reference is drawn."""
        return self._show_reference

    @property
    def show_sinogram(self):
        """Whether a sinogram given to the figure is painted."""
        return self._show_sinogram

    @property
    def show_recon(self):
        """Whether the phantom's silhouette and outlines are drawn."""
        return self._show_recon

    @property
    def show_compare(self):
        """Whether the comparison's artists are drawn in the four panels."""
        return self._show_compare

    def set_view(self, view_index):
        """Draw another view.

        Only the artists that move are redrawn, and the 3D camera and the panel
        limits are kept, so a camera the user has dragged survives the change
        and the ticks do not move under the drawing.

        Args:
            view_index (int): the view to draw.
        """
        self._view_index = self._checked_view_index(view_index)
        self._sync_slider()
        self._refresh()

    def set_show_trajectory(self, flag):
        """Turn the source's path over all views on or off.

        The path is one polyline per panel, whatever the number of views, and
        it is computed once and kept.

        Args:
            flag (bool): whether to draw the path.
        """
        flag = bool(flag)
        if flag == self._show_trajectory:
            return
        self._show_trajectory = flag
        self._sync_trajectory_check()
        for _, artist in self._trajectory_artists():
            artist.set_visible(flag)
        # The comparison's path answers to the comparison's toggle as well, so
        # its visibility is set from both flags.
        self._apply_compare_visibility()
        # The path reaches beyond one view's source position, so the panel
        # limits change with it and the whole figure repaints.
        self._refresh(rebuild_limits=True)

    def set_zoom(self, mode):
        """Switch the 3D panel between the whole scan and the volume.

        In ``'scan'`` the source, the detector, and the volume share one cube.
        In ``'volume'`` the cube is about ``ZOOM_VOLUME_WIDTH_FACTOR``
        volume extents wide and centered on the volume, and the axes limits
        clip the rays and the rotation axis where they leave it.

        Args:
            mode (str): one of ``ZOOM_MODES``.
        """
        mode = self._checked_zoom(mode)
        if mode == self._zoom:
            return
        self._zoom = mode
        self._sync_zoom_check()
        self._refresh(rebuild_limits=True)

    def set_show_reference(self, flag):
        """Turn the angle-0 reference on or off.

        The reference does not move with the slider, so it belongs to the
        background, and turning it off repaints the whole figure.  The
        source-path toggle works the same way.

        Args:
            flag (bool): whether to draw the reference.
        """
        flag = bool(flag)
        if flag == self._show_reference:
            return
        self._show_reference = flag
        self._sync_reference_check()
        for _, artist in self._reference_artists:
            artist.set_visible(flag)
        self._refresh(rebuild_limits=True)

    def set_show_sinogram(self, flag):
        """Turn the painted sinogram on or off.

        Hiding the sinogram hides its image and takes the words "with
        sinogram" out of the detector panel's title.  It removes nothing, so
        the array stays and showing the sinogram again costs no rebuilding.  A
        figure that holds no sinogram takes this call and draws nothing.

        Args:
            flag (bool): whether to paint the sinogram.
        """
        flag = bool(flag)
        if flag == self._show_sinogram:
            return
        self._show_sinogram = flag
        self._sync_sinogram_check()
        self._apply_sinogram_visibility()
        # The title and the footer say what is drawn, and both are settled by
        # the repaint below.
        self._refresh(rebuild_limits=True)

    def set_show_recon(self, flag):
        """Turn the phantom's silhouette and its outlines on or off.

        The two silhouette fills, the outline in each projected panel, the
        outline in the 3D panel, and the projected outline on the detector face
        are hidden together, because they are one drawing of one array.  The
        two legends keep their phantom entry while the phantom is hidden: a
        legend names what the figure holds, and the toggle beside it says
        whether that is drawn.

        Args:
            flag (bool): whether to draw the phantom.
        """
        flag = bool(flag)
        if flag == self._show_recon:
            return
        self._show_recon = flag
        self._sync_recon_check()
        self._apply_recon_visibility()
        # The silhouette is static, so it belongs to the background.
        self._refresh(rebuild_limits=True)

    def set_show_compare(self, flag):
        """Turn the comparison's drawn artists on or off.

        Only the drawing is hidden.  The comparison window stays open and the
        text panel keeps its comparison block, because those are the numbers a
        calibration user is reading; the block's heading says that the drawing
        is hidden.  The comparison's source path also follows the source-path
        toggle, so it is drawn only when both toggles are on.

        Args:
            flag (bool): whether to draw the comparison.
        """
        flag = bool(flag)
        if flag == self._show_compare:
            return
        self._show_compare = flag
        self._sync_compare_check()
        self._apply_compare_visibility()
        self._update_static_text()
        self._refresh(rebuild_limits=True)

    def set_compare(self, compare):
        """Draw a second geometry over the first, or stop drawing one.

        A comparison also opens the window that tables every difference, and
        removing the comparison closes that window.  A new comparison gets a
        window of its own, because the table's size is fixed when the window is
        built.

        Args:
            compare: a ``GeometryScene``, a ``TomographyModel``, a dictionary
                of parameter overrides applied to a copy of this scene's
                parameters, or None to remove the comparison.
        """
        self._remove_compare_artists()
        if self.compare_figure is not None:
            plt.close(self.compare_figure)
            self.compare_figure = None
        self.compare_scene = None
        self._compare_quantities = None
        self._compare_differences = None
        self._compare_trajectory = None
        # A new comparison gets the whole line budget back; the old one may
        # have been cut down to fit.
        self._compare_line_limit = None
        if compare is not None:
            self._install_compare(compare)
        self._create_legends()
        self._update_static_text()
        self._refresh(rebuild_limits=True)

    def set_sinogram(self, sinogram, vmin=None, vmax=None):
        """Paint a sinogram on the detector face, or stop painting one.

        The panel draws the view the slider is on, and a view change replaces
        the image's data.  The gray scale is fixed over the whole array, so
        stepping through the views compares them.

        Args:
            sinogram (array_like): an array of shape
                ``(num_views, num_det_rows, num_det_channels)``, or None to
                remove the sinogram.  A detector larger than
                ``SINOGRAM_DISPLAY_PIXELS`` pixels across is subsampled for
                display, on the device that holds the array.
            vmin, vmax (float, optional): the gray scale's ends.  None
                (default) takes the smallest and the largest value of the
                sinogram as displayed.

        Raises:
            ValueError: if the array's shape is not the scan's sinogram shape.
        """
        self._install_sinogram(sinogram, vmin=vmin, vmax=vmax)
        # The image belongs to the detector panel's background until it is
        # drawn once, and the titles change with it, so the whole figure
        # repaints.
        self._refresh(rebuild_limits=True)

    def set_recon(self, recon, threshold=None):
        """Draw a reconstruction's silhouette in the volume box, or stop.

        The silhouette is the support of the array: the voxels whose absolute
        value is above the threshold.  It is drawn in the top view and the side
        view, and it does not move with the view, because the object is the
        thing this drawing holds fixed.  The 3D panel draws up to
        ``PHANTOM_SECTION_COUNT`` outlines of the support in planes across its
        thinnest direction, and that panel's legend names how many planes were
        drawn against how many the support reaches.

        The array is read a chunk at a time where it lives, so a tensor on a
        GPU is thresholded there and never crosses to the host as a whole.

        Args:
            recon (array_like): an array of shape ``recon_shape``, as
                (rows, columns, slices), or None to remove the silhouette.
            threshold (float, optional): the absolute value a voxel must exceed
                to belong to the support.  None (default) uses
                ``DEFAULT_RECON_THRESHOLD_FRACTION`` of the largest
                absolute value in ``recon``.

        Raises:
            ValueError: if the array's shape is not the scan's recon shape.
        """
        self._install_recon(recon, threshold)
        # The silhouette is static, so it is part of the background.
        self._refresh(rebuild_limits=True)

    def save(self, path, dpi=110):
        """Write the figure to an image file.

        The artists a view change updates are marked animated, which keeps them
        out of an ordinary full draw so that the partial redraw can put them
        back.  A saved file must hold them, so they are unmarked for the
        duration of the write.

        A comparison window is written as a second file beside the first, with
        ``_comparison`` added to the name.  Saving ``scan.png`` while a
        comparison is drawn therefore also writes ``scan_comparison.png``.

        Args:
            path (str): the file to write.  The extension chooses the format.
            dpi (int, optional): dots per inch.

        Returns:
            str: the path of the main figure, which is the argument given.
        """
        self._suspend_blit = True
        try:
            self._set_animated(False)
            self.figure.savefig(path, dpi=dpi, facecolor='white')
        finally:
            self._set_animated(self._animate_moving())
            self._suspend_blit = False
            self._background = None
        if self.compare_figure is not None:
            self.compare_figure.savefig(_companion_path(path), dpi=dpi,
                                        facecolor='white')
        return path

    def show(self, block=True):
        """Open a window on the figure.

        A comparison opens a second window, and this method opens it too:
        ``plt.show`` shows every figure pyplot holds, and both figures are
        built through pyplot.

        On a backend with no window this prints a line and returns, leaving the
        figure available for :meth:`save`.

        Args:
            block (bool, optional): whether ``show`` waits for the window to
                close.
        """
        backend = matplotlib.get_backend().lower()
        if backend in NONINTERACTIVE_BACKENDS:
            print(f'The {backend} backend has no window; nothing was shown.  '
                  'Use GeometryFigure.save to write a file.')
            return
        plt.show(block=block)

    # ------------------------------------------------------------------
    # Checked arguments and small lookups
    # ------------------------------------------------------------------

    def _checked_view_index(self, view_index):
        view_index = int(view_index)
        if not 0 <= view_index < self.scene.num_views:
            raise IndexError(f'view_index {view_index} is outside '
                             f'[0, {self.scene.num_views}).')
        return view_index

    @staticmethod
    def _checked_zoom(mode):
        mode = str(mode)
        if mode not in ZOOM_MODES:
            raise ValueError(f'Unknown zoom {mode!r}; expected one of '
                             f'{ZOOM_MODES}.')
        return mode

    def _compare_index(self, view_index):
        """The view of the comparison that goes with a view of the primary.

        The comparison follows the slider, so the two share a view index.  A
        comparison with fewer views holds its last view instead of raising.
        """
        return min(int(view_index), self.compare_scene.num_views - 1)

    def _compare_view(self):
        """The comparison's ``ViewScene`` for the current view, or None."""
        if self.compare_scene is None:
            return None
        return self.compare_scene.view(self._compare_index(self._view_index))

    def _source_trajectory(self):
        """The source's path over all views, computed once and kept."""
        if self._trajectory is None:
            sources, _ = self.scene.trajectory()
            self._trajectory = sources
        return self._trajectory

    def _reference_view(self):
        """The angle-0 reference view, computed once and kept."""
        if self._reference is None:
            self._reference = self.scene.reference_view()
        return self._reference

    def _compare_source_trajectory(self):
        """The comparison's source path over all views, computed once."""
        if self._compare_trajectory is None:
            sources, _ = self.compare_scene.trajectory()
            self._compare_trajectory = sources
        return self._compare_trajectory

    def _trajectory_artists(self):
        """The (axes, artist) pairs that draw a source path."""
        pairs = [(axes, artist) for axes, artist in self._trajectory_lines]
        pairs.extend(self._compare_trajectory_lines)
        return pairs

    def _view_label(self, scene=None, view_index=None):
        """A short phrase naming what makes a view different.

        A rotating geometry is named by its angle, and the translation geometry
        by its translation vector.
        """
        scene = self.scene if scene is None else scene
        index = self._view_index if view_index is None else view_index
        if scene.kind == 'translation':
            vector = scene.translation_vectors[index]
            return ('translation ('
                    + ', '.join(_three_figures(value) for value in vector)
                    + ') ALU')
        angle = float(scene.angles[index])
        label = (f'angle {angle:.3f} rad '
                 f'({np.degrees(angle):.1f} deg)')
        if scene.kind == 'multiaxis':
            elevation = float(scene.elevations[index])
            label += f', elevation {elevation:.3f} rad'
        elif scene.kind == 'cone':
            shift = float(scene.z_shifts[index])
            if shift != 0.0:
                label += f', z shift {_three_figures(shift)} ALU'
        return label

    # ------------------------------------------------------------------
    # Layout and widgets
    # ------------------------------------------------------------------

    def _build_panels(self, figsize, title):
        """Create the figure and its five axes.

        The 3D view takes the whole left column, because it needs the room.
        The top view and the side view share the upper right, and the detector
        face and the text panel share the lower right.  The widget row goes
        under all of them.
        """
        self.figure = plt.figure(figsize=figsize)
        grid = self.figure.add_gridspec(
            2, 3, width_ratios=(1.35, 1.0, 1.0), left=0.04, right=0.985,
            bottom=GRID_BOTTOM, top=0.90, wspace=0.30, hspace=0.28)
        self.ax_3d = self.figure.add_subplot(grid[:, 0], projection='3d')
        self.ax_top = self.figure.add_subplot(grid[0, 1])
        self.ax_side = self.figure.add_subplot(grid[0, 2])
        self.ax_detector = self.figure.add_subplot(grid[1, 1])
        # The detector face gives up the bottom of its cell, so that its legend
        # has a band outside the panel to sit in; see _create_legends.  The
        # panel's box is reshaped to its equal aspect at every draw, so this
        # takes width from the panel only when its box is taller than the
        # shortened cell.
        cell = self.ax_detector.get_subplotspec().get_position(self.figure)
        self.ax_detector.set_position(
            (cell.x0, DETECTOR_PANEL_BOTTOM, cell.width,
             cell.y1 - DETECTOR_PANEL_BOTTOM))
        self.ax_text = self.figure.add_subplot(grid[1, 2])
        self.panel_axes = (self.ax_3d, self.ax_top, self.ax_side,
                           self.ax_detector, self.ax_text)

        if title is None:
            title = (f'{self._quantities["geometry_kind"]} geometry, '
                     f'sinogram {self._quantities["sinogram_shape_text"]}, '
                     f'recon {self._quantities["recon_shape_text"]}')
        self.figure.suptitle(title, fontsize=11)

        # The opaque rectangle a partial redraw paints over the slider row
        # before drawing it again; animated=True keeps it out of full draws.
        self._clear_rect = Rectangle((0, 0), 1, 1,
                                     facecolor=self.figure.get_facecolor(),
                                     edgecolor='none', animated=True,
                                     transform=IdentityTransform())
        self.figure.add_artist(self._clear_rect)

    def _create_widgets(self, wanted):
        """Create the view slider and the six toggles.

        The slider steps by one view and never by a fraction, and its own draw
        is turned off so that a step goes through this class's redraw instead.
        A scan with one view has nothing to slide, so the slider axes is
        hidden.  These are the conventions ``mbirtorch/viewer.py`` uses for its
        slice slider.

        The three overlay toggles are built whatever data the figure holds, so
        a toggle whose overlay is absent is drawn and does nothing.  An array
        given later through :meth:`set_sinogram`, :meth:`set_recon`, or
        :meth:`set_compare` then has its toggle ready and in the state the
        toggle is showing.
        """
        self.view_slider = None
        self.trajectory_check = None
        self.zoom_check = None
        self.reference_check = None
        self.sinogram_check = None
        self.recon_check = None
        self.compare_check = None
        self._slider_axes = None
        if not wanted:
            return

        self._slider_axes = self.figure.add_axes(SLIDER_RECT)
        if self.scene.num_views > 1:
            self.view_slider = Slider(
                self._slider_axes, label='View', valmin=0,
                valmax=self.scene.num_views - 1, valinit=self._view_index,
                valstep=1, valfmt='%0.0f')
            self.view_slider.drawon = False
            self.view_slider.label.set_fontsize(WIDGET_FONT_SIZE)
            self.view_slider.valtext.set_fontsize(WIDGET_FONT_SIZE)
            self.view_slider.on_changed(self._on_slider)
        else:
            self._slider_axes.set_visible(False)

        self.trajectory_check = self._add_check(
            TRAJECTORY_CHECK_RECT, 'source path', self._show_trajectory,
            self._on_trajectory_check)
        self.zoom_check = self._add_check(
            ZOOM_CHECK_RECT, '3D zoom to volume', self._zoom == 'volume',
            self._on_zoom_check)
        self.reference_check = self._add_check(
            REFERENCE_CHECK_RECT, 'angle-0 reference', self._show_reference,
            self._on_reference_check)
        self.sinogram_check = self._add_check(
            SINOGRAM_CHECK_RECT, 'sinogram', self._show_sinogram,
            self._on_sinogram_check)
        self.recon_check = self._add_check(
            RECON_CHECK_RECT, PHANTOM_NAME, self._show_recon,
            self._on_recon_check)
        self.compare_check = self._add_check(
            COMPARE_CHECK_RECT, 'comparison', self._show_compare,
            self._on_compare_check)

    def _add_check(self, rect, label, state, handler):
        """Create one toggle of the widget row.

        The six toggles differ only in where they sit, what they are called,
        which state they start in, and what a click calls, so one routine
        builds them all.  The frame is turned off, because a toggle is a box
        and a label and not a panel.

        Args:
            rect (tuple): where the toggle goes, as (left, bottom, width,
                height) in figure coordinates.
            label (str): the toggle's one label.
            state (bool): whether it starts checked.
            handler (callable): what a click on it calls.

        Returns:
            CheckButtons: the toggle.
        """
        axes = self.figure.add_axes(rect)
        axes.set_frame_on(False)
        check = CheckButtons(axes, [label], [bool(state)])
        for text in check.labels:
            text.set_fontsize(WIDGET_FONT_SIZE)
        check.on_clicked(handler)
        return check

    def _on_slider(self, value):
        """The view slider moved."""
        if self._syncing_widgets:
            return
        index = int(round(float(value)))
        if index == self._view_index:
            return
        self.set_view(index)

    def _on_trajectory_check(self, _label):
        """The source-path toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_show_trajectory(self.trajectory_check.get_status()[0])

    def _on_zoom_check(self, _label):
        """The 3D zoom toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_zoom('volume' if self.zoom_check.get_status()[0] else 'scan')

    def _on_reference_check(self, _label):
        """The angle-0 reference toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_show_reference(self.reference_check.get_status()[0])

    def _on_sinogram_check(self, _label):
        """The sinogram toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_show_sinogram(self.sinogram_check.get_status()[0])

    def _on_recon_check(self, _label):
        """The phantom toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_show_recon(self.recon_check.get_status()[0])

    def _on_compare_check(self, _label):
        """The comparison toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_show_compare(self.compare_check.get_status()[0])

    def _sync_slider(self):
        """Move the slider to the current view without calling back."""
        if self.view_slider is None:
            return
        if int(round(float(self.view_slider.val))) == self._view_index:
            return
        self._syncing_widgets = True
        try:
            self.view_slider.set_val(self._view_index)
        finally:
            self._syncing_widgets = False

    def _set_check(self, check, state):
        """Match one toggle to a state, without calling its handler back.

        A toggle is moved by clicking it, which calls the handler, and the
        handler would call the method that is already running.  The flag this
        sets is what every handler reads first.

        Args:
            check: the ``CheckButtons``, or None when the figure has no
                widgets.
            state (bool): the state the toggle is to show.
        """
        if check is None or check.get_status()[0] == bool(state):
            return
        self._syncing_widgets = True
        try:
            check.set_active(0)
        finally:
            self._syncing_widgets = False

    def _sync_trajectory_check(self):
        """Match the toggle to the state, without calling back."""
        self._set_check(self.trajectory_check, self._show_trajectory)

    def _sync_zoom_check(self):
        """Match the zoom toggle to the state, without calling back."""
        self._set_check(self.zoom_check, self._zoom == 'volume')

    def _sync_reference_check(self):
        """Match the reference toggle to the state, without calling back."""
        self._set_check(self.reference_check, self._show_reference)

    def _sync_sinogram_check(self):
        """Match the sinogram toggle to the state, without calling back."""
        self._set_check(self.sinogram_check, self._show_sinogram)

    def _sync_recon_check(self):
        """Match the phantom toggle to the state, without calling back."""
        self._set_check(self.recon_check, self._show_recon)

    def _sync_compare_check(self):
        """Match the comparison toggle to the state, without calling back."""
        self._set_check(self.compare_check, self._show_compare)

    # ------------------------------------------------------------------
    # Creating the artists
    # ------------------------------------------------------------------

    def _create_artists(self):
        """Create every artist once, for the primary geometry.

        The artists split in two.  A static artist is drawn by a full repaint
        and belongs to the background: the volume box, the region of
        reconstruction, the rotation axis or translation path, the voxel
        marker, the detector grid, the two detector-face reference markers, the
        angle-0 reference, and the text block.  A moving artist carries the
        current view and is redrawn on every view change: the source, the
        detector outline, the rays, the projected outlines of the volume box
        and of the region of reconstruction, the offset segments, the arc, the
        labels of the source and the detector, the side view's
        ``recon_slice_offset`` label, the panel titles, and the text panel's
        footer.  A moving artist is marked animated only where the partial
        redraw runs; see :meth:`_animate_moving`.

        The two data overlays are created after these, by
        :meth:`_install_sinogram` and :meth:`_install_recon`, because a caller
        may add or remove either one later.  The sinogram image joins the
        moving artists, and so does the phantom's outline on the detector face,
        which is the one part of the phantom that a view change moves.  The two
        silhouette images and the phantom's other outlines are static.
        """
        view = self.scene.view(self._view_index)
        self._create_3d_artists(view)
        self._create_top_artists(view)
        self._create_side_artists(view)
        self._create_detector_artists(view)
        self._create_reference_artists()
        self._create_text_artists()
        self._trajectory_lines = [(self.ax_3d, self._path_3d),
                                  (self.ax_top, self._path_top),
                                  (self.ax_side, self._path_side)]
        self._compare_trajectory_lines = []
        self._set_animated(self._animate_moving())

    def _set_animated(self, flag):
        """Mark or unmark every moving artist as animated."""
        for _, artist in self._moving + self._compare_moving:
            artist.set_animated(bool(flag))
        if self._arrow_3d is not None:
            self._arrow_3d.set_animated(bool(flag))

    def _moving_line(self, axes, color, **kwargs):
        """One empty line artist that a view change will fill in."""
        line, = axes.plot([], [], color=color, **kwargs)
        self._moving.append((axes, line))
        return line

    def _moving_line_3d(self, axes, color, **kwargs):
        """One empty 3D line artist that a view change will fill in."""
        line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                          color=color, **AXLIM_CLIP, **kwargs)
        self._moving.append((axes, line))
        return line

    def _moving_text(self, axes, text, color, vertical):
        """One small label that a view change moves to what it names.

        The label is placed at a data point and offset from it in points, so
        that the gap between the label and its marker is the same gap at every
        panel scale.  :func:`_place_beside` sets the horizontal part of that
        offset per view and keeps the vertical part given here.

        Args:
            axes: the panel.
            text (str): the label.
            color: the color of the element the label names.
            vertical (float): the vertical offset from the data point, in
                points.  A positive offset puts the label above the point and
                a negative one puts it below.

        Returns:
            The annotation, already registered as a moving artist.
        """
        label = axes.annotate(text, xy=(0.0, 0.0),
                              textcoords='offset points',
                              xytext=(LABEL_GAP_POINTS, vertical),
                              fontsize=ANNOTATION_FONT_SIZE, color=color,
                              ha='left',
                              va='bottom' if vertical >= 0 else 'top')
        self._moving.append((axes, label))
        return label

    def _moving_text_3d(self, axes, text, color, vertical='baseline'):
        """One small 3D label that a view change moves to what it names.

        A 3D text artist has no offset in points, so the gap from the point it
        names is a leading space in the text, and the only vertical choice is
        the alignment.

        Args:
            axes: the 3D panel.
            text (str): the label.
            color: the color of the element the label names.
            vertical (str, optional): the vertical alignment, which puts the
                label on the line of its point, above it, or below it.
        """
        label = axes.text(0.0, 0.0, 0.0, text, color=color,
                          fontsize=ANNOTATION_FONT_SIZE,
                          verticalalignment=vertical)
        self._moving.append((axes, label))
        return label

    # --- the 3D panel ---

    def _create_3d_artists(self, view):
        """Create the 3D panel's artists and label its axes."""
        axes = self.ax_3d

        # Static: the volume box, the region of reconstruction, the rotation
        # axis or the translation path, and the voxel marker.  None of these
        # moves with the view, because the object is the thing held fixed.
        box = _joined([view.volume_corners[[first, second]]
                       for first, second in VOLUME_BOX_EDGES])
        axes.plot(box[:, 0], box[:, 1], box[:, 2], color=COLORS['volume'],
                  linewidth=1.0, label='volume box', **AXLIM_CLIP)
        self._create_3d_ror(view)
        if view.rotation_axis is not None:
            segment = _sampled_segment(view.rotation_axis[0],
                                       view.rotation_axis[1])
            axes.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                      color=COLORS['axis'], linewidth=1.2, linestyle='--',
                      label='rotation axis', **AXLIM_CLIP)
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, 0], path[:, 1], path[:, 2], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5,
                      label='translation path', **AXLIM_CLIP)
        axes.plot([view.voxel0_center[0]], [view.voxel0_center[1]],
                  [view.voxel0_center[2]], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  label='voxel (0, 0, 0)', **AXLIM_CLIP, zorder=6)
        self._path_3d, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                   color=COLORS['trajectory'],
                                   linewidth=1.0, linestyle='-.',
                                   label='source path', **AXLIM_CLIP)
        self._path_3d.set_visible(self._show_trajectory)

        # Moving: the source, the detector, the rays, and the arc.
        source_label = ('source' if view.source is not None
                        else 'source (drawn)')
        self._source_3d = self._moving_line_3d(
            axes, COLORS['source'], marker='*', markersize=SOURCE_MARKER_SIZE,
            linestyle='none', label=source_label, zorder=6)
        self._detector_3d = self._moving_line_3d(
            axes, COLORS['detector'], linewidth=1.6, label='detector')
        self._face_3d = None
        if not self.scene.use_curved_detector:
            # A curved detector's face is not a polygon, so only its outline is
            # drawn.  A flat panel gets a lightly filled face, which tells the
            # near side of the detector from the far side.
            self._face_3d = Poly3DCollection(
                [view.detector_corners], facecolor=COLORS['detector'],
                alpha=0.12, edgecolor='none', **AXLIM_CLIP)
            axes.add_collection3d(self._face_3d)
            self._moving.append((axes, self._face_3d))
        self._rays_3d = self._moving_line_3d(axes, COLORS['rays'],
                                             linewidth=0.7)
        self._central_3d = self._moving_line_3d(
            axes, COLORS['central_ray'], linewidth=1.1, label='central ray')
        self._pixel0_3d = self._moving_line_3d(
            axes, COLORS['pixel0'], marker='x', linestyle='none',
            markersize=INDEX_MARKER_SIZE,
            markeredgewidth=INDEX_MARKER_WIDTH,
            label='detector pixel (0, 0)', zorder=6)
        self._arc_3d = self._moving_line_3d(axes, COLORS['axis'],
                                            linewidth=1.4)
        # The arc's label hangs below the end of the arc, which is a few
        # degrees of travel from the source, so that it does not run into the
        # source's own label.
        self._arc_text_3d = self._moving_text_3d(axes, ' source travel',
                                                 COLORS['axis'],
                                                 vertical='top')
        # Two spaces, not one: the source's star marker is wide enough to
        # reach under a label that starts one space away.
        self._source_text_3d = self._moving_text_3d(axes, '  source',
                                                    COLORS['source'])
        self._detector_text_3d = self._moving_text_3d(axes, ' detector',
                                                      COLORS['detector'])
        # The iso label hangs below its point, because the angle-0 reference's
        # label sits at the reference detector, which is a small distance from
        # the detector iso when the view angle is small.
        self._iso_text_3d = self._moving_text_3d(axes, ' ' + ISO_NAME,
                                                 COLORS['central_ray'],
                                                 vertical='top')

        axes.set_xlabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_zlabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        axes.view_init(elev=self._elevation_deg, azim=self._azimuth_deg,
                       roll=_camera_roll_deg())
        axes.set_box_aspect((1.0, 1.0, 1.0))
        self._title_3d = axes.set_title('', fontsize=TITLE_FONT_SIZE)
        self._moving.append((axes, self._title_3d))

    def _create_3d_ror(self, view):
        """Draw the region of reconstruction, when there is one to draw.

        The region is an elliptic cylinder about the rotation axis, so it does
        not move with the view.  It is drawn as its two rings and four
        uprights, joined into one polyline; the uprights make the shape read as
        a cylinder rather than as two unrelated ellipses.
        """
        cylinder = view.ror_cylinder
        if cylinder is None:
            return
        lower = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_min'])
        upper = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_max'])
        parts = [lower, upper]
        for step in range(0, ELLIPSE_SAMPLES - 1, (ELLIPSE_SAMPLES - 1) // 4):
            parts.append(np.stack([lower[step], upper[step]]))
        rings = _joined(parts)
        self.ax_3d.plot(rings[:, 0], rings[:, 1], rings[:, 2],
                        color=COLORS['ror'], linewidth=0.9,
                        label='region of reconstruction', **AXLIM_CLIP)

    # --- the two projected panels ---

    def _create_top_artists(self, view):
        """Create the top panel's artists: the xy plane seen from -z.

        The panel puts y on its horizontal axis, increasing to the left, and x
        on its vertical axis, increasing downward.  That is the orientation of
        the reference figure of Balke et al. (2018), in which the beam runs
        from left to right.
        """
        axes = self.ax_top
        first, second = TOP_PANEL_COLUMNS

        walk = view.volume_corners[list(_XY_FOOTPRINT_WALK)]
        axes.plot(walk[:, first], walk[:, second], color=COLORS['volume'],
                  linewidth=1.2)
        cylinder = view.ror_cylinder
        if cylinder is not None:
            ring = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                   cylinder['semi_axis_y'], cylinder['z_min'])
            axes.plot(ring[:, first], ring[:, second], color=COLORS['ror'],
                      linewidth=0.9)
        if view.rotation_axis is not None:
            # The rotation axis is a point in this plane.
            axes.plot([0.0], [0.0], marker='+', markersize=9,
                      markeredgewidth=1.4, color=COLORS['axis'],
                      linestyle='none')
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, first], path[:, second], color=COLORS['axis'],
                      linewidth=1.2, marker='o', markersize=3.0)
            # The path is a few ALU across while the panel spans the
            # source-detector distance, so it needs a label to be recognized.
            # The label hangs below the path on the screen, because the
            # angle-0 reference's label runs along the ray above it.
            axes.annotate('translation path',
                          xy=(float(np.mean(path[:, first])),
                              _screen_bottom(path[:, second])),
                          textcoords='offset points', xytext=(0, -5),
                          ha='center', va='top',
                          fontsize=ANNOTATION_FONT_SIZE, color=COLORS['axis'])
        axes.plot([view.voxel0_center[first]], [view.voxel0_center[second]],
                  marker='x', linestyle='none', markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  zorder=6)
        self._path_top, = axes.plot([], [], color=COLORS['trajectory'],
                                    linewidth=1.0, linestyle='-.')
        self._path_top.set_visible(self._show_trajectory)

        self._top = self._create_projected_moving_artists(axes)
        # Only the top view names the pixel-0 marker.  In the side view that
        # marker sits at the end of the detector, where the detector's own
        # label already is.  The label sits a whole line from the marker,
        # because a scan whose detector is short against the panel puts the
        # marker close to the detector iso, whose own two labels take the lines
        # nearer the detector.  Which side of the marker it sits on is chosen
        # per view by _update_projected_panel.
        self._top['pixel0_label'] = self._moving_text(
            axes, 'pixel (0,0)', COLORS['pixel0'], PIXEL0_LABEL_GAP_POINTS)
        self._arc_top = self._moving_line(axes, COLORS['axis'], linewidth=1.4)
        self._arrow_top = FancyArrowPatch(
            (0.0, 0.0), (0.0, 0.0), arrowstyle='-|>', mutation_scale=9,
            linewidth=1.4, color=COLORS['axis'], shrinkA=0.0, shrinkB=0.0)
        axes.add_patch(self._arrow_top)
        self._moving.append((axes, self._arrow_top))
        # Which side of the arc's end this label sits on is chosen per view by
        # _update_arc_and_trajectory.
        self._arc_text_top = axes.annotate(
            'source travel', xy=(0.0, 0.0), textcoords='offset points',
            xytext=(3, -ARC_LABEL_GAP_POINTS), fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['axis'])
        self._moving.append((axes, self._arc_text_top))
        axes.set_xlabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        self._title_top = axes.set_title(_top_view_title(),
                                         fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _create_side_artists(self, view):
        """Create the side panel's artists: the yz plane seen from +x.

        The panel puts y on its horizontal axis, increasing to the left, and z
        on its vertical axis, increasing downward, so that -z is at the top and
        the source of a view at angle 0 is on the left.
        """
        axes = self.ax_side
        first, second = SIDE_PANEL_COLUMNS

        walk = view.volume_corners[list(_YZ_FACE_WALK)]
        axes.plot(walk[:, first], walk[:, second], color=COLORS['volume'],
                  linewidth=1.2)
        axes.axhline(0.0, color=COLORS['axis'], linewidth=0.7, linestyle=':')

        # The volume's z center, which is recon_slice_offset, drawn as a
        # segment from z = 0 at the volume's high y edge and labeled above the
        # volume box.  A cone geometry's side view is many times wider than it
        # is tall, so the three labels of this panel go at three places along
        # it: the detector's at the near end, the volume's in the middle, and
        # the trajectory's in the far corner.
        z_min, z_max = self.scene.volume_z_range()
        z_center = 0.5 * (z_min + z_max)
        self._slice_offset_label = None
        self._slice_offset_point = None
        if z_center != 0.0:
            y_at = float(np.max(view.volume_corners[:, 1]))
            axes.plot([y_at, y_at], [0.0, z_center], color=COLORS['volume'],
                      linewidth=2.6, solid_capstyle='butt')
            # The label reads outward from the segment, away from the volume
            # box.  Over the middle of the box it ran into the row-offset
            # label, and below the segment it ran into the source's label,
            # which hangs below the source in this panel.  This panel is only a
            # few labels tall, so each of its labels needs its own place.
            #
            # Which side of the segment's end the label sits on changes with
            # the view, so the label is a moving artist and
            # _place_slice_offset_label puts it in place.  A static label above
            # the end was drawn over by the source's marker in the multiaxis
            # example, whose source rises and falls with the elevation.
            self._slice_offset_point = (y_at, z_center)
            self._slice_offset_label = self._moving_text(
                axes, f'recon_slice_offset {_three_figures(z_center)}',
                COLORS['volume'], LABEL_GAP_POINTS)
        if view.rotation_axis is not None:
            segment = view.rotation_axis
            axes.plot(segment[:, first], segment[:, second],
                      color=COLORS['axis'], linewidth=1.2, linestyle='--')
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, first], path[:, second], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5)
        axes.plot([view.voxel0_center[first]], [view.voxel0_center[second]],
                  marker='x', linestyle='none', markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  zorder=6)
        self._path_side, = axes.plot([], [], color=COLORS['trajectory'],
                                     linewidth=1.0, linestyle='-.')
        self._path_side.set_visible(self._show_trajectory)
        self._path_note_side = axes.annotate(
            '', xy=(0.98, 0.04), xycoords='axes fraction', va='bottom',
            ha='right', fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['trajectory'])
        self._path_note_side.set_visible(self._show_trajectory)

        # The source's label goes below the source in this panel.  Above it
        # is where the recon_slice_offset label sits, and the two ran into
        # each other in the curved cone figure.
        self._side = self._create_projected_moving_artists(
            axes, source_vertical=-7, name_iso=False)
        axes.set_xlabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        self._title_side = axes.set_title(_side_view_title(),
                                          fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _create_projected_moving_artists(self, axes, source_vertical=7,
                                         name_iso=True):
        """The moving artists the top and side panels share.

        The two panels are the same scene projected onto two different planes,
        so one routine creates the source, the detector outline, the rays, the
        central ray, the pixel marker, and the offset segment for both.

        Args:
            axes: the panel.
            source_vertical (float, optional): the vertical offset of the
                source's label, in points.
            name_iso (bool, optional): whether to label the detector iso in
                this panel.

        Returns:
            dict: the artists, keyed by name.
        """
        artists = dict(
            rays=self._moving_line(axes, COLORS['rays'], linewidth=0.7),
            edge_rays=self._moving_line(axes, EMPHASIZED_RAY_COLOR,
                                        linewidth=1.1),
            detector=self._moving_line(axes, COLORS['detector'],
                                       linewidth=1.6),
            central=self._moving_line(axes, COLORS['central_ray'],
                                      linewidth=1.1),
            offset=self._moving_line(axes, COLORS['detector'], linewidth=2.6,
                                     solid_capstyle='butt'),
            source=self._moving_line(axes, COLORS['source'], marker='*',
                                     markersize=SOURCE_MARKER_SIZE,
                                     linestyle='none', zorder=6),
            pixel0=self._moving_line(axes, COLORS['pixel0'], marker='x',
                                     linestyle='none',
                                     markersize=INDEX_MARKER_SIZE,
                                     markeredgewidth=INDEX_MARKER_WIDTH,
                                     zorder=6),
        )
        # The source's label is offset a little farther than the others,
        # because the central ray ends at the source and a smaller offset put
        # the label on that line.
        artists['source_label'] = self._moving_text(
            axes, 'source', COLORS['source'], source_vertical)
        artists['detector_label'] = self._moving_text(
            axes, 'detector', COLORS['detector'], 4)
        # The detector iso is where the central ray meets the detector, and it
        # is the point the two detector offsets are measured from.  Only the
        # top view names it, for the reason the pixel-0 marker is named there
        # alone: the side view is short and wide, and its detector already
        # carries its own label and the row-offset label at the same end.
        # _update_projected_panel puts this label and the offset label below on
        # the side of the detector away from its far corner.
        if name_iso:
            artists['iso_label'] = self._moving_text(
                axes, ISO_NAME, COLORS['central_ray'],
                ISO_LABEL_GAP_POINTS)
        # The label of the detector offset this panel shows.  It names the
        # offset segment, which runs from the detector iso to the detector
        # center, so it hangs at the iso like the label above, one line
        # farther out.
        artists['offset_label'] = self._moving_text(
            axes, '', COLORS['detector'],
            ISO_LABEL_GAP_POINTS + LABEL_LINE_POINTS)
        return artists

    # --- the detector-face panel ---

    def _create_detector_artists(self, view):
        """Create the detector-face panel's artists.

        The horizontal axis is the channel index, increasing to the right, and
        the vertical axis is the row index, increasing downward with row 0 at
        the top.  That is the view from the source toward the detector with -z
        up, and it is how ``imshow`` shows one view of a sinogram.  The grid,
        the detector iso, and the detector center depend only on the detector
        parameters, so they are static.  What moves with the view is the
        projected outline of the volume box and, when the scene has a region of
        reconstruction, the projected outline of that region.

        A phantom's projected outline moves with the view as well, and it is
        created by :meth:`_create_recon_detector_outline` instead of here,
        because the figure may be given a phantom later or have one taken away.
        """
        axes = self.ax_detector
        num_rows = self.scene.num_det_rows
        num_channels = self.scene.num_det_channels

        axes.add_patch(Rectangle((-0.5, -0.5), num_channels, num_rows,
                                 facecolor=COLORS['detector'], alpha=0.10,
                                 edgecolor=COLORS['detector'], linewidth=1.4))
        ((landing_row, landing_channel),
         (center_row, center_channel)) = \
            self.scene.detector_iso_and_center_indices()
        axes.plot([float(landing_channel)], [float(landing_row)], marker='o',
                  markersize=6, linestyle='none', markerfacecolor='none',
                  markeredgewidth=1.4, color=COLORS['central_ray'],
                  label=ISO_NAME)
        axes.plot([float(center_channel)], [float(center_row)], marker='s',
                  markersize=5, linestyle='none', markerfacecolor='none',
                  markeredgewidth=1.4, color=COLORS['detector'],
                  label=CENTER_NAME)
        axes.plot([0.0], [0.0], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['pixel0'],
                  label='pixel (0,0) = sino[v, 0, 0]')

        self._edges_inside = self._moving_line(axes, COLORS['volume'],
                                               linewidth=1.2,
                                               label='volume box')
        # No legend label: this line exists in every view and carries data
        # only where the volume projects past the detector's edge.
        self._edges_outside = self._moving_line(axes, COLORS['overshoot'],
                                                linewidth=1.6)
        # The region of reconstruction's two rims, split the same way.  This
        # is the shape the fit statement tests when the mask is on, so the
        # panel draws what the statement is about.  A scan with no mask has no
        # cylinder and gets neither line.
        self._ror_inside = None
        self._ror_outside = None
        if view.ror_outline_on_detector is not None:
            self._ror_inside = self._moving_line(
                axes, COLORS['ror'], linewidth=1.0,
                label='region of reconstruction')
            self._ror_outside = self._moving_line(axes, COLORS['overshoot'],
                                                  linewidth=1.4)
        axes.set_aspect('equal', adjustable='box')
        axes.set_xlabel('channel index', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('row index', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        self._title_detector = axes.set_title('', fontsize=TITLE_FONT_SIZE)
        self._moving.append((axes, self._title_detector))

    def _create_recon_detector_outline(self):
        """Create the phantom's projected outline on the detector face.

        The line is the same outline the 3D panel draws, projected onto the
        detector by the scene for the view drawn, so it moves with the view and
        :meth:`_update_detector_panel` fills it in.  It is dashed and in the
        volume's color, as the 3D outline is, so that the two read as one
        drawing of one array.

        The line is not split at the detector's edge, the way the volume box's
        projected outline is.  The phantom lies inside the volume, so a
        phantom's outline that leaves the grid leaves it inside the volume
        box's outline, and that outline already carries the overshoot color.
        """
        self._recon_detector_line = self._moving_line(
            self.ax_detector, COLORS['volume'],
            linewidth=RECON_OUTLINE_DETECTOR_LINEWIDTH, linestyle='--',
            label=PHANTOM_NAME)
        self._recon_outlines.append((self.ax_detector,
                                     self._recon_detector_line))

    # --- the angle-0 reference ---

    def _reference_label(self):
        """What the angle-0 reference's arrow is called in the top view.

        The name is two lines, because one line of it is wider than the top
        panel.  The translation geometry has no view angle, so its reference is
        the view action's identity said the other way: no translation.
        """
        if self.scene.kind == 'translation':
            return 'projection direction,\nno translation'
        return 'projection direction,\nangle 0'

    def _reference_short_label(self):
        """The same name, short enough for the 3D panel."""
        return ('no translation' if self.scene.kind == 'translation'
                else 'angle 0')

    def _create_reference_artists(self):
        """Draw the source and the detector at their angle-0 position.

        The reference is the geometry with the view action set to the identity,
        which ``GeometryScene.reference_view`` returns.  It answers what one
        view cannot: where the source and the detector stand at angle 0, and
        which way the geometry projects through the object from there.

        These artists do not depend on the view, so they are ordinary static
        artists and they belong to the background.  Their toggle therefore
        repaints the whole figure.  The 3D view and the top view get them.  The
        side view is the yz plane, in which a rotation about z moves nothing,
        so a reference drawn there would lie on top of the view drawn.
        """
        reference = self._reference_view()
        self._reference_artists = []
        dotted = dict(linestyle=REFERENCE_DOTS, linewidth=REFERENCE_LINEWIDTH,
                      alpha=REFERENCE_ALPHA)
        hollow_star = dict(marker='*', linestyle='none',
                           markersize=SOURCE_MARKER_SIZE,
                           markerfacecolor='none',
                           markeredgecolor=COLORS['source'],
                           markeredgewidth=1.3, alpha=REFERENCE_ALPHA)
        source = reference.source_draw
        origin = reference.detector_origin
        outline = reference.detector_outline
        # The arrowhead is drawn from a short segment at the detector end of
        # the central ray, so that it points the way the geometry projects.
        head_start = origin + REFERENCE_ARROW_FRACTION * (source - origin)

        def keep(axes, artist):
            """Register one reference artist and set its visibility."""
            artist.set_visible(self._show_reference)
            self._reference_artists.append((axes, artist))
            return artist

        axes = self.ax_3d
        keep(axes, axes.plot([source[0]], [source[1]], [source[2]],
                             label='angle-0 reference', **AXLIM_CLIP,
                             **hollow_star)[0])
        keep(axes, axes.plot(outline[:, 0], outline[:, 1], outline[:, 2],
                             color=COLORS['detector'], **AXLIM_CLIP,
                             **dotted)[0])
        central = _sampled_segment(source, origin)
        keep(axes, axes.plot(central[:, 0], central[:, 1], central[:, 2],
                             color=COLORS['central_ray'], **AXLIM_CLIP,
                             **dotted)[0])
        direction = origin - head_start
        self._reference_arrow_3d = keep(axes, axes.quiver(
            head_start[0], head_start[1], head_start[2],
            direction[0], direction[1], direction[2],
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA,
            arrow_length_ratio=3.0, linewidth=REFERENCE_LINEWIDTH))
        self._reference_arrow_point = head_start
        # The label goes at the middle of the reference detector's low-row
        # edge.  On the arrowhead, which lands at the middle of that detector,
        # it fell on the current detector's own label instead.
        corners = reference.detector_corners
        label_at = 0.5 * (corners[0] + corners[1])
        self._reference_text_3d = keep(axes, axes.text(
            label_at[0], label_at[1], label_at[2],
            ' ' + self._reference_short_label(),
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA,
            fontsize=ANNOTATION_FONT_SIZE))

        axes = self.ax_top
        across, down = TOP_PANEL_COLUMNS
        keep(axes, axes.plot([source[across]], [source[down]],
                             **hollow_star)[0])
        keep(axes, axes.plot(outline[:, across], outline[:, down],
                             color=COLORS['detector'], **dotted)[0])
        keep(axes, axes.plot([source[across], origin[across]],
                             [source[down], origin[down]],
                             color=COLORS['central_ray'], **dotted)[0])
        head = FancyArrowPatch(
            (head_start[across], head_start[down]),
            (origin[across], origin[down]),
            arrowstyle='-|>', mutation_scale=10,
            linewidth=REFERENCE_LINEWIDTH, color=COLORS['central_ray'],
            alpha=REFERENCE_ALPHA, shrinkA=0.0, shrinkB=0.0)
        axes.add_patch(head)
        keep(axes, head)
        # The top view's label reads along the ray, which is the line it
        # names.
        # The label sits in the panel's upper left corner and not on the ray
        # it names.  The ray runs across the middle of this panel, between the
        # source's labels at one end and the detector's three at the other,
        # and a two-line label anywhere along it ran into one of them in at
        # least one of the six geometries.  The corner is the one part of the
        # panel that no geometry draws in, and the dotted line, its hollow
        # star, and its arrowhead are the only dotted artists in the panel, so
        # the label is not ambiguous there.
        keep(axes, axes.annotate(
            self._reference_label(), xy=REFERENCE_LABEL_CORNER,
            xycoords='axes fraction', fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA, ha='left',
            va='top'))

    # --- the text panel ---

    def _create_text_artists(self):
        """Create the text panel's three text artists.

        The panel holds a static block at the top, a comparison block above the
        footer, and a footer that names the view drawn.  The split exists for
        speed: the static block is about twenty lines of monospace text, and
        rendering it costs more than every line and marker of the four drawing
        panels together, so a view change must not redraw it.
        """
        axes = self.ax_text
        axes.set_axis_off()
        self._static_text = axes.text(
            0.0, 1.0, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left')
        # The comparison block sits under the static block, and
        # _place_compare_text moves it to where that block ends.
        self._compare_text = axes.text(
            0.0, 0.30, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left', color=COLORS['compare'])
        self._footer_text = axes.text(
            0.0, 0.10, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left')
        self._moving.append((axes, self._footer_text))
        axes.set_title('Derived quantities', fontsize=TITLE_FONT_SIZE)
        self._update_static_text()

    def _update_static_text(self, keep_font=False):
        """Fill in the static text block and the comparison block.

        Args:
            keep_font (bool, optional): whether to keep the font size the panel
                has now.  New content starts from the panel's own size, but a
                re-flow of the comparison section during the fitting passes
                must not undo a font size that fitting has already chosen.
        """
        quantities = self._quantities

        def triple(*keys):
            return ', '.join(_three_figures(quantities[key]) for key in keys)

        rows = [
            ('geometry', quantities['geometry_kind'], ''),
            ('views', str(quantities['num_views']), ''),
            ('sinogram (v, r, c)', quantities['sinogram_shape_text'], ''),
            ('recon (r, c, s)', quantities['recon_shape_text'], ''),
            ('magnification', _three_figures(quantities['magnification']), ''),
            ('fan angle', _three_figures(quantities['fan_angle_deg']), 'deg'),
            ('cone angle', _three_figures(quantities['cone_angle_deg']),
             'deg'),
            ('lateral FoV', _three_figures(quantities['lateral_fov_alu']),
             'ALU'),
            ('axial FoV', _three_figures(quantities['axial_fov_alu']), 'ALU'),
            ('voxel pitch x, y, z',
             triple('delta_voxel_x', 'delta_voxel_y', 'delta_voxel_z'), 'ALU'),
            ('volume extent x,y,z',
             triple('volume_extent_x', 'volume_extent_y', 'volume_extent_z'),
             'ALU'),
            ('volume z center', _three_figures(quantities['volume_z_center']),
             'ALU'),
            ('detector w by h',
             _three_figures(quantities['detector_width']) + ' by '
             + _three_figures(quantities['detector_height']), 'ALU'),
            ('det pitch chan, row',
             triple('detector_channel_pitch', 'detector_row_pitch'), 'ALU'),
            ('det center du, dv',
             triple('detector_center_u', 'detector_center_v'), 'ALU'),
            ('helical travel',
             _three_figures(quantities['helical_travel_alu']), 'ALU'),
            # The fit statement, in its two halves: the channels, and the
            # rows or, for a helical scan, the swept coverage.
            ('lateral fit', _lateral_fit_text(quantities), ''),
            ('axial fit', _axial_fit_text(quantities), ''),
            # How many views the fit statement's shape leaves the detector in.
            # A helical scan leaves it in every view by design, so the count
            # is the number that says whether "no" means a scan that is wrong
            # or a scan that is helical.
            ('leaves det in views',
             f'{quantities["views_leaving_detector"]} of '
             f'{quantities["num_views"]}', ''),
        ]
        width = max(len(name) for name, _, _ in rows)
        lines = [f'{name:<{width}} : {value}{" " + unit if unit else ""}'
                 for name, value, unit in rows]
        lines.append('')
        # Two notes about the drawing itself.  The first line says what the
        # offsets of the detector center are measured from and which way z is
        # drawn, which is the display convention every panel follows; the two
        # facts share a line because the panel is short of lines.  The scene's
        # own note then names every position that is a drawing choice.
        for note in (f'(du, dv): {ISO_NAME} to {CENTER_NAME}; '
                     f'{_convention_note()}', quantities['drawing_note']):
            lines.extend(textwrap.wrap(note, width=TEXT_PANEL_WRAP_WIDTH))
        self._static_text.set_text('\n'.join(lines))

        compare_lines = self._comparison_lines()
        self._compare_text.set_text('\n'.join(compare_lines))
        self._compare_text.set_visible(bool(compare_lines))
        # New content starts from the panel's own font size; _fit_text_font
        # takes it down from there if the blocks are taller than the panel.
        if not keep_font:
            self._text_font_size = (COMPARING_FONT_SIZE
                                    if self.compare_scene is not None
                                    else TEXT_PANEL_FONT_SIZE)
        self._set_text_font(self._text_font_size)

    def _set_text_font(self, size):
        """Put one font size on the text panel's three blocks."""
        for artist in (self._static_text, self._compare_text,
                       self._footer_text):
            artist.set_fontsize(float(size))

    def _fit_text_font(self, renderer, panel):
        """Shrink the text panel's font until its blocks fit the panel.

        How tall the three blocks are depends on the geometry.  A parallel or
        multiaxis scan's drawing note is five or six lines where a cone scan's
        is two, and at the panel's own font size those blocks ran past the
        bottom of the panel and into the widget row.  The font size is
        therefore scaled by the room the panel has, which is measured with the
        renderer for the reason :meth:`_place_text_blocks` measures.  The scale
        is proportional, so one step lands close and the caller's next pass
        settles it.

        Returns:
            bool: whether the font changed, which means the figure has to be
            drawn again before the blocks can be placed.
        """
        blocks = [self._static_text]
        if self._compare_text.get_visible():
            blocks.append(self._compare_text)
        blocks.append(self._footer_text)
        counts = [artist.get_text().count('\n') + 1 for artist in blocks]
        # One blank line separates one block from the next.
        lines = sum(counts) + len(blocks) - 1
        extent = self._static_text.get_window_extent(renderer)
        line_height = extent.height / max(counts[0], 1)
        needed = lines * line_height
        room = TEXT_PANEL_FILL * panel.height
        # Half a line of tolerance, because a font's line height does not
        # scale exactly with its size: without the tolerance each pass found
        # the blocks a little too tall and shrank them again, and the passes
        # never settled.
        if needed <= room + 0.5 * line_height or line_height <= 0.0:
            return False
        size = max(TEXT_PANEL_FONT_MINIMUM,
                   self._text_font_size * room / needed)
        if size >= self._text_font_size - 0.05:
            return False
        self._text_font_size = size
        self._set_text_font(size)
        return True

    def _difference_groups(self):
        """The two geometries' differences, split into parameters and derived.

        The split is by name: a name that either geometry kind requires as a
        parameter is a parameter, and every other name is a derived quantity.
        ``GeometryScene.differences`` already returns the parameters first, so
        each list keeps the order it was given in.  The answer is computed once
        and kept, because computing it compares the derived quantities of both
        scenes, and both the text panel and the comparison window ask for it.

        Returns:
            (list, list): the parameter differences and the derived-quantity
            differences, each as ``(name, primary, comparison)`` triples.
        """
        if self._compare_differences is None:
            names = set(required_parameter_names(self.scene.kind))
            names.update(required_parameter_names(self.compare_scene.kind))
            parameters, derived = [], []
            for row in self.scene.differences(self.compare_scene):
                (parameters if row[0] in names else derived).append(row)
            self._compare_differences = (parameters, derived)
        return self._compare_differences

    def _comparison_lines(self):
        """The text panel's comparison section, as a list of lines.

        The section lists the parameters that differ, one per entry, as
        ``name: primary -> comparison``.  Those are the numbers a calibration
        user changed, so they are the ones to see beside the drawing.  The
        derived quantities that differ are counted in a last sentence instead,
        which points at the comparison window; that window holds the whole
        table.

        The parameter list is capped twice: at :data:`MAX_COMPARISON_ENTRIES`
        entries, and at the number of lines the panel has room for, which
        :meth:`_place_text_blocks` measures.  What is left out is counted in a
        line of its own.  When the room runs out, the sentence about the
        derived quantities goes first and the parameter entries last, because
        the entries are what a calibration user changed and the window holds
        the rest anyway.

        The section stays as it is while the comparison toggle is off, because
        these numbers are what a calibration user reads and the toggle hides a
        drawing and not a number.  The heading then says that the drawing is
        hidden.
        """
        if self.compare_scene is None:
            return []
        parameters, derived = self._difference_groups()
        heading = ('Comparison (dashed):' if self._show_compare
                   else 'Comparison (dashed, hidden):')
        lines = [heading]
        if not parameters and not derived:
            lines.append('  nothing differs')
            return lines

        blocks = [textwrap.wrap(_difference_text(name, mine, theirs),
                                width=TEXT_PANEL_WRAP_WIDTH,
                                initial_indent='  ',
                                subsequent_indent='      ')
                  for name, mine, theirs in
                  parameters[:MAX_COMPARISON_ENTRIES]]
        limit = self._compare_line_limit
        shown = 0
        for block in blocks:
            if limit is not None and len(lines) + len(block) + 1 > limit:
                break
            lines.extend(block)
            shown += 1
        left_out = len(parameters) - shown
        if left_out:
            noun = 'parameter' if left_out == 1 else 'parameters'
            lines.append(f'  and {left_out} more {noun}')
        sentence = textwrap.wrap(_derived_difference_sentence(len(derived)),
                                 width=TEXT_PANEL_WRAP_WIDTH,
                                 initial_indent='  ', subsequent_indent='  ')
        if limit is None or len(lines) + len(sentence) <= limit:
            lines.extend(sentence)
        return lines

    def _footer_lines(self):
        """The text panel's footer, as a list of lines."""
        lines = [f'view drawn : {self._view_index}']
        lines.extend(textwrap.wrap('view: ' + self._view_label(),
                                   width=TEXT_PANEL_WRAP_WIDTH))
        if self.compare_scene is not None:
            index = self._compare_index(self._view_index)
            if index != self._view_index:
                lines.extend(textwrap.wrap(
                    f'comparison holds view {index}: '
                    + self._view_label(self.compare_scene, index),
                    width=TEXT_PANEL_WRAP_WIDTH))
        # One line names the data overlays drawn, because neither of them has
        # a legend entry of its own.
        overlays = self._overlay_names()
        if overlays:
            lines.extend(textwrap.wrap('overlays : ' + '; '.join(overlays),
                                       width=TEXT_PANEL_WRAP_WIDTH))
        return lines

    # --- the legends ---

    def _create_legends(self):
        """Build the legend of the 3D panel and of the detector face.

        The legends are rebuilt when a comparison is added or removed, because
        the comparison adds an entry to each of them, and when a phantom is
        added or removed, because the phantom adds an entry to each of them
        too.  The phantom's entries come from its outline in the 3D panel and
        from its projected outline on the detector face.  An entry stays while
        the phantom's toggle hides those outlines: a legend names what the
        figure holds, and the toggle says whether it is drawn.

        The 3D panel's legend sits in the panel's upper left corner, where the
        drawing leaves room.  The detector face's legend sits outside its
        panel, in the band under it: inside the panel it covered the projected
        outlines and the sinogram there.  The legend is placed in the figure's
        own coordinates rather than the panel's, because the panel's box
        changes shape with the detector's row and channel counts while the
        band does not.  The legend still belongs to the detector's axes, so it
        carries that panel's entries and is drawn with that panel.
        """
        self.ax_3d.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE,
                          framealpha=0.85, borderpad=0.3, labelspacing=0.25)
        cell = self.ax_detector.get_subplotspec().get_position(self.figure)
        # borderaxespad is the gap a legend leaves between its anchor and
        # itself.  It is zero here, so that the anchor is the legend's own
        # bottom edge and the band it sits in can be read from the constants.
        self.ax_detector.legend(
            loc='lower center',
            bbox_to_anchor=(0.5 * (cell.x0 + cell.x1),
                            DETECTOR_LEGEND_BOTTOM),
            bbox_transform=self.figure.transFigure,
            ncol=DETECTOR_LEGEND_COLUMNS, borderaxespad=0.0,
            fontsize=LEGEND_FONT_SIZE, framealpha=0.85, borderpad=0.3,
            labelspacing=0.25)

    def _place_text_blocks(self):
        """Stack the text panel's three blocks from the top of the panel.

        The static block, the comparison block, and the footer follow one
        another with a blank line between them.  Where one block ends is
        measured with the renderer rather than estimated from the font size,
        because the line spacing of a text block depends on the font
        matplotlib resolves.  An estimate put the blocks on top of each other.

        Returns:
            bool: whether a block moved, which means the figure has to be
            drawn again.
        """
        canvas = self.figure.canvas
        if not hasattr(canvas, 'get_renderer'):
            return False
        renderer = canvas.get_renderer()
        panel = self.ax_text.get_window_extent(renderer)
        if panel.height <= 0.0:
            return False

        def line_height(artist):
            extent = artist.get_window_extent(renderer)
            count = artist.get_text().count('\n') + 1
            return extent, extent.height / max(count, 1)

        def place(artist, top):
            """Put an artist's top at a display height.

            Returns:
                bool: whether the artist moved.
            """
            target = float((top - panel.y0) / panel.height)
            if abs(target - float(artist.get_position()[1])) < 0.004:
                return False
            artist.set_position((0.0, target))
            return True

        # The font size is chosen first, because the places of the blocks
        # depend on how tall a line is.  Measuring an artist reports its
        # current font and not the font of the last draw, so the size and the
        # places settle in one pass.
        font_changed = self._fit_text_font(renderer, panel)
        moved = font_changed
        extent, height = line_height(self._static_text)
        next_top = extent.y0 - height
        if self._compare_text.get_visible():
            moved |= place(self._compare_text, next_top)
            extent, compare_height = line_height(self._compare_text)
            next_top = extent.y0 - compare_height
        moved |= place(self._footer_text, next_top)
        # The comparison section is cut only in a pass that left the font
        # alone.  A smaller font gives every block more lines, and the cut can
        # never be taken back, so cutting in the same pass that shrinks the
        # font drops entries the panel turns out to have room for.  That is
        # what happened when the fit statement grew a second line.
        if self._compare_text.get_visible() and not font_changed:
            moved |= self._limit_comparison_lines(renderer, panel,
                                                  compare_height)
        return moved

    def _limit_comparison_lines(self, renderer, panel, line_height):
        """Cut the comparison section down to the lines the panel has room for.

        The three text blocks together can be taller than the panel, which a
        comparison between two unrelated geometries makes likely.  The
        comparison section is the one that gives way, because the derived
        quantities and the view drawn are what the panel is for.  The limit
        only ever falls, so this settles instead of adding and dropping the
        same entry.

        Returns:
            bool: whether the section was cut, which means the figure has to be
            drawn again.
        """
        compare = self._compare_text.get_window_extent(renderer)
        footer = self._footer_text.get_window_extent(renderer)
        room = compare.y1 - panel.y0 - footer.height - line_height
        fits = int(max(room, 0.0) // max(line_height, 1.0))
        current = self._compare_text.get_text().count('\n') + 1
        if fits >= current:
            return False
        limit = fits if self._compare_line_limit is None else min(
            fits, self._compare_line_limit)
        if limit == self._compare_line_limit:
            return False
        self._compare_line_limit = limit
        self._update_static_text(keep_font=True)
        return True

    # ------------------------------------------------------------------
    # The comparison overlay
    # ------------------------------------------------------------------

    def _install_compare(self, compare):
        """Build the comparison scene, its artists, and its window."""
        self.compare_scene = self._as_compare_scene(compare)
        self._compare_quantities = self.compare_scene.derived_quantities()
        self._compare_differences = None
        self._compare_trajectory = None
        self._create_compare_artists()
        self._create_compare_window()
        self._update_static_text()

    def _create_compare_window(self):
        """Build the second figure, which tables every difference.

        Why a window of its own.  The text panel has room for a few lines, and
        two geometries that differ in several parameters differ in many derived
        quantities, so the panel could not show them all.  This window has one
        row per difference and no cap but its own height, and the text panel
        keeps the parameters and a count.

        What the table holds.  The three columns are the name, the primary
        geometry's value, and the comparison's.  The parameters come first,
        then a row of dashes, then the derived quantities.  Every row is one
        text artist in a monospace font, with its three columns padded to a
        fixed width, so the columns line up whatever the values are.  The font
        is taken down from :data:`COMPARE_WINDOW_FONT_SIZE` when the longest
        row would otherwise reach past the window's width.

        The window is drawn on one axes with its own axis turned off.  The axes
        covers the table's area exactly, so a row's place in it is its row
        number over the row count, and the spacing on the screen is
        :data:`COMPARE_ROW_HEIGHT_IN`.
        """
        lines = self._compare_table_lines()
        height = min(COMPARE_WINDOW_MAX_HEIGHT_IN,
                     COMPARE_WINDOW_TITLE_IN + COMPARE_WINDOW_MARGIN_IN
                     + len(lines) * COMPARE_ROW_HEIGHT_IN)
        self.compare_figure = plt.figure(
            figsize=(COMPARE_WINDOW_WIDTH_IN, height))
        self.compare_figure.suptitle(COMPARE_WINDOW_TITLE,
                                     fontsize=TITLE_FONT_SIZE + 1)
        # The desktop user has two windows open, so the second one says what it
        # is.  A backend with no window has no manager to tell, which is what
        # the check is for.
        manager = getattr(self.compare_figure.canvas, 'manager', None)
        if hasattr(manager, 'set_window_title'):
            manager.set_window_title(COMPARE_WINDOW_NAME)

        side = COMPARE_WINDOW_MARGIN_IN / COMPARE_WINDOW_WIDTH_IN
        bottom = COMPARE_WINDOW_MARGIN_IN / height
        top = 1.0 - COMPARE_WINDOW_TITLE_IN / height
        axes = self.compare_figure.add_axes(
            (side, bottom, 1.0 - 2.0 * side, top - bottom))
        axes.set_axis_off()

        room_points = 72.0 * (COMPARE_WINDOW_WIDTH_IN
                              - 2.0 * COMPARE_WINDOW_MARGIN_IN)
        longest = max(len(line) for line in lines)
        size = min(COMPARE_WINDOW_FONT_SIZE,
                   room_points / (MONOSPACE_ADVANCE * longest))
        for index, line in enumerate(lines):
            axes.text(0.0, 1.0 - (index + 0.5) / len(lines), line,
                      transform=axes.transAxes, family='monospace',
                      fontsize=size, va='center', ha='left',
                      # The first row is the column headings.
                      fontweight='bold' if index == 0 else 'normal')

    def _compare_table_lines(self):
        """The comparison window's table, as a list of monospace lines.

        The first line is the column headings, and a line of dashes separates
        the parameters from the derived quantities.  The table is cut to the
        rows the tallest window holds, and the rows left out are counted in a
        last line.

        Returns:
            list of str: the lines, each with its three columns padded so that
            the columns line up.
        """
        parameters, derived = self._difference_groups()
        rows = [COMPARE_TABLE_HEADING]
        for name, mine, theirs in parameters + derived:
            rows.append((name, *_difference_values(mine, theirs)))
        if not parameters and not derived:
            rows.append(('nothing differs', '', ''))
        widths = [max(len(row[column]) for row in rows)
                  for column in range(len(COMPARE_TABLE_HEADING))]
        # The rule sits where the parameters end, which is one row past the
        # headings.  It is drawn as dashes in every column, so it reads as a
        # line across the table.  A table with only one of the two kinds in it
        # needs no rule.
        if parameters and derived:
            rows.insert(1 + len(parameters),
                        tuple('-' * width for width in widths))

        gap = ' ' * COMPARE_COLUMN_GAP
        lines = [gap.join(entry.ljust(width)
                          for entry, width in zip(row, widths)).rstrip()
                 for row in rows]
        fits = int((COMPARE_WINDOW_MAX_HEIGHT_IN - COMPARE_WINDOW_TITLE_IN
                    - COMPARE_WINDOW_MARGIN_IN) // COMPARE_ROW_HEIGHT_IN)
        if len(lines) > fits:
            left_out = len(lines) - (fits - 1)
            lines = lines[:fits - 1] + [f'and {left_out} more']
        return lines

    def _as_compare_scene(self, compare):
        """A ``GeometryScene`` for the comparison.

        A dictionary of parameter overrides is applied to a copy of this
        scene's parameters, so that a comparison can be asked for as
        ``compare=dict(det_channel_offset=12.5)``.  A model is read the same
        way the primary scene was, with the same drawing options.
        """
        if isinstance(compare, GeometryScene):
            return compare
        if isinstance(compare, dict):
            return self.scene.with_parameters(compare)
        return GeometryScene.from_model(compare,
                                        **self.scene.drawing_options())

    def _create_compare_artists(self):
        """Create the comparison's artists in all four drawing panels.

        The comparison is drawn in one color with dashed lines: its detector
        outline, its source, its central ray, its projected volume outline on
        the detector face, and its pixel-0 marker.  The volume box is shared
        unless the two volumes differ, in which case the comparison's box is
        drawn too.

        On the detector face the comparison draws its volume box and not its
        region of reconstruction.  Two dashed ellipses over the primary's two
        would make that panel hard to read, and the primary's own ellipse is
        the one the fit statement is about.
        """
        color = COLORS['compare']
        dashed = dict(linestyle=COMPARE_DASHES, linewidth=COMPARE_LINEWIDTH)

        def moving(axes, three_d=False, label=None, **kwargs):
            if three_d:
                line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                  color=color, **AXLIM_CLIP, label=label,
                                  **kwargs)
            else:
                line, = axes.plot([], [], color=color, label=label, **kwargs)
            self._compare_moving.append((axes, line))
            return line

        self._compare = {}
        self._compare['detector_3d'] = moving(
            self.ax_3d, three_d=True, label='comparison', **dashed)
        self._compare['central_3d'] = moving(self.ax_3d, three_d=True,
                                             **dashed)
        self._compare['source_3d'] = moving(
            self.ax_3d, three_d=True, marker='*',
            markersize=SOURCE_MARKER_SIZE - 3, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)
        self._compare['pixel0_3d'] = moving(
            self.ax_3d, three_d=True, marker='+', linestyle='none',
            markersize=INDEX_MARKER_SIZE, markeredgewidth=INDEX_MARKER_WIDTH)
        for name, axes in (('top', self.ax_top), ('side', self.ax_side)):
            self._compare[f'detector_{name}'] = moving(axes, **dashed)
            self._compare[f'central_{name}'] = moving(axes, **dashed)
            self._compare[f'source_{name}'] = moving(
                axes, marker='*', markersize=SOURCE_MARKER_SIZE - 3,
                linestyle='none', markerfacecolor='none', markeredgewidth=1.4)
            self._compare[f'pixel0_{name}'] = moving(
                axes, marker='+', linestyle='none',
                markersize=INDEX_MARKER_SIZE,
                markeredgewidth=INDEX_MARKER_WIDTH)
        # The comparison is named once, beside its detector in the top view,
        # because a label in every panel would say the same thing four times.
        # It hangs farther below the detector than the primary's labels do, so
        # that it does not run into the channel-offset label.
        label = self.ax_top.annotate(
            'comparison', xy=(0.0, 0.0), textcoords='offset points',
            xytext=(LABEL_GAP_POINTS, -14), fontsize=ANNOTATION_FONT_SIZE,
            color=color, ha='left', va='top')
        self._compare_moving.append((self.ax_top, label))
        self._compare['label_top'] = label
        self._compare['edges_detector'] = moving(
            self.ax_detector, label='comparison', **dashed)
        self._compare['landing_detector'] = moving(
            self.ax_detector, marker='o', markersize=6, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)
        self._compare['center_detector'] = moving(
            self.ax_detector, marker='s', markersize=5, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)

        # The volume box is the object, and the object is what a comparison
        # usually shares.  Only a comparison whose recon parameters differ gets
        # its own box, drawn as a static artist because the object does not
        # move with the view.
        compare_corners = self.compare_scene.volume_corners()
        if not np.allclose(compare_corners, self.scene.volume_corners()):
            box = _joined([compare_corners[[first, second]]
                           for first, second in VOLUME_BOX_EDGES])
            line, = self.ax_3d.plot(box[:, 0], box[:, 1], box[:, 2],
                                    color=color, **AXLIM_CLIP, **dashed)
            self._compare_static.append((self.ax_3d, line))
            for axes, walk, columns in (
                    (self.ax_top, _XY_FOOTPRINT_WALK, TOP_PANEL_COLUMNS),
                    (self.ax_side, _YZ_FACE_WALK, SIDE_PANEL_COLUMNS)):
                points = compare_corners[list(walk)]
                line, = axes.plot(points[:, columns[0]],
                                  points[:, columns[1]], color=color,
                                  **dashed)
                self._compare_static.append((axes, line))

        # The comparison's own detector grid, when its detector has a different
        # shape.  An identical grid would only draw a dashed line on top of the
        # primary's rectangle.
        if (self.compare_scene.num_det_rows != self.scene.num_det_rows
                or self.compare_scene.num_det_channels
                != self.scene.num_det_channels):
            grid = self.ax_detector.add_patch(Rectangle(
                (-0.5, -0.5), self.compare_scene.num_det_channels,
                self.compare_scene.num_det_rows, facecolor='none',
                edgecolor=color, linewidth=COMPARE_LINEWIDTH,
                linestyle='--'))
            self._compare_static.append((self.ax_detector, grid))

        # The comparison's source path follows the source-path toggle as the
        # primary's does, and the comparison's own toggle as well.
        self._compare_trajectory_lines = []
        for axes in (self.ax_3d, self.ax_top, self.ax_side):
            if axes is self.ax_3d:
                line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                  color=color, linewidth=1.0, linestyle='-.',
                                  **AXLIM_CLIP)
            else:
                line, = axes.plot([], [], color=color, linewidth=1.0,
                                  linestyle='-.')
            self._compare_static.append((axes, line))
            self._compare_trajectory_lines.append((axes, line))
        self._apply_compare_visibility()
        self._set_animated(self._animate_moving())

    def _apply_compare_visibility(self):
        """Show or hide every comparison artist, following the toggles.

        A comparison artist is drawn when the comparison toggle is on.  The
        comparison's source path answers to the source-path toggle as well, so
        that the two paths are drawn together or not at all.
        """
        for _, artist in self._compare_moving + self._compare_static:
            artist.set_visible(self._show_compare)
        for _, line in self._compare_trajectory_lines:
            line.set_visible(self._show_compare and self._show_trajectory)

    def _remove_compare_artists(self):
        """Remove every comparison artist from its axes."""
        for _, artist in self._compare_moving + self._compare_static:
            artist.remove()
        self._compare_moving = []
        self._compare_static = []
        self._compare_trajectory_lines = []
        self._compare = {}

    # ------------------------------------------------------------------
    # The data overlays
    # ------------------------------------------------------------------

    def _install_sinogram(self, sinogram, vmin=None, vmax=None):
        """Create, replace, or remove the sinogram image on the detector face.

        The image is a moving artist, because a view change replaces its data
        with that view of the array.  It is drawn above the detector's
        translucent rectangle and below every line and marker, so the projected
        outlines of the volume box and of the region of reconstruction stay
        readable over it.  The gray scale is fixed over the whole array and not
        per view, so stepping through the views compares them.

        A detector larger than :data:`SINOGRAM_DISPLAY_PIXELS` pixels across is
        subsampled for display, on the device that holds the array, and the
        image's extent puts each kept sample at the detector row and channel it
        was taken from.

        Args:
            sinogram (array_like): the array, or None to remove the image.
            vmin, vmax (float, optional): the gray scale's ends.  None takes
                the smallest and the largest value of the sinogram as
                displayed.
        """
        _reject_divided('sinogram', sinogram)
        if self._sinogram_image is not None:
            self._sinogram_image.remove()
            self._moving = [pair for pair in self._moving
                            if pair[1] is not self._sinogram_image]
            self._sinogram_image = None
        self._sinogram = None
        self._sinogram_stride = 1
        if sinogram is None:
            return
        shape = _shape_of(sinogram)
        expected = tuple(self.scene.sinogram_shape)
        if shape != expected:
            raise ValueError(f'The sinogram has shape {shape}, and '
                             f"this scan's sinogram shape is {expected}.")
        # The same stride in both directions keeps the kept pixels square.
        # The array is sliced as it was given, so a tensor is subsampled on
        # its own device and only the kept pixels are brought to the host.
        largest = max(self.scene.num_det_rows, self.scene.num_det_channels)
        stride = max(1, -(-largest // SINOGRAM_DISPLAY_PIXELS))
        values = _to_host(sinogram[:, ::stride, ::stride])
        self._sinogram = values
        self._sinogram_stride = stride
        axes = self.ax_detector
        # The extent puts kept sample (a, b) at data coordinates channel
        # b * stride and row a * stride, which is the detector pixel the
        # sample was taken from.  With a stride of 1 these are the numbers
        # imshow uses by default with origin='upper'.  They are written out
        # because this panel's limits come from the geometry and not from the
        # image.  The panel's row axis is inverted, so row 0 is drawn at the
        # top, which is how imshow shows one view of a sinogram.
        extent = (-0.5 * stride, (values.shape[2] - 0.5) * stride,
                  (values.shape[1] - 0.5) * stride, -0.5 * stride)
        low = float(np.min(values)) if vmin is None else float(vmin)
        high = float(np.max(values)) if vmax is None else float(vmax)
        self._sinogram_image = axes.imshow(
            values[self._view_index], cmap=SINOGRAM_COLORMAP,
            interpolation='nearest', origin='upper', extent=extent,
            vmin=low, vmax=high, zorder=SINOGRAM_ZORDER)
        self._moving.append((axes, self._sinogram_image))
        self._apply_sinogram_visibility()
        self._set_animated(self._animate_moving())

    def _apply_sinogram_visibility(self):
        """Show or hide the painted sinogram, following its toggle."""
        if self._sinogram_image is not None:
            self._sinogram_image.set_visible(self._show_sinogram)

    def _install_recon(self, recon, threshold=None):
        """Create, replace, or remove the reconstruction silhouette.

        The silhouette is two images, one in the top view and one in the side
        view.  Each is the support projected along the one object coordinate
        its panel does not draw.  Each of those panels also gets the outline of
        the support, and the 3D panel gets the outlines of the support in a few
        planes across the direction the support is thinnest along.  All of
        these are static artists, because the object does not move with the
        view.  The detector face gets those same outlines projected onto it,
        which is a moving artist, because the view is what the projection
        depends on.

        The array is never held as a whole boolean support.  It is read a chunk
        of slices at a time, each chunk is thresholded where it lives, and only
        the three projections and the section planes reach the host.

        Both legends are built again at the end, because the 3D outline carries
        the phantom's entry in the 3D panel's legend and the projected outline
        carries its entry in the detector face's legend.  A call that removes
        the phantom therefore takes both entries out.

        Args:
            recon (array_like): the array, or None to remove the silhouette.
            threshold (float, optional): the absolute value a voxel must exceed
                to belong to the support.  None takes
                :data:`DEFAULT_RECON_THRESHOLD_FRACTION` of the largest
                absolute value in the array.
        """
        _reject_divided('recon', recon)
        for _, artist in self._recon_images + self._recon_outlines:
            artist.remove()
        # The projected outline is a moving artist, so it also leaves the list
        # the partial redraw walks; an artist left there after its axes has
        # dropped it would be drawn on the next view change.
        self._moving = [pair for pair in self._moving
                        if pair[1] is not self._recon_detector_line]
        self._recon_detector_line = None
        self._recon_images = []
        self._recon_outlines = []
        self._recon_outline_parts = []
        self._recon_projections = None
        self._recon_section_axis = None
        self._recon_section_indices = []
        self._recon_section_total = 0
        self._recon_section_factors = []
        self._recon_threshold = None
        self._recon_threshold_given = None
        if recon is None:
            self._create_legends()
            return
        shape = _shape_of(recon)
        expected = tuple(self.scene.recon_shape)
        if shape != expected:
            raise ValueError(f'The reconstruction has shape {shape}, '
                             f"and this scan's recon shape is {expected}.")
        if threshold is None:
            level = (DEFAULT_RECON_THRESHOLD_FRACTION
                     * _largest_magnitude(recon))
        else:
            level = float(threshold)
            self._recon_threshold_given = level
        self._recon_threshold = level
        self._recon_projections = _support_projections(recon, level)
        # An empty support draws two empty silhouettes and no outline, so the
        # text panel still names the array that was given.
        # The top view is the xy plane, so the support is projected along z,
        # which is the slice index.  What is left is indexed (row i, column j),
        # which is (y, x); the panel draws y across the screen and x down it,
        # so the transpose puts x on the image's rows.  The side view is the yz
        # plane, so the support is projected along x, which is the column
        # index, and the same transpose puts z on the image's rows.
        self._create_silhouette(self.ax_top, self._recon_projections['xy'].T,
                                TOP_PANEL_COLUMNS)
        self._create_silhouette(self.ax_side, self._recon_projections['yz'].T,
                                SIDE_PANEL_COLUMNS)
        self._recon_outline_parts = self._support_outline_parts(recon, level)
        if self._recon_outline_parts:
            self._create_support_outline_3d()
            self._create_recon_detector_outline()
        self._apply_recon_visibility()
        self._create_legends()
        # The projected outline joined the moving artists, so it takes the
        # animated flag the partial-redraw path gives them.
        self._set_animated(self._animate_moving())

    def _create_silhouette(self, axes, support, columns):
        """Draw one panel's silhouette image and its outline, and keep them.

        The fill is one color drawn through a masked array, so the outside of
        the support is transparent and the panel's own lines read through it.
        The fill is also light, which leaves the support faint against a busy
        panel.  The outline of that same support is therefore drawn over the
        fill, as a solid line through the outer faces of its voxels.

        Args:
            axes: the panel.
            support (ndarray): the support projected onto this panel's plane,
                indexed first by the voxel index the panel draws down the
                screen and then by the one it draws across.
            columns (tuple): which object coordinates the panel puts on its
                horizontal and its vertical axis, as indices into (x, y, z).
        """
        horizontal, vertical = columns
        low, high = self._volume_box_corners()
        # imshow's extent is (left, right, bottom, top) in data coordinates,
        # and origin='upper' puts the array's first row at the "top" value.
        # The array's first row and its first column are the voxels at index 0,
        # which sit at the ``low`` corner, so "top" and "left" are that
        # corner's coordinates.  The panel's inverted axes then turn the image
        # the same way they turn every line drawn over it.
        extent = (low[horizontal], high[horizontal],
                  high[vertical], low[vertical])
        filled = np.ma.masked_where(~support, np.ones(support.shape))
        image = axes.imshow(filled, cmap=ListedColormap([COLORS['volume']]),
                            interpolation='nearest', origin='upper',
                            extent=extent, alpha=RECON_FILL_ALPHA,
                            vmin=0.0, vmax=1.0, zorder=SILHOUETTE_ZORDER)
        self._recon_images.append((axes, image))

        # The cells of that image, as their edges in the panel's own
        # coordinates.  The image spans the volume box, and one cell of it is
        # one voxel, so a row of the image has one more edge than it has cells.
        down = np.linspace(low[vertical], high[vertical],
                           support.shape[0] + 1)
        across = np.linspace(low[horizontal], high[horizontal],
                             support.shape[1] + 1)
        outline = _mask_outline(support, across, down)
        if outline.shape[0] == 0:
            return
        line, = axes.plot(outline[:, 0], outline[:, 1],
                          color=COLORS['volume'],
                          linewidth=RECON_OUTLINE_LINEWIDTH,
                          zorder=RECON_OUTLINE_ZORDER)
        self._recon_outlines.append((axes, line))

    def _support_outline_parts(self, values, level):
        """The phantom's outline in three dimensions, one part per section.

        A section is the outline of the support in one plane, and the sections
        lie across the axis the support is thinnest along, so that each of them
        cuts the support where it is widest.  A phantom need not be a box and
        need not be one connected piece, and a section says what the support is
        in its own plane, holes and separate pieces included.

        At most :data:`PHANTOM_SECTION_COUNT` sections are drawn, spread evenly
        from the first plane the support reaches to the last.  Each may hold
        its share of :data:`PHANTOM_OUTLINE_POINT_BUDGET` points, and a section
        whose outline would be longer is drawn from a coarsened mask, so a
        field of small blobs stays a drawing the panel can hold rather than
        tens of thousands of points.

        Only the planes the sections cover are read from the array, and each is
        thresholded where it lives.  Every position comes from the scene's own
        ``voxel_centers``, at fractional voxel indices, so the viewer computes
        no position of its own.

        Args:
            values (array_like): the reconstruction, as (rows, cols, slices).
            level (float): the absolute value a voxel must exceed to belong to
                the support.

        Returns:
            list of ndarray: one part per section, each (N, 3) as (x, y, z)
            with a row of NaN between one segment of the section and the next.
            The list is empty when the support has no voxel in it.
        """
        axis = _section_axis(self._recon_projections)
        if axis is None:
            return []
        low, high = _section_extents(self._recon_projections)[axis]
        kind = PHANTOM_SECTION_KIND
        sections = _section_positions(low, high, PHANTOM_SECTION_COUNT, kind)
        share = PHANTOM_OUTLINE_POINT_BUDGET // len(sections)
        parts, factors = [], []
        for position, planes in sections:
            mask = _section_mask(values, axis, planes, level)
            outline, factor = _budgeted_outline(mask, share)
            factors.append(factor)
            parts.append(self._section_points(axis, position, outline))
        self._recon_section_axis = axis
        self._recon_section_indices = [position for position, _ in sections]
        self._recon_section_total = high - low + 1
        self._recon_section_factors = factors
        return parts

    def _section_points(self, axis, position, outline):
        """One section's outline in object coordinates, (N, 3) as (x, y, z).

        The outline comes in as (across, down) fractional indices in the
        section's own plane, and which voxel index each of those is depends on
        the axis the sections lie across; see :func:`_section_mask`.  The third
        index is the section's position along that axis.  The rows of NaN that
        separate one segment from the next stay NaN.

        Args:
            axis (str): ``'x'``, ``'y'``, or ``'z'``.
            position (float): the section's plane index.
            outline (ndarray): (N, 2) as (across, down).
        """
        across, down = outline[:, 0], outline[:, 1]
        plane = np.full(outline.shape[0], float(position))
        if axis == 'z':
            ijk = np.column_stack([down, across, plane])
        elif axis == 'y':
            ijk = np.column_stack([plane, down, across])
        else:
            ijk = np.column_stack([down, plane, across])
        points = self.scene.voxel_centers(ijk)
        points[~np.isfinite(ijk).all(axis=1)] = np.nan
        return points

    def _section_label(self):
        """What the 3D panel's legend calls the phantom's outline.

        The label counts the sections drawn against the planes the support
        reaches, because the panel draws a few planes of the support and not
        its whole shape, and a reader has to know which.
        """
        word = 'slabs' if PHANTOM_SECTION_KIND == 'slab' else 'sections'
        drawn = len(self._recon_section_indices)
        total = self._recon_section_total
        axis = self._recon_section_axis
        if drawn < total:
            return f'{PHANTOM_NAME} ({drawn} of {total} {word} across {axis})'
        return f'{PHANTOM_NAME} ({total} {word} across {axis})'

    def _create_support_outline_3d(self):
        """Draw the phantom's outline in the 3D panel, and keep it.

        The 3D panel draws no silhouette, because a filled shape there would
        hide the geometry behind it.  It draws this outline instead, which says
        where the phantom sits in all three directions.  The outline is dashed,
        so that it is not read as the volume box, which is the same color and
        solid.  Its sections, which :meth:`_support_outline_parts` builds, are
        drawn as one line whose data carries a row of NaN between one polyline
        and the next.
        """
        points = _joined(self._recon_outline_parts)
        line, = self.ax_3d.plot(points[:, 0], points[:, 1], points[:, 2],
                                color=COLORS['volume'],
                                linewidth=RECON_OUTLINE_3D_LINEWIDTH,
                                linestyle='--', label=self._section_label(),
                                **AXLIM_CLIP)
        self._recon_outlines.append((self.ax_3d, line))

    def _apply_recon_visibility(self):
        """Show or hide the phantom's fills and outlines, following its toggle.

        The two fills, the outline in each projected panel, the outline in the
        3D panel, and the projected outline on the detector face are one
        drawing of one array, so one flag governs them all.
        """
        for _, artist in self._recon_images + self._recon_outlines:
            artist.set_visible(self._show_recon)

    def _volume_box_corners(self):
        """The volume box's corner at voxel (0, 0, 0) and its opposite one.

        Each corner is (x, y, z).  Both come from the scene's own
        ``voxel_centers``, at the fractional voxel indices half a voxel outside
        the first voxel and the last one, so the box holds every voxel and the
        viewer computes no position of its own.  The voxel pitches are
        positive, so the first of the two is the smaller value on every axis.
        """
        rows, cols, slices = self.scene.recon_shape
        corners = self.scene.voxel_centers(
            [[-0.5, -0.5, -0.5],
             [rows - 0.5, cols - 0.5, slices - 0.5]])
        return corners[0], corners[1]

    def _overlay_names(self):
        """What the text panel's footer calls the overlays drawn, as a list.

        The line names what is drawn and not what the figure holds, so an
        overlay whose toggle is off is left out of it.  With both toggles off
        the footer has no overlay line at all, which is the same line a figure
        with no overlay shows.  The state of each toggle is in the widget row,
        beside its label.

        The threshold is named the way it was chosen: a caller's own threshold
        is printed as the number it is, and the default is printed as the
        fraction of the largest absolute value that it is.  A phantom whose
        section outlines were coarsened says so too, because a coarsened
        outline covers the support rather than following it exactly.
        """
        names = []
        if self._sinogram_image is not None and self._show_sinogram:
            names.append('sinogram')
        if self._recon_images and self._show_recon:
            if self._recon_threshold_given is None:
                fraction = _three_figures(DEFAULT_RECON_THRESHOLD_FRACTION)
                entry = f'recon above {fraction} max'
            else:
                entry = ('recon above '
                         + _three_figures(self._recon_threshold_given))
            coarsest = max(self._recon_section_factors, default=1)
            if coarsest > 1:
                entry += f', outline coarsened by {coarsest}'
            names.append(entry)
        return names

    # ------------------------------------------------------------------
    # Updating the artists for one view
    # ------------------------------------------------------------------

    def _refresh(self, rebuild_limits=False):
        """Update every moving artist for the current view and repaint.

        The panel limits are rebuilt when they are asked for, when they are
        not set yet, or when the current view's content would fall outside
        them.  A rebuild moves the ticks, so it forces a full repaint; the
        common case reuses the background and repaints only what moved.

        Args:
            rebuild_limits (bool): whether to rebuild the limits in any case.
        """
        view = self.scene.view(self._view_index)
        compare_view = self._compare_view()
        if rebuild_limits or not self._limits_hold(view, compare_view):
            self._compute_limits(view, compare_view)
            self._apply_limits()
            rebuild_limits = True

        self._update_moving_artists(view, compare_view)
        if rebuild_limits:
            self._full_redraw()
        else:
            self._fast_redraw()

    def _update_moving_artists(self, view, compare_view):
        """Replace the data of every moving artist, in place."""
        self._update_3d_panel(view)
        self._update_projected_panel(self._top, view, TOP_PANEL_COLUMNS,
                                     TOP_EDGE_RAYS,
                                     self.scene.det_channel_offset,
                                     CHANNEL_OFFSET_LABEL)
        self._update_projected_panel(self._side, view, SIDE_PANEL_COLUMNS,
                                     SIDE_EDGE_RAYS, self.scene.row_offset,
                                     ROW_OFFSET_LABEL)
        self._place_slice_offset_label(view)
        self._update_arc_and_trajectory(view)
        self._update_detector_panel(view)
        self._clip_3d_artists(view)
        self._footer_text.set_text('\n'.join(self._footer_lines()))
        self._update_titles()
        if compare_view is not None:
            self._update_compare_artists(compare_view)

    def _clip_3d_artists(self, view):
        """Hide the 3D artists that lie outside the 3D panel's cube.

        Two kinds of 3D artist are not clipped by the axes limits, the way a
        3D line drawn with ``AXLIM_CLIP`` is under matplotlib 3.10 and later:
        a text artist and an arrowhead built by ``quiver``.  In the volume zoom the source sits far outside
        the cube, and its label would be drawn where that point projects,
        which is outside the panel and on top of the rest of the figure.  Each
        such artist is hidden when the point it is attached to is outside the
        cube.

        Args:
            view (ViewScene): the current view's primitives.
        """
        cube = (self.ax_3d.get_xlim(), self.ax_3d.get_ylim(),
                self.ax_3d.get_zlim())
        arc = view.rotation_direction_arc
        artists = [(self._source_text_3d, view.source_draw, True),
                   (self._detector_text_3d, _far_corner(view, (0, 1, 2)),
                    True),
                   (self._iso_text_3d, view.detector_origin, True),
                   (self._arc_text_3d, None if arc is None else arc[-1],
                    arc is not None),
                   (self._arrow_3d, None if arc is None else arc[-2],
                    arc is not None and self._arrow_3d is not None),
                   (self._reference_text_3d,
                    self._reference_text_3d.get_position_3d(),
                    self._show_reference),
                   (self._reference_arrow_3d, self._reference_arrow_point,
                    self._show_reference)]
        for artist, point, drawn in artists:
            if artist is None:
                continue
            if not drawn:
                artist.set_visible(False)
                continue
            point = np.asarray(point, dtype=np.float64).reshape(1, 3)
            artist.set_visible(_within(point, cube))

    def _update_titles(self):
        """Put the view drawn into the titles of the drawing panels."""
        label = self._view_label()
        self._title_3d.set_text(f'3D view, view {self._view_index}, {label}')
        # A sinogram painted on the detector face gets no legend entry, so the
        # title is where the panel says that it is drawn.  A sinogram the
        # toggle hides is not drawn, so the title does not name it.
        painted = self._sinogram_image is not None and self._show_sinogram
        note = ', with sinogram' if painted else ''
        # The row order is a display choice, so the detector panel's title
        # names it on its own line above the view drawn.
        self._title_detector.set_text(
            f'{_detector_view_title()}\n'
            f'view {self._view_index}, {label}{note}')

    def _update_3d_panel(self, view):
        """Update the 3D panel's moving artists."""
        self._source_3d.set_data_3d(*_point_3d(view.source_draw))
        outline = view.detector_outline
        self._detector_3d.set_data_3d(outline[:, 0], outline[:, 1],
                                      outline[:, 2])
        if self._face_3d is not None:
            self._face_3d.set_verts([view.detector_corners])
            # A 3D collection projects its vertices when the whole axes is
            # drawn.  A partial redraw draws the artist alone, so the
            # projection is asked for here.  Before the first draw the axes
            # has no projection matrix, and that first draw makes one.
            if self.ax_3d.M is not None:
                self._face_3d.do_3d_projection()
        rays = _joined([_sampled_segment(ray[0], ray[1])
                        for ray in view.corner_rays])
        self._rays_3d.set_data_3d(rays[:, 0], rays[:, 1], rays[:, 2])
        central = _sampled_segment(view.source_draw, view.detector_origin)
        self._central_3d.set_data_3d(central[:, 0], central[:, 1],
                                     central[:, 2])
        self._pixel0_3d.set_data_3d(*_point_3d(view.detector_pixel0))
        self._iso_text_3d.set_position_3d(tuple(view.detector_origin))
        self._source_text_3d.set_position_3d(tuple(view.source_draw))
        self._detector_text_3d.set_position_3d(
            tuple(_far_corner(view, (0, 1, 2))))
        if self._show_trajectory:
            path = self._source_trajectory()
            self._path_3d.set_data_3d(path[:, 0], path[:, 1], path[:, 2])

    def _update_projected_panel(self, artists, view, columns, edge_pair,
                                offset, offset_name):
        """Update one projected panel's moving artists.

        Args:
            artists (dict): the panel's artists, from
                :meth:`_create_projected_moving_artists`.
            view (ViewScene): the current view's primitives.
            columns (tuple): which object coordinates go on the horizontal and
                the vertical axis, as indices into (x, y, z);
                :data:`TOP_PANEL_COLUMNS` or :data:`SIDE_PANEL_COLUMNS`.
            edge_pair (tuple): which two of the four corner rays bound this
                panel's plane, as indices into ``view.corner_rays``.
            offset (float): the detector offset this panel shows, which is
                ``det_channel_offset`` for the top view and the row offset for
                the side view.
            offset_name (str): the name of that offset, for its label.
        """
        first, second = columns

        def flat(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            return points[:, first], points[:, second]

        rays = view.corner_rays
        # The two corner rays that bound this plane are drawn darker, so that
        # the fan in the top view and the cone in the side view read as
        # outlines.  The top view's pair is the two corners at the extremes of
        # the channel direction and the side view's is the two at the extremes
        # of the row direction.
        other_pair = tuple(index for index in range(4)
                           if index not in edge_pair)
        edge = _joined([rays[index] for index in edge_pair])
        other = _joined([rays[index] for index in other_pair])
        artists['edge_rays'].set_data(*flat(edge))
        artists['rays'].set_data(*flat(other))

        artists['detector'].set_data(*flat(view.detector_outline))
        artists['central'].set_data(*flat(np.stack([view.source_draw,
                                                    view.detector_origin])))
        artists['source'].set_data(*flat(view.source_draw))
        artists['pixel0'].set_data(*flat(view.detector_pixel0))

        # Each label goes beside the thing it names.  The detector's goes at
        # the end of the detector farthest from the pixel-0 marker, which
        # keeps it off the middle of the panel, where the rays and the offset
        # segment are, and away from the pixel-0 label.  Every label takes the
        # panel's own side rule, which reads it into the panel and so keeps it
        # inside the panel at either end of the detector.
        anchor = _far_corner(view, columns)
        pixel0 = view.detector_pixel0
        _place_beside(artists['source_label'],
                      (view.source_draw[first], view.source_draw[second]))
        _place_beside(artists['detector_label'],
                      (anchor[first], anchor[second]))

        # The labels at the detector iso stack on the side of the detector
        # away from its far corner, where the detector's own label is.  A
        # detector short against the panel puts the iso and that corner close
        # together, and stacking the other way ran the labels into each other.
        iso_point = (view.detector_origin[first], view.detector_origin[second])
        away = -_screen_step(anchor[second] - view.detector_origin[second])
        iso_label = artists.get('iso_label')
        gap = ISO_LABEL_GAP_POINTS
        if iso_label is not None:
            _place_beside(iso_label, iso_point, vertical=away * gap)
            gap += LABEL_LINE_POINTS
        offset_label = artists['offset_label']
        offset_label.set_visible(float(offset) != 0.0)
        if float(offset) != 0.0:
            offset_label.set_text(f'{offset_name} {_three_figures(offset)}')
            _place_beside(offset_label, iso_point, vertical=away * gap)

        # The pixel-0 marker sits at the end of the detector opposite the far
        # corner, so the labels at the detector iso stack toward it.  Its own
        # label therefore reads outward from the marker, on the side away from
        # the iso.  A fixed side put the label above the marker whichever way
        # the detector was turned, and at the views where the iso was above the
        # marker the label landed on the iso's own label, a collision of the
        # "detector iso" label against the "pixel (0,0)" label.
        pixel0_label = artists.get('pixel0_label')
        if pixel0_label is not None:
            outward = _screen_step(pixel0[second]
                                   - view.detector_origin[second])
            _place_beside(pixel0_label, (pixel0[first], pixel0[second]),
                          vertical=outward * PIXEL0_LABEL_GAP_POINTS)

        # The offset segment runs from the detector iso to the center of the
        # detector grid.  Projected onto this panel it shows exactly the offset
        # this panel is about, because the other offset is perpendicular to the
        # plane.
        if float(offset) == 0.0:
            artists['offset'].set_data([], [])
        else:
            artists['offset'].set_data(
                *flat(np.stack([view.detector_origin, view.detector_center])))

    def _place_slice_offset_label(self, view):
        """Put the side view's recon_slice_offset label beside its segment.

        The label names the segment that runs from z = 0 to the volume's z
        center at the volume's high y edge.  It sits at the end of that
        segment, on the side away from the source, so that the source's marker
        cannot be drawn over it.  Which side that is changes with the view in a
        multiaxis scan, whose source rises and falls with the elevation, and in
        a helical scan, whose source rises through the scan.

        Args:
            view (ViewScene): the current view's primitives.
        """
        label = self._slice_offset_label
        if label is None:
            return
        _, vertical_column = SIDE_PANEL_COLUMNS
        end_z = self._slice_offset_point[1]
        source_z = float(view.source_draw[vertical_column])
        # _screen_step reports which way a step up in z points on the screen,
        # so this is +1 when the source is drawn above the segment's end.  The
        # label then goes below it, and the other way around.
        above = _screen_step(source_z - end_z)
        # The gap is a whole line and not LABEL_GAP_POINTS alone, for two
        # reasons.  The source's marker is SOURCE_MARKER_SIZE points wide, so
        # it reaches half of that past the source and would still touch a label
        # one gap away when the source sits at the segment's end.  And the
        # detector's row-offset label reads back toward the volume from the
        # detector iso, which is at nearly the same height in a side view that
        # is many times wider than it is tall; a line of clearance keeps the
        # two apart.
        gap = LABEL_GAP_POINTS + LABEL_LINE_POINTS
        # The horizontal side is the one that reads outward from the volume
        # box, which is a step toward larger y.
        _place_beside(label, self._slice_offset_point,
                      side=_screen_step(1.0), vertical=-above * gap)

    def _update_arc_and_trajectory(self, view):
        """Update the rotation arc, its arrowhead, and the source path.

        The arc, its direction, and its arrowhead direction all come from
        ``ViewScene.rotation_direction_arc``, whose last two points give the
        direction the head points.  The 3D arrowhead is drawn by matplotlib's
        own ``quiver``, which builds it from a start and a direction, so it is
        made again for each view rather than updated.
        """
        arc = view.rotation_direction_arc
        if self._arrow_3d is not None:
            self._arrow_3d.remove()
            self._arrow_3d = None
        for artist in (self._arc_3d, self._arc_text_3d, self._arc_top,
                       self._arrow_top, self._arc_text_top):
            artist.set_visible(arc is not None)
        if arc is not None:
            self._arc_3d.set_data_3d(arc[:, 0], arc[:, 1], arc[:, 2])
            self._arc_text_3d.set_position_3d(tuple(arc[-1]))
            start, end = arc[-2], arc[-1]
            direction = end - start
            self._arrow_3d = self.ax_3d.quiver(
                start[0], start[1], start[2],
                direction[0], direction[1], direction[2],
                color=COLORS['axis'], arrow_length_ratio=3.0, linewidth=1.4)
            self._arrow_3d.set_animated(self._animate_moving())
            # quiver widens the axes limits to hold its arrow, so the panel's
            # own limits go back on afterwards.
            self._apply_3d_limits()

            top_first, top_second = TOP_PANEL_COLUMNS
            self._arc_top.set_data(arc[:, top_first], arc[:, top_second])
            self._arrow_top.set_positions(
                (start[top_first], start[top_second]),
                (end[top_first], end[top_second]))
            # The arc's label reads into the panel, because the arc's end
            # sits at the source's radius, which is near a panel edge.
            # It hangs on the side of that end away from the source, which is
            # the point the arc runs from.  The source carries its own label a
            # few points away on its own side, and with both labels between the
            # two points they ran into each other.
            ahead = _screen_step(end[top_second]
                                 - view.source_draw[top_second])
            _place_beside(self._arc_text_top,
                          (end[top_first], end[top_second]),
                          vertical=ahead * ARC_LABEL_GAP_POINTS)

        if self._show_trajectory:
            path = self._source_trajectory()
            self._path_top.set_data(path[:, TOP_PANEL_COLUMNS[0]],
                                    path[:, TOP_PANEL_COLUMNS[1]])
            self._path_side.set_data(path[:, SIDE_PANEL_COLUMNS[0]],
                                     path[:, SIDE_PANEL_COLUMNS[1]])
            travel = self._quantities['helical_travel_alu']
            self._path_note_side.set_text(
                f'source z range {_three_figures(travel)} ALU')
        for artist in (self._path_top, self._path_side, self._path_3d,
                       self._path_note_side):
            artist.set_visible(self._show_trajectory)

    def _update_detector_panel(self, view):
        """Update the projected outlines on the detector face.

        The volume box's twelve edges are drawn as two polylines rather than as
        twelve artists: one for the parts on the detector and one for the parts
        past its edge, joined by rows of NaN.  An edge that crosses the
        boundary is sampled and split, so the red part starts exactly where the
        outline leaves the grid.  The region of reconstruction's two rims are
        drawn the same way, as two more polylines, when the scene reports them.

        A phantom's outline is drawn here too, as one more polyline.  It is not
        split at the detector's edge: the phantom lies inside the volume, whose
        own outline is already split and already carries the overshoot color.

        A sinogram painted on this panel is updated here as well.  A view
        change replaces the image's data and nothing else: its extent, its
        colormap, and its color scale are the same for every view.
        """
        if self._sinogram_image is not None:
            self._sinogram_image.set_data(self._sinogram[self._view_index])
        self._update_recon_detector_outline()

        outline = np.asarray(view.volume_outline_on_detector, dtype=np.float64)
        edges = np.stack([outline[[first, second]]
                          for first, second in VOLUME_BOX_EDGES])
        self.detector_volume_edges = edges
        inside, outside = self._split_edges(edges)
        # The panel's axes are (channel, row) and the scene reports
        # (row, channel).
        self._edges_inside.set_data(inside[:, 1], inside[:, 0])
        self._edges_outside.set_data(outside[:, 1], outside[:, 0])

        if self._ror_inside is None:
            return
        # Each rim is a closed polyline, so its segments are its consecutive
        # pairs of points.  The two rims' segments are split as one set,
        # because the two lines drawn hold both rims together.
        rims = np.asarray(view.ror_outline_on_detector, dtype=np.float64)
        segments = np.concatenate([np.stack([rim[:-1], rim[1:]], axis=1)
                                   for rim in rims])
        inside, outside = self._split_edges(segments)
        self._ror_inside.set_data(inside[:, 1], inside[:, 0])
        self._ror_outside.set_data(outside[:, 1], outside[:, 0])

    def _update_recon_detector_outline(self):
        """Put the phantom's outline on the detector face for this view.

        Every section of the 3D outline is projected in one call, because the
        scene's ``project_points`` is the one route to the detector for every
        drawn thing.  Only the finite points go through it, and each projected
        pair goes back at the row it came from, so the rows of NaN that break
        one segment from the next survive the trip and no NaN reaches the
        model.  The projected points are then cut back into the sections and
        joined by rows of NaN.  The panel's axes are (channel, row) and the
        scene reports (row, channel).
        """
        if self._recon_detector_line is None:
            return
        parts = self._recon_outline_parts
        points = np.concatenate(parts)
        finite = np.isfinite(points).all(axis=1)
        pairs = np.full((points.shape[0], 2), np.nan)
        if finite.any():
            rows, channels = self.scene.project_points(points[finite],
                                                       self._view_index)
            pairs[finite, 0] = channels
            pairs[finite, 1] = rows
        breaks = np.cumsum([part.shape[0] for part in parts])[:-1]
        drawn = _joined(np.split(pairs, breaks), 2)
        self._recon_detector_line.set_data(drawn[:, 0], drawn[:, 1])

    def _split_edges(self, edges):
        """The projected segments, split into the parts on and off the grid.

        Args:
            edges (ndarray): the segments, (M, 2, 2), each a start and an end
                as (row, channel).

        Returns:
            (ndarray, ndarray): the parts inside the grid and the parts outside
            it, each a NaN-separated polyline of (row, channel) pairs.
        """
        inside_parts, outside_parts = [], []
        for edge in edges:
            if self._inside_detector(edge).all():
                inside_parts.append(edge)
                continue
            fraction = np.linspace(0.0, 1.0, EDGE_CLIP_SAMPLES)[:, None]
            points = edge[0][None, :] + fraction * (edge[1] - edge[0])[None, :]
            inside = self._inside_detector(points)
            for span, is_inside in _runs_of_equal_flags(inside):
                piece = points[span]
                if piece.shape[0] < 2:
                    continue
                (inside_parts if is_inside else outside_parts).append(piece)
        return (_joined(inside_parts, 2), _joined(outside_parts, 2))

    def _inside_detector(self, row_channel):
        """Whether each (row, channel) pair lies on the detector grid."""
        pairs = np.asarray(row_channel, dtype=np.float64).reshape(-1, 2)
        return ((pairs[:, 0] >= -0.5)
                & (pairs[:, 0] <= self.scene.num_det_rows - 0.5)
                & (pairs[:, 1] >= -0.5)
                & (pairs[:, 1] <= self.scene.num_det_channels - 0.5))

    def _update_compare_artists(self, view):
        """Update the comparison overlay for its view."""
        compare = self._compare
        outline = view.detector_outline
        compare['detector_3d'].set_data_3d(outline[:, 0], outline[:, 1],
                                           outline[:, 2])
        central = np.stack([view.source_draw, view.detector_origin])
        compare['central_3d'].set_data_3d(central[:, 0], central[:, 1],
                                          central[:, 2])
        source = view.source_draw
        pixel0 = view.detector_pixel0
        compare['source_3d'].set_data_3d(*_point_3d(source))
        compare['pixel0_3d'].set_data_3d(*_point_3d(pixel0))
        for name, (first, second) in (('top', TOP_PANEL_COLUMNS),
                                      ('side', SIDE_PANEL_COLUMNS)):
            compare[f'detector_{name}'].set_data(outline[:, first],
                                                 outline[:, second])
            compare[f'central_{name}'].set_data(central[:, first],
                                                central[:, second])
            compare[f'source_{name}'].set_data([source[first]],
                                               [source[second]])
            compare[f'pixel0_{name}'].set_data([pixel0[first]],
                                               [pixel0[second]])

        # The comparison's label sits beside the end of its detector that is
        # farthest from its pixel-0 marker, as the primary's does.
        anchor = _far_corner(view, TOP_PANEL_COLUMNS)
        _place_beside(compare['label_top'],
                      (anchor[TOP_PANEL_COLUMNS[0]],
                       anchor[TOP_PANEL_COLUMNS[1]]))

        indices = np.asarray(view.volume_outline_on_detector,
                             dtype=np.float64)
        edges = _joined([indices[[first, second]]
                         for first, second in VOLUME_BOX_EDGES], 2)
        compare['edges_detector'].set_data(edges[:, 1], edges[:, 0])
        ((landing_row, landing_channel),
         (center_row, center_channel)) = \
            self.compare_scene.detector_iso_and_center_indices()
        compare['landing_detector'].set_data([float(landing_channel)],
                                             [float(landing_row)])
        compare['center_detector'].set_data([float(center_channel)],
                                            [float(center_row)])
        if self._show_trajectory:
            path = self._compare_source_trajectory()
            for axes, line in self._compare_trajectory_lines:
                if axes is self.ax_3d:
                    line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                elif axes is self.ax_top:
                    line.set_data(path[:, TOP_PANEL_COLUMNS[0]],
                                  path[:, TOP_PANEL_COLUMNS[1]])
                else:
                    line.set_data(path[:, SIDE_PANEL_COLUMNS[0]],
                                  path[:, SIDE_PANEL_COLUMNS[1]])

    # ------------------------------------------------------------------
    # Panel limits
    # ------------------------------------------------------------------

    def _object_point_groups(self, view):
        """Every object-frame point one view's drawing occupies.

        The panel limits are worked out from these, so the list must hold
        everything a panel draws in object coordinates.  The detector-face
        panel is in index coordinates and is handled separately.
        """
        groups = [view.volume_corners, view.detector_outline,
                  view.source_draw.reshape(1, 3),
                  view.detector_origin.reshape(1, 3),
                  view.detector_center.reshape(1, 3),
                  view.corner_rays.reshape(-1, 3)]
        if view.rotation_axis is not None:
            groups.append(view.rotation_axis)
        if view.translation_path is not None:
            groups.append(view.translation_path)
        if view.rotation_direction_arc is not None:
            groups.append(view.rotation_direction_arc)
        return groups

    def _limit_sample_indices(self):
        """The views the panel limits are computed from.

        A scan of 1800 views is not walked through: the limits come from
        :data:`LIMIT_SAMPLE_VIEWS` views spread over the scan, plus the view
        drawn, plus any view that has been found to fall outside the limits
        before.  A rotating geometry's views are rotations of one another about
        z, so a sample bounds them all in radius; the check in
        :meth:`_limits_hold` covers what a sample can miss.
        """
        count = self.scene.num_views
        if count <= LIMIT_SAMPLE_VIEWS:
            sample = set(range(count))
        else:
            sample = set(int(index) for index in np.round(
                np.linspace(0, count - 1, LIMIT_SAMPLE_VIEWS)))
        sample |= self._limit_views
        sample.add(self._view_index)
        self._limit_views = sample
        return sorted(sample)

    def _compute_limits(self, view, compare_view):
        """Work out and store the limits of the four drawing panels."""
        groups = []
        outlines = []
        for index in self._limit_sample_indices():
            sampled = self.scene.view(index)
            groups.extend(self._object_point_groups(sampled))
            outlines.append(sampled.volume_outline_on_detector)
            if self.compare_scene is not None:
                other = self.compare_scene.view(self._compare_index(index))
                groups.extend(self._object_point_groups(other))
                outlines.append(other.volume_outline_on_detector)
        groups.extend(self._object_point_groups(view))
        outlines.append(view.volume_outline_on_detector)
        if compare_view is not None:
            groups.extend(self._object_point_groups(compare_view))
            outlines.append(compare_view.volume_outline_on_detector)
        if self._show_trajectory:
            groups.append(self._source_trajectory())
            if self.compare_scene is not None:
                groups.append(self._compare_source_trajectory())
        # The angle-0 reference is measured whether or not it is drawn, so
        # that turning it on cannot put it outside limits already set.  The
        # limits are computed once, and the reference never moves.
        reference = self._reference_view()
        groups.extend([reference.source_draw.reshape(1, 3),
                       reference.detector_outline,
                       reference.detector_origin.reshape(1, 3)])

        points = np.concatenate([np.asarray(group, dtype=np.float64)
                                 .reshape(-1, 3) for group in groups])
        self._limits['scan'] = _cube_bounds(points)
        self._limits['volume'] = self._volume_cube()
        self._limits['top'] = _bounds(points[:, list(TOP_PANEL_COLUMNS)])
        self._limits['side'] = _bounds(points[:, list(SIDE_PANEL_COLUMNS)])

        index_points = np.concatenate(
            [np.asarray(outline, dtype=np.float64).reshape(-1, 2)
             for outline in outlines]
            + [np.array([[-0.5, -0.5],
                         [self.scene.num_det_rows - 0.5,
                          self.scene.num_det_channels - 0.5]])])
        margin = PANEL_MARGIN * max(self.scene.num_det_rows,
                                    self.scene.num_det_channels)
        row_low, row_high = (float(np.min(index_points[:, 0])) - margin,
                             float(np.max(index_points[:, 0])) + margin)
        channel_low, channel_high = (
            float(np.min(index_points[:, 1])) - margin,
            float(np.max(index_points[:, 1])) + margin)
        self._limits['detector'] = ((row_low, row_high),
                                    (channel_low, channel_high))

    def _volume_cube(self):
        """The cube the ``'volume'`` zoom uses, as three (low, high) pairs.

        The cube is centered on the volume box and is
        :data:`ZOOM_VOLUME_WIDTH_FACTOR` times the volume's largest extent
        wide, so the volume fills the middle of the panel and the rotation axis
        and the nearest rays are still in the picture.
        """
        corners = self.scene.volume_corners()
        if self.compare_scene is not None:
            corners = np.concatenate([corners,
                                      self.compare_scene.volume_corners()])
        low = corners.min(axis=0)
        high = corners.max(axis=0)
        center = 0.5 * (low + high)
        extent = ZOOM_VOLUME_WIDTH_FACTOR * float(np.max(high - low))
        if extent <= 0.0:
            extent = 1.0
        return tuple((float(center[index] - 0.5 * extent),
                      float(center[index] + 0.5 * extent))
                     for index in range(3))

    def _limits_hold(self, view, compare_view):
        """Whether the current view's content fits the limits already set."""
        if not self._limits:
            return False
        groups = self._object_point_groups(view)
        outlines = [view.volume_outline_on_detector]
        if compare_view is not None:
            groups.extend(self._object_point_groups(compare_view))
            outlines.append(compare_view.volume_outline_on_detector)
        points = np.concatenate([np.asarray(group, dtype=np.float64)
                                 .reshape(-1, 3) for group in groups])
        if not _within(points[:, list(TOP_PANEL_COLUMNS)],
                       self._limits['top']):
            return False
        if not _within(points[:, list(SIDE_PANEL_COLUMNS)],
                       self._limits['side']):
            return False
        if self._zoom == 'scan' and not _within(points, self._limits['scan']):
            return False
        index_points = np.concatenate(
            [np.asarray(outline, dtype=np.float64).reshape(-1, 2)
             for outline in outlines])
        return _within(index_points, self._limits['detector'])

    def _apply_limits(self):
        """Put the stored limits on the four drawing panels.

        Autoscaling is turned off afterwards, so that no artist's data can move
        a panel's limits and invalidate the background behind it.
        """
        self._apply_3d_limits()
        # Both projected panels put y on their horizontal axis and turn it
        # around, so that y increases to the left and the source of a view at
        # angle 0 is on the left.  Their vertical axes are x for the top view
        # and z for the side view, and both increase downward.  These two
        # lines are the top and side panels' one reading of Z_UP_SIGN.
        for axes, key in ((self.ax_top, 'top'), (self.ax_side, 'side')):
            horizontal, vertical = self._limits[key]
            axes.set_xlim(*_screen_pair(*horizontal))
            axes.set_ylim(*_screen_pair(*vertical))
            axes.set_autoscalex_on(False)
            axes.set_autoscaley_on(False)
        # The detector face keeps the channel index increasing to the right and
        # turns the row axis around, which puts row 0 at the top.
        rows, channels = self._limits['detector']
        self.ax_detector.set_xlim(*channels)
        self.ax_detector.set_ylim(*_screen_pair(*rows))
        self.ax_detector.set_autoscalex_on(False)
        self.ax_detector.set_autoscaley_on(False)

    def _apply_3d_limits(self):
        """Put the current zoom's cube on the 3D panel.

        The limits are the cube in increasing order on all three axes.  What
        puts -z at the top of this panel is the camera's roll and not a
        reversed pair of limits; see :func:`_camera_roll_deg`.
        """
        cube = self._limits['volume' if self._zoom == 'volume' else 'scan']
        axes = self.ax_3d
        axes.set_xlim(*cube[0])
        axes.set_ylim(*cube[1])
        axes.set_zlim(*cube[2])
        axes.set_box_aspect((1.0, 1.0, 1.0))
        axes.set_autoscalex_on(False)
        axes.set_autoscaley_on(False)
        axes.set_autoscalez_on(False)

    # ------------------------------------------------------------------
    # Painting: full repaints and partial redraws
    # ------------------------------------------------------------------

    def _animate_moving(self):
        """Whether the moving artists are marked animated.

        A full draw skips an animated artist, and only the partial-redraw path
        paints one.  So the moving artists may be animated only where that path
        runs: on a backend in ``BLIT_BACKENDS`` whose canvas reports blit
        support, with blitting enabled.  Everywhere else they stay ordinary
        artists that a full draw paints.  Marking them animated on such a
        backend would leave the source, the detector, and everything the slider
        moves invisible, which is what happened on the macosx backend before
        this rule existed.
        """
        canvas = self.figure.canvas
        return (self.enable_blit
                and matplotlib.get_backend().lower() in BLIT_BACKENDS
                and bool(getattr(canvas, 'supports_blit', False)))

    def _blit_usable(self):
        """Whether the partial-redraw fast path can be used right now.

        The rule follows ``mbirtorch/viewer.py``: only on a backend where the
        fast path is verified, and only when the canvas reports blit support.
        Everywhere else a view change repaints the whole figure, which is
        correct and slower.  A save suspends the path for its duration.
        """
        return self._animate_moving() and not self._suspend_blit

    def _full_redraw(self):
        """Repaint the whole figure and cache the new background.

        The text blocks are settled in both paths.  Their font size and their
        places can only be measured with a renderer, so they are settled after
        a draw and the figure is drawn again.  A figure built with ``blit``
        off takes the same passes.  Without them its three text blocks keep
        the places they were created with and print on top of one another.
        Only the background differs between the paths, because a figure that
        cannot blit has no background to cache.
        """
        self._background = None
        canvas = self.figure.canvas
        blit_usable = self._blit_usable()
        draw = canvas.draw if blit_usable else canvas.draw_idle
        draw()
        for _ in range(3):
            # Two passes settle the blocks; the third is a guard.
            if not self._place_text_blocks():
                break
            if blit_usable:
                self._background = None
            draw()
        if blit_usable and self._background is None:
            # The backend did not report the draw, so the background is taken
            # here instead of in the draw handler.
            self._capture_background()

    def _fast_redraw(self):
        """Restore the background, redraw what moved, and blit."""
        canvas = self.figure.canvas
        if not self._blit_usable():
            canvas.draw_idle()
            return
        if self._background is None:
            self._full_redraw()
            return
        canvas.restore_region(self._background)
        self._redraw_slider_row()
        self._draw_moving_and_blit()

    def _on_draw_event(self, event):
        """Take a new background whenever the whole figure is repainted.

        A full repaint happens when the user drags the 3D camera, resizes the
        window, or when this class asks for one.  The moving artists are
        animated and so are absent from that repaint, which makes it exactly
        the background a partial redraw needs.
        """
        if self._suspend_blit or not self._blit_usable():
            return
        if event is not None and event.canvas is not self.figure.canvas:
            return
        self._capture_background()

    def _capture_background(self):
        """Store the current canvas as the background and draw what moves."""
        canvas = self.figure.canvas
        try:
            self._background = canvas.copy_from_bbox(self.figure.bbox)
        except Exception:
            self._background = None
            return
        self._draw_moving_and_blit()

    def _draw_moving_and_blit(self):
        """Draw every moving artist onto the canvas and blit the figure."""
        canvas = self.figure.canvas
        for axes, artist in self._moving + self._compare_moving:
            if artist.get_visible():
                axes.draw_artist(artist)
        if self._arrow_3d is not None and self._arrow_3d.get_visible():
            self.ax_3d.draw_artist(self._arrow_3d)
        canvas.blit(self.figure.bbox)
        try:
            canvas.flush_events()
        except NotImplementedError:
            pass

    def _redraw_slider_row(self):
        """Repaint the slider, which the restored background holds stale.

        The slider's bar and value text are not animated, so the background
        carries them as they were when it was taken.  An opaque rectangle is
        painted over the row and the slider axes is drawn again on top, which
        is what ``mbirtorch/viewer.py`` does for its own slider rows.
        """
        if self.view_slider is None or not self._slider_axes.get_visible():
            return
        canvas = self.figure.canvas
        renderer = canvas.get_renderer()
        try:
            region = self._slider_axes.get_tightbbox(renderer).padded(6)
        except Exception:
            region = self._slider_axes.bbox.padded(6)
        # Union with the last region so that a value text that grew shorter
        # leaves no part of the old one behind.
        if self._slider_region is not None:
            region = Bbox.union([region, self._slider_region])
        self._slider_region = region
        self._clear_rect.set_bounds(region.x0, region.y0, region.width,
                                    region.height)
        self._clear_rect.draw(renderer)
        self._slider_axes.draw(renderer)


def _finish_2d_panel(axes):
    """Give a 2D panel equal aspect, small tick labels, and a light grid.

    The aspect is kept equal by reshaping the axes box rather than by padding
    the data, so a panel whose content is wide and short draws as a wide short
    strip.  Padding the data instead would squeeze the content into a band
    across the middle of the panel.
    """
    axes.set_aspect('equal', adjustable='box')
    axes.tick_params(labelsize=TICK_FONT_SIZE)
    axes.grid(True, linewidth=0.3, alpha=0.4)


# Figures opened with block=False, kept alive here in case the caller drops
# the return value.  The next blocking call closes them, as the slice viewer's
# registry does (mbirtorch/viewer.py, _NONBLOCKING_VIEWERS).
_NONBLOCKING_FIGURES = []


def geometry_viewer(model_or_scene, view_index=0, show_trajectory=False,
                    compare=None, show_reference=True, zoom=DEFAULT_ZOOM,
                    title=None, figsize=(15.0, 9.0),
                    elevation_deg=DEFAULT_ELEVATION_DEG,
                    azimuth_deg=DEFAULT_AZIMUTH_DEG, sinogram=None,
                    vmin=None, vmax=None, recon=None, recon_threshold=None,
                    show_sinogram=True, show_recon=True, show_compare=True,
                    block=True):
    """Launch the interactive geometry viewer on a model.

    This function builds a
    :class:`~mbirtorch.geometry_figure.GeometryFigure`, shows it, and returns
    it.  It is used the way ``mbirtorch.slice_viewer`` is used: one call opens
    the window, and ``block`` decides whether the call waits for the window to
    close.

    The window shows five panels for one view of the scan: a 3D view, a top
    view of the xy plane, a side view of the yz plane, the detector face in row
    and channel index, and the derived numbers.  A slider steps through the
    views, and six toggles sit beside it.  Three of them turn the source's
    path, the 3D zoom to the volume, and the angle-0 reference on and off, and
    three turn the sinogram, the phantom, and the comparison on and off.  Every
    panel draws negative z at the top, and the beam runs from the source on the
    left to the detector on the right.  Two arrays can be drawn beside the
    geometry: a sinogram on the detector face and a reconstruction's silhouette
    in the volume box.

    Args:
        model_or_scene: a ``TomographyModel`` or a ``GeometryScene``.
        view_index (int, optional): the view to draw first.  Defaults to 0.
        show_trajectory (bool, optional): whether to draw the source's path
            over all views.  Defaults to False.
        compare (optional): a second geometry drawn dashed over the first: a
            ``GeometryScene``, a model, or a dictionary of parameter overrides
            such as ``dict(det_channel_offset=12.5)``.  Defaults to None.  A
            comparison opens a second window, which tables every difference
            between the two geometries.
        show_reference (bool, optional): whether to draw the source and the
            detector at their angle-0 position.  Defaults to True.
        zoom (str, optional): ``'scan'`` (default) fits the source, the
            detector, and the volume in the 3D panel; ``'volume'`` fits the
            volume.
        title (str, optional): the figure title.  None (default) names the
            geometry and the shapes.
        figsize (tuple, optional): the figure size in inches.
        elevation_deg, azimuth_deg (float, optional): the 3D camera.
        sinogram (array_like, optional): a sinogram of shape
            ``(num_views, num_det_rows, num_det_channels)``, painted on the
            detector face for the view drawn.  Defaults to None, which paints
            none.  A detector larger than ``SINOGRAM_DISPLAY_PIXELS`` pixels
            across is subsampled for display, on the device that holds the
            array.
        vmin, vmax (float, optional): the gray scale's ends.  Default None
            takes the smallest and the largest value of the sinogram as
            displayed.
        recon (array_like, optional): a reconstruction or a phantom of shape
            ``recon_shape``, drawn as a silhouette in the volume box of the top
            view and the side view.  Defaults to None, which draws none.  The
            array is read a chunk at a time where it lives, so a tensor on a
            GPU is thresholded there, and the 3D panel draws up to
            ``PHANTOM_SECTION_COUNT`` outlines of the support in planes across
            its thinnest direction, with the legend naming the count.
        recon_threshold (float, optional): the absolute value above which a
            voxel belongs to that silhouette.  Defaults to None, which uses
            ``DEFAULT_RECON_THRESHOLD_FRACTION`` of the largest absolute
            value in ``recon``.
        show_sinogram (bool, optional): whether the sinogram starts drawn.
            Defaults to True.  Its toggle turns it on and off.
        show_recon (bool, optional): whether the phantom starts drawn.
            Defaults to True.
        show_compare (bool, optional): whether the comparison starts drawn.
            Defaults to True.
        block (bool, optional): If True (default), block until the window is
            closed.  If False, leave the window open and return immediately;
            the window becomes fully interactive when the next blocking call
            runs, and that blocking call closes every earlier nonblocking
            window when it returns.

    Returns:
        GeometryFigure: the figure.  Nonblocking callers may keep it to change
        the view or save an image; a module-level registry also keeps it alive
        if the return value is dropped.
    """
    figure = GeometryFigure(model_or_scene, view_index=view_index,
                            show_trajectory=show_trajectory, figsize=figsize,
                            title=title, elevation_deg=elevation_deg,
                            azimuth_deg=azimuth_deg, compare=compare,
                            zoom=zoom, show_reference=show_reference,
                            sinogram=sinogram, vmin=vmin, vmax=vmax,
                            recon=recon, recon_threshold=recon_threshold,
                            show_sinogram=show_sinogram,
                            show_recon=show_recon,
                            show_compare=show_compare)
    figure.show(block=block)
    if not block:
        _NONBLOCKING_FIGURES.append(figure)
        return figure
    # The blocking show returned, so every open window has been closed.  Close
    # the earlier nonblocking figures too, so they do not accumulate.  Each of
    # them may carry a comparison window, which is a figure of its own.
    for nonblocking in _NONBLOCKING_FIGURES:
        plt.close(nonblocking.figure)
        if nonblocking.compare_figure is not None:
            plt.close(nonblocking.compare_figure)
    _NONBLOCKING_FIGURES.clear()
    # Under TkAgg, collect now on the main thread: matplotlib's TkAgg backend
    # leaves orphaned tkinter objects that a later background-thread GC would
    # finalize with no Tk mainloop running ("main thread is not in main loop").
    if matplotlib.get_backend() == 'TkAgg':
        import gc
        gc.collect()
    return figure

