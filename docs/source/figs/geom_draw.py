"""Drawing helpers shared by the geometry figures of the documentation.

Every figure is a schematic in one 2D axes: the source on the left, the
object on the rotation axis in the middle, and the detector on the right.
Coordinates are in arbitrary drawing units.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

OBJECT_FILL = '#f9bcbc'
OBJECT_TOP = '#fbdada'
PANEL_FILL = '#f2e4d4'
SHADOW_FILL = '#fbdada'
RAY_COLOR = '0.45'
FONT_SIZE = 13


def new_axes(width=8.0, height=4.2):
    """Return a figure and an equal-aspect axes with no frame."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def cylinder(ax, cx, cy, rx=1.1, ry=0.18, height=1.2, label='object'):
    """Draw the cylindrical object centered at (cx, cy).  ``label`` None draws no label."""
    t = np.linspace(np.pi, 2 * np.pi, 60)
    ax.add_patch(Rectangle((cx - rx, cy - height / 2), 2 * rx, height,
                           facecolor=OBJECT_FILL, edgecolor='none'))
    ax.fill(cx + rx * np.cos(t), cy - height / 2 + ry * np.sin(t), color=OBJECT_FILL)
    ax.plot(cx + rx * np.cos(t), cy - height / 2 + ry * np.sin(t), 'k', lw=1.2)
    ax.plot([cx - rx, cx - rx], [cy - height / 2, cy + height / 2], 'k', lw=1.2)
    ax.plot([cx + rx, cx + rx], [cy - height / 2, cy + height / 2], 'k', lw=1.2)
    ax.add_patch(Ellipse((cx, cy + height / 2), 2 * rx, 2 * ry,
                         facecolor=OBJECT_TOP, edgecolor='k', lw=1.2))
    if label:
        ax.text(cx + rx + 0.15, cy - height / 2 - 0.3, label, fontsize=FONT_SIZE, va='top')


def rotation_axis(ax, cx, cy, height=1.2, label='view angles'):
    """Draw the vertical rotation axis through (cx, cy) with a curved arrow."""
    top = cy + height / 2 + 1.1
    ax.annotate('', xy=(cx, top), xytext=(cx, cy - height / 2 - 0.6),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))
    arrow = FancyArrowPatch((cx + 0.55, top - 0.45), (cx - 0.25, top - 0.35),
                            connectionstyle='arc3,rad=0.6', arrowstyle='-|>',
                            mutation_scale=12, color='k', lw=1.1)
    ax.add_patch(arrow)
    ax.text(cx + 0.35, top - 0.05, label, fontsize=FONT_SIZE)


def panel(ax, x, cy, half_height=1.6, depth=0.9, tilt=0.0):
    """Draw the detector panel and return its four corners.

    The panel's left edge is at x; the right edge is nearer the viewer, so it
    is drawn taller and displaced by ``depth``.  ``tilt`` rotates the panel
    about its center, in radians.  Corners are returned in the order
    left-bottom, left-top, right-top, right-bottom.
    """
    up = np.array([-np.sin(tilt), np.cos(tilt)])
    c = np.array([x, cy])
    d = np.array([depth, 0.0])
    corners = np.array([c - half_height * up,
                        c + half_height * up,
                        c + (half_height + 0.3) * up + d,
                        c - (half_height + 0.3) * up + d])
    ax.add_patch(Polygon(corners, closed=True, facecolor=PANEL_FILL,
                         edgecolor='k', lw=1.4))
    ax.text(corners[3, 0] - 0.3, corners[3, 1] - 0.15, 'Detector',
            fontsize=FONT_SIZE, va='top', ha='right')
    return corners


def shadow(ax, corners, fraction=0.75):
    """Draw the object's shadow on the panel as an inset of the panel."""
    center = corners.mean(axis=0)
    inset = center + fraction * (corners - center)
    ax.add_patch(Polygon(inset, closed=True, facecolor=SHADOW_FILL,
                         edgecolor='none'))


def source(ax, x, y, label='Source', label_dy=-0.35):
    """Draw the source as an eight-point star with a label below it."""
    ax.plot(x, y, marker=(8, 1, 0), markersize=16, color='k')
    ax.text(x, y + label_dy, label, fontsize=FONT_SIZE, ha='center', va='top')


def ray(ax, p, q):
    """Draw one dotted ray from p to q."""
    ax.plot([p[0], q[0]], [p[1], q[1]], ':', color=RAY_COLOR, lw=1.2)
