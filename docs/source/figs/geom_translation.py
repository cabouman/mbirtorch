"""Translation geometry: cone beam views of a large thin plate that is translated between
views rather than rotated.  The beam illuminates one patch of the plate at a time."""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
import geom_draw as g

fig, ax = g.new_axes()
cx, cy = 3.6, 0.0
sx, sy = -0.6, 0.0
det_x, det_depth, det_half = 7.0, 0.9, 1.6
corners = g.panel(ax, det_x, cy, half_height=det_half, depth=det_depth)
g.shadow(ax, corners)
for corner in corners:
    g.ray(ax, (sx, sy), corner)

# The plate faces the source and is drawn like the detector panel, seen obliquely, its
# broad face widening toward the viewer.
thick, half_h, depth = 0.07, 1.9, 1.8
x_e = cx - 0.65                       # the edge; the beam crosses the face to its right
grow = 0.3 * depth / det_depth        # the same perspective as the detector panel
face = [(x_e + thick, cy - half_h), (x_e + thick + depth, cy - half_h - grow),
        (x_e + thick + depth, cy + half_h + grow), (x_e + thick, cy + half_h)]
ax.add_patch(Polygon(face, closed=True, facecolor=g.OBJECT_TOP, edgecolor='k', lw=1.2))

# The patch of the face the beam passes through: the rays to the far detector corners
# cross the plate at its near-left point, the rays to the near corners a little further
# along and a little taller.
u0, u1 = 0.65, 0.65 + det_depth * (cx - sx) / (det_x - sx)
y0 = det_half * (cx - sx) / (det_x - sx)
y1 = (det_half + 0.3) * (cx - sx) / (det_x + det_depth - sx)
patch = [(x_e + u0, cy - y0), (x_e + u1, cy - y1), (x_e + u1, cy + y1), (x_e + u0, cy + y0)]
ax.add_patch(Polygon(patch, closed=True, facecolor=g.OBJECT_FILL, edgecolor='none'))
ax.text(x_e + thick + depth + 0.1, cy - half_h - grow + 0.1, 'object', fontsize=g.FONT_SIZE, va='top')

# The object moves between views: up and down, and across the beam.
for p, q in (((x_e - 0.55, cy - 0.8), (x_e - 0.55, cy + 0.8)),
             ((x_e - 0.3, cy + half_h + 0.3), (x_e + 1.0, cy + half_h + 0.75))):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='<|-|>', mutation_scale=12,
                                 color='k', lw=1.1))
ax.text(x_e - 2.2, cy + half_h + 0.45, 'translations', fontsize=g.FONT_SIZE)
g.source(ax, sx, sy)
ax.set_xlim(-1.4, 8.3)
ax.set_ylim(-2.9, 3.0)
