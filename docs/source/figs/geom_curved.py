"""Cone-beam geometry with a curved detector."""
import geom_draw as g

fig, ax = g.new_axes()
cx, cy = 3.6, 0.0
sx, sy = -0.6, 0.0
corners = g.panel(ax, 7.0, cy, curved=True, label='Curved detector')
g.shadow(ax, corners)
for corner in corners:
    g.ray(ax, (sx, sy), corner)
g.cylinder(ax, cx, cy)
g.rotation_axis(ax, cx, cy)
g.source(ax, sx, sy)
ax.set_xlim(-1.4, 8.3)
ax.set_ylim(-2.4, 2.6)
