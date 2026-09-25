"""Parallel-beam geometry: the source is at infinity, so the rays are parallel."""
import geom_draw as g

fig, ax = g.new_axes()
cx, cy = 3.6, 0.0
corners = g.panel(ax, 7.0, cy)
g.shadow(ax, corners)
for y in (corners[0, 1], corners[1, 1], cy - 0.6, cy + 0.6):
    g.ray(ax, (-1.2, y), (7.0, y))
g.cylinder(ax, cx, cy)
g.rotation_axis(ax, cx, cy)
g.source(ax, -0.9, 0.0, label='Source at\ninfinity')
ax.set_xlim(-1.4, 8.3)
ax.set_ylim(-2.4, 2.6)
