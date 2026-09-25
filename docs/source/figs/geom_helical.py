"""Helical cone-beam geometry: the source path is a helix about the rotation
axis, drawn in the frame of the object."""
import numpy as np
import geom_draw as g

fig, ax = g.new_axes()
cx, cy = 3.6, 0.0
radius, ry, turns = 2.1, 0.32, 2.0
t = np.pi + np.linspace(0.0, 2 * np.pi * turns, 400)
hx = cx + radius * np.cos(t)
hy = cy - 1.1 + 2.2 * (t - t[0]) / (t[-1] - t[0]) + ry * np.sin(t)
front = np.sin(t) < 0          # the half of each turn nearer the viewer

corners = g.panel(ax, 7.0, cy)
g.shadow(ax, corners)
sx, sy = hx[0], hy[0]          # the source at the start of the helix
for corner in corners:
    g.ray(ax, (sx, sy), corner)

# The far half of the helix, then the object, then the near half.
ax.plot(np.where(front, np.nan, hx), np.where(front, np.nan, hy), '--', color='0.6', lw=1.1)
g.cylinder(ax, cx, cy)
g.rotation_axis(ax, cx, cy)
ax.plot(np.where(front, hx, np.nan), np.where(front, hy, np.nan), '-', color='k', lw=1.3)
ax.text(cx + radius + 0.15, cy + 0.9, 'source path', fontsize=g.FONT_SIZE, va='center')
g.source(ax, sx, sy, label='Source', label_dy=-0.3)
ax.set_xlim(-1.4, 8.3)
ax.set_ylim(-2.4, 2.6)
