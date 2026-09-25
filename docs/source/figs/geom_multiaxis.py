"""Multi-axis parallel geometry: parallel rays at an elevation angle to the
rotation axis, as in laminography."""
import numpy as np
import geom_draw as g

fig, ax = g.new_axes(height=4.8)
cx, cy = 3.6, 0.0
elevation = np.deg2rad(22.0)
direction = np.array([np.cos(elevation), np.sin(elevation)])
up = np.array([-np.sin(elevation), np.cos(elevation)])
center = np.array([cx, cy])
panel_x = 7.0
corners = g.panel(ax, panel_x, cy + (panel_x - cx) * np.tan(elevation), tilt=elevation)
g.shadow(ax, corners)
for offset in (-1.6, -0.6, 0.6, 1.6):
    start = center + offset * up - 5.0 * direction
    end = center + offset * up + (panel_x - cx) / np.cos(elevation) * direction
    g.ray(ax, start, end)
g.cylinder(ax, cx, cy)
g.rotation_axis(ax, cx, cy)

# The elevation angle, between the horizontal and a ray, marked on the
# ray that passes below the object.
p = center - 1.6 * up - 1.9 * direction
ax.plot([p[0], p[0] + 1.5], [p[1], p[1]], '-', color='k', lw=0.9)
t = np.linspace(0.0, elevation, 30)
ax.plot(p[0] + 1.1 * np.cos(t), p[1] + 1.1 * np.sin(t), '-', color='k', lw=0.9)
ax.text(p[0] + 1.25, p[1] + 0.12, 'elevation angle', fontsize=g.FONT_SIZE, va='bottom')

sp = center - 0.6 * up - 4.6 * direction
g.source(ax, sp[0], sp[1], label='Source at\ninfinity')
ax.set_xlim(-1.4, 8.6)
ax.set_ylim(-2.7, 3.6)
