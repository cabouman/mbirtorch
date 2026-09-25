"""4D reconstruction: one continuous scan is divided into overlapping angular
windows, and one volume is reconstructed per window (time frame)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import geom_draw as g

fig, ax = g.new_axes(width=8.0, height=5.0)
num_frames = 5
step, span = 60.0, 120.0              # frame start step and frame span in degrees
scale = 6.4 / 360.0                    # drawing units per degree
x0, y_axis = 0.6, 3.6

# The scan: a view-angle axis over one rotation.
ax.annotate('', xy=(x0 + 385 * scale, y_axis), xytext=(x0 - 0.2, y_axis),
            arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))
for deg in range(0, 361, 60):
    x = x0 + deg * scale
    ax.plot([x, x], [y_axis - 0.07, y_axis + 0.07], 'k', lw=1)
    ax.text(x, y_axis + 0.15, f'{deg}°', fontsize=10, ha='center')
ax.text(x0 + 385 * scale + 0.15, y_axis, 'view angle\n(time)', fontsize=11, va='center')

# The overlapping windows, one per frame, one row per frame.
for k in range(num_frames):
    start = k * step
    y = y_axis - 0.5 - 0.34 * k
    ax.add_patch(FancyBboxPatch((x0 + start * scale, y - 0.13), span * scale, 0.26,
                                boxstyle='round,pad=0.02', facecolor=g.OBJECT_FILL,
                                edgecolor='k', lw=1))
    ax.text(x0 + (start + span / 2) * scale, y, f'frame {k + 1}', fontsize=10,
            ha='center', va='center')

# One reconstructed volume per frame, under its window, with a feature that
# moves from frame to frame.
for k in range(num_frames):
    cx = x0 + (k * step + span / 2) * scale
    cy = 0.7
    g.cylinder(ax, cx, cy, rx=0.42, ry=0.08, height=0.7, label=None)
    ax.add_patch(Circle((cx - 0.24 + 0.12 * k, cy + 0.35), 0.1,
                        facecolor='0.35', edgecolor='none'))
    ax.text(cx, cy - 0.6, f'volume {k + 1}', fontsize=10, ha='center', va='top')

ax.set_xlim(-0.2, 8.6)
ax.set_ylim(-0.4, 4.1)
