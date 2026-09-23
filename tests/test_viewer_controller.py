"""Crash guard for the slice viewer, run headlessly on Agg.

The viewer is built over one volume and over three, rendered to a PNG, and
the image area is sampled to confirm that pixels were actually drawn.
"""

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from mbirtorch.viewers.slice_figure import SliceViewer
# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def _process(fig, name, x, y, button=None):
    event = MouseEvent(name, fig.canvas, x, y, button=button)
    fig.canvas.callbacks.process(name, event)


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


@pytest.fixture
def make_viewer():
    created = []

    def _make(*datasets, **kwargs):
        viewer = SliceViewer(*datasets, **kwargs)
        viewer.fig.canvas.draw()
        created.append(viewer)
        return viewer

    yield _make
    for viewer in created:
        plt.close(viewer.fig)
# ---------------------------------------------------------------------------
# Toolbar interaction (zoom-to-rectangle, pan, and the menu)
# ---------------------------------------------------------------------------

class TestToolbarInteraction:
    @staticmethod
    def attach_toolbar(viewer):
        from matplotlib.backend_bases import NavigationToolbar2
        toolbar = NavigationToolbar2(viewer.fig.canvas)
        viewer.fig.canvas.toolbar = toolbar
        return toolbar

    @staticmethod
    def zoom_gesture(viewer, ax, d0, d1, buttons):
        from matplotlib.backend_bases import MouseEvent
        x0, y0 = ax.transData.transform(d0)
        x1, y1 = ax.transData.transform(d1)
        _process(viewer.fig, 'button_press_event', x0, y0, 1)
        for t in (0.3, 0.6, 1.0):
            event = MouseEvent('motion_notify_event', viewer.fig.canvas,
                               x0 + t * (x1 - x0), y0 + t * (y1 - y0),
                               buttons=buttons)
            viewer.fig.canvas.callbacks.process('motion_notify_event', event)
        _process(viewer.fig, 'button_release_event', x1, y1, 1)
# ---------------------------------------------------------------------------
# Blitting and snapshots
# ---------------------------------------------------------------------------

class TestRendering:
    @pytest.mark.parametrize('n_volumes', [1, 3])
    def test_snapshot_contains_image_pixels(self, make_viewer, tmp_path,
                                            n_volumes):
        volumes = [make_volume((32, 32, 8), s) for s in range(n_volumes)]
        viewer = make_viewer(*volumes, title='snapshot test')
        path = str(tmp_path / f'snapshot_{n_volumes}.png')
        viewer.fig.savefig(path)
        assert os.path.getsize(path) > 30_000
        # The panel region must not be blank: sample the image area.
        from matplotlib.image import imread
        pixels = imread(path)
        ax_bbox = viewer.axes[0].bbox
        height = viewer.fig.canvas.get_width_height()[1]
        row = int(height - (ax_bbox.y0 + ax_bbox.y1) / 2)
        col_range = slice(int(ax_bbox.x0), int(ax_bbox.x1))
        panel_row = pixels[row, col_range, :3]
        assert panel_row.std() > 0.05  # random noise, not a flat fill
