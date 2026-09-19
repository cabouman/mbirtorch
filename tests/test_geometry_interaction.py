"""Headless crash guard for the geometry figure of `geometry_figure.py`.

Each of the six scan geometries is walked through every control -- the view
slider, the zoom, the trajectory toggle, and a comparison overlay -- and the
figure is then saved.  The controls touch different code for different
geometries: the translation geometry has no rotation arc, the parallel and
multiaxis geometries have no source position, and a curved detector has no
filled face.  An artist that one geometry does not create must not break a
redraw.

The test runs under the Agg backend and opens no window.
"""

import os
import sys

import pytest

import matplotlib
matplotlib.use('Agg')  # the tests draw into a buffer and open no window

import geometry_probe as probe
from mbirtorch.viewers.geometry_scene import GeometryScene
from mbirtorch.viewers.geometry_figure import GeometryFigure

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

#: How many channels the comparison's detector offset is moved by.  Ten
#: channels is the size the comparison test uses.
COMPARISON_CHANNEL_SHIFT = 10


def build_figure(name='cone flat', **kwargs):
    """Build the scene and a figure for one probe configuration."""
    cfg = CONFIGS_BY_NAME[name]
    scene = GeometryScene.from_model(probe.build_model(cfg))
    return scene, GeometryFigure(scene, **kwargs)


def close(figure):
    """Close a figure's matplotlib figures so that the tests do not pile up.

    A comparison opens a second figure, which is closed here as well.
    """
    import matplotlib.pyplot as plt
    if figure.compare_figure is not None:
        plt.close(figure.compare_figure)
    plt.close(figure.figure)


# ── the comparison overlay ──────────────────────────────────────────────────

def compare_overrides(scene, channels=COMPARISON_CHANNEL_SHIFT):
    """The override dictionary that moves the detector by whole channels."""
    return dict(det_channel_offset=(scene.det_channel_offset
                                    + channels * scene.delta_det_channel))


# ── the partial redraw draws the same picture as a full repaint ─────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_every_geometry_takes_every_control(name, tmp_path):
    """Each of the six geometries survives the slider, the toggles, and a
    comparison.

    The controls touch different code for different geometries: the
    translation geometry has no rotation arc, the parallel and multiaxis
    geometries have no source position, and a curved detector has no filled
    face.  The test walks every geometry through every control and saves the
    result, so an artist that one geometry does not create cannot break a
    redraw.
    """
    scene, figure = build_figure(name, view_index=1)
    try:
        figure.set_zoom('volume')
        figure.set_view(3)
        figure.set_show_trajectory(True)
        figure.set_compare(compare_overrides(scene))
        figure.set_view(4)
        figure.set_zoom('scan')
        figure.set_show_trajectory(False)
        figure.set_compare(None)
        path = str(tmp_path / f'{name.replace(" ", "_")}.png')
        figure.save(path, dpi=80)
        assert os.path.getsize(path) > 10 * 1024
    finally:
        close(figure)


# ── the geometry_viewer entry point ─────────────────────────────────────────

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
