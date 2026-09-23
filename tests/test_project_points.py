"""Gates for TomographyModel.project_points on all four geometries.

Three things are checked against each other.  The projector itself, through the
centroid of a voxel's forward-projected footprint; project_points, which is
meant to state the projector's geometry for points; and an independent numpy
prediction written from a plain geometric statement (tests/geometry_probe.py),
which never calls any of the projector's own formulas.  The centroid
comparison has a half-pixel gate, because the footprint weights move a centroid
by about a tenth of a pixel; the prediction comparison is tight, because it is
the same map computed two independent ways.
"""

import numpy as np
import pytest

import geometry_probe as probe

# The centroid of a footprint is not exactly the projected center: the weights
# are cut off at the ends of the footprint, which moves the centroid by about a
# tenth of a pixel.  Half a pixel is the gate these tests use.
GATE_PIXELS = 0.5

# The gate for two computations of the same geometry.  Both run in float64 on
# the same view parameters, so the difference is rounding only.
EXACT_PIXELS = 1e-8

CONFIG_NAMES = [cfg['name'] for cfg in probe.CONFIGS]


def _config(name):
    """The probe configuration with this name."""
    return [cfg for cfg in probe.CONFIGS if cfg['name'] == name][0]


def _probe_points(cfg, g):
    """The probe voxels and the object-frame center of each one, (N, 3)."""
    voxels = probe.probe_voxels(cfg['recon_shape'])
    points = np.array([probe.voxel_center_xyz(i, j, k, g) for i, j, k in voxels])
    return voxels, points


def _config_with_stored_view_params(cfg, model):
    """A copy of ``cfg`` whose per-view numbers are the ones the model stores.

    The model classes keep their angles, z shifts, and translation vectors as
    float32, so the numbers written in the configuration are not quite the
    numbers the model works from.  The gap is about 1e-7 of a radian, which
    moves an index by about 1e-6 of a pixel -- enough to swamp a tight
    comparison, and nothing to do with the geometry.  Reading the view
    parameters back from the model removes it.  The prediction itself is
    unchanged: it is still the probe's own numpy geometry.
    """
    stored = np.asarray(model.get_params(model.get_params('view_params_name')),
                        dtype=np.float64)
    updated = dict(cfg)
    if cfg['kind'] == 'parallel':
        updated['angles'] = stored
    elif cfg['kind'] == 'cone':
        updated['angles'] = stored[:, 0]
        updated['z_shifts'] = stored[:, 1]
    elif cfg['kind'] == 'multiaxis':
        updated['angles'] = stored[:, 0]
        updated['elevations'] = stored[:, 1]
    else:
        updated['translation_vectors'] = stored
    return updated


@pytest.mark.parametrize('name', CONFIG_NAMES)
def test_project_points_matches_the_projector(name):
    """A projected point lands where the projector puts the voxel's mass."""
    cfg = _config(name)
    model = probe.build_model(cfg)
    g = probe.geometry_scalars(cfg)
    voxels, points = _probe_points(cfg, g)
    num_views = cfg['sinogram_shape'][0]
    row, channel = model.project_points(points, list(range(num_views)))

    compared = 0
    for index, voxel in enumerate(voxels):
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        for view, (reason, row_centroid, channel_centroid) in enumerate(measured):
            if reason is not None:
                # The footprint ran off the detector, so its centroid says
                # nothing about where the point landed.
                continue
            compared += 1
            assert abs(row[view, index] - row_centroid) < GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: row {row[view, index]} '
                f'against centroid {row_centroid}')
            assert abs(channel[view, index] - channel_centroid) < GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: channel '
                f'{channel[view, index]} against centroid {channel_centroid}')
    assert compared > 0, f'{name}: no voxel-view pair could be compared'


@pytest.mark.parametrize('name', CONFIG_NAMES)
def test_project_points_matches_the_independent_prediction(name):
    """The map agrees with a prediction written from the geometry alone."""
    cfg = _config(name)
    model = probe.build_model(cfg)
    g = probe.geometry_scalars(cfg)
    stated = _config_with_stored_view_params(cfg, model)
    predictor = probe.PREDICTORS[cfg['kind']]
    voxels, points = _probe_points(cfg, g)
    num_views = cfg['sinogram_shape'][0]
    row, channel = model.project_points(points, list(range(num_views)))

    for index, voxel in enumerate(voxels):
        for view in range(num_views):
            row_pred, channel_pred = predictor(*voxel, view, stated, g, {})
            assert abs(row[view, index] - row_pred) < EXACT_PIXELS, (
                f'{name} voxel {voxel} view {view}: row {row[view, index]} '
                f'against prediction {row_pred}')
            assert abs(channel[view, index] - channel_pred) < EXACT_PIXELS, (
                f'{name} voxel {voxel} view {view}: channel '
                f'{channel[view, index]} against prediction {channel_pred}')
