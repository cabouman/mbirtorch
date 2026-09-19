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


@pytest.mark.parametrize('name', CONFIG_NAMES)
def test_a_sequence_of_views_stacks_the_single_view_answers(name):
    """Asking for several views gives the single-view answers, stacked."""
    cfg = _config(name)
    model = probe.build_model(cfg)
    g = probe.geometry_scalars(cfg)
    _, points = _probe_points(cfg, g)
    num_views = cfg['sinogram_shape'][0]

    for views in (list(range(num_views)), [num_views - 1, 0]):
        row, channel = model.project_points(points, views)
        assert row.shape == (len(views), points.shape[0])
        assert channel.shape == (len(views), points.shape[0])
        for position, view in enumerate(views):
            row_one, channel_one = model.project_points(points, view)
            assert np.allclose(row[position], row_one, rtol=0.0, atol=1e-12)
            assert np.allclose(channel[position], channel_one, rtol=0.0,
                               atol=1e-12)


def test_one_point_and_the_shape_and_index_checks():
    """One point, and every input the method is supposed to refuse."""
    cfg = _config('cone flat')
    model = probe.build_model(cfg)
    num_views = cfg['sinogram_shape'][0]
    point = [1.5, -2.5, 0.75]

    row_one, channel_one = model.project_points(point, 0)
    row_list, channel_list = model.project_points([point], 0)
    assert row_one.shape == (1,) and channel_one.shape == (1,)
    assert np.array_equal(row_one, row_list)
    assert np.array_equal(channel_one, channel_list)

    for bad_points in (np.zeros(4), np.zeros((2, 2)), 3.0):
        with pytest.raises(ValueError):
            model.project_points(bad_points, 0)

    for bad_view in (-1, num_views):
        with pytest.raises(IndexError):
            model.project_points([point], bad_view)
    with pytest.raises(ValueError):
        model.project_points([point], 0.5)

    row, channel = model.project_points([point], np.array([], dtype=np.int64))
    assert row.shape == (0, 1) and channel.shape == (0, 1)


def test_parallel_rows_are_slice_indices():
    """Parallel beam sends slice k to row k, and ignores the row geometry."""
    cfg = _config('parallel')
    model = probe.build_model(cfg)
    g = probe.geometry_scalars(cfg)
    voxels, points = _probe_points(cfg, g)
    num_views = cfg['sinogram_shape'][0]

    row, channel = model.project_points(points, list(range(num_views)))
    for index, (_, _, k) in enumerate(voxels):
        assert np.allclose(row[:, index], k, rtol=0.0, atol=1e-9), (
            f'voxel slice {k} landed on rows {row[:, index]}')

    # The row of a parallel projection is a recon slice index, so the detector
    # row pitch and offset cannot enter it.
    model.set_params(no_warning=True, delta_det_row=2.7, det_row_offset=5.5)
    moved_row, moved_channel = model.project_points(points,
                                                    list(range(num_views)))
    assert np.array_equal(moved_row, row)
    assert np.array_equal(moved_channel, channel)


def test_the_curved_detector_rows_use_the_tangent_plane():
    """A curved panel spaces its rows on the tangent plane, not the cylinder."""
    cfg = _config('cone curved')
    model = probe.build_model(cfg)
    g = probe.geometry_scalars(cfg)
    stated = _config_with_stored_view_params(cfg, model)
    voxels, points = _probe_points(cfg, g)
    num_views = cfg['sinogram_shape'][0]
    row, _ = model.project_points(points, list(range(num_views)))

    largest_cylinder_gap = 0.0
    for index, voxel in enumerate(voxels):
        for view in range(num_views):
            plane_row, _ = probe.predict_cone(*voxel, view, stated, g, {})
            cylinder_row, _ = probe.predict_cone(*voxel, view, stated, g,
                                                 {'curved_row': 'cylinder'})
            assert abs(row[view, index] - plane_row) < EXACT_PIXELS
            largest_cylinder_gap = max(largest_cylinder_gap,
                                       abs(row[view, index] - cylinder_row))
    assert largest_cylinder_gap > 0.1, (
        'the two candidate row rules are too close together here to tell '
        f'them apart (largest gap {largest_cylinder_gap})')
