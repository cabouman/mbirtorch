"""Headless tests for mbirtorch/viewers/geometry_scene.py.

The gate test is ``test_projection_matches_projector``.  For each of the six
scan geometries of ``geometry_probe.py`` it forward-projects single voxels
with the real projector, measures where each footprint lands on the detector,
and compares that with ``GeometryScene.project_points``.  A viewer that failed
this test would draw a geometry the projector does not use.

The rest of the tests check the fixed-object picture, the drawing primitives,
the parameter interface, and four edge cases: a single detector row, an
infinite source-detector distance, a curved detector, and the translation
geometry.
"""

import numpy as np
import pytest

import mbirtorch
import geometry_probe as probe
from mbirtorch.viewers.geometry_scene import (CLOCKWISE_FROM_PLUS_Z,
                                              COUNTERCLOCKWISE_FROM_PLUS_Z,
                                              CURVED_ARC_SAMPLES,
                                              GeometryScene, RIM_SAMPLES,
                                              ROTATION_ARC_SAMPLES)

# The gate: the largest allowed difference between a measured footprint
# centroid and the scene's prediction, in detector pixels.  Half a pixel is the
# tolerance the projection tests use.
GATE_PIXELS = 0.5

# Position and direction comparisons are geometric identities, so they get a
# tight tolerance rather than the gate's half pixel.
GEOMETRY_TOLERANCE = 1e-8

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

# Filled in by the gate test and printed by this file's __main__ block, so the
# run record can quote a number per geometry.
MAX_ERRORS = {}


def build_scene(cfg):
    """Build the model of one probe configuration and a scene from it."""
    model = probe.build_model(cfg)
    return model, GeometryScene.from_model(model)


def uv_to_indices(scene, u, v):
    """Fractional detector indices of detector coordinates (u, v), the inverse
    of ``scene.indices_to_uv``, kept in the tests so that a drawing primitive
    can be turned back into an index and compared with the projection."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    channel = ((u + scene.det_channel_offset) / scene.delta_det_channel
               + (scene.num_det_channels - 1) / 2.0)
    row = ((v + scene.row_offset) / scene.row_pitch
           + (scene.num_det_rows - 1) / 2.0)
    return row, channel


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_projection_matches_projector(name):
    """Every projected voxel center lands where the projector puts it.

    The comparison uses footprint centroids, which are only meaningful when the
    footprint is at least about as wide as a detector pixel.  Every probe
    configuration keeps the voxel pitch near the detector pitch for that
    reason, and the probe's own record explains the failure mode.
    """
    cfg = CONFIGS_BY_NAME[name]
    model, scene = build_scene(cfg)

    max_row_error = 0.0
    max_channel_error = 0.0
    num_pairs = 0
    for voxel in probe.probe_voxels(cfg['recon_shape']):
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        center = scene.voxel_centers([voxel])
        for view, (reason, row, channel) in enumerate(measured):
            if reason is not None:
                continue
            row_pred, channel_pred = scene.project_points(center, view)
            row_error = abs(row - float(row_pred[0]))
            channel_error = abs(channel - float(channel_pred[0]))
            assert row_error <= GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: row {row} against '
                f'predicted {float(row_pred[0])}')
            assert channel_error <= GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: channel {channel} against '
                f'predicted {float(channel_pred[0])}')
            max_row_error = max(max_row_error, row_error)
            max_channel_error = max(max_channel_error, channel_error)
            num_pairs += 1

    assert num_pairs > 0
    MAX_ERRORS[name] = (num_pairs, max_row_error, max_channel_error)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_voxel_centers_match_the_probe(name):
    """The scene's voxel coordinates agree with the probe's own rule."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    scalars = probe.geometry_scalars(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    expected = np.array([probe.voxel_center_xyz(i, j, k, scalars)
                         for i, j, k in voxels])
    assert np.allclose(scene.voxel_centers(voxels), expected,
                       atol=GEOMETRY_TOLERANCE)


# ── the fixed-object picture ─────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_volume_outline_matches_project_points(name):
    """view(v).volume_outline_on_detector is project_points of the corners."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    for view in range(scene.num_views):
        view_scene = scene.view(view)
        row, channel = scene.project_points(view_scene.volume_corners, view)
        expected = np.stack([row, channel], axis=1)
        assert view_scene.volume_outline_on_detector.shape == (8, 2)
        assert np.allclose(view_scene.volume_outline_on_detector, expected,
                           atol=GEOMETRY_TOLERANCE)


def detector_indices_from_moved_detector(scene, view_scene, point):
    """Detector indices of one object-frame point, from the moved primitives.

    This repeats the projection using only the drawing primitives of
    ``view(v)``: the source position (or the ray direction), the detector
    origin, and the detector's two axes, all of which carry the inverse of the
    view's action.  Agreement with ``project_points`` therefore shows that the
    inverse action was applied consistently to the source and to the detector.

    A flat detector is a plane, so the ray from the source through the point is
    intersected with it.  A curved detector's channel coordinate is arc length,
    so the angle from the central ray is used instead.  This file's
    ``uv_to_indices`` takes (u, v) in that order and returns (row, channel).
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    origin = view_scene.detector_origin
    u_axis = view_scene.detector_u_axis
    v_axis = view_scene.detector_v_axis

    if view_scene.source is None:
        # Parallel rays: slide the point along the ray onto the detector plane.
        direction = view_scene.ray_direction
        step = np.dot(origin - point, direction)
        hit = point + step * direction
        u = np.dot(hit - origin, u_axis)
        v = np.dot(hit - origin, v_axis)
        return uv_to_indices(scene, u, v)

    to_point = point - view_scene.source
    forward = np.dot(to_point, view_scene.ray_direction)
    if scene.use_curved_detector:
        radius = scene.source_detector_dist
        u = radius * np.arctan2(np.dot(to_point, u_axis), forward)
        v = radius * np.dot(to_point, v_axis) / forward
    else:
        # The plane through detector_origin with the detector's normal.
        normal = view_scene.detector_normal
        denominator = np.dot(to_point, normal)
        step = np.dot(origin - view_scene.source, normal) / denominator
        hit = view_scene.source + step * to_point
        u = np.dot(hit - origin, u_axis)
        v = np.dot(hit - origin, v_axis)
    return uv_to_indices(scene, u, v)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_moved_source_and_detector_reproduce_the_projection(name):
    """The drawn source and detector project a point where the scene says.

    This is the check that the fixed-object picture is right.  The scene's
    ``project_points`` moves the object and holds the source fixed, which is
    what the projector does.  ``view(v)`` holds the object fixed and moves the
    source and the detector.  Re-deriving the detector indices from the moved
    primitives must give the same answer.
    """
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = scene.voxel_centers(voxels)
    for view in range(scene.num_views):
        view_scene = scene.view(view)
        rows, channels = scene.project_points(centers, view)
        for index, point in enumerate(centers):
            row, channel = detector_indices_from_moved_detector(
                scene, view_scene, point)
            assert abs(float(row) - rows[index]) < 1e-6
            assert abs(float(channel) - channels[index]) < 1e-6


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_detector_center_sits_at_the_grid_center(name):
    """The detector center is the (u, v) point that maps to the grid center."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    row, channel = uv_to_indices(scene, -scene.det_channel_offset,
                                 -scene.row_offset)
    assert float(row) == pytest.approx((scene.num_det_rows - 1) / 2.0)
    assert float(channel) == pytest.approx((scene.num_det_channels - 1) / 2.0)
    # The scene reports the same pair, beside the detector iso, which is the
    # point at (u, v) = (0, 0).
    iso, center = scene.detector_iso_and_center_indices()
    assert center == (pytest.approx((scene.num_det_rows - 1) / 2.0),
                      pytest.approx((scene.num_det_channels - 1) / 2.0))
    iso_row, iso_channel = uv_to_indices(scene, 0.0, 0.0)
    assert iso == (pytest.approx(float(iso_row)),
                   pytest.approx(float(iso_channel)))
    quantities = scene.derived_quantities()
    assert quantities['detector_center_u'] == pytest.approx(
        -scene.det_channel_offset)
    assert quantities['detector_center_v'] == pytest.approx(-scene.row_offset)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_view_primitives_have_the_promised_shapes(name):
    """Every primitive of every view has the shape a drawing expects."""
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    view_scene = scene.view(0)
    assert view_scene.source_draw.shape == (3,)
    assert view_scene.ray_direction.shape == (3,)
    assert np.linalg.norm(view_scene.ray_direction) == pytest.approx(1.0)
    assert view_scene.detector_center.shape == (3,)
    assert view_scene.detector_u_axis.shape == (3,)
    assert view_scene.detector_v_axis.shape == (3,)
    assert view_scene.detector_corners.shape == (4, 3)
    assert view_scene.detector_pixel0.shape == (3,)
    assert view_scene.volume_corners.shape == (8, 3)
    assert view_scene.voxel0_center.shape == (3,)
    assert view_scene.corner_rays.shape == (4, 2, 3)
    assert view_scene.detector_outline.ndim == 2
    assert view_scene.detector_outline.shape[1] == 3
    # The outline is closed.
    assert np.allclose(view_scene.detector_outline[0],
                       view_scene.detector_outline[-1],
                       atol=GEOMETRY_TOLERANCE)
    # Every corner ray ends on its corner.  Where the source is finite each
    # ray starts at it; where it is not, the rays are parallel instead.
    for index in range(4):
        assert np.allclose(view_scene.corner_rays[index, 1],
                           view_scene.detector_corners[index],
                           atol=GEOMETRY_TOLERANCE)
        if not scene.is_parallel_type:
            assert np.allclose(view_scene.corner_rays[index, 0],
                               view_scene.source_draw,
                               atol=GEOMETRY_TOLERANCE)

    sources, centers = scene.trajectory()
    assert sources.shape == (scene.num_views, 3)
    assert centers.shape == (scene.num_views, 3)
    assert np.allclose(sources[0], view_scene.source_draw,
                       atol=GEOMETRY_TOLERANCE)
    assert np.allclose(centers[0], view_scene.detector_center,
                       atol=GEOMETRY_TOLERANCE)


# ── the parallel row rule ────────────────────────────────────────────────────

def test_parallel_rows_ignore_the_detector_row_parameters():
    """A parallel scene's rows do not move when the row parameters change.

    The parallel projector sends recon slice m to detector row m, so
    ``delta_det_row`` and ``det_row_offset`` take no part.  A scene that used
    them would draw the rows in the wrong places.
    """
    cfg = CONFIGS_BY_NAME['parallel']
    model, scene = build_scene(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = scene.voxel_centers(voxels)
    rows_before, _ = scene.project_points(centers, 3)

    changed = scene.with_parameters(dict(delta_det_row=3.7,
                                         det_row_offset=6.25))
    rows_after, _ = changed.project_points(centers, 3)
    assert np.allclose(rows_after, rows_before, atol=GEOMETRY_TOLERANCE)
    assert changed.row_pitch == pytest.approx(scene.delta_voxel)
    assert changed.row_offset == 0.0

    # The same rule holds for the drawn detector height.
    assert changed.detector_size()[1] == pytest.approx(
        scene.num_det_rows * scene.delta_voxel)


# ── edge case: a single detector row ─────────────────────────────────────────

def single_row_parallel_config():
    """A parallel configuration with one detector row and one recon slice.

    The parallel model requires the slice count to equal the detector row
    count, so a one-row detector means a one-slice volume.
    """
    return dict(
        name='parallel one row',
        kind='parallel',
        sinogram_shape=(4, 1, 40),
        recon_shape=(10, 12, 1),
        angles=probe.VIEW_ANGLES[:4],
        params=dict(delta_det_channel=1.1, delta_det_row=0.9,
                    det_channel_offset=1.35, det_row_offset=-0.85,
                    delta_voxel=1.0, voxel_row_aspect=1.25,
                    voxel_slice_aspect=1.0),
    )


def test_single_detector_row_projects_correctly_and_draws_finite_primitives():
    """One detector row: the projector agrees and every primitive is finite.

    The parallel case is checked against the real projector, and both the
    parallel and the cone case must still produce a drawable outline and finite
    derived quantities.
    """
    cfg = single_row_parallel_config()
    model, scene = build_scene(cfg)
    assert scene.num_det_rows == 1
    # The probe's voxel list assumes at least three slices.  With one slice,
    # every voxel sits in slice 0, so clamp the slice index and drop repeats.
    voxels = sorted({(i, j, 0) for i, j, _ in
                     probe.probe_voxels(cfg['recon_shape'])})
    num_pairs = 0
    for voxel in voxels:
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        center = scene.voxel_centers([voxel])
        for view, (reason, row, channel) in enumerate(measured):
            if reason is not None:
                continue
            row_pred, channel_pred = scene.project_points(center, view)
            assert abs(row - float(row_pred[0])) <= GATE_PIXELS
            assert abs(channel - float(channel_pred[0])) <= GATE_PIXELS
            num_pairs += 1
    assert num_pairs > 0

    view_scene = scene.view(0)
    assert np.all(np.isfinite(view_scene.detector_outline))
    assert np.all(np.isfinite(view_scene.corner_rays))
    quantities = scene.derived_quantities()
    assert quantities['detector_height'] == pytest.approx(scene.delta_voxel)
    assert np.isfinite(quantities['axial_fov_alu'])

    # The same detector under a cone geometry.
    cone_cfg = dict(CONFIGS_BY_NAME['cone flat'])
    cone_cfg = dict(cone_cfg, name='cone one row', sinogram_shape=(8, 1, 48),
                    recon_shape=(10, 12, 1))
    _, cone_scene = build_scene(cone_cfg)
    assert cone_scene.num_det_rows == 1
    cone_quantities = cone_scene.derived_quantities()
    assert 0.0 < cone_quantities['cone_angle_deg'] < 2.0
    cone_view = cone_scene.view(2)
    assert np.all(np.isfinite(cone_view.detector_outline))
    assert cone_view.detector_outline.shape == (5, 3)


# ── edge case: an infinite source-detector distance ──────────────────────────

def infinite_distance_cone_config():
    """A cone configuration whose source-detector distance is infinite."""
    cfg = dict(CONFIGS_BY_NAME['cone flat'])
    return dict(cfg, name='cone infinite sdd',
                source_detector_dist=np.inf, source_iso_dist=100.0)


def test_infinite_source_detector_dist_is_a_parallel_projection():
    """An infinite source-detector distance gives magnification one.

    The cone model accepts the infinite distance, and its projector then uses
    u = x and v = z, which is a parallel projection that still uses
    ``delta_det_row`` and ``det_row_offset`` for its rows.  The scene must say
    the same, and it must place the source and the detector by the drawing
    rule, because neither has a finite position.
    """
    cfg = infinite_distance_cone_config()
    model, scene = build_scene(cfg)
    assert scene.magnification == pytest.approx(1.0)
    assert scene.is_parallel_type is True
    assert scene.row_pitch == pytest.approx(scene.delta_det_row)
    assert scene.row_offset == pytest.approx(scene.det_row_offset)

    view_scene = scene.view(0)
    assert view_scene.source is None
    assert np.all(np.isfinite(view_scene.source_draw))
    assert np.all(np.isfinite(view_scene.detector_outline))
    assert 'infinite' in scene.derived_quantities()['drawing_note']

    # The projector must agree with the scene here as it does elsewhere.
    num_pairs = 0
    for voxel in probe.probe_voxels(cfg['recon_shape']):
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        center = scene.voxel_centers([voxel])
        for view, (reason, row, channel) in enumerate(measured):
            if reason is not None:
                continue
            row_pred, channel_pred = scene.project_points(center, view)
            assert abs(row - float(row_pred[0])) <= GATE_PIXELS
            assert abs(channel - float(channel_pred[0])) <= GATE_PIXELS
            num_pairs += 1
    assert num_pairs > 0


# ── edge case: a curved detector ─────────────────────────────────────────────

def test_curved_detector_outline_lies_on_the_cylinder():
    """A curved outline is an arc at the cylinder's radius from the source.

    The cylinder's axis passes through the source parallel to z, so every point
    of the detector surface is at ``source_detector_dist`` from the source when
    measured in the xy plane.
    """
    cfg = CONFIGS_BY_NAME['cone curved']
    _, scene = build_scene(cfg)
    assert scene.use_curved_detector is True
    for view in (0, 3, 5):
        view_scene = scene.view(view)
        assert view_scene.detector_outline.shape == (2 * CURVED_ARC_SAMPLES + 1, 3)
        offsets = view_scene.detector_outline - view_scene.source
        horizontal = np.hypot(offsets[:, 0], offsets[:, 1])
        assert np.allclose(horizontal, scene.source_detector_dist, atol=1e-6)
        corner_offsets = view_scene.detector_corners - view_scene.source
        assert np.allclose(np.hypot(corner_offsets[:, 0], corner_offsets[:, 1]),
                           scene.source_detector_dist, atol=1e-6)


def test_curved_detector_rows_use_the_tangent_plane():
    """The curved row rule is the tangent plane and not the cylinder crossing.

    The two rules differ by hypot(x, source_iso_dist - y) / (source_iso_dist -
    y), so they separate only at a large fan angle.  The curved probe
    configuration has a large fan angle for that reason, and the projector was
    measured to follow the tangent plane.  This test states the difference so
    that a later change to the cylinder rule would be caught.
    """
    cfg = CONFIGS_BY_NAME['cone curved']
    _, scene = build_scene(cfg)
    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = scene.voxel_centers(voxels)
    view = 0
    rows, _ = scene.project_points(centers, view)

    projector_frame = scene.to_projector_frame(centers, view)
    x = projector_frame[:, 0]
    depth = scene.source_iso_dist - projector_frame[:, 1]
    z = projector_frame[:, 2]
    # The per-point magnification of a cone geometry, stated once in
    # mbirtorch.cone_beam._cone_xy_mag as
    # 1 / (1 / magnification - y / source_detector_dist), which is
    # source_detector_dist / depth here.
    y = projector_frame[:, 1]
    magnification = 1.0 / (1.0 / scene.magnification
                           - y / scene.source_detector_dist)
    tangent_v = magnification * z
    cylinder_v = scene.source_detector_dist * z / np.hypot(x, depth)
    tangent_rows, _ = uv_to_indices(scene, np.zeros_like(tangent_v), tangent_v)
    cylinder_rows, _ = uv_to_indices(scene, np.zeros_like(cylinder_v),
                                     cylinder_v)
    assert np.allclose(rows, tangent_rows, atol=GEOMETRY_TOLERANCE)
    assert np.max(np.abs(tangent_rows - cylinder_rows)) > 0.5


# ── the rotation axis and the translation path ───────────────────────────────

def test_rotating_scans_draw_an_axis_and_a_translation_scan_draws_its_path():
    """A rotating scan draws an axis through the volume's z range; a
    translation scan draws the path of the rotation center instead."""
    for name in ('parallel', 'cone flat', 'multiaxis'):
        cfg = CONFIGS_BY_NAME[name]
        _, scene = build_scene(cfg)
        view_scene = scene.view(0)
        assert view_scene.translation_path is None
        axis = view_scene.rotation_axis
        assert axis.shape == (2, 3)
        assert np.allclose(axis[:, :2], 0.0, atol=GEOMETRY_TOLERANCE)
        z_min, z_max = scene.volume_z_range()
        assert axis[0, 2] <= z_min + GEOMETRY_TOLERANCE
        assert axis[1, 2] >= z_max - GEOMETRY_TOLERANCE

    cfg = CONFIGS_BY_NAME['translation']
    _, scene = build_scene(cfg)
    view_scene = scene.view(0)
    assert view_scene.rotation_axis is None
    assert view_scene.translation_path.shape == (scene.num_views, 3)
    assert np.allclose(view_scene.translation_path,
                       probe.TRANSLATION_VECTORS, atol=1e-6)
    # In the fixed-object picture the source moves by plus the translation
    # vector, so the source path is the path shifted to the source.
    sources, _ = scene.trajectory()
    expected = (np.array([0.0, scene.source_iso_dist, 0.0])
                + probe.TRANSLATION_VECTORS)
    assert np.allclose(sources, expected, atol=1e-6)
    # No mask is drawn: the translation model turns the mask off.
    assert scene.ror_cylinder() is None


# ── edge case: a helical cone scan ───────────────────────────────────────────

def test_helical_trajectory_rises_with_the_z_shift():
    """The drawn source rises toward +z as the helical shift grows.

    The projector subtracts the shift from the object's z, so in the
    fixed-object picture the source and the detector move toward +z by the
    shift.
    """
    cfg = CONFIGS_BY_NAME['cone helical']
    _, scene = build_scene(cfg)
    sources, centers = scene.trajectory()
    assert np.allclose(sources[:, 2], probe.HELICAL_Z_SHIFTS, atol=1e-6)
    assert np.all(np.diff(sources[:, 2]) > 0)
    # The detector center rises by the same shift.
    assert np.allclose(centers[:, 2] - centers[0, 2],
                       probe.HELICAL_Z_SHIFTS - probe.HELICAL_Z_SHIFTS[0],
                       atol=1e-6)
    # The source turns clockwise seen from +z, so its in-plane angle from the
    # +y axis follows minus the view angle.
    angles = np.arctan2(sources[:, 0], sources[:, 1])
    assert np.allclose(angles, np.arctan2(np.sin(probe.VIEW_ANGLES),
                                          np.cos(probe.VIEW_ANGLES)),
                       atol=1e-6)
    assert scene.derived_quantities()['helical_travel_alu'] == pytest.approx(
        float(np.ptp(probe.HELICAL_Z_SHIFTS)))


# ── the multiaxis source-side convention ────────────────────────────────────

def test_multiaxis_source_side_is_a_documented_choice():
    """A positive elevation puts the source below the xy plane by default.

    The convention flag reverses the source and the detector and leaves every
    detector index unchanged, because a parallel projection is the same in both
    directions along a ray.
    """
    cfg = CONFIGS_BY_NAME['multiaxis']
    model = probe.build_model(cfg)
    default_scene = GeometryScene.from_model(model)
    flipped_scene = GeometryScene.from_model(
        model, multiaxis_source_on_plus_y=False)

    view = 0
    assert probe.MULTIAXIS_ELEVATIONS[view] > 0
    default_view = default_scene.view(view)
    assert default_view.source_draw[2] < 0.0
    assert default_view.detector_center[2] > 0.0
    assert default_view.source is None

    flipped_view = flipped_scene.view(view)
    assert flipped_view.source_draw[2] > 0.0

    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = default_scene.voxel_centers(voxels)
    for index in range(default_scene.num_views):
        expected = default_scene.project_points(centers, index)
        actual = flipped_scene.project_points(centers, index)
        assert np.allclose(actual[0], expected[0], atol=GEOMETRY_TOLERANCE)
        assert np.allclose(actual[1], expected[1], atol=GEOMETRY_TOLERANCE)
    assert 'convention' in default_scene.derived_quantities()['drawing_note']


# ── the region of reconstruction and the drawing distance ───────────────────

def test_ror_cylinder_matches_the_inscribed_ellipse():
    """The drawn region is the mask's own ellipse, through the centers of the
    outermost voxels.

    mbirtorch's default mask, ``vcd_utils.get_2d_ror_mask``, keeps the voxels
    whose centers lie inside that ellipse, so its semi-axes are (n - 1) / 2
    voxel pitches.  The projector's shadow of a volume of ones inside the mask
    matches this ellipse and not the one half a voxel larger, which is what
    ``get_support_radius`` bounds.
    """
    cfg = CONFIGS_BY_NAME['cone flat']
    _, scene = build_scene(cfg)
    cylinder = scene.ror_cylinder()
    half_x = 0.5 * (scene.num_cols - 1) * scene.delta_voxel
    half_y = 0.5 * (scene.num_rows - 1) * scene.delta_voxel_row
    assert cylinder['semi_axis_x'] == pytest.approx(half_x)
    assert cylinder['semi_axis_y'] == pytest.approx(half_y)
    assert cylinder['radius'] == pytest.approx(max(half_x, half_y))
    z_min, z_max = scene.volume_z_range()
    assert cylinder['z_min'] == pytest.approx(z_min)
    assert cylinder['z_max'] == pytest.approx(z_max)

    assert scene.with_parameters(
        dict(use_ror_mask=False)).ror_cylinder() is None


def test_drawing_distance_does_not_change_the_projection():
    """The drawing distance moves the picture and not the detector indices.

    A parallel projection's detector coordinates do not depend on where along
    the rays the detector plane sits, so a different drawing distance must give
    the same indices and a different drawn position.
    """
    cfg = CONFIGS_BY_NAME['parallel']
    model = probe.build_model(cfg)
    near = GeometryScene.from_model(model, drawing_distance_factor=1.5)
    far = GeometryScene.from_model(model, drawing_distance_factor=4.0)
    assert far.drawing_distance > near.drawing_distance

    voxels = probe.probe_voxels(cfg['recon_shape'])
    centers = near.voxel_centers(voxels)
    for view in range(near.num_views):
        expected = near.project_points(centers, view)
        actual = far.project_points(centers, view)
        assert np.allclose(actual[0], expected[0], atol=GEOMETRY_TOLERANCE)
        assert np.allclose(actual[1], expected[1], atol=GEOMETRY_TOLERANCE)
    assert (np.linalg.norm(far.view(0).detector_center)
            > np.linalg.norm(near.view(0).detector_center))


# ── the fit statement: which shape, which rule, and the swept coverage ──────

#: The flat cone configuration with its detector narrowed to this many
#: channels.  At 28 channels the volume box's corners project past the
#: detector's edge while the region-of-reconstruction cylinder still lands
#: inside it, which is the pair of answers
#: ``test_the_fit_report_tests_the_right_shape_and_splits_lateral_from_axial``
#: is about.
NARROW_CHANNEL_COUNT = 28

#: The relative tolerance of the swept-coverage identities.  Each identity is
#: exact in exact arithmetic, and the computation is a handful of
#: multiplications and divisions in float64, so only rounding separates the two
#: sides.
COVERAGE_TOLERANCE = 1e-9


def narrow_detector_cone_scene(use_ror_mask=True):
    """The flat cone scene with a detector too narrow for its volume box.

    The probe's own flat cone configuration keeps the volume well inside the
    detector, so it cannot show the difference between the two shapes.  This
    copy narrows the detector to :data:`NARROW_CHANNEL_COUNT` channels and
    changes nothing else.
    """
    _, scene = build_scene(CONFIGS_BY_NAME['cone flat'])
    return scene.with_parameters(
        dict(sinogram_shape=(scene.num_views, scene.num_det_rows,
                             NARROW_CHANNEL_COUNT),
             use_ror_mask=use_ror_mask))


def test_the_fit_report_tests_the_right_shape_and_splits_lateral_from_axial():
    """The fit statement tests the region of reconstruction, not the box, and
    reports the lateral and the axial answer apart.

    The reconstruction box is the square around the region of reconstruction,
    so its corners stick out past the field of view, and without this rule the
    fit statement reads "no" for a scan that reconstructs nothing outside the
    detector.  The scene must test the cylinder whenever the mask describes
    one.

    The probe's flat cone configuration cannot show this, because its volume
    was chosen to sit well inside the detector and both shapes fit.  The
    detector is narrowed to :data:`NARROW_CHANNEL_COUNT` channels instead,
    which puts the box's corners outside it and leaves the cylinder inside.

    An automatically sized cone scan then shows the other half: its box misses
    the detector by tens of channels while the cylinder inside it misses by
    less than one, and the cylinder misses in the rows, so the lateral answer
    is yes and the axial answer is no.  One answer would hide which direction
    misses.
    """
    cfg = CONFIGS_BY_NAME['cone flat']
    _, cone_scene = build_scene(cfg)
    fits, overshoot = cone_scene.volume_fits_detector()
    assert fits is True
    assert overshoot == 0.0
    grown_box = cone_scene.with_parameters(dict(recon_shape=(60, 60, 8)))
    fits, overshoot = grown_box.volume_fits_detector()
    assert fits is False
    assert overshoot > 1.0

    masked = narrow_detector_cone_scene(use_ror_mask=True)
    report = masked.fit_report()
    assert report['shape'] == 'cylinder'
    assert report['fits'] is True
    assert report['worst_overshoot_pixels'] == 0.0
    assert report['worst_channel_overshoot_pixels'] == 0.0
    assert report['worst_row_overshoot_pixels'] == 0.0
    assert report['views_leaving_detector'] == 0

    unmasked = narrow_detector_cone_scene(use_ror_mask=False)
    report = unmasked.fit_report()
    assert report['shape'] == 'box'
    assert report['fits'] is False
    assert report['worst_channel_overshoot_pixels'] > 1.0
    # The box leaves the detector sideways and not in the rows.
    assert report['worst_row_overshoot_pixels'] == 0.0
    # The count is a count of views and not a yes or no: the box's corners
    # swing past the detector's edge in some views and not in others.
    assert 0 < report['views_leaving_detector'] < unmasked.num_views

    # An automatically sized cone scan: the box misses by tens of channels, the
    # cylinder inside it misses in the rows only.  The beam narrows toward the
    # source, so the top of the cylinder on the source side sits outside the
    # rows the detector covers, and the statement still reads "no" with a
    # number that names a real truncation instead of a corner nobody
    # reconstructs.
    model = mbirtorch.ConeBeamModel(
        (180, 96, 128), np.linspace(0, 2 * np.pi, 180, endpoint=False),
        source_detector_dist=512.0, source_iso_dist=256.0, compile_mode='off')
    automatic = GeometryScene.from_model(model)
    cylinder = automatic.fit_report()
    box = automatic.with_parameters(dict(use_ror_mask=False)).fit_report()

    assert box['shape'] == 'box' and cylinder['shape'] == 'cylinder'
    assert box['worst_channel_overshoot_pixels'] > 10.0
    assert cylinder['worst_channel_overshoot_pixels'] == 0.0
    assert cylinder['worst_row_overshoot_pixels'] > 1.0
    assert cylinder['fits_laterally'] is True
    assert cylinder['fits_axially'] is False
    assert cylinder['fits'] is False

    # A volume grown past the field of view misses laterally as well.
    grown = automatic.with_parameters(dict(recon_shape=(200, 200, 96)))
    grown_report = grown.fit_report()
    assert grown_report['fits_laterally'] is False
    assert grown_report['fits'] is False


def test_the_swept_coverage_is_one_views_coverage_plus_the_helical_travel():
    """A scan that does not travel sweeps exactly one view's axial coverage,
    and helical travel adds itself to that range.

    Every view of a non-traveling scan covers the same z range on the rotation
    axis, so the swept range is that range.  Its length is the detector's
    height divided by the magnification, which is the axial field of view, and
    it is centered on the z the central ray lands at, which is minus the row
    offset divided by the magnification.

    The helical comparison holds every parameter of the helical configuration
    fixed and sets the per-view z shifts to zero, so the only difference
    between the two scenes is the travel.
    """
    _, scene = build_scene(CONFIGS_BY_NAME['cone flat'])
    report = scene.fit_report()
    length = report['swept_z_max'] - report['swept_z_min']
    height, magnification = scene.detector_size()[1], scene.magnification
    assert length == pytest.approx(height / magnification,
                                   rel=COVERAGE_TOLERANCE)
    assert length == pytest.approx(scene.derived_quantities()['axial_fov_alu'],
                                   rel=COVERAGE_TOLERANCE)
    center = 0.5 * (report['swept_z_min'] + report['swept_z_max'])
    assert center == pytest.approx(-scene.row_offset / magnification,
                                   rel=COVERAGE_TOLERANCE)

    _, scene = build_scene(CONFIGS_BY_NAME['cone helical'])
    view_params = np.asarray(scene.params['view_params_array'],
                             dtype=np.float64).copy()
    view_params[:, 1] = 0.0
    still = scene.with_parameters(dict(view_params_array=view_params))

    moving_report, still_report = scene.fit_report(), still.fit_report()
    moving = moving_report['swept_z_max'] - moving_report['swept_z_min']
    standing = still_report['swept_z_max'] - still_report['swept_z_min']
    assert scene.helical_travel() > 0.0
    assert still.helical_travel() == 0.0
    assert moving - standing == pytest.approx(scene.helical_travel(),
                                              rel=COVERAGE_TOLERANCE)


def test_a_helical_scan_is_judged_by_the_helical_rule():
    """A helical scan is asked about its channels and its swept z extent.

    Three helical scans are checked.  The probe's own helical configuration has
    a volume small enough to land on the detector in every view, so it leaves
    the detector in no view at all.  The second grows that volume until it is
    taller than one view's axial coverage, which is what an automatically sized
    helical scan is: it then leaves the detector in every view, and the
    statement still reads "yes" because the detector sweeps the whole volume
    over the scan.  Without the helical rule the statement reads "no" for
    every helical scan.

    The third puts the z shifts in two groups far enough apart that the two
    coverage ranges do not meet.  Its volume runs from the first group to the
    second, so it lies between ``swept_z_min`` and ``swept_z_max`` and yet part
    of it is covered by no view.  The union of the per-view ranges reports this
    and their hull does not, which is why the union is what
    ``z_extent_covered`` uses.

    A helical scan leaves the detector either in no view or in every view, and
    not in some of them, whenever what it leaves is the axial coverage.  The
    volume's z extent is the same in every view, and so is one view's coverage,
    so the two either overlap in every view or in none.
    """
    _, scene = build_scene(CONFIGS_BY_NAME['cone helical'])
    report = scene.fit_report()
    assert report['helical_rule'] is True
    assert report['views_leaving_detector'] == 0
    assert report['z_extent_covered'] is True
    assert report['worst_channel_overshoot_pixels'] == 0.0
    assert report['fits'] is True
    assert report['fits_laterally'] is True and report['fits_axially'] is True

    # The same scan with a volume taller than one view's axial coverage.
    tall = scene.with_parameters(dict(recon_shape=(10, 12, 25)))
    tall_report = tall.fit_report()
    z_min, z_max = tall.volume_z_range()
    one_view = tall.detector_size()[1] / tall.magnification
    assert z_max - z_min > one_view
    assert tall_report['views_leaving_detector'] == tall.num_views
    assert tall_report['worst_row_overshoot_pixels'] > 0.0
    assert tall_report['worst_channel_overshoot_pixels'] == 0.0
    assert tall_report['z_extent_covered'] is True
    assert tall_report['fits'] is True
    # The answer is the two questions of the helical rule and nothing else,
    # and the axial half is the swept coverage, not the rows.
    assert tall_report['fits_axially'] is True
    assert tall_report['fits'] == (
        tall_report['worst_channel_overshoot_pixels'] == 0.0
        and tall_report['z_extent_covered'])

    # Views in two groups leave a z range that no view covers.
    view_params = np.asarray(scene.params['view_params_array'],
                             dtype=np.float64).copy()
    view_params[:, 1] = np.repeat([0.0, 40.0], scene.num_views // 2)
    gapped = scene.with_parameters(dict(view_params_array=view_params,
                                        recon_shape=(10, 12, 50),
                                        recon_slice_offset=20.0))

    report = gapped.fit_report()
    z_min, z_max = gapped.volume_z_range()
    assert report['helical_rule'] is True
    # The hull holds the whole volume; the union does not.
    assert report['swept_z_min'] < z_min and z_max < report['swept_z_max']
    assert report['z_extent_covered'] is False
    assert report['worst_channel_overshoot_pixels'] == 0.0
    assert report['fits'] is False


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_ror_outline_is_the_two_projected_rims(name):
    """The drawn region outline is project_points of the scene's fit points.

    The outline must come from the one projection the scene has, for the reason
    the volume box's outline must: a second projection could disagree with the
    projector.  The rim at ``z_min`` also has to project to lower row indices
    than the rim at ``z_max``, point for point, because a larger z lands on a
    larger row in every one of these geometries.
    """
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    for view_index in (0, scene.num_views - 1):
        view = scene.view(view_index)
        if scene.ror_cylinder() is None:
            assert view.ror_outline_on_detector is None
            continue
        outline = view.ror_outline_on_detector
        assert outline.shape == (2, RIM_SAMPLES, 2)
        row, channel = scene.project_points(scene.fit_points(), view_index)
        expected = np.stack([row, channel], axis=1).reshape(2, RIM_SAMPLES, 2)
        assert np.allclose(outline, expected, atol=GEOMETRY_TOLERANCE)
        assert np.all(outline[0, :, 0] < outline[1, :, 0])

    # The translation model turns the mask off, so it has no outline to draw.
    if name == 'translation':
        assert scene.ror_cylinder() is None
        assert scene.view(0).ror_outline_on_detector is None


# ── the corner rays of a source-free geometry ───────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_corner_rays_are_parallel_where_there_is_no_source(name):
    """A geometry with no source position gets four parallel rays.

    Four rays converging on the drawn source would be the picture of a cone
    beam.  Each ray must instead be parallel to the ray direction and as long
    as the drawn central ray, so that the five rays start in one plane.
    """
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    view = scene.view(2)
    rays = view.corner_rays
    if not scene.is_parallel_type:
        assert view.source is not None
        return

    assert view.source is None
    central_length = float(np.linalg.norm(view.source_draw
                                          - view.detector_origin))
    for index in range(4):
        segment = rays[index, 1] - rays[index, 0]
        length = float(np.linalg.norm(segment))
        assert length == pytest.approx(central_length)
        assert np.allclose(segment / length, view.ray_direction,
                           atol=GEOMETRY_TOLERANCE)
    # Parallel means no two rays meet: the four starting points are as far
    # apart as the four corners they end on.
    for index in range(4):
        for other in range(index + 1, 4):
            start_gap = np.linalg.norm(rays[index, 0] - rays[other, 0])
            end_gap = np.linalg.norm(rays[index, 1] - rays[other, 1])
            assert float(start_gap) == pytest.approx(float(end_gap))


# ── the rotation-direction arc ──────────────────────────────────────────────

def test_rotation_direction_arc_radius_and_sense():
    """The arc starts at the source, keeps its radius, and turns its way.

    The probe's view angles rise with the view index, so the source turns
    clockwise seen from +z, which is a falling azimuth.  A scan whose angles
    fall gets the arc the other way.  A translation scan and a one-view scan
    say nothing about travel and get no arc.
    """
    for name, cfg in CONFIGS_BY_NAME.items():
        _, scene = build_scene(cfg)
        view = scene.view(2)

        if scene.kind == 'translation':
            assert view.rotation_direction_arc is None, name
            assert view.source_travel_sense is None, name
            continue

        arc = view.rotation_direction_arc
        assert arc.shape == (ROTATION_ARC_SAMPLES, 3), name
        assert np.allclose(arc[0], view.source_draw,
                           atol=GEOMETRY_TOLERANCE), name
        radius = np.hypot(arc[:, 0], arc[:, 1])
        assert np.allclose(radius, radius[0], atol=GEOMETRY_TOLERANCE), name
        assert np.allclose(arc[:, 2], view.source_draw[2],
                           atol=GEOMETRY_TOLERANCE), name

        # The probe's angles rise from view 2 to view 3, so the source turns
        # clockwise seen from +z: the azimuth falls along the arc.
        assert scene.angles[3] > scene.angles[2], name
        azimuth = np.unwrap(np.arctan2(arc[:, 1], arc[:, 0]))
        assert azimuth[-1] < azimuth[0], name
        assert view.source_travel_sense == CLOCKWISE_FROM_PLUS_Z, name

    # A scan whose angles fall turns the other way.
    _, scene = build_scene(CONFIGS_BY_NAME['cone flat'])
    falling = np.asarray(scene.params['view_params_array'],
                         dtype=np.float64).copy()
    falling[:, 0] = -falling[:, 0]
    reversed_scene = scene.with_parameters(dict(view_params_array=falling))

    view = reversed_scene.view(2)
    assert reversed_scene.angles[3] < reversed_scene.angles[2]
    azimuth = np.unwrap(np.arctan2(view.rotation_direction_arc[:, 1],
                                   view.rotation_direction_arc[:, 0]))
    assert azimuth[-1] > azimuth[0]
    assert view.source_travel_sense == COUNTERCLOCKWISE_FROM_PLUS_Z

    # One view says nothing about which way the source travels.
    first_row = np.asarray(scene.params['view_params_array'],
                           dtype=np.float64)[:1]
    single = scene.with_parameters(
        dict(sinogram_shape=(1, scene.num_det_rows, scene.num_det_channels),
             view_params_array=first_row))
    view = single.view(0)
    assert view.rotation_direction_arc is None
    assert view.source_travel_sense is None


# ── the trajectory over all views ───────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_trajectory_matches_the_per_view_scenes(name):
    """The vectorized trajectory equals the per-view scenes it replaces.

    ``trajectory`` computes the whole scan at once so that an 1800-view model
    redraws at interactive speed.  This test is the check that the fast form
    and the plain form agree.
    """
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    sources, centers = scene.trajectory()
    assert sources.shape == (scene.num_views, 3)
    assert centers.shape == (scene.num_views, 3)
    for view_index in range(scene.num_views):
        view = scene.view(view_index)
        assert np.allclose(sources[view_index], view.source_draw,
                           atol=GEOMETRY_TOLERANCE), view_index
        assert np.allclose(centers[view_index], view.detector_center,
                           atol=GEOMETRY_TOLERANCE), view_index


# ── comparing two scenes ────────────────────────────────────────────────────

def test_with_parameters_copies_and_replaces():
    """A copy carries the drawing options and the one changed parameter."""
    cfg = CONFIGS_BY_NAME['cone flat']
    _, scene = build_scene(cfg)
    shifted = scene.with_parameters(
        dict(det_channel_offset=scene.det_channel_offset
             + 10.0 * scene.delta_det_channel))

    assert shifted is not scene
    assert shifted.kind == scene.kind
    assert shifted.drawing_options() == scene.drawing_options()
    assert scene.params['det_channel_offset'] == cfg['params'][
        'det_channel_offset']
    # Ten channels of offset move a fixed point's image by ten channels.
    point = [[0.0, 0.0, 0.0]]
    row, channel = scene.project_points(point, 2)
    row_shifted, channel_shifted = shifted.project_points(point, 2)
    assert float(channel_shifted[0] - channel[0]) == pytest.approx(10.0)
    assert float(row_shifted[0] - row[0]) == pytest.approx(0.0)


# ── the angle-0 reference view ───────────────────────────────────────────────

#: The tolerance of the field-by-field comparison of two views.  The two are
#: built by the same code from the same numbers, so they agree far better than
#: this.
REFERENCE_TOLERANCE = 1e-9


def assert_views_agree(first, second, tolerance=REFERENCE_TOLERANCE):
    """Fail unless two ViewScene objects agree entry by entry."""
    names = list(first.__dataclass_fields__)
    assert names == list(second.__dataclass_fields__)
    for name in names:
        mine, theirs = getattr(first, name), getattr(second, name)
        if mine is None or theirs is None:
            assert mine is None and theirs is None, name
        elif isinstance(mine, (int, str, bool)):
            assert mine == theirs, name
        elif isinstance(mine, dict):
            assert sorted(mine) == sorted(theirs), name
            for key in mine:
                assert np.allclose(mine[key], theirs[key], atol=tolerance), (
                    f'{name}[{key}]')
        else:
            assert np.shape(mine) == np.shape(theirs), name
            assert np.allclose(mine, theirs, atol=tolerance), name


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_reference_view_is_the_zero_angle_zero_shift_view(name):
    """The reference view is the view action's identity, in every geometry.

    Its source sits on the +y axis at ``source_iso_dist`` wherever the source
    has a position, with no x component, no helical z shift, and no
    translation, and the detector lies opposite it on the -y side.  A multiaxis
    geometry has no position free of elevation, so its reference keeps view 0's
    elevation and only the azimuth goes to zero.  Where view 0 is already the
    identity, the reference and view 0 agree entry by entry.
    """
    cfg = CONFIGS_BY_NAME[name]
    _, scene = build_scene(cfg)
    reference = scene.reference_view()
    # Azimuth zero: the source and the detector lie in the y-z plane.
    assert abs(float(reference.source_draw[0])) < REFERENCE_TOLERANCE
    assert abs(float(reference.detector_origin[0])) < REFERENCE_TOLERANCE

    if name == 'multiaxis':
        elevation = float(scene.elevations[0])
        expected = np.array([0.0, -np.cos(elevation), np.sin(elevation)])
        assert np.allclose(reference.ray_direction, expected,
                           atol=REFERENCE_TOLERANCE)
    else:
        # No helical z shift and no elevation: the gantry lies in the xy plane.
        assert abs(float(reference.source_draw[2])) < REFERENCE_TOLERANCE
        assert abs(float(reference.detector_origin[2])) < REFERENCE_TOLERANCE

    if reference.source is not None:
        # The source sits on the +y axis and the detector opposite it, whatever
        # view 0's angle and translation are.
        expected = np.array([0.0, scene.source_iso_dist, 0.0])
        assert np.allclose(reference.source, expected,
                           atol=REFERENCE_TOLERANCE)
        assert np.allclose(reference.source_draw, expected,
                           atol=REFERENCE_TOLERANCE)
        assert reference.detector_origin[1] < 0.0
        if name == 'cone flat':
            # The probe's flat cone scan starts at -0.3 radians, so its view 0
            # source is somewhere else.
            assert float(scene.angles[0]) != 0.0
            assert not np.allclose(scene.view(0).source, expected, atol=1e-3)
        if name == 'translation':
            assert not np.allclose(scene.translation_vectors[0], 0.0)

    if name in ('parallel', 'cone flat'):
        # With view 0 at angle 0 and no z shift, view 0 is the identity, so the
        # reference must be that view entry by entry.
        identity_cfg = dict(cfg)
        identity_cfg['angles'] = np.linspace(
            0.0, 2.0 * np.pi, len(cfg['angles']), endpoint=False)
        _, identity_scene = build_scene(identity_cfg)
        assert float(identity_scene.angles[0]) == 0.0
        assert_views_agree(identity_scene.reference_view(),
                           identity_scene.view(0))


if __name__ == '__main__':
    # Print the per-geometry gate errors for the run record.
    for cfg in probe.CONFIGS:
        test_projection_matches_projector(cfg['name'])
    print(f'{"geometry":16s} {"pairs":>6s} {"max row err":>12s} '
          f'{"max chan err":>13s}')
    for name, (pairs, row_error, channel_error) in MAX_ERRORS.items():
        print(f'{name:16s} {pairs:6d} {row_error:12.4f} {channel_error:13.4f}')
