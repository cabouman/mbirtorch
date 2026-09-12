"""Six scan geometries, an independent prediction of where a voxel lands, and
a measurement of where the projector actually puts it.

This is a helper module, not a test file.  It holds three things that the
tests of :meth:`TomographyModel.project_points` and of the geometry viewer
both need:

* ``CONFIGS``: six small scan geometries, one per model class plus the flat,
  curved, and helical cone variants.  Each one is deliberately awkward --
  non-integer detector offsets of both signs, two different detector pitches,
  a voxel row aspect that is never one -- so that a sign or a pitch that is
  wrong anywhere shows up somewhere.
* ``predict_parallel`` and its three siblings: where a voxel center should
  land on the detector, written from a plain statement of the geometry with
  numpy vectors and a ray/plane intersection.  These never call any of the
  projector's own formulas, so agreement with the projector is evidence and
  not a tautology.  Keep it that way.
* ``measure_footprint_centroids``: where the projector actually puts a voxel,
  measured as the intensity-weighted centroid of its forward-projected
  footprint.

The geometric statement being tested, in words.  A voxel index (i, j, k) has
object coordinates (x, y, z): the column index j runs along x, the row index i
runs along y, and the slice index k runs along z.  Each view applies an action
to the object and leaves the source and detector fixed.  Parallel and cone
rotate the object about the z axis by the view angle.  Multiaxis rotates the
object about z by the azimuth and sends the rays along a direction tilted out
of the xy plane by the elevation.  Translation shifts the object by a vector.
Cone and translation put the point source on the positive y axis at
source_iso_dist from the origin, and the detector on the negative y side at
source_detector_dist from the source.  A detector row index counts the z
direction and a channel index counts the x direction.  The offsets
det_channel_offset and det_row_offset move the detector grid relative to the
central ray.

The ``flips`` argument of the predictors negates one convention at a time.
The tests use it to show that the curved detector's row rule is the
tangent-plane rule and not the cylinder rule.
"""

import numpy as np
import torch

import mbirtorch

# A footprint whose total mass is below this is treated as entirely off the
# detector.  The projected mass of one voxel is of order one.
MASS_FLOOR = 1e-9

# View angles shared by the geometries that rotate the object.  They are not
# equally spaced and one of them is negative.
VIEW_ANGLES = np.array([-0.30, 0.0, 0.21, 0.55, 1.10, 1.90, 2.40, 3.00])

# Per-view elevation for the multiaxis geometry, in radians.  Most views sit at
# about 20 degrees; one view is negative so that the elevation sign is
# exercised, and one is near zero.
MULTIAXIS_ELEVATIONS = np.array([0.349, 0.349, 0.300, -0.200,
                                 0.349, 0.349, 0.100, 0.349])

# Per-view z shift for the helical cone geometry.  The shifts are all
# non-negative, so they are not symmetric about zero.
HELICAL_Z_SHIFTS = np.array([0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2])

# Per-view object translation for the translation geometry, (num_views, 3) as
# (t_x, t_y, t_z).  Every component is nonzero in every view.
TRANSLATION_VECTORS = np.stack([
    np.array([-4.0, -2.0, 0.7, 2.0, 4.0, -3.0, 1.0, 3.0]),      # t_x
    np.array([-2.5, -1.5, 0.5, 1.5, 2.5, -0.5, 1.0, 2.0]),      # t_y
    np.array([-1.2, -0.6, 0.3, 0.9, 1.5, -0.9, 0.6, 1.2]),      # t_z
], axis=1)

# The six geometries.  Each entry carries the model class and its constructor
# arguments, the detector and voxel parameters set after construction, and the
# reconstruction shape.  The reconstruction shape is set by hand rather than
# taken from auto_set_recon_geometry so that the volume is comfortably smaller
# than the detector's field of view and footprints stay on the detector.
# Offsets are non-integer and of both signs across the six entries, the two
# detector pitches differ wherever the model allows it, and voxel_row_aspect is
# never one.
CONFIGS = [
    dict(
        name='parallel',
        kind='parallel',
        sinogram_shape=(8, 16, 40),
        recon_shape=(10, 12, 16),          # slices must equal detector rows
        angles=VIEW_ANGLES,
        params=dict(delta_det_channel=1.1, delta_det_row=0.9,
                    det_channel_offset=1.35, det_row_offset=-0.85,
                    delta_voxel=1.0, voxel_row_aspect=1.25,
                    voxel_slice_aspect=1.0),
    ),
    dict(
        name='cone flat',
        kind='cone',
        sinogram_shape=(8, 24, 48),
        recon_shape=(10, 12, 8),
        angles=VIEW_ANGLES,
        source_detector_dist=200.0,
        source_iso_dist=100.0,
        use_curved_detector=False,
        z_shifts=np.zeros(8),
        params=dict(delta_det_channel=1.1, delta_det_row=1.3,
                    det_channel_offset=-1.70, det_row_offset=0.65,
                    delta_voxel=1.0, voxel_row_aspect=1.2,
                    voxel_slice_aspect=0.8, recon_slice_offset=1.4),
    ),
    dict(
        # A short source distance and a wide detector give this entry a large
        # fan angle, which is what separates the two candidate row rules for a
        # curved detector (see predict_cone).
        name='cone curved',
        kind='cone',
        sinogram_shape=(8, 48, 64),
        recon_shape=(10, 12, 8),
        angles=VIEW_ANGLES,
        source_detector_dist=100.0,
        source_iso_dist=26.0,
        use_curved_detector=True,
        z_shifts=np.zeros(8),
        params=dict(delta_det_channel=2.0, delta_det_row=1.3,
                    det_channel_offset=2.30, det_row_offset=-1.15,
                    delta_voxel=1.0, voxel_row_aspect=1.2,
                    voxel_slice_aspect=1.0, recon_slice_offset=1.4),
    ),
    dict(
        name='cone helical',
        kind='cone',
        sinogram_shape=(8, 28, 48),
        recon_shape=(10, 12, 8),
        angles=VIEW_ANGLES,
        source_detector_dist=200.0,
        source_iso_dist=100.0,
        use_curved_detector=False,
        z_shifts=HELICAL_Z_SHIFTS,
        params=dict(delta_det_channel=1.1, delta_det_row=1.3,
                    det_channel_offset=-0.95, det_row_offset=1.40,
                    delta_voxel=1.0, voxel_row_aspect=1.2,
                    voxel_slice_aspect=0.8, recon_slice_offset=1.4),
    ),
    dict(
        name='multiaxis',
        kind='multiaxis',
        sinogram_shape=(8, 28, 48),
        recon_shape=(10, 12, 8),
        angles=VIEW_ANGLES,
        elevations=MULTIAXIS_ELEVATIONS,
        params=dict(delta_det_channel=1.1, delta_det_row=0.95,
                    det_channel_offset=1.85, det_row_offset=-1.35,
                    delta_voxel=1.0, voxel_row_aspect=1.3,
                    voxel_slice_aspect=1.15, recon_slice_offset=-0.9),
    ),
    dict(
        name='translation',
        kind='translation',
        sinogram_shape=(8, 24, 48),
        recon_shape=(10, 12, 8),
        translation_vectors=TRANSLATION_VECTORS,
        source_detector_dist=200.0,
        source_iso_dist=100.0,
        params=dict(delta_det_channel=1.1, delta_det_row=1.25,
                    det_channel_offset=-2.15, det_row_offset=1.45,
                    delta_voxel=1.0, voxel_row_aspect=1.15,
                    voxel_slice_aspect=0.9),
    ),
]


# ── model construction and parameter readout ─────────────────────────────────

def build_model(cfg):
    """Construct the model for one configuration and set its parameters.

    Compilation is off so that the run stays fast on a CPU and the geometry
    code runs in plain torch.  The devices are pinned to the CPU so that the
    footprint measurement gives the same answer on every machine.
    """
    kind = cfg['kind']
    shape = cfg['sinogram_shape']
    if kind == 'parallel':
        model = mbirtorch.ParallelBeamModel(shape, cfg['angles'],
                                            compile_mode='off')
    elif kind == 'cone':
        model = mbirtorch.ConeBeamModel(
            shape, cfg['angles'],
            source_detector_dist=cfg['source_detector_dist'],
            source_iso_dist=cfg['source_iso_dist'],
            helical_z_shifts=cfg['z_shifts'],
            use_curved_detector=cfg['use_curved_detector'],
            compile_mode='off')
    elif kind == 'multiaxis':
        angle_pairs = np.stack([cfg['angles'], cfg['elevations']], axis=1)
        model = mbirtorch.MultiAxisParallelModel(shape, angle_pairs,
                                                 compile_mode='off')
    elif kind == 'translation':
        model = mbirtorch.TranslationModel(
            shape, cfg['translation_vectors'],
            source_detector_dist=cfg['source_detector_dist'],
            source_iso_dist=cfg['source_iso_dist'],
            compile_mode='off')
    else:
        raise ValueError(f'unknown geometry kind {kind}')
    model.configure_devices(devices=['cpu'])
    # Setting recon_shape explicitly keeps the volume smaller than the
    # detector's field of view; no_warning suppresses the notice that the
    # reconstruction parameters were changed by hand.
    model.set_params(no_warning=True, recon_shape=cfg['recon_shape'],
                     **cfg['params'])
    model.verify_valid_params()
    return model


def build_timing_model():
    """The 1800-view helical cone-beam model the slider timing test uses.

    Compilation is off and nothing is ever projected: the viewer reads
    parameters and projects points, and the test measures a slider step.
    """
    num_views = 1800
    detector_shape = (40, 64)
    recon_shape = (24, 24, 16)
    num_turns = 2.0
    helical_travel = 30.0
    angles = np.linspace(0.0, num_turns * 2.0 * np.pi, num_views,
                         endpoint=False)
    z_shifts = np.linspace(0.0, helical_travel, num_views)
    model = mbirtorch.ConeBeamModel(
        (num_views,) + detector_shape, angles,
        source_detector_dist=400.0, source_iso_dist=200.0,
        helical_z_shifts=z_shifts, compile_mode='off')
    model.set_params(no_warning=True, recon_shape=recon_shape)
    model.verify_valid_params()
    return model


def geometry_scalars(cfg):
    """The derived lengths the prediction needs, gathered in one dictionary."""
    p = cfg['params']
    delta_voxel = p['delta_voxel']
    scalars = dict(
        num_det_rows=cfg['sinogram_shape'][1],
        num_det_channels=cfg['sinogram_shape'][2],
        num_rows=cfg['recon_shape'][0],
        num_cols=cfg['recon_shape'][1],
        num_slices=cfg['recon_shape'][2],
        delta_voxel=delta_voxel,
        delta_voxel_row=delta_voxel * p['voxel_row_aspect'],
        delta_voxel_slice=delta_voxel * p['voxel_slice_aspect'],
        delta_det_channel=p['delta_det_channel'],
        delta_det_row=p['delta_det_row'],
        det_channel_offset=p['det_channel_offset'],
        det_row_offset=p['det_row_offset'],
        recon_slice_offset=p.get('recon_slice_offset', 0.0),
    )
    for key in ('source_detector_dist', 'source_iso_dist'):
        if key in cfg:
            scalars[key] = cfg[key]
    return scalars


# ── the geometric statement, written independently of the projector ──────────

def voxel_center_xyz(i, j, k, g):
    """Object coordinates of the center of voxel (i, j, k).

    The volume is centered on the origin in x and y.  In z it is centered on
    recon_slice_offset.  The three pitches are the voxel pitch times the row
    and slice aspect ratios.
    """
    y = g['delta_voxel_row'] * (i - (g['num_rows'] - 1) / 2.0)
    x = g['delta_voxel'] * (j - (g['num_cols'] - 1) / 2.0)
    z = (g['delta_voxel_slice'] * (k - (g['num_slices'] - 1) / 2.0)
         + g['recon_slice_offset'])
    return np.array([x, y, z], dtype=np.float64)


def rotate_about_z(point, angle):
    """Rotate a point about the z axis by ``angle``.

    The sense is the standard right-handed one: for a positive angle the point
    turns from the +x axis toward the +y axis, which is counterclockwise when
    seen from a viewpoint on the +z axis looking toward the origin.
    """
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = point
    return np.array([c * x - s * y, s * x + c * y, z])


def ray_plane_intersection(origin, direction, plane_point, plane_normal):
    """Where the ray from ``origin`` along ``direction`` meets a plane."""
    denominator = np.dot(direction, plane_normal)
    step = np.dot(plane_point - origin, plane_normal) / denominator
    return origin + step * direction


def detector_indices(u, v, g, flips):
    """Detector (row, channel) indices of a point at detector coordinates (u, v).

    The grid is centered on the detector: index (num - 1) / 2 sits at the
    detector's middle.  A positive det_channel_offset moves the image toward
    higher channel index, which is the same as moving the grid's center to
    u = -det_channel_offset.  det_row_offset works the same way in v.  The
    ``flips`` entries let the caller negate either offset, which is how the
    two signs are confirmed by measurement.
    """
    channel_offset = flips.get('channel_offset', 1.0) * g['det_channel_offset']
    row_offset = flips.get('row_offset', 1.0) * g['det_row_offset']
    channel = ((u + channel_offset) / g['delta_det_channel']
               + (g['num_det_channels'] - 1) / 2.0)
    row = ((v + row_offset) / g['delta_det_row']
           + (g['num_det_rows'] - 1) / 2.0)
    return row, channel


def predict_parallel(i, j, k, view, cfg, g, flips):
    """Predicted (row, channel) for the parallel geometry.

    Statement.  The object turns about z by the view angle.  The rays are
    parallel to the -y direction, so the source is infinitely far away on the
    +y side and the detector plane is the xz plane.  The channel coordinate of
    a point is its x coordinate.  The row grid is not the detector row grid.  A
    detector row index equals the reconstruction slice index, so the row
    coordinate is z measured in units of the voxel slice pitch about the
    volume's z center.  delta_det_row and det_row_offset take no part.
    """
    angle = flips.get('rotation_sense', 1.0) * cfg['angles'][view]
    point = rotate_about_z(voxel_center_xyz(i, j, k, g), angle)
    ray_direction = np.array([0.0, -1.0, 0.0])
    detector_point = ray_plane_intersection(point, ray_direction,
                                            np.zeros(3), ray_direction)
    u = detector_point[0]
    # The row grid follows the volume's z center, so measure the height from
    # that center.  The parallel model has no recon_slice_offset and the term
    # is zero; it is written out so the rule stays readable.
    v = detector_point[2] - g['recon_slice_offset']
    channel = ((u + flips.get('channel_offset', 1.0) * g['det_channel_offset'])
               / g['delta_det_channel'] + (g['num_det_channels'] - 1) / 2.0)
    row = v / g['delta_voxel_slice'] + (g['num_det_rows'] - 1) / 2.0
    return row, channel


def predict_cone(i, j, k, view, cfg, g, flips):
    """Predicted (row, channel) for the cone geometries, flat and curved.

    Statement.  The object turns about z by the view angle, and a helical scan
    also shifts the object by -z_shift along z.  The source is a point at
    (0, source_iso_dist, 0).  A flat detector is the plane
    y = source_iso_dist - source_detector_dist, with its u axis along +x and
    its v axis along +z.  The detector coordinates of a voxel are where the ray
    from the source through the voxel center meets that plane.

    A curved detector is a cylinder of radius source_detector_dist whose axis
    passes through the source parallel to z.  Its channel coordinate is arc
    length along the cylinder, measured from the central ray and positive
    toward +x.  Its row coordinate is the one measured empirically: the
    projector uses the flat tangent-plane height, not the height at which the
    ray crosses the cylinder.  Passing flips['curved_row'] = 'cylinder'
    selects the cylinder-crossing height instead, which is how the choice was
    settled.
    """
    angle = flips.get('rotation_sense', 1.0) * cfg['angles'][view]
    point = rotate_about_z(voxel_center_xyz(i, j, k, g), angle)
    # The stated sign is -1: the object moves toward -z as the shift grows.
    # The audit variant passes +1 to show that the measurement rejects it.
    z_shift_sign = flips.get('z_shift', -1.0)
    point = point + np.array([0.0, 0.0,
                              z_shift_sign * cfg['z_shifts'][view]])
    sdd = g['source_detector_dist']
    sid = g['source_iso_dist']
    source = np.array([0.0, sid, 0.0])

    if not cfg['use_curved_detector']:
        plane_point = np.array([0.0, sid - sdd, 0.0])
        plane_normal = np.array([0.0, 1.0, 0.0])
        hit = ray_plane_intersection(source, point - source,
                                     plane_point, plane_normal)
        u = hit[0]
        v = hit[2]
    else:
        # Horizontal components of the vector from the source to the voxel.
        # The central ray points along -y, so the signed angle away from it,
        # positive toward +x, is atan2(dx, -dy).
        dx = point[0] - source[0]
        dy = point[1] - source[1]
        u = sdd * np.arctan2(dx, -dy)
        if flips.get('curved_row', 'plane') == 'cylinder':
            # Height where the ray crosses the cylinder of radius sdd.
            v = sdd * point[2] / np.hypot(dx, dy)
        else:
            # Height where the ray crosses the plane tangent to the cylinder
            # at the central ray, which is the flat-detector height.
            v = sdd * point[2] / (-dy)
    return detector_indices(u, v, g, flips)


def predict_multiaxis(i, j, k, view, cfg, g, flips):
    """Predicted (row, channel) for the multiaxis parallel geometry.

    Statement.  The object turns about z by the azimuth.  The rays are
    parallel and their direction is tilted out of the xy plane by the
    elevation: the assumed direction of travel is
    (0, -cos(elevation), +sin(elevation)).  The detector plane is perpendicular
    to that direction.  Its u axis is +x, so the channel coordinate is
    unaffected by the tilt.  Its v axis is (0, sin(elevation), cos(elevation)),
    which is the +z axis turned by the elevation, so the row coordinate picks
    up the in-plane depth y as well as z.  Passing flips['elevation'] = -1
    negates the elevation, which is how its sign was confirmed.

    Note on what the measurement can and cannot settle.  The projection is
    parallel, so it fixes the ray line but not which end of it holds the
    source.  The direction of travel above is the choice that matches the
    other three models, whose source sits on the +y side.  It follows that a
    positive elevation puts the source below the xy plane.  That last statement
    is a convention, not a measured fact.
    """
    azimuth = flips.get('rotation_sense', 1.0) * cfg['angles'][view]
    elevation = flips.get('elevation', 1.0) * cfg['elevations'][view]
    point = rotate_about_z(voxel_center_xyz(i, j, k, g), azimuth)
    ray_direction = np.array([0.0, -np.cos(elevation), np.sin(elevation)])
    axis_u = np.array([1.0, 0.0, 0.0])
    axis_v = np.array([0.0, np.sin(elevation), np.cos(elevation)])
    # The detector plane may be placed anywhere along the ray direction; the
    # coordinates in its own axes do not depend on that placement.  Put it
    # through the origin.
    hit = ray_plane_intersection(point, ray_direction,
                                 np.zeros(3), ray_direction)
    return detector_indices(np.dot(hit, axis_u), np.dot(hit, axis_v), g, flips)


def predict_translation(i, j, k, view, cfg, g, flips):
    """Predicted (row, channel) for the translation geometry.

    Statement.  The source and the flat detector sit exactly as in the cone
    geometry, and the object never turns.  Each view moves the object by minus
    the view's translation vector: a positive t_x moves the object toward -x, a
    positive t_y moves it toward -y and so away from the source, and a positive
    t_z moves it toward -z.  Passing flips['translation'] = +1 moves the object
    by plus the vector instead, which is how the three signs were confirmed.
    There is no recon_slice_offset in this model, so the volume is centered on
    z = 0.
    """
    # The stated sign is -1: the object moves by minus the translation vector.
    # The audit variant passes +1 to show that the measurement rejects it.
    sign = flips.get('translation', -1.0)
    point = voxel_center_xyz(i, j, k, g) + sign * cfg['translation_vectors'][view]
    sdd = g['source_detector_dist']
    sid = g['source_iso_dist']
    source = np.array([0.0, sid, 0.0])
    hit = ray_plane_intersection(source, point - source,
                                 np.array([0.0, sid - sdd, 0.0]),
                                 np.array([0.0, 1.0, 0.0]))
    return detector_indices(hit[0], hit[2], g, flips)


PREDICTORS = dict(parallel=predict_parallel, cone=predict_cone,
                  multiaxis=predict_multiaxis, translation=predict_translation)


# ── measurement ──────────────────────────────────────────────────────────────

def probe_voxels(recon_shape):
    """The voxels to project.

    The list holds the eight corners of the volume, the center voxel, and three
    off-center voxels.  Each off-center voxel has, in all three directions, an
    index that is neither at an end nor at the center.  Those three make every
    sign in the index-to-coordinate rule visible, because a corner or a center
    voxel hides a sign error that swaps two symmetric positions.
    """
    num_rows, num_cols, num_slices = recon_shape
    voxels = [(i, j, k) for i in (0, num_rows - 1)
              for j in (0, num_cols - 1)
              for k in (0, num_slices - 1)]
    voxels.append((num_rows // 2, num_cols // 2, num_slices // 2))
    voxels.append((2, num_cols - 3, num_slices // 2 + 1))
    voxels.append((num_rows - 3, 2, 1))
    voxels.append((num_rows // 2 + 1, 3, num_slices - 2))
    return voxels


def measure_footprint_centroids(model, cfg, voxel):
    """Project one voxel and return the per-view footprint centroids.

    The voxel is passed to sparse_forward_project as a one-pixel cylinder whose
    values are zero except in the voxel's own slice.  For each view the routine
    returns the intensity-weighted centroid in (row, channel), or a reason for
    rejecting that view.
    """
    i, j, k = voxel
    _, num_cols, num_slices = cfg['recon_shape']
    num_views, num_det_rows, num_det_channels = cfg['sinogram_shape']
    pixel_index = torch.tensor([i * num_cols + j], dtype=torch.int64)
    values = torch.zeros((1, num_slices), dtype=torch.float32)
    values[0, k] = 1.0
    sinogram = model.sparse_forward_project(values, pixel_index).cpu().numpy()

    row_grid = np.arange(num_det_rows)
    channel_grid = np.arange(num_det_channels)
    # A footprint may reach the edge of the detector in the row direction only
    # in the geometries that spread a voxel over rows.  Parallel beam sends
    # slice k to row k with no spreading, so its row 0 and last row are lit by
    # the end slices without any cut.
    row_axis_can_clip = cfg['kind'] != 'parallel'

    results = []
    for view in range(num_views):
        footprint = sinogram[view]
        mass = footprint.sum()
        if mass < MASS_FLOOR:
            results.append(('off detector', None, None))
            continue
        touches_channel_edge = (footprint[:, 0].sum() > 0
                                or footprint[:, -1].sum() > 0)
        touches_row_edge = (footprint[0, :].sum() > 0
                            or footprint[-1, :].sum() > 0)
        if touches_channel_edge or (row_axis_can_clip and touches_row_edge):
            results.append(('clipped at detector edge', None, None))
            continue
        row_centroid = float((footprint.sum(axis=1) * row_grid).sum() / mass)
        channel_centroid = float((footprint.sum(axis=0) * channel_grid).sum()
                                 / mass)
        results.append((None, row_centroid, channel_centroid))
    return results
