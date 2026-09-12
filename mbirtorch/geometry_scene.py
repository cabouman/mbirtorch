"""Drawable description of an mbirtorch scan geometry, in plain numpy.

What this module is for.  A tomography model holds numbers: distances, pitches,
offsets, and per-view angles or vectors.  A user cannot see a geometry error in
those numbers, but can see it in a picture.  This module turns the numbers into
the points, lines, and polylines a picture needs.  It imports numpy only, and
no matplotlib, so every geometric statement it makes can be tested without a
display.

The two layers.  ``GeometryScene`` is the model layer: it holds the parameters,
maps object points to detector indices, and returns the primitives for one
view.  A separate matplotlib module draws those primitives and owns the widgets.
This split follows the slice viewer of ``mbirtorch/viewer.py``, whose pure-numpy
``VolumeStack`` carries all of the data logic.  The scene reads the model's
parameters through ``get_params`` and projects points through
:meth:`TomographyModel.project_points`, so it holds no projection formula of
its own.

Every geometric statement a drawing makes is made here.  The drawing layer
places primitives on axes and computes no positions of its own, so this module
also carries the two drawings that are annotations rather than geometry: the
rays of a geometry whose source is infinitely far away, and the arc that shows
which way the source travels.

The object frame.  All four geometries share one right-handed (x, y, z) frame
whose z axis is the rotation axis.  Voxel (i, j, k) of the reconstruction sits
at

    x = delta_voxel * (j - (num_cols - 1) / 2)
    y = delta_voxel_row * (i - (num_rows - 1) / 2)
    z = delta_voxel_slice * (k - (num_slices - 1) / 2) + recon_slice_offset

with ``delta_voxel_row = voxel_row_aspect * delta_voxel`` and
``delta_voxel_slice = voxel_slice_aspect * delta_voxel``.  The volume is
centered on the origin in x and y.

Two pictures of the same geometry.  The projector moves the object and holds the
source and the detector fixed.  A scanner user thinks of the object as fixed and
the gantry as moving.  This module computes detector indices in the projector's
picture, because that is what the projector does, and returns drawable
primitives in the user's picture, because that is what a drawing needs.  The
conversion is the inverse of the view's action on the object.

One more view is drawable beside the scan's own views.  ``reference_view``
returns the primitives of the position the view action's identity gives, which
is the source and the detector at angle 0.  A drawing uses it as a fixed
reference for the view it shows.

Two positions in a drawing are choices and not geometry.  A parallel projection
has no source position, and a cone geometry with an infinite source-detector
distance has neither a source nor a detector position.  Both cases place the
source and the detector at a distance derived from the volume's size; see
``drawing_distance``.

The conventions this module implements are stated in
:meth:`TomographyModel.project_points`, and were confirmed against the
projector by the tests.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = ['GeometryScene', 'ViewScene', 'GEOMETRY_KINDS',
           'required_parameter_names', 'CURVED_ARC_SAMPLES', 'RIM_SAMPLES',
           'DEFAULT_DRAWING_DISTANCE_FACTOR', 'MULTIAXIS_SOURCE_ON_PLUS_Y',
           'ROTATION_ARC_SWEEP', 'ROTATION_ARC_SAMPLES',
           'CLOCKWISE_FROM_PLUS_Z', 'COUNTERCLOCKWISE_FROM_PLUS_Z',
           'DIFFERENCE_EXCLUDED_QUANTITIES', 'values_are_equal']


#: The geometry kinds this module knows.  Each names one mbirtorch model class.
GEOMETRY_KINDS = ('parallel', 'cone', 'multiaxis', 'translation')

#: Number of points sampled along one edge of a curved detector's outline.  The
#: outline is a closed polyline through the bottom edge and back along the top
#: edge, so it holds 2 * CURVED_ARC_SAMPLES + 1 points.
CURVED_ARC_SAMPLES = 33

#: Number of points sampled around one rim of the region-of-reconstruction
#: cylinder; see :meth:`GeometryScene.fit_points`.  The rim is a closed
#: polyline whose last point repeats its first, so it holds 90 distinct points,
#: one every four degrees.  A finite sample can miss the true widest point of a
#: projected rim, and at this spacing that miss is far below half a detector
#: pixel, the tolerance the projection tests use.
RIM_SAMPLES = 91

#: Default multiple of the volume's largest half-extent used as the drawing
#: distance for a geometry whose source or detector has no physical position.
#: This is a drawing choice, not a property of any model.  At 1.5 the drawn
#: source of a parallel-type geometry sat close enough to the volume box to
#: touch it in the 3D panel, so the factor is 2.5.
DEFAULT_DRAWING_DISTANCE_FACTOR = 2.5

#: Default multiple of the volume's z half-extent used for the length of the
#: drawn rotation axis, so that the axis sticks out past the volume box.
DEFAULT_ROTATION_AXIS_EXTENSION = 1.25

#: The angle the rotation-direction arc sweeps, in radians, and the number of
#: points it is drawn with.  The arc starts at the drawn source and follows the
#: circle the source travels on, so it needs no radius of its own.  Both are
#: drawing choices and neither affects a detector index.
ROTATION_ARC_SWEEP = 0.5
ROTATION_ARC_SAMPLES = 17

#: The two values ``ViewScene.source_travel_sense`` can take.  The sense is
#: named as it is seen from a point on the +z axis looking toward the origin,
#: which is the view the top panel of a drawing shows.
CLOCKWISE_FROM_PLUS_Z = 'clockwise seen from +z'
COUNTERCLOCKWISE_FROM_PLUS_Z = 'counterclockwise seen from +z'

#: Entries of ``derived_quantities`` that :meth:`GeometryScene.differences`
#: does not report.  Each is a sentence that explains the numbers beside it, so
#: a comparison listing it would print a paragraph twice and would say nothing
#: the other entries do not already say.
DIFFERENCE_EXCLUDED_QUANTITIES = ('angle_note', 'drawing_note')

#: The multiaxis source-side convention.  A parallel projection is the same in
#: both directions along a ray, so measurement cannot say which end of a ray
#: holds the source.  True puts the source on the +y side, as the cone and
#: translation geometries do.  A positive elevation then puts the source below
#: the xy plane and the detector center above it.  Setting this to False
#: reverses the direction of travel and swaps the two, and changes nothing about
#: the detector indices a point receives.
MULTIAXIS_SOURCE_ON_PLUS_Y = True


# Parameter names read from a model, by geometry kind.  ``from_model`` reads
# exactly these names through ``model.get_params``, so the scene depends on the
# parameter interface and not on any model class.
_COMMON_PARAMETER_NAMES = (
    'sinogram_shape', 'recon_shape', 'delta_voxel', 'voxel_row_aspect',
    'voxel_slice_aspect', 'delta_det_channel', 'delta_det_row',
    'det_channel_offset', 'det_row_offset', 'use_ror_mask',
)

# The parallel and translation models have no recon_slice_offset parameter and
# raise on that name, so it is requested only where it exists.
_KIND_PARAMETER_NAMES = {
    'parallel': ('angles',),
    'cone': ('view_params_array', 'source_detector_dist', 'source_iso_dist',
             'use_curved_detector', 'recon_slice_offset'),
    'multiaxis': ('angles', 'recon_slice_offset'),
    'translation': ('translation_vectors', 'source_detector_dist',
                    'source_iso_dist'),
}


def required_parameter_names(kind):
    """The parameter names a scene of this geometry kind needs.

    Args:
        kind (str): one of ``GEOMETRY_KINDS``.

    Returns:
        tuple of str: the names, in a fixed order.
    """
    if kind not in GEOMETRY_KINDS:
        raise ValueError(f'Unknown geometry kind {kind!r}; '
                         f'expected one of {GEOMETRY_KINDS}.')
    return _COMMON_PARAMETER_NAMES + _KIND_PARAMETER_NAMES[kind]


def _rotation_about_z(angle):
    """The 3x3 matrix that rotates a point about the z axis by ``angle``.

    The sense is the right-handed one: a positive angle carries the +x axis
    toward the +y axis, which is counterclockwise seen from a point on the +z
    axis looking toward the origin.
    """
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0],
                     [sine, cosine, 0.0],
                     [0.0, 0.0, 1.0]])


def values_are_equal(first, second, tolerance=1e-12):
    """Whether two parameter or derived values are the same value.

    A parameter can be a float, an integer, a bool, a string, a shape tuple, or
    a per-view array, and one of the two can be None, which means the parameter
    does not exist for that geometry kind.  Floats are compared with a relative
    tolerance rather than exactly, because two computed floats that mean the
    same value can differ by rounding.  An infinite ``source_detector_dist``
    equals another infinity of the same sign.

    Args:
        first, second: the two values.
        tolerance (float, optional): the relative tolerance for numbers.

    Returns:
        bool: whether the two are the same value.
    """
    if first is None or second is None:
        return first is None and second is None
    if isinstance(first, str) or isinstance(second, str):
        return str(first) == str(second)
    try:
        left = np.asarray(first, dtype=np.float64)
        right = np.asarray(second, dtype=np.float64)
    except (TypeError, ValueError):
        return bool(first == second)
    if left.shape != right.shape:
        return False
    if left.size == 0:
        return True
    return bool(np.allclose(left, right, rtol=tolerance, atol=0.0,
                            equal_nan=True))


def _merged_intervals(intervals):
    """Overlapping (low, high) pairs joined into the fewest pairs that cover
    the same set.

    The pairs are sorted and then walked once.  A pair that starts at or before
    the end of the pair being built extends it; a pair that starts after it
    begins a new one, which is the gap that makes the union smaller than the
    hull.

    Args:
        intervals (sequence): the (low, high) pairs, in any order.

    Returns:
        list of [low, high]: the merged pairs, in increasing order.
    """
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def _unit(vector):
    """``vector`` scaled to unit length."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError('Cannot normalize a zero vector.')
    return vector / norm


def _model_with_parameters(model, overrides):
    """A new model of the same class with some parameters replaced.

    The copy is for drawing only: it is never used to project data, so
    compilation is off.  It is built directly from the model's own
    parameters rather than with ``build_model`` or ``copy_ct_model``, so
    that the automatic reconstruction geometry does not follow an override.
    The constructor runs that automatic pass once, and every value it sets
    is then replaced through ``set_params`` by the model's own values, with
    the overrides applied last.  The copy therefore holds exactly the
    model's values and the overrides, even when an override is a detector
    parameter that the automatic pass would follow.  Warnings from the
    constructor are silenced: the original model already issued them.
    """
    required, optional, regularization = model.get_all_params()
    constructor_args = {name: value for name, value in required.items()
                        if name != 'geometry_type'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        copy = type(model)(**constructor_args, compile_mode='off')
    copy.set_params(no_compile=True, no_warning=True, **optional)
    copy.set_params(no_compile=True, no_warning=True, **regularization)
    copy.set_params(no_compile=True, no_warning=True, **overrides)
    return copy


@dataclass
class ViewScene:
    """The drawable primitives of one view, in the fixed-object picture.

    Every array is float64 and every position is in object-frame ALU.  The
    object stays where ``GeometryScene.voxel_centers`` puts it, and the source
    and the detector carry the view's motion.

    Attributes:
        view_index (int): the view these primitives belong to.
        source (ndarray or None): the source position, (3,).  None for a
            geometry whose source is infinitely far away, which is the parallel
            and multiaxis geometries and a cone geometry with an infinite
            source-detector distance.
        source_draw (ndarray): a source position that is always finite, (3,).
            Equal to ``source`` when that exists.  Otherwise a point one
            drawing distance back along the ray direction, which is a drawing
            choice.
        ray_direction (ndarray): unit vector, (3,), pointing from the source
            toward the detector along the central ray.
        detector_center (ndarray): the physical center of the detector grid,
            (3,).  This is the point at detector coordinates
            (u, v) = (-det_channel_offset, -det_row_offset), which is not on
            the central ray when either offset is nonzero.
        detector_origin (ndarray): the point where the central ray meets the
            detector, (3,), which is detector coordinates (0, 0).
        detector_u_axis (ndarray): unit vector, (3,), along which the channel
            index increases.  For a curved detector this is the tangent to the
            cylinder at the central ray.
        detector_v_axis (ndarray): unit vector, (3,), along which the row index
            increases.
        detector_normal (ndarray): unit vector, (3,), equal to
            ``ray_direction`` for a flat detector.
        detector_outline (ndarray): closed polyline through the detector's
            physical corners, (M, 3).  Five points for a flat panel.  For a
            curved detector, the bottom edge sampled along the arc, the top
            edge sampled back, and a repeat of the first point.
        detector_corners (ndarray): the four physical corners, (4, 3), in the
            order (low channel, low row), (high channel, low row), (high
            channel, high row), (low channel, high row).
        detector_pixel0 (ndarray): the physical center of detector pixel
            (row 0, channel 0), (3,).
        volume_corners (ndarray): the eight outer corners of the volume box,
            (8, 3).  These sit on the outer faces of the corner voxels, half a
            voxel beyond the corner voxel centers.
        voxel0_center (ndarray): the center of voxel (0, 0, 0), (3,).
        rotation_axis (ndarray or None): a segment along the rotation axis,
            (2, 3), spanning the volume's z extent and a little past it.  None
            for the translation geometry, which does not rotate.
        translation_path (ndarray or None): the path of the projector frame's
            origin through the object, (V, 3), for the translation geometry.
            This is the line the central ray sweeps through the object, and it
            takes the rotation axis's place in a drawing.  None otherwise.
        corner_rays (ndarray): four segments ending at the four detector
            corners, (4, 2, 3), in the same corner order as
            ``detector_corners``.  A geometry with a finite source gets four
            segments that start at that source.  A parallel-type geometry gets
            four segments parallel to ``ray_direction``, each starting in the
            plane through ``source_draw``, so a parallel-type corner ray is as
            long as the drawn central ray.  They stay parallel rather than
            converging on ``source_draw``, because converging rays would draw
            a cone beam.
        volume_outline_on_detector (ndarray): the detector indices of the eight
            volume corners, (8, 2), as (row, channel).
        ror_outline_on_detector (ndarray or None): the detector indices of the
            region of reconstruction's two rims, (2, RIM_SAMPLES, 2), as
            (row, channel).  Entry 0 is the rim at ``z_min`` and entry 1 the
            rim at ``z_max``, each a closed polyline whose last point repeats
            its first.  The two rims bound the whole cylinder's projection, for
            the reason :meth:`GeometryScene.fit_points` gives.  None when there
            is no cylinder, which is the case ``ror_cylinder`` reports as None.
        ror_cylinder (dict or None): the region of reconstruction, when
            ``use_ror_mask`` is True.  Keys: ``center`` (3,), ``semi_axis_x``,
            ``semi_axis_y``, ``radius``, ``z_min``, ``z_max``.  None when the
            mask is off or is a custom array.
        rotation_direction_arc (ndarray or None): a short arc on the circle the
            source travels, (M, 3), starting at ``source_draw`` and sweeping
            ``ROTATION_ARC_SWEEP`` radians in the direction the source moves
            from this view to the next one.  The last two points give the
            direction an arrowhead at the end of the arc should point.  None
            for the translation geometry, which does not rotate, and None when
            the direction cannot be told: a single view, two neighboring views
            at one angle, or a source on the rotation axis.
        source_travel_sense (str or None): the same direction in words, either
            ``CLOCKWISE_FROM_PLUS_Z`` or ``COUNTERCLOCKWISE_FROM_PLUS_Z``.
            None exactly when ``rotation_direction_arc`` is None.
    """

    view_index: int
    source: object
    source_draw: np.ndarray
    ray_direction: np.ndarray
    detector_center: np.ndarray
    detector_origin: np.ndarray
    detector_u_axis: np.ndarray
    detector_v_axis: np.ndarray
    detector_normal: np.ndarray
    detector_outline: np.ndarray
    detector_corners: np.ndarray
    detector_pixel0: np.ndarray
    volume_corners: np.ndarray
    voxel0_center: np.ndarray
    rotation_axis: object
    translation_path: object
    corner_rays: np.ndarray
    volume_outline_on_detector: np.ndarray
    ror_outline_on_detector: object = field(default=None)
    ror_cylinder: object = field(default=None)
    rotation_direction_arc: object = field(default=None)
    source_travel_sense: object = field(default=None)


class GeometryScene:
    """The drawable geometry of one tomography model.

    A scene is built from a model.  It reads the model's parameters by name
    through ``get_params``, and it projects points through the model's
    ``project_points``; nothing else about the model is touched.
    :func:`required_parameter_names` lists the names each geometry kind needs.

    Args:
        model: a ``TomographyModel`` subclass instance.
        drawing_distance_factor (float, optional): multiple of the volume's
            largest half-extent used as the drawing distance where a source or
            a detector has no physical position.  Defaults to
            ``DEFAULT_DRAWING_DISTANCE_FACTOR``.
        multiaxis_source_on_plus_y (bool, optional): the multiaxis source-side
            convention; see ``MULTIAXIS_SOURCE_ON_PLUS_Y``.
        rotation_axis_extension (float, optional): multiple of the volume's z
            half-extent used for the drawn rotation axis's half length.
        rotation_arc_sweep (float, optional): the angle in radians that the
            rotation-direction arc sweeps; see ``ROTATION_ARC_SWEEP``.

    Attributes:
        kind (str): the geometry kind.
        num_views (int): the number of views.
        num_det_rows, num_det_channels (int): the detector's shape.
        num_rows, num_cols, num_slices (int): the volume's shape.
        magnification (float): the scale from a point at the origin to its
            image on the detector.  One for a parallel projection.
        is_parallel_type (bool): True when the source is infinitely far away,
            so that the projection has no finite source position.
        drawing_distance (float): the distance used where a position is a
            drawing choice.
        row_pitch (float): the detector row pitch the projector actually uses.
            This is ``delta_det_row`` for every geometry except the parallel
            one, which sends recon slice m to detector row m and therefore uses
            ``delta_voxel``.
        row_offset (float): the detector row offset the projector actually
            uses, which is zero for the parallel geometry and
            ``det_row_offset`` otherwise.
    """

    def __init__(self, model,
                 drawing_distance_factor=DEFAULT_DRAWING_DISTANCE_FACTOR,
                 multiaxis_source_on_plus_y=MULTIAXIS_SOURCE_ON_PLUS_Y,
                 rotation_axis_extension=DEFAULT_ROTATION_AXIS_EXTENSION,
                 rotation_arc_sweep=ROTATION_ARC_SWEEP):
        self.model = model
        self.kind = self.kind_of_model(model)
        self.params = {name: model.get_params(name)
                       for name in required_parameter_names(self.kind)}
        self.drawing_distance_factor = float(drawing_distance_factor)
        self.multiaxis_source_on_plus_y = bool(multiaxis_source_on_plus_y)
        self.rotation_axis_extension = float(rotation_axis_extension)
        self.rotation_arc_sweep = float(rotation_arc_sweep)

        self._read_shapes()
        self._read_detector()
        self._read_view_parameters()
        self._read_source_and_detector_distances()
        self._set_drawing_distance()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_model(cls, model, **kwargs):
        """Build a scene from an mbirtorch model.

        The parameters are read one name at a time through ``model.get_params``,
        and the geometry kind comes from the model's ``geometry_type``
        parameter, with the class name as a fallback.  Nothing else about the
        model is touched, so a change inside a model class cannot change what
        this scene draws.

        Args:
            model: a ``TomographyModel`` subclass instance.
            **kwargs: passed to the constructor.

        Returns:
            GeometryScene
        """
        return cls(model, **kwargs)

    @staticmethod
    def kind_of_model(model):
        """The geometry kind of a model.

        The four model classes record their identity differently.  The cone,
        parallel, and translation models set ``geometry_type`` to ``'cone'``,
        ``'parallel'``, and ``'translation'``.  The multiaxis model sets it to
        the class-identity string ``str(type(self))``, which contains both
        "multiaxis" and "parallel", so "multiaxis" is tested first.
        """
        try:
            geometry_type = str(model.get_params('geometry_type'))
        except Exception:
            geometry_type = ''
        text = (geometry_type + ' ' + type(model).__name__).lower()
        for keyword, kind in (('multiaxis', 'multiaxis'),
                              ('cone', 'cone'),
                              ('translation', 'translation'),
                              ('parallel', 'parallel')):
            if keyword in text:
                return kind
        raise ValueError('Cannot tell the geometry kind of a model whose '
                         f'geometry_type is {geometry_type!r} and whose class '
                         f'is {type(model).__name__}.')

    def _read_shapes(self):
        """Read the sinogram and reconstruction shapes and the voxel pitches."""
        sinogram_shape = tuple(int(n) for n in self.params['sinogram_shape'])
        recon_shape = tuple(int(n) for n in self.params['recon_shape'])
        if len(sinogram_shape) != 3 or len(recon_shape) != 3:
            raise ValueError('sinogram_shape and recon_shape must each have '
                             'three entries.')
        self.sinogram_shape = sinogram_shape
        self.recon_shape = recon_shape
        self.num_views, self.num_det_rows, self.num_det_channels = sinogram_shape
        self.num_rows, self.num_cols, self.num_slices = recon_shape

        self.delta_voxel = float(self.params['delta_voxel'])
        self.voxel_row_aspect = float(self.params['voxel_row_aspect'])
        self.voxel_slice_aspect = float(self.params['voxel_slice_aspect'])
        self.delta_voxel_row = self.voxel_row_aspect * self.delta_voxel
        self.delta_voxel_slice = self.voxel_slice_aspect * self.delta_voxel
        self.recon_slice_offset = float(self.params.get('recon_slice_offset',
                                                        0.0))
        self.use_ror_mask = self.params['use_ror_mask']

    def _read_detector(self):
        """Read the detector pitches and offsets, and the pitches actually used.

        The parallel geometry is the one exception: the projector sends recon
        slice m to detector row m, so its row pitch is ``delta_voxel`` and its
        row offset is zero.  ``delta_det_row`` and ``det_row_offset`` take no
        part in a parallel projection at all, and a drawing that used them
        would show rows in the wrong places.
        """
        self.delta_det_channel = float(self.params['delta_det_channel'])
        self.delta_det_row = float(self.params['delta_det_row'])
        self.det_channel_offset = float(self.params['det_channel_offset'])
        self.det_row_offset = float(self.params['det_row_offset'])
        if self.kind == 'parallel':
            self.row_pitch = self.delta_voxel
            self.row_offset = 0.0
        else:
            self.row_pitch = self.delta_det_row
            self.row_offset = self.det_row_offset

    def _read_view_parameters(self):
        """Read the per-view angles, z shifts, elevations, or translations."""
        num_views = self.num_views
        self.angles = np.zeros(num_views)
        self.elevations = np.zeros(num_views)
        self.z_shifts = np.zeros(num_views)
        self.translation_vectors = None

        if self.kind == 'parallel':
            angles = np.asarray(self.params['angles'], dtype=np.float64)
            self._check_view_count(angles.shape[0], 'angles')
            self.angles = angles.reshape(-1)
        elif self.kind == 'cone':
            view_params = np.asarray(self.params['view_params_array'],
                                     dtype=np.float64)
            if view_params.ndim != 2 or view_params.shape[1] != 2:
                raise ValueError('view_params_array must have shape '
                                 f'(num_views, 2); got {view_params.shape}.')
            self._check_view_count(view_params.shape[0], 'view_params_array')
            self.angles = view_params[:, 0]
            self.z_shifts = view_params[:, 1]
        elif self.kind == 'multiaxis':
            angles = np.asarray(self.params['angles'], dtype=np.float64)
            if angles.ndim != 2 or angles.shape[1] != 2:
                raise ValueError('multiaxis angles must have shape '
                                 f'(num_views, 2); got {angles.shape}.')
            self._check_view_count(angles.shape[0], 'angles')
            self.angles = angles[:, 0]
            self.elevations = angles[:, 1]
        else:
            vectors = np.asarray(self.params['translation_vectors'],
                                 dtype=np.float64)
            if vectors.ndim != 2 or vectors.shape[1] != 3:
                raise ValueError('translation_vectors must have shape '
                                 f'(num_views, 3); got {vectors.shape}.')
            self._check_view_count(vectors.shape[0], 'translation_vectors')
            self.translation_vectors = vectors

    def _check_view_count(self, count, name):
        if int(count) != self.num_views:
            raise ValueError(f'{name} holds {count} views but sinogram_shape '
                             f'holds {self.num_views}.')

    def _read_source_and_detector_distances(self):
        """Read the two distances and work out the magnification.

        A cone geometry accepts an infinite source-detector distance, and its
        magnification is then one.  Such a geometry is a parallel projection
        that still uses ``delta_det_row`` and ``det_row_offset`` for its rows,
        unlike the parallel model.  Its source and detector have no finite
        position, so both become drawing choices.
        """
        self.source_detector_dist = None
        self.source_iso_dist = None
        self.use_curved_detector = False

        if self.kind in ('cone', 'translation'):
            self.source_detector_dist = float(
                self.params['source_detector_dist'])
            self.source_iso_dist = float(self.params['source_iso_dist'])
        if self.kind == 'cone':
            self.use_curved_detector = bool(self.params['use_curved_detector'])

        if self.kind in ('parallel', 'multiaxis'):
            self.magnification = 1.0
            self.is_parallel_type = True
        elif np.isinf(self.source_detector_dist):
            if self.kind == 'translation':
                raise ValueError('The translation geometry needs a finite '
                                 'source_detector_dist; every view would '
                                 'otherwise carry the same information.')
            if self.use_curved_detector:
                raise ValueError('A curved detector needs a finite '
                                 'source_detector_dist, which is its radius.')
            self.magnification = 1.0
            self.is_parallel_type = True
        else:
            if self.source_iso_dist == 0.0:
                raise ValueError('source_iso_dist must not be zero.')
            self.magnification = (self.source_detector_dist
                                  / self.source_iso_dist)
            self.is_parallel_type = False

    def _set_drawing_distance(self):
        """Choose the distance used where a position is a drawing choice.

        The rule is a multiple of the volume's largest half-extent, taken over
        all three axes so that the drawn source and detector always sit outside
        the volume box.  This distance affects only where things are drawn.  It
        never affects a detector index, because a parallel projection's
        detector coordinates do not depend on where along the ray the detector
        plane sits.
        """
        half_extents = self.volume_half_extents()
        largest = float(max(half_extents))
        if largest <= 0.0:
            largest = max(self.delta_voxel, 1.0)
        self.drawing_distance = self.drawing_distance_factor * largest

    # ------------------------------------------------------------------
    # The volume
    # ------------------------------------------------------------------

    def volume_half_extents(self):
        """Half the volume's physical size along x, y, and z, as a 3-tuple.

        The z entry is measured from the origin rather than from the volume's
        own center, so it includes ``recon_slice_offset``.  That makes it the
        largest distance from the origin to a corner along z, which is what the
        drawing distance needs.
        """
        half_x = 0.5 * self.num_cols * self.delta_voxel
        half_y = 0.5 * self.num_rows * self.delta_voxel_row
        half_z = (0.5 * self.num_slices * self.delta_voxel_slice
                  + abs(self.recon_slice_offset))
        return float(half_x), float(half_y), float(half_z)

    def voxel_centers(self, ijk):
        """Object coordinates of the centers of the given voxels.

        Args:
            ijk (array_like): voxel indices, (N, 3) as (i, j, k), where i is the
                row index, j the column index, and k the slice index.  Fractional
                indices are allowed.

        Returns:
            ndarray: object coordinates, (N, 3) as (x, y, z).
        """
        ijk = np.asarray(ijk, dtype=np.float64).reshape(-1, 3)
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        x = self.delta_voxel * (j - (self.num_cols - 1) / 2.0)
        y = self.delta_voxel_row * (i - (self.num_rows - 1) / 2.0)
        z = (self.delta_voxel_slice * (k - (self.num_slices - 1) / 2.0)
             + self.recon_slice_offset)
        return np.stack([x, y, z], axis=1)

    def volume_corners(self):
        """The eight outer corners of the volume box, (8, 3).

        A corner sits on the outer face of a corner voxel, half a voxel beyond
        that voxel's center, so the box encloses every voxel.
        """
        half_x, half_y, _ = self.volume_half_extents()
        half_z = 0.5 * self.num_slices * self.delta_voxel_slice
        z_center = self.recon_slice_offset
        corners = [(sx * half_x, sy * half_y, z_center + sz * half_z)
                   for sx in (-1.0, 1.0)
                   for sy in (-1.0, 1.0)
                   for sz in (-1.0, 1.0)]
        return np.array(corners, dtype=np.float64)

    def volume_z_range(self):
        """The volume's z extent as a (z_min, z_max) pair."""
        half_z = 0.5 * self.num_slices * self.delta_voxel_slice
        return (self.recon_slice_offset - half_z,
                self.recon_slice_offset + half_z)

    def ror_cylinder(self):
        """The region of reconstruction, or None when there is no mask to draw.

        mbirtorch's default mask, ``vcd_utils.get_2d_ror_mask``, keeps the
        voxels whose centers lie inside the ellipse through the centers of the
        outermost voxels, and applies it to every slice, so the region is an
        elliptic cylinder about the rotation axis.  The semi-axes are therefore
        (n - 1) / 2 voxel pitches in x and in y, half a voxel less than half
        the volume's physical width.  The projector's shadow of a volume of
        ones inside the mask matches this ellipse to about a tenth of a
        detector channel, and the ellipse half a voxel larger by about one
        channel.  ``vcd_utils.get_support_radius`` is the half voxel larger on
        purpose, because it bounds the outer edge of every voxel the projectors
        update.  The ``radius`` entry is the larger semi-axis.  In z the
        cylinder spans the volume's full extent, because the mask has no z part
        and a voxel's material reaches its outer face.

        A ``use_ror_mask`` value of False means no mask.  A custom array cannot
        be described as an ellipse, so this returns None for that too.
        """
        if self.use_ror_mask is not True:
            return None
        semi_axis_x = 0.5 * (self.num_cols - 1) * self.delta_voxel
        semi_axis_y = 0.5 * (self.num_rows - 1) * self.delta_voxel_row
        z_min, z_max = self.volume_z_range()
        return dict(center=np.array([0.0, 0.0, 0.5 * (z_min + z_max)]),
                    semi_axis_x=float(semi_axis_x),
                    semi_axis_y=float(semi_axis_y),
                    radius=float(max(semi_axis_x, semi_axis_y)),
                    z_min=float(z_min), z_max=float(z_max))

    # ------------------------------------------------------------------
    # The view action
    # ------------------------------------------------------------------

    def view_action(self, view_index):
        """The affine map from the object frame to the projector frame.

        The projector holds the source and the detector fixed and moves the
        object.  For view v that motion is ``q = R p + d``.  The parallel,
        cone, and multiaxis geometries rotate about z, and a helical cone scan
        also shifts the object toward -z by the view's z shift.  The
        translation geometry moves the object by minus the view's translation
        vector and does not rotate.

        Returns:
            (ndarray, ndarray): the 3x3 rotation R and the offset d, (3,).
        """
        view_index = self._check_view_index(view_index)
        offset = np.zeros(3)
        if self.kind == 'translation':
            rotation = np.eye(3)
            offset = -self.translation_vectors[view_index]
        else:
            rotation = _rotation_about_z(self.angles[view_index])
            if self.kind == 'cone':
                offset = np.array([0.0, 0.0, -self.z_shifts[view_index]])
        return rotation, offset

    def to_projector_frame(self, points_xyz, view_index):
        """Object-frame points, (N, 3), mapped to the projector frame."""
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        rotation, offset = self.view_action(view_index)
        return points @ rotation.T + offset

    def to_object_frame(self, points_xyz, view_index):
        """Projector-frame points, (N, 3), mapped back to the object frame.

        This is the inverse of the view's action, and it is what turns the
        projector's picture into the fixed-object picture a drawing needs.
        """
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        rotation, offset = self.view_action(view_index)
        return (points - offset) @ rotation

    def direction_to_object_frame(self, vectors, view_index):
        """Projector-frame directions, (N, 3), mapped to the object frame.

        A direction carries the rotation but not the offset.
        """
        vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        rotation, _ = self.view_action(view_index)
        return vectors @ rotation

    def _check_view_index(self, view_index):
        view_index = int(view_index)
        if not 0 <= view_index < self.num_views:
            raise IndexError(f'view_index {view_index} is outside '
                             f'[0, {self.num_views}).')
        return view_index

    # ------------------------------------------------------------------
    # Projection onto the detector
    # ------------------------------------------------------------------

    def detector_iso_and_center_indices(self):
        """The fractional (row, channel) indices of the detector iso and of the
        detector center.  The detector iso is where the central ray meets the
        detector, which is (u, v) = (0, 0), and the detector center is the
        middle of the grid, index (num - 1) / 2 on each axis."""
        iso_row = (self.row_offset / self.row_pitch
                   + (self.num_det_rows - 1) / 2.0)
        iso_channel = (self.det_channel_offset / self.delta_det_channel
                       + (self.num_det_channels - 1) / 2.0)
        center_row = (self.num_det_rows - 1) / 2.0
        center_channel = (self.num_det_channels - 1) / 2.0
        return ((float(iso_row), float(iso_channel)),
                (float(center_row), float(center_channel)))

    def indices_to_uv(self, row, channel):
        """Detector coordinates (u, v) of fractional detector indices.

        This is the index rule that places the detector's own grid in a
        drawing: the detector center, the corners, and pixel 0.  It is not a
        projection.  The grid is centered on the detector: index (num - 1) / 2
        sits at the grid's middle.  A positive ``det_channel_offset`` moves the
        grid toward negative u, which moves a fixed object's image toward a
        higher channel index, and ``det_row_offset`` acts the same way in v.

        The row pitch and row offset used here are ``row_pitch`` and
        ``row_offset``, which differ from the parameters for the parallel
        geometry; see :meth:`_read_detector`.
        """
        row = np.asarray(row, dtype=np.float64)
        channel = np.asarray(channel, dtype=np.float64)
        u = ((channel - (self.num_det_channels - 1) / 2.0)
             * self.delta_det_channel - self.det_channel_offset)
        v = ((row - (self.num_det_rows - 1) / 2.0) * self.row_pitch
             - self.row_offset)
        return u, v

    def project_points(self, points_xyz, view_index):
        """Fractional detector indices of object-frame points.

        This is the projector's own map; see
        :meth:`TomographyModel.project_points`.  The scene adds no formula of
        its own.  The points are given at their unshifted, unrotated
        positions, which is the frame ``voxel_centers`` returns, and the
        model applies the view's action to them.

        Args:
            points_xyz (array_like): object-frame points, (N, 3).
            view_index (int or sequence of int): one view, or several, as the
                model method allows.

        Returns:
            (ndarray, ndarray): row and channel, fractional.  For one view each
            is (N,); for a sequence of views each is (num_selected_views, N).
        """
        return self.model.project_points(points_xyz, view_index)

    # ------------------------------------------------------------------
    # The source and the detector in the projector frame
    # ------------------------------------------------------------------

    def _source_projector_frame(self):
        """The source position in the projector frame, or None.

        None means the source is infinitely far away, which is the case for the
        parallel and multiaxis geometries and for a cone geometry with an
        infinite source-detector distance.
        """
        if self.is_parallel_type:
            return None
        return np.array([0.0, self.source_iso_dist, 0.0])

    def _ray_direction_projector_frame(self, view_index):
        """The central ray's unit direction in the projector frame.

        The direction points from the source toward the detector.  It is
        constant except for the multiaxis geometry, whose per-view elevation
        tilts it out of the xy plane.  The multiaxis direction of travel is
        ``(0, -cos(elevation), sin(elevation))`` under the source-on-+y
        convention; see ``MULTIAXIS_SOURCE_ON_PLUS_Y``.
        """
        if self.kind != 'multiaxis':
            return np.array([0.0, -1.0, 0.0])
        elevation = self.elevations[self._check_view_index(view_index)]
        direction = np.array([0.0, -np.cos(elevation), np.sin(elevation)])
        if not self.multiaxis_source_on_plus_y:
            direction = -direction
        return direction

    def _detector_axes_projector_frame(self, view_index):
        """The detector's u and v unit axes in the projector frame.

        For a curved detector the u axis is the tangent to the cylinder at the
        central ray.
        """
        u_axis = np.array([1.0, 0.0, 0.0])
        if self.kind == 'multiaxis':
            elevation = self.elevations[self._check_view_index(view_index)]
            v_axis = np.array([0.0, np.sin(elevation), np.cos(elevation)])
        else:
            v_axis = np.array([0.0, 0.0, 1.0])
        return u_axis, v_axis

    def _detector_origin_projector_frame(self, view_index):
        """Where the central ray meets the detector, in the projector frame.

        For a finite cone or translation geometry this is a physical position:
        the detector plane sits at ``source_detector_dist`` from the source on
        the far side of the origin.  For a parallel projection the position
        along the ray is a drawing choice, and the drawing distance is used.
        """
        direction = self._ray_direction_projector_frame(view_index)
        if self.is_parallel_type:
            return self.drawing_distance * direction
        return (np.array([0.0, self.source_iso_dist, 0.0])
                + self.source_detector_dist * direction)

    def _uv_to_projector_frame(self, u, v, view_index):
        """Physical positions of detector coordinates (u, v), (N, 3).

        A flat detector is a plane, and (u, v) are Cartesian coordinates on it.
        A curved detector is a cylinder of radius ``source_detector_dist``
        whose axis passes through the source parallel to z.  On the cylinder u
        is arc length from the central ray and v is height along z, so a pixel
        at (u, v) sits at angle ``u / source_detector_dist`` around the axis.
        The row spacing on that surface is the tangent-plane spacing the
        projector uses, so a curved detector's rows are drawn at equal heights
        rather than at equal angles.
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        u, v = np.broadcast_arrays(u, v)
        if self.kind == 'cone' and self.use_curved_detector:
            theta = u / self.source_detector_dist
            radius = self.source_detector_dist
            x = radius * np.sin(theta)
            y = self.source_iso_dist - radius * np.cos(theta)
            return np.stack([x, y, v], axis=1)
        origin = self._detector_origin_projector_frame(view_index)
        u_axis, v_axis = self._detector_axes_projector_frame(view_index)
        return (origin[None, :] + u[:, None] * u_axis[None, :]
                + v[:, None] * v_axis[None, :])

    def detector_uv_extent(self):
        """The detector's (u, v) extent as ((u_min, u_max), (v_min, v_max)).

        The extent runs from the outer edge of the first pixel to the outer
        edge of the last, which is index -0.5 to index num - 0.5.
        """
        u_low, v_low = self.indices_to_uv(-0.5, -0.5)
        u_high, v_high = self.indices_to_uv(self.num_det_rows - 0.5,
                                            self.num_det_channels - 0.5)
        return ((float(u_low), float(u_high)), (float(v_low), float(v_high)))

    def detector_size(self):
        """The detector's physical (width, height) in ALU.

        The height uses ``row_pitch``, so a parallel geometry's detector is as
        tall as the volume's z extent rather than as tall as
        ``num_det_rows * delta_det_row``.
        """
        return (float(self.num_det_channels * self.delta_det_channel),
                float(self.num_det_rows * self.row_pitch))

    # ------------------------------------------------------------------
    # Drawable primitives
    # ------------------------------------------------------------------

    def _detector_outline_uv(self):
        """The detector outline as (u, v) samples of a closed polyline.

        A flat panel needs only its four corners and a repeat of the first.  A
        curved detector's edges are arcs, so each of the two long edges is
        sampled at ``CURVED_ARC_SAMPLES`` points.
        """
        (u_min, u_max), (v_min, v_max) = self.detector_uv_extent()
        if not (self.kind == 'cone' and self.use_curved_detector):
            u = np.array([u_min, u_max, u_max, u_min, u_min])
            v = np.array([v_min, v_min, v_max, v_max, v_min])
            return u, v
        arc = np.linspace(u_min, u_max, CURVED_ARC_SAMPLES)
        u = np.concatenate([arc, arc[::-1], arc[:1]])
        v = np.concatenate([np.full(CURVED_ARC_SAMPLES, v_min),
                            np.full(CURVED_ARC_SAMPLES, v_max),
                            np.array([v_min])])
        return u, v

    def _detector_corners_uv(self):
        """The four detector corners as (u, v), in a walk around the panel."""
        (u_min, u_max), (v_min, v_max) = self.detector_uv_extent()
        u = np.array([u_min, u_max, u_max, u_min])
        v = np.array([v_min, v_min, v_max, v_max])
        return u, v

    def view(self, view_index):
        """The drawable primitives of one view, in the fixed-object picture.

        Args:
            view_index (int): the view.

        Returns:
            ViewScene: see that class for the entries.
        """
        view_index = self._check_view_index(view_index)

        def to_object(points):
            return self.to_object_frame(points, view_index)

        direction = self._ray_direction_projector_frame(view_index)
        ray_direction = _unit(
            self.direction_to_object_frame(direction, view_index)[0])

        source_projector = self._source_projector_frame()
        if source_projector is None:
            source = None
        else:
            source = to_object(source_projector)[0]

        detector_origin = to_object(
            self._detector_origin_projector_frame(view_index))[0]
        if source is None:
            # No finite source position exists, so the drawing places one at
            # the drawing distance back along the ray from where the central
            # ray meets the detector plane.  Every parallel-type detector
            # origin is itself one drawing distance out from the origin, so
            # this puts the drawn source symmetrically on the other side.
            source_draw = detector_origin - 2.0 * self.drawing_distance * ray_direction
        else:
            source_draw = source

        u_axis_p, v_axis_p = self._detector_axes_projector_frame(view_index)
        detector_u_axis = _unit(
            self.direction_to_object_frame(u_axis_p, view_index)[0])
        detector_v_axis = _unit(
            self.direction_to_object_frame(v_axis_p, view_index)[0])
        detector_normal = _unit(np.cross(detector_u_axis, detector_v_axis))

        detector_center = to_object(self._uv_to_projector_frame(
            -self.det_channel_offset, -self.row_offset, view_index))[0]

        outline_u, outline_v = self._detector_outline_uv()
        detector_outline = to_object(self._uv_to_projector_frame(
            outline_u, outline_v, view_index))
        corner_u, corner_v = self._detector_corners_uv()
        detector_corners = to_object(self._uv_to_projector_frame(
            corner_u, corner_v, view_index))

        pixel0_u, pixel0_v = self.indices_to_uv(0.0, 0.0)
        detector_pixel0 = to_object(self._uv_to_projector_frame(
            pixel0_u, pixel0_v, view_index))[0]

        if self.is_parallel_type:
            # Four rays converging on the drawn source would draw a cone beam,
            # which is not this geometry.  The rays are parallel to the ray
            # direction instead, and each is drawn as long as the drawn central
            # ray, so that the four rays and the central ray start in one
            # plane through the drawn source.
            length = float(np.linalg.norm(source_draw - detector_origin))
            starts = detector_corners - length * ray_direction[None, :]
            corner_rays = np.stack([starts, detector_corners], axis=1)
        else:
            corner_rays = np.stack([np.stack([source_draw, corner])
                                    for corner in detector_corners])

        arc, travel_sense = self._source_travel(view_index, source_draw)

        volume_corners = self.volume_corners()
        row, channel = self.project_points(volume_corners, view_index)
        volume_outline_on_detector = np.stack([row, channel], axis=1)

        # The region of reconstruction's own outline on the detector, from the
        # same projection the box outline uses.  The two rims come back as one
        # array of points, so the result is split into one entry per rim.
        cylinder = self.ror_cylinder()
        ror_outline_on_detector = None
        if cylinder is not None:
            rim_row, rim_channel = self.project_points(self.fit_points(),
                                                       view_index)
            ror_outline_on_detector = np.stack(
                [rim_row, rim_channel], axis=1).reshape(2, RIM_SAMPLES, 2)

        if self.kind == 'translation':
            rotation_axis = None
            translation_path = self.translation_vectors.copy()
        else:
            z_min, z_max = self.volume_z_range()
            z_center = 0.5 * (z_min + z_max)
            half = self.rotation_axis_extension * 0.5 * (z_max - z_min)
            if half <= 0.0:
                half = self.rotation_axis_extension * self.delta_voxel_slice
            rotation_axis = np.array([[0.0, 0.0, z_center - half],
                                      [0.0, 0.0, z_center + half]])
            translation_path = None

        return ViewScene(
            view_index=view_index,
            source=source,
            source_draw=source_draw,
            ray_direction=ray_direction,
            detector_center=detector_center,
            detector_origin=detector_origin,
            detector_u_axis=detector_u_axis,
            detector_v_axis=detector_v_axis,
            detector_normal=detector_normal,
            detector_outline=detector_outline,
            detector_corners=detector_corners,
            detector_pixel0=detector_pixel0,
            volume_corners=volume_corners,
            voxel0_center=self.voxel_centers([[0, 0, 0]])[0],
            rotation_axis=rotation_axis,
            translation_path=translation_path,
            corner_rays=corner_rays,
            volume_outline_on_detector=volume_outline_on_detector,
            ror_outline_on_detector=ror_outline_on_detector,
            ror_cylinder=cylinder,
            rotation_direction_arc=arc,
            source_travel_sense=travel_sense,
        )

    def _source_travel(self, view_index, source_draw):
        """Which way the source moves from this view to the next one.

        The arc lies on the circle about the rotation axis that the source
        travels on, so its radius and its height are the drawn source's own.
        It starts at the drawn source and sweeps ``rotation_arc_sweep``
        radians.  The direction comes from the sign of the step between this
        view's angle and the next one, so a model whose angles fall gets an arc
        the other way.

        The sense follows the projector's convention.  A growing view angle
        turns the object counterclockwise seen from +z, so in a drawing that
        holds the object fixed the source turns clockwise.  The source's
        azimuth is ``pi / 2`` minus the view angle, which is why a rising
        angle gives a falling azimuth.

        Args:
            view_index (int): the view.
            source_draw (ndarray): the drawn source position of that view.

        Returns:
            (ndarray or None, str or None): the arc, (M, 3), and the sense in
            words.  Both are None when the direction cannot be told: the
            translation geometry, a single view, two neighboring views at one
            angle, or a source on the rotation axis.
        """
        if self.kind == 'translation' or self.num_views < 2:
            return None, None
        neighbor = (view_index + 1 if view_index + 1 < self.num_views
                    else view_index - 1)
        step = float(self.angles[neighbor] - self.angles[view_index])
        if neighbor < view_index:
            step = -step
        if step == 0.0:
            return None, None
        radius = float(np.hypot(source_draw[0], source_draw[1]))
        if radius <= 0.0:
            return None, None

        start = float(np.arctan2(source_draw[1], source_draw[0]))
        sweep = -np.sign(step) * self.rotation_arc_sweep
        azimuth = start + np.linspace(0.0, sweep, ROTATION_ARC_SAMPLES)
        arc = np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth),
                        np.full(ROTATION_ARC_SAMPLES, float(source_draw[2]))],
                       axis=1)
        sense = (CLOCKWISE_FROM_PLUS_Z if sweep < 0.0
                 else COUNTERCLOCKWISE_FROM_PLUS_Z)
        return arc, sense

    def reference_view(self):
        """The drawable primitives of the zero-angle reference position.

        This is where the source and the detector sit when the view action is
        the identity.  A drawing of one view cannot say which way the geometry
        projects at angle 0, and a user reads every view angle from that
        position, so a viewer draws it as a fixed reference.

        The identity differs by geometry kind.  It is angle 0 and z shift 0 for
        the parallel and cone geometries, azimuth 0 for the multiaxis geometry,
        and a zero translation vector for the translation geometry.  The
        multiaxis reference keeps the elevation of view 0, because a multiaxis
        geometry has no position free of elevation: its rays are tilted out of
        the xy plane in every view.

        The reference is built by replacing view 0's own view parameters with
        the identity and taking that copy's view 0.  Every entry therefore
        means what the same entry of :meth:`view` means, and a scan whose view
        0 is already the identity gets its own view 0 back.

        Returns:
            ViewScene: the primitives, with ``view_index`` 0.
        """
        if self.kind == 'parallel':
            angles = self.angles.copy()
            angles[0] = 0.0
            overrides = dict(angles=angles)
        elif self.kind == 'cone':
            view_params = np.stack([self.angles, self.z_shifts], axis=1)
            view_params[0] = 0.0
            overrides = dict(view_params_array=view_params)
        elif self.kind == 'multiaxis':
            angles = np.stack([self.angles, self.elevations], axis=1)
            angles[0, 0] = 0.0
            overrides = dict(angles=angles)
        else:
            vectors = self.translation_vectors.copy()
            vectors[0] = 0.0
            overrides = dict(translation_vectors=vectors)
        return self.with_parameters(overrides).view(0)

    def _ray_directions_all_views(self):
        """The central ray's unit direction in the projector frame, (V, 3).

        This is the vectorized form of
        :meth:`_ray_direction_projector_frame`, used by :meth:`trajectory`.
        """
        if self.kind != 'multiaxis':
            return np.tile(np.array([0.0, -1.0, 0.0]), (self.num_views, 1))
        elevation = self.elevations
        directions = np.stack([np.zeros_like(elevation),
                               -np.cos(elevation),
                               np.sin(elevation)], axis=1)
        if not self.multiaxis_source_on_plus_y:
            directions = -directions
        return directions

    def _detector_origins_all_views(self, directions):
        """Where the central ray meets the detector, projector frame, (V, 3).

        This is the vectorized form of
        :meth:`_detector_origin_projector_frame`.
        """
        if self.is_parallel_type:
            return self.drawing_distance * directions
        return (np.array([0.0, self.source_iso_dist, 0.0])
                + self.source_detector_dist * directions)

    def _detector_centers_all_views(self, directions, origins):
        """The center of the detector grid, projector frame, (V, 3).

        This is the vectorized form of ``_uv_to_projector_frame`` evaluated at
        (u, v) = (-det_channel_offset, -row_offset), which is the grid center.
        """
        u = -self.det_channel_offset
        v = -self.row_offset
        if self.kind == 'cone' and self.use_curved_detector:
            radius = self.source_detector_dist
            theta = u / radius
            point = np.array([radius * np.sin(theta),
                              self.source_iso_dist - radius * np.cos(theta),
                              v])
            return np.tile(point, (self.num_views, 1))
        if self.kind == 'multiaxis':
            elevation = self.elevations
            v_axis = np.stack([np.zeros_like(elevation),
                               np.sin(elevation),
                               np.cos(elevation)], axis=1)
        else:
            v_axis = np.tile(np.array([0.0, 0.0, 1.0]), (self.num_views, 1))
        u_axis = np.array([1.0, 0.0, 0.0])
        return origins + u * u_axis + v * v_axis

    def _to_object_frame_all_views(self, points):
        """One projector-frame point per view, (V, 3), in the object frame.

        This is :meth:`to_object_frame` applied view by view, written out over
        the whole scan at once.  The rotation about z is written as its two
        rows rather than as a matrix product, so no per-view matrix is built.
        """
        points = np.asarray(points, dtype=np.float64).reshape(self.num_views, 3)
        if self.kind == 'translation':
            return points + self.translation_vectors
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if self.kind == 'cone':
            z = z + self.z_shifts
        cosine, sine = np.cos(self.angles), np.sin(self.angles)
        return np.stack([x * cosine + y * sine,
                         -x * sine + y * cosine,
                         z], axis=1)

    def trajectory(self):
        """The source and detector-center paths over all views.

        The paths are computed over the whole scan at once, without building a
        :class:`ViewScene` per view, because a helical scan of 1800 views is a
        size the viewer must draw at interactive speed.
        ``test_trajectory_matches_the_per_view_scenes`` checks the two ways
        against each other.

        Returns:
            (ndarray, ndarray): source positions, (V, 3), and detector centers,
            (V, 3), in the fixed-object picture.  Where no finite source
            position exists, the source entry is the drawn point that
            :attr:`ViewScene.source_draw` uses.
        """
        directions = self._ray_directions_all_views()
        origins = self._detector_origins_all_views(directions)
        if self.is_parallel_type:
            # The same rule source_draw uses: one drawing distance back from
            # the detector origin on each side of the volume.
            sources = origins - 2.0 * self.drawing_distance * directions
        else:
            sources = np.tile(np.array([0.0, self.source_iso_dist, 0.0]),
                              (self.num_views, 1))
        centers = self._detector_centers_all_views(directions, origins)
        return (self._to_object_frame_all_views(sources),
                self._to_object_frame_all_views(centers))

    # ------------------------------------------------------------------
    # Derived numbers
    # ------------------------------------------------------------------

    def fit_shape(self):
        """Which shape the fit statement tests, as ``'cylinder'`` or ``'box'``.

        The shape is the cylinder whenever :meth:`ror_cylinder` describes one,
        because that cylinder is the region the reconstruction actually solves
        for.  The automatic reconstruction box is the square around that
        cylinder, so its corners stick out past the field of view by
        construction and asking whether they land on the detector answers a
        question nobody asked.  The shape is the box when there is no mask.
        """
        return 'cylinder' if self.ror_cylinder() is not None else 'box'

    def fit_points(self):
        """Object-frame points that bound the fit shape's projection, (N, 3).

        For the box the points are the eight :meth:`volume_corners`.  For the
        cylinder they are its two rims, the ellipse at ``z_min`` and the
        ellipse at ``z_max``, each sampled at :data:`RIM_SAMPLES` points, with
        the rim at ``z_min`` first.

        Why the two rims bound the whole cylinder.  A point's channel index
        does not depend on its z in any of the four geometries, and the two
        rims run through the same x and y values as the rest of the cylinder,
        so the rims reach every channel index the cylinder reaches.  A point's
        row index is an affine function of its z when x and y are held fixed,
        so along each line of the cylinder parallel to the axis the row index
        is largest at one end and smallest at the other, and both ends are on a
        rim.  The projected rims therefore bound the projected cylinder.

        Returns:
            ndarray: the points, (8, 3) for the box and
            (2 * RIM_SAMPLES, 3) for the cylinder.
        """
        cylinder = self.ror_cylinder()
        if cylinder is None:
            return self.volume_corners()
        angle = np.linspace(0.0, 2.0 * np.pi, RIM_SAMPLES)
        x = cylinder['center'][0] + cylinder['semi_axis_x'] * np.cos(angle)
        y = cylinder['center'][1] + cylinder['semi_axis_y'] * np.sin(angle)
        rims = [np.stack([x, y, np.full(RIM_SAMPLES, float(height))], axis=1)
                for height in (cylinder['z_min'], cylinder['z_max'])]
        return np.concatenate(rims)

    def fit_report(self):
        """Whether the scan's detector covers the region it reconstructs.

        The statement is geometric and ignores the projector's point spread, so
        a shape that just fits can still spread a little past the edge.

        Two rules are used.  A scan that does not travel along the axis is
        asked whether the fit shape lands on the detector in every view.  A
        helical cone scan is asked something else, because its volume is taller
        than one view's detector by design and so leaves the detector in every
        view.  It is asked whether the shape stays inside the detector's
        channel range, and whether the detector's axial coverage sweeps the
        whole volume over the scan.

        The axial coverage of one view is worked out from
        :meth:`project_points` and not from a second formula.  The two points
        (0, 0, z_min) and (0, 0, z_max) on the line x = y = 0 are projected,
        and the row index is an affine function of z along that line in every
        geometry, so the two rows give the line that is then solved for the z
        at row -0.5 and at row ``num_det_rows`` - 0.5.  That pair of z values
        is the view's coverage.  A view whose two rows come out equal says
        nothing about z and is left out of the union, as is every view when
        the volume has no z extent.  For the translation geometry the line
        x = y = 0 is not a rotation axis, because that geometry does not
        rotate; the computation needs no special case all the same.

        Every view is projected in one call, and the overshoots and the
        coverages are then computed with array operations, so the report costs
        one call to :meth:`project_points` however many views the scan has.

        Returns:
            dict: the entries are

            ``shape``: what was tested, from :meth:`fit_shape`.

            ``worst_overshoot_pixels``, ``worst_channel_overshoot_pixels``,
            ``worst_row_overshoot_pixels``: how far the shape reaches past a
            detector edge, in detector pixels, over all views.  Each is zero
            when nothing reaches past that edge.

            ``views_leaving_detector``: the number of views in which the shape
            reaches past any detector edge.

            ``helical_rule``: whether the helical rule above was used.

            ``swept_z_min``, ``swept_z_max``: the lowest and highest z on the
            line x = y = 0 that any view's detector covers.  Both are nan when
            no view has an axial coverage.

            ``z_extent_covered``: whether every z of :meth:`volume_z_range`
            lies in the union of the per-view coverage intervals.  The union is
            merged interval by interval, so a scan that leaves a gap between
            two groups of views reads False even though the gap lies between
            ``swept_z_min`` and ``swept_z_max``.

            ``fits_laterally``: whether the shape stays inside the detector's
            channel range in every view.

            ``fits_axially``: whether the shape stays inside the detector's row
            range in every view or, under the helical rule, whether
            ``z_extent_covered`` is True.

            ``fits``: the answer, which is both of the two above.  Without the
            helical rule it is True when the shape lands inside the detector in
            every view.  With the helical rule it is True when the channel
            overshoot is zero in every view and ``z_extent_covered`` is True.
        """
        points = self.fit_points()
        num_shape_points = int(points.shape[0])
        z_min, z_max = self.volume_z_range()
        # The two axis points ride along with the shape's points, and every
        # view is projected in one call, so the whole report costs one call to
        # project_points.
        probe_points = np.concatenate([points,
                                       np.array([[0.0, 0.0, z_min],
                                                 [0.0, 0.0, z_max]])])
        row, channel = self.project_points(probe_points,
                                           np.arange(self.num_views))

        # One overshoot per view, from the shape's own columns.
        row_over = self._overshoot(row[:, :num_shape_points],
                                   self.num_det_rows)
        channel_over = self._overshoot(channel[:, :num_shape_points],
                                       self.num_det_channels)
        worst_row = float(np.max(row_over, initial=0.0))
        worst_channel = float(np.max(channel_over, initial=0.0))
        views_leaving = int(np.count_nonzero(
            np.maximum(row_over, channel_over) > 0.0))

        # The axial coverage, view by view, from the two axis columns.  The
        # row index is an affine function of z along the line x = y = 0, so
        # the two projected rows give that line's slope and intercept, and
        # the coverage is the z range that lands between the detector's first
        # and last row edge.  A view whose two rows are equal says nothing
        # about z and is dropped, as is every view when the volume has no z
        # extent.
        row_at_z_min = row[:, num_shape_points]
        row_at_z_max = row[:, num_shape_points + 1]
        if z_max == z_min:
            usable = np.zeros(row_at_z_min.shape, dtype=bool)
        else:
            usable = row_at_z_max != row_at_z_min
        if np.any(usable):
            slope = ((row_at_z_max[usable] - row_at_z_min[usable])
                     / (z_max - z_min))
            intercept = row_at_z_min[usable] - slope * z_min
            first = (-0.5 - intercept) / slope
            last = (self.num_det_rows - 0.5 - intercept) / slope
            lows = np.minimum(first, last)
            highs = np.maximum(first, last)
            intervals = list(zip(lows.tolist(), highs.tolist()))
            swept_z_min = float(np.min(lows))
            swept_z_max = float(np.max(highs))
            covered = any(low <= z_min and z_max <= high
                          for low, high in _merged_intervals(intervals))
        else:
            swept_z_min = swept_z_max = float('nan')
            covered = False

        worst = max(worst_row, worst_channel)
        helical_rule = self.kind == 'cone' and self.helical_travel() > 0.0
        # The lateral question is the same for every scan.  The axial question
        # is about the rows for a scan that does not travel, and about the
        # swept coverage for one that does (the two answers are reported
        # separately, because one answer would hide which direction misses).
        fits_laterally = worst_channel <= 0.0
        fits_axially = covered if helical_rule else worst_row <= 0.0
        fits = fits_laterally and fits_axially
        return dict(
            shape=self.fit_shape(),
            fits_laterally=bool(fits_laterally),
            fits_axially=bool(fits_axially),
            worst_overshoot_pixels=float(worst),
            worst_channel_overshoot_pixels=float(worst_channel),
            worst_row_overshoot_pixels=float(worst_row),
            views_leaving_detector=int(views_leaving),
            helical_rule=bool(helical_rule),
            swept_z_min=float(swept_z_min),
            swept_z_max=float(swept_z_max),
            z_extent_covered=bool(covered),
            fits=bool(fits),
        )

    def volume_fits_detector(self):
        """The two entries of :meth:`fit_report` that older callers ask for.

        Returns:
            (bool, float): the report's ``fits`` and
            ``worst_overshoot_pixels``.
        """
        report = self.fit_report()
        return report['fits'], report['worst_overshoot_pixels']

    @staticmethod
    def _overshoot(indices, count):
        """How far the given indices reach past the ends of a detector axis.

        The reduction is along the last axis, so an array of one view's
        indices gives one number and an array of shape (num_views, N) gives
        one number per view.
        """
        indices = np.asarray(indices, dtype=np.float64)
        low = np.max(-0.5 - indices, axis=-1)
        high = np.max(indices - (count - 0.5), axis=-1)
        return np.maximum(0.0, np.maximum(low, high))

    def _fan_and_cone_angles(self):
        """The full fan and cone angles in degrees, and a note about them.

        Each angle is the angle the detector subtends at the source, measured
        edge to edge, so a detector offset makes the fan asymmetric about the
        central ray but does not change this total.  A parallel projection's
        rays do not converge, so both angles are zero.
        """
        if self.is_parallel_type:
            return 0.0, 0.0, ('The rays are parallel, so the source subtends '
                              'no angle and both angles are zero.')
        (u_min, u_max), (v_min, v_max) = self.detector_uv_extent()
        distance = self.source_detector_dist
        if self.use_curved_detector:
            # On the cylinder the fan angle is arc length over radius exactly.
            fan = (u_max - u_min) / distance
            note = ('The fan angle is the detector arc divided by the '
                    'source-detector distance.  The cone angle uses the plane '
                    'tangent to the cylinder at the central ray, which is '
                    'where the projector places the rows.')
        else:
            fan = np.arctan(u_max / distance) - np.arctan(u_min / distance)
            note = 'Both angles are subtended at the source by the flat panel.'
        cone = np.arctan(v_max / distance) - np.arctan(v_min / distance)
        return float(np.degrees(fan)), float(np.degrees(cone)), note

    def helical_travel(self):
        """The total helical travel in ALU, zero for a non-helical scan."""
        if self.kind != 'cone':
            return 0.0
        return float(np.ptp(self.z_shifts))

    def derived_quantities(self):
        """The numbers a text panel reports, as plain floats, ints, and strings.

        Returns:
            dict: the keys are stable names.  The entries are

            ``geometry_kind``, ``num_views``, ``sinogram_shape_text`` and
            ``recon_shape_text``: what geometry this is and how big it is.

            ``magnification``: the scale from a point at the origin to its
            image on the detector.

            ``fan_angle_deg`` and ``cone_angle_deg``: the angles the detector
            subtends at the source, edge to edge, with ``angle_note``
            explaining them.

            ``lateral_fov_alu`` and ``axial_fov_alu``: the detector's width and
            height divided by the magnification, which is the region at the
            rotation axis the detector can see.  The axial value includes the
            helical travel.

            ``delta_voxel_x``, ``delta_voxel_y``, ``delta_voxel_z``: the voxel
            pitches along the three object axes.

            ``volume_extent_x``, ``volume_extent_y``, ``volume_extent_z``,
            ``volume_z_center``: the volume's physical size and z center.

            ``detector_width``, ``detector_height``, ``detector_row_pitch``,
            ``detector_channel_pitch``: the detector's physical size and the
            pitches used to draw it.

            ``detector_center_u`` and ``detector_center_v``: where the center
            of the detector grid sits relative to the central ray.

            ``helical_travel_alu``: the range of the per-view z shifts.

            ``volume_fits_detector``: whether the detector covers the region
            the scan reconstructs, which is :meth:`fit_report`'s ``fits``, and
            ``fits_laterally`` and ``fits_axially`` are its two halves, the
            channel question and the row or swept-coverage question.
            ``fit_shape`` names the shape that was tested,
            ``worst_overshoot_pixels`` says by how much the worst point of it
            misses the detector, ``worst_channel_overshoot_pixels`` and
            ``worst_row_overshoot_pixels`` split that miss between the two
            detector directions, and ``views_leaving_detector`` counts the
            views in which the shape reaches past an edge.
            ``helical_fit_rule`` says whether the helical rule was used, and
            ``swept_z_min``, ``swept_z_max``, and ``z_extent_covered``
            describe the detector's axial coverage over the scan.  The entries
            of :meth:`fit_report` explain all of these.

            ``drawing_distance`` and ``drawing_note``: the distance used where
            a position is a drawing choice, and what was chosen.
        """
        fan, cone, angle_note = self._fan_and_cone_angles()
        width, height = self.detector_size()
        extent_x, extent_y, _ = self.volume_half_extents()
        z_min, z_max = self.volume_z_range()
        travel = self.helical_travel()
        fit = self.fit_report()

        quantities = dict(
            geometry_kind=self.kind,
            num_views=int(self.num_views),
            sinogram_shape_text=str(self.sinogram_shape),
            recon_shape_text=str(self.recon_shape),
            magnification=float(self.magnification),
            fan_angle_deg=fan,
            cone_angle_deg=cone,
            angle_note=angle_note,
            lateral_fov_alu=float(width / self.magnification),
            axial_fov_alu=float(height / self.magnification + travel),
            delta_voxel_x=float(self.delta_voxel),
            delta_voxel_y=float(self.delta_voxel_row),
            delta_voxel_z=float(self.delta_voxel_slice),
            volume_extent_x=float(2.0 * extent_x),
            volume_extent_y=float(2.0 * extent_y),
            volume_extent_z=float(z_max - z_min),
            volume_z_center=float(0.5 * (z_min + z_max)),
            detector_width=float(width),
            detector_height=float(height),
            detector_channel_pitch=float(self.delta_det_channel),
            detector_row_pitch=float(self.row_pitch),
            detector_center_u=float(-self.det_channel_offset),
            detector_center_v=float(-self.row_offset),
            helical_travel_alu=travel,
            volume_fits_detector=fit['fits'],
            fits_laterally=fit['fits_laterally'],
            fits_axially=fit['fits_axially'],
            fit_shape=fit['shape'],
            worst_overshoot_pixels=fit['worst_overshoot_pixels'],
            worst_channel_overshoot_pixels=fit[
                'worst_channel_overshoot_pixels'],
            worst_row_overshoot_pixels=fit['worst_row_overshoot_pixels'],
            views_leaving_detector=fit['views_leaving_detector'],
            helical_fit_rule=fit['helical_rule'],
            swept_z_min=fit['swept_z_min'],
            swept_z_max=fit['swept_z_max'],
            z_extent_covered=fit['z_extent_covered'],
            drawing_distance=float(self.drawing_distance),
            drawing_note=self.drawing_note(),
        )
        return quantities

    def drawing_note(self):
        """A short note naming every position in the drawing that is a choice.

        The note is printed in the viewer's text panel, which has room for a
        few lines only, so each version is as short as its facts allow.
        """
        if self.kind == 'parallel':
            return ('Source and detector positions are drawing choices, at '
                    'the drawing distance.  Rows are drawn at the delta_voxel '
                    'pitch; delta_det_row and det_row_offset take no part in '
                    'a parallel projection.')
        if self.kind == 'multiaxis':
            side = '+y' if self.multiaxis_source_on_plus_y else '-y'
            return ('Source and detector positions are drawing choices, at '
                    f'the drawing distance, with the source on the {side} '
                    'side by convention: a parallel projection is the same '
                    'from both ends.')
        if self.kind == 'cone' and self.is_parallel_type:
            return ('The source-detector distance is infinite: the '
                    'magnification is one, and both positions are drawing '
                    'choices at the drawing distance.')
        return 'Every position drawn is taken from the parameters.'

    # ------------------------------------------------------------------
    # Comparing two scenes
    # ------------------------------------------------------------------

    def drawing_options(self):
        """The drawing choices this scene was built with, as a dictionary.

        These are the constructor arguments that affect only where things are
        drawn.  A copy of this scene passes them on, so that a comparison
        drawing uses the same choices as the drawing it is compared with.
        """
        return dict(
            drawing_distance_factor=self.drawing_distance_factor,
            multiaxis_source_on_plus_y=self.multiaxis_source_on_plus_y,
            rotation_axis_extension=self.rotation_axis_extension,
            rotation_arc_sweep=self.rotation_arc_sweep,
        )

    def with_parameters(self, overrides):
        """A copy of this scene with some parameter values replaced.

        This is how a viewer draws a second geometry that differs from the
        first in a few numbers, which is the calibration use: a vendor geometry
        against the same geometry with an estimated offset.

        Args:
            overrides (dict): parameter values to replace, keyed by mbirtorch
                parameter name.  Every name must be one this geometry kind
                uses, so that a misspelled name raises instead of being
                ignored.

        Returns:
            GeometryScene: a new scene of the same kind, with the same drawing
            options.  A scene's own checks still run on the copy, so an
            override that makes no sense raises the same error it would raise
            for a model built that way.
        """
        allowed = required_parameter_names(self.kind)
        unknown = [name for name in overrides if name not in allowed]
        if unknown:
            raise ValueError(f'These are not parameters of a {self.kind} '
                             f'geometry: {sorted(unknown)}.  The names this '
                             f'kind uses are {list(allowed)}.')
        copy = _model_with_parameters(self.model, overrides)
        return GeometryScene(copy, **self.drawing_options())

    def differences(self, other):
        """What differs between this scene and another one.

        Both the parameters and the derived quantities are compared, because a
        user checking a geometry against a second one wants to see the changed
        parameter and its consequences in one list.  The two sentences named in
        ``DIFFERENCE_EXCLUDED_QUANTITIES`` are left out.

        Args:
            other (GeometryScene): the scene to compare with.  It may be of
                another geometry kind.

        Returns:
            list of (str, object, object): one entry per differing name, as the
            name, this scene's value, and the other scene's value.  The
            parameters come first, in the order
            :func:`required_parameter_names` gives them, then the derived
            quantities in the order :meth:`derived_quantities` gives them.  A
            value of None means the name is not a parameter of that scene's
            geometry kind.  A per-view array counts as one entry, and its two
            values are the two arrays.
        """
        names = list(required_parameter_names(self.kind))
        for name in required_parameter_names(other.kind):
            if name not in names:
                names.append(name)

        rows = []
        for name in names:
            mine = self.params.get(name)
            theirs = other.params.get(name)
            if not values_are_equal(mine, theirs):
                rows.append((name, mine, theirs))

        my_quantities = self.derived_quantities()
        their_quantities = other.derived_quantities()
        for name, value in my_quantities.items():
            if name in DIFFERENCE_EXCLUDED_QUANTITIES:
                continue
            other_value = their_quantities.get(name)
            if not values_are_equal(value, other_value):
                rows.append((name, value, other_value))
        return rows
