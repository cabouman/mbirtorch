"""Geometric calibration from the sinogram.

The functions here run after a scanner reader's ``get_sino_and_model`` and before
reconstruction.  They estimate scan geometry that the vendor metadata got wrong or left out, and
they show a user the evidence behind an estimate.  This module includes the reduced problem
that every estimator runs on, a parameter sweep that reconstructs one slice per candidate value,
a rotation-direction check, the conjugate-view estimators for ``det_channel_offset`` and
``det_rotation``, and the one function that applies a result.

Every function takes the sinogram and the model and works through the model's own
``forward_project``, ``back_project``, and ``recon_direct``.  No function here changes the caller's
model or sinogram except :func:`apply_calibration`, which is documented as the one that does.

The order of preprocessing matters.  Run in this order:
 1. defective-pixel interpolation, background offset correction, and stripe removal
 2. the functions here
 3. ``align_sino_views``.
Stripe removal comes first because a gain stripe sits at a fixed channel and a geometry estimate
would take it for a feature of the object.  ``align_sino_views`` comes last because it shifts each
view on its own.  A wrong ``det_channel_offset`` looks like a per-view shift, so aligning first
would remove part of the error a calibration is meant to find.
"""

import math
import warnings
from typing import NamedTuple

import numpy as np
import torch

from .. import _sharding
from ..cone_beam import ConeBeamModel
from ..multiaxis_parallel import MultiAxisParallelModel
from ..parallel_beam import ParallelBeamModel
from ..utilities import copy_ct_model
from ..vcd_utils import get_support_radius
from . import pipeline
from .utilities import _rotation_kernel, sino_high_pass_filtering

__all__ = ['CalibrationResult', 'build_reduced_problem', 'reduce_sinogram', 'parameter_sweep',
           'check_rotation_direction', 'apply_calibration', 'estimate_det_channel_offset',
           'estimate_det_rotation', 'conjugate_difference']

# The parameters parameter_sweep accepts.  'det_rotation' is not a model parameter.  It is applied
# by resampling the sinogram, and the sweep does that per candidate.
_SWEEP_PARAMETERS = ('det_channel_offset', 'det_row_offset', 'det_rotation')

# Views of the full sinogram read per step when a reduced sinogram is built.  One step's block
# holds this many views by the kept rows, plus the rotation margin when there is one, by every
# channel, so the transient stays small at any sinogram size.
_REDUCE_VIEW_BATCH = 64

# Views rotated per step when a detector rotation is applied in place.
_ROTATE_VIEW_BATCH = 30

# The largest detector rotation, in radians, that a sweep, an estimator, or a difference image
# accepts.  LEAP caps its detector tilt at the same five degrees.
_MAX_DET_ROTATION = math.radians(5.0)


def _check_det_rotation(det_rotation):
    """Refuse a detector rotation beyond the cap.  A detector tilt is a small correction, and the
    resampling that applies it degrades with the angle."""
    if abs(float(det_rotation)) > _MAX_DET_ROTATION:
        raise ValueError(f'A det_rotation of {math.degrees(float(det_rotation)):.2f} degrees is beyond '
                         f'the {math.degrees(_MAX_DET_ROTATION):.0f} degree limit.')


class CalibrationResult(NamedTuple):
    """What one estimator returns.

    Attributes:
        parameter (str): the name of the estimated quantity, for example ``'det_channel_offset'``.
        value (float): the estimate.
        score (float): the score at the estimate.  Lower is better for every method here.
        candidates (ndarray): the values that were scored, shape ``(num_candidates,)``.
        scores (ndarray): the score at each candidate, shape ``(num_candidates,)``.
        method (str): the name of the scoring method.
        reduction (dict): the reduced problem the scores were computed on, as returned by
            :func:`build_reduced_problem`.
    """
    parameter: str
    value: float
    score: float
    candidates: np.ndarray
    scores: np.ndarray
    method: str
    reduction: dict


# ── geometry helpers ──────────────────────────────────────────────────────────────────────────────

def _geometry_kind(ct_model):
    """Classify a model as 'parallel', 'cone', or 'multiaxis', or raise for anything else.

    The translation geometry has no rotation, so none of the quantities calibrated here apply to
    it, and it is refused by name.
    """
    if isinstance(ct_model, ConeBeamModel):
        return 'cone'
    if isinstance(ct_model, MultiAxisParallelModel):
        return 'multiaxis'
    if isinstance(ct_model, ParallelBeamModel):
        return 'parallel'
    raise TypeError(f'geometry_calibration supports ConeBeamModel, ParallelBeamModel, and '
                    f'MultiAxisParallelModel; got {type(ct_model).__name__}.')


def _is_helical(ct_model):
    """True for a cone-beam model with any nonzero per-view axial shift."""
    if not isinstance(ct_model, ConeBeamModel):
        return False
    z_shifts = np.asarray(ct_model.get_params('view_params_array'))[:, 1]
    return bool(np.any(z_shifts != 0))


def _slab_row_window(ct_model, z_lo, z_hi, binned_row_margin=0):
    """The detector rows ``[lo, hi)`` that any ray through the slab ``z_lo <= z <= z_hi`` can reach.

    The window is computed from the model's own geometry, so a sinogram cropped to it carries every
    measurement the slab contributes to.  Cropping to the window makes the reduced problem small.
    It also removes most of the measurements that come only from material outside the slab.

    For cone beam a voxel at axial position z lands on the detector at height ``v = z * pixel_mag``.
    The magnification ``pixel_mag`` depends on the voxel's in-plane depth, and its range over the
    support is ``ConeBeamModel.pixel_magnification_bounds``, which is also what the axial padding
    in ``ConeBeamModel.auto_set_recon_geometry`` uses in the other direction, from rows to z.  For
    multiaxis parallel beam a voxel at axial position z and in-plane depth y lands at
    ``v = z * cos(elevation) + y * sin(elevation)``, with ``|y| <= r`` for the support radius
    ``r``, as ``_multiaxis_vertical_terms`` writes it.  In both geometries the detector row of a
    height v is ``(v + det_row_offset) / delta_det_row + center_row``.

    The window takes the extreme v over the slab's two faces and the support.  It then widens that
    range by one voxel's vertical footprint at each end, by one row below and two above for the
    rounding, and by ``binned_row_margin`` rows at each end.

    Args:
        ct_model: a cone or multiaxis model.
        z_lo, z_hi (float): the slab's axial extent in ALU, including the half-voxel at each end.
        binned_row_margin (int): extra rows kept at each end, in this model's rows.

    Returns:
        tuple: ``(lo, hi, clipped)``.  ``lo`` and ``hi`` are clipped to the detector, and
        ``clipped`` is True when the clip removed rows the window asked for.

    Raises:
        ValueError: when no detector row can see the slab.
    """
    kind = _geometry_kind(ct_model)
    sinogram_shape, recon_shape = ct_model.get_params(['sinogram_shape', 'recon_shape'])
    num_rows = int(sinogram_shape[1])
    delta_det_row, det_row_offset, delta_voxel, voxel_row_aspect, voxel_slice_aspect, use_ror_mask = \
        ct_model.get_params(['delta_det_row', 'det_row_offset', 'delta_voxel', 'voxel_row_aspect',
                             'voxel_slice_aspect', 'use_ror_mask'])
    delta_voxel_slice = voxel_slice_aspect * delta_voxel
    support_radius = get_support_radius(recon_shape, voxel_row_aspect * delta_voxel, delta_voxel,
                                        use_ror_mask=use_ror_mask)

    if kind == 'cone':
        mags = np.array(ct_model.pixel_magnification_bounds())
        if np.isinf(mags[1]):
            # The source lies inside the support, so no ray bound holds.  Keep every row.
            return 0, num_rows, False
        v_values = np.outer(mags, [z_lo, z_hi])
        footprint = float(mags.max()) * delta_voxel_slice
    else:
        elevations = np.asarray(ct_model.get_params('angles'))[:, 1]
        cos_el, sin_el = np.cos(elevations), np.sin(elevations)
        v_values = np.stack([z * cos_el + y * sin_el for z in (z_lo, z_hi)
                             for y in (-support_radius, support_radius)])
        # A tilted voxel's vertical footprint is bounded by its diagonal.
        footprint = math.hypot(delta_voxel_slice, max(delta_voxel, voxel_row_aspect * delta_voxel))

    center_row = (num_rows - 1) / 2.0
    v_min, v_max = float(v_values.min()) - footprint, float(v_values.max()) + footprint
    lo = math.floor((v_min + det_row_offset) / delta_det_row + center_row) - 1 - int(binned_row_margin)
    hi = math.ceil((v_max + det_row_offset) / delta_det_row + center_row) + 2 + int(binned_row_margin)
    if hi <= 0 or lo >= num_rows:
        raise ValueError(f'No detector row can see the slab from z = {z_lo:.4g} to {z_hi:.4g} ALU: '
                         f'its rows would be {lo} to {hi} on a detector of {num_rows} rows.')
    return max(0, lo), min(num_rows, hi), bool(lo < 0 or hi > num_rows)


def _rotation_row_margin(det_rotation, max_row_distance, num_channels):
    """Rows beyond a window that a rotation of the detector can sample from.

    An output pixel at row i and channel j reads the input at a row displaced by
    ``(cos(a) - 1) * (i - center_row) + sin(a) * (j - center_col)``.  Over the detector's channels
    that displacement is at most ``(1 - cos(a)) * |i - center_row| + |sin(a)| * num_channels / 2``.
    The caller passes the largest ``|i - center_row|`` in its window.  One row is added for the
    bilinear neighbor.
    """
    a = abs(float(det_rotation))
    bound = (1.0 - math.cos(a)) * max_row_distance + math.sin(a) * num_channels / 2.0
    return int(math.ceil(bound)) + 1


# ── the reduced problem ───────────────────────────────────────────────────────────────────────────

def build_reduced_problem(ct_model, *, view_stride=4, bin_factor=2, num_slab_slices=8,
                          slice_index=None, row_margin=0):
    """Build the smaller model that a calibration search scores candidates on.

    The reduced model keeps every ``view_stride``-th view, bins the detector by ``bin_factor`` in
    rows and in channels, and reconstructs a thin slab of ``num_slab_slices`` slices centered on
    recon slice ``slice_index`` of the full model.  Each of the three reductions keeps the geometry
    in ALU unchanged, so a value estimated on the reduced model applies to the full model as it is.
    The detector pitches are multiplied by the bin factor, and the reconstruction geometry is
    recomputed from them, so the reduced model's field of view equals the full model's.  The
    detector offsets are in ALU and do not change.  The bin factor must divide the row and channel
    counts exactly.  A dropped leftover channel would move the detector center by half a bin, and
    that is a bias in ``det_channel_offset`` of the size this module exists to find.

    The slab is selected differently per geometry.  In parallel beam detector row r is recon slice
    r, so the slab is a band of detector rows.  In cone beam and multiaxis parallel beam the slab is
    set through ``recon_shape`` and ``recon_slice_offset``.  The detector rows are then cropped to
    the rows that rays through the slab can reach, which :func:`_slab_row_window` computes, and
    the row offset is compensated for the crop.

    A thin slab makes a search cheap, and it has a cost.  Rays through the slab also cross material
    outside it, which the slab does not represent.  A score that compares the data with a
    projection of the slab therefore carries a term that the slab cannot explain.  A caller that
    needs the whole axial extent passes ``num_slab_slices=None``, which keeps every detector row
    and the automatic slice count.  A helical cone-beam scan always keeps the whole extent,
    because every ray through a slab comes from a different axial position.

    The reduced model is pinned to the full model's lead device, so the scores are reproducible and
    the model does not run its own device search.  It inherits the full model's ``compile_mode``.
    A caller-supplied ``use_ror_mask`` array has the full model's shape and cannot serve the
    reduced one, so the reduced model uses the default mask instead.

    Args:
        ct_model (TomographyModel): a cone, parallel, or multiaxis parallel model.  Not modified.
        view_stride (int, optional): keep every ``view_stride``-th view.  Must divide the view
            count.  Defaults to 4.
        bin_factor (int, optional): detector binning factor in rows and channels.  Must divide both
            detector counts.  Defaults to 2.
        num_slab_slices (int or None, optional): recon slices in the slab, or None for the whole
            axial extent.  A helical scan always keeps the whole extent.  Defaults to 8.
        slice_index (int, optional): the full model's recon slice at the center of the slab.  None
            (the default) is the middle slice.
        row_margin (int, optional): extra full-resolution detector rows kept on each side of the
            slab's row window, for a sweep that moves where the slab lands on the detector.
            Parallel beam ignores it.  Defaults to 0.

    Returns:
        tuple: ``(reduced_model, reduction)``.  ``reduction`` is a dict that records the reduction,
        and it is what :func:`reduce_sinogram` needs to reduce a sinogram or a weights array to
        match.  Its entries are these:

        - ``'geometry'``, ``'view_stride'``, ``'bin_factor'``;
        - ``'row_window'``, the full-resolution rows kept, as ``(lo, hi)``;
        - ``'axial_thinning'``, False when the whole axial extent is kept;
        - ``'slice_index'``, the requested slice of the full model, and ``'slice_in_slab'``, the
          reduced model's slice whose center is nearest to it;
        - ``'num_slab_slices'`` and ``'slab_z_center'``, which is None unless a cone or multiaxis
          slab was selected;
        - ``'det_row_offset_shift'``, what the row crop added to the reduced model's
          ``det_row_offset``, so that a candidate value for the full model is set on the reduced
          model as the candidate plus this shift;
        - ``'full_sinogram_shape'``, ``'sinogram_shape'``, ``'recon_shape'``, and ``'devices'``.

    Raises:
        TypeError: for a translation model.
        ValueError: when ``view_stride`` does not divide the view count, when ``bin_factor`` does not
            divide the detector counts, when ``slice_index`` is outside the recon, or when no
            detector row can see the requested slab.
    """
    kind = _geometry_kind(ct_model)
    view_stride, bin_factor = int(view_stride), int(bin_factor)
    if num_slab_slices is not None:
        num_slab_slices = int(num_slab_slices)
    if view_stride < 1 or bin_factor < 1 or (num_slab_slices is not None and num_slab_slices < 1):
        raise ValueError('view_stride, bin_factor, and num_slab_slices must each be at least 1; got '
                         f'{view_stride}, {bin_factor}, {num_slab_slices}.')
    num_views, num_det_rows, num_det_channels = (int(s) for s in ct_model.get_params('sinogram_shape'))
    if num_views % view_stride != 0:
        raise ValueError(f'view_stride must divide the view count: {num_views} views, stride '
                         f'{view_stride}.  A stride that divides keeps a 360-degree scan\'s opposite '
                         'views.')
    if num_det_rows % bin_factor != 0 or num_det_channels % bin_factor != 0:
        raise ValueError(f'bin_factor must divide both detector counts exactly: {num_det_rows} rows '
                         f'and {num_det_channels} channels, factor {bin_factor}.  A dropped leftover '
                         'channel would move the detector center by half a bin.')
    full_recon_shape = tuple(int(s) for s in ct_model.get_params('recon_shape'))
    if slice_index is None:
        slice_index = (full_recon_shape[2] - 1) // 2
    slice_index = int(slice_index)
    if not 0 <= slice_index < full_recon_shape[2]:
        raise ValueError(f'slice_index {slice_index} is outside the recon, which has '
                         f'{full_recon_shape[2]} slices.')
    axial_thinning = num_slab_slices is not None and not _is_helical(ct_model)

    # First reduction: a copy with the kept views and the binned detector, at the full row count.
    # The detector pitches grow by the bin factor and the recon geometry is recomputed from them,
    # so the copy covers the same field of view in ALU with coarser voxels.  This copy is only read
    # for its geometry.  It never projects, so it compiles nothing.
    required, _, _ = ct_model.get_all_params()
    angles = np.asarray(required['angles'])[::view_stride]
    copy_kwargs = dict(new_angles=angles, new_num_det_rows=num_det_rows // bin_factor,
                       new_num_det_cols=num_det_channels // bin_factor)
    if kind == 'cone':
        copy_kwargs['new_helical_z_shifts'] = np.asarray(required['helical_z_shifts'])[::view_stride]
    binned = copy_ct_model(ct_model, **copy_kwargs)
    if not isinstance(ct_model.get_params('use_ror_mask'), bool):
        binned.set_params(use_ror_mask=True)
    delta_det_channel, delta_det_row = ct_model.get_params(['delta_det_channel', 'delta_det_row'])
    binned.set_params(delta_det_channel=bin_factor * delta_det_channel,
                      delta_det_row=bin_factor * delta_det_row)
    binned.auto_set_recon_geometry()
    binned_rows = num_det_rows // bin_factor

    # Second reduction: the slab, and the detector rows it needs.
    slab_z_center = None
    if not axial_thinning:
        row_lo, row_hi = 0, binned_rows
    elif kind == 'parallel':
        # Row r is slice r.  The slab is the binned rows around the binned row that holds the
        # requested slice.
        row_center = slice_index // bin_factor
        row_lo = max(0, min(row_center - num_slab_slices // 2, binned_rows - num_slab_slices))
        row_hi = min(binned_rows, row_lo + num_slab_slices)
    else:
        slab_z_center = float(ct_model.recon_slice_z(slice_index))
        delta_voxel, voxel_slice_aspect = binned.get_params(['delta_voxel', 'voxel_slice_aspect'])
        half_height = 0.5 * num_slab_slices * voxel_slice_aspect * delta_voxel
        binned_row_margin = math.ceil(row_margin / bin_factor)
        row_lo, row_hi, clipped = _slab_row_window(binned, slab_z_center - half_height,
                                                   slab_z_center + half_height,
                                                   binned_row_margin=binned_row_margin)
        if clipped and binned_row_margin > 0:
            warnings.warn('The slab sits near the edge of the detector, so the row margin a sweep '
                          'asked for was cut off by the detector edge.  Candidates that move the '
                          'slab toward that edge lose part of their data.')

    reduced = copy_ct_model(binned, new_num_det_rows=row_hi - row_lo)
    # The model reads compile_mode when it first builds its projectors, which has not happened yet.
    reduced.compile_mode = ct_model.compile_mode
    det_row_offset_shift = 0.0
    if kind != 'parallel' and (row_lo, row_hi) != (0, binned_rows):
        # The crop moves the detector center, and the row offset moves with it by the rule
        # apply_detector_crop uses: half the difference of the rows removed at the two ends.
        crop_top, crop_bottom = row_lo, binned_rows - row_hi
        det_row_offset_shift = (crop_bottom - crop_top) / 2.0 * binned.get_params('delta_det_row')
        reduced.set_params(det_row_offset=binned.get_params('det_row_offset') + det_row_offset_shift)
    reduced.auto_set_recon_geometry()
    if slab_z_center is not None:
        recon_rows, recon_cols, _ = reduced.get_params('recon_shape')
        reduced.set_params(recon_shape=(recon_rows, recon_cols, num_slab_slices),
                           recon_slice_offset=slab_z_center)
    reduced.configure_devices(devices=[ct_model.torch_device])

    # The reduced model's slice nearest the requested slice.  In parallel beam it is the row's
    # place in the band.  With a slab it is the slab's middle, which is exact for an odd slice
    # count and half a slice off for an even one.  With the whole extent kept, the reduced volume
    # has its own automatic slice grid, and the requested slice's axial position is looked up on it.
    if kind == 'parallel':
        slice_in_slab = slice_index // bin_factor - row_lo
    elif axial_thinning:
        slice_in_slab = (num_slab_slices - 1) // 2
    else:
        slice_in_slab = reduced.nearest_recon_slice(float(ct_model.recon_slice_z(slice_index)))

    reduction = {
        'geometry': kind,
        'view_stride': view_stride,
        'bin_factor': bin_factor,
        'row_window': (row_lo * bin_factor, row_hi * bin_factor),
        'axial_thinning': axial_thinning,
        'slice_index': slice_index,
        'slice_in_slab': int(slice_in_slab),
        'num_slab_slices': int(reduced.get_params('recon_shape')[2]),
        'slab_z_center': slab_z_center,
        'det_row_offset_shift': float(det_row_offset_shift),
        'full_sinogram_shape': (num_views, num_det_rows, num_det_channels),
        'sinogram_shape': tuple(int(s) for s in reduced.get_params('sinogram_shape')),
        'recon_shape': tuple(int(s) for s in reduced.get_params('recon_shape')),
        'devices': [str(d) for d in reduced.sino_placement.devices],
    }
    return reduced, reduction


def reduce_sinogram(sino, reduction, *, det_rotation=0.0):
    """Reduce a sinogram, or a weights array, to match a reduced model.

    The reduction keeps every ``view_stride``-th view, crops the rows to the reduced model's row
    window, and averages each ``bin_factor`` by ``bin_factor`` block of detector pixels.  A nonzero
    ``det_rotation`` rotates the kept rows first, at full resolution and about the full detector's
    center, so the result equals a crop of the rotated full sinogram.  The full sinogram is read in
    view batches and never copied whole, and the result is a small new array.

    A weights array reduced this way holds the mean weight of each bin.  For inverse-variance
    weights the weight of a binned measurement would be ``bin_factor ** 2`` times that mean.  The
    factor is the same for every candidate, so it does not change which candidate scores lowest.

    Args:
        sino (ndarray or tensor): the full sinogram, shape ``reduction['full_sinogram_shape']``.  A
            host array or a device tensor; an array in the divided device form is refused.
        reduction (dict): the record returned by :func:`build_reduced_problem`.  A record may name
            the views to keep as ``'view_indices'`` in place of the stride.
        det_rotation (float, optional): detector rotation in radians to apply before the crop.
            Defaults to 0.0.

    Returns:
        ndarray: float32, shape ``reduction['sinogram_shape']``.
    """
    _sharding.reject_shards('reduce_sinogram', sino=sino)
    full_shape = tuple(reduction['full_sinogram_shape'])
    if tuple(sino.shape) != full_shape:
        raise ValueError(f'reduce_sinogram: the sinogram has shape {tuple(sino.shape)}, and the '
                         f'reduction was built for {full_shape}.')
    num_views, num_rows, num_channels = full_shape
    stride, bin_factor = reduction['view_stride'], reduction['bin_factor']
    row_lo, row_hi = reduction['row_window']
    # The views kept are every stride-th view, or an explicit list when the record names one.
    view_indices = reduction.get('view_indices')
    if view_indices is None:
        view_indices = np.arange(0, num_views, stride)
    view_indices = np.asarray(view_indices, dtype=np.int64)
    kept_views, kept_rows = int(view_indices.size), row_hi - row_lo
    out = np.empty((kept_views, kept_rows // bin_factor, num_channels // bin_factor), dtype=np.float32)

    # A rotation reads rows beyond the window.  The block read per batch is widened by the rows
    # the rotation can sample, and the rotation turns about the full detector's center, expressed
    # in the block's own row indices.
    rotate = float(det_rotation) != 0.0
    if rotate:
        center_row = (num_rows - 1) / 2.0
        max_row_distance = max(abs(row_lo - center_row), abs(row_hi - 1 - center_row))
        margin = _rotation_row_margin(det_rotation, max_row_distance, num_channels)
        band_lo, band_hi = max(0, row_lo - margin), min(num_rows, row_hi + margin)
        center = (center_row - band_lo, (num_channels - 1) / 2.0)
    else:
        band_lo, band_hi = row_lo, row_hi
    device = torch.device(reduction['devices'][0])

    with torch.no_grad():
        for k0 in range(0, kept_views, _REDUCE_VIEW_BATCH):
            k1 = min(k0 + _REDUCE_VIEW_BATCH, kept_views)
            batch = view_indices[k0:k1]
            block = pipeline._stage_batch(sino[batch, band_lo:band_hi, :], device)
            if rotate:
                block = _rotation_kernel(block, det_rotation, center=center)
                block = block[:, row_lo - band_lo:row_hi - band_lo, :]
            # Average each bin_factor x bin_factor block of detector pixels.
            binned = block.reshape(k1 - k0, kept_rows // bin_factor, bin_factor,
                                   num_channels // bin_factor, bin_factor).mean(dim=(2, 4))
            out[k0:k1] = binned.to('cpu', torch.float32).numpy()
    return out


# ── the parameter sweep ───────────────────────────────────────────────────────────────────────────

def parameter_sweep(ct_model, sino, parameter, values, *, slice_index=None, filter_name='ramp'):
    """Reconstruct one slice per candidate value of a geometry parameter, for viewing.

    This is the manual calibration workflow.  A user looks at the stack in the slice viewer, picks
    the candidate whose slice is sharpest or free of rings, and sets the value on the model.  Each
    slice is a direct reconstruction from every view at the full channel resolution, with no view
    stride and no binning.  For a parallel or circular cone-beam scan it comes from a one-slice
    problem built with :func:`build_reduced_problem`, whose detector is cropped to the rows that
    rays through the slice can reach.  Each candidate then costs one filter pass over those rows
    and one back projection into one slice.  The row crop is small for a slice near the center of
    the volume and grows with the slice's distance from it, because the cone widens.  A helical
    scan keeps every row and every slice, and the requested slice is read out of the whole volume.

    Until the detector offsets become call-time inputs of the projectors, setting one on the
    reduced model rebuilds its projector bindings, and the first changed value costs one retrace of
    the compiled projection bodies.  Later values do not.

    The candidate index is the last axis, which is the axis the slice viewer pages through by
    default::

        from mbirtorch.preprocess import geometry_calibration
        values = np.linspace(-4.0, 4.0, 17)
        slices = geometry_calibration.parameter_sweep(ct_model, sino, 'det_channel_offset', values)
        mbirtorch.slice_viewer(slices, title='det_channel_offset sweep')

    Args:
        ct_model (TomographyModel): a cone, parallel, or multiaxis parallel model.  Not modified.
        sino (ndarray or tensor): the sinogram, shape ``ct_model.get_params('sinogram_shape')``.
            Not modified.
        parameter (str): ``'det_channel_offset'`` or ``'det_row_offset'``, in ALU, or
            ``'det_rotation'``, in radians.  ``det_row_offset`` is refused for parallel beam, which
            does not use it.  A row-offset sweep on a cone-beam scan shows the object at a
            different height in each candidate, because a wrong row offset shifts the volume.  Its
            sharpness changes only in proportion to the cone angle, so the stack is a way to see
            where the object sits rather than a way to pick the offset by sharpness.
            ``det_rotation`` is refused for a curved detector, whose channel coordinate is an arc
            rather than a distance in the detector plane.
        values (sequence of float): the candidate values.
        slice_index (int, optional): the full model's recon slice to reconstruct.  None (the
            default) is the middle slice.
        filter_name (str, optional): the direct reconstruction filter.  Defaults to ``'ramp'``.

    Returns:
        ndarray: float32 stack of shape ``(num_recon_rows, num_recon_cols, num_candidates)``.
    """
    kind = _geometry_kind(ct_model)
    _sharding.reject_shards('parameter_sweep', sino=sino)
    if parameter not in _SWEEP_PARAMETERS:
        raise ValueError(f'parameter_sweep accepts {", ".join(_SWEEP_PARAMETERS)}; got {parameter!r}.')
    if parameter == 'det_row_offset' and kind == 'parallel':
        raise ValueError('det_row_offset has no effect in parallel beam geometry, so there is nothing '
                         'to sweep.')
    if parameter == 'det_rotation' and kind == 'cone' and ct_model.get_params('use_curved_detector'):
        raise ValueError('det_rotation cannot be applied to a curved detector: the rotation resamples '
                         'a flat detector plane.')
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError('values must be a non-empty sequence of finite numbers.')
    if parameter == 'det_rotation':
        _check_det_rotation(np.max(np.abs(values)))

    # A row-offset candidate moves where the slice lands on the detector, so the row window is
    # widened by the largest move among the candidates.
    row_margin = 0
    if parameter == 'det_row_offset':
        current = float(ct_model.get_params('det_row_offset'))
        row_margin = math.ceil(np.max(np.abs(values - current)) / ct_model.get_params('delta_det_row'))
    reduced, reduction = build_reduced_problem(ct_model, view_stride=1, bin_factor=1,
                                               num_slab_slices=1, slice_index=slice_index,
                                               row_margin=row_margin)
    if parameter != 'det_rotation':
        sino_reduced = reduce_sinogram(sino, reduction)

    recon_rows, recon_cols, _ = reduction['recon_shape']
    slice_in_slab = reduction['slice_in_slab']
    stack = np.empty((recon_rows, recon_cols, values.size), dtype=np.float32)
    for k, value in enumerate(values):
        if parameter == 'det_rotation':
            sino_reduced = reduce_sinogram(sino, reduction, det_rotation=float(value))
        elif parameter == 'det_row_offset':
            reduced.set_params(det_row_offset=float(value) + reduction['det_row_offset_shift'])
        else:
            reduced.set_params(**{parameter: float(value)})
        recon = reduced.recon_direct(sino_reduced, filter_name=filter_name)
        stack[:, :, k] = np.asarray(recon)[:, :, slice_in_slab]
    return stack


# ── the rotation-direction check ──────────────────────────────────────────────────────────────────

def _direct_residual_score(ct_model, sino, filtered_sino=None, row_fraction=0.5):
    """The normalized high-pass residual of a direct reconstruction, over the central rows.

    The model reconstructs the sinogram directly, forward projects the result, and high-pass
    filters both the sinogram and the projection with ``sino_high_pass_filtering``.  The score is
    the mean squared difference divided by the mean squared filtered sinogram.  A direct
    reconstruction is one filtered back projection and does not iterate against the data, so the
    residual measures how consistent the data are with the model's geometry.  The high-pass filter
    removes the smooth mismatch that scatter, cupping, and the cone-beam approximation leave, and
    it keeps the edges that a geometry error displaces.

    The residual is taken over the central ``row_fraction`` of the detector rows.  The direct
    reconstruction's own error grows with the cone angle, so the outer rows add error that has
    nothing to do with the geometry under test.  On synthetic cone-beam scans, restricting the
    residual from every row to the central half raised the ratio of the wrong-direction score to
    the right-direction score by a factor of 1.35 to 2.3.  Narrower bands changed the ratio by
    less than 20 percent.

    Args:
        ct_model: the reduced model.
        sino (ndarray): the reduced sinogram.
        filtered_sino (ndarray, optional): ``sino_high_pass_filtering(sino)``, when the caller has
            it already.
        row_fraction (float, optional): the fraction of central rows scored.  Defaults to 0.5.
    """
    if filtered_sino is None:
        filtered_sino = sino_high_pass_filtering(sino)
    recon = ct_model.recon_direct(sino)
    projection = ct_model.forward_project(recon)
    filtered_projection = sino_high_pass_filtering(projection)
    num_rows = sino.shape[1]
    keep = max(1, int(round(num_rows * row_fraction)))
    lo = (num_rows - keep) // 2
    filtered_sino = filtered_sino[:, lo:lo + keep]
    filtered_projection = filtered_projection[:, lo:lo + keep]
    energy = np.mean(filtered_sino ** 2, dtype=np.float64)
    if energy == 0.0:
        raise ValueError('The reduced sinogram is zero over the scored rows, so no score exists.')
    residual = np.mean((filtered_sino - filtered_projection) ** 2, dtype=np.float64)
    return float(residual / energy)


# The ratio of the worse direction's score to the better one below which check_rotation_direction
# warns that its answer rests on a small margin.  The value is provisional.  The smallest ratio seen
# on synthetic data was 2.2, at 32 views, 16 rows, and 32 channels, and the ratio falls with the
# fan angle, so a narrow-fan scan may sit below it.
_DIRECTION_MIN_RATIO = 1.5


def check_rotation_direction(ct_model, sino, *, view_stride=4, bin_factor=2):
    """Decide whether the view angles run in the right direction for a cone-beam scan.

    A reversed rotation direction is a common metadata failure, and its symptom is a
    reconstruction that is subtly warped rather than obviously wrong.  The check scores the
    geometry as given and the geometry with every view angle negated, and reports which scores
    lower.  With the wrong direction each measurement is assigned to a ray whose angle is off by
    twice its fan angle.  The data are then inconsistent with the model away from the center
    channel, and a direct reconstruction does not reproduce them.

    The score is the high-pass residual of a direct reconstruction over the central detector rows,
    computed by :func:`_direct_residual_score`.  It runs on a reduced problem that keeps every
    fourth view and bins the detector by two, and that keeps the whole axial extent.  The whole
    extent is kept because a thin slab cannot explain the measurements that pass through material
    outside it, and on synthetic scans that unexplained part hid the difference between the two
    directions.  The separation grows with the fan angle, so the check is most reliable on a scan
    with a wide fan.  Both scores are returned so the caller can see the margin, and the function
    warns when the worse score is less than 1.5 times the better one.  That threshold is
    provisional.  The scores depend on the size of the reduced problem and on the fixed pixel
    widths of the high-pass filter, so a ratio measured on one scan does not transfer to another.

    Only cone beam is supported.  For parallel beam, negating the angles mirrors
    the reconstruction and changes nothing else, so the direction cannot be decided from the data.
    For multiaxis parallel beam with nonzero elevation the two directions do differ, and the check
    does not support that geometry yet.  A helical scan is refused as well.

    Args:
        ct_model (ConeBeamModel): the model.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.
        view_stride (int, optional): passed to :func:`build_reduced_problem`.  Defaults to 4.
        bin_factor (int, optional): passed to :func:`build_reduced_problem`.  Defaults to 2.

    Returns:
        CalibrationResult: ``parameter`` is ``'rotation_direction'``, ``candidates`` is
        ``[1.0, -1.0]`` for the angles as given and negated, and ``value`` is the better of the two.
        A value of -1.0 means the angles should be negated; :func:`apply_calibration` does that.
    """
    kind = _geometry_kind(ct_model)
    if kind == 'parallel':
        raise ValueError('check_rotation_direction needs a cone-beam model.  In parallel-beam '
                         'geometry negating the view angles mirrors the reconstruction, so the two '
                         'directions cannot be told apart from the data.')
    if kind == 'multiaxis':
        raise ValueError('check_rotation_direction supports cone beam only; a '
                         'multiaxis parallel model is not supported yet.')
    if _is_helical(ct_model):
        raise ValueError('check_rotation_direction does not support a helical scan yet.')
    _sharding.reject_shards('check_rotation_direction', sino=sino)
    reduction_kwargs = dict(view_stride=view_stride, bin_factor=bin_factor, num_slab_slices=None)

    reduced, reduction = build_reduced_problem(ct_model, **reduction_kwargs)
    sino_reduced = reduce_sinogram(sino, reduction)
    filtered = sino_high_pass_filtering(sino_reduced)
    # The reversed model is the full model with every angle negated, reduced the same way.  The
    # reduction depends on the geometry and not on the angle signs, so one reduced sinogram serves
    # both.
    required, _, _ = ct_model.get_all_params()
    reversed_full = copy_ct_model(ct_model, new_angles=-np.asarray(required['angles']),
                                  new_helical_z_shifts=np.asarray(required['helical_z_shifts']))
    reversed_full.compile_mode = ct_model.compile_mode
    reversed_reduced, _ = build_reduced_problem(reversed_full, **reduction_kwargs)

    candidates = np.array([1.0, -1.0])
    scores = np.array([_direct_residual_score(reduced, sino_reduced, filtered),
                       _direct_residual_score(reversed_reduced, sino_reduced, filtered)])
    best = int(np.argmin(scores))
    ratio = float(scores.max() / max(scores.min(), 1e-30))
    if ratio < _DIRECTION_MIN_RATIO:
        warnings.warn(f'check_rotation_direction: the worse direction scored only {ratio:.2f} times '
                      f'the better one, below the margin of {_DIRECTION_MIN_RATIO} that the check '
                      'expects.  The answer may be unreliable; a narrow fan angle gives a small '
                      'margin.')
    return CalibrationResult(parameter='rotation_direction', value=float(candidates[best]),
                             score=float(scores[best]), candidates=candidates, scores=scores,
                             method='direct_residual', reduction=reduction)


# ── applying a result ─────────────────────────────────────────────────────────────────────────────

def _resolve_work_device():
    """The device a host array's views are rotated on: the process's default device."""
    from ..tomography_model import _resolve_device
    return _resolve_device('auto')


def _rotate_views_in_place(sino, det_rotation):
    """Rotate every view of ``sino`` by ``det_rotation`` radians, writing each batch back in place.

    The sinogram may be a writable floating-point host array or a floating-point tensor on any
    device.  The rotation runs in float32 and the result is written back in the array's own dtype.
    A read-only host array is refused, because the alternative is a second full-size sinogram.
    """
    if torch.is_tensor(sino):
        if not torch.is_floating_point(sino):
            raise TypeError(f'apply_calibration needs a floating-point sinogram; got {sino.dtype}.')
        device = sino.device if sino.device.type != 'cpu' else _resolve_work_device()
    else:
        if not isinstance(sino, np.ndarray):
            raise TypeError('apply_calibration needs the sinogram as a numpy array or a torch tensor, '
                            f'so the rotation can be written in place; got {type(sino).__name__}.')
        if not np.issubdtype(sino.dtype, np.floating):
            raise TypeError(f'apply_calibration needs a floating-point sinogram; got {sino.dtype}.')
        if not sino.flags.writeable:
            raise ValueError('apply_calibration rotates the sinogram in place, and the array is '
                             'read-only.  Pass a writable copy.')
        device = _resolve_work_device()
    num_views = sino.shape[0]
    with torch.no_grad():
        for j in range(0, num_views, _ROTATE_VIEW_BATCH):
            end = min(j + _ROTATE_VIEW_BATCH, num_views)
            rotated = _rotation_kernel(pipeline._stage_batch(sino[j:end], device), det_rotation)
            if torch.is_tensor(sino):
                sino[j:end] = rotated.to(sino.device, sino.dtype)
            else:
                sino[j:end] = rotated.cpu().numpy()
    return sino


def apply_calibration(ct_model, sino, results):
    """Apply calibration results to the model and the sinogram.

    This is the only function in the module that changes state.  A model parameter is set on
    ``ct_model`` with ``set_params``.  A detector rotation is not a model parameter, so it is applied
    by rotating every view of ``sino`` in place, one batch of views at a time, with no second
    full-size sinogram.  A rotation direction of -1 negates every view angle of the model.

    After a change to ``det_row_offset`` the cone-beam recon geometry that was derived from the
    old value stays as it was.  Re-running ``ct_model.auto_set_recon_geometry()`` afterward is the
    caller's decision, because it also resets any ``recon_shape`` the caller chose.

    Args:
        ct_model (TomographyModel): the model to change.
        sino (ndarray or tensor): the sinogram, changed in place when a rotation is applied.  A host
            array must be writable and floating point.
        results (CalibrationResult, or dict or sequence of CalibrationResult): the results to apply.
            A dict is read for its values, as :func:`calibrate_geometry` returns one.

    Returns:
        tuple: ``(ct_model, sino)``, the same two objects after the changes.

    Raises:
        ValueError: for a result whose ``parameter`` this function does not know how to apply.
    """
    if isinstance(results, CalibrationResult):
        results = [results]
    elif isinstance(results, dict):
        results = list(results.values())
    else:
        results = list(results)
    for result in results:
        if not isinstance(result, CalibrationResult):
            raise TypeError(f'apply_calibration expects CalibrationResult values; got '
                            f'{type(result).__name__}.')

    for result in results:
        name, value = result.parameter, float(result.value)
        if name in ('det_channel_offset', 'det_row_offset'):
            ct_model.set_params(**{name: value})
        elif name == 'det_rotation':
            if value != 0.0:
                _sharding.reject_shards('apply_calibration', sino=sino)
                sino = _rotate_views_in_place(sino, value)
        elif name == 'rotation_direction':
            if value == -1.0:
                # The angles are the first column of the model's per-view parameter array for cone
                # and multiaxis, and the whole array for parallel beam.  Setting the array rebuilds
                # the projectors, which hold their own copies of it.
                view_params_name = ct_model.get_params('view_params_name')
                view_params = np.array(ct_model.get_params(view_params_name), copy=True)
                if view_params.ndim == 2:
                    view_params[:, 0] *= -1
                else:
                    view_params *= -1
                ct_model.set_params(**{view_params_name: view_params})
            elif value != 1.0:
                raise ValueError(f'rotation_direction must be 1.0 or -1.0; got {value}.')
        else:
            raise ValueError(f'apply_calibration does not know how to apply {name!r}.')
    return ct_model, sino


# ── the conjugate-view method ─────────────────────────────────────────────────────────────────────

# Defaults of the conjugate-view estimator.  Each is described where it is used.
_CONJUGATE_OFFSET_HALF_RANGE_CHANNELS = 4.0     # search range on each side of the model's value
_CONJUGATE_OFFSET_MAX_SLIDES = 8                # the search window moves this many times at most
_CONJUGATE_MIN_REGION_FRACTION = 0.25           # the least fraction of channels a comparison may use
_CONJUGATE_OFFSET_TOLERANCE_CHANNELS = 0.01     # where the search stops, as a fraction of a channel
_CONJUGATE_NUM_ROWS = 16                        # detector rows compared, before the cone-beam limit
_CONJUGATE_VIEW_STRIDE = 1                      # every reference view is kept
_CONJUGATE_TRIM_FRACTION = 0.1                  # fraction of the worst view pairs dropped
_CONJUGATE_EDGE_MARGIN = 4                      # channels excluded at each edge beyond the shift
# A scan has a full rotation when no gap between neighboring view angles exceeds both this many
# times the median gap and this many radians.  The rule accepts irregular spacing, a few dropped
# views, and more than one turn, and it refuses a scan over a half rotation, whose largest gap is
# the missing half.
_CONJUGATE_MAX_GAP_RATIO = 3.0
_CONJUGATE_MAX_GAP = math.radians(5.0)


def _view_angles(ct_model):
    """The view angles in radians of a parallel or cone model, as a float64 array."""
    required, _, _ = ct_model.get_all_params()
    return np.asarray(required['angles'], dtype=np.float64).ravel()


def _angular_gaps(angles):
    """The gaps between neighboring distinct view angles on the circle, in radians."""
    wrapped = np.unique(np.mod(angles, 2 * np.pi))
    if wrapped.size < 2:
        return np.array([2 * np.pi])
    return np.diff(np.append(wrapped, wrapped[0] + 2 * np.pi))


def _angular_coverage(angles):
    """The angular range the views cover, in radians: the full circle minus the largest gap between
    neighboring distinct angles.  A full rotation gives 2 pi minus one view spacing, and views over
    a half rotation give about pi."""
    return float(2 * np.pi - _angular_gaps(angles).max())


def _require_conjugate_geometry(ct_model, parameter, det_rotation=0.0):
    """Refuse the geometries the conjugate-view method cannot serve, with the reason."""
    kind = _geometry_kind(ct_model)
    if kind == 'multiaxis':
        raise ValueError('The conjugate-view method does not support a multiaxis parallel model yet.')
    if _is_helical(ct_model):
        raise ValueError('The conjugate-view method needs an opposite view at the same axial '
                         'position, which a helical scan does not have.')
    gaps = _angular_gaps(_view_angles(ct_model))
    if gaps.max() > max(_CONJUGATE_MAX_GAP_RATIO * np.median(gaps), _CONJUGATE_MAX_GAP):
        raise ValueError('The conjugate-view method needs views over a full rotation.  The angles '
                         f'cover {math.degrees(2 * np.pi - gaps.max()):.1f} degrees, with a gap of '
                         f'{math.degrees(gaps.max()):.1f} degrees between neighboring views.')
    if det_rotation != 0.0:
        _check_det_rotation(det_rotation)
        if kind == 'cone' and ct_model.get_params('use_curved_detector'):
            raise ValueError('A det_rotation cannot be applied to a curved detector.  The rotation '
                             'resamples a flat detector plane.')
    if parameter == 'det_rotation' and kind == 'cone' and ct_model.get_params('use_curved_detector'):
        raise ValueError('det_rotation cannot be estimated on a curved detector.  The rotation '
                         'resamples a flat detector plane.')


def _fourier_shift_channels(array, shift, spectrum=None):
    """Shift every row of ``array`` along its last axis by ``shift`` samples.

    The shift is applied in the Fourier domain, which is exact for a band-limited signal.  It is
    circular, so the samples that leave one end of a row enter at the other.  Positive shifts move
    content toward higher channel indices.  A caller that shifts the same array many times passes
    its precomputed ``spectrum``, the result of ``np.fft.rfft(array, axis=-1)``.
    """
    if shift == 0.0 and spectrum is None:
        return array
    num_channels = array.shape[-1]
    if spectrum is None:
        spectrum = np.fft.rfft(array, axis=-1)
    phase = np.exp(-2j * np.pi * np.fft.rfftfreq(num_channels) * shift).astype(np.complex64)
    return np.fft.irfft(spectrum * phase, n=num_channels, axis=-1).astype(np.float32)


class _ConjugatePairs:
    """The data behind a conjugate-view score.

    An instance holds a band of detector rows from every kept view.  For each kept view and
    channel it records which view holds the opposite ray.  The opposite of the ray at view angle
    ``beta`` and fan angle ``gamma`` lies at view angle ``beta + pi - 2 * gamma`` and fan angle
    ``-gamma``, in the sign conventions of ``cone_beam._cone_pixel_xy_mag``.  Parallel beam is the
    case ``gamma = 0``.  The partner view is interpolated linearly between the two kept views
    nearest that angle.

    The reference views are every view at the record's view stride, and their partners are drawn
    from every view.  A stride therefore thins the references without moving any partner, so it
    saves memory at the cost of fewer pairs and does not blur the partners.  Every view is a
    reference at stride 1, so each unordered pair is compared from both sides.  Comparing each pair
    from one side only, with the references limited to a half rotation, raised the first-pass error
    on an off-axis rod from 0.03 to 0.3 channels, because the interpolation of a partner view
    errs in opposite directions on the two sides and the two cancel.  The memory held is one band
    for the references, one for their opposites, one for the partner views, and the spectrum of
    the opposites.

    The fan angle of a channel depends on the channel offset.  The partners are computed once, at
    ``pairing_offset``, and a candidate offset ``d`` channels away moves a channel's partner angle
    by ``2 d delta / sdd``.  The estimator therefore makes a second pass with the pairs rebuilt
    at its first estimate.

    Args:
        ct_model: a parallel or cone model.
        reduction (dict or None): a record from :func:`build_reduced_problem`, whose view stride,
            bin factor, and row window are used.  None builds a record that keeps every view,
            bins nothing, and takes a band of rows around the row that the central plane of the
            scan reaches.
        num_rows (int or None): the band height when ``reduction`` is None.  None takes the
            default, reduced for cone beam so that the opposite rays through the band land within
            about one row of each other across the support.
        pairing_offset (float or None): the channel offset in ALU that the fan angles are computed
            at.  None is the model's current value.
    """

    def __init__(self, ct_model, reduction=None, num_rows=None, pairing_offset=None):
        self.kind = _geometry_kind(ct_model)
        num_views, num_det_rows, num_det_channels = (int(s) for s in ct_model.get_params('sinogram_shape'))
        delta_det_channel, delta_det_row, det_channel_offset, det_row_offset = ct_model.get_params(
            ['delta_det_channel', 'delta_det_row', 'det_channel_offset', 'det_row_offset'])
        if reduction is None:
            reduction = self._default_reduction(ct_model, num_rows)
        elif self.kind == 'cone':
            # A caller's row window is used as given.  The cone-beam comparison is only sound
            # near the central plane, so a window that leaves that plane out gets a warning.
            central_row = (num_det_rows - 1) / 2.0 + det_row_offset / delta_det_row
            lo, hi = reduction['row_window']
            if not lo <= central_row < hi:
                warnings.warn('The reduction\'s row window does not contain the row the central '
                              'plane reaches, so the cone-beam conjugate comparison is biased '
                              'by the cone angle.')
        self.reduction = dict(reduction)
        stride, bin_factor = int(reduction['view_stride']), int(reduction['bin_factor'])
        self.delta = bin_factor * float(delta_det_channel)
        self.num_channels = num_det_channels // bin_factor
        self.model_offset = float(det_channel_offset)
        self.pairing_offset = self.model_offset if pairing_offset is None else float(pairing_offset)

        # The reference views are every stride-th view, and their partners come from every view.
        angles = _view_angles(ct_model)
        self.reference_indices = np.arange(0, num_views, stride)
        self.num_views = int(self.reference_indices.size)
        # The record describes the reference views, so that reduce_sinogram with this record and
        # the difference image of conjugate_difference have the same shape.
        self.reduction['view_indices'] = self.reference_indices
        self.reduction['sinogram_shape'] = (self.num_views,) + tuple(self.reduction['sinogram_shape'][1:])

        # The opposite ray's view angle, per reference view and per channel of the mirrored
        # opposite.  Column m of the mirrored array holds channel 2c - m of the partner view, whose
        # detector coordinate is u = (c - m) delta - d, and the ray there is the opposite of a
        # reference ray at fan angle -gamma(u).  The partner therefore lies at
        # beta + pi + 2 gamma(u), which is beta + pi - 2 gamma((m - c) delta + d).
        center_channel = (self.num_channels - 1) / 2.0
        u = (np.arange(self.num_channels) - center_channel) * self.delta + self.pairing_offset
        if self.kind == 'cone':
            source_detector_dist = float(ct_model.get_params('source_detector_dist'))
            if np.isinf(source_detector_dist):
                gamma = np.zeros_like(u)
            elif ct_model.get_params('use_curved_detector'):
                gamma = u / source_detector_dist
            else:
                gamma = np.arctan(u / source_detector_dist)
        else:
            gamma = np.zeros_like(u)
        target = angles[self.reference_indices][:, None] + np.pi - 2.0 * gamma[None, :]
        low, high, self.partner_weight = self._partners(angles, target)
        # The partner views are read once, as a compact set, and the partner indices are remapped
        # into that set.
        self.partner_indices = np.unique(np.concatenate([low.ravel(), high.ravel()]))
        self.partner_low = np.searchsorted(self.partner_indices, low)
        self.partner_high = np.searchsorted(self.partner_indices, high)

    @staticmethod
    def _default_reduction(ct_model, num_rows):
        """A reduction record for the band of rows around the scan's central plane."""
        num_views, num_det_rows, num_det_channels = (int(s) for s in ct_model.get_params('sinogram_shape'))
        delta_det_row, det_row_offset = ct_model.get_params(['delta_det_row', 'det_row_offset'])
        if num_rows is None:
            num_rows = _CONJUGATE_NUM_ROWS
            if isinstance(ct_model, ConeBeamModel):
                # Opposite rays through a point off the central plane reach the detector at
                # different heights.  At a distance r from the axis and a height of m rows the
                # difference is about m * 2 r / sid rows, so the band is limited to the rows where
                # that difference stays within one row.
                min_mag, _ = ct_model.pixel_magnification_bounds()
                source_detector_dist, source_iso_dist = ct_model.get_params(
                    ['source_detector_dist', 'source_iso_dist'])
                if not np.isinf(source_detector_dist):
                    support_radius = source_detector_dist / min_mag - source_iso_dist
                    half = max(1, math.floor(source_iso_dist / (2.0 * support_radius)))
                    num_rows = min(num_rows, 2 * half + 1)
        num_rows = max(1, min(int(num_rows), num_det_rows))
        # The row the central plane reaches is where the detector height v is zero, and the band
        # is centered on it.
        central_row = (num_det_rows - 1) / 2.0 + det_row_offset / delta_det_row
        lo = int(round(central_row - (num_rows - 1) / 2.0))
        lo = max(0, min(lo, num_det_rows - num_rows))
        stride = _CONJUGATE_VIEW_STRIDE
        return {'geometry': _geometry_kind(ct_model), 'view_stride': stride, 'bin_factor': 1,
                'row_window': (lo, lo + num_rows), 'axial_thinning': True,
                'full_sinogram_shape': (num_views, num_det_rows, num_det_channels),
                'sinogram_shape': (num_views // stride, num_rows, num_det_channels),
                'devices': [str(ct_model.torch_device)]}

    @staticmethod
    def _partners(angles, target):
        """For each target angle, the two kept views that bracket it on the circle and the weight
        of the second.  Returns three arrays of the target's shape."""
        wrapped = np.mod(angles, 2 * np.pi)
        order = np.argsort(wrapped)
        sorted_angles = wrapped[order]
        num_views = angles.size
        t = np.mod(target, 2 * np.pi)
        position = np.searchsorted(sorted_angles, t)
        high = position % num_views
        low = (position - 1) % num_views
        gap = np.mod(sorted_angles[high] - sorted_angles[low], 2 * np.pi)
        gap = np.where(gap == 0.0, 2 * np.pi, gap)
        weight = np.mod(t - sorted_angles[low], 2 * np.pi) / gap
        return order[low], order[high], weight.astype(np.float32)

    def read_bands(self, sino, det_rotation=0.0):
        """The band of the reference views and the band of the partner views."""
        partner = dict(self.reduction, view_indices=self.partner_indices)
        return (reduce_sinogram(sino, self.reduction, det_rotation=det_rotation),
                reduce_sinogram(sino, partner, det_rotation=det_rotation))

    def pairs(self, sino, det_rotation=0.0, bands=None):
        """The band of every reference view, and the mirrored opposite ray of every element of it.

        Args:
            sino: the full sinogram, read through :func:`reduce_sinogram` unless ``bands`` is given.
            det_rotation (float): a rotation applied by :func:`reduce_sinogram`, bilinear.
            bands (tuple of ndarray, optional): the reference and partner bands already read, as
                the rotation estimate supplies them.

        Returns:
            tuple of ndarray: ``(views, opposites)``, each of shape ``(num_reference_views,
            num_rows, num_channels)`` in float32.  Element ``[i, r, n]`` of ``opposites`` is the
            measurement of the ray opposite to element ``[i, r, n]`` of ``views``, placed at the
            mirrored channel.  The two agree up to a shift of twice the channel offset.
        """
        views, partners = self.read_bands(sino, det_rotation) if bands is None else bands
        mirrored = partners[:, :, ::-1]
        opposites = np.empty_like(views)
        columns = np.arange(self.num_channels)
        for i in range(self.num_views):
            low = mirrored[self.partner_low[i], :, columns]      # (channels, rows)
            high = mirrored[self.partner_high[i], :, columns]
            weight = self.partner_weight[i][:, None]
            opposites[i] = ((1.0 - weight) * low + weight * high).T
        return views, opposites

    def channel_margin(self, max_abs_offset):
        """Channels excluded at each edge of the comparison for offsets up to ``max_abs_offset``."""
        return int(math.ceil(2.0 * abs(max_abs_offset) / self.delta)) + _CONJUGATE_EDGE_MARGIN

    def prepare(self, views, opposites, margin):
        """What the score needs from a pair set, computed once for every candidate.

        Returns:
            dict: the interior channel region, the views over it, their per-pair mean square, and
            the spectrum of the opposites along the channel axis.
        """
        region = slice(margin, self.num_channels - margin)
        if region.stop <= region.start:
            raise ValueError(f'A margin of {margin} channels at each edge leaves none of the '
                             f'{self.num_channels} channels to compare.  The margin follows the largest '
                             'offset the search can reach; narrow the bounds or center them nearer zero.')
        interior = np.ascontiguousarray(views[:, :, region], dtype=np.float32)
        energy = np.mean(interior.astype(np.float64) ** 2, axis=(1, 2))
        if not np.any(energy > 0.0):
            raise ValueError('The compared band of the sinogram is zero, so no score exists.')
        return {'region': region, 'views': interior, 'energy': energy,
                'spectrum': np.fft.rfft(opposites, axis=-1)}

    def per_pair(self, prepared, opposites, det_channel_offset):
        """The mean squared difference between each view and its shifted opposites, per pair."""
        shift = 2.0 * float(det_channel_offset) / self.delta
        shifted = _fourier_shift_channels(opposites, shift, spectrum=prepared['spectrum'])
        difference = prepared['views'] - shifted[:, :, prepared['region']]
        return np.mean(difference.astype(np.float64) ** 2, axis=(1, 2))

    def keep_set(self, prepared, opposites, det_channel_offset):
        """The pairs kept by the trimmed mean: all but the fraction that agree worst at one offset,
        with each pair's difference measured against its own energy, so that the views with the
        most object in them are not the ones dropped.  The set is chosen once, so that every
        candidate is scored on the same pairs."""
        relative = self.per_pair(prepared, opposites, det_channel_offset) / np.maximum(prepared['energy'], 1e-30)
        num_kept = max(1, int(round(relative.size * (1.0 - _CONJUGATE_TRIM_FRACTION))))
        return np.sort(np.argsort(relative)[:num_kept])

    def score(self, prepared, opposites, det_channel_offset, keep):
        """The conjugate-view score at one channel offset: the mean squared difference over the
        kept pairs divided by the mean square of their views."""
        per_pair = self.per_pair(prepared, opposites, det_channel_offset)
        return float(per_pair[keep].mean() / prepared['energy'][keep].mean())


def _search_minimum(score_fn, bounds, num_coarse, tolerance):
    """Find the minimum of a scalar score over ``bounds``.

    A coarse pass evaluates ``num_coarse`` equally spaced candidates, which shows whether the curve
    has one minimum.  A golden-section search then narrows the bracket around the coarse minimum
    until it is shorter than ``tolerance``.  Every evaluation is kept.

    Returns:
        tuple: ``(best, candidates, scores, notes)``.  ``candidates`` and ``scores`` are sorted by
        candidate and hold every evaluation.  ``notes`` is a list of strings describing anything the
        caller should warn about: a coarse minimum at an edge of the bounds, or more than one
        local minimum on the coarse curve.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not hi > lo:
        raise ValueError(f'bounds must satisfy lo < hi; got {bounds}.')
    num_coarse = max(3, int(num_coarse))
    coarse = np.linspace(lo, hi, num_coarse)
    evaluated = {float(x): float(score_fn(x)) for x in coarse}
    coarse_scores = np.array([evaluated[float(x)] for x in coarse])
    notes = []
    best = int(np.argmin(coarse_scores))
    interior = coarse_scores[1:-1]
    local_minima = int(np.sum((interior < coarse_scores[:-2]) & (interior <= coarse_scores[2:])))
    if local_minima > 1:
        notes.append(f'the score curve has {local_minima} local minima on the coarse grid')
    if best in (0, num_coarse - 1):
        notes.append('the coarse minimum sits at an edge of the bounds')
    a = float(coarse[max(best - 1, 0)])
    b = float(coarse[min(best + 1, num_coarse - 1)])

    # Golden-section search on [a, b].  Each step drops the worse end and keeps one interior point.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = b - ratio * (b - a)
    x2 = a + ratio * (b - a)
    f1, f2 = float(score_fn(x1)), float(score_fn(x2))
    evaluated[x1], evaluated[x2] = f1, f2
    while b - a > tolerance:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - ratio * (b - a)
            f1 = float(score_fn(x1))
            evaluated[x1] = f1
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + ratio * (b - a)
            f2 = float(score_fn(x2))
            evaluated[x2] = f2
    candidates = np.array(sorted(evaluated))
    scores = np.array([evaluated[x] for x in candidates])
    return float(candidates[int(np.argmin(scores))]), candidates, scores, notes


def estimate_det_channel_offset(ct_model, sino, *, method='auto', bounds=None, num_coarse=11,
                                reduction=None, det_rotation=0.0, num_rows=None):
    """Estimate ``det_channel_offset`` from the sinogram by comparing each view with its opposite.

    In a scan over a full rotation every ray is measured twice, once from each side.  A voxel at
    in-plane position x projects to channel ``(x + det_channel_offset) / delta_det_channel`` from
    the detector center.  After a half rotation it projects to the mirrored position, so a view
    and its mirrored opposite differ by a shift of twice the offset.  The estimator scores
    candidate offsets by that shift and returns the one at which the views and their opposites
    agree best.

    For cone beam each channel's opposite ray lies at a view angle that depends on the fan angle,
    and the method pairs each channel with that view, interpolated between the two nearest.
    Opposite rays through points off the central plane reach the detector at different heights.
    The cone-beam comparison therefore uses a band of rows around the central plane, and the
    estimate degrades as the fan angle and the cone angle grow.  On synthetic data at a full fan
    angle of 20 degrees the bias was 0.02 to 0.03 channels.

    The search evaluates ``num_coarse`` candidates across ``bounds``, then narrows the bracket
    around the best one by golden section to a hundredth of a channel, in about 24 evaluations.
    Every candidate and score of that search is returned, so a flat or double minimum is visible.
    The function warns when the coarse curve has more than one minimum or its minimum sits at an
    edge of the bounds.  A trimmed mean over view pairs drops the tenth of the pairs that agree
    worst at the best candidate of a first coarse grid, which costs ``num_coarse`` more
    evaluations, so a few corrupted views do not move the estimate.

    Three limits apply.  The comparison holds the kept views' band, its mirrored opposites, and the
    spectrum of the opposites, which is about three times the band in memory, so the view stride
    and the band height bound it.  An offset scan, whose detector is displaced by hundreds of
    channels, is not served: the search range is a few channels and the comparison excludes only
    the channels the shift wraps.  A sinogram in the divided device form is refused.

    Args:
        ct_model (TomographyModel): a parallel or cone model.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.
        method (str, optional): ``'auto'`` or ``'conjugate'``.  Both select the conjugate-view
            method.  A scan the method cannot serve raises; a later method will be the fallback.
            Defaults to ``'auto'``.
        bounds (tuple of float, optional): the search range in ALU.  None (the default) is a window
            of four channels on each side of the model's current value.  When the coarse minimum
            sits at an edge of that window, the window moves to center on the edge, at the same
            width, up to eight times, so the default reaches offsets of about 36 channels.  The
            window stops moving when the channels excluded for the circular shift would leave less
            than a quarter of the detector to compare.  A range given here is not moved.
        num_coarse (int, optional): candidates in the coarse pass.  Defaults to 11.
        reduction (dict, optional): a record from :func:`build_reduced_problem` whose view stride,
            bin factor, and row window the comparison uses.  None (the default) keeps every view
            at full resolution over a band of rows around the central plane.
        det_rotation (float, optional): a detector rotation in radians applied to the views before
            the comparison.  Defaults to 0.0.
        num_rows (int, optional): the band height when ``reduction`` is None.

    Returns:
        CalibrationResult: ``parameter`` is ``'det_channel_offset'``, ``value`` is the estimate in
        ALU, and ``method`` is ``'conjugate'``.  ``reduction`` records the rows and views compared,
        the channels excluded at each edge, the rotation applied, the pairs kept, and the search
        notes.

    Raises:
        ValueError: for a multiaxis model, a helical scan, views that do not cover a full
            rotation, or a rotation on a curved detector.
    """
    if method not in ('auto', 'conjugate'):
        raise ValueError(f"estimate_det_channel_offset supports method 'auto' or 'conjugate'; got "
                         f"{method!r}.")
    _require_conjugate_geometry(ct_model, 'det_channel_offset', det_rotation)
    _sharding.reject_shards('estimate_det_channel_offset', sino=sino)
    problem = _ConjugatePairs(ct_model, reduction, num_rows)
    user_bounds = bounds
    if bounds is None:
        half_range = _CONJUGATE_OFFSET_HALF_RANGE_CHANNELS * problem.delta
        bounds = (problem.model_offset - half_range, problem.model_offset + half_range)
    margin = problem.channel_margin(max(abs(bounds[0]), abs(bounds[1])))
    tolerance = _CONJUGATE_OFFSET_TOLERANCE_CHANNELS * problem.delta

    def search(problem):
        """One pass over the current bounds: choose the kept pairs at the best candidate of a
        coarse grid scored on every pair, then search on that fixed set.  The bands are read per
        pass, because the set of partner views depends on the pairing offset."""
        views, opposites = problem.pairs(sino, det_rotation=det_rotation)
        prepared = problem.prepare(views, opposites, margin)
        every_pair = np.arange(problem.num_views)
        coarse = np.linspace(bounds[0], bounds[1], max(3, int(num_coarse)))
        coarse_best = coarse[int(np.argmin([problem.score(prepared, opposites, x, every_pair)
                                            for x in coarse]))]
        keep = problem.keep_set(prepared, opposites, float(coarse_best))
        return _search_minimum(lambda offset: problem.score(prepared, opposites, offset, keep),
                               bounds, num_coarse, tolerance) + (keep,)

    # A coarse minimum at an edge of the window means the window is in the wrong place, so the
    # window, at its fixed width, moves to center on that edge and the search repeats, up to a
    # limit.  The width is kept so that the coarse grid keeps its spacing.  The default window of
    # four channels on each side does not hold the offset of an uncalibrated scan in general; a
    # 7.5 channel offset on a 512 channel detector needed one move.  The channels excluded at each
    # edge grow with the largest offset in the window, and the window stops moving when they
    # would leave less than a quarter of the channels to compare.
    best, candidates, scores, notes, keep = search(problem)
    slides = 0
    while ('the coarse minimum sits at an edge of the bounds' in notes and user_bounds is None
           and slides < _CONJUGATE_OFFSET_MAX_SLIDES):
        half_width = 0.5 * (bounds[1] - bounds[0])
        moved = (best - half_width, best + half_width)
        moved_margin = problem.channel_margin(max(abs(moved[0]), abs(moved[1])))
        if problem.num_channels - 2 * moved_margin < _CONJUGATE_MIN_REGION_FRACTION * problem.num_channels:
            notes.append('the search window could not move further, because the channels excluded '
                         'for the circular shift would leave less than a quarter of the detector')
            break
        bounds, margin = moved, moved_margin
        best, candidates, scores, notes, keep = search(problem)
        slides += 1

    # Two passes for cone beam.  The fan angle of a channel, and so its partner view, depends on
    # the offset, and the first pass pairs at the model's value.  The second pass pairs at the
    # first estimate.  On synthetic data at a 20 degree fan the first pass alone erred by about
    # one percent of the offset, and the second pass removed that trend; it doubles the cost.
    first_pass = best
    if problem.kind == 'cone' and abs(best - problem.pairing_offset) > tolerance:
        problem = _ConjugatePairs(ct_model, reduction, num_rows, pairing_offset=best)
        best, candidates, scores, notes, keep = search(problem)
    for note in notes:
        warnings.warn(f'estimate_det_channel_offset: {note}.')
    record = dict(problem.reduction, num_pairs=problem.num_views, pairs_kept=int(keep.size),
                  channel_margin=margin, det_rotation=float(det_rotation),
                  pairing_offset=problem.pairing_offset, first_pass=first_pass,
                  bounds=(float(bounds[0]), float(bounds[1])), search_notes=notes)
    return CalibrationResult(parameter='det_channel_offset', value=best,
                             score=float(scores[np.searchsorted(candidates, best)]),
                             candidates=candidates, scores=scores, method='conjugate',
                             reduction=record)


# Defaults of the rotation estimate.  The search covers the five degree cap on each side of zero
# and stops at this many radians.  Below this edge displacement, in pixels, the estimate is in the
# regime where the resampling of a candidate angle biases it, and the function warns.  On the
# cluster the error was 4.5 percent at 0.89 pixels and under 0.5 percent from 1.34 pixels upward.
_CONJUGATE_ROTATION_TOLERANCE = math.radians(0.005)
_CONJUGATE_MIN_EDGE_DISPLACEMENT = 1.0


def _rotated_band(sino, reduction, det_rotation):
    """The band of rows named by ``reduction``, rotated by ``det_rotation`` about the full
    detector's center with cubic interpolation.

    The band is read from the sinogram with a margin of rows on each side, so the rotation samples
    nothing outside the rows it has, and it is cropped afterward.  The cubic kernel is used because
    the bilinear one smooths the data by an amount that grows with the angle, which biases a
    search over the angle toward its bounds on cone-beam data.  The cubic kernel's bias is 10 to 24
    percent of the angle when the rotation displaces the edge pixel by less than half a pixel, a
    few percent up to one pixel, and under 0.5 percent beyond that; the measurement is recorded
    with the plans for this feature.
    """
    import cv2
    num_views, num_rows, num_channels = reduction['full_sinogram_shape']
    row_lo, row_hi = reduction['row_window']
    center_row = (num_rows - 1) / 2.0
    margin = _rotation_row_margin(det_rotation, max(abs(row_lo - center_row), abs(row_hi - 1 - center_row)),
                                  num_channels)
    band_lo, band_hi = max(0, row_lo - margin), min(num_rows, row_hi + margin)
    wide = dict(reduction, row_window=(band_lo, band_hi))
    wide['sinogram_shape'] = (reduction['sinogram_shape'][0], (band_hi - band_lo) // reduction['bin_factor'],
                              reduction['sinogram_shape'][2])
    band = reduce_sinogram(sino, wide)
    if det_rotation == 0.0:
        return band[:, row_lo - band_lo:row_hi - band_lo, :]
    bin_factor = reduction['bin_factor']
    matrix = cv2.getRotationMatrix2D(((band.shape[2] - 1) / 2.0, (center_row - band_lo) / bin_factor),
                                     math.degrees(det_rotation), 1.0)
    out = np.empty((band.shape[0], (row_hi - row_lo) // bin_factor, band.shape[2]), dtype=np.float32)
    for i in range(band.shape[0]):
        rotated = cv2.warpAffine(band[i], matrix, (band.shape[2], band.shape[1]), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[i] = rotated[(row_lo - band_lo) // bin_factor:(row_hi - band_lo) // bin_factor]
    return out


def estimate_det_rotation(ct_model, sino, *, method='auto', bounds=None, num_coarse=11,
                          reduction=None, det_channel_offset=None, num_rows=None):
    """Estimate the detector rotation, in radians, by comparing each view with its opposite.

    A detector rotated by an angle about the optical axis records every view rotated by that
    angle.  Mirroring a view in channels reverses the sign of that rotation, so a view and its
    mirrored opposite differ by twice the angle.  Each candidate angle is applied to a band of
    rows from every kept view by cubic resampling about the detector center, the views are paired
    with their opposites as in :func:`estimate_det_channel_offset`, and the candidate at which
    they agree best is returned.  The comparison shifts the opposites by twice the channel
    offset, which is the model's current value unless ``det_channel_offset`` is given, so
    estimate the offset first.

    Resampling the band at a candidate angle smooths it, and the smoothing biases the estimate
    when the rotation displaces the edge pixel of the detector by less than about one pixel.  On
    synthetic data at 512 and 1024 channels the cubic kernel's bias was 10 to 24 percent of the
    angle below half a pixel of edge displacement, 4.5 percent at 0.89 pixels, and under 0.5
    percent from 1.34 pixels upward.  The function warns when the estimate displaces the edge
    pixel by less than one pixel.  A rotation handled inside the projectors would need no
    resampling.

    Args:
        ct_model (TomographyModel): a parallel or flat-detector cone model.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.
        method (str, optional): ``'auto'`` or ``'conjugate'``.  Defaults to ``'auto'``.
        bounds (tuple of float, optional): the search range in radians, within five degrees of
            zero.  None (the default) is the full five degrees on each side.
        num_coarse (int, optional): candidates in the coarse pass.  Defaults to 11.
        reduction (dict, optional): as in :func:`estimate_det_channel_offset`.
        det_channel_offset (float, optional): the channel offset in ALU used in the comparison.
            None (the default) is the model's current value.
        num_rows (int, optional): the band height when ``reduction`` is None.

    Returns:
        CalibrationResult: ``parameter`` is ``'det_rotation'``, ``value`` is the angle in radians
        that :func:`apply_calibration` should apply, and ``method`` is ``'conjugate'``.

    Raises:
        ValueError: for a multiaxis model, a helical scan, a curved detector, views that do not
            cover a full rotation, or bounds beyond the five degree cap.
    """
    if method not in ('auto', 'conjugate'):
        raise ValueError(f"estimate_det_rotation supports method 'auto' or 'conjugate'; got {method!r}.")
    _require_conjugate_geometry(ct_model, 'det_rotation')
    _sharding.reject_shards('estimate_det_rotation', sino=sino)
    if bounds is None:
        bounds = (-_MAX_DET_ROTATION, _MAX_DET_ROTATION)
    _check_det_rotation(max(abs(bounds[0]), abs(bounds[1])))
    problem = _ConjugatePairs(ct_model, reduction, num_rows, pairing_offset=det_channel_offset)
    offset = problem.pairing_offset
    margin = problem.channel_margin(offset)

    def pairs_at(det_rotation):
        partner = dict(problem.reduction, view_indices=problem.partner_indices)
        return problem.pairs(sino, bands=(_rotated_band(sino, problem.reduction, float(det_rotation)),
                                          _rotated_band(sino, partner, float(det_rotation))))

    # The pairs kept by the trimmed mean are chosen at the model's own geometry, with no
    # rotation applied, and every candidate is scored on that set.
    views, opposites = pairs_at(0.0)
    prepared = problem.prepare(views, opposites, margin)
    keep = problem.keep_set(prepared, opposites, offset)

    def score_at(det_rotation):
        views, opposites = pairs_at(det_rotation)
        return problem.score(problem.prepare(views, opposites, margin), opposites, offset, keep)

    best, candidates, scores, notes = _search_minimum(score_at, bounds, num_coarse,
                                                      _CONJUGATE_ROTATION_TOLERANCE)
    edge_displacement = abs(best) * problem.num_channels / 2.0
    if edge_displacement < _CONJUGATE_MIN_EDGE_DISPLACEMENT:
        notes.append(f'the estimate displaces the edge channels by {edge_displacement:.2f} pixels, '
                     'where the resampling of each candidate biases it by up to 25 percent of the angle')
    for note in notes:
        warnings.warn(f'estimate_det_rotation: {note}.')
    record = dict(problem.reduction, num_pairs=problem.num_views, pairs_kept=int(keep.size),
                  channel_margin=margin, det_channel_offset=offset, search_notes=notes)
    return CalibrationResult(parameter='det_rotation', value=best,
                             score=float(scores[np.searchsorted(candidates, best)]),
                             candidates=candidates, scores=scores, method='conjugate',
                             reduction=record)


def conjugate_difference(ct_model, sino, *, det_channel_offset=None, det_rotation=0.0,
                         reduction=None, num_rows=None):
    """The difference between each view and its mirrored opposite, as an image stack for viewing.

    The conjugate-view score is a normalized mean square of this image, so the stack shows what
    that number summarizes.  At the true channel offset and rotation the difference holds noise
    and the residue of the fan and cone angles.  A wrong offset shows as doubled edges displaced
    along the channel axis.  A wrong rotation shows as edges displaced vertically, by an amount
    that grows toward the edge channels.  The shift is circular, so the channels within a few
    samples of the edges are not meaningful.

    Args:
        ct_model (TomographyModel): a parallel or cone model.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.
        det_channel_offset (float, optional): the offset in ALU to compare at.  None (the default)
            is the model's current value.
        det_rotation (float, optional): the rotation in radians applied before the comparison.
            Defaults to 0.0.
        reduction (dict, optional): as in :func:`estimate_det_channel_offset`.
        num_rows (int, optional): the band height when ``reduction`` is None.

    Returns:
        ndarray: float32 of shape ``(num_kept_views, num_rows, num_channels)``.
    """
    _require_conjugate_geometry(ct_model, 'det_channel_offset', det_rotation)
    _sharding.reject_shards('conjugate_difference', sino=sino)
    problem = _ConjugatePairs(ct_model, reduction, num_rows, pairing_offset=det_channel_offset)
    offset = problem.pairing_offset
    views, opposites = problem.pairs(sino, det_rotation=det_rotation)
    return views - _fourier_shift_channels(opposites, 2.0 * offset / problem.delta)
