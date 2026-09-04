"""Geometric calibration from the sinogram.

The functions here run after a scanner reader's ``get_sino_and_model`` and before
reconstruction.  They estimate scan geometry that the vendor metadata got wrong or left out, and
they show a user the evidence behind an estimate.  This module includes the reduced problem
that every estimator runs on, a parameter sweep that reconstructs one slice per candidate value,
a rotation-direction check, and the one function that applies a result.

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
           'check_rotation_direction', 'apply_calibration']

# The parameters parameter_sweep accepts.  'det_rotation' is not a model parameter.  It is applied
# by resampling the sinogram, and the sweep does that per candidate.
_SWEEP_PARAMETERS = ('det_channel_offset', 'det_row_offset', 'det_rotation')

# Views of the full sinogram read per step when a reduced sinogram is built.  One step's block
# holds this many views by the kept rows, plus the rotation margin when there is one, by every
# channel, so the transient stays small at any sinogram size.
_REDUCE_VIEW_BATCH = 64

# Views rotated per step when a detector rotation is applied in place.
_ROTATE_VIEW_BATCH = 30


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
        reduction (dict): the record returned by :func:`build_reduced_problem`.
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
    kept_views, kept_rows = num_views // stride, row_hi - row_lo
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
            block = pipeline._stage_batch(sino[k0 * stride:k1 * stride:stride, band_lo:band_hi, :],
                                          device)
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
