"""Geometry calibration from the sinogram.

Two functions estimate scan geometry that the vendor metadata got wrong or left out: the detector
channel offset, which sets the center of rotation, and the rotation of the detector about the
optical axis.  Each returns one number and changes nothing.  The caller sets the offset with
``set_params`` and removes the rotation with ``correct_det_rotation``.  The third function,
``align_sino_views``, removes the small per-view shifts that remain after the geometry is set.

The order of preprocessing matters.  Run in this order:
 1. defective-pixel interpolation, background offset correction, and stripe removal
 2. the two estimators
 3. ``align_sino_views``.
Stripe removal comes first because a gain stripe sits at a fixed channel, and a geometry estimate
would take it for a feature of the object.  ``align_sino_views`` comes last because a wrong
``det_channel_offset`` looks like a per-view shift, which aligning first would partly remove.
"""

import math
import warnings

import numpy as np
import torch

from .. import _sharding
from ..cone_beam import ConeBeamModel
from ..multiaxis_parallel import MultiAxisParallelModel
from ..parallel_beam import ParallelBeamModel
from . import _pipeline

__all__ = ['estimate_det_channel_offset', 'estimate_det_rotation', 'align_sino_views']

# This many views of the full sinogram are read per step when a band of rows is read.
_READ_VIEW_BATCH = 64

# The rotation estimate searches within this angle of zero, in radians.  A detector tilt is a small
# correction, and the resampling that applies it degrades with the angle.
_MAX_DET_ROTATION = math.radians(5.0)

_OFFSET_HALF_RANGE_CHANNELS = 4.0     # search range on each side of the model's value
_OFFSET_MAX_SLIDES = 8                # the search window moves this many times at most
_OFFSET_TOLERANCE_CHANNELS = 0.01     # where the search stops, as a fraction of a channel
_MIN_REGION_FRACTION = 0.25           # the least fraction of channels a comparison may use
_NUM_COARSE = 11                      # candidates scored on the coarse grid of a search
_NUM_ROWS = 16                        # detector rows compared, before the cone-beam limit
_TRIM_FRACTION = 0.1                  # fraction of the worst view pairs dropped
_EDGE_MARGIN = 4                      # channels excluded at each edge beyond the shift

# A scan covers a full rotation when no gap between neighboring view angles exceeds both this
# many times the median gap and this many radians.  A scan over a half rotation is refused.
_MAX_GAP_RATIO = 3.0
_MAX_GAP = math.radians(5.0)

# The rotation search stops when its bracket is shorter than the first constant, in radians.
# The second is an edge displacement in pixels, below which the function warns.
_ROTATION_TOLERANCE = math.radians(0.005)
_MIN_EDGE_DISPLACEMENT = 1.0


# ── geometry checks ───────────────────────────────────────────────────────────────────────────────

def _geometry_kind(ct_model):
    """Classify a model as 'parallel', 'cone', or 'multiaxis', or raise for anything else.

    The translation geometry has no rotation, so neither quantity estimated here applies to it,
    and it is refused by name.
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


def _require_conjugate_geometry(ct_model, parameter):
    """Refuse the geometries the opposite-view comparison cannot serve, with the reason."""
    kind = _geometry_kind(ct_model)
    if kind == 'multiaxis':
        raise ValueError(f'{parameter} cannot be estimated for a multiaxis parallel model yet.')
    if _is_helical(ct_model):
        raise ValueError(f'{parameter} is estimated by comparing each view with the opposite view at '
                         'the same axial position, which a helical scan does not have.')
    gaps = _angular_gaps(_view_angles(ct_model))
    if gaps.max() > max(_MAX_GAP_RATIO * np.median(gaps), _MAX_GAP):
        raise ValueError(f'{parameter} is estimated by comparing each view with its opposite, which '
                         f'needs views over a full rotation.  The angles cover '
                         f'{math.degrees(2 * np.pi - gaps.max()):.1f} degrees, with a gap of '
                         f'{math.degrees(gaps.max()):.1f} degrees between neighboring views.')
    if parameter == 'det_rotation' and kind == 'cone' and ct_model.get_params('use_curved_detector'):
        raise ValueError('det_rotation cannot be estimated on a curved detector.  The rotation '
                         'resamples a flat detector plane.')


# ── reading a band of rows ────────────────────────────────────────────────────────────────────────

def _read_band(sino, view_indices, row_window, device):
    """The rows ``row_window`` of the views ``view_indices``, as a float32 host array of shape
    ``(len(view_indices), rows, channels)``."""
    row_lo, row_hi = row_window
    view_indices = np.asarray(view_indices, dtype=np.int64)
    out = np.empty((view_indices.size, row_hi - row_lo, sino.shape[2]), dtype=np.float32)
    with torch.no_grad():
        for k0 in range(0, view_indices.size, _READ_VIEW_BATCH):
            k1 = min(k0 + _READ_VIEW_BATCH, view_indices.size)
            block = _pipeline._stage_batch(sino[view_indices[k0:k1], row_lo:row_hi, :], device)
            out[k0:k1] = block.to('cpu', torch.float32).numpy()
    return out


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


def _rotated_band(sino, view_indices, row_window, device, det_rotation):
    """The band ``row_window`` of the views ``view_indices``, rotated by ``det_rotation`` about the
    full detector's center with cubic interpolation.

    The band is read from the sinogram with a margin of rows on each side, so the rotation samples
    nothing outside the rows it has, and it is cropped afterward.  The cubic kernel is used because
    the bilinear one smooths the data by an amount that grows with the angle, which biases a search
    over the angle toward its bounds on cone-beam data.
    """
    import cv2
    num_rows, num_channels = sino.shape[1], sino.shape[2]
    row_lo, row_hi = row_window
    if det_rotation == 0.0:
        return _read_band(sino, view_indices, row_window, device)
    center_row = (num_rows - 1) / 2.0
    margin = _rotation_row_margin(det_rotation, max(abs(row_lo - center_row), abs(row_hi - 1 - center_row)),
                                  num_channels)
    band_lo, band_hi = max(0, row_lo - margin), min(num_rows, row_hi + margin)
    band = _read_band(sino, view_indices, (band_lo, band_hi), device)
    matrix = cv2.getRotationMatrix2D(((num_channels - 1) / 2.0, center_row - band_lo),
                                     math.degrees(det_rotation), 1.0)
    out = np.empty((band.shape[0], row_hi - row_lo, num_channels), dtype=np.float32)
    for i in range(band.shape[0]):
        rotated = cv2.warpAffine(band[i], matrix, (num_channels, band.shape[1]), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[i] = rotated[row_lo - band_lo:row_hi - band_lo]
    return out


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


# ── the opposite-view comparison ──────────────────────────────────────────────────────────────────

class _ConjugatePairs:
    """The data behind an opposite-view score.

    An instance holds a band of detector rows around the central plane of the scan, from every
    view, and, for each view and channel, which view holds the opposite ray.  The opposite of the
    ray at view angle ``beta`` and fan angle ``gamma`` lies at view angle ``beta + pi - 2 * gamma``
    and fan angle ``-gamma``, in the sign conventions of ``cone_beam._cone_pixel_xy_mag``.
    Parallel beam is the case ``gamma = 0``.  The partner view is interpolated linearly between the
    two views nearest that angle.  Every view is a reference, so each unordered pair is compared
    from both sides and the interpolation errors of the two sides cancel.

    The fan angle of a channel depends on the channel offset, so the partners are computed once at
    ``pairing_offset``, and a candidate offset ``d`` channels away moves a channel's partner angle
    by ``2 d delta / sdd``.

    Args:
        ct_model: a parallel or cone model.
        pairing_offset (float or None): the channel offset in ALU that the fan angles are computed
            at.  None is the model's current value.
    """

    def __init__(self, ct_model, pairing_offset=None):
        self.kind = _geometry_kind(ct_model)
        num_views, num_det_rows, num_det_channels = (int(s) for s in ct_model.get_params('sinogram_shape'))
        delta_det_channel, det_channel_offset = ct_model.get_params(['delta_det_channel', 'det_channel_offset'])
        self.delta = float(delta_det_channel)
        self.num_views = num_views
        self.num_channels = num_det_channels
        self.model_offset = float(det_channel_offset)
        self.pairing_offset = self.model_offset if pairing_offset is None else float(pairing_offset)
        self.row_window = self._band_rows(ct_model)
        self.device = ct_model.torch_device

        # These are the opposite ray's view angles, one per view and per channel.  A reference ray
        # at fan angle -gamma(u) has its opposite at beta + pi + 2 gamma(u), where
        # u = (m - c) delta + d for mirrored column m.
        angles = _view_angles(ct_model)
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
        target = angles[:, None] + np.pi - 2.0 * gamma[None, :]
        low, high, self.partner_weight = self._partners(angles, target)
        self.partner_indices = np.unique(np.concatenate([low.ravel(), high.ravel()]))
        self.partner_low = np.searchsorted(self.partner_indices, low)
        self.partner_high = np.searchsorted(self.partner_indices, high)

    @staticmethod
    def _band_rows(ct_model):
        """The rows ``(lo, hi)`` of the band around the scan's central plane."""
        num_det_rows = int(ct_model.get_params('sinogram_shape')[1])
        delta_det_row, det_row_offset = ct_model.get_params(['delta_det_row', 'det_row_offset'])
        num_rows = _NUM_ROWS
        if isinstance(ct_model, ConeBeamModel):
            # Opposite rays through a point off the central plane reach the detector at different
            # heights.  The band keeps that difference within one row.
            min_mag, _ = ct_model.pixel_magnification_bounds()
            source_detector_dist, source_iso_dist = ct_model.get_params(
                ['source_detector_dist', 'source_iso_dist'])
            if not np.isinf(source_detector_dist):
                support_radius = source_detector_dist / min_mag - source_iso_dist
                half = max(1, math.floor(source_iso_dist / (2.0 * support_radius)))
                num_rows = min(num_rows, 2 * half + 1)
        num_rows = max(1, min(num_rows, num_det_rows))
        # The central plane reaches the row where the detector height v is zero, and the band is
        # centered there.
        central_row = (num_det_rows - 1) / 2.0 + det_row_offset / delta_det_row
        lo = int(round(central_row - (num_rows - 1) / 2.0))
        lo = max(0, min(lo, num_det_rows - num_rows))
        return lo, lo + num_rows

    @staticmethod
    def _partners(angles, target):
        """For each target angle, the two views that bracket it on the circle and the weight of the
        second.  Returns three arrays of the target's shape."""
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

    def bands(self, sino, det_rotation=0.0):
        """The band of every view and the band of the partner views, rotated by ``det_rotation``."""
        return (_rotated_band(sino, np.arange(self.num_views), self.row_window, self.device, det_rotation),
                _rotated_band(sino, self.partner_indices, self.row_window, self.device, det_rotation))

    def pairs(self, bands):
        """The band of every view, and the mirrored opposite ray of every element of it.

        Args:
            bands (tuple of ndarray): the view and partner bands from :meth:`bands`.

        Returns:
            tuple of ndarray: ``(views, opposites)``, each of shape ``(num_views, num_rows,
            num_channels)`` in float32.  Element ``[i, r, n]`` of ``opposites`` is the measurement
            of the ray opposite to element ``[i, r, n]`` of ``views``, placed at the mirrored
            channel.  The two agree up to a shift of twice the channel offset.
        """
        views, partners = bands
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
        return int(math.ceil(2.0 * abs(max_abs_offset) / self.delta)) + _EDGE_MARGIN

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
        num_kept = max(1, int(round(relative.size * (1.0 - _TRIM_FRACTION))))
        return np.sort(np.argsort(relative)[:num_kept])

    def score(self, prepared, opposites, det_channel_offset, keep):
        """The opposite-view score at one channel offset: the mean squared difference over the
        kept pairs divided by the mean square of their views."""
        per_pair = self.per_pair(prepared, opposites, det_channel_offset)
        return float(per_pair[keep].mean() / prepared['energy'][keep].mean())


def _search_minimum(score_fn, bounds, tolerance):
    """Find the minimum of a scalar score over ``bounds``.

    A coarse pass evaluates ``_NUM_COARSE`` equally spaced candidates, which shows whether the
    curve has one minimum.  A golden-section search then narrows the bracket around the coarse
    minimum until it is shorter than ``tolerance``.

    Returns:
        tuple: ``(best, notes)``.  ``notes`` is a list of strings describing anything the caller
        should warn about: a coarse minimum at an edge of the bounds, or more than one local
        minimum on the coarse curve.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not hi > lo:
        raise ValueError(f'bounds must satisfy lo < hi; got {bounds}.')
    coarse = np.linspace(lo, hi, _NUM_COARSE)
    evaluated = {float(x): float(score_fn(x)) for x in coarse}
    coarse_scores = np.array([evaluated[float(x)] for x in coarse])
    notes = []
    best = int(np.argmin(coarse_scores))
    interior = coarse_scores[1:-1]
    local_minima = int(np.sum((interior < coarse_scores[:-2]) & (interior <= coarse_scores[2:])))
    if local_minima > 1:
        notes.append(f'the score curve has {local_minima} local minima on the coarse grid')
    if best in (0, _NUM_COARSE - 1):
        notes.append('the coarse minimum sits at an edge of the bounds')
    a = float(coarse[max(best - 1, 0)])
    b = float(coarse[min(best + 1, _NUM_COARSE - 1)])

    # The golden-section search runs on [a, b].  Each step drops the worse end and
    # keeps one interior point.
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
    return min(evaluated, key=evaluated.get), notes


# ── the estimators ────────────────────────────────────────────────────────────────────────────────

def estimate_det_channel_offset(ct_model, sino, *, bounds=None):
    """Estimate ``det_channel_offset`` from the sinogram by comparing each view with its opposite.

    In a scan over a full rotation every ray is measured twice, once from each side.  A voxel at
    in-plane position x projects to channel ``(x + det_channel_offset) / delta_det_channel`` from
    the detector center, and after a half rotation it projects to the mirrored position.  A view
    and its mirrored opposite therefore differ by a shift of twice the offset, and the estimate is
    the candidate offset at which they agree best.  For cone beam the opposite of a channel lies at
    a view angle that depends on the fan angle, and the comparison uses a band of rows around the
    central plane.

    The search scores candidates across ``bounds`` on a coarse grid and then narrows the bracket by
    golden section to a hundredth of a channel.  It warns when the coarse curve has more than one
    minimum or its minimum sits at an edge of the bounds.  A trimmed mean drops the tenth of the
    view pairs that agree worst, so a few corrupted views do not move the estimate.

    The method needs views over a full rotation and an opposite view at the same axial position, so
    it refuses a short scan and a helical scan.  A multiaxis parallel model is not supported yet.
    An offset scan, whose detector is displaced by hundreds of channels, is not served, because the
    search range is a few channels.  A sinogram in the divided device form is refused.

    Args:
        ct_model (TomographyModel): a parallel or cone model.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.
        bounds (tuple of float, optional): the search range in ALU.  None (the default) is a window
            of four channels on each side of the model's current value.  That window moves to
            center on the edge where the coarse minimum sits, at the same width, up to eight times.
            A range given here is not moved.

    Returns:
        float: the estimate in ALU, to set with ``ct_model.set_params(det_channel_offset=...)``.

    Raises:
        ValueError: for a multiaxis model, a helical scan, or views that do not cover a full
            rotation.
    """
    _require_conjugate_geometry(ct_model, 'det_channel_offset')
    _sharding.reject_shards('estimate_det_channel_offset', sino=sino)
    problem = _ConjugatePairs(ct_model)
    user_bounds = bounds
    if bounds is None:
        half_range = _OFFSET_HALF_RANGE_CHANNELS * problem.delta
        bounds = (problem.model_offset - half_range, problem.model_offset + half_range)
    margin = problem.channel_margin(max(abs(bounds[0]), abs(bounds[1])))
    tolerance = _OFFSET_TOLERANCE_CHANNELS * problem.delta

    def search(problem):
        """One pass over the current bounds: choose the kept pairs at the best candidate of a
        coarse grid scored on every pair, then search on that fixed set.  The bands are read per
        pass, because the set of partner views depends on the pairing offset."""
        views, opposites = problem.pairs(problem.bands(sino))
        prepared = problem.prepare(views, opposites, margin)
        every_pair = np.arange(problem.num_views)
        coarse = np.linspace(bounds[0], bounds[1], _NUM_COARSE)
        coarse_best = coarse[int(np.argmin([problem.score(prepared, opposites, x, every_pair)
                                            for x in coarse]))]
        keep = problem.keep_set(prepared, opposites, float(coarse_best))
        return _search_minimum(lambda offset: problem.score(prepared, opposites, offset, keep),
                               bounds, tolerance)

    # A coarse minimum at an edge of the window means the window is in the wrong place,
    # so the window moves to center on that edge and the search repeats.
    best, notes = search(problem)
    slides = 0
    while ('the coarse minimum sits at an edge of the bounds' in notes and user_bounds is None
           and slides < _OFFSET_MAX_SLIDES):
        half_width = 0.5 * (bounds[1] - bounds[0])
        moved = (best - half_width, best + half_width)
        moved_margin = problem.channel_margin(max(abs(moved[0]), abs(moved[1])))
        if problem.num_channels - 2 * moved_margin < _MIN_REGION_FRACTION * problem.num_channels:
            notes.append('the search window could not move further, because the channels excluded '
                         'for the circular shift would leave less than a quarter of the detector')
            break
        bounds, margin = moved, moved_margin
        best, notes = search(problem)
        slides += 1

    # Cone beam uses two passes.  The partner view of a channel depends on the offset, so the
    # second pass pairs at the estimate from the first pass.
    if problem.kind == 'cone' and abs(best - problem.pairing_offset) > tolerance:
        problem = _ConjugatePairs(ct_model, pairing_offset=best)
        best, notes = search(problem)
    for note in notes:
        warnings.warn(f'estimate_det_channel_offset: {note}.')
    return float(best)


def estimate_det_rotation(ct_model, sino, *, bounds=None):
    """Estimate the detector rotation, in radians, by comparing each view with its opposite.

    A detector rotated about the optical axis records every view rotated by that angle, and
    mirroring a view in channels reverses the sign of that rotation, so a view and its mirrored
    opposite differ by twice the angle.  Each candidate angle is applied to a band of rows from
    every view by cubic resampling about the detector center, the views are paired with their
    opposites as in :func:`estimate_det_channel_offset`, and the candidate at which they agree best
    is returned.  The comparison shifts the opposites by twice the model's ``det_channel_offset``,
    so estimate and set the offset first.

    Resampling smooths the band, so an estimate that displaces the edge channels by less than about
    one pixel is uncertain, and the function warns in that case.  When the scanner loader supplies
    a detector tilt, prefer it over the estimate, and check the slices far from the central plane
    before applying an estimate, because a rotation displaces those slices most.

    Args:
        ct_model (TomographyModel): a parallel or flat-detector cone model.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.
        bounds (tuple of float, optional): the search range in radians, within five degrees of
            zero.  None (the default) is the full five degrees on each side.

    Returns:
        float: the angle in radians, to remove with
        :func:`~mbirtorch.preprocess.correct_det_rotation`.

    Raises:
        ValueError: for a multiaxis model, a helical scan, a curved detector, views that do not
            cover a full rotation, or bounds beyond the five degree cap.
    """
    _require_conjugate_geometry(ct_model, 'det_rotation')
    _sharding.reject_shards('estimate_det_rotation', sino=sino)
    if bounds is None:
        bounds = (-_MAX_DET_ROTATION, _MAX_DET_ROTATION)
    if max(abs(bounds[0]), abs(bounds[1])) > _MAX_DET_ROTATION:
        raise ValueError(f'bounds must lie within {math.degrees(_MAX_DET_ROTATION):.0f} degrees of '
                         f'zero; got {tuple(math.degrees(b) for b in bounds)} degrees.')
    problem = _ConjugatePairs(ct_model)
    offset = problem.pairing_offset
    margin = problem.channel_margin(offset)

    # The pairs kept by the trimmed mean are chosen with no rotation applied, and every candidate
    # is scored on that set.
    views, opposites = problem.pairs(problem.bands(sino))
    prepared = problem.prepare(views, opposites, margin)
    keep = problem.keep_set(prepared, opposites, offset)

    def score_at(det_rotation):
        views, opposites = problem.pairs(problem.bands(sino, float(det_rotation)))
        return problem.score(problem.prepare(views, opposites, margin), opposites, offset, keep)

    best, notes = _search_minimum(score_at, bounds, _ROTATION_TOLERANCE)
    edge_displacement = abs(best) * problem.num_channels / 2.0
    if edge_displacement < _MIN_EDGE_DISPLACEMENT:
        notes.append(f'the estimate displaces the edge channels by {edge_displacement:.2f} pixels, '
                     'below the one pixel at which it is reliable.  Prefer a vendor tilt when the '
                     'scanner loader supplies one, and check the slices far from the central plane '
                     'before applying it')
    for note in notes:
        warnings.warn(f'estimate_det_rotation: {note}.')
    return float(best)


# ── view alignment ────────────────────────────────────────────────────────────────────────────────

def _sino_high_pass_filtering(sino, sigma_row=3.0, sigma_col=15.0, subtract_view_mean=True):
    """
    High-pass filter for 3D cone-beam sinogram.

    Args:
        sino (numpy array or tensor): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        sigma_row (float, optional): Gaussian sigma along detector rows (vertical). Use smaller value than sigma_col.
        Defaults to 3.0.
        sigma_col (float, optional): Gaussian sigma along detector channels (horizontal). Defaults to 15.0.
        subtract_view_mean (bool, optional): If True, subtract per-view mean (DC offset removal). Defaults to True.

    Returns:
        filtered_sino (numpy array): High-pass filtered sinogram, same shape as input.
    """
    import cv2

    if isinstance(sino, torch.Tensor):
        sino = sino.detach().cpu().numpy()
    sino_np = np.asarray(sino)
    if sino_np.ndim != 3:
        raise ValueError(f"Expected shape (num_views, num_det_rows, num_det_channels), got {sino_np.shape}")

    num_views, num_det_rows, num_det_channels = sino_np.shape
    filtered_sino = np.empty_like(sino_np)

    for view in range(num_views):
        single_view = sino_np[view]

        if subtract_view_mean:
            single_view = single_view - single_view.mean()

        loss_pass_estimate = cv2.GaussianBlur(
            single_view,
            ksize=(0, 0),
            sigmaX=sigma_col,
            sigmaY=sigma_row,
            borderType=cv2.BORDER_REFLECT,
        )

        filtered_sino[view] = single_view - loss_pass_estimate

    return filtered_sino


def _estimate_sino_view_offset(ct_model, sino, recon_direct):
    """
    Estimate per-view 2D shifts for a sinogram.

    This function estimate the shifts in three steps:
    1. Forward project the preliminary reconstruction using the CT model.
    2. Apply high-pass filtering to both the sinogram and the
        forward projection of the preliminary reconstruction.
    3. For each view, estimate a 2D shift that aligns the sinogram view
        to the corresponding forward-projected view using an image alignment method from OpenCV

    Args:
        ct_model (mt.TomographyModel): A CT model object that defined the CT geometry.
        sino (numpy array or tensor): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        recon_direct (numpy array or tensor): A preliminary 3D reconstruction of the sinogram.

    Returns:
        estimated_shifts (numpy.array): A (num_views, 2) array of per-view shift (y, x) in pixels.
            Each shift specified how much the corresponding sinogram slice should be shifted to match forward projection.
            Positive x shifts the view right. Positive y shifts the view down.
    """
    import cv2

    recon_shape = ct_model.get_params('recon_shape')
    if tuple(recon_direct.shape) != tuple(recon_shape):
        raise ValueError("Input recon shape does not match ct_model's recon shape.")

    sino_from_recon = ct_model.forward_project(recon_direct)

    filtered_sino = _sino_high_pass_filtering(sino)
    filtered_sino_from_recon = _sino_high_pass_filtering(sino_from_recon)

    num_slices, num_rows, num_channels = sino.shape
    estimated_shifts = np.zeros((num_slices, 2))

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    for slice_index in range(num_slices):
        sino_from_recon_view = np.asarray(filtered_sino_from_recon[slice_index, :, :], dtype=np.float32)
        sino_view = np.asarray(filtered_sino[slice_index, :, :], dtype=np.float32)
        cc, warp_matrix = cv2.findTransformECC(sino_from_recon_view, sino_view, warp_matrix,
                                               cv2.MOTION_TRANSLATION)
        estimated_shifts[slice_index, 0] = -warp_matrix[1, 2]
        estimated_shifts[slice_index, 1] = -warp_matrix[0, 2]

    return estimated_shifts


def _translate_views_bilinear(sino, shifts):
    """Shift each view of a sinogram by its own (dy, dx) with bilinear interpolation, zero outside.

    Matches a linear scale-and-translate with unit scale: output(i, j) samples the input at
    (i - dy, j - dx); samples outside the view are zero.
    """
    sino = torch.as_tensor(np.asarray(sino))
    shifts = np.asarray(shifts, dtype=np.float64)
    num_views, num_rows, num_cols = sino.shape
    dtype = sino.dtype
    grid_i, grid_j = torch.meshgrid(torch.arange(num_rows, dtype=dtype),
                                    torch.arange(num_cols, dtype=dtype), indexing='ij')
    out = torch.empty_like(sino)
    for v in range(num_views):
        dy, dx = float(shifts[v, 0]), float(shifts[v, 1])
        src_row = grid_i - dy
        src_col = grid_j - dx
        lower_row = torch.floor(src_row)
        lower_col = torch.floor(src_col)
        frac_row = src_row - lower_row
        frac_col = src_col - lower_col
        r0 = torch.clamp(lower_row.to(torch.int64), 0, num_rows - 1)
        r1 = torch.clamp(torch.ceil(src_row).to(torch.int64), 0, num_rows - 1)
        c0 = torch.clamp(lower_col.to(torch.int64), 0, num_cols - 1)
        c1 = torch.clamp(torch.ceil(src_col).to(torch.int64), 0, num_cols - 1)
        view = sino[v]
        shifted = (((1.0 - frac_row) * (1.0 - frac_col)) * view[r0, c0]
                   + ((1.0 - frac_row) * frac_col) * view[r0, c1]
                   + (frac_row * (1.0 - frac_col)) * view[r1, c0]
                   + (frac_row * frac_col) * view[r1, c1])
        in_bounds = ((src_row >= 0) & (src_row <= num_rows - 1)
                     & (src_col >= 0) & (src_col <= num_cols - 1)).to(dtype)
        out[v] = shifted * in_bounds
    return out


def align_sino_views(ct_model, sino, recon_direct):
    """
    Shift each view of a sinogram to align it with the forward projection of a first reconstruction.

    A 2D shift is estimated for each view by comparing it with the forward projection of
    ``recon_direct``, and the view is shifted by that amount.  This corrects small per-view motion
    of the object.

    Args:
        ct_model (TomographyModel): The model of the scan.
        sino (numpy array or tensor): Sinogram, shape (num_views, num_det_rows, num_det_channels).
        recon_direct (numpy array or tensor): A first reconstruction of ``sino``, such as the output
            of :meth:`~mbirtorch.TomographyModel.recon_direct`.

    Returns:
        numpy.ndarray: The aligned sinogram, the shape of ``sino``.
    """
    estimated_shifts = _estimate_sino_view_offset(ct_model, sino, recon_direct)

    return _translate_views_bilinear(sino, estimated_shifts).cpu().numpy()
