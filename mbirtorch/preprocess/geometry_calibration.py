"""Geometry calibration from the sinogram.

``estimate_det_channel_offset`` estimates the detector channel offset, which sets the center of
rotation, from the sinogram alone.  It returns one number and changes nothing; the caller sets it
with ``set_params``.

``fit_det_alignment`` fits each view to the forward projection of a first reconstruction and so
finds the detector offsets, the detector rotation, and the per-view shifts.  It needs a
reconstruction, which is the costly part.  It returns values for the model and corrections for the
data, and ``correct_det_alignment`` resamples the views with those corrections.
``align_sino_views`` is deprecated in favor of the two.

The order of preprocessing matters.  Run in this order:
 1. defective-pixel interpolation, background offset correction, and stripe removal
 2. ``estimate_det_channel_offset``
 3. ``fit_det_alignment`` and ``correct_det_alignment``.
Stripe removal comes first because a gain stripe sits at a fixed channel, and a geometry estimate
would take it for a feature of the object.  The reprojection fit comes last because a wrong
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
from .corrections import _resample_views

__all__ = ['estimate_det_channel_offset', 'fit_det_alignment', 'correct_det_alignment',
           'align_sino_views']

# This many views of the full sinogram are read per step when a band of rows is read.
_READ_VIEW_BATCH = 64

_NUM_COARSE = 11                      # candidates scored on the coarse grid of a search
_NUM_ROWS = 16                        # detector rows compared, before the cone-beam limit

# A scan covers a full rotation when no gap between neighboring view angles exceeds both this
# many times the median gap and this many radians.  A scan over a half rotation is refused.
_MAX_GAP_RATIO = 3.0
_MAX_GAP = math.radians(5.0)

# A view whose correlation with its reprojection, after the fit, is below this is a failed fit.
_ECC_MIN_CORRELATION = 0.2
# The fit of a view stops after this many iterations or when the correlation gains less than this.
_ECC_MAX_ITERATIONS = 200
_ECC_EPS = 1e-6


# ── geometry checks ───────────────────────────────────────────────────────────────────────────────

def _geometry_kind(ct_model):
    """Classify a model as 'parallel', 'cone', 'multiaxis', or 'other'."""
    if isinstance(ct_model, ConeBeamModel):
        return 'cone'
    if isinstance(ct_model, MultiAxisParallelModel):
        return 'multiaxis'
    if isinstance(ct_model, ParallelBeamModel):
        return 'parallel'
    return 'other'


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


def _unsuitable_reason(ct_model):
    """Why the opposite-view comparison cannot serve this scan, or None when it can.

    The comparison needs a parallel or cone-beam scan over a full rotation, so that every view has
    an opposite view at the same axial position.
    """
    kind = _geometry_kind(ct_model)
    if kind == 'other':
        return f'{type(ct_model).__name__} has no opposite views'
    if kind == 'multiaxis':
        return 'a multiaxis parallel scan is not supported yet'
    if _is_helical(ct_model):
        return 'a helical scan has no opposite view at the same axial position'
    gaps = _angular_gaps(_view_angles(ct_model))
    if gaps.max() > max(_MAX_GAP_RATIO * np.median(gaps), _MAX_GAP):
        return (f'the views cover {math.degrees(2 * np.pi - gaps.max()):.1f} degrees with a gap of '
                f'{math.degrees(gaps.max()):.1f} degrees, so not every view has an opposite view')
    return None


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

    def bands(self, sino):
        """The band of every view and the band of the partner views."""
        return (_read_band(sino, np.arange(self.num_views), self.row_window, self.device),
                _read_band(sino, self.partner_indices, self.row_window, self.device))

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


# ── the channel offset ────────────────────────────────────────────────────────────────────────────

_HP_SIGMA_CHANNELS = 15.0      # width of the blur removed from each profile, as in align_sino_views
_HP_SEARCH_FRACTION = 0.25     # half-width of the integer shift search, as a fraction of the detector
_HP_MIN_PEAK = 0.2             # a correlation peak below this is reported as weak
_HP_MAX_SPREAD = 1.0           # per-view peaks further than this many channels from the global one


def _high_pass_profiles(band):
    """One high-passed profile per view from a band of rows: the rows are averaged, the mean is
    removed, and a Gaussian blur of ``_HP_SIGMA_CHANNELS`` along the channels is subtracted.  The
    blur extends each end value past the edge, which keeps the ends of a short profile from
    biasing the estimate, as reflecting the profile does."""
    from scipy.ndimage import gaussian_filter1d
    profile = band.astype(np.float64).mean(axis=1)
    profile -= profile.mean(axis=1, keepdims=True)
    return profile - gaussian_filter1d(profile, _HP_SIGMA_CHANNELS, axis=1, mode='nearest')


def _overlap_correlation(views, opposites, shifts):
    """The normalized correlation between every view and its opposite moved by each integer shift,
    over the channels the two cover in common.

    Returns:
        tuple: ``(global_curve, per_view_curves)``.  ``global_curve[k]`` sums the products over
        every view before normalizing, so views with more content weigh more.
        ``per_view_curves[v, k]`` normalizes each view on its own.
    """
    num_views, num_channels = views.shape
    num = np.empty((num_views, shifts.size))
    den_v = np.empty_like(num)
    den_o = np.empty_like(num)
    for k, s in enumerate(shifts):
        lo, hi = max(0, s), min(num_channels, num_channels + s)
        v = views[:, lo:hi]
        o = opposites[:, lo - s:hi - s]
        num[:, k] = (v * o).sum(axis=1)
        den_v[:, k] = (v * v).sum(axis=1)
        den_o[:, k] = (o * o).sum(axis=1)
    global_curve = num.sum(axis=0) / np.sqrt(den_v.sum(axis=0) * den_o.sum(axis=0))
    per_view = num / np.sqrt(np.maximum(den_v * den_o, 1e-30))
    return global_curve, per_view


def _fractional_correlation(views, opposites, shift_int, fraction):
    """The global normalized correlation at the shift ``shift_int + fraction``.

    The opposites are moved by the integer part by indexing, over the channels the two cover in
    common.  The fraction is then split: the views move by half of it one way and the opposites
    by half the other way, both resampled by a cubic spline, so that the two sides are blurred
    alike and the correlation does not favor whole-channel shifts.  Three channels at each end,
    where the resampling has no neighbors, are left out.
    """
    from scipy.ndimage import shift as ndshift
    num_channels = views.shape[1]
    lo, hi = max(0, shift_int), min(num_channels, num_channels + shift_int)
    v = ndshift(views[:, lo:hi], (0.0, -0.5 * fraction), order=3, mode='nearest')[:, 3:-3]
    o = ndshift(opposites[:, lo - shift_int:hi - shift_int], (0.0, 0.5 * fraction), order=3, mode='nearest')[:, 3:-3]
    return float((v * o).sum() / math.sqrt((v * v).sum() * (o * o).sum()))


def _estimate_offset_by_correlation(ct_model, sino, pairing_offset=None):
    """One pass of the channel offset estimate: the correlation of high-passed profiles of each view
    and its mirrored opposite, searched over integer shifts and refined to a fraction of a channel.

    Returns:
        dict: ``offset`` in ALU; ``shift`` in channels, which is twice the offset; ``peak``, the
        correlation at the peak; ``parabola`` and ``refined``, the two sub-pixel estimates of the
        shift; ``spread``, the median distance of the per-view peaks from the global one in
        channels; and ``curve`` and ``shifts``, the global correlation curve.
    """
    problem = _ConjugatePairs(ct_model, pairing_offset=pairing_offset)
    views, opposites = problem.pairs(problem.bands(sino))
    views, opposites = _high_pass_profiles(views), _high_pass_profiles(opposites)
    delta = problem.delta
    center = int(round(2.0 * problem.model_offset / delta))
    half = int(round(_HP_SEARCH_FRACTION * problem.num_channels))
    # The window of integer shifts is centered on the model's value.  A peak at an edge of the
    # window means the window is in the wrong place, so it moves to center on the peak, a few
    # times at most, and the overlap never drops below half the detector.
    limit = problem.num_channels // 2
    for _ in range(4):
        shifts = np.arange(max(center - half, -limit), min(center + half, limit) + 1)
        curve, per_view = _overlap_correlation(views, opposites, shifts)
        k = int(np.argmax(curve))
        if 1 < k < shifts.size - 2 or abs(int(shifts[k])) >= limit:
            break
        center = int(shifts[k])
    best = int(shifts[k])
    if 0 < k < shifts.size - 1:
        a, b, c = curve[k - 1], curve[k], curve[k + 1]
        parabola = best + 0.5 * (a - c) / (a - 2.0 * b + c)
    else:
        parabola = float(best)
    score = lambda f: -_fractional_correlation(views, opposites, best, f)
    fraction, _ = _search_minimum(score, (-1.0, 1.0), 1.0 / 64)
    refined = best + fraction
    per_view_peaks = shifts[np.argmax(per_view, axis=1)]
    spread = float(np.median(np.abs(per_view_peaks - refined)))
    return dict(offset=refined * delta / 2.0, shift=refined, peak=float(curve[k]), parabola=parabola,
                refined=refined, spread=spread, curve=curve, shifts=shifts, kind=problem.kind)


def estimate_det_channel_offset(ct_model, sino):
    """Estimate the detector channel offset of a scan over a full rotation.

    The estimate is the value of ``det_channel_offset`` at which each view agrees with its opposite
    view.  In a full rotation every ray is measured twice, once from each side, so a view and its
    mirrored opposite view are the same up to a shift of twice the channel offset.  The function
    finds that shift by correlating high-passed profiles of the two, taken from a band of rows at
    the central plane, and refines it to a fraction of a channel.

    The scan must be a parallel or cone-beam scan over a full rotation.  For any other scan, such
    as a short scan, a helical scan, or a multiaxis scan, the function warns and returns the
    model's current value.  It also warns when the opposite views match weakly.

    Args:
        ct_model (TomographyModel): the model of the scan.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.

    Returns:
        float: the estimate in ALU, to set with ``ct_model.set_params(det_channel_offset=...)``.

    Example:
        >>> det_channel_offset = mtp.estimate_det_channel_offset(ct_model, sino)  # from the sinogram alone
        >>> ct_model.set_params(det_channel_offset=det_channel_offset)  # update the model
    """
    reason = _unsuitable_reason(ct_model)
    if reason is not None:
        warnings.warn(f'estimate_det_channel_offset: {reason}, so det_channel_offset was left at the '
                      f'model\'s value.')
        return float(ct_model.get_params('det_channel_offset'))
    _sharding.reject_shards('estimate_det_channel_offset', sino=sino)
    result = _estimate_offset_by_correlation(ct_model, sino)
    if result['kind'] == 'cone':
        result = _estimate_offset_by_correlation(ct_model, sino, pairing_offset=result['offset'])
    if result['peak'] < _HP_MIN_PEAK:
        warnings.warn(f'estimate_det_channel_offset: the correlation peak is weak ({result["peak"]:.2f}).')
    if result['spread'] > _HP_MAX_SPREAD:
        warnings.warn(f'estimate_det_channel_offset: the per-view peaks spread {result["spread"]:.1f} '
                      'channels around the estimate.')
    if abs(result['parabola'] - result['refined']) > 0.25:
        warnings.warn(f'estimate_det_channel_offset: the two sub-pixel estimates differ by '
                      f'{abs(result["parabola"] - result["refined"]):.2f} channels.')
    return float(result['offset'])


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

    The blur replicates the edge value outside the view.  A reflected border would make the
    high-passed copy of a shifted view differ from the shifted high-passed view near the edges,
    which biases a fit between the two toward too small a shift.

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
            borderType=cv2.BORDER_REPLICATE,
        )

        filtered_sino[view] = single_view - loss_pass_estimate

    return filtered_sino


def _fit_views(ct_model, sino, recon_direct, rotation):
    """Fit each view to its reprojection and return the fitted transforms about the detector center.

    Returns ``(angles, shifts, failed)``: the rotation of each view in radians, in the convention
    of ``correct_det_rotation``; the sampling offset of each view in pixels as ``(row, channel)``,
    which is where a view's content sits relative to the reprojection once the rotation about the
    detector center is taken out; and a boolean mask of the views whose fit failed, which hold zero.
    """
    import cv2

    recon_shape = ct_model.get_params('recon_shape')
    if tuple(recon_direct.shape) != tuple(recon_shape):
        raise ValueError("Input recon shape does not match ct_model's recon shape.")

    sino_from_recon = ct_model.forward_project(recon_direct)
    filtered_sino = _sino_high_pass_filtering(sino)
    filtered_sino_from_recon = _sino_high_pass_filtering(sino_from_recon)

    num_views, num_rows, num_cols = filtered_sino.shape
    center = np.array([(num_rows - 1) / 2.0, (num_cols - 1) / 2.0])
    motion = cv2.MOTION_EUCLIDEAN if rotation else cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, _ECC_MAX_ITERATIONS, _ECC_EPS)
    angles = np.zeros(num_views)
    shifts = np.zeros((num_views, 2))
    failed = np.zeros(num_views, dtype=bool)
    for v in range(num_views):
        template = np.ascontiguousarray(filtered_sino_from_recon[v], dtype=np.float32)
        view = np.ascontiguousarray(filtered_sino[v], dtype=np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            correlation, warp = cv2.findTransformECC(template, view, warp, motion, criteria)
        except cv2.error:
            failed[v] = True
            continue
        if not correlation >= _ECC_MIN_CORRELATION:
            failed[v] = True
            continue
        # The warp maps a reprojection pixel (x, y) to the view pixel that holds the same content:
        # view(R p + t) = reprojection(p), with R the rotation about the image corner.  Rewritten
        # about the detector center, the translation becomes t + (R - I) center.
        theta = math.atan2(warp[1, 0], warp[0, 0]) if rotation else 0.0
        t = np.array([warp[1, 2], warp[0, 2]], dtype=np.float64)        # (row, channel)
        rot = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        angles[v] = theta
        shifts[v] = t + rot @ center - center
    return angles, shifts, failed


def fit_det_alignment(ct_model, sino, recon_direct, rotation=False):
    """Fit each view to the reprojection of a first reconstruction for the detector offsets and rotation.

    Each view is compared with the forward projection of ``recon_direct``.  The transform that
    aligns the two, a 2D shift and optionally a rotation about the detector center, is fit per view
    on high-passed copies of both.  The median over views of the fitted offsets is the global value
    of each offset parameter.  Each view's deviation from the median, and its rotation, are
    corrections for the data.

    Returns the two as dicts.  ``model_params`` holds values to set on the model with
    ``set_params``: ``det_channel_offset`` and ``det_row_offset``, in ALU.  ``view_params`` holds
    one array of length ``num_views`` per key, to apply to the data with
    :func:`correct_det_alignment`: ``det_channel_offset`` and ``det_row_offset`` are the amount by
    which each view's offset exceeds the model value, in ALU, and, when ``rotation`` is True,
    ``det_rotation`` is the rotation to remove from each view, in radians, in the convention of
    :func:`~mbirtorch.preprocess.correct_det_rotation`.

    A view whose fit does not converge, or whose correlation with its reprojection stays weak, is
    reported in a warning.  It is left out of the medians and gets a zero deviation and the median
    rotation.  There is no warning on the spread of the per-view values, since per-view motion is
    what the per-view values are for.

    Args:
        ct_model (TomographyModel): The model of the scan.  It is not changed.
        sino (numpy array or tensor): Sinogram, shape (num_views, num_det_rows, num_det_channels).
        recon_direct (numpy array or tensor): A first reconstruction of ``sino``, such as the output
            of :meth:`~mbirtorch.TomographyModel.recon_direct`.
        rotation (bool, optional): Fit a rotation about the detector center as well as a shift.
            Defaults to False.

    Returns:
        tuple: ``(model_params, view_params)``, the two dicts described above.

    Example:
        >>> recon_direct = ct_model.recon_direct(sino)  # reconstruct object
        >>> model_params, view_params = mtp.fit_det_alignment(ct_model, sino, recon_direct, rotation=True)
        >>> ct_model.set_params(**model_params)  # update offset parameters
        >>> sino = mtp.correct_det_alignment(ct_model, sino, view_params)  # correct per-view jitter and rotation
    """
    angles, shifts, failed = _fit_views(ct_model, sino, recon_direct, rotation)
    num_views = len(angles)
    if failed.all():
        raise ValueError("No view could be aligned with its reprojection.")
    if failed.any():
        warnings.warn(f"The alignment fit failed for {int(failed.sum())} of {num_views} views, "
                      f"starting with view {int(np.flatnonzero(failed)[0])}.  Each gets no per-view "
                      "correction and the median rotation.")

    delta_row, delta_channel = (float(d) for d in ct_model.get_params(['delta_det_row', 'delta_det_channel']))
    row_offset, channel_offset = (float(d) for d in ct_model.get_params(['det_row_offset', 'det_channel_offset']))
    # A view whose content sits s pixels higher than the reprojection has an offset s * pitch
    # larger than the model's.
    row_values = row_offset + shifts[:, 0] * delta_row
    channel_values = channel_offset + shifts[:, 1] * delta_channel
    good = ~failed
    model_params = {'det_channel_offset': float(np.median(channel_values[good])),
                    'det_row_offset': float(np.median(row_values[good]))}
    view_params = {'det_channel_offset': np.where(good, channel_values - model_params['det_channel_offset'], 0.0),
                   'det_row_offset': np.where(good, row_values - model_params['det_row_offset'], 0.0)}
    if rotation:
        view_params['det_rotation'] = np.where(good, angles, np.median(angles[good]))
    return model_params, view_params


def correct_det_alignment(ct_model, sino, view_params, batch_size=30, devices=None):
    """Resample each view of a sinogram by the per-view corrections from :func:`fit_det_alignment`.

    Each view is rotated about the detector center by its ``det_rotation`` and moved by its
    ``det_channel_offset`` and ``det_row_offset`` deviations, converted from ALU to pixels with the
    model's pixel pitches, in one bicubic resampling.  A sample from outside the detector takes the
    nearest edge value.  Any key may be left out, and a missing key means no correction of that kind.
    The model supplies only the pixel pitches and is not changed.

    Args:
        ct_model (TomographyModel): The model of the scan.
        sino (numpy array or tensor): Sinogram, shape (num_views, num_det_rows, num_det_channels).
        view_params (dict): Arrays of length ``num_views`` under any of the keys
            ``det_channel_offset`` and ``det_row_offset`` (ALU) and ``det_rotation`` (radians), as
            :func:`fit_det_alignment` returns them.
        batch_size (int, optional): Views resampled at a time.  Defaults to 30.
        devices (sequence or None, optional): The first of these is the device used.  None uses the
            first CUDA device, or the default device when there is none.  Defaults to None.

    Returns:
        numpy.ndarray: The corrected sinogram, the shape of ``sino``.

    Example:
        >>> recon_direct = ct_model.recon_direct(sino)  # reconstruct object
        >>> model_params, view_params = mtp.fit_det_alignment(ct_model, sino, recon_direct, rotation=True)
        >>> ct_model.set_params(**model_params)  # update offset parameters
        >>> sino = mtp.correct_det_alignment(ct_model, sino, view_params)  # correct per-view jitter and rotation

        To skip the per-view shifts and remove only each view's rotation:

        >>> sino = mtp.correct_det_alignment(ct_model, sino, {'det_rotation': view_params['det_rotation']})
    """
    _pipeline.reject_shards('correct_det_alignment', sino=sino)
    sino = torch.as_tensor(np.asarray(sino)) if not isinstance(sino, torch.Tensor) else sino
    num_views = sino.shape[0]
    delta_row, delta_channel = (float(d) for d in ct_model.get_params(['delta_det_row', 'delta_det_channel']))

    def per_view(key, scale):
        values = view_params.get(key)
        if values is None:
            return np.zeros(num_views)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (num_views,):
            raise ValueError(f"view_params['{key}'] must have one value per view, got shape {values.shape}.")
        return values / scale

    angles = per_view('det_rotation', 1.0)
    offsets = np.stack([per_view('det_row_offset', delta_row), per_view('det_channel_offset', delta_channel)], axis=1)

    device = torch.device(_pipeline.permitted_devices(devices)[0])
    output = np.empty(tuple(sino.shape), dtype=np.float32 if sino.dtype == torch.float32 else np.float64)
    with torch.no_grad():
        for lo in range(0, num_views, batch_size):
            hi = min(lo + batch_size, num_views)
            batch = sino[lo:hi].to(device)
            out = _resample_views(batch, torch.as_tensor(angles[lo:hi], dtype=torch.float32, device=device),
                                  torch.as_tensor(offsets[lo:hi], dtype=torch.float32, device=device))
            output[lo:hi] = out.cpu().numpy()
    return output


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

    Deprecated.  Use :func:`fit_det_alignment` and :func:`correct_det_alignment` instead, which also
    put the global part of the shift into the model and can fit a detector rotation.  This function
    warns and will be removed in a later release.

    A 2D shift is estimated for each view by comparing it with the forward projection of
    ``recon_direct``, and the view is shifted by that amount.  This corrects small per-view motion
    of the object.  The whole shift of each view is applied to the data, and the model is left as it
    is.

    Args:
        ct_model (TomographyModel): The model of the scan.
        sino (numpy array or tensor): Sinogram, shape (num_views, num_det_rows, num_det_channels).
        recon_direct (numpy array or tensor): A first reconstruction of ``sino``, such as the output
            of :meth:`~mbirtorch.TomographyModel.recon_direct`.

    Returns:
        numpy.ndarray: The aligned sinogram, the shape of ``sino``.
    """
    warnings.warn("align_sino_views is deprecated; use fit_det_alignment and correct_det_alignment.",
                  DeprecationWarning, stacklevel=2)
    model_params, view_params = fit_det_alignment(ct_model, sino, recon_direct, rotation=False)
    # The model keeps its offsets, so each view's correction includes the global part.
    for key in ('det_channel_offset', 'det_row_offset'):
        view_params[key] = view_params[key] + (model_params[key] - float(ct_model.get_params(key)))
    return correct_det_alignment(ct_model, sino, view_params)
