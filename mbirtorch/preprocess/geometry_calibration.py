"""Geometry calibration from the sinogram.

``estimate_det_channel_offset`` estimates the detector channel offset, which sets the center of
rotation, from the sinogram.  It returns one number and changes nothing; the caller sets it with
``set_params``.  ``align_sino_views`` removes the small per-view shifts that remain after the
geometry is set.

The order of preprocessing matters.  Run in this order:
 1. defective-pixel interpolation, background offset correction, and stripe removal
 2. ``estimate_det_channel_offset``
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

__all__ = ['estimate_det_channel_offset', 'align_sino_views']

# This many views of the full sinogram are read per step when a band of rows is read.
_READ_VIEW_BATCH = 64

_NUM_COARSE = 11                      # candidates scored on the coarse grid of a search
_NUM_ROWS = 16                        # detector rows compared, before the cone-beam limit

# A scan covers a full rotation when no gap between neighboring view angles exceeds both this
# many times the median gap and this many radians.  A scan over a half rotation is refused.
_MAX_GAP_RATIO = 3.0
_MAX_GAP = math.radians(5.0)


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
    """Estimate ``det_channel_offset`` from the sinogram by comparing each view with its opposite.

    In a scan over a full rotation every ray is measured twice, once from each side, so a view and
    its mirrored opposite differ by a shift of twice the channel offset.  The shift is found by
    correlating high-passed profiles of the two over a band of rows at the central plane, as in
    :func:`align_sino_views`, and refined to a fraction of a channel.  The function warns when the
    match is weak.

    A scan without opposite views, such as a short scan, a helical scan, or a multiaxis scan, gets
    a warning and the model's current value back unchanged.

    Args:
        ct_model (TomographyModel): the model of the scan.  Not modified.
        sino (ndarray or tensor): the sinogram.  Not modified.

    Returns:
        float: the estimate in ALU, to set with ``ct_model.set_params(det_channel_offset=...)``.
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
