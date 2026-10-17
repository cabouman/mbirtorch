"""Tests of the redundancy weighting of cone-beam FDK.

When the views of a circular scan do not cover a whole number of rotations, or when a helical
scan sees each height over such a range, ``ConeBeamModel.recon_fdk`` multiplies the sinogram by
the weights of ``redundancy_weights`` or ``helical_redundancy_weights`` before the filter.  The
tests check the weights themselves, the detection of the scan range, and the reconstruction
of an off-center object against the reconstruction from a full rotation.
"""

import numpy as np
import pytest
import torch

import mbirtorch

# The small problem of the reconstruction tests.  The object is off center, so a wrong fan-angle
# sign or missing weights leaves a large error.
_NUM_VIEWS = 72
_NUM_ROWS = 8
_NUM_CHANNELS = 48
_SOURCE_DETECTOR_DIST = 4 * _NUM_CHANNELS
_SOURCE_ISO_DIST = 2 * _NUM_CHANNELS
# A weighted scan must reconstruct as well as a full rotation, within this margin on the
# relative error.
_WEIGHTED_MARGIN = 0.02
# The plain FDK of a short scan must be worse than this.
_PLAIN_SHORT_SCAN_MIN_ERROR = 0.08
# The plain FDK of more than one rotation is less wrong than that of a short scan, but the
# weights must still improve it by at least this factor.  (An odd number of half rotations,
# such as one and a half, measures almost every line the same number of times, so the plain
# FDK is nearly right there; the tests use one and a quarter rotations instead.)
_OVERSCAN_MIN_IMPROVEMENT = 1.2
# The sum of the weights of the measurements of one line, on a fine scan, within this
# tolerance of its exact value.
_REDUNDANCY_TOL = 1e-3
# The helical tests use this many views per rotation and this many rotations.
_HELICAL_VIEWS_PER_ROTATION = 72
_HELICAL_ROTATIONS = 4
_HELICAL_NUM_ROWS = 16


def _model(angles, device, helical_z_shifts=None):
    shape = (len(angles), _HELICAL_NUM_ROWS if helical_z_shifts is not None else _NUM_ROWS,
             _NUM_CHANNELS)
    model = mbirtorch.ConeBeamModel(shape, angles, source_detector_dist=_SOURCE_DETECTOR_DIST,
                                    source_iso_dist=_SOURCE_ISO_DIST,
                                    helical_z_shifts=helical_z_shifts)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


def _fan_angle(model):
    """The full in-plane angle that the detector subtends at the source."""
    num_channels = int(model.get_params('sinogram_shape')[2])
    delta_det_channel, source_detector_dist = model.get_params(
        ['delta_det_channel', 'source_detector_dist'])
    return 2.0 * np.arctan(0.5 * num_channels * delta_det_channel / source_detector_dist)


def _views(num_views, start, scan_range):
    """Equally spaced views over [start, start + scan_range) with the end point excluded."""
    return start + scan_range * np.arange(num_views) / num_views


def _off_center_image(model):
    """A cylinder away from the rotation axis on a wider block, as a (rows, cols) array."""
    num_rows, num_cols, _ = model.get_params('recon_shape')
    delta_voxel = model.get_params('delta_voxel')
    y = (np.arange(num_rows) - (num_rows - 1) / 2) * delta_voxel
    x = (np.arange(num_cols) - (num_cols - 1) / 2) * delta_voxel
    half = 0.5 * num_cols * delta_voxel
    radius_sq = (x[None, :] - 0.3 * half) ** 2 + (y[:, None] + 0.2 * half) ** 2
    cylinder = radius_sq < (0.25 * half) ** 2
    block = (np.abs(x[None, :]) < 0.7 * half) & (np.abs(y[:, None]) < 0.5 * half)
    return (0.1 * block + 0.2 * cylinder).astype(np.float32)


def _off_center_object(model):
    """The off-center image repeated over every slice, as a torch tensor."""
    num_slices = model.get_params('recon_shape')[2]
    phantom = np.repeat(_off_center_image(model)[:, :, None], num_slices, axis=2)
    return torch.as_tensor(phantom, device=model.torch_device)


def _relative_error(recon, phantom, slices):
    """The relative error over the given slices of the volume."""
    recon = torch.as_tensor(np.asarray(recon), device=phantom.device)
    diff = recon[:, :, slices] - phantom[:, :, slices]
    return float(diff.norm() / phantom[:, :, slices].norm())


def _interior_error(recon, phantom):
    """The relative error away from the slice ends of the volume."""
    num_slices = phantom.shape[2]
    return _relative_error(recon, phantom, slice(num_slices // 4, num_slices - num_slices // 4))


def test_scan_range_and_whole_rotations(device):
    """Whole rotations have no weights; a short scan and one and a half rotations have a
    weight for every view and channel."""
    # The model stores the angles as float32, so the range is exact to about 1e-7.
    full = _model(_views(_NUM_VIEWS, 0.3, 2 * np.pi), device)
    assert full.get_scan_range() == pytest.approx(2 * np.pi, rel=1e-6)
    assert full.redundancy_weights() is None
    assert _model(_views(2 * _NUM_VIEWS, 0.3, 4 * np.pi), device).redundancy_weights() is None
    short_range = np.pi + _fan_angle(full)
    for scan_range in (short_range, 3 * np.pi):
        model = _model(_views(_NUM_VIEWS, 0.3, scan_range), device)
        assert model.get_scan_range() == pytest.approx(scan_range, rel=1e-6)
        weights = model.redundancy_weights()
        assert weights.shape == (_NUM_VIEWS, _NUM_CHANNELS) and weights.dtype == np.float32
        assert np.all(weights >= 0) and np.all(np.isfinite(weights))


def test_too_short_scan_warns(device):
    """A scan shorter than 180 degrees plus the fan angle gets no weights and a warning."""
    model = _model(_views(_NUM_VIEWS, 0.0, 0.9 * np.pi), device)
    with pytest.warns(UserWarning, match='less than 180 degrees'):
        assert model.redundancy_weights() is None


@pytest.mark.parametrize('extra', [0.2, np.pi + 0.5])
def test_weights_of_one_line_add_to_one(device, extra):
    """On a fine scan, the weights of all the measurements of a line in the central plane add
    up to the full-rotation factor scan_range / pi.  The scan is a short scan or about one
    and a quarter rotations."""
    num_views = 3000
    scan_range = np.pi + _fan_angle(_model(_views(1, 0.0, 1.0), device)) + extra
    model = _model(_views(num_views, 0.0, scan_range), device)
    weights = model.redundancy_weights().astype(np.float64)
    delta_det_channel, det_channel_offset, source_detector_dist = model.get_params(
        ['delta_det_channel', 'det_channel_offset', 'source_detector_dist'])
    u = (np.arange(_NUM_CHANNELS) - (_NUM_CHANNELS - 1) / 2) * delta_det_channel \
        - det_channel_offset
    fan = np.arctan(u / source_detector_dist)
    beta = (np.arange(num_views) + 0.5) * scan_range / num_views
    checked = 0
    for channel in range(0, _NUM_CHANNELS, 4):
        mirror = _NUM_CHANNELS - 1 - channel  # The channel at -u.
        # The same line is measured at beta + 2 pi n in this channel and at
        # beta + pi - 2 atan(u / sdd) + 2 pi n in the mirror channel.
        total = np.zeros(num_views)
        for n in range(-2, 3):
            same = beta + 2 * np.pi * n
            inside = (same >= 0) & (same <= scan_range)
            total += np.where(inside, np.interp(same, beta, weights[:, channel]), 0.0)
            conjugate = beta + np.pi - 2 * fan[channel] + 2 * np.pi * n
            inside = (conjugate >= 0) & (conjugate <= scan_range)
            total += np.where(inside, np.interp(conjugate, beta, weights[:, mirror]), 0.0)
        # Views near the ends of a short scan have their partners outside the range of the
        # interpolation; keep the views whose own bin and partners lie inside.
        keep = (beta > scan_range / num_views) & (beta < scan_range - scan_range / num_views)
        assert np.allclose(total[keep], scan_range / np.pi, atol=_REDUNDANCY_TOL), channel
        checked += int(np.sum(keep))
    assert checked > 0


@pytest.mark.parametrize('start', [0.0, 1.0])
def test_short_scan_fdk_matches_full_rotation(device, start):
    """FDK of a short scan with the weights reconstructs an off-center object as well as FDK
    of a full rotation.  Without the weights, the short scan is far off."""
    full = _model(_views(_NUM_VIEWS, start, 2 * np.pi), device)
    phantom = _off_center_object(full)
    full_error = _interior_error(full.recon_fdk(full.forward_project(phantom)), phantom)
    short = _model(_views(_NUM_VIEWS, start, np.pi + _fan_angle(full)), device)
    sinogram = short.forward_project(phantom)
    weighted_error = _interior_error(short.recon_fdk(sinogram), phantom)
    plain_error = _interior_error(short.recon_fdk(sinogram, redundancy_weighting=False),
                                  phantom)
    assert weighted_error < full_error + _WEIGHTED_MARGIN, (weighted_error, full_error)
    assert plain_error > _PLAIN_SHORT_SCAN_MIN_ERROR, plain_error


def test_overscan_fdk_matches_full_rotation(device):
    """FDK of one and a quarter rotations with the weights reconstructs as well as FDK of a
    full rotation, and better than the plain FDK of the same views."""
    full = _model(_views(_NUM_VIEWS, 0.0, 2 * np.pi), device)
    phantom = _off_center_object(full)
    full_error = _interior_error(full.recon_fdk(full.forward_project(phantom)), phantom)
    over = _model(_views(5 * _NUM_VIEWS // 4, 0.0, 2.5 * np.pi), device)
    sinogram = over.forward_project(phantom)
    weighted_error = _interior_error(over.recon_fdk(sinogram), phantom)
    plain_error = _interior_error(over.recon_fdk(sinogram, redundancy_weighting=False),
                                  phantom)
    assert weighted_error < full_error + _WEIGHTED_MARGIN, (weighted_error, full_error)
    assert plain_error > _OVERSCAN_MIN_IMPROVEMENT * weighted_error, (plain_error,
                                                                      weighted_error)


def _helical_model(pitch, device):
    """A helical scan of several rotations at the given pitch, with the z shifts linear in the
    view angle."""
    num_views = _HELICAL_VIEWS_PER_ROTATION * _HELICAL_ROTATIONS
    angles = _views(num_views, 0.0, 2 * np.pi * _HELICAL_ROTATIONS)
    circular = _model(angles, device, helical_z_shifts=np.zeros(num_views))
    det_height_iso = _HELICAL_NUM_ROWS * circular.get_params('delta_det_row') \
        / circular.get_magnification()
    travel_per_rotation = pitch * det_height_iso
    z_shifts = travel_per_rotation * angles / (2 * np.pi)
    z_shifts = z_shifts - 0.5 * (z_shifts[0] + z_shifts[-1])
    return _model(angles, device, helical_z_shifts=z_shifts)


def _covered_slices(model):
    """The slices that the helical scan sees over the whole range of one height."""
    num_rows = int(model.get_params('sinogram_shape')[1])
    det_height_iso = num_rows * model.get_params('delta_det_row') / model.get_magnification()
    z_shifts = np.asarray(model.get_params('view_params_array'))[:, 1]
    z_k = model.recon_slice_z()
    covered = (z_k >= z_shifts.min() + 0.5 * det_height_iso) \
        & (z_k <= z_shifts.max() - 0.5 * det_height_iso)
    indices = np.flatnonzero(covered)
    assert indices.size >= 2
    return slice(int(indices[0]), int(indices[-1]) + 1)


@pytest.mark.parametrize('pitch', [1.0, 1.5, 0.8])
def test_helical_fdk_weights(device, pitch):
    """At pitch 1 each height is seen over one rotation and no weights apply.  At pitch 1.5
    (a short scan per height) and at pitch 0.8 (one and a quarter rotations per height) the
    weights bring the FDK of the covered slices to the accuracy of a circular full rotation,
    and the plain FDK is worse."""
    helical = _helical_model(pitch, device)
    if pitch == 1.0:
        assert helical.helical_redundancy_weights() is None
        return
    weights = helical.helical_redundancy_weights()
    assert weights.shape == (_HELICAL_NUM_ROWS, _NUM_CHANNELS) and np.all(weights >= 0)
    phantom = _off_center_object(helical)
    sinogram = helical.forward_project(phantom)
    covered = _covered_slices(helical)
    weighted_error = _relative_error(helical.recon_fdk(sinogram), phantom, covered)
    plain_error = _relative_error(helical.recon_fdk(sinogram, redundancy_weighting=False),
                                  phantom, covered)
    # The circular reference: a full rotation of the same geometry and volume.
    num_views = _HELICAL_VIEWS_PER_ROTATION
    circular = _model(_views(num_views, 0.0, 2 * np.pi), device,
                      helical_z_shifts=np.zeros(num_views))
    circular.set_params(recon_shape=helical.get_params('recon_shape'),
                        recon_slice_offset=helical.get_params('recon_slice_offset'))
    reference_error = _relative_error(circular.recon_fdk(circular.forward_project(phantom)),
                                      phantom, covered)
    assert weighted_error < reference_error + _WEIGHTED_MARGIN, (weighted_error, reference_error)
    assert plain_error > _OVERSCAN_MIN_IMPROVEMENT * weighted_error, (plain_error,
                                                                      weighted_error)


def test_recon_fdk_leaves_the_input_unchanged(device):
    """The weighting makes a new tensor and does not change the caller's sinogram."""
    model = _model(_views(_NUM_VIEWS, 0.0, np.pi + _fan_angle(_model([0.0], device))), device)
    phantom = _off_center_object(model)
    sinogram = torch.as_tensor(model.forward_project(phantom), device=model.torch_device)
    before = sinogram.clone()
    model.recon_fdk(sinogram)
    assert torch.equal(sinogram, before)
