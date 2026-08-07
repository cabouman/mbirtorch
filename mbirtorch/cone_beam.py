"""ConeBeamModel, ported from mbirjax.cone_beam: flat and curved detectors,
circular and helical scans, the multi-device banding seams, and DC damping.

Structure: cone projection is two separable fans.  The HORIZONTAL fan maps a
voxel to detector channels exactly as in parallel beam, except the projected
coordinate, width, and weight are magnification-dependent PER PIXEL.  The
VERTICAL fan maps each slice of a voxel cylinder to a RANGE of detector rows
(the cone angle); both directions derive that row map from the single affine
pair (m0, W_p_r) of :func:`_cone_vertical_affine`.  The forward vertical fan
is formulated from the DETECTOR side (for each detector row, which voxels
project onto it), matching the back projector by construction so the pair
stays exactly adjoint; the back vertical fan gathers each pixel's detector
column onto the recon slices with the weight rule

    A = clip((W_p_r + 1) / 2 - |m_p - m|, 0, min(1, W_p_r)) / cos_phi

(validity-masked, then raised to coeff_power -- the mbirjax
vertical_fan_band_gather rule, including its historical arithmetic order).

The drivers batch over views like the parallel drivers; the dominant
transients are (view_batch, P, S) and (view_batch, P, R), so the effective
view batch shrinks under the same transient budget.
"""

import numpy as np
import torch
import warnings

from .horizontal_fan import fan_back_batch, fan_forward_batch
from .projectors import maybe_compile
from .tomography_model import TomographyModel
from .vcd_utils import get_support_radius

_F32 = torch.float32

# (a, b, p, c) for the slice damping s_k = (c t^p + a b^p)/(t^p + b^p),
# t_k = L |z_k| / (R dz) -- the "C4" preconditioner.  Not a public parameter;
# for sweeps set ct_model._dc_damping = (a, b, p, c) or None.  ON by default,
# matching mbirjax (the update direction is a positive definite reshaping of
# the gradient, so the MAP fixed point is unchanged; only the trajectory
# differs -- and the convergence-parity gate requires matching trajectories).
_DC_DAMPING_DEFAULT = (0.25, 100.0, 0.7, 0.5)


def _dc_damped_update_direction(forward_grad, prior_grad, forward_hess,
                                prior_hess, s_row):
    # d = -(g - (1 - s_k) gbar_k) / H with gbar_k the H^-1-weighted slice mean
    # of g over the subset's pixels; s == 1 reduces to the base -(g / H).
    g = forward_grad + prior_grad
    h_inv = 1.0 / (forward_hess + prior_hess)
    gbar = torch.sum(h_inv * g, dim=0) / torch.sum(h_inv, dim=0)
    return -(g - ((1.0 - s_row) * gbar)[None, :]) * h_inv


# ── geometry chains (pure, compiled) ─────────────────────────────────────────
def _cone_pixel_xy_mag(pixel_indices, angles, num_rows, num_cols, delta_voxel,
                       delta_voxel_row, magnification, source_detector_dist):
    """Rotated in-plane coordinates and the per-pixel magnification.

    Returns x (Vb, P), y (Vb, P), pixel_mag (Vb, P).  The magnification
    expression 1 / (1/M - y/SDD) is valid even at SDD = inf (mbirjax's
    geometry_xyz_to_uv_mag).
    """
    row_index = (pixel_indices // num_cols).to(_F32)
    col_index = (pixel_indices % num_cols).to(_F32)
    # Note the change in order from (i, j) to (y, x) (recon_ijk_to_xyz).
    y_tilde = delta_voxel_row * (row_index - (num_rows - 1) / 2.0)
    x_tilde = delta_voxel * (col_index - (num_cols - 1) / 2.0)
    cosine = torch.cos(angles)[:, None]
    sine = torch.sin(angles)[:, None]
    x = cosine * x_tilde[None, :] - sine * y_tilde[None, :]
    y = sine * x_tilde[None, :] + cosine * y_tilde[None, :]
    pixel_mag = 1.0 / (1.0 / magnification - y / source_detector_dist)
    return x, y, pixel_mag


def _cone_horizontal_data(pixel_indices, angles, num_rows, num_cols, num_channels,
                          delta_voxel, delta_voxel_row, delta_det_channel,
                          det_channel_offset, magnification, source_detector_dist,
                          use_curved_detector):
    """The horizontal fan's inputs for a view batch (compute_horizontal_data +
    the wrapper's center rounding).  All per-PIXEL: n_p, centers (int32),
    W_p_c, weight_scale are (Vb, P) -- the hfan contract (horizontal_fan.py)
    -- and pixel_mag is returned for the vertical fan.
    """
    x, y, pixel_mag = _cone_pixel_xy_mag(pixel_indices, angles, num_rows, num_cols,
                                         delta_voxel, delta_voxel_row,
                                         magnification, source_detector_dist)
    det_center_channel = (num_channels - 1) / 2.0
    if not use_curved_detector:
        u = pixel_mag * x
        theta = torch.atan2(u, torch.as_tensor(source_detector_dist, dtype=_F32,
                                               device=u.device))
    else:
        source_iso_dist = source_detector_dist / magnification
        u = source_detector_dist * torch.atan2(x, source_iso_dist - y)
        theta = u / source_detector_dist

    n_p = (u + det_channel_offset) / delta_det_channel + det_center_channel
    footprint_xy = torch.maximum((angles[:, None] - theta).cos().abs() * delta_voxel,
                                 (angles[:, None] - theta).sin().abs() * delta_voxel_row)
    if not use_curved_detector:
        # Foreshortening correction: the footprint widens at oblique angles on a
        # flat detector; the arc parameterisation absorbs it on a curved one.
        W_p_c = pixel_mag * (footprint_xy / delta_det_channel) / torch.cos(theta)
    else:
        W_p_c = pixel_mag * (footprint_xy / delta_det_channel)
    weight_scale = (delta_voxel_row * delta_voxel) / footprint_xy
    centers = torch.round(n_p).to(torch.int32)
    return n_p, centers, W_p_c, weight_scale, pixel_mag


def _cone_vertical_affine(pixel_mag, z_shifts, num_slices, delta_voxel_slice,
                          delta_det_row, det_row_offset, recon_slice_offset,
                          num_rows_r):
    """The vertical fan's affine map from GLOBAL slice index to detector row.

    Each (view, pixel) cylinder projects onto the detector rows through the
    affine map

        m(v, p, l) = m0 + W_p_r * l

    with ``l`` the GLOBAL slice index (band-independent by construction: m0 is
    anchored at global slice 0, so a slice band just restricts the range of
    ``l``).  The two projection directions consume the two algebraic forms of
    the SAME map -- the back body evaluates it directly (slice -> row), the
    forward body inverts it (row -> slice, ``k_m = (m - m0) / W_p_r``) -- so
    deriving both from this one pair keeps them consistent by construction
    instead of by parallel edits.

    THE (m0, W_p_r) PAIR IS THE SANCTIONED GEOMETRY BRIDGE: a fused kernel
    consumes exactly this pair (plus the hfan data contract of
    horizontal_fan.py) and needs nothing else of the cone vertical geometry.
    Extending the vertical fan means extending this function, not the bodies.

    Args:
        pixel_mag: per-(view, pixel) magnification, (Vb, P).
        z_shifts: per-view helical z shift, (Vb,).
        num_slices (int): the FULL recon slice count (the z anchor stays on it
            whatever band is requested).
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset:
            the vertical geometry scalars.
        num_rows_r (int): detector row count.

    Returns:
        (m0, W_p_r, z_offset): the row-center anchor at global slice 0 (Vb, P),
        the rows-per-slice slope (Vb, P), and the per-view z offset (Vb,) that
        both bodies' cone-angle chains share.
    """
    z_offset = recon_slice_offset - z_shifts                     # (Vb,)
    det_center_row = (num_rows_r - 1) / 2.0
    W_p_r = pixel_mag * delta_voxel_slice / delta_det_row        # (Vb, P)
    z_at_slice_0 = z_offset[:, None] - delta_voxel_slice * (num_slices - 1) / 2.0
    m0 = (pixel_mag * z_at_slice_0 + det_row_offset) / delta_det_row \
        + det_center_row                                         # (Vb, P)
    return m0, W_p_r, z_offset


def _cone_forward_view_batch(values, pixel_indices, view_params_batch,
                             num_rows_r, num_channels, num_recon_rows,
                             num_recon_cols, num_slices, delta_voxel,
                             delta_voxel_row, delta_voxel_slice,
                             delta_det_channel, delta_det_row,
                             det_channel_offset, det_row_offset, recon_slice_offset,
                             magnification, source_detector_dist,
                             use_curved_detector, psf_radius, bp_psf_radius,
                             slice_start=0, plan=None):
    """Cone forward for one view batch: the detector-side vertical fan, then the
    per-pixel horizontal fan scatter.  Returns (Vb, R, C).

    ``slice_start`` supports the banded sharded forward: ``values`` may be a
    slice BAND (P, L) whose global slice indices are [slice_start,
    slice_start + L); the z geometry stays anchored on the FULL num_slices
    center, gathers use band-local storage indices, and taps outside the band
    contribute zero -- so summing the per-band outputs over a tiling of the
    slice axis reproduces the unbanded projection exactly.  The default 0
    with L == num_slices is the unbanded case, bit-identical to before.

    ``plan`` is the memoization slot for a future sorted/CSR stream variant
    (per pixel-subset x view-range); unused today."""
    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    vb, num_pixels = n_p.shape
    dev = values.device

    # ── vertical fan (detector side; forward_vertical_fan_one_pixel) ─────────
    m0, W_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    # Scale the cylinder values by 1/cos(phi): phi is the vertical cone angle of
    # each (pixel, slice) voxel; 1/cos is the projection length through a voxel.
    band_len = values.shape[1]
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    z = (delta_voxel_slice * (k - (num_slices - 1) / 2.0))[None, None, :] \
        + z_offset[:, None, None]                                # (Vb, 1, L)
    v_slices = pixel_mag.unsqueeze(-1) * z                       # (Vb, P, L)
    cos_phi = torch.cos(torch.atan2(v_slices, torch.as_tensor(
        source_detector_dist, dtype=_F32, device=dev)))
    scaled_values = values[None, :, :] / cos_phi                 # (Vb, P, S)

    # Detector rows -> voxel fractional indices: the INVERSE of the shared
    # affine map (the back body below evaluates the direct form), so k_m is
    # affine in the row index with slope 1/W_p_r.
    m = torch.arange(num_rows_r, dtype=_F32, device=dev)         # (R,)
    k_m = (m[None, None, :] - m0.unsqueeze(-1)) / W_p_r.unsqueeze(-1)
    k_center = torch.round(k_m).to(torch.int64)                  # (Vb, P, R)

    slope = W_p_r.unsqueeze(-1)
    L_max_r = torch.clamp(W_p_r, max=1.0).unsqueeze(-1)
    m_p = slope * (k_center.to(_F32) - k_m)                      # projection offset

    det_col = torch.zeros((vb, num_pixels, num_rows_r), dtype=_F32, device=dev)
    for k_off in range(-bp_psf_radius, bp_psf_radius + 1):
        k_ind = k_center + k_off
        A = torch.clamp((slope + 1.0) / 2.0 - (m_p + slope * k_off).abs(), min=0.0)
        A = torch.minimum(A, L_max_r)
        A = A * ((k_ind >= slice_start)
                 & (k_ind < slice_start + band_len)).to(_F32)
        g = torch.gather(scaled_values, 2,
                         (k_ind - slice_start).clamp(0, band_len - 1))
        det_col = det_col + A * g

    # ── horizontal fan scatter (the shared kernel; per-view values) ──────────
    acc = fan_forward_batch((n_p, centers, W_p_c, weight_scale), det_col,
                            num_channels, psf_radius)
    return acc.permute(0, 2, 1)


def _cone_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                          num_rows_r, num_channels, num_recon_rows,
                          num_recon_cols, num_slices, delta_voxel,
                          delta_voxel_row, delta_voxel_slice,
                          delta_det_channel, delta_det_row, det_channel_offset,
                          det_row_offset, recon_slice_offset, magnification,
                          source_detector_dist, use_curved_detector, psf_radius,
                          bp_psf_radius, coeff_power=1, slice_start=0,
                          band_slices=None, plan=None):
    """Cone back projection for one view batch, summed over the batch's views:
    horizontal fan gather -> per-pixel detector columns -> vertical fan gather
    onto the slices.  Returns (P, S) -- or (P, band_slices) when a slice BAND
    [slice_start, slice_start + band_slices) is requested (the banded sharded
    back); the z geometry stays anchored on the full num_slices center, and
    the default (0, None) is the unbanded case, bit-identical to before.

    ``plan`` is the memoization slot for a future sorted/CSR stream variant
    (per pixel-subset x view-range); unused today."""
    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, pixel_mag = _cone_horizontal_data(
        pixel_indices, angles, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset,
        magnification, source_detector_dist, use_curved_detector)
    vb, num_pixels = n_p.shape
    dev = sino_batch.device

    # ── horizontal fan gather (the shared kernel, view axis kept) ────────────
    sino_T = sino_batch.permute(0, 2, 1).contiguous()            # (Vb, C, R)
    det_col = fan_back_batch(sino_T, (n_p, centers, W_p_c, weight_scale),
                             num_channels, psf_radius,
                             coeff_power=coeff_power, reduce_views=False)

    # ── vertical fan gather (compute_vertical_data + vertical_fan_band_gather)
    m0, W_p_r, z_offset = _cone_vertical_affine(
        pixel_mag, z_shifts, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, recon_slice_offset, num_rows_r)

    band_len = num_slices if band_slices is None else band_slices
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    z = (delta_voxel_slice * (k - (num_slices - 1) / 2.0))[None, None, :] \
        + z_offset[:, None, None]
    v_slices = pixel_mag.unsqueeze(-1) * z                       # (Vb, P, S)
    sdd_t = torch.as_tensor(source_detector_dist, dtype=_F32, device=dev)
    cos_phi = torch.cos(torch.atan2(v_slices, sdd_t))
    # Slice -> detector row: the DIRECT form of the shared affine map (the
    # forward body above inverts it).
    slope = W_p_r.unsqueeze(-1)
    m_p = m0.unsqueeze(-1) + slope * k[None, None, :]            # (Vb, P, S)
    m_center = torch.round(m_p).to(torch.int64)
    L_max_r = torch.clamp(slope, max=1.0)

    out = torch.zeros((num_pixels, band_len), dtype=_F32, device=dev)
    for m_off in range(-psf_radius, psf_radius + 1):
        mm = m_center + m_off
        L = torch.clamp((slope + 1.0) / 2.0 - (m_p - mm.to(_F32)).abs(), min=0.0)
        A = torch.minimum(L, L_max_r) / cos_phi
        A = A * ((mm >= 0) & (mm < num_rows_r)).to(_F32)
        if coeff_power != 1:
            A = A ** coeff_power
        g = torch.gather(det_col, 2, mm.clamp(0, num_rows_r - 1))
        out = out + torch.einsum("vps,vps->ps", A, g)
    return out


class ConeBeamModel(TomographyModel):
    """
    A class designed for handling forward and backward projections in a cone
    beam geometry, extending :class:`TomographyModel`.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels).
        angles (ndarray): 1D array of projection angles in radians.
        source_detector_dist (float): Distance from source to detector in ALU.
        source_iso_dist (float): Distance from source to iso (rotation center).
        helical_z_shifts (ndarray, optional): per-view z shift for helical
            scans; None (default) is a circular scan.
        use_curved_detector (bool): False (default) = flat panel; True = a
            cylindrical detector of radius source_detector_dist.
        device, view_batch_size, compile_mode: as in ParallelBeamModel.
    """

    def __init__(self, sinogram_shape, angles, source_detector_dist, source_iso_dist,
                 helical_z_shifts=None, use_curved_detector=False, device='auto',
                 view_batch_size=64, compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32).flatten()
        if helical_z_shifts is None:
            helical_z_shifts = np.zeros_like(angles)
        else:
            helical_z_shifts = np.asarray(helical_z_shifts, dtype=np.float32).flatten()
        if helical_z_shifts.shape != angles.shape:
            raise ValueError("Incompatible view dependent vector lengths: all "
                             "view-dependent vectors must have the same length.")
        view_params_array = np.stack([angles, helical_z_shifts], axis=1)
        super().__init__(sinogram_shape, device=device,
                         view_batch_size=view_batch_size, compile_mode=compile_mode,
                         geometry_type='cone', view_params_name='view_params_array',
                         view_params_array=view_params_array,
                         source_detector_dist=source_detector_dist,
                         source_iso_dist=source_iso_dist,
                         recon_slice_offset=0.0, axial_pad_fraction=0.0,
                         use_curved_detector=use_curved_detector)

    _dc_damping = _DC_DAMPING_DEFAULT

    def create_projectors(self):
        super().create_projectors()
        # Warm the DC-damping profile and its per-device compiled instances
        # EAGERLY (params- and layout-dependent): built lazily it raced the
        # per-device worker threads on the first subset.
        self._dc_damping_slice_profile()

    def _view_batch_bodies(self):
        return _cone_forward_view_batch, _cone_back_view_batch

    def _view_batch_args(self):
        gp_names = ['delta_det_row', 'delta_det_channel', 'det_row_offset',
                    'det_channel_offset', 'source_detector_dist', 'delta_voxel',
                    'voxel_row_aspect', 'voxel_slice_aspect', 'recon_slice_offset',
                    'use_curved_detector']
        (ddr, ddc, dro, dco, sdd, dv, vra, vsa, rso, curved) = self.get_params(gp_names)
        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')
        psf_radius, bp_psf_radius = self.get_psf_radii()
        return dict(num_rows_r=sinogram_shape[1], num_channels=sinogram_shape[2],
                    num_recon_rows=recon_shape[0], num_recon_cols=recon_shape[1],
                    num_slices=recon_shape[2], delta_voxel=dv,
                    delta_voxel_row=vra * dv, delta_voxel_slice=vsa * dv,
                    delta_det_channel=ddc, delta_det_row=ddr,
                    det_channel_offset=dco, det_row_offset=dro,
                    recon_slice_offset=rso, magnification=self.get_magnification(),
                    source_detector_dist=sdd, use_curved_detector=curved,
                    psf_radius=psf_radius, bp_psf_radius=bp_psf_radius)

    def _transient_cols(self, band_cols):
        # The cone bodies hold (Vb, P, S) and (Vb, P, R) transients whatever
        # the requested band, so the budget width is params-derived (the
        # calibrated rule; see the base hook's docstring).
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return max(int(recon_shape[2]), int(sinogram_shape[1]))

    def _dc_damping_slice_profile(self):
        """The per-slice damping vectors s_k, split per device, or None if
        disabled.  Circular: s_k from t_k = L |z_k| / (R dz); helical:
        view-averaged.  The full profile is computed on the host, any padded
        slice tail is filled with 1.0 (no damping -- inert, matching the
        forced-zero padded slices), and each device gets its own slice band
        plus its own compiled damping instance (per-device instances, like
        the subset updater's other compiled units).  Cached against the parameters
        and the device layout; _invalidate_device_caches drops it.

        Returns:
            (profiles, fns): per-device (local_slices,) tensors and compiled
            direction helpers, in device order; or None when disabled.
        """
        cfg = self._dc_damping
        if cfg is None:
            return None
        recon_shape = self.get_params('recon_shape')
        dv, slice_aspect, oz = self.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_slice_offset'])
        R = self.get_params('source_iso_dist')
        z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        rp = self.recon_placement
        key = (tuple(cfg), tuple(recon_shape), dv, slice_aspect, oz, R,
               float(z_shifts.min()), float(z_shifts.max()),
               tuple(str(d) for d in rp.devices), rp.padded_size)
        cache = getattr(self, '_dc_damping_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]

        a, b, p, c = cfg
        nz = recon_shape[2]
        L = recon_shape[0] * dv
        dz = slice_aspect * dv
        z = (np.arange(nz) - (nz - 1) / 2.0) * dz + oz

        def profile(t):
            return (c * t ** p + a * b ** p) / (t ** p + b ** p)

        if z_shifts.max() - z_shifts.min() == 0:
            s_prof = profile(L * np.abs(z - z_shifts[0]) / (R * dz))
        else:
            t = L * np.abs(z[:, None] - z_shifts[None, :]) / (R * dz)
            s_prof = profile(t).mean(axis=1)
        total = rp.padded_size if rp.padded_size is not None else nz
        if total > nz:
            s_prof = np.concatenate([s_prof, np.ones(total - nz)])
        profiles, fns = [], []
        for i, (dev, (s0, s1)) in enumerate(rp.shard_ranges(total)):
            profiles.append(torch.as_tensor(
                s_prof[s0:s1].astype(np.float32), device=dev))
            fns.append(maybe_compile(_dc_damped_update_direction,
                                     self.compile_enabled, instance_key=i))
        self._dc_damping_cache = (key, (profiles, fns))
        return profiles, fns

    def _get_update_direction(self, forward_grad, prior_grad, forward_hess,
                              prior_hess, pixel_indices, dev_index=0):
        # DC damping of each slice's update (qGGMRF and prox paths alike),
        # applied per shard: the slice means inside the damping formula are
        # shard-local, which is exactly right under slice sharding (every
        # slice lives on one shard).
        prof = self._dc_damping_slice_profile()
        if prof is None:
            return super()._get_update_direction(forward_grad, prior_grad,
                                                 forward_hess, prior_hess,
                                                 pixel_indices,
                                                 dev_index=dev_index)
        profiles, fns = prof
        prior_hess_t = (prior_hess if torch.is_tensor(prior_hess)
                        else torch.as_tensor(prior_hess, dtype=_F32,
                                             device=forward_grad.device))
        return fns[dev_index](forward_grad, prior_grad, forward_hess,
                              prior_hess_t, profiles[dev_index])


    def get_magnification(self):
        """magnification = source_detector_dist / source_iso_dist (1 at inf)."""
        source_detector_dist, source_iso_dist = self.get_params(
            ['source_detector_dist', 'source_iso_dist'])
        if np.isinf(source_detector_dist):
            return 1
        return source_detector_dist / source_iso_dist

    def verify_valid_params(self):
        """Check that all parameters are compatible for a reconstruction."""
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(
            ['sinogram_shape', 'view_params_array'])
        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])
        num_views, num_det_rows = sinogram_shape[:2]
        if view_params_array is None:
            raise ValueError("view_params_array was not set.")
        if tuple(view_params_array.shape) != (num_views, 2):
            raise ValueError('Number of view dependent parameter vectors must '
                             'equal the number of views.')
        if voxel_row_aspect <= 0 or voxel_slice_aspect <= 0:
            raise ValueError('Voxel aspect ratios must be positive.')
        # Check for cone angle > 45 degrees.
        source_detector_dist, delta_det_row, det_row_offset = self.get_params(
            ['source_detector_dist', 'delta_det_row', 'det_row_offset'])
        half_detector_height = delta_det_row * num_det_rows / 2 + abs(det_row_offset)
        if half_detector_height > source_detector_dist:
            warnings.warn('Cone angle is more than 45 degrees.  This will likely '
                          'produce recon artifacts.')

    def get_psf_radii(self):
        """The two integer psf radii, PURE (no attribute side effects): the
        channel radius from the maximum magnification (the forward horizontal
        tap radius and the back vertical tap radius) and the back/vertical
        voxel-per-detector radius (the forward vertical tap radius).  The
        directions deliberately use DIFFERENT vertical radii -- forward
        gathers voxels per detector row (bp radius), back gathers rows per
        voxel (psf radius) -- inherited from mbirjax.

        Returns:
            (psf_radius, bp_psf_radius) ints.
        """
        (delta_det_row, delta_det_channel, source_detector_dist, recon_shape,
         delta_voxel, voxel_row_aspect, voxel_slice_aspect) = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_detector_dist',
             'recon_shape', 'delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        delta_det = min(delta_det_row, delta_det_channel)

        if np.isinf(source_detector_dist):
            max_magnification = min_magnification = 1
        else:
            source_to_iso_dist = source_detector_dist / magnification
            half_width_x = 0.5 * recon_shape[1] * delta_voxel
            half_width_y = 0.5 * recon_shape[0] * delta_voxel_row
            half_xy_extent = max(half_width_x, half_width_y)
            max_magnification = source_detector_dist / (source_to_iso_dist - half_xy_extent)
            min_magnification = source_detector_dist / (source_to_iso_dist + half_xy_extent)

        max_voxel_pitch = max(delta_voxel, delta_voxel_row, delta_voxel_slice)
        psf_radius = int(np.ceil(np.ceil(max_voxel_pitch * max_magnification / delta_det) / 2))
        min_voxel_pitch = min(delta_voxel, delta_voxel_row, delta_voxel_slice)
        max_voxels_per_detector = delta_det / (min_magnification * min_voxel_pitch)
        bp_psf_radius = int(np.ceil(np.ceil(max_voxels_per_detector) / 2))
        return psf_radius, bp_psf_radius

    def get_psf_radius(self):
        """Integer radius of the channel psf (see :meth:`get_psf_radii`)."""
        return self.get_psf_radii()[0]

    @staticmethod
    def detector_mn_to_uv(m, n, delta_det_channel, delta_det_row, det_channel_offset,
                          det_row_offset, num_det_rows, num_det_channels):
        """Fractional detector indices (m, n) -> physical coordinates (u, v)."""
        det_center_row = (num_det_rows - 1) / 2.0
        det_center_channel = (num_det_channels - 1) / 2.0
        v = (m - det_center_row) * delta_det_row - det_row_offset
        u = (n - det_center_channel) * delta_det_channel - det_channel_offset
        return u, v

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Compute the automatic recon shape for cone beam reconstruction.

        The xy width is the detector field of view at iso; the axial height is
        the detector height at iso swept over any helical travel, plus per-end
        padding scaled by ``axial_pad_fraction`` (a fraction of 1 pads each end
        to the deepest z reached by any measured ray).  Verbatim mbirjax math.
        """
        delta_det_row, delta_det_channel = self.get_params(
            ['delta_det_row', 'delta_det_channel'])
        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()

        delta_voxel = delta_det_channel / magnification
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        sinogram_shape = self.get_params('sinogram_shape')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        num_recon_rows = int(np.round(num_det_channels
                                      * ((delta_det_channel / delta_voxel_row) / magnification)))
        num_recon_cols = int(np.round(num_det_channels
                                      * ((delta_det_channel / delta_voxel) / magnification)))

        z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        z_min, z_max = float(np.min(z_shifts)), float(np.max(z_shifts))
        z_travel = z_max - z_min
        H_iso = num_det_rows * (delta_det_row / magnification)
        num_recon_slices = max(1, int(np.ceil((H_iso + z_travel) / delta_voxel_slice)))
        recon_slice_offset = 0.5 * (z_min + z_max)

        # Per-end axial padding: an edge ray at height v diverges across the
        # support to |z| = |v| * (SID + R) / SDD; extend each end by its excess
        # over H_iso / 2, scaled by axial_pad_fraction.
        source_detector_dist, det_row_offset, det_channel_offset, use_ror_mask = \
            self.get_params(['source_detector_dist', 'det_row_offset',
                             'det_channel_offset', 'use_ror_mask'])
        support_radius = get_support_radius((num_recon_rows, num_recon_cols),
                                            delta_voxel_row, delta_voxel,
                                            use_ror_mask=use_ror_mask)
        _, v_row_low = self.detector_mn_to_uv(-0.5, 0.0, delta_det_channel,
                                              delta_det_row, det_channel_offset,
                                              det_row_offset, num_det_rows,
                                              num_det_channels)
        _, v_row_high = self.detector_mn_to_uv(num_det_rows - 0.5, 0.0,
                                               delta_det_channel, delta_det_row,
                                               det_channel_offset, det_row_offset,
                                               num_det_rows, num_det_channels)
        v_bot = max(float(v_row_low), float(v_row_high))
        v_top = min(float(v_row_low), float(v_row_high))
        z_per_v_far_side = 1.0 / float(magnification)
        if not np.isinf(source_detector_dist):
            z_per_v_far_side += support_radius / float(source_detector_dist)
        excess_bot = max(0.0, v_bot * z_per_v_far_side - float(H_iso) / 2)
        excess_top = max(0.0, -v_top * z_per_v_far_side - float(H_iso) / 2)

        axial_pad_fraction = self.get_params('axial_pad_fraction')
        if isinstance(axial_pad_fraction, (tuple, list)):
            if len(axial_pad_fraction) != 2:
                raise ValueError('axial_pad_fraction must be a float or a '
                                 '(top_fraction, bottom_fraction) pair; got '
                                 f'{axial_pad_fraction!r}.')
            pad_frac_top, pad_frac_bot = (float(f) for f in axial_pad_fraction)
        else:
            pad_frac_top = pad_frac_bot = float(axial_pad_fraction)
        if pad_frac_top < 0 or pad_frac_bot < 0:
            raise ValueError(f'axial_pad_fraction must be >= 0; got {axial_pad_fraction!r}.')
        num_slices_top = int(np.ceil(pad_frac_top * excess_top / delta_voxel_slice))
        num_slices_bot = int(np.ceil(pad_frac_bot * excess_bot / delta_voxel_slice))
        num_recon_slices += num_slices_top + num_slices_bot
        recon_slice_offset += 0.5 * (num_slices_bot - num_slices_top) * float(delta_voxel_slice)

        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning,
                        recon_shape=recon_shape, delta_voxel=delta_voxel,
                        recon_slice_offset=recon_slice_offset)

    # ── direct recon (FDK) ────────────────────────────────────────────────────
    def fdk_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """FDK filtering: the shared row filter with the FDK cosine pre-weight
        per detector element and the voxel-size scale alpha."""
        sinogram = self._shard_sinogram(sinogram)
        # Dims from the params, not the array: the sinogram may be a Shards
        # container under a multi-device configuration.
        _, num_rows, num_channels = (int(x) for x in
                                     self.get_params('sinogram_shape'))
        source_detector_dist = self.get_params('source_detector_dist')
        (delta_voxel, delta_det_row, delta_det_channel, voxel_row_aspect,
         voxel_slice_aspect) = self.get_params(
            ['delta_voxel', 'delta_det_row', 'delta_det_channel',
             'voxel_row_aspect', 'voxel_slice_aspect'])
        det_row_offset, det_channel_offset = self.get_params(
            ['det_row_offset', 'det_channel_offset'])
        voxel_volume = delta_voxel * (voxel_row_aspect * delta_voxel) \
            * (voxel_slice_aspect * delta_voxel)
        M_0 = self.get_magnification()

        # FDK cosine pre-weight (rows, channels), view-independent.
        m_grid, n_grid = np.meshgrid(np.arange(num_rows), np.arange(num_channels),
                                     indexing='ij')
        u_grid, v_grid = self.detector_mn_to_uv(m_grid, n_grid, delta_det_channel,
                                                delta_det_row, det_channel_offset,
                                                det_row_offset, num_rows, num_channels)
        weight_map = source_detector_dist / np.sqrt(
            source_detector_dist ** 2 + u_grid ** 2 + v_grid ** 2)
        weight_t = torch.as_tensor(weight_map.astype(np.float32),
                                   device=self.torch_device)

        alpha = delta_det_row / (voxel_volume * M_0)
        return self._apply_direct_recon_filter(sinogram, filter_name,
                                               filter_scale=alpha,
                                               output_sharded=output_sharded,
                                               row_weight=weight_t)

    def helical_fdk_z_weight(self, recon, sinogram):
        """Scale each helical FDK slice by the inverse of the fraction of the
        scan in which the slice is in view of the detector (verbatim mbirjax)."""
        num_views, num_rows, num_channels = self.get_params('sinogram_shape')
        helical_z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        (delta_voxel, voxel_slice_aspect, recon_shape, recon_slice_offset,
         delta_det_row) = self.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_shape',
             'recon_slice_offset', 'delta_det_row'])
        M_0 = self.get_magnification()
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        from . import _sharding
        if isinstance(recon, _sharding.Shards):
            # Per-shard weighting in GLOBAL slice coordinates: each shard
            # scales its own slices with its slice-range of the weight.
            w_full = self._helical_z_weight_row(sinogram)
            rp = recon.placement
            tensors = []
            for i, (dev, (s0, s1)) in enumerate(
                    rp.shard_ranges(int(recon_shape[2]))):
                w = torch.as_tensor(w_full[s0:s1].astype(np.float32), device=dev)
                tensors.append(recon.tensors[i] * w[None, None, :])
            return _sharding.Shards(tensors, rp)
        num_real_slices = recon_shape[2]
        k = np.arange(recon.shape[2])
        z_k = delta_voxel_slice * (k - (num_real_slices - 1) / 2.0) + recon_slice_offset
        det_half_height_iso = 0.5 * num_rows * delta_det_row / M_0
        visible = np.abs(z_k[:, None] - helical_z_shifts[None, :]) <= det_half_height_iso
        coverage = np.sum(visible, axis=1)
        z_weight = np.where(coverage > 0, num_views / np.maximum(coverage, 1), 0.0)
        # Padded device-form slices (k >= num_real_slices) are identically zero
        # by the forced-zero invariant and must remain so.  A no-op until a
        # sharding port pads the slice axis (recon.shape[2] == recon_shape[2]).
        z_weight = np.where(k < num_real_slices, z_weight, 0.0)
        w = torch.as_tensor(z_weight.astype(np.float32), device=recon.device)
        return recon * w[None, None, :]

    def _helical_z_weight_row(self, sinogram):
        """The full (num_real_slices,) helical z-weight row in global slice
        coordinates (the shared math of helical_fdk_z_weight, from params
        only, host numpy)."""
        num_views, num_rows, _ = self.get_params('sinogram_shape')
        helical_z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        (delta_voxel, voxel_slice_aspect, recon_shape, recon_slice_offset,
         delta_det_row) = self.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_shape',
             'recon_slice_offset', 'delta_det_row'])
        M_0 = self.get_magnification()
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        num_real_slices = recon_shape[2]
        k = np.arange(num_real_slices)
        z_k = delta_voxel_slice * (k - (num_real_slices - 1) / 2.0) \
            + recon_slice_offset
        det_half_height_iso = 0.5 * num_rows * delta_det_row / M_0
        visible = np.abs(z_k[:, None] - helical_z_shifts[None, :]) \
            <= det_half_height_iso
        coverage = np.sum(visible, axis=1)
        return np.where(coverage > 0, num_views / np.maximum(coverage, 1), 0.0)

    def fdk_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FDK reconstruction: standard filtering, then the exact adjoint
        of the forward projector as the backprojection.

        Note:
            FDK assumes equally spaced views over the full angular range and
            applies no short-scan redundancy weighting; for helical scans it is
            approximate regardless.  Best used as an initializer for ``recon()``.
        """
        # Place once at entry so the filter receives device-form data (a no-op
        # when already placed; a single device is the trivial 1-shard case).
        # The pipeline then stays on-device throughout -- fdk_filter then
        # back_project, both output_sharded=True (zero host transfer) --
        # exactly like ParallelBeamModel.fbp_recon.
        sinogram = self._shard_sinogram(sinogram)
        filtered_sinogram = self.fdk_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self.back_project(filtered_sinogram, output_sharded=True)
        helical_z_shifts = np.asarray(self.get_params('view_params_array'))[:, 1]
        if float(np.max(helical_z_shifts) - np.min(helical_z_shifts)) > 0:
            recon = self.helical_fdk_z_weight(recon, sinogram)
        return recon if output_sharded else self._gather_recon(recon)

    def direct_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fdk_recon(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fdk_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)
