"""MultiAxisParallelModel: parallel beam with a per-view elevation (tilt)
angle.  Each view has two angles --
azimuth (the standard tomography rotation about z) and elevation (the tilt of
the ray out of the xy plane).  Parallel beam laminography is a special case;
at zero elevation the geometry is mathematically equivalent to parallel beam.

Structure: two separable fans, as in cone and translation, and the bodies
below mirror those in cone_beam.py / translation_model.py.  The one place the
geometry differs is the detector row coordinate, v = z*cos(el) + y*sin(el)
(parallel beam has v = z), so the vertical fan's affine slice-to-row map has
slope delta_voxel_slice*cos(el)/delta_det_row and a per-pixel anchor that
carries the in-plane depth y.  The vertical fan's weights differ from cone's
and translation's: a pure interpolation weight with a per-view mass-conserving
amplitude (total vertical weight independent of elevation), and its footprint
is the largest of the voxel's three projected edges, so it never collapses to
zero -- which is why the FORWARD vertical fan is a scatter over slices rather
than the detector-side gather cone uses (a straight-down view has slope 0, and
a row-to-slice inversion would divide by it).
"""

import warnings

import numpy as np
import torch

from .horizontal_fan import fan_back_batch, fan_forward_batch
from .tomography_model import TomographyModel

_F32 = torch.float32


# ── geometry chains (pure, compiled) ─────────────────────────────────────────
def _multiaxis_horizontal_data(pixel_indices, azimuth, num_rows, num_cols,
                               num_channels, delta_voxel, delta_voxel_row,
                               delta_det_channel, det_channel_offset):
    """The horizontal fan's inputs for a view batch.  n_p and centers (int32)
    are (Vb, P); W_p_c and weight_scale are per-VIEW (Vb, 1) -- the footprint
    depends only on the azimuth (parallel beam, no magnification).  Also
    returns the rotated in-plane depth y (Vb, P) for the vertical fan.
    """
    row_index = (pixel_indices // num_cols).to(_F32)
    col_index = (pixel_indices % num_cols).to(_F32)
    y_tilde = delta_voxel_row * (row_index - (num_rows - 1) / 2.0)
    x_tilde = delta_voxel * (col_index - (num_cols - 1) / 2.0)
    cosine = torch.cos(azimuth)[:, None]
    sine = torch.sin(azimuth)[:, None]
    x = cosine * x_tilde[None, :] - sine * y_tilde[None, :]
    y = sine * x_tilde[None, :] + cosine * y_tilde[None, :]
    n_p = (x + det_channel_offset) / delta_det_channel + (num_channels - 1) / 2.0
    footprint_xy = torch.maximum(cosine.abs() * delta_voxel,
                                 sine.abs() * delta_voxel_row)       # (Vb, 1)
    W_p_c = footprint_xy / delta_det_channel
    weight_scale = (delta_voxel * delta_voxel_row) / footprint_xy
    centers = torch.round(n_p).to(torch.int32)
    return n_p, centers, W_p_c, weight_scale, y


def _multiaxis_vertical_terms(y, azimuth, elevation, num_slices,
                              delta_voxel, delta_voxel_row, delta_voxel_slice,
                              delta_det_row, det_row_offset, recon_slice_offset,
                              num_rows_r):
    """The vertical fan's terms: the affine slice-to-row map m(k) = m0 + slope*k
    (k the GLOBAL slice index; v = z*cos(el) + y*sin(el)), plus the per-view
    footprint, clip bound, and mass-conserving amplitude.

    The footprint is the LARGEST of the voxel's three projected edges on the
    detector v-axis (max, not sum -- the trapezoid's effective width is about
    the largest edge), so it never collapses to zero and the amplitude never
    divides by zero.  The amplitude makes the total vertical weight
    delta_voxel_slice/delta_det_row, independent of elevation.

    Returns:
        (m0 (Vb, P), slope (Vb, 1), W_p_r (Vb, 1), L_max (Vb, 1),
        scaling (Vb, 1)).
    """
    cos_el = torch.cos(elevation)[:, None]                           # (Vb, 1)
    sin_el = torch.sin(elevation)[:, None]
    z_0 = -delta_voxel_slice * (num_slices - 1) / 2.0 + recon_slice_offset
    v_0 = z_0 * cos_el + y * sin_el                                  # (Vb, P)
    m0 = (v_0 + det_row_offset) / delta_det_row + (num_rows_r - 1) / 2.0
    slope = (delta_voxel_slice * cos_el) / delta_det_row             # (Vb, 1)

    z_edge = delta_voxel_slice * cos_el.abs()
    x_edge = sin_el.abs() * torch.sin(azimuth)[:, None].abs() * delta_voxel
    y_edge = sin_el.abs() * torch.cos(azimuth)[:, None].abs() * delta_voxel_row
    W_p_r = torch.maximum(z_edge, torch.maximum(x_edge, y_edge)) / delta_det_row
    L_max = torch.clamp(W_p_r, max=1.0)
    scaling = (delta_voxel_slice / delta_det_row) / W_p_r
    return m0, slope, W_p_r, L_max, scaling


def _multiaxis_forward_view_batch(values, pixel_indices, view_params_batch,
                                  num_rows_r, num_channels, num_recon_rows,
                                  num_recon_cols, num_slices, delta_voxel,
                                  delta_voxel_row, delta_voxel_slice,
                                  delta_det_channel, delta_det_row,
                                  det_channel_offset, det_row_offset,
                                  recon_slice_offset, psf_radius,
                                  slice_start=0, plan=None):
    """Multiaxis forward for one view batch: the slice-scatter vertical fan,
    then the per-pixel horizontal fan scatter.  Returns (Vb, R, C).

    ``slice_start`` supports a slice-BANDED call exactly as in cone:
    ``values`` may be a band (P, L) with global indices [slice_start,
    slice_start + L); the slice-to-row map is anchored on the full num_slices
    center, so summing per-band outputs over a tiling of the slice axis
    reproduces the unbanded projection.

    ``plan`` is the memoization slot for a future sorted/CSR stream variant;
    unused today."""
    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    vb, num_pixels = n_p.shape
    dev = values.device

    # ── vertical fan (scatter input slices onto detector rows) ───────────────
    m0, slope, W_p_r, L_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)

    band_len = values.shape[1]
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    m_p = m0.unsqueeze(-1) + slope.unsqueeze(-1) * k[None, None, :]  # (Vb, P, L)
    m_center = torch.round(m_p).to(torch.int64)

    # Mass-conserving amplitude folded into the values; slices past the real
    # count are masked (the global validity test).
    valid_k = (k < num_slices).to(_F32)                              # (L,)
    scaled_values = values[None, :, :] * scaling.unsqueeze(-1) \
        * valid_k[None, None, :]                                     # (Vb, P, L)

    W_b = W_p_r.unsqueeze(-1)
    L_max_b = L_max.unsqueeze(-1)
    det_col = torch.zeros((vb, num_pixels, num_rows_r), dtype=_F32, device=dev)
    for m_off in range(-psf_radius, psf_radius + 1):
        m = m_center + m_off
        A = torch.clamp((W_b + 1.0) / 2.0 - (m_p - m.to(_F32)).abs(), min=0.0)
        A = torch.minimum(A, L_max_b)
        A = A * ((m >= 0) & (m < num_rows_r)).to(_F32)
        det_col.scatter_add_(2, m.clamp(0, num_rows_r - 1), scaled_values * A)

    # ── horizontal fan scatter (the shared kernel; per-view values) ──────────
    acc = fan_forward_batch((n_p, centers, W_p_c, weight_scale), det_col,
                            num_channels, psf_radius)
    return acc.permute(0, 2, 1)


def _multiaxis_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                               num_rows_r, num_channels, num_recon_rows,
                               num_recon_cols, num_slices, delta_voxel,
                               delta_voxel_row, delta_voxel_slice,
                               delta_det_channel, delta_det_row,
                               det_channel_offset, det_row_offset,
                               recon_slice_offset, psf_radius, coeff_power=1,
                               slice_start=0, band_slices=None, plan=None):
    """Multiaxis back projection for one view batch, summed over the batch's
    views: horizontal fan gather -> per-pixel detector columns -> vertical fan
    gather onto the slices.  Returns (P, S), or (P, band_slices) for a slice
    band, exactly as in cone.

    The vertical weight is a pure interpolation weight raised to coeff_power,
    with the per-view amplitude applied at the same power (the adjoint of the
    forward folding the amplitude into the values).

    ``plan`` is the memoization slot for a future sorted/CSR stream variant;
    unused today."""
    azimuth = view_params_batch[:, 0]
    elevation = view_params_batch[:, 1]
    n_p, centers, W_p_c, weight_scale, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, num_recon_rows, num_recon_cols, num_channels,
        delta_voxel, delta_voxel_row, delta_det_channel, det_channel_offset)
    vb, num_pixels = n_p.shape
    dev = sino_batch.device

    # ── horizontal fan gather (the shared kernel, view axis kept) ────────────
    sino_T = sino_batch.permute(0, 2, 1).contiguous()                # (Vb, C, R)
    det_col = fan_back_batch(sino_T, (n_p, centers, W_p_c, weight_scale),
                             num_channels, psf_radius,
                             coeff_power=coeff_power, reduce_views=False)

    # ── vertical fan gather ──────────────────────────────────────────────────
    m0, slope, W_p_r, L_max, scaling = _multiaxis_vertical_terms(
        y, azimuth, elevation, num_slices, delta_voxel, delta_voxel_row,
        delta_voxel_slice, delta_det_row, det_row_offset, recon_slice_offset,
        num_rows_r)

    band_len = num_slices if band_slices is None else band_slices
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    m_p = m0.unsqueeze(-1) + slope.unsqueeze(-1) * k[None, None, :]  # (Vb, P, S)
    m_center = torch.round(m_p).to(torch.int64)

    W_b = W_p_r.unsqueeze(-1)
    L_max_b = L_max.unsqueeze(-1)
    scale_pow = (scaling ** coeff_power).unsqueeze(-1)               # (Vb, 1, 1)
    out = torch.zeros((num_pixels, band_len), dtype=_F32, device=dev)
    for m_off in range(-psf_radius, psf_radius + 1):
        mm = m_center + m_off
        A = torch.clamp((W_b + 1.0) / 2.0 - (m_p - mm.to(_F32)).abs(), min=0.0)
        A = torch.minimum(A, L_max_b)
        A = A * ((mm >= 0) & (mm < num_rows_r)).to(_F32)
        if coeff_power != 1:
            A = A ** coeff_power
        A = A * scale_pow
        g = torch.gather(det_col, 2, mm.clamp(0, num_rows_r - 1))
        out = out + torch.einsum("vps,vps->ps", A, g)
    # Slices past the real count are inert (a no-op for an interior band).
    out = out * (k < num_slices).to(_F32)[None, :]
    return out


class MultiAxisParallelModel(TomographyModel):
    """
    Parallel beam geometry allowing for a per-view elevation (tilt) angle,
    extending :class:`TomographyModel`.

    Each view has two angles:
      - Azimuth: rotation around the object's z-axis (the standard tomography
        rotation, as in ParallelBeamModel).
      - Elevation: tilt of the ray vector out of the xy plane.

    When elevation = 0 this model is mathematically equivalent to
    ParallelBeamModel.  Parallel beam laminography is a special case.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels).
        angles (ndarray): (num_views, 2) array; angles[:, 0] is the azimuth
            and angles[:, 1] the elevation, in radians.
        view_batch_size, compile_mode: as in ParallelBeamModel.
    """

    def __init__(self, sinogram_shape, angles, view_batch_size=None,
                 compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32)
        if angles.ndim != 2 or angles.shape[1] != 2:
            raise ValueError(f"angles must have shape (num_views, 2). "
                             f"Got {angles.shape}.")
        if angles.shape[0] != sinogram_shape[0]:
            raise ValueError(f"Number of angle pairs ({angles.shape[0]}) must "
                             f"match number of views ({sinogram_shape[0]}).")
        if np.any(np.abs(angles[:, 1]) > np.pi / 4):
            warnings.warn("One or more elevation angles exceed 45 degrees. "
                          "This may degrade approximation quality.")
        # geometry_type is the class-identity string, so save/load resolves
        # the class by name.
        super().__init__(sinogram_shape,
                         view_batch_size=view_batch_size, compile_mode=compile_mode,
                         geometry_type=str(type(self)),
                         view_params_name='angles', angles=angles,
                         recon_slice_offset=0.0)

    # Multiaxis has its own floor family rather than borrowing parallel's,
    # because its crossover was measured separately.  The thresholds
    # themselves, the runs behind them, and their dates are the multiaxis
    # rows of _widening_floors.FLOORS; they are not restated here, because a
    # refresh moves them and a copy would go stale.  Historical note: the
    # 2026-08-17 reading that first justified a separate family (394 s at two
    # devices against 951 s at four, 1024-class) was torch's per-function
    # recompile budget filling, which projectors._raise_recompile_budget has
    # since fixed.
    _floor_family = 'multiaxis'

    def _view_batch_bodies(self):
        # No hand-written kernels for multiaxis yet; the compiled torch bodies
        # are the only bodies.
        return _multiaxis_forward_view_batch, _multiaxis_back_view_batch

    def _view_batch_args(self):
        gp_names = ['delta_det_row', 'delta_det_channel', 'det_row_offset',
                    'det_channel_offset', 'delta_voxel', 'voxel_row_aspect',
                    'voxel_slice_aspect', 'recon_slice_offset']
        (ddr, ddc, dro, dco, dv, vra, vsa, rso) = self.get_params(gp_names)
        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')
        return dict(num_rows_r=sinogram_shape[1], num_channels=sinogram_shape[2],
                    num_recon_rows=recon_shape[0], num_recon_cols=recon_shape[1],
                    num_slices=recon_shape[2], delta_voxel=dv,
                    delta_voxel_row=vra * dv, delta_voxel_slice=vsa * dv,
                    delta_det_channel=ddc, delta_det_row=ddr,
                    det_channel_offset=dco, det_row_offset=dro,
                    recon_slice_offset=rso,
                    psf_radius=self.get_psf_radius())

    def _transient_cols(self, band_cols):
        # The bodies hold (Vb, P, S) and (Vb, P, R) transients whatever the
        # requested band, so the budget width is params-derived (as in cone).
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return max(int(recon_shape[2]), int(sinogram_shape[1]))

    def get_magnification(self):
        """For parallel beam geometries, magnification is always 1.0."""
        return 1.0

    def verify_valid_params(self):
        """Verify parameters match the expected geometry constraints."""
        super().verify_valid_params()
        sinogram_shape = self.get_params('sinogram_shape')
        angles = np.asarray(self.get_params('angles'))
        if angles.shape[0] != sinogram_shape[0]:
            raise ValueError(f"View mismatch: {angles.shape[0]} angles for "
                             f"{sinogram_shape[0]} views.")
        if angles.shape[1] != 2:
            raise ValueError("Each view requires exactly 2 angles: "
                             "[azimuth, elevation].")

    def get_psf_radius(self):
        """Integer radius of the psf kernel: the max of the horizontal
        (in-plane footprint on the channels) and vertical (tilt footprint on
        the rows) radii, one radius for every view."""
        (delta_det_channel, delta_det_row, delta_voxel, voxel_row_aspect,
         voxel_slice_aspect) = self.get_params(
            ['delta_det_channel', 'delta_det_row', 'delta_voxel',
             'voxel_row_aspect', 'voxel_slice_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        max_in_plane_pitch = max(delta_voxel, delta_voxel_row)
        psf_radius_u = int(np.ceil(np.ceil(max_in_plane_pitch / delta_det_channel) / 2))

        # Vertical: one radius serves every view, so take each footprint
        # edge's max over the actual elevations (the z edge peaks at the
        # smallest tilt, the in-plane edge at the largest).
        angles = self.get_params('angles')
        if angles is not None:
            elevations = np.asarray(angles)[:, 1]
            max_abs_cos_el = float(np.max(np.abs(np.cos(elevations))))
            max_abs_sin_el = float(np.max(np.abs(np.sin(elevations))))
        else:
            max_abs_cos_el, max_abs_sin_el = 1.0, 0.0
        z_edge = max_abs_cos_el * delta_voxel_slice
        in_plane_edge = max_abs_sin_el * max(delta_voxel, delta_voxel_row)
        vertical_footprint = max(z_edge, in_plane_edge)
        psf_radius_v = int(np.ceil(np.ceil(vertical_footprint / delta_det_row) / 2))

        return max(psf_radius_u, psf_radius_v)

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Set the reconstruction shape from the largest bounding box that
        projects onto the detector at the given angles."""
        sinogram_shape = self.get_params('sinogram_shape')
        num_views, num_det_rows, num_det_channels = sinogram_shape
        delta_det_channel, delta_det_row = self.get_params(
            ['delta_det_channel', 'delta_det_row'])

        max_u = (num_det_channels * delta_det_channel) / 2.0
        max_v = (num_det_rows * delta_det_row) / 2.0

        angles = np.asarray(self.get_params('angles'))
        elevations = angles[:, 1]

        # XY radius from the channel coverage; z height from the row coverage
        # (v = z*cos(el) - t*sin(el); the cos is clamped so a top-down view
        # does not imply infinite z).
        max_R_xy = max_u
        min_cos_el = np.min(np.abs(np.cos(elevations)))
        min_cos_el = max(min_cos_el, 0.1)
        max_R_z = max_v / min_cos_el

        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])
        delta_voxel = delta_det_channel
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        num_recon_cols = int(np.floor(2 * max_R_xy / delta_voxel))
        num_recon_rows = int(np.floor(2 * max_R_xy / delta_voxel_row))
        num_recon_slices = int(np.floor(2 * max_R_z / delta_voxel_slice))

        self.set_params(recon_shape=(num_recon_rows, num_recon_cols, num_recon_slices),
                        delta_voxel=delta_voxel,
                        no_compile=no_compile, no_warning=no_warning)

    # ── direct recon (stacked 2-D FBP) ────────────────────────────────────────
    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """FBP filtering with the standard 1-D channel ramp (the shared row
        filter, as in ParallelBeamModel).

        The multiaxis geometry is a set of simultaneous fixed measurements
        with no acquisition trajectory, so the ``angles`` list order is
        arbitrary and the filter must not depend on it: a uniform per-view
        angular weight pi/num_views times the channel ramp.  This reduces
        exactly to ParallelBeamModel.fbp_filter at zero elevation; elevation
        is approximated in the filter and corrected by the iterative
        ``recon()``.
        """
        delta_voxel, voxel_row_aspect = self.get_params(
            ['delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(
            sinogram, filter_name, filter_scale=scaling_factor,
            output_sharded=output_sharded, row_weight=None)

    def recon_fbp(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FBP reconstruction: filter the sinogram, then apply the exact
        adjoint of the forward projector as the backprojection.

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            filter_name (string, optional): The name of the filter to use.
                Defaults to 'ramp'.
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the device form: a torch
                tensor on a single device, or a Shards container (one
                tensor per device) on a multi-device model.

        Returns:
            recon (numpy or tensor): The reconstructed volume.

        Note:
            The pi/num_views weight assumes equally spaced azimuths, and the
            geometry is treated as stacked 2-D FBP: the filter ignores
            elevation.  Multiaxis parallel beam is typically a limited-angle
            geometry, so this direct reconstruction is only approximate; it is
            intended as an initializer for the iterative ``recon()``.
        """
        # Settle the device layout before the first large allocation, as
        # recon() does: a no-op when the user already chose devices;
        # otherwise the automatic selection runs here, so a bare FBP call
        # spreads across the GPUs instead of landing whole on one.  The
        # workload tells the memory check to price this reconstruction rather
        # than the full recon the device count is chosen for.
        self._apply_device_policy(workload='direct')
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self.back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

    def recon_direct(self, sinogram, filter_name="ramp", output_sharded=False):
        """Direct reconstruction by stacked 2-D FBP; equivalent to
        :meth:`recon_fbp`.  See :meth:`TomographyModel.recon_direct` for the
        argument and return conventions."""
        return self.recon_fbp(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """Filtering as needed for a direct recon; equivalent to
        :meth:`fbp_filter`."""
        return self.fbp_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)


# Backward-compatible public API name used throughout docs/examples.
MultiAxisParallelBeamModel = MultiAxisParallelModel
