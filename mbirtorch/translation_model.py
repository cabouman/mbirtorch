"""TranslationModel: each view is a
cone beam projection of a translated object (translation computed tomography,
TCT).  Useful for 3D imaging of thin objects.

Structure: like cone, translation projection is two separable fans, and the
bodies below mirror cone_beam.py's.  The view parameter is a per-view
(t_x, t_y, t_z) object translation instead of a rotation angle: the in-plane
pixel coordinates are shifted, never rotated, and t_z plays the role cone's
helical z shift plays in the vertical fan.  The HORIZONTAL fan maps a voxel
to detector channels with per-pixel magnification; the VERTICAL fan maps
each slice of a voxel cylinder to a range of detector rows through the
affine pair (m0, W_p_r) of :func:`_translation_vertical_affine`.  Both
directions use the single psf_radius of :meth:`TranslationModel.get_psf_radius`
(translation has no separate forward vertical radius).

Known scale limit: at production TCT detector shapes
(~1900x3000 panels) the back projection holds (view_batch, P, rows) and
(view_batch, P, slices) transients, so large pixel batches are memory-bound
and the view batch shrinks accordingly.  What would relieve it is a change
to the projector drivers, not to this file: they currently tile over views
only, and tiling over the pixel axis as well -- the two-axis tiling
described in projectors.py -- would let the pixel batch shrink instead of the
view batch.  Nothing here
works around its absence.
"""

import warnings

import numpy as np
import torch

from .cone_beam import (ConeBeamModel, _cone_channel_coordinate,
                        _cone_row_coordinate)
from .geometry_rules import channel_index, pixel_xy, row_index
from .horizontal_fan import fan_back_batch, fan_forward_batch
from .tomography_model import TomographyModel

_F32 = torch.float32


def _translation_pixel_xy_mag(pixel_indices, t_x, t_y, num_rows, num_cols,
                              delta_voxel, delta_voxel_row, magnification,
                              source_detector_dist):
    """Translated in-plane coordinates and the per-pixel magnification.

    The object shifts by (-t_x, -t_y) per view and is never rotated.
    Returns x (Vb, P), y (Vb, P), and pixel_mag (Vb, P).
    """
    x_tilde, y_tilde = pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel,
                                delta_voxel_row)
    return _translation_xy_mag(x_tilde, y_tilde, t_x, t_y, magnification,
                               source_detector_dist)


def _translation_xy_mag(x_tilde, y_tilde, t_x, t_y, magnification,
                        source_detector_dist):
    """The same map as :func:`_translation_pixel_xy_mag`, taking positions
    rather than pixel indices.  Returns x, y, and pixel_mag, each (Vb, P)."""
    y = y_tilde[None, :] - t_y[:, None]
    x = x_tilde[None, :] - t_x[:, None]
    pixel_mag = 1.0 / (1.0 / magnification - y / source_detector_dist)
    return x, y, pixel_mag


def _translation_horizontal_data(pixel_indices, view_params_batch, num_rows,
                                 num_cols, num_channels, delta_voxel,
                                 delta_voxel_row, delta_det_channel,
                                 det_channel_offset, magnification,
                                 source_detector_dist):
    """The horizontal fan inputs for one view batch.

    Returns n_p, centers (int32), W_p_c, and weight_scale, each (Vb, P), plus
    pixel_mag for the vertical fan.  The weight scale is the projection length
    through a voxel, delta_voxel_row / cos(theta_p).
    """
    t_x = view_params_batch[:, 0]
    t_y = view_params_batch[:, 1]
    x, y, pixel_mag = _translation_pixel_xy_mag(
        pixel_indices, t_x, t_y, num_rows, num_cols, delta_voxel,
        delta_voxel_row, magnification, source_detector_dist)
    u = _cone_channel_coordinate(x, y, pixel_mag, magnification,
                                 source_detector_dist, False)
    theta = torch.atan2(u, torch.as_tensor(source_detector_dist, dtype=_F32,
                                           device=u.device))
    n_p = channel_index(u, delta_det_channel, det_channel_offset, num_channels)
    W_p_c = pixel_mag * (delta_voxel / delta_det_channel)
    weight_scale = delta_voxel_row / torch.cos(theta)
    centers = torch.round(n_p).to(torch.int32)
    return n_p, centers, W_p_c, weight_scale, pixel_mag


def _translation_vertical_affine(pixel_mag, t_z, num_slices, delta_voxel_slice,
                                 delta_det_row, det_row_offset, num_rows_r):
    """The vertical fan's affine map from global slice index to detector row.

    The map is m(v, p, l) = m0 + W_p_r * l.  Returns m0 (Vb, P), W_p_r (Vb, P),
    and z_offset (Vb,).
    """
    z_offset = -t_z                                              # (Vb,)
    W_p_r = pixel_mag * delta_voxel_slice / delta_det_row        # (Vb, P)
    z_at_slice_0 = z_offset[:, None] - delta_voxel_slice * (num_slices - 1) / 2.0
    m0 = row_index(_cone_row_coordinate(pixel_mag, z_at_slice_0),
                   delta_det_row, det_row_offset, num_rows_r)    # (Vb, P)
    return m0, W_p_r, z_offset


def _translation_forward_view_batch(values, pixel_indices, view_params_batch,
                                    num_rows_r, num_channels, num_recon_rows,
                                    num_recon_cols, num_slices, delta_voxel,
                                    delta_voxel_row, delta_voxel_slice,
                                    delta_det_channel, delta_det_row,
                                    det_channel_offset, det_row_offset,
                                    magnification, source_detector_dist,
                                    psf_radius, slice_start=0, plan=None):
    """Translation forward projection for one view batch.  Returns (Vb, R, C).

    ``values`` may be a slice band (P, L) with global indices starting at
    ``slice_start``.  The z geometry stays anchored on the full num_slices
    center, and taps outside the band contribute zero.  ``plan`` is ignored.
    """
    n_p, centers, W_p_c, weight_scale, pixel_mag = _translation_horizontal_data(
        pixel_indices, view_params_batch, num_recon_rows, num_recon_cols,
        num_channels, delta_voxel, delta_voxel_row, delta_det_channel,
        det_channel_offset, magnification, source_detector_dist)
    vb, num_pixels = n_p.shape
    dev = values.device
    t_z = view_params_batch[:, 2]

    # Vertical fan, detector side.
    m0, W_p_r, z_offset = _translation_vertical_affine(
        pixel_mag, t_z, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, num_rows_r)

    # 1/cos(phi) is the projection length through a voxel at cone angle phi.
    band_len = values.shape[1]
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    z = (delta_voxel_slice * (k - (num_slices - 1) / 2.0))[None, None, :] \
        + z_offset[:, None, None]                                # (Vb, 1, L)
    v_slices = pixel_mag.unsqueeze(-1) * z                       # (Vb, P, L)
    cos_phi = torch.cos(torch.atan2(v_slices, torch.as_tensor(
        source_detector_dist, dtype=_F32, device=dev)))
    scaled_values = values[None, :, :] / cos_phi

    # This is the inverse of the affine map, from detector rows to fractional
    # slice indices.  The back projection body evaluates the direct form.
    m = torch.arange(num_rows_r, dtype=_F32, device=dev)         # (R,)
    k_m = (m[None, None, :] - m0.unsqueeze(-1)) / W_p_r.unsqueeze(-1)
    k_center = torch.round(k_m).to(torch.int64)                  # (Vb, P, R)

    slope = W_p_r.unsqueeze(-1)
    L_max_r = torch.clamp(W_p_r, max=1.0).unsqueeze(-1)
    m_p = slope * (k_center.to(_F32) - k_m)                      # Projection offset.

    det_col = torch.zeros((vb, num_pixels, num_rows_r), dtype=_F32, device=dev)
    for k_off in range(-psf_radius, psf_radius + 1):
        k_ind = k_center + k_off
        A = torch.clamp((slope + 1.0) / 2.0 - (m_p + slope * k_off).abs(), min=0.0)
        A = torch.minimum(A, L_max_r)
        A = A * ((k_ind >= slice_start)
                 & (k_ind < slice_start + band_len)).to(_F32)
        g = torch.gather(scaled_values, 2,
                         (k_ind - slice_start).clamp(0, band_len - 1))
        det_col = det_col + A * g

    acc = fan_forward_batch((n_p, centers, W_p_c, weight_scale), det_col,
                            num_channels, psf_radius)
    return acc.permute(0, 2, 1)


def _translation_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                                 num_rows_r, num_channels, num_recon_rows,
                                 num_recon_cols, num_slices, delta_voxel,
                                 delta_voxel_row, delta_voxel_slice,
                                 delta_det_channel, delta_det_row,
                                 det_channel_offset, det_row_offset,
                                 magnification, source_detector_dist,
                                 psf_radius, coeff_power=1, slice_start=0,
                                 band_slices=None, plan=None):
    """Translation back projection for one view batch, summed over its views.

    Returns (P, S), or (P, band_slices) for a slice band.  ``plan`` is ignored.
    """
    n_p, centers, W_p_c, weight_scale, pixel_mag = _translation_horizontal_data(
        pixel_indices, view_params_batch, num_recon_rows, num_recon_cols,
        num_channels, delta_voxel, delta_voxel_row, delta_det_channel,
        det_channel_offset, magnification, source_detector_dist)
    vb, num_pixels = n_p.shape
    dev = sino_batch.device
    t_z = view_params_batch[:, 2]

    # Horizontal fan gather, with the view axis kept.
    sino_T = sino_batch.permute(0, 2, 1).contiguous()            # (Vb, C, R)
    det_col = fan_back_batch(sino_T, (n_p, centers, W_p_c, weight_scale),
                             num_channels, psf_radius,
                             coeff_power=coeff_power, reduce_views=False)

    # Vertical fan gather.
    m0, W_p_r, z_offset = _translation_vertical_affine(
        pixel_mag, t_z, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, num_rows_r)

    band_len = num_slices if band_slices is None else band_slices
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    z = (delta_voxel_slice * (k - (num_slices - 1) / 2.0))[None, None, :] \
        + z_offset[:, None, None]
    v_slices = pixel_mag.unsqueeze(-1) * z                       # (Vb, P, S)
    sdd_t = torch.as_tensor(source_detector_dist, dtype=_F32, device=dev)
    cos_phi = torch.cos(torch.atan2(v_slices, sdd_t))
    # This is the direct form of the affine map, from slice to detector row.
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


class TranslationModel(TomographyModel):
    """
    A class for forward and backward projections in a translation (TCT)
    geometry, extending :class:`TomographyModel`: each view is a cone beam
    projection of a translated object.  Useful for 3D imaging of thin objects.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels),
            where num_views is the number of translation steps.
        translation_vectors (ndarray): (num_views, 3) array of object
            translations (x, y, z) in ALU.  Each view moves the object by minus
            its vector, in the object frame that
            :meth:`~mbirtorch.TomographyModel.project_points` describes: a
            positive t_x moves the object toward -x, so its image moves toward
            lower channel index; a positive t_y moves it toward -y, away from
            the source, so its image shrinks toward the point where the central
            ray meets the detector; a positive t_z moves it toward -z, so its
            image moves toward lower row index.
        source_detector_dist (float): Distance from source to detector in ALU.
        source_iso_dist (float): Distance from source to isocenter in ALU.
        view_batch_size, compile_mode: as in ParallelBeamModel.
    """

    # The thresholds for this family are the translation rows of
    # _widening_floors.FLOORS.
    _floor_family = 'translation'

    def __init__(self, sinogram_shape, translation_vectors, source_detector_dist,
                 source_iso_dist, view_batch_size=None, compile_mode='auto'):
        translation_vectors = np.asarray(translation_vectors, dtype=np.float32)
        if translation_vectors.ndim != 2 or translation_vectors.shape[1] != 3:
            raise ValueError('translation_vectors must have shape '
                             f'(num_views, 3); got {translation_vectors.shape}.')
        # Row neighbor regularization is weak because the objects are thin.  There is
        # no cylindrical mask because the object usually spans the field of view.
        super().__init__(sinogram_shape,
                         view_batch_size=view_batch_size, compile_mode=compile_mode,
                         geometry_type='translation',
                         view_params_name='translation_vectors',
                         translation_vectors=translation_vectors,
                         source_detector_dist=source_detector_dist,
                         source_iso_dist=source_iso_dist,
                         qggmrf_nbr_wts=[0.1, 1.0, 1.0], use_ror_mask=False)
        # A larger line search cap has shown instabilities in this geometry.
        self.set_params(no_warning=True, max_alpha=1.3)

    def _view_batch_bodies(self):
        # Translation has no hand written kernels, only the compiled torch bodies.
        return _translation_forward_view_batch, _translation_back_view_batch

    def _project_points_batch(self, points, view_params):
        # points is (N, 3) float64 on the CPU.  view_params is (V, 3) holding
        # (t_x, t_y, t_z).  Returns (row, channel), each (V, N).
        ddr, ddc, dro, dco, sdd = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'det_row_offset',
             'det_channel_offset', 'source_detector_dist'])
        _, num_rows_r, num_channels = self.get_params('sinogram_shape')
        magnification = self.get_magnification()
        x, y, pixel_mag = _translation_xy_mag(
            points[:, 0], points[:, 1], view_params[:, 0], view_params[:, 1],
            magnification, sdd)
        u = _cone_channel_coordinate(x, y, pixel_mag, magnification, sdd, False)
        # Each view moves the object by minus its translation vector.
        z = points[:, 2][None, :] - view_params[:, 2][:, None]
        v = _cone_row_coordinate(pixel_mag, z)
        return (row_index(v, ddr, dro, num_rows_r),
                channel_index(u, ddc, dco, num_channels))

    def _view_batch_args(self):
        gp_names = ['delta_det_row', 'delta_det_channel', 'det_row_offset',
                    'det_channel_offset', 'source_detector_dist', 'delta_voxel',
                    'voxel_row_aspect', 'voxel_slice_aspect']
        (ddr, ddc, dro, dco, sdd, dv, vra, vsa) = self.get_params(gp_names)
        sinogram_shape = self.get_params('sinogram_shape')
        recon_shape = self.get_params('recon_shape')
        return dict(num_rows_r=sinogram_shape[1], num_channels=sinogram_shape[2],
                    num_recon_rows=recon_shape[0], num_recon_cols=recon_shape[1],
                    num_slices=recon_shape[2], delta_voxel=dv,
                    delta_voxel_row=vra * dv, delta_voxel_slice=vsa * dv,
                    delta_det_channel=ddc, delta_det_row=ddr,
                    det_channel_offset=dco, det_row_offset=dro,
                    magnification=self.get_magnification(),
                    source_detector_dist=sdd,
                    psf_radius=self.get_psf_radius())

    def _transient_cols(self, band_cols):
        # The bodies hold (Vb, P, S) and (Vb, P, R) temporaries whatever band is
        # requested, so the budget width comes from the parameters.
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return max(int(recon_shape[2]), int(sinogram_shape[1]))

    def get_magnification(self):
        """magnification = source_detector_dist / source_iso_dist."""
        source_detector_dist, source_iso_dist = self.get_params(
            ['source_detector_dist', 'source_iso_dist'])
        if np.isinf(source_detector_dist):
            raise ValueError('Distance from source to detector is infinite, '
                             'which means all translated projections have the '
                             'same information.')
        return source_detector_dist / source_iso_dist

    def verify_valid_params(self):
        """Check that all parameters are compatible for a reconstruction."""
        super().verify_valid_params()
        sinogram_shape, translation_vectors = self.get_params(
            ['sinogram_shape', 'translation_vectors'])
        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])
        if translation_vectors is None:
            raise ValueError("translation_vectors was not set.")
        if tuple(np.asarray(translation_vectors).shape) != (sinogram_shape[0], 3):
            raise ValueError('Number of translation vectors must equal the '
                             'number of views.')
        if voxel_row_aspect <= 0 or voxel_slice_aspect <= 0:
            raise ValueError('Voxel aspect ratios must be positive.')

    def get_psf_radius(self):
        """Integer radius of the psf kernel, from the maximum magnification
        over the translated positions (used by both fans; translation has no
        separate forward vertical radius)."""
        (delta_det_row, delta_det_channel, source_iso_dist, source_detector_dist,
         recon_shape, delta_voxel, translation_vectors, voxel_row_aspect,
         voxel_slice_aspect) = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_iso_dist',
             'source_detector_dist', 'recon_shape', 'delta_voxel',
             'translation_vectors', 'voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()   # Raises at infinite SDD.
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        delta_det = min(delta_det_row, delta_det_channel)

        max_translation = np.amax(np.asarray(translation_vectors), axis=0)
        source_to_closest_pixel = source_iso_dist \
            - 0.5 * recon_shape[0] * delta_voxel_row - max_translation[1]
        if source_to_closest_pixel <= 0:
            raise ValueError('Reconstruction volume extends into source - no '
                             'valid projection in this case.')
        max_magnification = source_detector_dist / source_to_closest_pixel

        max_voxel_pitch = max(delta_voxel, delta_voxel_slice)
        psf_radius = int(np.ceil(np.ceil(max_voxel_pitch * max_magnification
                                         / delta_det) / 2))
        if psf_radius > 4:
            warnings.warn('A single voxel may project onto 100 or more detector '
                          'elements, which may lead to artifacts. Consider '
                          'using smaller voxels.')
        return psf_radius

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Compute the automatic recon shape for translation reconstruction."""
        from .utilities import calc_tct_recon_params
        sinogram_shape = self.get_params('sinogram_shape')
        source_detector_dist, source_iso_dist = self.get_params(
            ['source_detector_dist', 'source_iso_dist'])
        delta_det_row, delta_det_channel = self.get_params(
            ['delta_det_row', 'delta_det_channel'])
        translation_vectors = np.asarray(self.get_params('translation_vectors'))
        voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['voxel_row_aspect', 'voxel_slice_aspect'])

        recon_shape, delta_voxel, voxel_row_aspect = calc_tct_recon_params(
            source_detector_dist, source_iso_dist, delta_det_row,
            delta_det_channel, sinogram_shape, translation_vectors,
            voxel_row_aspect, voxel_slice_aspect)

        self.set_params(no_compile=no_compile, no_warning=no_warning,
                        recon_shape=recon_shape, delta_voxel=delta_voxel,
                        voxel_row_aspect=voxel_row_aspect)

    def _check_lateral_truncation(self, sino_indicator):
        """Do nothing.  In translation tomography the object routinely spans
        the whole field of view, so sinogram support that touches the edge is
        normal rather than the defect the base check warns about."""
        return

    def fdk_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """FDK filtering: the shared row filter with the FDK cosine pre-weight
        per detector element and the voxel-size scale alpha (as in cone;
        translation reuses cone's detector coordinate map)."""
        sinogram = self._shard_sinogram(sinogram)
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

        # The FDK cosine pre-weight is (rows, channels) and does not vary with view.
        m_grid, n_grid = np.meshgrid(np.arange(num_rows), np.arange(num_channels),
                                     indexing='ij')
        u_grid, v_grid = ConeBeamModel.detector_mn_to_uv(
            m_grid, n_grid, delta_det_channel, delta_det_row,
            det_channel_offset, det_row_offset, num_rows, num_channels)
        weight_map = source_detector_dist / np.sqrt(
            source_detector_dist ** 2 + u_grid ** 2 + v_grid ** 2)
        weight_t = torch.as_tensor(weight_map.astype(np.float32),
                                   device=self.torch_device)

        alpha = delta_det_row / (voxel_volume * M_0)
        return self._apply_direct_recon_filter(sinogram, filter_name,
                                               filter_scale=alpha,
                                               output_sharded=output_sharded,
                                               row_weight=weight_t)

    def recon_fdk(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FDK reconstruction: standard filtering, then the exact adjoint
        of the forward projector as the backprojection.

        Note:
            Translation tomography is an inherently limited-angle geometry, so
            this direct reconstruction is only approximate; it is intended as
            an initializer for the iterative ``recon()``.
        """
        # The device layout is settled before the first large allocation, so a
        # bare FDK call spreads across the GPUs instead of landing on one.
        self._apply_device_policy(workload='direct')
        filtered_sinogram = self.fdk_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self.back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

    def recon_direct(self, sinogram, filter_name="ramp", output_sharded=False):
        """Direct reconstruction by the FDK algorithm; equivalent to
        :meth:`recon_fdk`.  See :meth:`TomographyModel.recon_direct` for the
        argument and return conventions."""
        return self.recon_fdk(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """Filtering as needed for a direct recon; equivalent to
        :meth:`fdk_filter`."""
        return self.fdk_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """Estimate the standard deviation of the reconstruction from the
        sinogram.  The estimate accounts for the anisotropic row pitch and is
        used to scale sigma_prox and sigma_x."""
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        voxel_row_aspect = self.get_params('voxel_row_aspect')
        delta_voxel_row = voxel_row_aspect * delta_voxel

        typical_sinogram_value = np.average(np.abs(sinogram),
                                            weights=sino_indicator)
        # The object is assumed to fill about half the row extent.
        fraction_of_fill = 0.5
        typical_path_length = fraction_of_fill * recon_shape[0] * delta_voxel_row
        return typical_sinogram_value / typical_path_length
