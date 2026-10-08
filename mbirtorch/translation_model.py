"""This module defines ``TranslationModel`` for translation computed tomography (TCT).  In
this geometry, each view is a cone beam projection of the object after a translation.  The
model is useful for 3D imaging of thin objects.

The projectors use two separable fans, as in the cone beam model.  Each view has an object
translation (t_x, t_y, t_z) in place of a rotation.  Both fans use the same psf radius.
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
    """This function returns the in-plane coordinates x and y of each pixel.  For each view, it
    shifts the coordinates by (-t_x, -t_y).  It also returns the magnification of each pixel.  Each output
    has shape (Vb, P), where Vb is the number of views in the batch and P is the number of pixels.
    """
    x_tilde, y_tilde = pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel,
                                delta_voxel_row)
    return _translation_xy_mag(x_tilde, y_tilde, t_x, t_y, magnification,
                               source_detector_dist)


def _translation_xy_mag(x_tilde, y_tilde, t_x, t_y, magnification,
                        source_detector_dist):
    """This function computes the same values as :func:`_translation_pixel_xy_mag`, but it
    takes positions in place of pixel indices.  It returns x, y, and pixel_mag, each of shape
    (Vb, P)."""
    y = y_tilde[None, :] - t_y[:, None]
    x = x_tilde[None, :] - t_x[:, None]
    pixel_mag = 1.0 / (1.0 / magnification - y / source_detector_dist)
    return x, y, pixel_mag


def _translation_horizontal_data(pixel_indices, view_params_batch, num_rows,
                                 num_cols, num_channels, delta_voxel,
                                 delta_voxel_row, delta_det_channel,
                                 det_channel_offset, magnification,
                                 source_detector_dist):
    """This function computes the values that the horizontal fan projector needs for one batch
    of views.  It returns n_p, centers, W_p_c, weight_scale, and pixel_mag.  Each has shape (Vb, P).
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
    """This function returns the affine map m = m0 + W_p_r * l of the vertical fan.  The map
    takes a slice index l of the full volume to a detector row m.  The function returns m0
    (Vb, P), W_p_r (Vb, P), and z_offset (Vb,).
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
    """This function returns the forward projection of one batch of views.  The output has shape
    (Vb, R, C).  The array ``values`` has shape (P, L).  It may hold only a contiguous block of
    L slices that starts at ``slice_start``.  This function does not use ``plan``.
    """
    n_p, centers, W_p_c, weight_scale, pixel_mag = _translation_horizontal_data(
        pixel_indices, view_params_batch, num_recon_rows, num_recon_cols,
        num_channels, delta_voxel, delta_voxel_row, delta_det_channel,
        det_channel_offset, magnification, source_detector_dist)
    vb, num_pixels = n_p.shape
    dev = values.device
    t_z = view_params_batch[:, 2]

    m0, W_p_r, z_offset = _translation_vertical_affine(
        pixel_mag, t_z, num_slices, delta_voxel_slice, delta_det_row,
        det_row_offset, num_rows_r)

    # 1/cos(phi) is the path length through a voxel at cone angle phi.
    band_len = values.shape[1]
    k = torch.arange(slice_start, slice_start + band_len, dtype=_F32, device=dev)
    z = (delta_voxel_slice * (k - (num_slices - 1) / 2.0))[None, None, :] \
        + z_offset[:, None, None]                                # (Vb, 1, L)
    v_slices = pixel_mag.unsqueeze(-1) * z                       # (Vb, P, L)
    cos_phi = torch.cos(torch.atan2(v_slices, torch.as_tensor(
        source_detector_dist, dtype=_F32, device=dev)))
    scaled_values = values[None, :, :] / cos_phi

    # The inverse of the affine map takes each detector row to a fractional slice index.
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
    """This function computes the back projection of one batch of views.  It sums the back
    projections of the individual views.  The output has shape (P, S), or (P, band_slices) when
    ``band_slices`` is given.
    This function does not use ``plan``.
    """
    n_p, centers, W_p_c, weight_scale, pixel_mag = _translation_horizontal_data(
        pixel_indices, view_params_batch, num_recon_rows, num_recon_cols,
        num_channels, delta_voxel, delta_voxel_row, delta_det_channel,
        det_channel_offset, magnification, source_detector_dist)
    vb, num_pixels = n_p.shape
    dev = sino_batch.device
    t_z = view_params_batch[:, 2]

    sino_T = sino_batch.permute(0, 2, 1).contiguous()            # (Vb, C, R)
    det_col = fan_back_batch(sino_T, (n_p, centers, W_p_c, weight_scale),
                             num_channels, psf_radius,
                             coeff_power=coeff_power, reduce_views=False)

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
    This class is used to compute forward and back projections in the translation (TCT)
    geometry.  The class extends :class:`TomographyModel`.  In this geometry, each view is a
    cone beam projection of the object after a translation.  The class is useful for 3D
    imaging of thin objects.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels).  num_views is
            the number of translation steps.
        translation_vectors (ndarray): A (num_views, 3) array of object translations
            (t_x, t_y, t_z) in ALU.  In each view, the object moves by the negative of the
            translation vector of that view.  A positive t_x moves the image toward a lower
            channel index.  A positive t_y moves the object away from the source.  A positive
            t_z moves the image toward a lower row index.
        source_detector_dist (float): The distance from the source to the detector in ALU.
        source_iso_dist (float): The distance from the source to the isocenter in ALU.
        view_batch_size: The same as in ``ParallelBeamModel``.
        compile_mode: The same as in ``ParallelBeamModel``.
    """

    # This name selects the translation table in ``_min_problem_sizes``, which lists minimum sinogram sizes.
    _size_table_key = 'translation'

    def __init__(self, sinogram_shape, translation_vectors, source_detector_dist,
                 source_iso_dist, view_batch_size=None, compile_mode='auto'):
        translation_vectors = np.asarray(translation_vectors, dtype=np.float32)
        if translation_vectors.ndim != 2 or translation_vectors.shape[1] != 3:
            raise ValueError('translation_vectors must have shape '
                             f'(num_views, 3); got {translation_vectors.shape}.')
        # The row neighbor weight is small because the objects are thin.  The model uses no
        # cylindrical mask because the object usually spans the field of view.
        super().__init__(sinogram_shape,
                         view_batch_size=view_batch_size, compile_mode=compile_mode,
                         geometry_type='translation',
                         view_params_name='translation_vectors',
                         translation_vectors=translation_vectors,
                         source_detector_dist=source_detector_dist,
                         source_iso_dist=source_iso_dist,
                         qggmrf_nbr_wts=[0.1, 1.0, 1.0], use_ror_mask=False)
        # A larger max_alpha has made reconstructions unstable in this geometry.
        self.set_params(no_warning=True, max_alpha=1.3)

    def _view_batch_bodies(self):
        return _translation_forward_view_batch, _translation_back_view_batch

    def _project_points_batch(self, points, view_params):
        # points is (N, 3) float64 on the CPU.  view_params is (V, 3) and holds (t_x, t_y, t_z).
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
        # The projector functions hold (Vb, P, S) and (Vb, P, R) temporary arrays for any requested block of slices.
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return max(int(recon_shape[2]), int(sinogram_shape[1]))

    def get_magnification(self):
        """This method returns the magnification, which is source_detector_dist / source_iso_dist."""
        source_detector_dist, source_iso_dist = self.get_params(
            ['source_detector_dist', 'source_iso_dist'])
        if np.isinf(source_detector_dist):
            raise ValueError('Distance from source to detector is infinite, '
                             'which means all translated projections have the '
                             'same information.')
        return source_detector_dist / source_iso_dist

    def verify_valid_params(self):
        """This method checks that all parameters are compatible for a reconstruction."""
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
        """This method returns the integer radius of the psf kernel.  It computes the radius
        from the largest magnification over all translated positions."""
        (delta_det_row, delta_det_channel, source_iso_dist, source_detector_dist,
         recon_shape, delta_voxel, translation_vectors, voxel_row_aspect,
         voxel_slice_aspect) = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_iso_dist',
             'source_detector_dist', 'recon_shape', 'delta_voxel',
             'translation_vectors', 'voxel_row_aspect', 'voxel_slice_aspect'])
        magnification = self.get_magnification()   # Raises an error when source_detector_dist is infinite.
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
        """This method sets the reconstruction shape and voxel size automatically from the
        translation geometry."""
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
        """This method does nothing.  The base class warns when the object touches the edge of
        the detector.  In translation tomography, the object normally spans the whole field of
        view, so this warning is not needed."""
        return

    def fdk_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method applies FDK filtering to the sinogram, as in the cone beam model.  It
        weights each detector element by the FDK cosine weight.  It then applies the row filter
        of :class:`TomographyModel`.  It scales the filter by a factor alpha that depends on the
        voxel size."""
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

        # The FDK cosine weight is a (rows, channels) array.  It is the same for every view.
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
        This method computes an FDK reconstruction.  It filters the sinogram with
        ``fdk_filter``.  It then back projects with the exact adjoint of the forward
        projector.

        Note:
            Translation tomography is a limited angle geometry, so this reconstruction is
            approximate.  It is intended as an initial value for ``recon()``.
        """
        # This call chooses the GPUs.  It must run before the first large allocation.
        self._apply_device_policy(workload='direct')
        filtered_sinogram = self.fdk_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self._back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

    def recon_direct(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method computes a direct reconstruction by calling :meth:`recon_fdk`.  The
        arguments and return value are as in :meth:`TomographyModel.recon_direct`."""
        return self.recon_fdk(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """This method applies the filter that a direct reconstruction needs.  It calls
        ``fdk_filter``."""
        return self.fdk_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)

    def recon_tct_direct(self, sinogram, weights=None):
        """
        Direct reconstruction derived for the translation geometry (TCT FBP) with a planar source trajectory:
        each detector row is padded with its edge values, pre-weighted, ramp filtered and cropped, then
        backprojected with the depth weight D_sd / (D_sp - y)^2 and averaged over the translation paths that
        record each voxel.  See :func:`mbirtorch.tct_direct.tct_direct_recon` for the formula.

        Unlike :meth:`recon_fdk`, which reuses the rotational FDK weights with the adjoint backprojector, this
        reconstruction follows the TCT direct reconstruction theory; like any direct method for this
        limited-angle geometry, it reconstructs only the frequencies the scan measures.

        Args:
            sinogram (numpy or torch array): Sinogram with shape (num_views, num_det_rows, num_det_channels).
            weights (numpy or torch array, optional): Sinogram weights; the valid detector panel is the bounding
                box of the elements with a positive weight.  Defaults to None (whole detector).

        Returns:
            numpy array: The reconstruction with shape ``recon_shape``.
        """
        from .tct_direct import tct_direct_recon
        return tct_direct_recon(self, sinogram, weights=weights)

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """This method estimates the standard deviation of the reconstruction from the
        sinogram.  The estimate uses the voxel size along the row axis."""
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        voxel_row_aspect = self.get_params('voxel_row_aspect')
        delta_voxel_row = voxel_row_aspect * delta_voxel

        typical_sinogram_value = np.average(np.abs(sinogram),
                                            weights=sino_indicator)
        # The estimate assumes that the object fills about half of the rows.
        fraction_of_fill = 0.5
        typical_path_length = fraction_of_fill * recon_shape[0] * delta_voxel_row
        return typical_sinogram_value / typical_path_length
