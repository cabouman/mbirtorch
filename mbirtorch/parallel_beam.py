"""ParallelBeamModel: the Phase 1 geometry, ported from mbirjax.parallel_beam.

The geometry math (compute_proj_data) is reproduced exactly, vectorized over a
batch of views: detector row r maps to recon slice r, and the horizontal fan's
inputs are the continuous channel coordinate n_p, its rounded center, the
projected width W_p_c, and the weight scale (in-plane voxel area over the
footprint length).
"""

import numpy as np
import torch

from .projectors import maybe_compile
from .tomography_model import TomographyModel

_F32 = torch.float32


def _parallel_hfan_math(pixel_indices, angles_batch, num_rows, num_cols,
                        num_channels, delta_det_channel, det_channel_offset,
                        delta_voxel, delta_voxel_row):
    """The pure geometry chain of compute_hfan_data_batched, split out so it
    can be torch.compiled (the scalar parameters specialize as constants; they
    are fixed per model)."""
    row_index = (pixel_indices // num_cols).to(_F32)
    col_index = (pixel_indices % num_cols).to(_F32)
    # Compute the un-rotated coordinates relative to iso.  Note the change in
    # order from (i, j) to (y, x) (recon_ij_to_x in mbirjax).
    y_tilde = delta_voxel_row * (row_index - (num_rows - 1) / 2.0)
    x_tilde = delta_voxel * (col_index - (num_cols - 1) / 2.0)

    # Precompute cosine and sine of the view angles, then do the rotation; only
    # the x coordinate is needed for the channel projection.
    cosine = torch.cos(angles_batch)[:, None]
    sine = torch.sin(angles_batch)[:, None]
    x = cosine * x_tilde[None, :] - sine * y_tilde[None, :]

    # Calculate indices on the detector grid.
    det_center_channel = (num_channels - 1) / 2.0
    n_p = (x + det_channel_offset) / delta_det_channel + det_center_channel

    # Compute the footprint of a voxel projected onto the channels, the
    # projected voxel width in channel units, and the weight scale.
    footprint_xy = torch.maximum(cosine.abs() * delta_voxel,
                                 sine.abs() * delta_voxel_row)
    W_p_c = footprint_xy / delta_det_channel
    weight_scale = (delta_voxel_row * delta_voxel) / footprint_xy
    L_max = torch.clamp(W_p_c, max=1.0)
    centers = torch.round(n_p).to(torch.int64)
    return n_p, centers, W_p_c, weight_scale, L_max


class ParallelBeamModel(TomographyModel):
    """
    A class designed for handling forward and backward projections in a parallel
    beam geometry, extending :class:`TomographyModel`.  This class offers
    specialized methods and parameters tailored for parallel beam setups.

    This class inherits all methods and properties from TomographyModel and
    overrides some to suit parallel beam geometrical requirements.  See the
    parent class for standard methods like setting parameters and performing
    projections and reconstructions.

    Parameters not included in the constructor can be set using the set_params
    method of TomographyModel.

    Args:
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form (views, rows, channels),
            where 'views' is the number of different projection angles, 'rows'
            correspond to the number of detector rows, and 'channels' index
            columns of the detector that are assumed to be aligned with the
            rotation axis.
        angles (ndarray):
            A 1D array of projection angles, in radians, specifying the angle of
            each projection relative to the origin.
        device (str): 'auto' (cuda > mps > cpu), or an explicit torch device
            string.  Device selection is an execution-environment choice, not a
            saved model parameter (the mbirjax configure_devices rationale).
        view_batch_size (int): views per eager kernel batch (the single
            memory/speed knob of the Phase 1 drivers).

    Example:
        >>> import numpy as np, mbirtorch
        >>> angles = np.linspace(0, np.pi, 180, endpoint=False)
        >>> model = mbirtorch.ParallelBeamModel((180, 256, 10), angles)
    """

    def __init__(self, sinogram_shape, angles, device='auto', view_batch_size=64,
                 compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32)
        super().__init__(sinogram_shape, device=device, view_batch_size=view_batch_size,
                         compile_mode=compile_mode,
                         geometry_type='parallel', view_params_name='angles',
                         angles=angles)

    def get_magnification(self):
        """
        Compute the scale factor from a voxel at iso (at the origin on the center
        of rotation) to its projection on the detector.  For parallel beam, this
        is 1, but it may be parameter-dependent for other geometries.

        Returns:
            (float): magnification
        """
        return 1.0

    def get_psf_radius(self):
        """Computes the integer radius of the PSF kernel for parallel beam
        projection: the maximum number of detector channels on either side of
        the center channel hit by a voxel."""
        delta_det_channel, delta_voxel, voxel_row_aspect = self.get_params(
            ['delta_det_channel', 'delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        max_footprint = max(delta_voxel, delta_voxel_row)
        return int(np.ceil(np.ceil(max_footprint / delta_det_channel) / 2))

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Compute the default recon size using the internal parameters
        delta_det_channel and delta_det_row plus the number of channels from the
        sinogram (verbatim mbirjax math).  Run this after changing geometry
        parameters such as ``delta_det_channel``; it resets ``recon_shape`` and
        ``delta_voxel`` to reasonable values."""
        delta_det_row, delta_det_channel = self.get_params(
            ['delta_det_row', 'delta_det_channel'])
        voxel_row_aspect = self.get_params('voxel_row_aspect')

        delta_voxel = self.get_params('delta_det_channel') / self.get_magnification()
        delta_voxel_row = voxel_row_aspect * delta_voxel

        sinogram_shape = self.get_params('sinogram_shape')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        magnification = self.get_magnification()
        num_recon_rows = int(np.ceil(num_det_channels * delta_det_channel
                                     / (delta_voxel_row * magnification)))
        num_recon_cols = int(np.ceil(num_det_channels * delta_det_channel
                                     / (delta_voxel * magnification)))
        num_recon_slices = int(np.round(num_det_rows * ((delta_det_row / delta_voxel)
                                                        / magnification)))
        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning,
                        recon_shape=recon_shape, delta_voxel=delta_voxel)

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.

        Note:
            Raises ValueError for invalid parameters.
        """
        super().verify_valid_params()
        sinogram_shape, angles, voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['sinogram_shape', 'angles', 'voxel_row_aspect', 'voxel_slice_aspect'])

        if voxel_row_aspect <= 0:
            raise ValueError('Voxel row aspect ratio must be positive. \n'
                             f'Got {voxel_row_aspect} for voxel_row_aspect.')
        if voxel_slice_aspect != 1.0:
            raise ValueError('Setting voxel slice aspect ratio is not supported for '
                             f'parallel beam model. \nGot {voxel_slice_aspect}.')
        if np.asarray(angles).shape[0] != sinogram_shape[0]:
            raise ValueError('Number of view dependent parameter vectors must equal '
                             'the number of views.')
        recon_shape = self.get_params('recon_shape')
        if recon_shape[2] != sinogram_shape[1]:
            raise ValueError('Number of recon slices must match number of sinogram '
                             f'rows. \nGot {recon_shape} for recon_shape and '
                             f'{sinogram_shape} for sinogram_shape')

    def compute_hfan_data_batched(self, pixel_indices, angles_batch):
        """Compute the quantities needed for horizontal projection, for a batch
        of views (the batched form of mbirjax's compute_proj_data plus the
        wrapper's center rounding).

        Parallel-beam mapping: the voxel cylinders are assumed to have slices
        aligned with detector rows, so a parallel beam maps a cylinder slice to
        a detector row and the fan mixes CHANNELS only.  n_p is the continuous
        projected channel coordinate; its rounded value is the fans' integer
        scatter/gather center, computed ONCE per (view, pixel) so forward and
        back stay consistent (see the Projectors class note).  The weight scale
        is the in-plane voxel area over the footprint length, a per-view scalar.

        Args:
            pixel_indices: (P,) int64 tensor of indices into the flattened
                (rows, cols) grid, on the model device.
            angles_batch: (Vb,) float32 tensor of view angles in radians.

        Returns:
            (n_p, centers, W_p_c, weight_scale, L_max): n_p/centers (Vb, P);
            the per-view scalars (Vb, 1).
        """
        gp_names = ['delta_det_channel', 'det_channel_offset', 'delta_voxel',
                    'voxel_row_aspect']
        delta_det_channel, det_channel_offset, delta_voxel, voxel_row_aspect = \
            self.get_params(gp_names)
        num_channels = self.get_params('sinogram_shape')[2]
        recon_shape = self.get_params('recon_shape')
        delta_voxel_row = voxel_row_aspect * delta_voxel
        fn = maybe_compile(_parallel_hfan_math, self.compile_enabled)
        return fn(pixel_indices, angles_batch, recon_shape[0], recon_shape[1],
                  num_channels, delta_det_channel, det_channel_offset,
                  delta_voxel, delta_voxel_row)

    # ── direct recon ──────────────────────────────────────────────────────────
    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FBP filtering on the given sinogram.

        Args:
            sinogram (numpy or tensor): input with shape
                (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter.  Defaults to "ramp".
            output_sharded (bool, optional): If False (default), return numpy;
                if True, return the device tensor (the mbirjax argument name,
                kept for API compatibility).

        Returns:
            The filtered sinogram.
        """
        # Voxel-size scaling factor: adjusts the filter to account for voxel
        # size.  For the theoretical derivation see the zip linked at
        # https://mbirjax.readthedocs.io/en/latest/theory.html
        # The FBP weight pi/num_views is folded into the filter by the shared
        # method; parallel beam has no FDK cosine pre-weight.
        delta_voxel, voxel_row_aspect = self.get_params(['delta_voxel',
                                                         'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(sinogram, filter_name,
                                               filter_scale=scaling_factor,
                                               output_sharded=output_sharded)

    def fbp_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform filtered back-projection (FBP) reconstruction on the given
        sinogram.

        Our implementation uses standard filtering of the sinogram, then uses
        the adjoint of the forward projector to perform the backprojection.
        This is different from many implementations, in which the
        backprojection is not exactly the adjoint of the forward projection.
        For a detailed theoretical derivation, see the zip file linked at
        https://mbirjax.readthedocs.io/en/latest/theory.html

        Note:
            FBP assumes the view angles are EQUALLY SPACED over the full angular
            range (the ``pi / num_views`` angular weight in the ramp filter).
            On nonuniformly-spaced or limited-angle data it is only approximate
            and is best used as an initializer for the iterative ``recon()``,
            which corrects the angular weighting.

        Args:
            sinogram (numpy or tensor): input with shape
                (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter.  Defaults to "ramp".
            output_sharded (bool, optional): If False (default), return numpy;
                if True, return the device tensor.

        Returns:
            recon (numpy or tensor): the reconstructed volume.
        """
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
        recon = self.back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else recon.cpu().numpy()

    def direct_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fbp_recon(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fbp_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)
