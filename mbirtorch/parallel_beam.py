"""ParallelBeamModel: the Phase 1 geometry, ported from mbirjax.parallel_beam.

The geometry math (compute_proj_data) is reproduced exactly, vectorized over a
batch of views: detector row r maps to recon slice r, and the horizontal fan's
inputs are the continuous channel coordinate n_p, its rounded center, the
projected width W_p_c, and the weight scale (in-plane voxel area over the
footprint length).
"""

import numpy as np
import torch

from .tomography_model import TomographyModel

_F32 = torch.float32


class ParallelBeamModel(TomographyModel):
    """Parallel-beam projection model.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels).
        angles (array): 1D projection angles in radians.
        device (str): 'auto' (cuda > mps > cpu), or an explicit torch device.
        view_batch_size (int): views per eager kernel batch (the memory knob).
    """

    def __init__(self, sinogram_shape, angles, device='auto', view_batch_size=64):
        angles = np.asarray(angles, dtype=np.float32)
        super().__init__(sinogram_shape, device=device, view_batch_size=view_batch_size,
                         geometry_type='parallel', view_params_name='angles',
                         angles=angles)

    def get_magnification(self):
        return 1.0

    def get_psf_radius(self):
        """Integer radius of the channel psf, as in mbirjax."""
        delta_det_channel, delta_voxel, voxel_row_aspect = self.get_params(
            ['delta_det_channel', 'delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        max_footprint = max(delta_voxel, delta_voxel_row)
        return int(np.ceil(np.ceil(max_footprint / delta_det_channel) / 2))

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """Default recon shape and voxel pitch from the sinogram shape
        (verbatim mbirjax math)."""
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
        """The horizontal fan's inputs for a batch of views (the batched form of
        mbirjax's compute_proj_data + the wrapper's center rounding).

        Args:
            pixel_indices: (P,) int64 tensor on the model device.
            angles_batch: (Vb,) float32 tensor of view angles.

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
        num_rows, num_cols = recon_shape[0], recon_shape[1]
        delta_voxel_row = voxel_row_aspect * delta_voxel

        row_index = (pixel_indices // num_cols).to(_F32)
        col_index = (pixel_indices % num_cols).to(_F32)
        y_tilde = delta_voxel_row * (row_index - (num_rows - 1) / 2.0)
        x_tilde = delta_voxel * (col_index - (num_cols - 1) / 2.0)

        cosine = torch.cos(angles_batch)[:, None]
        sine = torch.sin(angles_batch)[:, None]
        x = cosine * x_tilde[None, :] - sine * y_tilde[None, :]

        det_center_channel = (num_channels - 1) / 2.0
        n_p = (x + det_channel_offset) / delta_det_channel + det_center_channel

        footprint_xy = torch.maximum(cosine.abs() * delta_voxel,
                                     sine.abs() * delta_voxel_row)
        W_p_c = footprint_xy / delta_det_channel
        weight_scale = (delta_voxel_row * delta_voxel) / footprint_xy
        L_max = torch.clamp(W_p_c, max=1.0)
        centers = torch.round(n_p).to(torch.int64)
        return n_p, centers, W_p_c, weight_scale, L_max

    # ── direct recon ──────────────────────────────────────────────────────────
    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """FBP row filter (the voxel-size scaling folded into the filter)."""
        delta_voxel, voxel_row_aspect = self.get_params(['delta_voxel',
                                                         'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(sinogram, filter_name,
                                               filter_scale=scaling_factor,
                                               output_sharded=output_sharded)

    def fbp_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        """Filtered back-projection: filter, then the exact adjoint back
        projector.  Assumes equally spaced views over the full angular range."""
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
