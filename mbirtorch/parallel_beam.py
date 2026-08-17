"""ParallelBeamModel: reconstruction for parallel-beam scanner geometry.

The geometry math (compute_proj_data) is vectorized over a
batch of views: detector row r maps to recon slice r, and the horizontal fan's
inputs are the continuous channel coordinate n_p, its rounded center, the
projected width W_p_c, and the weight scale (in-plane voxel area over the
footprint length).
"""

import numpy as np
import torch

from .horizontal_fan import fan_back_batch, fan_forward_batch
from .tomography_model import TomographyModel

_F32 = torch.float32


def _parallel_hfan_math(pixel_indices, view_params_batch, num_rows, num_cols,
                        num_channels, delta_det_channel, det_channel_offset,
                        delta_voxel, delta_voxel_row):
    """The parallel geometry chain producing the hfan data contract (see
    horizontal_fan.py); the view parameters are the view angles here.  Pure,
    fused into the view-batch bodies below by torch.compile (the scalar
    parameters specialize as constants; they are fixed per model)."""
    row_index = (pixel_indices // num_cols).to(_F32)
    col_index = (pixel_indices % num_cols).to(_F32)
    # Compute the un-rotated coordinates relative to iso.  Note the change in
    # order from (i, j) to (y, x).
    y_tilde = delta_voxel_row * (row_index - (num_rows - 1) / 2.0)
    x_tilde = delta_voxel * (col_index - (num_cols - 1) / 2.0)

    # Precompute cosine and sine of the view angles, then do the rotation; only
    # the x coordinate is needed for the channel projection.
    cosine = torch.cos(view_params_batch)[:, None]
    sine = torch.sin(view_params_batch)[:, None]
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
    centers = torch.round(n_p).to(torch.int32)
    return n_p, centers, W_p_c, weight_scale


def _parallel_forward_view_batch(values, pixel_indices, view_params_batch,
                                 num_rows, num_cols, num_channels,
                                 delta_det_channel, det_channel_offset,
                                 delta_voxel, delta_voxel_row, psf_radius,
                                 slice_start=0, plan=None):
    """Parallel forward for one view batch: the geometry chain fused with
    the shared horizontal fan in ONE compiled body (detector row r is recon
    slice r, so the slice axis rides through the fan as the column axis and
    a slice band IS a row band -- banding needs no z anchor).

    ``plan`` is the memoization slot for a future sorted/CSR stream variant
    (per pixel-subset x view-range); unused today."""
    assert slice_start == 0
    hfan_data = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)
    block = fan_forward_batch(hfan_data, values, num_channels, psf_radius)
    return block.permute(0, 2, 1)


def _parallel_back_view_batch(sino_batch, pixel_indices, view_params_batch,
                              num_rows, num_cols, num_channels,
                              delta_det_channel, det_channel_offset,
                              delta_voxel, delta_voxel_row, psf_radius,
                              coeff_power=1, slice_start=0, band_slices=None,
                              plan=None):
    """Parallel back for one view batch, summed over the batch's views (the
    adjoint of :func:`_parallel_forward_view_batch`); rows==slices, so the
    input's row band is already the output's slice band.

    ``plan`` is the memoization slot for a future sorted/CSR stream variant
    (per pixel-subset x view-range); unused today."""
    assert slice_start == 0 and band_slices is None
    hfan_data = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)
    sino_T = sino_batch.permute(0, 2, 1).contiguous()
    return fan_back_batch(sino_T, hfan_data, num_channels, psf_radius,
                          coeff_power=coeff_power, reduce_views=True)


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
        view_batch_size (int or None, optional): number of views processed
            per projection call.  Smaller values reduce peak memory and may
            reduce speed.  None (default) chooses automatically.
        compile_mode (str, optional): 'auto' (default) compiles the
            computational kernels with torch.compile; 'off' runs without
            compilation.

    Example:
        >>> import numpy as np, mbirtorch
        >>> angles = np.linspace(0, np.pi, 180, endpoint=False)
        >>> model = mbirtorch.ParallelBeamModel((180, 256, 10), angles)
    """

    def __init__(self, sinogram_shape, angles,
                 view_batch_size=None, compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32)
        super().__init__(sinogram_shape, view_batch_size=view_batch_size,
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

    # Parallel beam ties detector row r to recon slice r 1:1 (see the base
    # attribute): the sharded drivers take the row-aligned path.
    rows_track_slices = True

    # The measured set of widening speed floors that governs this geometry's
    # automatic device count (see _widening_floors).
    _floor_family = 'parallel'

    # Never call the compiled parallel bodies with a single pixel: on linux
    # with torch 2.13.0, CPU inductor miscompiles that one-pixel case in both
    # bodies and lands the pixel's footprint one detector channel off (6.56e-02
    # relative error on the forward, 5.04e-02 on the back; eager is right, and
    # so is every width of two or more).  The driver pads a one-pixel call to
    # two and takes the padding back out, outside the compiled region --
    # projectors.forward_at_min_pixel_width holds the full measurement and the
    # argument that the padding cannot change a value.  Cone beam does not need
    # this and does not declare it.
    min_compiled_pixel_width = 2

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
        sinogram.  Run this after changing geometry
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

    def _view_batch_bodies(self):
        # The hand-written kernels are alternative BODIES: same signatures, so
        # nothing downstream of this hook changes.  Each is available wherever
        # BOTH availability gates pass -- the triton probe and the first-use
        # value self-check on the actual device (the probe-the-hardware
        # protocol; MBIRTORCH_DISABLE_TRITON=1 is the kill switch inside the
        # probe) -- and both kernels passed their composed performance gate,
        # so both are ON by default, gates alone (the cone protocol).
        #
        # Composed five-arm gate on H100, warm seeded vcd, pinned constants:
        # with BOTH kernels the 512 cell runs at 1.21x of jax's time and the
        # 1024 cell at 1.90x, at 0.63x of jax's memory -- the replacement
        # rule passes at both parallel gate cells (the compiled torch bodies
        # stood at 2.98x and 5.56x).  The forward kernel loses its ISOLATED
        # full-pixel-set bench at the 1024 cell (0.78x of the compiled body)
        # yet wins composition by 19-24% (the both-kernels arm vs the
        # back-only arm): vcd calls the forward on pixel subsets, where the
        # body pays transients and per-shape recompiles the kernel does not.
        from .kernel_availability import (parallel_back_kernel_usable,
                                          parallel_forward_kernel_usable)
        if parallel_back_kernel_usable(self)[0]:
            from .triton_parallel import _parallel_back_view_batch_triton
            back_body = _parallel_back_view_batch_triton
        else:
            back_body = _parallel_back_view_batch
        # Selection is layout-independent.  An interim rule once withheld the
        # forward kernel from sharded layouts: under the multi-device
        # drivers it disagreed with the torch forward by order one,
        # non-reproducibly, in both geometries.  The defect was the LAUNCH,
        # not the kernel: a Triton launch targets the launching thread's
        # current device, the per-device workers launch from threads whose
        # current device is 0, and the shard's consumers raced the misplaced
        # kernel.  The wrappers now bracket their launches on the tensors'
        # device, the repair is measured at the kernel-parity class on two
        # GPUs, and the standing kernel-times-sharding gate
        # (tests/test_kernels_sharded.py) holds the contract (the
        # kernel-sharding findings in the plans repo).
        if parallel_forward_kernel_usable(self)[0]:
            from .triton_parallel import _parallel_forward_view_batch_triton
            fwd_body = _parallel_forward_view_batch_triton
        else:
            fwd_body = _parallel_forward_view_batch
        return fwd_body, back_body

    def _view_batch_args(self):
        gp_names = ['delta_det_channel', 'det_channel_offset', 'delta_voxel',
                    'voxel_row_aspect']
        delta_det_channel, det_channel_offset, delta_voxel, voxel_row_aspect = \
            self.get_params(gp_names)
        num_channels = self.get_params('sinogram_shape')[2]
        recon_shape = self.get_params('recon_shape')
        return dict(num_rows=recon_shape[0], num_cols=recon_shape[1],
                    num_channels=num_channels,
                    delta_det_channel=delta_det_channel,
                    det_channel_offset=det_channel_offset,
                    delta_voxel=delta_voxel,
                    delta_voxel_row=voxel_row_aspect * delta_voxel,
                    psf_radius=self.get_psf_radius())

    # ── direct recon ──────────────────────────────────────────────────────────
    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FBP filtering on the given sinogram.

        Args:
            sinogram (numpy or tensor): input with shape
                (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter.  Defaults to "ramp".
            output_sharded (bool, optional): If False (default), return numpy;
                if True, return the device tensor.

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
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the device form: a torch
                tensor on a single device, or a Shards container (one
                tensor per device) on a multi-device model.

        Returns:
            recon (numpy or tensor): the reconstructed volume.
        """
        # Settle the device layout before the first large allocation, as
        # recon() does: a no-op when the user already chose devices;
        # otherwise the automatic selection runs here, so a bare FBP call
        # spreads across the GPUs instead of landing whole on one.  The
        # workload tells the memory check to price this reconstruction rather
        # than the full recon the device count is chosen for.
        self._apply_device_policy(workload='direct')
        # Place once at entry so the filter receives device-form data (a no-op
        # when already placed; a single device is the trivial 1-shard case).
        sinogram = self._shard_sinogram(sinogram)

        # Internal pipeline stage: keep the device form, no host transfer.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)

        # Keep the recon in the device form through the pipeline; the exit
        # handling below is the single place the output form is decided.
        recon = self.back_project(filtered_sinogram, output_sharded=True)
        return recon if output_sharded else self._gather_recon(recon)

    def direct_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        """Direct reconstruction by filtered backprojection (FBP); equivalent
        to :meth:`fbp_recon`.  See :meth:`TomographyModel.direct_recon` for
        the argument and return conventions."""
        return self.fbp_recon(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fbp_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)
