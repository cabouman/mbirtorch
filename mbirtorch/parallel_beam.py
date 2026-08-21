"""ParallelBeamModel: reconstruction for parallel-beam scanner geometry.

The geometry math (compute_proj_data) is vectorized over a
batch of views: detector row r maps to recon slice r, and the horizontal fan's
inputs are the continuous channel coordinate n_p, its rounded center, the
projected width W_p_c, and the weight scale (in-plane voxel area over the
footprint length).
"""

import os
import warnings

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
        # https://mbirtorch.readthedocs.io/en/latest/theory.html
        # The FBP weight pi/num_views is folded into the filter by the shared
        # method; parallel beam has no FDK cosine pre-weight.
        delta_voxel, voxel_row_aspect = self.get_params(['delta_voxel',
                                                         'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(sinogram, filter_name,
                                               filter_scale=scaling_factor,
                                               output_sharded=output_sharded)

    def recon_fbp(self, sinogram, filter_name="ramp", output_sharded=False):
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

    def recon_direct(self, sinogram, filter_name="ramp", output_sharded=False):
        """Direct reconstruction by filtered backprojection (FBP); equivalent
        to :meth:`recon_fbp`.  See :meth:`TomographyModel.recon_direct` for
        the argument and return conventions."""
        return self.recon_fbp(sinogram, filter_name=filter_name,
                              output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fbp_filter(sinogram, filter_name=filter_name,
                               output_sharded=output_sharded)

    def recon_split_sino(self, sino, weights=None, half_overlap=5, init_recon=None, max_iterations=15,
                         stop_threshold_change_pct=0.2, first_iteration=0, compute_prior_loss=False,
                         logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                         align_split_grid=False, slices_per_part=None):
        """
        This function reduces the memory needed for parallel beam MBIR reconstruction by splitting the
        detector rows into overlapping parts, reconstructing one part at a time, and stitching the
        reconstructions together.  Memory use drops by roughly the number of parts, since only one
        part is resident at a time.

        In parallel beam geometry detector row r is recon slice r, so a band of detector rows
        reconstructs exactly the matching band of slices and the parts decouple exactly in the
        forward model.  The overlap is there for the prior: it gives the voxels near a seam their
        neighbors on the other side, so the seam is not treated as a volume boundary.

        The arguments mirror TomographyModel.recon(), and the result is approximately equal to the
        reconstruction recon() returns.  Two differences: ``output_sharded`` is not accepted, and
        ``compute_prior_loss`` is accepted but unused.

        Args:
            sino (numpy or tensor): Full sinogram of shape (num_views, num_rows, num_cols).  A
                sharded array is not accepted.
            weights (numpy or tensor, optional): Optional sinogram weights with the same shape as
                `sino`.  Not accepted in sharded form, like `sino`.
            half_overlap (int): Number of detector rows, and therefore recon slices, kept past each
                side of a seam by the part on that side.  Every interior seam is then computed twice
                over ``2 * half_overlap`` slices, which is the span the stitch blends across.
            init_recon (optional): Same as in the recon method.  Not accepted
                in sharded form, like `sino`.
            max_iterations (int, optional): Same as in the recon method.
            stop_threshold_change_pct (float, optional): Same as in the recon method.
            first_iteration (int, optional): Same as in the TomographyModel.recon() method.
            compute_prior_loss (bool, optional): Accepted for interface compatibility; not
                currently used by the mbirtorch recon.
            logfile_path (str, optional): Same as in the TomographyModel.recon() method.  The parts'
                logs are merged into this single file, each under a section header.
            print_logs (bool, optional): Same as in the TomographyModel.recon() method.
            align_split_grid (bool, optional): Accepted for interface compatibility and does
                nothing here.  Rows and slices share one grid in parallel beam, so the sub-slice
                misalignment between the sinogram cut and the recon split that this flag corrects
                for cone beam cannot exist.
            slices_per_part (int, optional): Number of slices each part keeps, which sets the number
                of parts.  Must be at least ``2 * half_overlap``.  The default, None, chooses the
                fewest parts whose reconstruction is modeled to fit the available device memory,
                using the same memory model the reconstruction itself uses; a value of at least the
                number of slices asks for a single part, which is a plain recon().

        Returns:
            Tuple[np.ndarray, dict]: the reconstructed volume (numpy array), and a
                metadata dictionary containing recon and model parameters for each
                part, plus 'split_params' (the overlap, the number of parts, and the
                slice range each part contributes).  If the volume has too few slices
                to split at this half_overlap, the method warns, performs a standard
                recon() instead, and returns that result's dictionary (no per-part
                entries).  A single part chosen by the estimate, or asked for through
                `slices_per_part`, does the same without a warning.

        Raises:
            ValueError: If inputs are missing or shapes are inconsistent, if half_overlap < 2, if
                `slices_per_part` would leave a part with fewer than ``2 * half_overlap`` slices, or
                if `sino`, `weights`, or `init_recon` is in the sharded form.
            AssertionError: If array dimensions are invalid.

        Example:
            >>> import numpy as np
            >>> import mbirtorch
            >>> sino = np.ones((180, 64, 64), dtype=np.float32)  # (views, rows, cols)
            >>> model = mbirtorch.ParallelBeamModel(sinogram_shape=sino.shape,
            ...                                     angles=np.linspace(0, np.pi, 180))
            >>> recon, recon_info = model.recon_split_sino(sino, half_overlap=4)
        """
        from . import _sharding
        from .utilities import copy_ct_model, stitch_arrays, merge_log_files

        # -------- Basic validation --------
        if half_overlap < 2:
            raise ValueError('half_overlap must be >= 2.')
        if sino is None:
            raise ValueError("sino must be provided.")
        # An input already placed on the devices is refused.  Each part settles
        # a device layout of its own, so this method works from the host array.
        # Gathering here would leave the caller's placed copy on the devices
        # for the whole call, which is the memory the split is meant to save.
        # The initial reconstruction is included in the check: it is sliced on
        # the host below, which the device form does not support.
        if (isinstance(sino, _sharding.Shards)
                or isinstance(weights, _sharding.Shards)
                or isinstance(init_recon, _sharding.Shards)):
            raise ValueError(
                'recon_split_sino does not accept a sinogram, weights, or an '
                'initial reconstruction in sharded form.  Pass the host (numpy '
                'or tensor) arrays.')
        if not (hasattr(sino, "ndim") and sino.ndim == 3):
            raise AssertionError("sino must be a 3D array shaped (num_views, num_rows, num_cols).")
        if weights is not None and getattr(weights, "shape", None) != sino.shape:
            raise AssertionError("weights, if provided, must have the same shape as sino.")

        # Operate on the host: split here and let each part's recon re-shard its own part, so the
        # full sinogram is never on the devices at once (the memory saving).  The per-part slices
        # below are then cheap host views.
        if isinstance(sino, torch.Tensor):
            sino = sino.detach().cpu().numpy()
        sino = np.asarray(sino)
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            weights = np.asarray(weights)
        if init_recon is not None and isinstance(init_recon, torch.Tensor):
            # Same host-side treatment as sino/weights: host slicing keeps only one part's arrays
            # device-resident at a time.
            init_recon = self._gather_recon(init_recon)

        num_rows = sino.shape[1]
        recon_rows, recon_cols = self.get_params('recon_shape')[:2]

        # -------- Model builders shared by the estimate and the reconstruction --------
        def _part_model(num_part_rows):
            """A copy of this model covering ``num_part_rows`` detector rows, and therefore that
            many recon slices.  The recon rows and columns are set explicitly so a parent with a
            custom in-plane recon shape keeps it; the copy's own automatic pass would recompute it
            from the detector."""
            model = copy_ct_model(self, new_num_det_rows=num_part_rows)
            # The regularization values come from the parent, which derives them from the FULL
            # sinogram below, so a part must not re-derive them from its own partial data.
            model.set_params(no_warning=True, auto_regularize_flag=False)
            model.set_params(recon_shape=(recon_rows, recon_cols, num_part_rows))
            return model

        def _worst_part_model_rows(num_parts):
            """Rows in the largest part model at this part count: the largest kept part, plus
            half_overlap for each interior side it has.  A middle part has two, an end part of a
            two-part split has one, and a single part has none."""
            biggest_kept = -(-num_rows // num_parts)
            if num_parts == 1:
                return biggest_kept
            if num_parts == 2:
                return biggest_kept + half_overlap
            return biggest_kept + 2 * half_overlap

        # -------- Choose the number of parts --------
        # Each part must keep at least 2 * half_overlap slices, so that the overlaps at its two
        # seams do not run into each other, which bounds the number of parts.
        max_parts = num_rows // (2 * half_overlap)
        estimated = False
        if slices_per_part is not None:
            if slices_per_part < 2 * half_overlap:
                raise ValueError(
                    f'slices_per_part must be at least 2 * half_overlap = {2 * half_overlap}; '
                    f'got {slices_per_part}.')
            num_parts = -(-num_rows // int(slices_per_part))
            if num_parts > 1 and num_parts > max_parts:
                raise ValueError(
                    f'slices_per_part={slices_per_part} gives {num_parts} parts of about '
                    f'{num_rows // num_parts} slices each, which is below 2 * half_overlap = '
                    f'{2 * half_overlap}; use at most {max_parts} parts, or a smaller '
                    f'half_overlap.')
        elif max_parts < 2:
            # Fewer than 4 * half_overlap slices: no split leaves both parts with the slices their
            # overlaps need, so warn and do a normal MBIR recon.
            warnings.warn(
                "the volume has too few slices to split at this half_overlap; "
                "falling back to standard MBIR reconstruction.",
                UserWarning,
            )
            num_parts = 1
        else:
            # The fewest parts whose largest part model is priced to fit the devices.  The
            # candidates are built and discarded here, so nothing about the estimate is left
            # behind on this model or on them.
            num_parts = max_parts
            for candidate in range(1, max_parts + 1):
                if _part_model(_worst_part_model_rows(candidate))._fits_available_devices():
                    num_parts = candidate
                    break
            estimated = True

        if num_parts == 1:
            return self.recon(
                sino,
                weights=weights,
                init_recon=init_recon,
                max_iterations=max_iterations,
                stop_threshold_change_pct=stop_threshold_change_pct,
                first_iteration=first_iteration,
                logfile_path=logfile_path,
                print_logs=print_logs,
            )

        # -------- The kept slice ranges, which tile [0, num_rows) in nearly equal parts --------
        base, extra = divmod(num_rows, num_parts)
        part_ranges, start = [], 0
        for index in range(num_parts):
            stop = start + base + (1 if index < extra else 0)
            part_ranges.append((start, stop))
            start = stop

        # Regularization params come from the FULL sinogram; the parts copy them and set
        # auto_regularize_flag=False so they do not re-derive from their partial sinograms.
        self.auto_set_regularization_params(sino)

        def _recon_one_part(model_lo, model_hi, part_logfile_path):
            """Reconstruct one band of detector rows on the host; return (host_recon, recon_dict).

            Builds the part's model, sinogram slice, and weights, runs recon, and gathers the result
            to the host.  All the heavy state (the part model, the device recon) is local, so it is
            released when this returns -- only ONE part's inputs are resident at a time, which is
            the point of doing one part at a time.  The returned reconstruction is a host array.
            """
            model = _part_model(model_hi - model_lo)

            # Sinogram and weight slices are host VIEWS (nothing mutates them; weights=None passes
            # through so the part recon uses its constant-weight path with no ones array built).
            sino_part = sino[:, model_lo:model_hi, :]
            weights_part = None if weights is None else weights[:, model_lo:model_hi, :]
            # Rows are slices, so the part's initial reconstruction is the matching slice band.
            part_init = None if init_recon is None else init_recon[:, :, model_lo:model_hi]

            # recon() already returns a host NumPy array (its output_sharded=False gather), so the
            # part is on the host here.
            return model.recon(sino_part, weights=weights_part, init_recon=part_init,
                               max_iterations=max_iterations,
                               stop_threshold_change_pct=stop_threshold_change_pct,
                               first_iteration=first_iteration,
                               logfile_path=part_logfile_path,
                               print_logs=print_logs)

        # -------- Reconstruct the parts ONE AT A TIME (each part is built, recon'd, gathered to the
        # host, and freed before the next is built), so only one part's sino/weights/model and one
        # part's device recon are resident at any moment. --------
        # Each part logs to its own temp file; they are merged into logfile_path afterward
        # (in finally, so any part logs written before a failure are preserved).
        if logfile_path:
            log_path = os.path.expanduser(logfile_path)
            part_log_paths = [log_path + '.part{}'.format(index) for index in range(num_parts)]
        else:
            log_path, part_log_paths = None, [None] * num_parts
        part_recons, part_dicts = [], []
        try:
            for index, (lo, hi) in enumerate(part_ranges):
                # The part's model spans its kept rows plus half_overlap on each interior side.
                model_lo, model_hi = max(lo - half_overlap, 0), min(hi + half_overlap, num_rows)
                part_recon, part_dict = _recon_one_part(model_lo, model_hi, part_log_paths[index])
                part_recons.append(part_recon)
                part_dicts.append(part_dict)
        finally:
            if log_path:
                labels = ['recon_split_sino: part {} of {} (slices {}-{})'.format(
                    index + 1, num_parts, lo, hi - 1)
                    for index, (lo, hi) in enumerate(part_ranges)]
                merge_log_files(log_path, zip(labels, part_log_paths))

        # -------- Stitch the parts together --------
        # The parts are host arrays, so stitch_arrays (host-preserving) assembles the full volume ON
        # THE HOST -- the full recon is never rebuilt on a single device, which would defeat the
        # part-at-a-time memory saving (and OOM for a recon too large to fit whole on the GPUs).
        # half_overlap is used on both sides of each seam, so the total overlap between consecutive
        # parts is 2 * half_overlap, the same for every seam.  ramp_overlap determines which slices
        # are blended, which is usually less than 2 * half_overlap to avoid possible boundary
        # effects.  ramp_overlap should be even so that it applies equally to slices on either side
        # of the seam.
        ramp_overlap = 4
        ramp_overlap = min(ramp_overlap, half_overlap)
        ramp_overlap -= ramp_overlap % 2  # ensure even
        recon_full = stitch_arrays(part_recons, axis=2, overlap=2 * half_overlap,
                                   ramp_overlap=ramp_overlap)

        # -------- Construct full reconstruction dictionary --------
        # One entry per part, in part order.  The last three split_params entries have no parallel
        # beam meaning -- rows and slices share one grid, so there is no grid shift and no cut/split
        # mismatch -- and are carried so a reader of either geometry's dictionary finds the same
        # fields.
        recon_full_dict = {'recon_params_parts': [d.get('recon_params') for d in part_dicts],
                           'recon_log_parts': [d.get('recon_log', '# Log info not saved.')
                                               for d in part_dicts],
                           'notes_parts': [d.get('notes', '# No notes saved') for d in part_dicts],
                           'model_params_parts': [d.get('model_params') for d in part_dicts],
                           'split_params': {'half_overlap_sino': int(half_overlap),
                                            'half_overlap_recon': int(half_overlap),
                                            'num_parts': int(num_parts),
                                            'part_slice_ranges': [(int(lo), int(hi))
                                                                  for lo, hi in part_ranges],
                                            'slices_per_part': int(max(hi - lo for lo, hi
                                                                       in part_ranges)),
                                            'estimated': bool(estimated),
                                            'align_split_grid': bool(align_split_grid),
                                            'grid_shift_alu': 0.0,
                                            'split_cut_mismatch_slices': 0.0}, }

        return recon_full, recon_full_dict


def recon_simple_parallel(sinogram, angles, weights=None, sharpness=1.0,
                          max_iterations=15):
    """
    Functional interface for a basic parallel-beam reconstruction.

    This builds a :class:`ParallelBeamModel` with default geometry parameters
    and reconstructs in one call.  For anything beyond the arguments here --
    changing the voxel size or the recon shape, choosing devices, controlling
    the stopping rule or the logs, restarting from a previous reconstruction --
    create the model yourself and call
    :meth:`TomographyModel.recon`; see :class:`ParallelBeamModel` for the
    geometry arguments.

    Args:
        sinogram (numpy or tensor): 3D sinogram data with shape
            (num_views, num_det_rows, num_det_channels).
        angles (numpy or tensor): 1D array of projection angles in radians, one
            per view.
        weights (numpy or tensor, optional): 3D positive weights with the same
            shape as the sinogram.  Defaults to None (all 1s).
        sharpness (float, optional): higher values give crisper edges and more
            noise; lower values give softer edges and less noise.  Defaults
            to 1.0.
        max_iterations (int, optional): maximum number of iterations.  Defaults
            to 15.  Use max_iterations=0 for a filtered back projection scaled
            to fit the data.

    Returns:
        (recon, recon_dict): the reconstruction volume, and a dict
        with entries 'recon_params' (per-iteration traces and settings),
        'recon_log' (the run's log text), 'notes', and
        'model_params' (a snapshot of the model parameters).

    Example:
        >>> import numpy as np, mbirtorch
        >>> angles = np.linspace(0, np.pi, 180, endpoint=False)
        >>> recon, recon_dict = mbirtorch.recon_simple_parallel(sinogram, angles)
    """
    # The model's geometry is read off the sinogram's shape, which a divided
    # array does not have, so that form is refused here rather than failing on
    # a missing attribute in the line that builds the model.
    from . import _sharding
    _sharding.reject_shards('recon_simple_parallel', sinogram=sinogram,
                            weights=weights)
    # A torch sinogram or torch angles are converted here, so that the model
    # gets the same plain shape tuple and host angles either way.
    if torch.is_tensor(angles):
        angles = angles.detach().cpu().numpy()
    model = ParallelBeamModel(tuple(sinogram.shape), angles)
    model.set_params(sharpness=sharpness)
    return model.recon(sinogram, weights=weights, max_iterations=max_iterations)
