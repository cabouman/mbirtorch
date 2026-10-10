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

from .geometry_rules import channel_index, pixel_xy, rotate_about_z
from .horizontal_fan import fan_back_batch, fan_forward_batch
from .tomography_model import TomographyModel

_F32 = torch.float32


def _parallel_hfan_math(pixel_indices, view_params_batch, num_rows, num_cols,
                        num_channels, delta_det_channel, det_channel_offset,
                        delta_voxel, delta_voxel_row):
    """Return the horizontal fan data contract of horizontal_fan.py for the
    parallel geometry.  The view parameters are the view angles."""
    x_tilde, y_tilde = pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel,
                                delta_voxel_row)
    # Only the rotated x is needed.  It is the channel coordinate of a
    # parallel projection whose rays travel along -y.
    cosine = torch.cos(view_params_batch)[:, None]
    sine = torch.sin(view_params_batch)[:, None]
    x, _ = rotate_about_z(x_tilde, y_tilde, cosine, sine)
    n_p = channel_index(x, delta_det_channel, det_channel_offset, num_channels)

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
    """Parallel forward projection for one view batch.  Detector row r is
    recon slice r, so the slice axis passes through the fan as the column
    axis and a band of slices is a band of rows.

    ``plan`` is unused."""
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
    """Parallel back projection for one view batch, summed over the batch's
    views.  It is the adjoint of :func:`_parallel_forward_view_batch`.  Rows
    are slices, so the input's row band is already the output's slice band.

    ``plan`` is unused."""
    assert slice_start == 0 and band_slices is None
    hfan_data = _parallel_hfan_math(
        pixel_indices, view_params_batch, num_rows, num_cols, num_channels,
        delta_det_channel, det_channel_offset, delta_voxel, delta_voxel_row)
    sino_T = sino_batch.permute(0, 2, 1).contiguous()
    return fan_back_batch(sino_T, hfan_data, num_channels, psf_radius,
                          coeff_power=coeff_power, reduce_views=True)


class ParallelBeamModel(TomographyModel):
    """
    A model of the parallel beam geometry: the rays of every view are parallel and the object
    rotates about an axis parallel to the detector columns.

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels).  The channels run
            across the detector, perpendicular to the rotation axis.
        angles (ndarray): The rotation angle of each view, in radians.
        view_batch_size (int or None, optional): Views projected at a time.  Smaller values use
            less memory and may be slower.  None chooses automatically.  Defaults to None.
        compile_mode (str, optional): 'auto' compiles the projection kernels with
            ``torch.compile``; 'off' does not.  Defaults to 'auto'.

    Example:
        >>> angles = np.linspace(0, np.pi, 180, endpoint=False)
        >>> ct_model = mbirtorch.ParallelBeamModel((180, 256, 10), angles)
    """

    def __init__(self, sinogram_shape, angles,
                 view_batch_size=None, compile_mode='auto'):
        angles = np.asarray(angles, dtype=np.float32)
        super().__init__(sinogram_shape, view_batch_size=view_batch_size,
                         compile_mode=compile_mode,
                         geometry_type='parallel', view_params_name='angles',
                         angles=angles)

    def _project_points_batch(self, points, view_params):
        # points is (N, 3) float64 on the CPU.  view_params is (V,) view
        # angles.  Returns (row, channel), each of shape (V, N).
        ddc, dco = self.get_params(['delta_det_channel', 'det_channel_offset'])
        num_channels = self.get_params('sinogram_shape')[2]
        cosine = torch.cos(view_params)[:, None]
        sine = torch.sin(view_params)[:, None]
        x, _ = rotate_about_z(points[:, 0], points[:, 1], cosine, sine)
        channel = channel_index(x, ddc, dco, num_channels)
        # Detector row r receives recon slice r, with no spreading and no offset,
        # so the row of a point is the fractional slice index of its z.
        row = self._fractional_slice_index(points[:, 2])[None, :].expand(
            channel.shape[0], -1)
        return row, channel

    def get_magnification(self):
        """
        Compute the scale factor from a voxel at iso (at the origin on the center
        of rotation) to its projection on the detector.  For parallel beam, this
        is 1, but it may be parameter-dependent for other geometries.

        Returns:
            (float): magnification
        """
        return 1.0

    # Parallel beam ties detector row r to recon slice r, so the sharded
    # drivers take the row-aligned path.
    rows_track_slices = True

    # This name selects the measured speed floors that set the automatic
    # device count.  See _widening_floors.
    _floor_family = 'parallel'

    # The compiled parallel bodies are never called with a single pixel.  On linux
    # with torch 2.13.0 the CPU inductor miscompiles that case and places the pixel's
    # footprint one detector channel off.  The driver pads a one-pixel call to two
    # pixels and removes the padding outside the compiled region.
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
        # Each Triton kernel is used where the Triton probe and the first-use
        # value check both pass.  Set MBIRTORCH_DISABLE_TRITON=1 to turn them off.
        from .kernel_availability import (parallel_back_kernel_usable,
                                          parallel_forward_kernel_usable)
        if parallel_back_kernel_usable(self)[0]:
            from .triton_parallel import _parallel_back_view_batch_triton
            back_body = _parallel_back_view_batch_triton
        else:
            back_body = _parallel_back_view_batch
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
        # The scaling factor adjusts the filter for the voxel size.  For the
        # derivation see https://mbirtorch.readthedocs.io/en/latest/theory.html
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
        # The device layout is settled before the first large allocation.  The workload
        # name prices this direct reconstruction rather than a full iterative recon.
        self._apply_device_policy(workload='direct')
        # The sinogram is placed on the devices here, so the filter and the
        # back projection both run on the devices with no host transfer.
        sinogram = self._shard_sinogram(sinogram)
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name,
                                            output_sharded=True)
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
                         align_split_grid=False, slices_per_section=None, min_slices_per_section=200,
                         slices_per_part=None):
        """
        Reduce the memory parallel beam MBIR needs by splitting the detector rows into overlapping
        sections, reconstructing the sections separately, and joining the results.

        In parallel beam geometry detector row r is recon slice r, so a section of detector rows
        reconstructs exactly the matching section of slices.  The overlap is there for the prior: it
        gives the voxels near a seam their neighbors on the other side, so the seam is not treated
        as a volume boundary.

        The GPUs are divided into equal groups, as many as possible, so that each group can hold a
        section of at least ``min_slices_per_section`` slices.  The groups reconstruct sections
        side by side, and each group takes its sections one after another, so a volume of any
        size can be reconstructed.  When no smaller group holds such a section, all the GPUs
        reconstruct each section in turn.

        The arguments mirror TomographyModel.recon(), and the result is approximately equal to the
        reconstruction recon() returns.  Two differences: ``output_sharded`` is not accepted, and
        ``compute_prior_loss`` is accepted but unused.

        Args:
            sino (numpy or tensor): Full sinogram of shape (num_views, num_rows, num_cols).  A
                sharded array is not accepted.
            weights (numpy or tensor, optional): Optional sinogram weights with the same shape as
                `sino`.  Not accepted in sharded form, like `sino`.
            half_overlap (int): Number of detector rows, and therefore recon slices, kept past each
                side of a seam by the section on that side.  Every interior seam is then computed
                twice over ``2 * half_overlap`` slices, which is the span the join blends across.
            init_recon (optional): Same as in the recon method.  Not accepted
                in sharded form, like `sino`.
            max_iterations (int, optional): Same as in the recon method.
            stop_threshold_change_pct (float, optional): Same as in the recon method.
            first_iteration (int, optional): Same as in the TomographyModel.recon() method.
            compute_prior_loss (bool, optional): Accepted for interface compatibility; not
                currently used by the mbirtorch recon.
            logfile_path (str, optional): Same as in the TomographyModel.recon() method.  The
                sections' logs are merged into this single file, each under a header.
            print_logs (bool, optional): Same as in the TomographyModel.recon() method.
            align_split_grid (bool, optional): Accepted for interface compatibility and does
                nothing here.  Rows and slices share one grid in parallel beam, so the sub-slice
                misalignment between the sinogram cut and the recon split that this flag corrects
                for cone beam cannot exist.
            slices_per_section (int, optional): Number of slices each section keeps, which sets the
                number of sections.  Must be at least ``2 * half_overlap``.  The default, None,
                chooses the sections from the memory of the GPUs, as described above.  A value of
                at least the number of slices asks for a single section, which is a plain recon().
            min_slices_per_section (int, optional): The fewest slices a section may keep when the
                sections are chosen automatically.  Defaults to 200.
            slices_per_part (int, optional): Deprecated name of ``slices_per_section``.

        Returns:
            Tuple[np.ndarray, dict]: the reconstructed volume (numpy array), and a
                metadata dictionary containing recon and model parameters for each
                section, plus 'split_params' (the overlap, the number of sections, the
                slice range each section contributes, and the number of GPU groups and GPUs
                per group used).  If the volume has too few slices to split at this
                half_overlap, the method warns, performs a standard recon() instead, and
                returns that result's dictionary (no per-section entries).  A single section,
                chosen automatically or asked for through `slices_per_section`, does the same
                without a warning.

        Raises:
            ValueError: If inputs are missing or shapes are inconsistent, if half_overlap < 2, if
                `slices_per_section` would leave a section with fewer than ``2 * half_overlap``
                slices, or if `sino`, `weights`, or `init_recon` is in the sharded form.
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

        if slices_per_part is not None:
            warnings.warn('slices_per_part is deprecated; use slices_per_section.',
                          DeprecationWarning, stacklevel=2)
            if slices_per_section is None:
                slices_per_section = slices_per_part
        if half_overlap < 2:
            raise ValueError('half_overlap must be >= 2.')
        if sino is None:
            raise ValueError("sino must be provided.")
        # An input already placed on the devices is refused.  Each section settles a
        # device layout of its own, so this method works from host arrays.
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

        # The split is done on the host, so the full sinogram is never on the devices
        # at once.  The per-section slices below are host views.
        if isinstance(sino, torch.Tensor):
            sino = sino.detach().cpu().numpy()
        sino = np.asarray(sino)
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            weights = np.asarray(weights)
        if init_recon is not None and isinstance(init_recon, torch.Tensor):
            init_recon = self._gather_recon(init_recon)

        num_rows = sino.shape[1]
        recon_rows, recon_cols = self.get_params('recon_shape')[:2]

        def _section_model(num_section_rows):
            """Return a copy of this model covering ``num_section_rows``
            detector rows, and therefore that many recon slices."""
            model = copy_ct_model(self, new_num_det_rows=num_section_rows, no_warning=True)
            # The regularization values come from the parent, which derives them from
            # the full sinogram below.  A section must not derive its own from section data.
            model.set_params(no_warning=True, auto_regularize_flag=False)
            model.set_params(recon_shape=(recon_rows, recon_cols, num_section_rows))
            return model

        def _largest_section_rows(num_sections):
            """Return the number of rows in the largest section model at this
            section count.  It is the largest kept section plus half_overlap for
            each interior side it has."""
            biggest_kept = -(-num_rows // num_sections)
            if num_sections == 1:
                return biggest_kept
            if num_sections == 2:
                return biggest_kept + half_overlap
            return biggest_kept + 2 * half_overlap

        # Each section must keep at least 2 * half_overlap slices, so that the overlaps
        # at its two seams do not run into each other.
        max_sections = num_rows // (2 * half_overlap)
        weighted = weights is not None
        devices = self._split_devices()
        gpus_per_group = len(devices)
        estimated = False
        if slices_per_section is not None:
            if slices_per_section < 2 * half_overlap:
                raise ValueError(
                    f'slices_per_section must be at least 2 * half_overlap = {2 * half_overlap}; '
                    f'got {slices_per_section}.')
            num_sections = -(-num_rows // int(slices_per_section))
            if num_sections > 1 and num_sections > max_sections:
                raise ValueError(
                    f'slices_per_section={slices_per_section} gives {num_sections} sections of about '
                    f'{num_rows // num_sections} slices each, which is below 2 * half_overlap = '
                    f'{2 * half_overlap}; use at most {max_sections} sections, or a smaller '
                    f'half_overlap.')
            if num_sections > 1:
                section = _section_model(_largest_section_rows(num_sections))
                gpus_per_group = self._choose_gpus_per_group(
                    devices, lambda group: self._section_fits(section, group, weighted))
        elif max_sections < 2:
            # With fewer than 4 * half_overlap slices no split leaves both sections the
            # slices their overlaps need, so a normal MBIR recon runs instead.
            warnings.warn(
                "the volume has too few slices to split at this half_overlap; "
                "falling back to standard MBIR reconstruction.",
                UserWarning,
            )
            num_sections = 1
        else:
            # The devices are divided into equal groups, as many as possible, so that each
            # group can hold a section of at least min_slices_per_section slices.
            min_rows = min(int(min_slices_per_section) + 2 * half_overlap, num_rows)
            smallest_section = _section_model(min_rows)
            gpus_per_group = self._choose_gpus_per_group(
                devices, lambda group: self._section_fits(smallest_section, group, weighted))
            num_sections = max_sections
            if gpus_per_group == len(devices):
                # One group: the fewest sections whose largest model fits the devices.
                for candidate in range(1, max_sections + 1):
                    if _section_model(_largest_section_rows(candidate))._fits_available_devices():
                        num_sections = candidate
                        break
            else:
                # Several groups: the fewest sections that fit one group, and at least one
                # section per group while each keeps min_slices_per_section slices.
                group = devices[:gpus_per_group]
                for candidate in range(1, max_sections + 1):
                    if self._section_fits(_section_model(_largest_section_rows(candidate)),
                                          group, weighted):
                        num_sections = candidate
                        break
                num_groups = len(devices) // gpus_per_group
                num_sections = max(num_sections,
                                   min(num_groups, num_rows // int(min_slices_per_section), max_sections))
            estimated = True

        if num_sections == 1:
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

        # The kept slice ranges tile [0, num_rows) in nearly equal sections.
        base, extra = divmod(num_rows, num_sections)
        section_ranges, start = [], 0
        for index in range(num_sections):
            stop = start + base + (1 if index < extra else 0)
            section_ranges.append((start, stop))
            start = stop

        # The regularization parameters come from the full sinogram and its weights, the same
        # inputs recon uses, so the sections get the values recon would set.  The sections copy
        # them and set auto_regularize_flag=False.
        self.auto_set_regularization_params(sino, weights=weights)

        def _recon_one_section(model_lo, model_hi, section_logfile_path, devices, rng):
            """Reconstruct one band of detector rows and return (host_recon,
            recon_dict).

            ``devices`` is the section's group of devices, or None to let the
            section model choose its devices as recon does.  ``rng`` is the random
            generator for recon, or None.  The section's model,
            sinogram slice, and weights are local, so they are released when this
            returns.  The returned reconstruction is a host array.
            """
            model = _section_model(model_hi - model_lo)
            if devices is not None:
                model.configure_devices(devices=devices)

            # The sinogram and weight slices are host views, and nothing writes them.
            # A weights value of None passes through to the constant-weight path.
            sino_section = sino[:, model_lo:model_hi, :]
            weights_section = None if weights is None else weights[:, model_lo:model_hi, :]
            # Rows are slices, so the section's initial reconstruction is the
            # matching slice band.
            section_init = None if init_recon is None else init_recon[:, :, model_lo:model_hi]

            return model.recon(sino_section, weights=weights_section, init_recon=section_init,
                               max_iterations=max_iterations,
                               stop_threshold_change_pct=stop_threshold_change_pct,
                               first_iteration=first_iteration,
                               logfile_path=section_logfile_path,
                               print_logs=print_logs, rng=rng)

        # Each section logs to its own file.  The merge runs in a finally block so that logs
        # from a failure are kept.
        if logfile_path:
            log_path = os.path.expanduser(logfile_path)
            section_log_paths = [log_path + '.section{}'.format(index) for index in range(num_sections)]
        else:
            log_path, section_log_paths = None, [None] * num_sections

        def _job(index, lo, hi):
            """Return the job that reconstructs section ``index``.  The section's model spans its
            kept rows plus half_overlap on each interior side."""
            model_lo, model_hi = max(lo - half_overlap, 0), min(hi + half_overlap, num_rows)
            return lambda group, rng: _recon_one_section(model_lo, model_hi, section_log_paths[index],
                                                         group, rng)

        jobs = [_job(index, lo, hi) for index, (lo, hi) in enumerate(section_ranges)]
        try:
            results = self._run_split_sections(jobs, devices, gpus_per_group)
        finally:
            if log_path:
                labels = ['recon_split_sino: section {} of {} (slices {}-{})'.format(
                    index + 1, num_sections, lo, hi - 1)
                    for index, (lo, hi) in enumerate(section_ranges)]
                merge_log_files(log_path, zip(labels, section_log_paths))
        section_recons = [recon for recon, _ in results]
        section_dicts = [recon_dict for _, recon_dict in results]

        # stitch_arrays assembles the full volume on the host, with an overlap of
        # half_overlap on each side of every seam.  ramp_overlap sets which slices
        # are blended.  It is smaller than the overlap, and it is even.
        ramp_overlap = 4
        ramp_overlap = min(ramp_overlap, half_overlap)
        ramp_overlap -= ramp_overlap % 2
        recon_full = stitch_arrays(section_recons, axis=2, overlap=2 * half_overlap,
                                   ramp_overlap=ramp_overlap)

        # The dictionary holds one entry per section, in section order.  The last three
        # split_params entries have no meaning for parallel beam, and are carried
        # so that both geometries return the same fields.
        recon_full_dict = {'recon_params_sections': [d.get('recon_params') for d in section_dicts],
                           'recon_log_sections': [d.get('recon_log', '# Log info not saved.')
                                                  for d in section_dicts],
                           'notes_sections': [d.get('notes', '# No notes saved') for d in section_dicts],
                           'model_params_sections': [d.get('model_params') for d in section_dicts],
                           'split_params': {'half_overlap_sino': int(half_overlap),
                                            'half_overlap_recon': int(half_overlap),
                                            'num_sections': int(num_sections),
                                            'section_slice_ranges': [(int(lo), int(hi))
                                                                     for lo, hi in section_ranges],
                                            'slices_per_section': int(max(hi - lo for lo, hi
                                                                          in section_ranges)),
                                            'estimated': bool(estimated),
                                            'num_groups': len(devices) // gpus_per_group,
                                            'gpus_per_group': int(gpus_per_group),
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
    # The model's geometry is read from the sinogram's shape, which a divided
    # array does not have, so that form is refused.
    from . import _sharding
    _sharding.reject_shards('recon_simple_parallel', sinogram=sinogram,
                            weights=weights)
    # Torch angles are converted here so that the model gets host angles.
    if torch.is_tensor(angles):
        angles = angles.detach().cpu().numpy()
    model = ParallelBeamModel(tuple(sinogram.shape), angles)
    model.set_params(sharpness=sharpness)
    return model.recon(sinogram, weights=weights, max_iterations=max_iterations)
