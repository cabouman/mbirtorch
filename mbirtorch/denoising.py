"""QGGMRFDenoiser: a qGGMRF proximal-map image denoiser.

The denoiser uses the recon framework: the forward model is the IDENTITY (the
residual image plays the role of the error sinogram), so no projectors exist
and the VCD subset update reduces to the closed-form below.  A plain python
loop runs over the subsets of ONE fixed partition, in order.  Unlike
_vcd_recon, which reshuffles each iteration, the denoiser never reshuffles, so
a seeded partition makes the whole sweep deterministic.

There are two paths.  On one device the whole sweep runs through the compiled
in-place update below.  Across several devices the image is slice-sharded and
the sweep runs shard by shard, with the qGGMRF halos staged once per pass and
the four line-search sums combined on the lead device into one step size.
Both paths keep the line search on device, so neither forces a host
synchronization per subset.

A third form, :meth:`QGGMRFDenoiser.denoise_stack`, denoises a stack of
same-shaped volumes with shared parameters on one device.  Its update carries
a leading volume axis, so every volume has its own step size and its own
stopping test, and the result equals denoising the volumes one at a time.
"""

import datetime

import numpy as np
import torch

from . import _memory_ledger, _sharding

from . import qggmrf as _qggmrf
from . import vcd_utils
from ._memory_ledger import image_ell1, stack_ell1
from ._utils import _AUTO_REGULARIZATION_PARAM_NAMES, recon_param_names
from .projectors import maybe_compile
from .tomography_model import TomographyModel

_F32_EPS = float(np.finfo(np.float32).eps)
# The smallest sigma_x the stack statistics will set; see
# QGGMRFDenoiser.auto_set_regularization_params_from_stack.
_SIGMA_X_FLOOR = 1e-6
# The most voxels the whole-volume statistics read.  It is the budget
# estimate_image_noise_std already keeps for its own subsample.
_STATISTICS_POINT_BUDGET = 5_000_000


#: The tile grid the point budget is spent on, and the smallest tile edge
#: worth reading.  A grid of small tiles samples the whole field of view;
#: one block of the same area would sample only the middle of it.
#: Two tiles per axis, four in all, measured best: over five positions of an
#: object boundary its worst error against the whole-stack estimate was
#: 2.9e-2, against 4.3e-2 for one block and 6.2e-2 for four or eight tiles
#: per axis.
_TILE_GRID = 2
_MIN_TILE = 4


def _sample_tiles(num_rows, num_cols, num_leading, point_budget):
    """Tiles of two axes that together hold at most ``point_budget`` voxels,
    counting ``num_leading`` entries of a third axis that is left whole.

    The tiles are contiguous, so a neighbor difference inside one is between
    adjacent voxels; a strided subsample would compare voxels a stride apart,
    which measures something else.  They are spread evenly across both axes,
    first tile at one edge and last at the other, because the estimate must
    see the whole field of view.  One block of the same area would sample
    only the middle: at 30 frames of 512 cubed the budget buys 18 voxels per
    axis, so a single block reads a needle 3.5 percent as wide as the object,
    and misses the boundaries, where the largest neighbor differences are.

    Args:
        num_rows (int): the extent of the first tiled axis.
        num_cols (int): the extent of the second tiled axis.
        num_leading (int): the number of entries of the axis left whole.
        point_budget (int): the most voxels the tiles may hold together.

    Returns:
        (list of slice, list of slice): the rows and the columns of the tiles.
        Their product is the tile grid.  Each list holds one whole-axis slice
        when the budget is not reached.
    """
    num_rows, num_cols, num_leading = int(num_rows), int(num_cols), int(num_leading)
    total = num_leading * num_rows * num_cols
    if total <= point_budget or total <= 0:
        return [slice(0, num_rows)], [slice(0, num_cols)]
    # The extent each axis may keep, spent as a grid of equal tiles.
    side = max(2, int((point_budget / num_leading) ** 0.5))
    count = max(1, min(_TILE_GRID, side // _MIN_TILE))
    edge = max(2, side // count)

    def spread(extent):
        """The tiles of one axis, centered on equal shares of it.

        The shares are ``(index + 0.5) / count`` of the axis, so no tile sits
        against an edge.  Tiles placed edge to edge instead would, at the
        production budget, put four tiles nine voxels wide in the four
        corners of a 512 by 512 field, where a reconstruction holds only air.
        """
        width = min(edge, extent)
        if count == 1 or width >= extent:
            start = (extent - width) // 2
            return [slice(start, start + width)]
        starts = sorted({min(max(0, round((index + 0.5) * extent / count) - width // 2),
                             extent - width)
                         for index in range(count)})
        return [slice(start, start + width) for start in starts]

    return spread(num_rows), spread(num_cols)


def vcd_subset_denoiser(flat_image, flat_error_image, pixel_indices,
                        fm_constant, qggmrf_params, image_shape):
    """One VCD subset update for the identity forward model (the analog of
    vcd_subset_updater).  Mutates both state tensors in place and returns
    (flat_image, flat_error_image, ell1, alpha).

    The formulas and their order of operations are fixed by the golden-value
    tests (tests/test_denoiser.py); do not rearrange them."""
    # qGGMRF prior - compute the gradient and Hessian at each pixel in the set.
    prior_grad, prior_hess = _qggmrf.qggmrf_gradient_and_hessian_at_indices(
        flat_image, image_shape, pixel_indices, qggmrf_params)

    # "Back project" the residual - the forward Hessian is all 1s for the
    # qggmrf proximal map.
    cur_error_image = flat_error_image[pixel_indices]
    forward_grad = -fm_constant * cur_error_image
    forward_hess = 1

    # Compute the update direction in the recon domain.
    delta_recon_at_indices = -((forward_grad + prior_grad)
                               / (forward_hess + prior_hess))

    # Compute delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha.
    prior_linear = torch.sum(prior_grad * delta_recon_at_indices)

    # Estimated upper bound for the prior Hessian term.
    prior_quadratic_approx = torch.sum(prior_hess * delta_recon_at_indices ** 2)

    # The "sinogram-domain" direction IS the recon-domain direction (identity A).
    delta_sinogram = delta_recon_at_indices
    forward_linear = fm_constant * torch.sum(cur_error_image * delta_sinogram)
    forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram)

    # Compute the optimal update step.
    alpha_numerator = forward_linear - prior_linear
    alpha_denominator = forward_quadratic + prior_quadratic_approx + _F32_EPS
    alpha = alpha_numerator / alpha_denominator
    max_alpha = 1.5
    alpha = torch.clamp(alpha, _F32_EPS, max_alpha)

    delta_recon_at_indices = alpha * delta_recon_at_indices
    flat_image.index_add_(0, pixel_indices, delta_recon_at_indices)

    # Update the residual at the subset's pixels.
    cur_error_image = cur_error_image - alpha * delta_sinogram
    flat_error_image.index_copy_(0, pixel_indices, cur_error_image)
    ell1_for_subset = torch.sum(torch.abs(delta_recon_at_indices))
    return flat_image, flat_error_image, ell1_for_subset, alpha


def vcd_subset_denoiser_batched(flat_image, flat_error_image, pixel_indices,
                                fm_constant, qggmrf_params, image_shape, active):
    """One VCD subset update for a stack of images under the identity forward
    model: :func:`vcd_subset_denoiser` with a leading volume axis.

    The formulas are those of the single-image function.  The four line-search
    sums are reduced over the pixel and slice axes of each volume, so the step
    size ``alpha`` has one entry per volume and the volumes do not interact.
    ``active`` is a boolean tensor with one entry per volume; a volume whose
    entry is False gets a step of zero, so its image and its residual do not
    change.  Both state tensors, of shape (num_volumes, num_pixels,
    num_slices), are mutated in place.

    Returns:
        (flat_image, flat_error_image, ell1, alpha): the two state tensors,
        the ell-1 norm of each volume's step, shape (num_volumes,), and the
        step size of each volume, shape (num_volumes,).
    """
    prior_grad, prior_hess = _qggmrf.qggmrf_gradient_and_hessian_batched(
        flat_image, image_shape, pixel_indices, qggmrf_params)

    # The forward Hessian is all 1s for the qggmrf proximal map.
    cur_error_image = flat_error_image[:, pixel_indices]
    forward_grad = -fm_constant * cur_error_image
    forward_hess = 1

    delta_recon_at_indices = -((forward_grad + prior_grad)
                               / (forward_hess + prior_hess))

    # The line-search sums, one per volume.
    prior_linear = torch.sum(prior_grad * delta_recon_at_indices, dim=(1, 2))
    prior_quadratic_approx = torch.sum(prior_hess * delta_recon_at_indices ** 2,
                                       dim=(1, 2))

    delta_sinogram = delta_recon_at_indices
    forward_linear = fm_constant * torch.sum(cur_error_image * delta_sinogram,
                                             dim=(1, 2))
    forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram,
                                                dim=(1, 2))

    alpha_numerator = forward_linear - prior_linear
    alpha_denominator = forward_quadratic + prior_quadratic_approx + _F32_EPS
    alpha = alpha_numerator / alpha_denominator
    max_alpha = 1.5
    alpha = torch.clamp(alpha, _F32_EPS, max_alpha)
    # A frozen volume takes no step.
    alpha = torch.where(active, alpha, torch.zeros_like(alpha))

    delta_scaled = alpha[:, None, None] * delta_recon_at_indices
    flat_image.index_add_(1, pixel_indices, delta_scaled)

    cur_error_image = cur_error_image - delta_scaled
    flat_error_image.index_copy_(1, pixel_indices, cur_error_image)
    ell1_for_subset = torch.sum(torch.abs(delta_scaled), dim=(1, 2))
    return flat_image, flat_error_image, ell1_for_subset, alpha


def _volume_shape(image):
    """The shape of a 3D volume in any form the denoiser accepts: a numpy
    array, a torch tensor on any device, or a slice-sharded Shards.

    A denoiser image is sharded on the LAST axis, so its slice count is the
    sum of the per-shard widths; rows and columns are not sharded, and every
    shard holds all of them."""
    if isinstance(image, _sharding.Shards):
        first = image.tensors[0]
        num_slices = sum(int(t.shape[-1]) for t in image.tensors)
        return int(first.shape[0]), int(first.shape[1]), num_slices
    if torch.is_tensor(image):
        return tuple(int(n) for n in image.shape)
    return tuple(int(n) for n in np.asarray(image).shape)


def _subsample_to_host(image, row_step=1, col_step=1, slice_step=1):
    """Return ``numpy.asarray(image)[::row_step, ::col_step, ::slice_step]``
    for a 3D volume in any form the denoiser accepts, without ever holding
    the whole volume on the host.

    The denoiser's two statistics -- the noise estimate and the
    auto-regularization parameters -- each look at a small strided subsample
    of the image and at nothing else, so only that subsample needs to cross
    to the host.  For a tensor or for shards, the strided block is taken on
    the device that holds the data and only its elements are copied over.

    For sharded input the result is EXACT: the same elements in the same
    order as striding the assembled volume, because this is data movement
    rather than an approximation.  Rows and columns are not sharded, so every
    shard is strided identically on those two axes.  On the sharded last
    axis, shard k owns global slices ``[start_k, end_k)``, so the sampled
    global positions ``0, slice_step, 2 * slice_step, ...`` that land in that
    block are the local positions ``j`` with
    ``(start_k + j) % slice_step == 0`` -- they begin at local offset
    ``(-start_k) % slice_step`` and continue by ``slice_step``.  Taking each
    shard's block from that offset and concatenating on the last axis
    reproduces the strided volume slice for slice.  A shard that owns no
    slices, or one in which no sampled position lands, contributes a
    zero-width block, which changes nothing.

    Incidentally this also removes a latent failure on a single-device CUDA
    model: ``numpy.asarray`` raises on a CUDA tensor, so a caller's tensor
    handed straight to numpy would fail there.  Every tensor path here goes
    through ``.cpu()`` first.
    """
    def block_to_host(tensor, slice_start):
        """One tensor's strided block, made dense on its own device so that
        the copy crossing to the host carries only the sampled elements."""
        block = tensor[::row_step, ::col_step, slice_start::slice_step]
        return block.detach().contiguous().cpu().numpy()

    if isinstance(image, _sharding.Shards):
        placement = image.placement
        if placement.axis % 3 != 2:
            raise ValueError(
                'A denoiser image must be sharded on its last (slice) axis; '
                'got a placement on axis {}.'.format(placement.axis))
        num_slices = _volume_shape(image)[2]
        blocks = [block_to_host(tensor, (-start) % slice_step)
                  for tensor, (_dev, (start, _end))
                  in zip(image.tensors, placement.shard_ranges(num_slices))]
        return np.concatenate(blocks, axis=-1)
    if torch.is_tensor(image):
        return block_to_host(image, 0)
    return np.asarray(image)[::row_step, ::col_step, ::slice_step]


class QGGMRFDenoiser(TomographyModel):
    """
    The QGGMRFDenoiser uses the recon framework to implement a qggmrf proximal
    map denoiser.  The primary interface is through :meth:`denoise`.

    With default settings, and with X a clean image and W equal to AWGN of
    standard deviation sigma_noise, the result of :meth:`denoise` applied to
    X + W is the MAP estimate of the denoised image using the qGGMRF prior.

    :meth:`denoise` settles the device layout through the same
    once-per-model automatic policy ``recon`` uses, sized by the denoiser's
    own memory plan rather than by a reconstruction it will never run.
    ``configure_devices`` pins a layout explicitly, and an explicit layout is
    never second-guessed.  On a multi-device layout the image is divided
    across the devices by slice.

    Args:
        image_shape (tuple of int): shape of the images to denoise
            (3-dimensional).  To denoise a 2D image, use shape (1, m, n).
        compile_mode (str, optional): 'auto' (default) compiles the
            computational kernels with torch.compile; 'off' runs without
            compilation.
    """

    # The measured widening speed floors that govern this class's automatic
    # device count (see _widening_floors).  Both denoiser rows are sentinels:
    # sharded denoising lost at every size probed up to a billion image
    # voxels, so the automatic path holds a denoiser at one device and only
    # capacity widens it.  The family's floors are read in IMAGE VOXELS,
    # because this class's sinogram shape is its image shape.
    _floor_family = 'denoiser'

    def __init__(self, image_shape, compile_mode='auto'):
        if len(image_shape) != 3:
            raise ValueError('image_shape must be 3-dimensional. Got image_shape={}. '
                             'To denoise a 2D image, use shape (1, m, n).'.format(image_shape))
        super().__init__(image_shape, compile_mode=compile_mode,
                         view_params_name='None', sigma_noise=None)
        self.set_params(use_ror_mask=False)
        self.set_params(sharpness=0)   # the denoiser's default sharpness level
        # For qggmrf denoising a single fixed partition suffices.
        self.set_params(granularity=[16], partition_sequence=[0])

    def get_magnification(self):
        """Return 1 to satisfy the TomographyModel interface."""
        return 1.0

    def verify_valid_params(self):
        """Check that all parameters are compatible for a denoise."""
        super().verify_valid_params()
        sinogram_shape = self.get_params('sinogram_shape')
        image_shape = self.get_params('recon_shape')
        if tuple(image_shape) != tuple(sinogram_shape):
            raise ValueError('image_shape and sinogram_shape must be the same. \n'
                             f'Got {image_shape} for image_shape and '
                             f'{sinogram_shape} for sinogram_shape')

    def create_projectors(self):
        # The identity forward model has no projectors.
        pass

    def get_psf_radius(self):
        raise NotImplementedError('get_psf_radius is not implemented for QGGMRFDenoiser.')

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """The recon (image) shape equals the sinogram (input image) shape."""
        sinogram_shape = self.get_params('sinogram_shape')
        self.set_params(no_compile=no_compile, no_warning=no_warning,
                        recon_shape=sinogram_shape)

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        # sigma_y IS the noise level for the identity forward model.
        sigma_y = self.get_params('sigma_noise')
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def _check_lateral_truncation(self, sino_indicator):
        """No-op override: the denoiser's 'sinogram' is an ordinary image, and
        image content reaching the frame edge is normal."""
        return

    def estimate_image_noise_std(self, image):
        """
        Estimate the noise standard deviation from the image (two passes of
        support-indicator + neighbor-difference std).

        Only a strided subsample of at most about five million points is ever
        used, so the element count and the stride come from the image's
        SHAPE rather than from a host copy, and the subsample itself is taken
        through :func:`_subsample_to_host`.  A sharded image is therefore
        strided on its own devices and never brought over whole.  The stride
        arithmetic is unchanged, so any given image still yields the estimate
        it always did.
        """
        num_rows, num_cols, num_slices = _volume_shape(image)
        num_elements = num_rows * num_cols * num_slices
        num_pts_to_use = np.minimum(5_000_000, num_elements)
        stride = round((num_elements / num_pts_to_use) ** (1 / 3))
        small_image = _subsample_to_host(image, stride, stride, stride)

        support_indicator = self._get_sino_indicator(small_image, sigma_noise=0.0)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
        support_indicator = self._get_sino_indicator(small_image, sigma_noise=sigma_noise)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
        return sigma_noise

    def _get_estimate_of_recon_std(self, noisy_image, support_indicator):
        """Neighbor-difference std over the support (the denoiser's own recon-std
        estimate, replacing the projection-path-length formula).

        Each voxel of the support is compared with itself and with its three
        backward neighbors, one per axis, and the mean of those per-voxel
        standard deviations is returned.  A voxel on an edge takes its
        neighbor from the far side, which is what indexing with -1 did.

        The four values are read as shifted views of the image rather than
        gathered through an index array.  The gather held about sixteen
        arrays of the image's size, because ``np.where`` returns three int64
        arrays and the stack of four gathered copies doubles again; the
        shifted views hold about five at their peak, and the spread is taken
        in two passes, which is stable.
        """
        def views():
            """The voxel and its three backward neighbors, one array at a
            time, so that only one shifted copy exists at once."""
            yield noisy_image
            for axis in range(3):
                yield np.roll(noisy_image, 1, axis=axis)

        count = 4
        mean = np.array(noisy_image, dtype=noisy_image.dtype, copy=True)
        for index, value in enumerate(views()):
            if index:
                mean += value
        mean /= count
        spread = np.zeros_like(mean)
        deviation = np.empty_like(mean)
        for value in views():
            np.subtract(value, mean, out=deviation)
            deviation *= deviation
            spread += deviation
        spread /= count
        np.sqrt(spread, out=spread)
        return np.mean(spread[support_indicator.astype(bool)])

    def _get_sino_indicator(self, noisy_image, sigma_noise=None, verbose=1):
        """Binary support indicator for the noisy image: threshold at a small
        fraction of the mean magnitude plus the noise floor."""
        if sigma_noise is None:
            sigma_noise = self.get_params('sigma_noise')
        percent_noise_floor = 5.0
        threshold = (0.01 * percent_noise_floor) * np.mean(np.fabs(noisy_image)) + sigma_noise
        threshold = min(threshold, np.amax(noisy_image))
        return np.int8(noisy_image >= threshold)

    def recon(self, *args, **kwargs):
        raise NotImplementedError('recon is not implemented for QGGMRFDenoiser.  '
                                  'Use `denoise` instead.')

    def denoise(self, image, sigma_noise=None, use_ror_mask=False, init_image=None,
                max_iterations=15, stop_threshold_change_pct=0.2, first_iteration=0,
                logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                output_sharded=False):
        """
        Compute the MAP denoiser assuming AWGN and the 3D qGGMRF prior.

        The amount of denoising can be changed by changing sigma_noise.  If
        sigma_noise is None, it is estimated from the image.  Denoising strength
        can also be adjusted with the ``sharpness`` parameter (default 0.0).

        The first call settles the model's device layout, so it may raise the
        memory preflight's ``MemoryPreflightError`` when no device count
        holds the sweep.  ``MBIRTORCH_NUM_DEVICES`` caps the automatic count,
        and ``configure_devices`` fixes it outright.

        Args:
            image (numpy or tensor or Shards): the 3D volume to be denoised.
                The slice-sharded device form is accepted too, so a
                Plug-and-Play loop can feed back what a reconstruction
                returned with ``output_sharded=True``, provided the two models
                share a device layout (see
                :meth:`~mbirtorch.TomographyModel.configure_devices` and its
                ``like=`` argument).
            sigma_noise (float, optional): estimated noise std in the image.
                If None, estimated from the image.  ``sigma_y`` is kept equal
                to ``sigma_noise`` (for the identity forward model they are
                the same parameter), whether or not auto-regularization is on.
            use_ror_mask: restrict denoising to a masked region (False default;
                True for the inscribed ellipse, or a custom 2D mask).
            init_image (numpy or tensor or Shards, optional): initial image
                for the minimization, in a plain array or in the device form.
                Defaults to ``image``.
            max_iterations (int, optional): maximum VCD iterations.
            stop_threshold_change_pct (float, optional): stop when
                100 * ||delta||_1 / ||image||_1 drops below this.  0 guarantees
                exactly max_iterations.
            first_iteration (int, optional): iteration label offset for logs.
            logfile_path (str, optional): Path to the output log file ('~' expands to the
                user's home directory).  If None or empty, no log file is written.
                Defaults to '~/.mbirtorch/logs/recon.log'.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): if True return the device form
                (slice-sharded across several devices).

        Returns:
            (denoised_image, denoiser_dict): the denoised volume, and a dict
            with entries 'recon_params', 'recon_log', 'notes', and
            'model_params' (as in :meth:`TomographyModel.get_recon_dict`).

        Example:
            >>> denoiser = mbirtorch.QGGMRFDenoiser(noisy_image.shape)
            >>> denoised_image, d = denoiser.denoise(noisy_image, sigma_noise=0.1)
        """
        self._log_run_header(first_iteration, logfile_path, print_logs)
        # Settle the device layout before the image is placed.  The denoiser
        # prices its own plan: it has no projectors, so a recon-sized plan
        # would charge arrays it never allocates.  init_image rides along so a
        # caller-supplied initial image is priced as the fourth resident image.
        self._apply_device_policy(workload='denoise', init_recon=init_image)
        self._log_device_report()

        # The noise and regularization estimates below each run on a small
        # strided subsample of the image and never on the whole volume, so
        # each one brings over only the elements it reads.  A sharded input is
        # subsampled on its own devices, so no full copy crosses to the host
        # and a caller that keeps its volume on the devices (a plug-and-play
        # loop, say) pays no whole-volume transfer per denoise.
        self.set_params(no_warning=True, use_ror_mask=use_ror_mask)
        if sigma_noise is None:
            # This one strides all three axes itself, so it takes the image in
            # whatever form the caller supplied.  Handing it the row subsample
            # built below would change the estimate.
            sigma_noise = self.estimate_image_noise_std(image)
        # For the identity forward model sigma_y IS sigma_noise, so the two
        # are kept equal here rather than only in the flag-gated auto path:
        # a pinned denoiser (auto_regularize_flag=False, the Plug-and-Play
        # agent configuration) must still take its strength from sigma_noise.
        self.set_params(no_warning=True, sigma_noise=sigma_noise,
                        sigma_y=sigma_noise)
        self.logger.info('Initializing QGGMRFDenoiser')

        # Auto-regularization with the background-estimation warning suppressed.
        # auto_set_regularization_params begins by calling subsample_views,
        # which keeps every step_size-th row and reads nothing else, so giving
        # it those rows instead of the volume gives it exactly the same data:
        # one such subsample leaves at most 39 rows, and at 39 rows or fewer
        # subsample_views uses a step size of 1, so its own call passes them
        # through unchanged.  (Checked for every row count from 1 to 4999.)
        # The step comes from subsample_views itself, applied to the row
        # indices, rather than from a second copy of its rule here.
        num_rows = _volume_shape(image)[0]
        sampled_rows = self.subsample_views(np.arange(num_rows))
        row_step = int(sampled_rows[1] - sampled_rows[0]) if sampled_rows.size > 1 else 1
        small_image = _subsample_to_host(image, row_step=row_step)
        verbose = self.get_params('verbose')
        self.set_params(no_warning=True, verbose=0)
        regularization_params = self.auto_set_regularization_params(small_image)
        self.set_params(no_warning=True, verbose=verbose)

        # One fixed partition (sequential subsets; no per-iteration shuffle).
        image_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partition_sequence = self.get_params('partition_sequence')
        partition_index = partition_sequence[0]
        use_ror_mask = self.get_params('use_ror_mask')
        partitions = vcd_utils.gen_set_of_pixel_partitions(
            image_shape, [granularity[partition_index]],
            device=self.torch_device, use_ror_mask=use_ror_mask)
        partition = partitions[0]

        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = (b, sigma_x, p, q, T)
        max_iters = max_iterations
        stop_thresh = stop_threshold_change_pct / 100.0

        image_t = self._shard_recon(image)
        init_t = image_t if init_image is None else self._shard_recon(init_image)

        self.logger.info('Starting VCD iterations')
        if isinstance(image_t, _sharding.Shards):
            flat_image, nmae_update, alpha_values, num_iters = self._denoise_sharded(
                image_t, init_t, partition, fm_constant, qggmrf_params,
                tuple(image_shape), max_iters, stop_thresh, first_iteration, verbose)
            denoised = _sharding.Shards(
                [t.reshape(s.shape) for t, s in zip(flat_image.tensors, image_t.tensors)],
                flat_image.placement)
        else:
            # Single device: the whole sweep through one compiled in-place update.
            flat_image = init_t.clone().reshape((-1, image_shape[2])).contiguous()
            flat_error_image = (image_t.reshape((-1, image_shape[2]))
                                - flat_image).contiguous()
            subset_denoiser = maybe_compile(vcd_subset_denoiser, self.compile_enabled)
            nmae_update = np.zeros(max_iters)
            alpha_values = np.zeros(max_iters)
            num_iters = 0
            with torch.no_grad():
                for i in range(max_iters):
                    ell1_accum = 0.0
                    alpha_accum = 0.0
                    for k in range(partition.shape[0]):
                        flat_image, flat_error_image, ell1_subset, alpha_subset = \
                            subset_denoiser(flat_image, flat_error_image, partition[k],
                                            fm_constant, qggmrf_params, tuple(image_shape))
                        ell1_accum = ell1_accum + ell1_subset
                        alpha_accum = alpha_accum + alpha_subset

                    # Chunked rather than sum(abs) over the whole image: see
                    # image_ell1 for the temporary this avoids and for why the
                    # fused norm is not used instead.  A zero image gives nan
                    # rather than raising ZeroDivisionError, as in _vcd_recon's
                    # iteration statistics (a Plug-and-Play loop initialized
                    # at zero feeds one in).
                    image_l1 = float(image_ell1(flat_image))
                    nmae = (float(ell1_accum) / image_l1 if image_l1
                            else float('nan'))
                    nmae_update[i] = nmae
                    alpha_values[i] = float(alpha_accum) / partition.shape[0]
                    num_iters += 1
                    if verbose >= 1 and (i % 5) == 0:
                        self.logger.info('After iteration {} of a max of {}: Pct change={:.4f}'
                                         .format(i + first_iteration, max_iters, 100 * nmae))
                    if nmae < stop_thresh:
                        break
            denoised = flat_image.reshape(tuple(image_shape))

        recon_params = dict(zip(recon_param_names,
                                [int(num_iters), granularity, partition_sequence,
                                 None, None, regularization_params,
                                 [100 * float(v) for v in nmae_update[:num_iters]],
                                 [float(v) for v in alpha_values[:num_iters]],
                                 None]))
        # This call has written its last log line, so finish the file rather
        # than holding it open, as recon and prox_map do.  A call continuing
        # this run reopens it.
        self.close_log_file()
        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        denoiser_dict = self.get_recon_dict(recon_params, notes=notes)
        return (denoised if output_sharded else self._gather_recon(denoised)), denoiser_dict

    def _denoise_sharded(self, image_sh, init_sh, partition, fm_constant,
                         qggmrf_params, image_shape, max_iters, stop_thresh,
                         first_iteration, verbose):
        """Run the denoising sweep across devices on slice-sharded state.

        Mirrors _vcd_recon's sharded path: the qGGMRF halos are staged once
        per pass, each device computes its shard's prior and identity-forward
        terms, and the four line-search sums combine ON THE LEAD DEVICE into
        one step size (the same formula as vcd_subset_denoiser).

        The line search stays on device for the reason _vcd_recon states at
        its own combine: alpha is a scalar tensor, so no host synchronization
        is forced per subset.  Reading the four sums back as Python floats
        would cost 5 x n_devices device-to-host syncs per subset per pass,
        from inside worker threads, for a scalar that is only ever consumed
        on the devices again.  The single-device denoiser already keeps these
        as tensors (see :func:`vcd_subset_denoiser`), so this also puts the
        two paths on the same float32 arithmetic; the host syncs that remain
        are one per PASS, for the convergence test and the logged history.

        Returns (flat_image shards, nmae history, alpha history, num_iters).
        """
        devices = image_sh.placement.devices
        n = len(devices)
        pl = image_sh.placement
        dev0 = devices[0]

        def combine_on_lead(parts):
            """Sum per-shard 0-d tensor partials on the lead device: the
            identity on one device, scalar-sized device moves otherwise."""
            total = parts[0]
            for part in parts[1:]:
                total = total + _sharding.move_shard(part, dev0,
                                                     self.dev2dev_safe)
            return total

        # Flat (num_pixels, local_slices) shards; residual = image - init.
        # The pixel count is named rather than inferred: a shard that owns no
        # slices has no elements, and reshape cannot infer a row count from an
        # empty tensor whose column count is also zero.
        num_pixels = int(image_shape[0]) * int(image_shape[1])
        flat_image = _sharding.Shards(
            [t.reshape(num_pixels, t.shape[-1]).clone().contiguous()
             for t in init_sh.tensors], pl)
        flat_error = _sharding.Shards(
            [(a.reshape(num_pixels, a.shape[-1]) - b).contiguous()
             for a, b in zip(image_sh.tensors, flat_image.tensors)], pl)

        grad_hess = [maybe_compile(_qggmrf.qggmrf_gradient_and_hessian_at_indices,
                                   self.compile_enabled, instance_key=i)
                     for i in range(n)]
        idx_per_dev = [[torch.as_tensor(partition[k], dtype=torch.int64).to(d)
                        for d in devices] for k in range(partition.shape[0])]
        halos = {'left': [None] * n, 'right': [None] * n}

        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        num_iters = 0
        # ONE per-device thread pool for the whole sweep, as _vcd_recon keeps
        # for its loop: the two fan-outs per subset reuse it instead of
        # building and tearing down a private pool each time.  A caller that
        # already installed one (a reconstruction driving the denoiser) keeps
        # its own; n == 1 never needs one, since run_per_device short-circuits
        # to a direct call there.
        owns_pool = n > 1 and self._per_device_pool is None
        if owns_pool:
            self._per_device_pool = _sharding.device_pool(n)
        try:
            with torch.no_grad():
                for i in range(max_iters):
                    # Exchange the halos once per pass.
                    halos['left'], halos['right'] = _sharding.exchange_qggmrf_halos(
                        flat_image, self.dev2dev_safe)
                    ell1_accum = 0.0
                    alpha_accum = 0.0
                    for k in range(partition.shape[0]):
                        idx = idx_per_dev[k]

                        def terms_worker(j, dev):
                            grad, hess = grad_hess[j](
                                flat_image.tensors[j], image_shape, idx[j],
                                qggmrf_params, left_halo=halos['left'][j],
                                right_halo=halos['right'][j])
                            cur_error = flat_error.tensors[j][idx[j]]
                            forward_grad = -fm_constant * cur_error
                            delta = -((forward_grad + grad) / (1.0 + hess))
                            # 0-d tensors, not floats: they combine on the lead
                            # device below and are consumed back on the devices.
                            return (delta,
                                    torch.sum(grad * delta),
                                    torch.sum(hess * delta ** 2),
                                    fm_constant * torch.sum(cur_error * delta),
                                    fm_constant * torch.sum(delta * delta))

                        results = _sharding.run_per_device(
                            devices, terms_worker, executor=self._per_device_pool)
                        deltas = [r[0] for r in results]
                        prior_linear = combine_on_lead([r[1] for r in results])
                        prior_quadratic = combine_on_lead([r[2] for r in results])
                        forward_linear = combine_on_lead([r[3] for r in results])
                        forward_quadratic = combine_on_lead([r[4] for r in results])
                        alpha = ((forward_linear - prior_linear)
                                 / (forward_quadratic + prior_quadratic + _F32_EPS))
                        alpha = torch.clamp(alpha, _F32_EPS, 1.5)
                        # The step size is a scalar tensor on the lead device,
                        # so each shard needs its own copy to scale its delta.
                        alpha_per_device = (
                            [alpha] if n == 1 else
                            [_sharding.move_shard(alpha, dev, self.dev2dev_safe)
                             for dev in devices])

                        def apply_worker(j, dev):
                            step = alpha_per_device[j] * deltas[j]
                            flat_image.tensors[j].index_add_(0, idx[j], step)
                            flat_error.tensors[j].index_add_(0, idx[j], -step)
                            return torch.sum(torch.abs(step))

                        ell1_parts = _sharding.run_per_device(
                            devices, apply_worker, executor=self._per_device_pool)
                        ell1_accum = ell1_accum + combine_on_lead(ell1_parts)
                        alpha_accum = alpha_accum + alpha

                    # The three host reads per pass, all at this one
                    # synchronization point: the convergence test and the two
                    # logged histories need Python numbers.
                    # Chunked per shard, for the reason image_ell1 gives: it
                    # spares each device an image-shaped array of absolute
                    # values at the pass's one synchronization point.
                    image_l1 = float(combine_on_lead(
                        [image_ell1(t) for t in flat_image.tensors]))
                    # A zero image gives nan rather than raising, as on the
                    # single-device path.
                    nmae = (float(ell1_accum) / image_l1 if image_l1
                            else float('nan'))
                    nmae_update[i] = nmae
                    alpha_values[i] = float(alpha_accum) / partition.shape[0]
                    num_iters += 1
                    if verbose >= 1 and (i % 5) == 0:
                        self.logger.info('After iteration {} of a max of {}: Pct change={:.4f}'
                                         .format(i + first_iteration, max_iters, 100 * nmae))
                    if nmae < stop_thresh:
                        break
        finally:
            if owns_pool:
                self._per_device_pool.shutdown(wait=True)
                self._per_device_pool = None
        return flat_image, nmae_update, alpha_values, num_iters

    # ── denoising a stack of volumes at once ──────────────────────────────────
    def auto_set_regularization_params_from_stack(self, stack):
        """
        Set the regularization parameters (sigma_y, sigma_x, and sigma_prox)
        from a subsample of whole volumes of a stack, and return them as a
        dict.  The parameters change only when ``auto_regularize_flag`` is
        True, as in :meth:`auto_set_regularization_params`.

        About 20 volumes are chosen, evenly spaced along the volume axis:
        every volume when the stack holds at most 39, and every
        ``num_volumes // 20``-th volume otherwise, which is the rule
        :meth:`subsample_views` applies to views.  Only the chosen volumes
        cross to the host; a tensor is indexed on its own device first.  The
        chosen volumes are merged into one 3D array of shape
        ``(k * d0, d1, d2)`` and the statistics run on it with no further
        subsampling, so the neighbor differences along the first axis are
        between adjacent frames of one volume, except at the ``k - 1`` joins
        where the last frame of one chosen volume meets the first frame of the
        next.  :meth:`auto_set_regularization_params` reads a row subsample
        instead, whose row neighbors lie a stride apart, which is not what a
        stack of volumes needs.

        Args:
            stack (numpy or tensor): the volumes, with shape
                ``(num_volumes,) + image_shape``, on any device.

        Returns:
            dict: the values of ``sigma_y``, ``sigma_x``, and ``sigma_prox``.

        Raises:
            TypeError: if ``stack`` is in the divided device form.
        """
        _sharding.reject_shards('auto_set_regularization_params_from_stack',
                                stack=stack)
        names = list(_AUTO_REGULARIZATION_PARAM_NAMES)
        if self.get_params('auto_regularize_flag'):
            if not torch.is_tensor(stack):
                stack = np.asarray(stack)
            num_volumes = int(stack.shape[0])
            d0, d1, d2 = (int(n) for n in stack.shape[1:])
            # The volumes chosen are the ones subsample_views keeps when it is
            # handed the volume indices.  The same stride, applied to the stack
            # viewed as (num_volumes, d0 * d1, d2), picks those volumes on the
            # device that holds them and brings only them to the host.
            chosen = self.subsample_views(np.arange(num_volumes))
            step = int(chosen[1] - chosen[0]) if chosen.size > 1 else 1
            # The chosen volumes are cropped to a point budget before they
            # cross to the host.  The crop keeps the middle of the two
            # trailing axes and the whole of the first, so every neighbor
            # difference the estimate reads is still between adjacent voxels.
            rows, cols = _sample_tiles(d1, d2, int(chosen.size) * d0,
                                       _STATISTICS_POINT_BUDGET)
            blocks = []
            for row in rows:
                for col in cols:
                    tile = stack[:, :, row, col]
                    height, width = row.stop - row.start, col.stop - col.start
                    # Only this tile's chosen volumes cross to the host.
                    piece = _subsample_to_host(
                        tile.reshape(num_volumes, d0 * height, width), row_step=step)
                    blocks.append(piece.reshape(-1, height, width))
            # The tiles are stacked along the axis the volumes already join
            # on, so the joins between them are of the same kind.
            merged = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
            tiled = len(blocks) > 1 or merged.shape[1:] != (d1, d2)

            sino_indicator = self._get_sino_indicator(merged)
            if tiled:
                # The estimate reads each neighbor with a wrap, so entry 0 of
                # an axis is compared with the far end of that axis.  Over a
                # whole volume that is one plane in hundreds.  Over a tile it
                # is one column in the tile's width, which on a structured
                # image moves the estimate by tens of percent, so those
                # entries are dropped from the support.  The planes where one
                # tile meets the next are dropped for the same reason.
                sino_indicator[:, 0, :] = 0
                sino_indicator[:, :, 0] = 0
                sino_indicator[::merged.shape[0] // len(blocks)] = 0
            self.auto_set_sigma_y(merged, sino_indicator)
            recon_std = self._get_estimate_of_recon_std(merged, sino_indicator)
            if not np.isfinite(recon_std):
                recon_std = 0.0
            self.auto_set_sigma_x(recon_std)
            self.auto_set_sigma_prox(recon_std)
            # A stack dominated by background gives a sigma_x near zero, and
            # the sweep then divides by it and returns NaN for every volume.
            # The floor keeps sigma_x positive.
            sigma_x = self.get_params('sigma_x')
            if not np.isfinite(sigma_x) or sigma_x < _SIGMA_X_FLOOR:
                self.set_params(no_warning=True, sigma_x=_SIGMA_X_FLOOR)
        values = [float(v) for v in self.get_params(names)]
        return dict(zip(names, values))

    def auto_batch_size(self, volume_shape=None, init_supplied=False):
        """
        Return the number of volumes :meth:`denoise_stack` sweeps at once when
        its ``batch_size`` is None.

        On a CUDA device this is the largest batch whose sweep fits the
        device's free memory.  The sweep holds every image-shaped array of one
        volume's denoise plan once per volume in the batch, so the plan is
        priced for one volume by the memory ledger, scaled by the batch count,
        and compared with the free memory under the same margin the
        reconstruction preflight uses.  The free memory is read when this
        method is called.  On other devices no memory budget can be read, and
        the result is None, which :meth:`denoise_stack` takes to mean the
        whole stack.

        Args:
            volume_shape (tuple of int, optional): the shape of one volume.
                Defaults to the denoiser's image shape, which is the only
                shape this denoiser sweeps; any other shape raises.
            init_supplied (bool, optional): whether the sweep will be given an
                initial stack, which is one more image-shaped array per
                volume.  Defaults to False.

        Returns:
            int or None: volumes per batch, or None when the device reports no
            memory budget.

        Raises:
            ValueError: if ``volume_shape`` is not the denoiser's image shape.
            MemoryPreflightError: if a batch of one volume does not fit.
        """
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        if volume_shape is not None and \
                tuple(int(n) for n in volume_shape) != image_shape:
            raise ValueError(
                'auto_batch_size prices the sweep this denoiser runs, which is '
                f'over volumes of shape {image_shape}; got volume_shape '
                f'{tuple(volume_shape)}.  Build a QGGMRFDenoiser at that shape.')
        device = self.torch_device
        budget = _memory_ledger.device_budget_bytes(device)
        if budget is None:
            return None
        plan = _memory_ledger.plan_from_model(self, [device], workload='denoise')
        plan.init_recon_supplied = bool(init_supplied)
        ledger = _memory_ledger.estimate_peak_device_bytes(plan)
        batch = _memory_ledger.largest_denoise_batch(
            ledger, budget, margin=self.memory_preflight_margin)
        if batch < 1:
            need = _memory_ledger.denoise_batch_peak_bytes(ledger, 1)
            raise _memory_ledger.MemoryPreflightError(
                'a denoise sweep of one volume of shape {} is modeled at {:.2f} GB '
                'against {:.2f} GB free on {} (with a {:.0%} margin), so no batch '
                'fits.  Pass batch_size to denoise_stack to run anyway.'.format(
                    image_shape, need / 2 ** 30, budget / 2 ** 30, device,
                    self.memory_preflight_margin))
        return int(batch)

    def denoise_stack(self, stack, sigma_noise=None, init_stack=None,
                      max_iterations=15, stop_threshold_change_pct=0.2,
                      batch_size=None):
        """
        Denoise a stack of same-shaped volumes with shared parameters, each
        volume as :meth:`denoise` would denoise it alone.

        The volumes are independent.  The sweep carries a leading volume axis,
        each volume has its own line-search step size, and each volume has its
        own stopping test: a volume whose change falls below the threshold is
        frozen, its step is zero from then on, and the other volumes keep
        iterating.  The result and the per-volume iteration counts therefore
        equal those of calling :meth:`denoise` once per volume with the same
        parameters and the same pixel partition.

        The parameters are set once for the whole stack.  ``sigma_noise`` is
        shared by every volume, and ``sigma_y`` is kept equal to it.  When
        auto-regularization is on, the regularization parameters are set by
        :meth:`auto_set_regularization_params_from_stack` from a subsample of
        about 20 whole volumes, evenly spaced, with the neighbor differences
        taken between adjacent frames.  This differs from :meth:`denoise`,
        which reads a row subsample of a single image.  One pixel partition is
        drawn from the global numpy random generator and used by every volume,
        so a seeded call is reproducible.

        The sweep runs on the denoiser's device, in batches of ``batch_size``
        volumes.  The last batch is padded to the full size by repeating its
        last volume, so one compiled shape serves every batch of a call, and
        the padded results are discarded.  This method uses one device: a
        denoiser configured with more than one device raises.  No log file is
        written.

        Args:
            stack (numpy or tensor): the volumes to denoise, with shape
                (num_volumes,) + image_shape.
            sigma_noise (float, optional): noise std shared by every volume.
                None estimates it from the merged stack.
            init_stack (numpy or tensor, optional): initial image for each
                volume, with the shape of ``stack``.  Defaults to ``stack``.
            max_iterations (int, optional): maximum VCD iterations per volume.
            stop_threshold_change_pct (float, optional): a volume stops when
                100 * ||delta||_1 / ||volume||_1 drops below this.  0 runs
                every volume for exactly max_iterations.
            batch_size (int, optional): volumes swept at once.  None chooses
                the size with :meth:`auto_batch_size`, which is the whole stack
                on a device without a readable memory budget.

        Returns:
            (denoised_stack, info): the denoised volumes, numpy for numpy input
            and a tensor on the input's device for tensor input; and a dict
            with 'num_iterations' (one count per volume), 'nmae_pct' (one list
            per volume holding the percent change at each of its iterations),
            'regularization_params', and 'batch_size'.

        Raises:
            ValueError: if ``stack`` or ``init_stack`` has the wrong shape, if
                ``batch_size`` is below 1, or if the denoiser is configured
                with more than one device.
            TypeError: if ``stack`` or ``init_stack`` is in the divided device
                form.

        Example:
            >>> denoiser = mbirtorch.QGGMRFDenoiser(stack.shape[1:])
            >>> denoised, info = denoiser.denoise_stack(stack, sigma_noise=0.1)
        """
        _sharding.reject_shards('denoise_stack', stack=stack, init_stack=init_stack)
        if self.recon_placement.n_devices > 1:
            raise ValueError(
                'denoise_stack runs on one device, and this denoiser is '
                f'configured with {self.recon_placement.n_devices}.  Configure '
                'it with one device, or call denoise for the sharded sweep.')
        device = self.torch_device
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        stack_shape = tuple(int(n) for n in stack.shape)
        if len(stack_shape) != 4 or stack_shape[1:] != image_shape:
            raise ValueError(
                f'stack must have shape (num_volumes,) + {image_shape}; got '
                f'{stack_shape}.')
        if init_stack is not None and tuple(int(n) for n in init_stack.shape) != stack_shape:
            raise ValueError(
                f'init_stack must have the shape of stack, {stack_shape}; got '
                f'{tuple(init_stack.shape)}.')
        num_volumes = stack_shape[0]
        num_pixels = image_shape[0] * image_shape[1]
        num_slices = image_shape[2]

        # The output is allocated before the batch size is chosen, so that a
        # tensor output living on the sweep device is already counted in the
        # free-memory reading the automatic choice makes.
        stack_is_tensor = torch.is_tensor(stack)
        if stack_is_tensor:
            out = torch.empty(stack_shape, dtype=torch.float32, device=stack.device)
        else:
            out = np.empty(stack_shape, dtype=np.float32)
        if batch_size is None:
            batch_size = self.auto_batch_size(init_supplied=init_stack is not None)
        if batch_size is None or int(batch_size) > num_volumes:
            batch_size = num_volumes
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1; got {batch_size}.')

        # The noise estimate strides all three axes of the stack merged into
        # one 3D array, as denoise strides a single image.
        if sigma_noise is None:
            sigma_noise = self.estimate_image_noise_std(
                stack.reshape(-1, image_shape[1], image_shape[2]))
        # For the identity forward model sigma_y IS sigma_noise; kept equal
        # here for the pinned path too, as in denoise.
        self.set_params(no_warning=True, sigma_noise=sigma_noise,
                        sigma_y=sigma_noise)
        # The regularization parameters come from whole volumes, so the
        # neighbor differences they measure are between adjacent frames.
        regularization_params = self.auto_set_regularization_params_from_stack(stack)
        verbose = self.get_params('verbose')

        # One fixed partition over one volume's pixel grid, shared by every
        # volume.
        granularity = self.get_params('granularity')
        partition_sequence = self.get_params('partition_sequence')
        use_ror_mask = self.get_params('use_ror_mask')
        partition = vcd_utils.gen_set_of_pixel_partitions(
            image_shape, [granularity[partition_sequence[0]]],
            device=device, use_ror_mask=use_ror_mask)[0]

        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        qggmrf_params = (_qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts), sigma_x, p, q, T)
        stop_thresh = stop_threshold_change_pct / 100.0
        # One compiled instance per denoiser object, so that two denoisers
        # swept at the same time from different threads, on the same device or
        # on different ones, share no compiled state.
        subset_denoiser = maybe_compile(vcd_subset_denoiser_batched,
                                        self.compile_enabled, instance_key=id(self))

        def flat_on_device(block):
            """A block of volumes as a float32 (B, num_pixels, num_slices)
            tensor on the sweep device."""
            tensor = block if torch.is_tensor(block) else torch.as_tensor(block)
            tensor = tensor.to(device=device, dtype=torch.float32)
            return tensor.reshape(tensor.shape[0], num_pixels, num_slices)

        def padded(flat, pad):
            """The block with its last volume repeated pad more times."""
            if pad == 0:
                return flat
            return torch.cat([flat, flat[-1:].expand(pad, -1, -1)])

        num_iterations = np.zeros(num_volumes, dtype=int)
        nmae_pct = [[] for _ in range(num_volumes)]
        with torch.no_grad():
            for b0 in range(0, num_volumes, batch_size):
                b1 = min(b0 + batch_size, num_volumes)
                pad = batch_size - (b1 - b0)
                flat = padded(flat_on_device(stack[b0:b1]), pad)
                init_flat = (flat if init_stack is None
                             else padded(flat_on_device(init_stack[b0:b1]), pad))
                flat_image = init_flat.clone().contiguous()
                flat_error_image = (flat - flat_image).contiguous()
                counts, history = self._sweep_stack(
                    flat_image, flat_error_image, partition, fm_constant,
                    qggmrf_params, image_shape, max_iterations, stop_thresh,
                    subset_denoiser)
                real = b1 - b0
                result = flat_image[:real].reshape((real,) + image_shape)
                if stack_is_tensor:
                    out[b0:b1] = result.to(out.device)
                else:
                    out[b0:b1] = result.cpu().numpy()
                num_iterations[b0:b1] = counts[:real]
                nmae_pct[b0:b1] = history[:real]

        if verbose >= 1:
            self.logger.info(
                'Denoised {} volumes in batches of {}: {} to {} iterations per '
                'volume.'.format(num_volumes, batch_size,
                                 int(num_iterations.min()), int(num_iterations.max())))
        info = dict(num_iterations=num_iterations, nmae_pct=nmae_pct,
                    regularization_params=regularization_params,
                    batch_size=batch_size)
        return out, info

    @staticmethod
    def _sweep_stack(flat_image, flat_error_image, partition, fm_constant,
                     qggmrf_params, image_shape, max_iters, stop_thresh,
                     subset_denoiser):
        """Run the batched sweep in place on one batch of flat volumes.

        Each volume runs until its own change falls below ``stop_thresh`` or
        until ``max_iters``.  A volume that has stopped is frozen: it keeps
        its place in the batch with a step of zero.  The loop ends when no
        volume is active.

        Returns:
            (num_iterations, nmae_pct): the iteration count of each volume,
            and one list per volume of the percent change at each of its
            iterations.
        """
        num_vols = int(flat_image.shape[0])
        device = flat_image.device
        active_host = np.ones(num_vols, dtype=bool)
        active = torch.ones(num_vols, dtype=torch.bool, device=device)
        counts = np.zeros(num_vols, dtype=int)
        history = [[] for _ in range(num_vols)]
        for i in range(max_iters):
            ell1_accum = torch.zeros(num_vols, dtype=flat_image.dtype, device=device)
            for k in range(partition.shape[0]):
                flat_image, flat_error_image, ell1_subset, _alpha = subset_denoiser(
                    flat_image, flat_error_image, partition[k], fm_constant,
                    qggmrf_params, tuple(image_shape), active)
                ell1_accum = ell1_accum + ell1_subset
            # One host read per iteration: the stopping test needs Python
            # numbers.  The ratio is formed in float64 on the host, as
            # denoise forms it, and a zero volume gives nan rather than
            # raising, as there.
            stats = torch.stack([ell1_accum, stack_ell1(flat_image)])
            stats = stats.cpu().numpy().astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                nmae = stats[0] / stats[1]
            for j in np.flatnonzero(active_host):
                history[j].append(100.0 * float(nmae[j]))
                counts[j] = i + 1
            active_host &= ~(nmae < stop_thresh)
            if not active_host.any():
                break
            active = torch.tensor(active_host, device=device)
        return counts, history


def median_filter3d(x, max_block_gb=4.0, return_min_max=False):
    """
    Apply a 27-point (3x3x3) median filter to a 3-D array using replicated
    (edge) boundary conditions.  Optionally also return the min and max of
    each 27-point neighborhood.

    Args:
        x (ndarray or tensor): Input array.
        max_block_gb (float, optional): A rough upper bound on the amount of
            memory in GB to use for the filtering.  Defaults to 4.0.
        return_min_max (bool, optional): If True, the output is a tuple
            (median, min, max).

    Returns:
        ndarray or tensor (or tuple of 3): An array of the same shape and
        dtype as ``x`` containing the median-filtered result, numpy for
        numpy input and tensor for tensor input.

    Raises:
        TypeError: If ``x`` is in the divided device form.

    Note:
        The array is processed in blocks along axis 0 so that roughly
        ``max_block_gb`` of temporary data exists at once.  If axis 0 is
        short relative to another axis, swapping axis 0 with the long axis
        first may use less memory.

    Example:
        >>> import numpy as np
        >>> import mbirtorch
        >>> vol = np.arange(27.).reshape(3, 3, 3)
        >>> mbirtorch.median_filter3d(vol)
    """
    import torch.nn.functional as F
    from .tomography_model import _resolve_device

    # The filter works on one array on one device, and each output voxel needs
    # the 26 around it, so a slice-divided volume would need its neighboring
    # slices exchanged between devices.  It is refused rather than being taken
    # for numpy by the check just below, which fails on a torch dtype message
    # that says nothing about where the array actually is.
    _sharding.reject_shards('median_filter3d', x=x)
    was_numpy = not isinstance(x, torch.Tensor)
    if was_numpy:
        xt = torch.as_tensor(np.asarray(x), device=_resolve_device('auto'))
    else:
        xt = x
    d0, d1, d2 = xt.shape
    x_gb = xt.numel() * 4 / (1024**3)
    num_blocks = int(np.ceil(27 * x_gb / max_block_gb))
    block_size = max(d0 // max(num_blocks, 1), 1)

    # 1) Pad every dim by 1 for the edge‐replicated halo
    xp = F.pad(xt[None, None], (1, 1, 1, 1, 1, 1), mode='replicate')[0, 0]   # (d0+2, d1+2, d2+2)

    # 2) Pad d0 *further* up to a multiple of block_size, only at the end so fixed-size blocks tile it
    n_blocks = (d0 + block_size - 1) // block_size
    padded_Z = n_blocks * block_size
    pad_extra = padded_Z - d0
    if pad_extra > 0:
        xp = F.pad(xp[None, None], (0, 0, 0, 0, 0, pad_extra), mode='replicate')[0, 0]

    med_blocks, min_blocks, max_blocks = [], [], []
    with torch.no_grad():
        for i in range(n_blocks):
            z0 = i * block_size
            block = xp[z0:z0 + block_size + 2]

            # the 27‐roll → stack → median recipe on this small block
            patches = [
                torch.roll(block, shifts=(dz, dy, dx), dims=(0, 1, 2))
                for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            ]
            stacked = torch.stack(patches, dim=0)          # (27, blkZ+2, d1+2, d2+2)
            filtered = torch.median(stacked, dim=0).values
            med_blocks.append(filtered[1:-1, 1:-1, 1:-1])
            if return_min_max:
                min_blocks.append(torch.min(stacked, dim=0).values[1:-1, 1:-1, 1:-1])
                max_blocks.append(torch.max(stacked, dim=0).values[1:-1, 1:-1, 1:-1])

    def stitch(blocks):
        out = torch.cat(blocks, dim=0)[:d0]
        return out.cpu().numpy() if was_numpy else out

    if return_min_max:
        return stitch(med_blocks), stitch(min_blocks), stitch(max_blocks)
    return stitch(med_blocks)
