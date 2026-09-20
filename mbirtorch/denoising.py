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
import itertools

import numpy as np
import torch

from . import _memory_ledger, _sharding

from . import qggmrf as _qggmrf
from . import vcd_utils
from ._memory_ledger import image_ell1, stack_ell1
from ._utils import _AUTO_REGULARIZATION_PARAM_NAMES, Param, recon_param_names
from .projectors import maybe_compile
from .tomography_model import TomographyModel

# Each denoiser takes the next number at construction, and that number names
# its own compiled instances.
_instance_counter = itertools.count(1)

_F32_EPS = float(np.finfo(np.float32).eps)
# The stack statistics never set sigma_x below this value.
_SIGMA_X_FLOOR = 1e-6
# The whole-volume statistics read at most this many voxels.
_STATISTICS_POINT_BUDGET = 5_000_000

# The sampled statistics use this many tiles per axis, and a tile edge shorter
# than _MIN_TILE is not read.
_TILE_GRID = 2
_MIN_TILE = 4


def _sample_tiles(num_rows, num_cols, num_leading, point_budget):
    """Return tiles of two axes that together hold at most ``point_budget``
    voxels, counting ``num_leading`` entries of a third axis left whole.

    The tiles are contiguous, so a neighbor difference inside a tile is taken
    between adjacent voxels.  The tiles are spread across both axes so that
    the estimate sees the whole field of view.

    Returns:
        (list of slice, list of slice): the rows and the columns of the tiles.
        Their product is the tile grid.  Each list holds one whole-axis slice
        when the budget is not reached.
    """
    num_rows, num_cols, num_leading = int(num_rows), int(num_cols), int(num_leading)
    total = num_leading * num_rows * num_cols
    if total <= point_budget or total <= 0:
        return [slice(0, num_rows)], [slice(0, num_cols)]
    side = max(2, int((point_budget / num_leading) ** 0.5))
    count = max(1, min(_TILE_GRID, side // _MIN_TILE))
    edge = max(2, side // count)

    def spread(extent):
        """Return the tiles of one axis.  Each tile is centered on an equal
        share of the axis, so no tile lies against an edge."""
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
    prior_grad, prior_hess = _qggmrf.qggmrf_gradient_and_hessian_at_indices(
        flat_image, image_shape, pixel_indices, qggmrf_params)

    # The forward Hessian is all 1s for the qggmrf proximal map.
    cur_error_image = flat_error_image[pixel_indices]
    forward_grad = -fm_constant * cur_error_image
    forward_hess = 1

    delta_recon_at_indices = -((forward_grad + prior_grad)
                               / (forward_hess + prior_hess))

    # These two sums are delta^T grad Q(x_hat; x'=x_hat) and an upper bound
    # on the prior Hessian term.  Both are used to find the step size alpha.
    prior_linear = torch.sum(prior_grad * delta_recon_at_indices)
    prior_quadratic_approx = torch.sum(prior_hess * delta_recon_at_indices ** 2)

    # The forward model is the identity, so the sinogram domain direction is
    # the recon domain direction.
    delta_sinogram = delta_recon_at_indices
    forward_linear = fm_constant * torch.sum(cur_error_image * delta_sinogram)
    forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram)

    alpha_numerator = forward_linear - prior_linear
    alpha_denominator = forward_quadratic + prior_quadratic_approx + _F32_EPS
    alpha = alpha_numerator / alpha_denominator
    max_alpha = 1.5
    alpha = torch.clamp(alpha, _F32_EPS, max_alpha)

    delta_recon_at_indices = alpha * delta_recon_at_indices
    flat_image.index_add_(0, pixel_indices, delta_recon_at_indices)

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


def _is_stack(image):
    """True when the image is a stack of volumes rather than one volume.  The
    divided device form always holds one volume."""
    if isinstance(image, _sharding.Shards):
        return False
    return len(np.shape(image)) == 4


def _mask_key(use_ror_mask):
    """Return a comparable form of the mask setting.  A custom mask is an
    array, which does not compare as one value."""
    if use_ror_mask is True or use_ror_mask is False:
        return use_ror_mask
    return np.asarray(use_ror_mask).tobytes()


def _volume_shape(image):
    """Return the shape of a 3D volume given as a numpy array, a torch
    tensor, or a Shards.  A denoiser image is sharded on the last axis, which
    is the slice axis."""
    if isinstance(image, _sharding.Shards):
        first = image.tensors[0]
        num_slices = sum(int(t.shape[-1]) for t in image.tensors)
        return int(first.shape[0]), int(first.shape[1]), num_slices
    if torch.is_tensor(image):
        return tuple(int(n) for n in image.shape)
    return tuple(int(n) for n in np.asarray(image).shape)


def _subsample_to_host(image, row_step=1, col_step=1, slice_step=1):
    """Return ``numpy.asarray(image)[::row_step, ::col_step, ::slice_step]``
    for a 3D volume given as a numpy array, a tensor, or a Shards.  The
    stride is applied on the device that holds the data, so only the sampled
    elements are copied to the host.

    For sharded input the result is exact.  Shard k owns global slices
    ``[start_k, end_k)``, so its sampled local positions begin at offset
    ``(-start_k) % slice_step``.  Concatenating those blocks on the last axis
    reproduces the strided volume.
    """
    def block_to_host(tensor, slice_start):
        """Return one tensor's strided block as a numpy array.  The block is
        made dense on its own device before it is copied to the host."""
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

    Every call settles its own noise level, regularization parameters, and
    pixel partition, so a caller who denoises a different volume each time
    gets parameters fitted to that volume.  A caller who needs one fixed
    operator instead calls :meth:`initialize_denoiser` once and then passes
    ``do_initialization=False``, which reuses what that call settled.

    Args:
        image_shape (tuple of int): shape of the images to denoise
            (3-dimensional).  To denoise a 2D image, use shape (1, m, n).
        compile_mode (str, optional): 'auto' (default) compiles the
            computational kernels with torch.compile; 'off' runs without
            compilation.
    """

    # This name selects the speed floors in _widening_floors that set the
    # automatic device count, read in image voxels.  Sharded denoising was
    # slower at every size measured, so the automatic path uses one device
    # unless memory capacity requires more.
    _floor_family = 'denoiser'

    def __init__(self, image_shape, compile_mode='auto'):
        if len(image_shape) != 3:
            raise ValueError('image_shape must be 3-dimensional. Got image_shape={}. '
                             'To denoise a 2D image, use shape (1, m, n).'.format(image_shape))
        super().__init__(image_shape, compile_mode=compile_mode,
                         view_params_name='None', sigma_noise=None)
        # Setting the noise level must not clear the model's caches, so
        # sigma_noise is registered without the recompile flag.
        self.params['sigma_noise'] = Param(self.get_params('sigma_noise'), False)
        # This holds what initialize_denoiser settled: the pixel partition, the
        # regularization parameters, the noise level, and the signature they
        # were settled at.
        self.denoise_data = None
        # This records whether the cached partition came from the caller.
        self._partition_supplied = False
        # This names the compiled instances of this object.  A counter is used
        # rather than the object's address, because an address is handed out
        # again once the object at it is freed.
        self._instance_key = next(_instance_counter)
        self.set_params(use_ror_mask=False)
        self.set_params(sharpness=0)
        # A single fixed partition suffices for qggmrf denoising.
        self.set_params(granularity=[16], partition_sequence=[0])

    def _invalidate_device_caches(self):
        """Drop the denoiser's cache along with the caches of the base model,
        because the cached partition is a tensor on the model's device."""
        super()._invalidate_device_caches()
        self.denoise_data = None

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
        # For the identity forward model sigma_y is the noise level.
        sigma_y = self.get_params('sigma_noise')
        self.set_params(no_warning=True, sigma_y=sigma_y, auto_regularize_flag=True)

    def _check_lateral_truncation(self, sino_indicator):
        """Do nothing.  The denoiser's input is an ordinary image, and image
        content reaching the frame edge is normal."""
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
        """Return an estimate of the noise standard deviation.  Each voxel of
        the support is compared with its three backward neighbors, one per
        axis, and the mean of the per-voxel standard deviations is returned.
        A voxel on an edge takes its neighbor from the far side of that axis.
        """
        def views():
            """Yield the voxel array and its three backward neighbor arrays.
            Only one shifted copy exists at a time."""
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
        """Return a binary support indicator for the noisy image.  The
        threshold is a small fraction of the mean magnitude plus the noise
        floor."""
        if sigma_noise is None:
            sigma_noise = self.get_params('sigma_noise')
        percent_noise_floor = 5.0
        threshold = (0.01 * percent_noise_floor) * np.mean(np.fabs(noisy_image)) + sigma_noise
        threshold = min(threshold, np.amax(noisy_image))
        return np.int8(noisy_image >= threshold)

    def recon(self, *args, **kwargs):
        raise NotImplementedError('recon is not implemented for QGGMRFDenoiser.  '
                                  'Use `denoise` instead.')

    # ── the initialization cache ──────────────────────────────────────────────
    def _denoise_signature(self):
        """Return the settings the cache is valid for: the image shape, the
        two partition settings, the mask, and the device."""
        recon_shape, granularity, partition_sequence, use_ror_mask = self.get_params(
            ['recon_shape', 'granularity', 'partition_sequence', 'use_ror_mask'])
        return (tuple(int(n) for n in recon_shape), tuple(granularity),
                tuple(partition_sequence), _mask_key(use_ror_mask),
                str(self.torch_device))

    def _draw_partition(self, rng=None):
        """Return a pixel partition of the image at the number of subsets the
        granularity and the partition sequence name."""
        image_shape, granularity, partition_sequence, use_ror_mask = self.get_params(
            ['recon_shape', 'granularity', 'partition_sequence', 'use_ror_mask'])
        return vcd_utils.gen_set_of_pixel_partitions(
            image_shape, [granularity[partition_sequence[0]]],
            device=self.torch_device, use_ror_mask=use_ror_mask, rng=rng)[0]

    def _validated_partition(self, partition):
        """Return a supplied partition as a contiguous int64 tensor on the
        model's device, after checking that it partitions the pixels the mask
        allows.

        Two rows may hold the same index, as a drawn partition does when it
        pads its last rows, but one row may not hold an index twice.
        """
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        values = (partition.detach().cpu().numpy() if torch.is_tensor(partition)
                  else np.asarray(partition))
        if values.ndim != 2:
            raise ValueError('a denoiser partition has two dimensions, (subsets, '
                             f'indices); got shape {values.shape}.')
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError('a denoiser partition holds integer pixel indices; got '
                             f'dtype {values.dtype}.')
        num_pixels = image_shape[0] * image_shape[1]
        if values.size == 0:
            raise ValueError('a denoiser partition holds at least one pixel index; '
                             'got an empty array.')
        if values.min() < 0 or values.max() >= num_pixels:
            raise ValueError(
                f'a denoiser partition indexes the {image_shape[0]} by {image_shape[1]} '
                f'pixel grid, so every value lies in [0, {num_pixels}); got values from '
                f'{int(values.min())} to {int(values.max())}.')
        for row in range(values.shape[0]):
            if np.unique(values[row]).size != values.shape[1]:
                raise ValueError(
                    f'subset {row} of a denoiser partition holds an index twice.')
        allowed = np.asarray(vcd_utils.gen_full_indices(
            image_shape, use_ror_mask=self.get_params('use_ror_mask')))
        present = np.unique(values)
        if not np.array_equal(present, np.sort(allowed)):
            raise ValueError(
                'the subsets of a denoiser partition together cover exactly the pixels '
                f'the mask allows, which are {allowed.size} of {num_pixels}; they cover '
                f'{present.size}.')
        return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.int64,
                               device=self.torch_device)

    def _settle_regularization(self, image):
        """Return the regularization parameters of the sweep.  With
        auto-regularization on they are estimated from the image, which is
        then required; with it off the current values are read."""
        names = list(_AUTO_REGULARIZATION_PARAM_NAMES)
        if not self.get_params('auto_regularize_flag'):
            return dict(zip(names, [float(v) for v in self.get_params(names)]))
        if image is None:
            raise ValueError(
                'initialize_denoiser estimates the regularization parameters from an '
                'image, because auto_regularize_flag is on.  Pass image=, or set the '
                'parameters yourself and turn the flag off.')
        if _is_stack(image):
            # The parameters of a stack come from whole volumes, so the neighbor
            # differences they measure are between adjacent frames.
            return self.auto_set_regularization_params_from_stack(image)
        # auto_set_regularization_params starts by calling subsample_views, which
        # keeps every step_size-th row, so passing those rows gives it the same data.
        num_rows = _volume_shape(image)[0]
        sampled_rows = self.subsample_views(np.arange(num_rows))
        row_step = int(sampled_rows[1] - sampled_rows[0]) if sampled_rows.size > 1 else 1
        small_image = _subsample_to_host(image, row_step=row_step)
        verbose = self.get_params('verbose')
        self.set_params(no_warning=True, verbose=0)
        params = self.auto_set_regularization_params(small_image)
        self.set_params(no_warning=True, verbose=verbose)
        return params

    def initialize_denoiser(self, image=None, sigma_noise=None, partition=None,
                            rng=None):
        """
        Settle everything a sweep needs that depends on the image or on a
        random draw, and store it in ``denoise_data`` for later calls.

        :meth:`denoise` and :meth:`denoise_stack` call this themselves when
        they are asked to initialize or find no cache.  Call it directly to
        make one denoiser a fixed operator: initialize once, then pass
        ``do_initialization=False`` on every call, and every call then sweeps
        with the same partition and the same parameters.

        The device layout is settled first, so that nothing settled later
        drops what this stores.  A change of the device layout clears the
        cache.  So does a change of ``recon_shape``, and a call that finds
        ``granularity``, ``partition_sequence``, or ``use_ror_mask`` moved
        draws the partition again.

        Args:
            image (numpy or tensor or Shards, optional): one volume, or a
                stack of volumes with a leading volume axis.  The noise level
                and the regularization parameters are estimated from it.
            sigma_noise (float, optional): the noise level.  None estimates it
                from ``image``, and with no image the model's current value is
                kept.  ``sigma_y`` is set equal to it.
            partition (array or tensor, optional): the pixel partition every
                sweep uses, of shape (subsets, indices).  None draws one.
            rng (numpy.random.Generator, optional): the generator the
                partition is drawn from.  Defaults to None, the global
                np.random state.

        Returns:
            dict: the cache, with entries 'partition', 'regularization_params',
            'sigma_noise', and 'signature'.

        Raises:
            ValueError: if no noise level can be found, if
                ``auto_regularize_flag`` is on and no image is given, or if a
                supplied partition does not partition the pixels the mask
                allows.

        Example:
            >>> denoiser.initialize_denoiser(image=volume, sigma_noise=0.1)
            >>> out, d = denoiser.denoise(volume, do_initialization=False)
        """
        # The device layout is settled before anything is placed, so that the
        # partition below lands on the device the sweep will use.
        self._apply_device_policy(workload='denoise')
        image_shape = tuple(int(n) for n in self.get_params('recon_shape'))
        if sigma_noise is None and image is not None:
            # This estimate strides all three axes itself, so it takes the
            # image in whatever form the caller supplied.  A stack is read as
            # one volume, with its volumes joined along the row axis.
            volume = (image.reshape(-1, image_shape[1], image_shape[2])
                      if _is_stack(image) else image)
            sigma_noise = self.estimate_image_noise_std(volume)
        if sigma_noise is None:
            sigma_noise = self.get_params('sigma_noise')
        if sigma_noise is None:
            raise ValueError(
                'initialize_denoiser needs a noise level.  Pass sigma_noise, pass an '
                'image to estimate it from, or set sigma_noise on the model.')
        # For the identity forward model sigma_y is sigma_noise, so the two
        # are kept equal even when auto-regularization is off.
        self.set_params(no_warning=True, sigma_noise=float(sigma_noise),
                        sigma_y=float(sigma_noise))
        regularization_params = self._settle_regularization(image)
        if partition is None:
            partition_tensor = self._draw_partition(rng=rng)
            self._partition_supplied = False
        else:
            partition_tensor = self._validated_partition(partition)
            self._partition_supplied = True
        self.denoise_data = {'partition': partition_tensor,
                             'regularization_params': regularization_params,
                             'sigma_noise': float(sigma_noise),
                             'signature': self._denoise_signature()}
        return self.denoise_data

    def _denoise_setup(self, image, sigma_noise, do_initialization):
        """Return the pixel partition and the regularization parameters of the
        sweep about to run.

        The cache is built when this call asks for it or when there is none.
        Otherwise the cache is reused: a noise level given to the call
        replaces the cached one, the statistics are not computed again, and a
        cache settled at other partition settings has its partition drawn
        again.
        """
        if do_initialization or self.denoise_data is None:
            if not do_initialization and self._partition_supplied:
                raise ValueError(
                    'the partition given to initialize_denoiser was dropped by a change '
                    'of the device layout, so this call has none to reuse.  Call '
                    'initialize_denoiser again with the partition.')
            data = self.initialize_denoiser(image=image, sigma_noise=sigma_noise)
            return data['partition'], dict(data['regularization_params'])

        data = self.denoise_data
        # fm_constant is one scalar, so a new noise level costs nothing to use
        # and the parameters reported carry the value used.
        level = data['sigma_noise'] if sigma_noise is None else float(sigma_noise)
        self.set_params(no_warning=True, sigma_noise=level, sigma_y=level)
        data['sigma_noise'] = level
        signature = self._denoise_signature()
        if signature != data['signature']:
            if self._partition_supplied:
                raise ValueError(
                    'the partition given to initialize_denoiser was made for other '
                    'settings than this call uses.  Call initialize_denoiser again with '
                    'a partition for the current settings.')
            data['partition'] = self._draw_partition()
            data['signature'] = signature
            self.logger.info('The pixel partition was drawn again, because the '
                             'partition settings changed.')
        return data['partition'], dict(data['regularization_params'], sigma_y=level)

    def denoise(self, image, sigma_noise=None, use_ror_mask=None, init_image=None,
                max_iterations=15, stop_threshold_change_pct=0.2, first_iteration=0,
                logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                output_sharded=False, do_initialization=True):
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
            use_ror_mask: restrict denoising to a masked region (None default,
                which keeps the model's current setting; False for no mask,
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
            do_initialization (bool, optional): If True, settle the noise
                level, the regularization parameters, and the pixel partition
                for this call through :meth:`initialize_denoiser`.  False
                reuses what a previous call or a direct call to
                :meth:`initialize_denoiser` settled, so that every call is the
                same operator.  Defaults to True.

        Returns:
            (denoised_image, denoiser_dict): the denoised volume, and a dict
            with entries 'recon_params', 'recon_log', 'notes', and
            'model_params' (as in :meth:`TomographyModel.get_recon_dict`).

        Example:
            >>> denoiser = mbirtorch.QGGMRFDenoiser(noisy_image.shape)
            >>> denoised_image, d = denoiser.denoise(noisy_image, sigma_noise=0.1)
        """
        self._log_run_header(first_iteration, logfile_path, print_logs)
        # The device layout is settled before the image is placed.  The plan
        # is priced for a denoise, which has no projectors.
        self._apply_device_policy(workload='denoise', init_recon=init_image)
        self._log_device_report()

        if use_ror_mask is not None:
            self.set_params(no_warning=True, use_ror_mask=use_ror_mask)
        self.logger.info('Initializing QGGMRFDenoiser')
        # The sweep uses one fixed partition.  The subsets run in order and
        # are not reshuffled between iterations.
        partition, regularization_params = self._denoise_setup(
            image, sigma_noise, do_initialization)

        image_shape, granularity = self.get_params(['recon_shape', 'granularity'])
        partition_sequence = self.get_params('partition_sequence')
        verbose = self.get_params('verbose')

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
            # On one device the whole sweep runs through one compiled
            # in-place update.
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

                    # A zero image gives nan rather than raising
                    # ZeroDivisionError.
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
        # The last log line is written, so the file is closed.  A later call
        # reopens it.
        self.close_log_file()
        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        denoiser_dict = self.get_recon_dict(recon_params, notes=notes)
        return (denoised if output_sharded else self._gather_recon(denoised)), denoiser_dict

    def _denoise_sharded(self, image_sh, init_sh, partition, fm_constant,
                         qggmrf_params, image_shape, max_iters, stop_thresh,
                         first_iteration, verbose):
        """Run the denoising sweep across devices on slice-sharded state.

        The qGGMRF halos are staged once per pass.  Each device computes its
        own shard's prior and forward terms.  The four line-search sums are
        combined on the lead device into one step size, using the formula of
        :func:`vcd_subset_denoiser`.  The step size stays a tensor on the
        device, so no host synchronization is forced per subset.  One host
        synchronization per pass reads the convergence test and the logged
        history.

        Returns (flat_image shards, nmae history, alpha history, num_iters).
        """
        devices = image_sh.placement.devices
        n = len(devices)
        pl = image_sh.placement
        dev0 = devices[0]

        def combine_on_lead(parts):
            """Sum the per-shard scalar tensors on the lead device."""
            total = parts[0]
            for part in parts[1:]:
                total = total + _sharding.move_shard(part, dev0,
                                                     self.dev2dev_safe)
            return total

        # The state is held as flat (num_pixels, local_slices) shards.  The pixel
        # count is explicit because reshape cannot infer it for a shard with no slices.
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
        # One thread pool serves the whole sweep, and a caller that already
        # installed a pool keeps its own.
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
                            # The four sums are scalar tensors, not floats, and
                            # are combined on the lead device.
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
                        # Each shard needs its own copy of the step size to
                        # scale its delta.
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

                    # The convergence test needs Python numbers, so this is the one
                    # host synchronization point of the pass.  A zero image gives nan.
                    image_l1 = float(combine_on_lead(
                        [image_ell1(t) for t in flat_image.tensors]))
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
            # The volumes chosen are the ones subsample_views keeps when it
            # is handed the volume indices.
            chosen = self.subsample_views(np.arange(num_volumes))
            step = int(chosen[1] - chosen[0]) if chosen.size > 1 else 1
            # The chosen volumes are cropped to a point budget before they cross to
            # the host.  The crop keeps the whole first axis.
            rows, cols = _sample_tiles(d1, d2, int(chosen.size) * d0,
                                       _STATISTICS_POINT_BUDGET)
            blocks = []
            for row in rows:
                for col in cols:
                    tile = stack[:, :, row, col]
                    height, width = row.stop - row.start, col.stop - col.start
                    piece = _subsample_to_host(
                        tile.reshape(num_volumes, d0 * height, width), row_step=step)
                    blocks.append(piece.reshape(-1, height, width))
            # The tiles are stacked along the axis on which the volumes
            # already join.
            merged = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
            tiled = len(blocks) > 1 or merged.shape[1:] != (d1, d2)

            sino_indicator = self._get_sino_indicator(merged)
            if tiled:
                # The estimate reads each neighbor with a wrap, so entry 0 of an
                # axis is compared with the far end of that axis.  Those entries
                # are dropped from the support, and so are the planes where one
                # tile meets the next.
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
            sigma_x = self.get_params('sigma_x')
            if not np.isfinite(sigma_x) or sigma_x < _SIGMA_X_FLOOR:
                self.set_params(no_warning=True, sigma_x=_SIGMA_X_FLOOR)
        values = [float(v) for v in self.get_params(names)]
        return dict(zip(names, values))

    def auto_batch_size(self, volume_shape=None):
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
        whole stack.  A budget can be read on a CUDA device and on no other,
        so on the CPU and on an Apple GPU the whole stack is swept at once.

        The count does not depend on whether an initial stack is given.  The
        sweep holds two arrays per volume either way, the working image and
        the residual.  The image starts from the initial stack when there is
        one and from the input when there is not, each taken over when the
        sweep owns it and copied otherwise, and the residual is formed in the
        input's own buffer when the sweep owns that and in a new array when
        it does not; see ``overwrite_input`` on :meth:`denoise_stack`.

        Args:
            volume_shape (tuple of int, optional): the shape of one volume.
                Defaults to the denoiser's image shape, which is the only
                shape this denoiser sweeps; any other shape raises.

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
                      batch_size=None, overwrite_input=False,
                      do_initialization=True):
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
        used by every volume: it is drawn from the global numpy random
        generator, so a seeded call is reproducible, and
        :meth:`initialize_denoiser` settles one that every later call reuses.

        The sweep runs on the denoiser's device, in batches of ``batch_size``
        volumes.  The last batch is padded to the full size by repeating its
        last volume, so one compiled shape serves every batch of a call, and
        the padded results are discarded.  This method uses one device: a
        denoiser configured with more than one device raises.  No log file is
        written.

        The sweep holds two arrays per volume, the working image and the
        residual.  A stack swept as one batch is returned as that working
        image, with no separate output array; several batches write into
        one.  By default the caller's tensors are never written: a tensor
        already on the sweep device in float32 is cloned into the working
        image, so the call holds the input beside the two.  With
        ``overwrite_input`` the sweep takes such a tensor over instead, which
        saves one array per volume; the caller must not read it afterward.

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
            overwrite_input (bool, optional): let the sweep write ``stack``
                and ``init_stack`` in place when they are float32 tensors
                already on the sweep device, rather than clone them.  A numpy
                array is never written.  Defaults to False.
            do_initialization (bool, optional): If True, settle the noise
                level, the regularization parameters, and the pixel partition
                for this call through :meth:`initialize_denoiser`.  False
                reuses what a previous call or a direct call to
                :meth:`initialize_denoiser` settled, so that every call is the
                same operator.  Defaults to True.

        Returns:
            (denoised_stack, info): the denoised volumes, numpy for numpy input
            and a tensor on the input's device for tensor input, which shares
            the input's storage when ``overwrite_input`` let the sweep write
            it; and a dict
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
        if init_stack is stack:
            # The default start is the input.  Treating the two as one keeps the
            # in-place forms below from using one buffer as both image and residual.
            init_stack = None
        num_volumes = stack_shape[0]
        num_pixels = image_shape[0] * image_shape[1]
        num_slices = image_shape[2]

        stack_is_tensor = torch.is_tensor(stack)
        out = None
        if batch_size is None:
            if stack_is_tensor:
                # The output is allocated before auto_batch_size reads the free
                # memory, so that an output on the sweep device is counted.
                out = torch.empty(stack_shape, dtype=torch.float32, device=stack.device)
            batch_size = self.auto_batch_size()
        if batch_size is None or int(batch_size) > num_volumes:
            batch_size = num_volumes
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1; got {batch_size}.')
        # One batch returns the sweep's working image and allocates no output
        # array.  Several batches write into one output on the input's device.
        single = batch_size == num_volumes
        if single:
            out = None
        elif out is None:
            out = (torch.empty(stack_shape, dtype=torch.float32, device=stack.device)
                   if stack_is_tensor else np.empty(stack_shape, dtype=np.float32))

        # Every volume shares one fixed partition of one volume's pixel grid.
        partition, regularization_params = self._denoise_setup(
            stack, sigma_noise, do_initialization)
        # The sweep device is read after the layout is settled, so that the
        # volumes land where the partition did.
        device = self.torch_device
        verbose = self.get_params('verbose')

        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        qggmrf_params = (_qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts), sigma_x, p, q, T)
        stop_thresh = stop_threshold_change_pct / 100.0
        # Each denoiser object gets its own compiled instance, so that two
        # denoisers swept at the same time share no compiled state.  The key
        # is the object's own number, because an address is handed out again
        # once the object at it is freed.
        subset_denoiser = maybe_compile(vcd_subset_denoiser_batched,
                                        self.compile_enabled,
                                        instance_key=self._instance_key)

        def flat_on_device(block):
            """Return a block of volumes as a float32 (B, num_pixels,
            num_slices) tensor on the sweep device, together with a flag
            saying whether the sweep may write it.

            The sweep owns any block it had to copy.  A caller's tensor that
            already sits on the sweep device in float32 and reshapes as a
            view stays the caller's, and the sweep may write it only under
            ``overwrite_input``.  A numpy array is never written.
            """
            tensor = block if torch.is_tensor(block) else torch.as_tensor(block)
            moved = tensor.to(device=device, dtype=torch.float32)
            flat = moved.reshape(moved.shape[0], num_pixels, num_slices)
            owned = (moved is not tensor or flat.data_ptr() != moved.data_ptr()
                     or (overwrite_input and torch.is_tensor(block)))
            return flat, owned

        def padded(flat, pad):
            """Return the block with its last volume repeated pad more
            times."""
            if pad == 0:
                return flat
            return torch.cat([flat, flat[-1:].expand(pad, -1, -1)])

        num_iterations = np.zeros(num_volumes, dtype=int)
        nmae_pct = [[] for _ in range(num_volumes)]
        with torch.no_grad():
            for b0 in range(0, num_volumes, batch_size):
                b1 = min(b0 + batch_size, num_volumes)
                pad = batch_size - (b1 - b0)
                flat, owned = flat_on_device(stack[b0:b1])
                if pad:
                    flat, owned = padded(flat, pad), True
                if init_stack is None:
                    # The sweep writes its image in place, so it uses the input
                    # only when it owns it.  The residual is then zero.
                    flat_image = flat if owned else flat.clone()
                    flat_error_image = torch.zeros_like(flat_image)
                else:
                    init_flat, init_owned = flat_on_device(init_stack[b0:b1])
                    if pad:
                        init_flat, init_owned = padded(init_flat, pad), True
                    flat_image = init_flat if init_owned else init_flat.clone()
                    # The residual is formed in the input's own buffer when
                    # the sweep owns that buffer, and in a new array otherwise.
                    flat_error_image = (flat.sub_(flat_image) if owned
                                        else flat - flat_image)
                flat_image = flat_image.contiguous()
                flat_error_image = flat_error_image.contiguous()
                # The residual carries the data term, so the input is not
                # read again.
                del flat
                counts, history = self._sweep_stack(
                    flat_image, flat_error_image, partition, fm_constant,
                    qggmrf_params, image_shape, max_iterations, stop_thresh,
                    subset_denoiser)
                real = b1 - b0
                result = flat_image[:real].reshape((real,) + image_shape)
                if single:
                    out = result.to(stack.device) if stack_is_tensor else result.cpu().numpy()
                elif stack_is_tensor:
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
        until ``max_iters``.  A volume that has stopped keeps its place in
        the batch and takes a step of zero.  The loop ends when no volume is
        active.

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
            # The stopping test needs Python numbers, so there is one host read per
            # iteration.  The ratio is formed in float64, and a zero volume gives nan.
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

    # Each output voxel needs the 26 voxels around it, so the slice-divided form
    # is refused rather than exchanging neighboring slices between devices.
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

    # Every axis is padded by 1 to give the edge replicated boundary.
    xp = F.pad(xt[None, None], (1, 1, 1, 1, 1, 1), mode='replicate')[0, 0]

    # Axis 0 is padded further, at the end only, up to a multiple of
    # block_size, so that fixed-size blocks tile it.
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

            patches = [
                torch.roll(block, shifts=(dz, dy, dx), dims=(0, 1, 2))
                for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            ]
            stacked = torch.stack(patches, dim=0)
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
