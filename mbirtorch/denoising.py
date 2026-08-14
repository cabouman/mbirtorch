"""QGGMRFDenoiser, ported from mbirjax.denoising.

The denoiser uses the recon framework to implement a qGGMRF proximal-map
denoiser: the forward model is the IDENTITY (the residual image plays the role
of the error sinogram), so no projectors exist and the VCD subset update
reduces to the closed-form below.  The mbirjax jit/fori_loop machinery is
replaced by a plain python loop over the (sequential, unshuffled) subsets of
ONE fixed partition -- the mbirjax denoiser iterates subsets in order, unlike
vcd_recon's per-iteration shuffle, so a seeded partition makes the
whole sweep deterministic.

Two paths, as in mbirjax: on one device the whole sweep runs through the
compiled in-place update below; across several devices the image is
slice-sharded and the sweep runs shard by shard, with the qGGMRF halos
staged once per pass and the four line-search sums combined on the lead
device into one step size.  Both paths keep the line search on device, so
neither forces a host synchronization per subset.
"""

import datetime

import numpy as np
import torch

from . import _sharding

from . import qggmrf as _qggmrf
from . import vcd_utils
from ._utils import recon_param_names
from .projectors import maybe_compile
from .tomography_model import TomographyModel

_F32_EPS = float(np.finfo(np.float32).eps)


def vcd_subset_denoiser(flat_image, flat_error_image, pixel_indices,
                        fm_constant, qggmrf_params, image_shape):
    """One VCD subset update for the identity forward model (the analog of
    vcd_subset_updater; verbatim mbirjax math).  Mutates both state tensors in
    place and returns (flat_image, flat_error_image, ell1, alpha)."""
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


class QGGMRFDenoiser(TomographyModel):
    """
    The QGGMRFDenoiser uses the recon framework to implement a qggmrf proximal
    map denoiser.  The primary interface is through :meth:`denoise`.

    With default settings, and with X a clean image and W equal to AWGN of
    standard deviation sigma_noise, the result of :meth:`denoise` applied to
    X + W is the MAP estimate of the denoised image using the qGGMRF prior.

    The denoiser runs on one device by default.  Multi-device denoising is
    explicit: call ``configure_devices`` first, and the image is then divided
    across the devices by slice.  The automatic device count that ``recon``
    uses does not apply here.

    Args:
        image_shape (tuple of int): shape of the images to denoise
            (3-dimensional).  To denoise a 2D image, use shape (1, m, n).
        compile_mode (str, optional): 'auto' (default) compiles the
            computational kernels with torch.compile; 'off' runs without
            compilation.
    """

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
        support-indicator + neighbor-difference std, as in mbirjax).
        """
        image = np.asarray(image)
        num_pts_to_use = np.minimum(5_000_000, image.size)
        stride = round((image.size / num_pts_to_use) ** (1 / 3))
        small_image = image[::stride, ::stride, ::stride]

        support_indicator = self._get_sino_indicator(small_image, sigma_noise=0.0)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
        support_indicator = self._get_sino_indicator(small_image, sigma_noise=sigma_noise)
        sigma_noise = self._get_estimate_of_recon_std(small_image, support_indicator)
        return sigma_noise

    def _get_estimate_of_recon_std(self, noisy_image, support_indicator):
        """Neighbor-difference std over the support (the denoiser's own recon-std
        estimate, replacing the projection-path-length formula)."""
        inds = np.where(support_indicator)
        vals = np.stack([noisy_image[inds[0], inds[1], inds[2]],
                         noisy_image[inds[0] - 1, inds[1], inds[2]],
                         noisy_image[inds[0], inds[1] - 1, inds[2]],
                         noisy_image[inds[0], inds[1], inds[2] - 1]], axis=0)
        return np.mean(np.std(vals, axis=0))

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

        Args:
            image (numpy or tensor): the 3D volume to be denoised.
            sigma_noise (float, optional): estimated noise std in the image.
                If None, estimated from the image.
            use_ror_mask: restrict denoising to a masked region (False default;
                True for the inscribed ellipse, or a custom 2D mask).
            init_image (numpy or tensor, optional): initial image for the
                minimization.  Defaults to ``image``.
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
        self._log_device_report()   # the layout is explicit or the default; final either way

        # The noise and regularization estimates below index the image on the
        # host; a sharded input supplies a host copy for them (statistics
        # only -- the sweep itself consumes the sharded form).
        host_image = image
        if isinstance(image, _sharding.Shards):
            from .utilities import _to_host
            host_image = _to_host(image)

        self.set_params(no_warning=True, use_ror_mask=use_ror_mask)
        if sigma_noise is None:
            sigma_noise = self.estimate_image_noise_std(host_image)
        self.set_params(no_warning=True, sigma_noise=sigma_noise)
        self.logger.info('Initializing QGGMRFDenoiser')

        # Auto-regularization with the background-estimation warning suppressed.
        verbose = self.get_params('verbose')
        self.set_params(no_warning=True, verbose=0)
        regularization_params = self.auto_set_regularization_params(np.asarray(host_image))
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

                    nmae = float(ell1_accum) / float(torch.sum(torch.abs(flat_image)))
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
        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        denoiser_dict = self.get_recon_dict(recon_params, notes=notes)
        return (denoised if output_sharded else self._gather_recon(denoised)), denoiser_dict

    def _denoise_sharded(self, image_sh, init_sh, partition, fm_constant,
                         qggmrf_params, image_shape, max_iters, stop_thresh,
                         first_iteration, verbose):
        """Run the denoising sweep across devices on slice-sharded state.

        Mirrors vcd_recon's sharded path: the qGGMRF halos are staged once
        per pass, each device computes its shard's prior and identity-forward
        terms, and the four line-search sums combine ON THE LEAD DEVICE into
        one step size (the same formula as vcd_subset_denoiser).

        The line search stays on device for the reason vcd_recon states at
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
        flat_image = _sharding.Shards(
            [t.reshape(-1, t.shape[-1]).clone().contiguous()
             for t in init_sh.tensors], pl)
        flat_error = _sharding.Shards(
            [(a.reshape(-1, a.shape[-1]) - b).contiguous()
             for a, b in zip(image_sh.tensors, flat_image.tensors)], pl)

        interface_masks = self._qggmrf_interface_masks()
        grad_hess = [maybe_compile(_qggmrf.qggmrf_gradient_and_hessian_at_indices,
                                   self.compile_enabled, instance_key=i)
                     for i in range(n)]
        idx_per_dev = [[torch.as_tensor(partition[k], dtype=torch.int64).to(d)
                        for d in devices] for k in range(partition.shape[0])]
        halos = {'left': [None] * n, 'right': [None] * n}

        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        num_iters = 0
        # ONE per-device thread pool for the whole sweep, as vcd_recon keeps
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
                    # Halos once per pass, as in mbirjax's sharded denoiser.
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
                                right_halo=halos['right'][j],
                                interface_mask=(None if interface_masks is None
                                                else interface_masks[j]))
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
                    image_l1 = combine_on_lead([torch.sum(torch.abs(t))
                                                for t in flat_image.tensors])
                    nmae = float(ell1_accum) / float(image_l1)
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
