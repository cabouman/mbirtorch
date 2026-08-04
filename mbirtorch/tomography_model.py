"""TomographyModel: the single-device VCD engine, FBP, and projector wrappers.

Ported from mbirjax.tomography_model, Phase 1 scope (port_plan.md section 5):
the VCD engine (numpy partitions, subset updater, on-device line search,
positivity), FBP via torch.fft, the auto-regularization chain, and the public
numpy-at-the-boundary API.  Deliberately not ported in Phase 1: sharding and
placements, the tile policy, prox_map/denoiser, checkpoint resume, save/load,
and the memory-management machinery jax needed (donation and .delete() become
plain in-place ops here).

Value-parity intent: every formula follows the mbirjax source read 2026-08-04,
in the same order of operations, so a seeded run matches a seeded mbirjax run
iteration for iteration (the Phase 1 convergence-parity gate).
"""

import logging
import math
import warnings

import numpy as np
import torch

from . import qggmrf as _qggmrf
from . import tomography_utils, vcd_utils
from ._utils import _AUTO_REGULARIZATION_PARAM_NAMES, recon_param_names
from .parameter_handler import ParameterHandler
from .projectors import Projectors

_F32_EPS = float(np.finfo(np.float32).eps)


def _resolve_device(device):
    """'auto' -> cuda if available, else mps, else cpu; else the given device."""
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TomographyModel(ParameterHandler):
    """Base class: geometry classes supply the per-view-batch fan geometry
    (``compute_hfan_data_batched``), the psf radius, and the auto recon
    geometry; this class owns the projction wrappers and the VCD engine."""

    def __init__(self, sinogram_shape, device='auto', view_batch_size=64, **kwargs):
        super().__init__()
        self.torch_device = _resolve_device(device)
        self.view_batch_size = view_batch_size
        self.logger = logging.getLogger('mbirtorch')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Insert the geometry's own parameters (e.g. angles, view_params_name)
        # as new Param entries, then record the sinogram shape.
        from ._utils import Param
        for key, val in kwargs.items():
            self.params[key] = Param(val, True)
        self.set_params(no_compile=True, no_warning=True,
                        sinogram_shape=tuple(int(s) for s in sinogram_shape))

        # Geometry-derived defaults (recon_shape, delta_voxel), then the
        # projectors, then a validity check -- the mbirjax construction order.
        self.auto_set_recon_geometry(no_compile=True, no_warning=True)
        self.create_projectors()
        self.verify_valid_params()

    # ── hooks for geometry subclasses ─────────────────────────────────────────
    def create_projectors(self):
        self.projector_functions = Projectors(self)

    def get_magnification(self):
        raise NotImplementedError

    def get_psf_radius(self):
        raise NotImplementedError

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        raise NotImplementedError

    def compute_hfan_data_batched(self, pixel_indices, view_params_batch):
        raise NotImplementedError

    def direct_recon(self, sinogram, filter_name=None, output_sharded=False):
        raise NotImplementedError

    # ── projection wrappers (numpy at the public boundary) ────────────────────
    def sparse_forward_project(self, voxel_values, pixel_indices):
        """Cylinders at ``pixel_indices`` -> full sinogram (tensor)."""
        return self.projector_functions.sparse_forward_project(voxel_values, pixel_indices)

    def sparse_back_project(self, sinogram, pixel_indices, coeff_power=1):
        """Sinogram -> cylinders at ``pixel_indices`` (tensor)."""
        return self.projector_functions.sparse_back_project(sinogram, pixel_indices,
                                                            coeff_power=coeff_power)

    def _full_indices(self):
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        return vcd_utils.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)

    def forward_project(self, recon, output_sharded=False):
        """Full-volume forward projection.

        ``output_sharded`` keeps the device tensor (the name matches the
        mbirjax API; here it simply means "skip the numpy exit").
        """
        recon_shape = self.get_params('recon_shape')
        recon = torch.as_tensor(recon, dtype=torch.float32, device=self.torch_device)
        indices = torch.as_tensor(self._full_indices(), dtype=torch.int64,
                                  device=self.torch_device)
        voxel_values = recon.reshape(-1, recon.shape[-1])[indices]
        sinogram = self.sparse_forward_project(voxel_values, indices)
        return sinogram if output_sharded else sinogram.cpu().numpy()

    def back_project(self, sinogram, output_sharded=False):
        """Full-volume back projection (zeros outside the ROR mask)."""
        recon_shape = self.get_params('recon_shape')
        sinogram = torch.as_tensor(sinogram, dtype=torch.float32, device=self.torch_device)
        indices = torch.as_tensor(self._full_indices(), dtype=torch.int64,
                                  device=self.torch_device)
        cylinders = self.sparse_back_project(sinogram, indices)
        recon = torch.zeros((recon_shape[0] * recon_shape[1], cylinders.shape[-1]),
                            dtype=torch.float32, device=self.torch_device)
        recon[indices] = cylinders
        recon = recon.reshape(tuple(recon_shape[:2]) + (cylinders.shape[-1],))
        return recon if output_sharded else recon.cpu().numpy()

    def compute_hessian_diagonal(self, weights=None, output_sharded=False):
        """Back projection of the weights with squared coefficients (all pixels,
        matching mbirjax's arange over the full grid)."""
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        if weights is None:
            weights = torch.ones(tuple(sinogram_shape), dtype=torch.float32,
                                 device=self.torch_device)
        elif tuple(weights.shape) != tuple(sinogram_shape):
            raise ValueError('Weights must be constant or an array compatible with sinogram'
                             f'\nGot weights.shape = {tuple(weights.shape)}, but '
                             f'sinogram.shape = {tuple(sinogram_shape)}')
        indices = torch.arange(recon_shape[0] * recon_shape[1], dtype=torch.int64,
                               device=self.torch_device)
        hessian = self.sparse_back_project(weights, indices, coeff_power=2)
        hessian = hessian.reshape((recon_shape[0], recon_shape[1], hessian.shape[-1]))
        return hessian if output_sharded else hessian.cpu().numpy()

    def get_voxels_at_indices(self, recon, indices):
        return recon.reshape((-1, recon.shape[-1]))[indices]

    # ── auto-regularization (verbatim-math numpy ports) ───────────────────────
    def auto_set_regularization_params(self, sinogram, weights=None):
        # Host-side statistics: accept tensors (any device) or numpy.
        if torch.is_tensor(sinogram):
            sinogram = sinogram.cpu().numpy()
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        if self.get_params('auto_regularize_flag'):
            num_real_views = self.get_params('sinogram_shape')[0]
            small_sinogram = self.subsample_views(sinogram, num_real_views=num_real_views)
            small_weights = 1 if weights is None else self.subsample_views(
                weights, num_real_views=num_real_views)

            sino_indicator = self._get_sino_indicator(small_sinogram,
                                                      verbose=self.get_params('verbose'))
            self._check_lateral_truncation(sino_indicator)
            self.auto_set_sigma_y(small_sinogram, sino_indicator, small_weights)

            recon_std = self._get_estimate_of_recon_std(small_sinogram, sino_indicator)
            self.auto_set_sigma_x(recon_std)
            self.auto_set_sigma_prox(recon_std)

        values = [float(v) for v in self.get_params(list(_AUTO_REGULARIZATION_PARAM_NAMES))]
        return dict(zip(_AUTO_REGULARIZATION_PARAM_NAMES, values))

    def _check_lateral_truncation(self, sino_indicator):
        if np.all(sino_indicator):
            return
        edge_frac = float(np.mean(np.logical_or(sino_indicator[:, :, 0],
                                                sino_indicator[:, :, -1])))
        if edge_frac > 0.02 and self.get_params('verbose') > 0:
            warnings.warn(
                f"Lateral FoV truncation detected: the object support reaches the detector's "
                f"edge channels in {edge_frac:.0%} of the sampled view-rows.  Consider using "
                f"scale_recon_shape(s, s) where s >= 1.1 to improve image quality.")

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        snr_db = self.get_params('snr_db')
        magnification = self.get_magnification()
        delta_voxel, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_channel'])

        signal_rms = float(np.average(weights * np.asarray(sinogram) ** 2, None,
                                      sino_indicator) ** 0.5)
        rel_noise_std = 10 ** (-snr_db / 20)
        default_pixel_pitch = delta_det_channel / magnification
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        sigma_y = np.float32(rel_noise_std * signal_rms *
                             (pixel_pitch_relative_to_default ** 0.5))
        self.set_params(no_warning=True, sigma_y=float(sigma_y), auto_regularize_flag=True)

    def auto_set_sigma_x(self, recon_std):
        sharpness = self.get_params('sharpness')
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=float(sigma_x), auto_regularize_flag=True)

    def auto_set_sigma_prox(self, recon_std):
        sharpness = self.get_params('sharpness')
        sigma_prox = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_prox=float(sigma_prox),
                        auto_regularize_flag=True)

    @staticmethod
    def subsample_views(array, max_views_to_use=20, num_real_views=None):
        num_views = array.shape[0] if num_real_views is None else num_real_views
        max_views_to_use = min(max_views_to_use, num_views)
        step_size = max(num_views // max_views_to_use, 1)
        return np.array(array[:num_views][::step_size])

    @staticmethod
    def _get_sino_indicator(sinogram, verbose=1):
        sinogram = np.asarray(sinogram)
        if np.iscomplexobj(sinogram):
            raise TypeError("sinogram must be real-valued; got complex dtype.")
        if not np.isfinite(sinogram).all():
            raise ValueError("sinogram contains NaN and/or Inf values.")

        left, right = vcd_utils.estimate_background_cluster_boundaries(sinogram)
        threshold = right + (right - left)

        max_sino = np.max(sinogram)
        if max_sino <= 0:
            if verbose > 0:
                warnings.warn("Sinogram contains no positive values. This may lead to a "
                              "contrast reversed reconstruction.")
            return np.ones_like(sinogram, dtype=np.int8)
        if max_sino < threshold:
            if verbose > 0:
                warnings.warn('\nUnable to determine sinogram background. This may affect '
                              'regularization.\n')
            return np.ones_like(sinogram, dtype=np.int8)

        object_level = 0.25
        object_median = np.median(sinogram[sinogram >= threshold])
        object_threshold = object_level * object_median
        return np.int8(sinogram >= object_threshold)

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        delta_det_channel = self.get_params('delta_det_channel')
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        magnification = self.get_magnification()
        num_det_channels = sinogram.shape[-1]

        typical_sinogram_value = np.average(np.abs(sinogram), weights=sino_indicator)
        typical_path_length_space = (2 * recon_shape[0] * recon_shape[1]) / (
                recon_shape[0] + recon_shape[1]) * delta_voxel
        typical_path_length_sino = num_det_channels * delta_det_channel / magnification
        typical_path_length = np.minimum(typical_path_length_space, typical_path_length_sino)
        return typical_sinogram_value / typical_path_length

    # ── direct recon (FBP) machinery ──────────────────────────────────────────
    def _apply_direct_recon_filter(self, sinogram, filter_name, filter_scale,
                                   output_sharded=False):
        """Scale the (tiny) filter by filter_scale * pi / num_views, then filter
        every detector row ('valid'); the scalar fold keeps everything f32."""
        sinogram = torch.as_tensor(sinogram, dtype=torch.float32, device=self.torch_device)
        num_channels = sinogram.shape[2]
        num_views = self.get_params('sinogram_shape')[0]
        recon_filter = tomography_utils.generate_direct_recon_filter(
            num_channels, filter_name=filter_name)
        recon_filter = recon_filter * np.float32(filter_scale * (np.pi / num_views))
        filter_t = torch.as_tensor(recon_filter, device=self.torch_device)
        filtered = tomography_utils.apply_row_filter(sinogram, filter_t)
        return filtered if output_sharded else filtered.cpu().numpy()

    # ── loss / stats (mirrors get_forward_model_loss + _vcd_iteration_stats) ──
    @staticmethod
    def get_forward_model_loss(error_sinogram, sigma_y, weights=None, normalize=True):
        if weights is None:
            weights = 1
            avg_weight = 1
        elif not torch.is_tensor(weights) or weights.ndim == 0:
            avg_weight = weights
        else:
            avg_weight = torch.mean(weights)
        if normalize:
            weighted_sq_sum = torch.sum(error_sinogram * error_sinogram * weights)
            loss = torch.sqrt(weighted_sq_sum /
                              (avg_weight * float(error_sinogram.numel()))) / sigma_y
        else:
            loss = (1.0 / (2 * sigma_y ** 2)) * torch.sum(
                (error_sinogram * error_sinogram) * weights)
        return loss

    @staticmethod
    def _vcd_iteration_stats(error_sinogram, flat_recon, sigma_y, weights=None):
        fm_loss = TomographyModel.get_forward_model_loss(error_sinogram, sigma_y, weights)
        recon_l1 = torch.sum(torch.abs(flat_recon))
        es_rmse = torch.sqrt(torch.sum(error_sinogram * error_sinogram)
                             / float(error_sinogram.numel()))
        return fm_loss, recon_l1, es_rmse

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights,
                             fm_constant, const_weights):
        forward_linear = fm_constant * torch.sum(weighted_error_sinogram * delta_sinogram)
        if const_weights:
            forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram)
        else:
            forward_quadratic = fm_constant * torch.sum(
                delta_sinogram * delta_sinogram * weights)
        return forward_linear, forward_quadratic

    # ── the VCD engine ────────────────────────────────────────────────────────
    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        """One-subset update closure; mirrors mbirjax's create_vcd_subset_updater
        with in-place torch state updates replacing donation."""
        positivity_flag = self.get_params('positivity_flag')
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = (b, sigma_x, p, q, T)
        sigma_prox = self.get_params('sigma_prox')
        recon_shape = self.get_params('recon_shape')
        max_alpha = self.get_params('max_alpha')

        const_weights = not torch.is_tensor(weights)
        if const_weights and abs(weights - 1) > 1e-5:
            raise ValueError('Constant weights must have value 1.')

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices):
            # Prior gradient/Hessian at the subset's cylinders.
            if prox_input is None:
                prior_grad, prior_hess = _qggmrf.qggmrf_gradient_and_hessian_at_indices(
                    flat_recon, recon_shape, pixel_indices, qggmrf_params)
            else:
                prior_hess = 1 / (sigma_prox ** 2)
                prior_grad = _qggmrf.prox_gradient_at_indices(
                    flat_recon, prox_input, pixel_indices, sigma_prox)

            weighted_error_sinogram = (error_sinogram if const_weights
                                       else weights * error_sinogram)

            # Forward-model gradient and Hessian on the subset.
            forward_grad = -fm_constant * self.sparse_back_project(
                weighted_error_sinogram, pixel_indices)
            forward_hess = fm_constant * fm_hessian[pixel_indices]

            # Diagonally-preconditioned direction.
            delta_recon_at_indices = -((forward_grad + prior_grad)
                                       / (forward_hess + prior_hess))

            prior_linear = torch.sum(prior_grad * delta_recon_at_indices)
            prior_quadratic_approx = torch.sum(prior_hess * delta_recon_at_indices ** 2)

            # Sinogram-domain direction and the on-device line search.
            delta_sinogram = self.sparse_forward_project(delta_recon_at_indices,
                                                         pixel_indices)
            forward_linear, forward_quadratic = self.get_forward_lin_quad(
                weighted_error_sinogram, delta_sinogram, weights, fm_constant,
                const_weights)

            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + _F32_EPS
            alpha = alpha_numerator / alpha_denominator
            alpha = torch.clamp(alpha, _F32_EPS, max_alpha)

            if positivity_flag is True:
                recon_at_indices = flat_recon[pixel_indices]
                pos_constant = 1.0 / (alpha + _F32_EPS)
                delta_recon_at_indices = torch.maximum(
                    -pos_constant * recon_at_indices, delta_recon_at_indices)
                delta_sinogram = self.sparse_forward_project(delta_recon_at_indices,
                                                             pixel_indices)

            # Apply the update in place (the donation-free torch idiom).
            delta_recon_at_indices = alpha * delta_recon_at_indices
            flat_recon.index_add_(0, pixel_indices, delta_recon_at_indices)
            delta_sumsq_subset = torch.sum(
                delta_recon_at_indices * delta_recon_at_indices, dim=0)
            error_sinogram.sub_(alpha * delta_sinogram)

            ell1_for_subset = torch.sum(torch.abs(delta_recon_at_indices))
            return (flat_recon, error_sinogram, ell1_for_subset, alpha,
                    delta_sumsq_subset)

        return vcd_subset_updater

    def vcd_partition_iterator(self, vcd_subset_updater, flat_recon, error_sinogram,
                               partition):
        """One full pass over the partition's subsets in np.random order (the
        same RNG call as mbirjax, for trace parity)."""
        ell1_for_partition = 0
        alpha_sum = 0
        delta_sumsq_partition = 0
        subset_indices = np.random.permutation(partition.shape[0])

        for index in subset_indices:
            subset = partition[index]
            (flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset,
             delta_sumsq_subset) = vcd_subset_updater(flat_recon, error_sinogram, subset)
            ell1_for_partition += ell1_for_subset
            alpha_sum += alpha_for_subset
            delta_sumsq_partition = delta_sumsq_partition + delta_sumsq_subset

        return (flat_recon, error_sinogram, ell1_for_partition,
                alpha_sum / partition.shape[0], delta_sumsq_partition)

    def vcd_recon(self, sinogram, partitions, partition_sequence,
                  stop_threshold_change_pct, weights=None, init_recon=None,
                  prox_input=None, first_iteration=0):
        """The VCD loop for a given set of partitions and sequence (single
        device; mirrors mbirjax.vcd_recon minus sharding and checkpointing)."""
        self.verify_valid_params()
        dev = self.torch_device
        recon_shape = self.get_params('recon_shape')

        constant_weights = weights is None
        if constant_weights:
            weights = 1
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=dev)

        sinogram = torch.as_tensor(sinogram, dtype=torch.float32, device=dev)

        scale_recon_to_sinogram = init_recon is None
        if init_recon is None:
            self.logger.info('Starting direct recon for initial reconstruction')
            init_recon = self.direct_recon(sinogram, output_sharded=True)
        elif isinstance(init_recon, int):
            init_recon = torch.full(tuple(recon_shape), float(init_recon),
                                    dtype=torch.float32, device=dev)
        else:
            init_recon = torch.as_tensor(init_recon, dtype=torch.float32, device=dev)

        if tuple(init_recon.shape) != tuple(recon_shape):
            raise ValueError(f"init_recon does not have the correct shape. Expected "
                             f"{tuple(recon_shape)}, got {tuple(init_recon.shape)}.")

        # Initialize the error sinogram, scaling the init to the sinogram:
        # alpha* = argmin ||y - alpha A x0||_W^2.
        self.logger.info('Initializing error sinogram')
        error_sinogram = self.forward_project(init_recon, output_sharded=True)
        weighted_error_sinogram = (error_sinogram if constant_weights
                                   else weights * error_sinogram)
        wtd_err_sino_norm = torch.sum(weighted_error_sinogram * error_sinogram)
        if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
            alpha = (torch.sum(weighted_error_sinogram * sinogram)
                     / wtd_err_sino_norm).item()
        else:
            alpha = 1
        error_sinogram = sinogram - alpha * error_sinogram
        init_recon = alpha * init_recon

        if prox_input is not None:
            prox_input = torch.as_tensor(prox_input, dtype=torch.float32, device=dev)
            prox_input = prox_input.reshape((-1, prox_input.shape[-1]))

        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        self.logger.info('Computing Hessian diagonal')
        hess_weights = None if constant_weights else weights
        fm_hessian = self.compute_hessian_diagonal(weights=hess_weights,
                                                   output_sharded=True)
        fm_hessian = fm_hessian.reshape((-1, fm_hessian.shape[-1]))

        flat_recon = init_recon.reshape((-1, init_recon.shape[-1])).contiguous()

        stat_weights = 1 if constant_weights else weights
        vcd_subset_updater = self.create_vcd_subset_updater(
            fm_hessian, weights=stat_weights, prox_input=prox_input)

        self.logger.info('Starting VCD iterations')
        max_iters = partition_sequence.size
        fm_rmse = np.zeros(max_iters)
        pm_loss = np.zeros(max_iters)
        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        delta_norm_per_slice = np.zeros((max_iters, recon_shape[2]))
        num_iters = 0
        for i in range(max_iters):
            partition = partitions[partition_sequence[i]]
            (flat_recon, error_sinogram, ell1_for_partition, alpha,
             delta_sumsq_partition) = self.vcd_partition_iterator(
                vcd_subset_updater, flat_recon, error_sinogram, partition)

            fm_loss_i, recon_l1, es_rmse = self._vcd_iteration_stats(
                error_sinogram, flat_recon, sigma_y, stat_weights)
            fm_rmse[i] = float(fm_loss_i)
            nmae_update[i] = float(ell1_for_partition) / float(recon_l1)
            alpha_values[i] = float(alpha)
            delta_norm_per_slice[i] = np.sqrt(
                delta_sumsq_partition.cpu().numpy())[:recon_shape[2]]

            if verbose >= 1:
                self.logger.info(
                    '\nAfter iteration {} of a max of {}: Pct change={:.4f}, '
                    'Forward loss={:.4f}'.format(i + first_iteration,
                                                 max_iters + first_iteration,
                                                 100 * nmae_update[i], fm_rmse[i]))
                self.logger.info(f'Relative step size (alpha)={alpha_values[i]:.2f}, '
                                 f'Error sino RMSE={float(es_rmse):.4f}')
                self.logger.info('Number subsets = {}'.format(partition.shape[0]))
            num_iters += 1
            if nmae_update[i] < stop_threshold_change_pct / 100:
                self.logger.warning('Change threshold stopping condition reached')
                break

        recon_3d = flat_recon.reshape(tuple(recon_shape[:2]) + (flat_recon.shape[-1],))
        losses = (fm_rmse[:num_iters], pm_loss[:num_iters], nmae_update[:num_iters],
                  alpha_values[:num_iters], delta_norm_per_slice[:num_iters])
        return recon_3d, losses

    def initialize_recon(self, sinogram, weights=None, init_recon=None,
                         max_iterations=15, first_iteration=0):
        """Partitions, sequence, input checks, and auto-regularization."""
        recon_shape, granularity, use_ror_mask = self.get_params(
            ['recon_shape', 'granularity', 'use_ror_mask'])
        partitions = vcd_utils.gen_set_of_pixel_partitions(
            recon_shape, granularity, device=self.torch_device,
            use_ror_mask=use_ror_mask)

        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = vcd_utils.gen_partition_sequence(
            partition_sequence, max_iterations=max_iterations)
        partition_sequence = partition_sequence[first_iteration:]

        sinogram_np = np.asarray(sinogram) if not torch.is_tensor(sinogram) \
            else sinogram.cpu().numpy()
        if np.iscomplexobj(sinogram_np):
            raise TypeError("sinogram must be real-valued; got complex dtype.")
        if not np.isfinite(sinogram_np).all():
            raise ValueError("sinogram contains NaN and/or Inf values.")
        if weights is not None:
            weights_np = np.asarray(weights) if not torch.is_tensor(weights) \
                else weights.cpu().numpy()
            if not np.isfinite(weights_np).all():
                raise ValueError("weights contains NaN and/or Inf values.")
            if (weights_np < 0).any():
                raise ValueError("weights contain negative values.")
            if (weights_np == 0).all():
                raise ValueError("all weights are zero.")

        regularization_params = self.auto_set_regularization_params(sinogram_np,
                                                                    weights=weights)
        return (sinogram, weights, init_recon, partitions, partition_sequence,
                granularity, regularization_params)

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=15,
              stop_threshold_change_pct=0.2, first_iteration=0):
        """MBIR reconstruction via multi-granular VCD (see mbirjax.recon).

        Reproducibility: partitions and subset order draw from numpy's global
        RNG; call ``np.random.seed(seed)`` first for a reproducible run.

        Returns:
            (recon, recon_dict): numpy volume and a dict with 'recon_params'
            and 'model_params'.
        """
        (sinogram, weights, init_recon, partitions, partition_sequence, granularity,
         regularization_params) = self.initialize_recon(
            sinogram, weights, init_recon, max_iterations, first_iteration)

        with torch.inference_mode():
            recon, loss_vectors = self.vcd_recon(
                sinogram, partitions, partition_sequence,
                stop_threshold_change_pct, weights=weights, init_recon=init_recon,
                first_iteration=first_iteration)

        partition_sequence = [int(val) for val in partition_sequence]
        fm_rmse = [float(val) for val in loss_vectors[0]]
        prior_loss = [0]
        stop_pct = [100 * float(val) for val in loss_vectors[2]]
        alpha_values = [float(val) for val in loss_vectors[3]]
        delta_norm_per_slice = [[float(v) for v in row] for row in loss_vectors[4]]
        num_iterations = len(fm_rmse)
        recon_params = dict(zip(recon_param_names,
                                [num_iterations, granularity, partition_sequence,
                                 fm_rmse, prior_loss, regularization_params,
                                 stop_pct, alpha_values, delta_norm_per_slice]))
        recon_dict = {'recon_params': recon_params,
                      'model_params': {k: v.val for k, v in self.params.items()}}
        return recon.cpu().numpy(), recon_dict

    @staticmethod
    def gen_weights(sinogram, weight_type):
        return vcd_utils.gen_weights(sinogram, weight_type)

    def reshape_recon(self, recon):
        recon_shape = self.get_params('recon_shape')
        return recon.reshape(recon_shape)
