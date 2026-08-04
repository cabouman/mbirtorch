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
        """
        Perform a full forward projection at all voxels in the region of
        reconstruction (the pixels selected by the ROR mask; see
        :func:`vcd_utils.get_2d_ror_mask`).

        Args:
            recon (numpy or tensor): 3D volume with shape
                (num_recon_rows, num_recon_cols, num_recon_slices).
            output_sharded (bool, optional): If False (default), return a numpy
                array.  If True, return the device tensor (the mbirjax argument
                name, kept for API compatibility; here it means "skip the numpy
                exit").

        Returns:
            The sinogram, shape (num_views, num_det_rows, num_det_channels).
        """
        recon_shape = self.get_params('recon_shape')
        recon = torch.as_tensor(recon, dtype=torch.float32, device=self.torch_device)
        indices = torch.as_tensor(self._full_indices(), dtype=torch.int64,
                                  device=self.torch_device)
        voxel_values = recon.reshape(-1, recon.shape[-1])[indices]
        sinogram = self.sparse_forward_project(voxel_values, indices)
        return sinogram if output_sharded else sinogram.cpu().numpy()

    def back_project(self, sinogram, output_sharded=False):
        """
        Perform a full back projection at all voxels in the region of
        reconstruction (zeros outside the ROR mask).

        Args:
            sinogram (numpy or tensor): 3D array with shape
                (num_views, num_det_rows, num_det_channels).
            output_sharded (bool, optional): If False (default), return a numpy
                array.  If True, return the device tensor.

        Returns:
            The back projection, shape (num_recon_rows, num_recon_cols,
            num_recon_slices).
        """
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
        """
        Computes the diagonal of the Hessian matrix, which is computed by doing
        a backprojection of the weight matrix except using the square of the
        coefficients in the backprojection to a given voxel.  If weights is not
        None, it must be an array with the same shape as the sinogram; if None,
        constant weights of 1 are used.

        The indices cover ALL pixels of the grid (matching mbirjax's arange over
        the full grid, not the ROR-masked set).

        Args:
            weights (numpy or tensor, optional): 3D positive weights with the
                same shape as the sinogram.  Defaults to all 1s.
            output_sharded (bool, optional): If False (default), return numpy;
                if True, return the device tensor.

        Returns:
            Diagonal of the Hessian matrix with the same shape as the recon.
        """
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
        """
        Automatically sets the regularization parameters (sigma_y, sigma_x, and
        sigma_prox) used in MBIR reconstruction based on the provided sinogram
        and optional weights.

        Args:
            sinogram (ndarray or tensor): 3D sinogram with shape
                (num_views, num_det_rows, num_det_channels).
            weights (ndarray or tensor, optional): 3D weights array with the same
                shape as the sinogram.  Defaults to all 1s.

        Returns:
            dict containing the parameters sigma_y, sigma_x, sigma_prox.

        Notes:
            The method adjusts the regularization parameters only if
            `auto_regularize_flag` is set to True within the model's parameters.
            Inputs are cast to numpy before calculation (the statistics run on
            the host, on a view subsample, exactly as in mbirjax).
        """
        # Host-side statistics: accept tensors (any device) or numpy.
        if torch.is_tensor(sinogram):
            sinogram = sinogram.cpu().numpy()
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        if self.get_params('auto_regularize_flag'):
            # Estimate the regularization stats from a view subsample (see
            # subsample_views) -- both cheap and independent of sinogram size.
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
        """Warn if the sinogram support reaches the detector's edge channels
        (lateral FoV truncation).

        Args:
            sino_indicator (ndarray): binary support indicator from
                :meth:`_get_sino_indicator`, shaped (views, rows, channels) --
                typically view-subsampled.
        """
        if np.all(sino_indicator):
            # An all-ones indicator is either the undeterminable-background
            # fallback (which has already warned on its own) or support
            # genuinely everywhere -- indistinguishable here, so skip rather
            # than risk a spurious warning on the fallback.
            return
        edge_frac = float(np.mean(np.logical_or(sino_indicator[:, :, 0],
                                                sino_indicator[:, :, -1])))
        if edge_frac > 0.02 and self.get_params('verbose') > 0:
            warnings.warn(
                f"Lateral FoV truncation detected: the object support reaches the detector's "
                f"edge channels in {edge_frac:.0%} of the sampled view-rows.  Consider using "
                f"scale_recon_shape(s, s) where s >= 1.1 to improve image quality.")

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        """
        Sets the value of the parameter sigma_y for use in MBIR reconstruction.

        Args:
            sinogram (ndarray): 3D sinogram with shape
                (num_views, num_det_rows, num_det_channels), typically
                view-subsampled.
            sino_indicator (ndarray): a binary mask that indicates the region of
                sinogram support; same shape as sinogram.
            weights (ndarray, optional): 3D positive weights with the same shape
                as the sinogram.  Defaults to all 1s.
        """
        snr_db = self.get_params('snr_db')
        magnification = self.get_magnification()
        delta_voxel, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_channel'])

        # Compute RMS value of sinogram excluding empty space
        signal_rms = float(np.average(weights * np.asarray(sinogram) ** 2, None,
                                      sino_indicator) ** 0.5)

        # Convert snr to relative noise standard deviation
        rel_noise_std = 10 ** (-snr_db / 20)

        # This section adjusts the regularization when the reconstruction
        # resolution is greater or less than normal.  For normal resolution,
        # pixel_pitch_relative_to_default = 1.0; low resolution >> 1.0; high
        # resolution << 1.0.  The default pixel pitch is the detector pixel
        # pitch in the recon plane given the magnification.
        default_pixel_pitch = delta_det_channel / magnification
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        # Compute sigma_y and scale by relative pixel pitch
        sigma_y = np.float32(rel_noise_std * signal_rms *
                             (pixel_pitch_relative_to_default ** 0.5))
        self.set_params(no_warning=True, sigma_y=float(sigma_y), auto_regularize_flag=True)

    def auto_set_sigma_x(self, recon_std):
        """
        Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction
        with the qGGMRF prior.

        Args:
            recon_std (float): Estimated standard deviation of the reconstruction
                from :meth:`_get_estimate_of_recon_std`.
        """
        sharpness = self.get_params('sharpness')
        # Compute sigma_x as a fraction of the typical recon value.
        # 0.2 is an empirically determined constant.
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=float(sigma_x), auto_regularize_flag=True)

    def auto_set_sigma_prox(self, recon_std):
        """
        Compute the automatic value of ``sigma_prox`` for use in MBIR
        reconstruction with the proximal map prior.

        Args:
            recon_std (float): Estimated standard deviation of the reconstruction
                from :meth:`_get_estimate_of_recon_std`.
        """
        sharpness = self.get_params('sharpness')
        # Compute sigma_prox as a fraction of the typical recon value.
        # 0.2 is an empirically determined constant.
        sigma_prox = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_prox=float(sigma_prox),
                        auto_regularize_flag=True)

    @staticmethod
    def subsample_views(array, max_views_to_use=20, num_real_views=None):
        """Return an evenly-spaced subsample of approximately ``max_views_to_use``
        views (axis 0) as a host numpy array.

        Statistical sinogram estimates -- the support indicator, the RMS /
        typical value, the auto-regularization stats -- do not need every view,
        so they are computed on such a subsample.  Callers that need to
        subsample a companion array (e.g. weights) the same way just call this
        again with the same arguments (the stride depends only on the view
        count).

        Args:
            array (ndarray or tensor): array batched along axis 0 (views).
            max_views_to_use (int, optional): approximate number of views to
                retain.  Defaults to 20.
            num_real_views (int or None, optional): if set, sample only
                ``array[:num_real_views]``.

        Returns:
            numpy.ndarray: the view-subsampled array on the host.
        """
        num_views = array.shape[0] if num_real_views is None else num_real_views
        max_views_to_use = min(max_views_to_use, num_views)
        step_size = max(num_views // max_views_to_use, 1)
        return np.array(array[:num_views][::step_size])

    @staticmethod
    def _get_sino_indicator(sinogram, verbose=1):
        """
        Compute a binary mask that indicates the region of sinogram support.

        Typically called on a view SUBSAMPLE (see :meth:`subsample_views`), not
        the full sinogram: this runs several host-side reductions.

        Args:
            sinogram (ndarray): 3D sinogram with shape
                (num_views, num_det_rows, num_det_channels).
            verbose (int, optional): Verbosity level.  Defaults to 1.

        Returns:
            (ndarray): int8 support indicator with the same shape as the input.
        """
        # Sometimes users accidentally create complex sinograms when they take
        # the -log.  So we check for complex numbers or NaNs and raise an error.
        sinogram = np.asarray(sinogram)
        if np.iscomplexobj(sinogram):
            raise TypeError("sinogram must be real-valued; got complex dtype.")
        if not np.isfinite(sinogram).all():
            raise ValueError("sinogram contains NaN and/or Inf values.")

        # Compute an initial threshold that results in a non-empty region that
        # contains no background: the background cluster's right boundary plus
        # one cluster width of safety.
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

        # Compute a final threshold that is a fraction of the median of the
        # object region.
        object_level = 0.25
        object_median = np.median(sinogram[sinogram >= threshold])
        object_threshold = object_level * object_median
        return np.int8(sinogram >= object_threshold)

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """
        Estimate the standard deviation of the reconstruction from the sinogram.
        This is used to scale sigma_prox and sigma_x in MBIR reconstruction.

        Args:
            sinogram (ndarray): 3D sinogram with shape
                (num_views, num_det_rows, num_det_channels), typically
                view-subsampled.
            sino_indicator (ndarray): a binary mask that indicates the region of
                sinogram support; same shape as sinogram.
        """
        delta_det_channel = self.get_params('delta_det_channel')
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        magnification = self.get_magnification()
        num_det_channels = sinogram.shape[-1]

        # Compute the typical magnitude of a sinogram value
        typical_sinogram_value = np.average(np.abs(sinogram), weights=sino_indicator)

        # Compute a typical projection path length based on the soft minimum of
        # the recon width and height
        typical_path_length_space = (2 * recon_shape[0] * recon_shape[1]) / (
                recon_shape[0] + recon_shape[1]) * delta_voxel

        # Compute a typical projection path length based on the detector column width
        typical_path_length_sino = num_det_channels * delta_det_channel / magnification

        # Compute a typical projection path as the minimum of the two estimates
        typical_path_length = np.minimum(typical_path_length_space, typical_path_length_sino)

        # Compute a typical recon value by dividing the average sinogram value by
        # a typical projection path length
        return typical_sinogram_value / typical_path_length

    # ── direct recon (FBP) machinery ──────────────────────────────────────────
    def _apply_direct_recon_filter(self, sinogram, filter_name, filter_scale,
                                   output_sharded=False):
        """Shared FBP row-filter for direct reconstruction.

        Scales the recon filter by ``filter_scale * pi / num_views`` -- folded
        into the (tiny) filter array, NOT applied as an out-of-place
        full-sinogram multiply (which would promote f32 -> f64 via np.pi and
        ~double peak memory -- the mbirjax lesson carried over).

        Equally-spaced-angle assumption: the ``pi / num_views`` factor is the
        angular quadrature weight ``d(theta)`` of the backprojection sum that
        approximates the FBP angular integral, so it assumes the views are
        EQUALLY SPACED over the conventional full angular range (the [0, pi)
        period for parallel beam, via the conjugate-ray symmetry).  For
        nonuniformly-spaced angles, limited-angle scans, or short scans this
        scalar is only approximate -- acceptable for direct recon as a quick
        analytic image or an MBIR initializer (iterative ``recon()`` absorbs a
        global angular mis-weighting in a few iterations); a STANDALONE direct
        recon on such data is not quantitatively accurate -- prefer ``recon()``.

        Args:
            sinogram: (num_views, num_rows, num_channels); numpy or tensor.
            filter_name (str): filter for generate_direct_recon_filter ('ramp').
            filter_scale (float): geometry-specific filter scaling
                (FBP: 1/(delta_voxel * delta_voxel_row)).
            output_sharded (bool): True returns the device tensor; False numpy.

        Returns:
            The filtered sinogram.
        """
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
        """
        Calculate the loss function for the forward model from the error
        sinogram and weights, where
        error_sinogram = measured_sinogram - forward_proj(recon).

        Args:
            error_sinogram (tensor): 3D error sinogram with shape
                (num_views, num_det_rows, num_det_channels).
            sigma_y (float): Estimate obtained from auto_set_sigma_y or
                get_params('sigma_y').
            weights (tensor, optional): 3D positive weights with the same shape
                as the sinogram.  Defaults to all 1s.
            normalize (bool, optional, default=True): If True, return the
                weight-normalized RMSE form; otherwise the unnormalized
                weighted squared error.

        Returns:
            The loss as a device scalar tensor.
        """
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
        """
        Compute forward model terms used in line-search updates:
        ``forward_linear = fm_constant * sum(weighted_error_sinogram * delta_sinogram)``
        and
        ``forward_quadratic = fm_constant * sum(delta_sinogram^2 * weights)``.

        Args:
            weighted_error_sinogram (tensor): weights * error_sinogram (or the
                error sinogram itself under constant weights).
            delta_sinogram (tensor): forward projection of the update direction.
            weights (tensor or constant): the sinogram weights.
            fm_constant (float): 1 / sigma_y^2.
            const_weights (bool): True if the weights are the constant 1.

        Returns:
            tuple: ``(forward_linear, forward_quadratic)`` as device scalars.
        """
        forward_linear = fm_constant * torch.sum(weighted_error_sinogram * delta_sinogram)
        if const_weights:
            forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram)
        else:
            forward_quadratic = fm_constant * torch.sum(
                delta_sinogram * delta_sinogram * weights)
        return forward_linear, forward_quadratic

    # ── the VCD engine ────────────────────────────────────────────────────────
    def _get_update_direction(self, forward_grad, prior_grad, forward_hess,
                              prior_hess, pixel_indices):
        """Return the update direction for one subset of pixels.

        The base implementation is the preconditioned gradient:
            update_direction = -(forward_grad + prior_grad) / (forward_hess + prior_hess)
        Geometry subclasses may override this to apply a different preconditioner
        (the mbirjax preconditioning seam, kept for architectural parity).

        Rule for overrides (from mbirjax): to preserve the cost's minimizers,
        this function must return a linear positive definite transformation of
        the total gradient, update_direction = -M (forward_grad + prior_grad)
        with M positive definite.

        Args:
            forward_grad: data-term gradient, shape (num_subset_pixels, num_slices).
            prior_grad: prior gradient, same shape.
            forward_hess: data-term Hessian diagonal, same shape.
            prior_hess: prior curvature, same shape (a scalar on the
                proximal-map path).
            pixel_indices: flat in-plane indices of this subset.  Unused by the
                base implementation; spatially-aware overrides may use it.

        Returns:
            The update direction, same shape as forward_grad.
        """
        return -((forward_grad + prior_grad) / (forward_hess + prior_hess))

    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        """
        Create a function to update a subset of pixels in the recon and error
        sinogram (mirrors mbirjax's create_vcd_subset_updater, with in-place
        torch state updates replacing jax's buffer donation).

        Args:
            fm_hessian (tensor): (num_pixels, num_slices) diagonal of the
                Hessian for the forward model loss.
            weights (tensor or 1): 3D positive weights with the same shape as
                the sinogram, or the constant 1.
            prox_input (tensor, optional): input for the proximal map, flattened
                to (num_pixels, num_slices).

        Returns:
            (callable) vcd_subset_updater(flat_recon, error_sinogram,
            pixel_indices) that updates the recon and error sinogram in place.
        """
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
            """
            Calculate an iteration of the VCD algorithm on a single subset of the
            partition.  Each application should return a better reconstruction.

            The combination (error_sinogram, recon) forms an overcomplete state
            that makes computation efficient, maintained under the invariant
                error_sinogram = measured_sinogram - forward_proj(recon).

            Args:
                flat_recon (tensor): (num_recon_rows x num_recon_cols,
                    num_recon_slices); updated IN PLACE.
                error_sinogram (tensor): (num_views, num_det_rows,
                    num_det_channels); updated IN PLACE.
                pixel_indices (int tensor): 1D array of pixel indices.

            Returns:
                flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset,
                delta_sumsq_subset: the state (updated to reduce the overall
                loss), the L1 norm of this subset's recon change, the relative
                step size, and the per-slice sum of squared update values.
            """
            # Compute the prior model gradient and Hessian (i.e., second
            # derivative) terms at each pixel in the index set.
            if prox_input is None:
                # qGGMRF prior.
                prior_grad, prior_hess = _qggmrf.qggmrf_gradient_and_hessian_at_indices(
                    flat_recon, recon_shape, pixel_indices, qggmrf_params)
            else:
                # Proximal map prior: pointwise, so the Hessian is a scalar.
                prior_hess = 1 / (sigma_prox ** 2)
                prior_grad = _qggmrf.prox_gradient_at_indices(
                    flat_recon, prox_input, pixel_indices, sigma_prox)

            # Compute the forward model gradient and Hessian at each pixel in the
            # index set.  Assumes Loss(delta) =
            # 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2.
            weighted_error_sinogram = (error_sinogram if const_weights
                                       else weights * error_sinogram)

            # Back project to get the gradient; note fm_constant = 1/sigma_y^2.
            forward_grad = -fm_constant * self.sparse_back_project(
                weighted_error_sinogram, pixel_indices)

            # Get the forward Hessian for this subset.
            forward_hess = fm_constant * fm_hessian[pixel_indices]

            # Compute the update direction in the recon domain -- the per-subset
            # preconditioning seam (base: the diagonally-preconditioned
            # direction; geometry models may override _get_update_direction).
            delta_recon_at_indices = self._get_update_direction(
                forward_grad, prior_grad, forward_hess, prior_hess, pixel_indices)

            # Compute delta^T \nabla Q(x_hat; x'=x_hat) for use in finding alpha.
            prior_linear = torch.sum(prior_grad * delta_recon_at_indices)

            # Estimated upper bound for the prior Hessian term.
            prior_quadratic_approx = torch.sum(prior_hess * delta_recon_at_indices ** 2)

            # Compute the update direction in the sinogram domain.
            delta_sinogram = self.sparse_forward_project(delta_recon_at_indices,
                                                         pixel_indices)
            forward_linear, forward_quadratic = self.get_forward_lin_quad(
                weighted_error_sinogram, delta_sinogram, weights, fm_constant,
                const_weights)

            # Compute the optimal update step.  The line search stays ON DEVICE
            # (alpha is a scalar tensor; no host synchronization per subset).
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + _F32_EPS
            alpha = alpha_numerator / alpha_denominator
            alpha = torch.clamp(alpha, _F32_EPS, max_alpha)

            # Enforce the positivity constraint if desired: clip updates so that
            # recon + alpha * delta >= 0, then recompute the sinogram projection.
            if positivity_flag is True:
                recon_at_indices = flat_recon[pixel_indices]
                pos_constant = 1.0 / (alpha + _F32_EPS)
                delta_recon_at_indices = torch.maximum(
                    -pos_constant * recon_at_indices, delta_recon_at_indices)
                delta_sinogram = self.sparse_forward_project(delta_recon_at_indices,
                                                             pixel_indices)

            # Perform sparse updates at the index locations, IN PLACE.  In jax
            # this required buffer donation (out-of-place per-subset updates
            # leak via sharded-array reference cycles); in torch a plain
            # index_add_ / sub_ is the whole mechanism.  The per-slice sum of
            # squared updates is the per-slice convergence diagnostic
            # (delta_norm_per_slice in the recon dict).
            delta_recon_at_indices = alpha * delta_recon_at_indices
            flat_recon.index_add_(0, pixel_indices, delta_recon_at_indices)
            delta_sumsq_subset = torch.sum(
                delta_recon_at_indices * delta_recon_at_indices, dim=0)

            # Update the error sinogram:
            # error_sinogram <- error_sinogram - alpha * delta_sinogram.
            error_sinogram.sub_(alpha * delta_sinogram)

            ell1_for_subset = torch.sum(torch.abs(delta_recon_at_indices))
            return (flat_recon, error_sinogram, ell1_for_subset, alpha,
                    delta_sumsq_subset)

        return vcd_subset_updater

    def vcd_partition_iterator(self, vcd_subset_updater, flat_recon, error_sinogram,
                               partition):
        """
        Calculate a full iteration of the VCD algorithm by scanning over the
        subsets of the partition.  Each iteration should return a better
        reconstruction.  The error_sinogram should always satisfy
        error_sinogram = measured_sinogram - forward_proj(recon).

        Args:
            vcd_subset_updater (callable): function to apply to each subset.
            flat_recon (tensor): (num_recon_rows x num_recon_cols,
                num_recon_slices); updated in place across subsets.
            error_sinogram (tensor): (num_views, num_det_rows,
                num_det_channels); updated in place across subsets.
            partition (int tensor): 2D array where partition[subset_index] gives
                a 1D array of pixel indices.

        Returns:
            (flat_recon, error_sinogram, ell1_for_partition, alpha,
            delta_sumsq_partition): the updated state; the summed L1 recon
            change over all subsets; alpha averaged over the subsets; and the
            per-slice sum of squared update values accumulated over the
            partition's subsets -- since the subsets tile the in-mask pixels,
            this is the squared per-slice L2 norm of the iteration's total
            update.
        """
        # Loop over the subsets of the partition, using random subset_indices to
        # order them (the same np.random call as mbirjax, for trace parity).
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
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate
        Descent algorithm for a given set of partitions and a prescribed
        partition sequence (single device; mirrors mbirjax.vcd_recon minus
        sharding and checkpoint resume).

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            partitions (list): K partitions, each an (N_subsets, N_indices)
                integer index tensor of voxels to be updated in a flattened recon.
            partition_sequence (ndarray): sequence of integers specifying which
                partition is used at each iteration.
            stop_threshold_change_pct (float): stop when the NMAE percent change
                from one iteration to the next falls below this value.
            weights (numpy or tensor, optional): 3D positive weights with the
                same shape as the sinogram.  Defaults to all 1s.
            init_recon (array or int or None): initial reconstruction.  If None,
                direct_recon (FBP) is used.  An int gives a constant volume.
            prox_input (array, optional): reconstruction input to a proximal map.
            first_iteration (int, optional): iteration offset for restarts (used
                only in the printed iteration labels here).

        Returns:
            (recon, recon_stats): the 3D reconstruction tensor and a tuple of
            per-iteration stats (fm_rmse, pm_loss, nmae_update, alpha_values,
            delta_norm_per_slice), where nmae_update is
            ||recon(i+1) - recon(i)||_1 / ||recon(i+1)||_1.
        """
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

        # Initialize the error sinogram.  We find the optimal alpha to minimize
        # (1/2) ||y - alpha A x0||_weights^2, where y is the sinogram and x0 is
        # init_recon, and scale both the error sinogram and the init by it.
        # The scaling applies only to the default (direct-recon) init; a
        # user-supplied init_recon is used as-is (scale_recon_to_sinogram).
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

        # Initialize the diagonal of the Hessian of the forward model: the back
        # projection of the weights with squared coefficients (constant weights
        # use an all-ones sinogram).
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
        """
        Do the parameter initialization needed for recon: generate the set of
        voxel partitions and the partition sequence, validate the inputs, and
        run auto-regularization.

        Args:
            See :meth:`recon` for arguments.

        Returns:
            sinogram, weights, init_recon, partitions, partition_sequence,
            granularity, regularization_params
        """
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
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate
        Descent algorithm.  This function takes care of generating its own
        partitions and partition sequence.

        To restart a recon using the same partition sequence, set
        first_iteration to the number of iterations completed so far and set
        init_recon to the output of the previous recon; this continues the
        partition sequence from where the previous recon left off.

        Reproducibility note: the pixel partitions are drawn from numpy's
        global random number generator, so reconstructions vary slightly from
        run to run.  For a reproducible result, call ``np.random.seed(seed)``
        before calling this method.

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            weights (numpy or tensor, optional): 3D positive weights with the
                same shape as the sinogram.  Defaults to None (all 1s).
            init_recon (array, int, or None, optional): initial reconstruction.
                If None, direct_recon is called with default arguments.
            max_iterations (int, optional): maximum number of VCD iterations.
            stop_threshold_change_pct (float, optional): stop when
                100 * ||delta_recon||_1 / ||recon||_1 between iterations drops
                below this value.  Defaults to 0.2; set 0 to guarantee exactly
                max_iterations.
            first_iteration (int, optional): the number of iterations previously
                completed when restarting a recon.  Defaults to 0.

        Returns:
            (recon, recon_dict): the numpy reconstruction volume, and a dict
            with entries 'recon_params' (per-iteration traces and settings) and
            'model_params' (a snapshot of the model parameters).
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
