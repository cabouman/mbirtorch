"""TomographyModel: the VCD reconstruction loop, FBP, and projector wrappers.

The module holds the VCD loop (numpy partitions, subset updater, on-device
line search, positivity), FBP via torch.fft, the auto-regularization chain, the
placement and gather functions, the multi-device VCD loop, and the public
numpy-at-the-boundary API.  The loop's working state is always per-device
(:class:`_sharding.Shards`); a single device is the one-shard case.  The
checkpoint-resume path mutates the caller's arrays in place.

The order of operations in every formula is deliberate.  A seeded run
reproduces a reconstruction iteration for iteration, so do not reorder the
arithmetic.
"""

import contextlib
import datetime
import io
import math
import os
import warnings

import numpy as np
import torch

from . import _memory_ledger
from . import _sharding
from . import _widening_floors
from . import qggmrf as _qggmrf
from . import tomography_utils, vcd_utils
from .memory_stats import get_memory_stats
from ._utils import _AUTO_REGULARIZATION_PARAM_NAMES, recon_param_names
from .parameter_handler import ParameterHandler
from .projectors import Projectors, maybe_compile

_F32_EPS = float(np.finfo(np.float32).eps)

# This is the number of pixels in one transferred cylinder batch in the multi-device forward
# projection.  Setting forward_project_pixel_batch on the model overrides it.
FORWARD_PIXEL_BATCH = 32768


# These module level functions hold the fused per-subset arithmetic, so that
# each one is compiled once per process.
def _diagonal_update_direction(forward_grad, prior_grad, forward_hess, prior_hess):
    return -((forward_grad + prior_grad) / (forward_hess + prior_hess))


def _prior_line_terms(prior_grad, prior_hess, delta):
    return (torch.sum(prior_grad * delta),
            torch.sum(prior_hess * delta * delta))


def _forward_lin_quad_const(weighted_error_sinogram, delta_sinogram, fm_constant):
    return (fm_constant * torch.sum(weighted_error_sinogram * delta_sinogram),
            fm_constant * torch.sum(delta_sinogram * delta_sinogram))


def _forward_lin_quad_weighted(error_sinogram, delta_sinogram, weights, fm_constant):
    # The product of the weights and the error is fused into the reductions, so
    # no array of sinogram size is created for each subset.
    return (fm_constant * torch.sum(weights * error_sinogram * delta_sinogram),
            fm_constant * torch.sum(delta_sinogram * delta_sinogram * weights))


def _apply_update(flat_recon, error_sinogram, pixel_indices, delta_scaled,
                  alpha, delta_sinogram):
    # This updates the reconstruction and the error sinogram in place, in one compiled region.
    # The state tensors are returned so that callers rebind them rather than use the side effect.
    flat_recon.index_add_(0, pixel_indices, delta_scaled)
    delta_sumsq = torch.sum(delta_scaled * delta_scaled, dim=0)
    # This subtracts alpha times delta from the error sinogram.  The addcmul_ form reads alpha from
    # a zero dimensional device tensor.  The sub_ form would force a host synchronization.
    error_sinogram.addcmul_(delta_sinogram, alpha, value=-1)
    ell1 = torch.sum(torch.abs(delta_scaled))
    return flat_recon, error_sinogram, delta_sumsq, ell1


def _resolve_device(device):
    """'auto' -> cuda if available, else mps, else cpu; else the given device."""
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def gpu_devices():
    """The GPU devices torch can use, as a tuple.

    Every CUDA device when CUDA is available, the MPS device alone when CUDA
    is absent and MPS is present, and an empty tuple when there is no GPU.
    The tuple reports the hardware and reads no environment variable.
    """
    if torch.cuda.is_available():
        return tuple(torch.device('cuda', i) for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        return (torch.device('mps'),)
    return ()


def cpu_devices():
    """The CPU device, as a one-element tuple.

    torch presents one CPU device however many cores the machine has, so a
    pool of CPU devices has one entry.
    """
    return (torch.device('cpu'),)


def default_devices():
    """The devices a run uses when none are named, as a list: the GPU devices
    when there are any, and otherwise the CPU device."""
    return list(gpu_devices()) or list(cpu_devices())


def _array_extremes(array):
    """Return the minimum and maximum of ``array``, computed where the array
    already is.

    One pass answers every input check below.  A NaN anywhere propagates into
    both values, and an infinity appears in the extreme on its own side.  Both
    reductions run through torch, so a numpy array is wrapped rather than
    copied, and a device array is never brought back to the host.
    """
    if torch.is_tensor(array):
        tensor = array
    else:
        with warnings.catch_warnings():
            # This function only reads the array, so torch's warning about a
            # host array that is not writable does not apply.
            warnings.filterwarnings('ignore', message='.*not writable.*')
            tensor = torch.as_tensor(array)
    return float(tensor.min()), float(tensor.max())


class TomographyModel(ParameterHandler):
    """
    Base class for all tomography geometries.  It provides projection
    (:meth:`forward_project`, :meth:`back_project`), reconstruction
    (:meth:`recon`, :meth:`prox_map`, :meth:`recon_direct`), device
    configuration (:meth:`configure_devices`), and parameter handling.
    Users construct a geometry subclass (for example ``ConeBeamModel`` or
    ``ParallelBeamModel``) rather than this class.

    Constructor args common to all geometries:

    * sinogram_shape (tuple of int): (num_views, num_det_rows,
      num_det_channels).
    * view_batch_size (int, optional): views per projection call.  None
      (default) lets each projection method choose.
    * compile_mode (str, optional): 'auto' (default) compiles the
      computational kernels with torch.compile; 'off' runs without
      compilation.
    """

    def __init__(self, sinogram_shape, view_batch_size=None,
                 compile_mode='auto', **kwargs):
        super().__init__()
        # The device state resolves on first use, so that inspecting a model or
        # calling configure_devices never touches a device that was not chosen.
        self._torch_device = None
        self._sino_placement = None
        self._recon_placement = None
        self._projector_functions = None
        # This is the number of views per call in the batched drivers.  None means the body's
        # default, and the driver may reduce the batch to stay within its memory budget.
        self.view_batch_size = view_batch_size
        # A compile_mode of 'auto' compiles the main chains with torch.compile,
        # and 'off' runs them eagerly.
        self.compile_mode = compile_mode
        # This caches the prox initialization, so that a Plug and Play loop
        # calls initialize_recon only once.
        self.prox_data = None
        self._dc_damping_cache = None
        self.dev2dev_safe = True     # configure_devices probes this.
        # This becomes False once configure_devices is called.  An explicit
        # layout belongs to the caller permanently.
        self.device_layout_is_automatic = True
        # This records the (sinogram_shape, recon_shape) pair that the current automatic layout
        # was decided from.  _apply_device_policy decides again when the shapes do not match.
        self._settled_shapes = None
        # This is the workload, either 'recon' or 'direct', that the settled layout's capacity
        # check was made against.  A call that allocates more than that check priced runs it again.
        self._settled_workload = None
        # These are the device counts the automatic choice rejected, for the
        # run log.
        self.device_choice_rejections = []
        self._speed_floor_fallback = None
        self._speed_floor_held = None
        # The preflight margin covers memory the ledger cannot see, such as fragmentation and
        # CUDA workspaces outside torch.  The preflight runs when the automatic layout is decided,
        # so setting skip_memory_preflight later has no effect until the shapes change.
        self.skip_memory_preflight = False
        self.memory_preflight_margin = 0.15
        # These two exist for a test harness to read.  Nothing in the library
        # reads them.
        self.last_memory_ledger = None
        self.last_memory_calibration = None
        # _vcd_recon owns the per-device thread pool.  It is None outside a
        # reconstruction.
        self._per_device_pool = None

        # The geometry's own parameters, such as angles, are added as new Param
        # entries.
        from ._utils import Param
        for key, val in kwargs.items():
            self.params[key] = Param(val, True)
        self.set_params(no_compile=True, no_warning=True,
                        sinogram_shape=tuple(int(s) for s in sinogram_shape))

        # The order matters.  The geometry defaults come first, then the
        # projectors, then the validity check.
        self.auto_set_recon_geometry(no_compile=True, no_warning=True)
        self.verify_valid_params()

    @property
    def compile_enabled(self):
        return self.compile_mode != 'off'

    # Each device property below resolves on first read and can also be
    # assigned directly by configure_devices and by the automatic widening.
    @property
    def torch_device(self):
        """The model's lead device, resolved on first use.

        Resolution is 'auto': cuda if available, else mps, else cpu.  A
        caller who wants something else calls
        ``configure_devices(devices=[...])``, which sets this before anything
        reads it, so no device is ever touched that the caller did not ask
        for."""
        if self._torch_device is None:
            self._torch_device = _resolve_device('auto')
        return self._torch_device

    @torch_device.setter
    def torch_device(self, value):
        self._torch_device = torch.device(value)

    # Sinogram arrays are sharded by view, which is axis 0.  Reconstruction
    # arrays are sharded by slice, which is the last axis.
    @property
    def sino_placement(self):
        if self._sino_placement is None:
            self._sino_placement = _sharding.Placement([self.torch_device],
                                                       axis=0)
        return self._sino_placement

    @sino_placement.setter
    def sino_placement(self, value):
        self._sino_placement = value

    @property
    def recon_placement(self):
        if self._recon_placement is None:
            self._recon_placement = _sharding.Placement([self.torch_device],
                                                        axis=-1)
        return self._recon_placement

    @recon_placement.setter
    def recon_placement(self, value):
        self._recon_placement = value

    @property
    def projector_functions(self):
        """The projection driver, built on first use against the layout then
        in force."""
        if self._projector_functions is None:
            self.create_projectors()
        return self._projector_functions

    @projector_functions.setter
    def projector_functions(self, value):
        self._projector_functions = value

    # ── hooks for geometry subclasses ─────────────────────────────────────────
    def create_projectors(self):
        self.projector_functions = Projectors(self)

    def get_magnification(self):
        """Return the magnification for this geometry.  Each geometry model
        defines this; parallel beam returns 1.0."""
        raise NotImplementedError

    def get_psf_radius(self):
        raise NotImplementedError

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """
        Set the automatic value of the recon shape and voxel pitch using the
        geometry parameters and sinogram shape.  Each geometry model defines
        this.

        Note: This function should be run after changing geometry parameters
        such as ``delta_det_channel``.  It will set reconstruction parameters
        such as ``recon_shape`` and ``delta_voxel`` to reasonable values.

        Args:
            no_compile (bool, optional): If True, do not rebuild the
                projectors.  Defaults to False.
            no_warning (bool, optional): If True, do not issue warnings.
                Defaults to False.
        """
        raise NotImplementedError

    def _view_batch_bodies(self):
        """Return the geometry's projection bodies for one view batch, as
        (forward, back).

        These must be module level functions that take parameter values built
        by :meth:`_view_batch_args`.  A bound method would hold the model in the
        module level compile cache, and a parameter read inside the compiled
        region would make dynamo trace the parameter machinery."""
        raise NotImplementedError(
            f'{type(self).__name__} defines no per-view-batch projection '
            'bodies.')

    def _view_batch_args(self):
        """Return the argument dictionary for this geometry's projection
        bodies.  Every parameter read happens here, outside the traced region,
        on every call.  The values must not be frozen at build time."""
        raise NotImplementedError

    def _transient_cols(self, band_cols):
        """Return the column count of this geometry's largest temporary array
        for one view, which sets the driver's view batch budget.

        The base class returns the band length.  A geometry whose slice band
        projects onto many detector rows, such as cone beam, overrides this
        with a width derived from its parameters.  Changing the value changes
        batch sizes, the order of floating point summation, and peak memory."""
        return band_cols

    def recon_direct(self, sinogram, filter_name=None, output_sharded=False):
        """
        Do a direct (non-iterative) reconstruction, typically using a form of
        filtered backprojection.  The implementation details are geometry
        specific, and recon_direct may not be available for all geometries.

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            filter_name (string, optional): The name of the filter to use.
                Every geometry's implementation defaults to 'ramp'.
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the device form: a torch
                tensor on a single device, or a Shards container (one
                tensor per device) on a multi-device model.

        Returns:
            recon (numpy or tensor): The reconstructed volume.

        Note:
            An implementation settles the device layout before its first large
            allocation, with ``self._apply_device_policy(workload='direct')``
            as its first statement.  Without that call a direct reconstruction
            on a model whose layout the caller has not fixed runs whole on the
            lead device; with it, the memory check prices the direct
            reconstruction rather than the full recon the device count is
            chosen for.
        """
        raise NotImplementedError

    def recon_split_sino(self, sino, weights=None, half_overlap=5, init_recon=None,
                         max_iterations=15, stop_threshold_change_pct=0.2,
                         first_iteration=0, compute_prior_loss=False,
                         logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                         align_split_grid=False):
        """
        Perform MBIR reconstruction with less memory than :meth:`recon` by
        splitting the detector rows into overlapping row bands -- two halves
        for cone beam, one or more parts for parallel beam -- reconstructing
        one band at a time, and stitching the results.  The output is
        approximately equal to the output of :meth:`recon`.

        The split arithmetic is geometry specific, and recon_split_sino may
        not be available for all geometries; geometries without an
        implementation raise ``NotImplementedError``.

        Args:
            sino (numpy or tensor): Full sinogram of shape (num_views, num_rows, num_cols).  A
                sharded array is not accepted.
            weights (numpy or tensor, optional): Optional sinogram weights with the same shape as
                `sino`.  Not accepted in sharded form, like `sino`.
            half_overlap (int, optional): Number of overlapping detector rows
                kept past the split in each half.  Defaults to 5.
            init_recon (optional): Same as in :meth:`recon`.  Not accepted in
                sharded form, like `sino`.
            max_iterations (int, optional): Same as in :meth:`recon`.
            stop_threshold_change_pct (float, optional): Same as in :meth:`recon`.
            first_iteration (int, optional): Same as in :meth:`recon`.
            compute_prior_loss (bool, optional): Accepted for interface
                compatibility; not currently used.
            logfile_path (str, optional): Same as in :meth:`recon`.  The two
                halves' logs are merged into this single file.
            print_logs (bool, optional): Same as in :meth:`recon`.
            align_split_grid (bool, optional): If True, shift the recon slice
                grid by up to half a slice to align the split with the
                sinogram cut, which removes seam stripes.  Defaults to False.

        Returns:
            Tuple[np.ndarray, dict]: the reconstructed volume, and a metadata
            dictionary with the recon and model parameters for each half plus
            ``'split_params'`` (the overlaps and any alignment shift used).
            If the split would leave either half too thin, the method warns,
            performs a standard :meth:`recon` instead, and returns that
            result's dictionary (no per-half entries).
        """
        raise NotImplementedError(
            f'recon_split_sino is not implemented for {type(self).__name__}.')

    def recon_plastic_metal(self, sino, weights, num_BH_iterations=3, num_constraint_update_iter=10,
                            stop_threshold_change_pct=0.2, num_metal=1, order=3, alpha=1, beta=0.002,
                            gamma=0.1, verbose=0, max_iterations=15,
                            logfile_path='~/.mbirtorch/logs/recon.log',
                            radial_margin=None, top_margin=None, bottom_margin=None):
        """
        Perform iterative metal artifact reduction using plastic-metal beam hardening correction.  If num_metal is 0,
        then this performs a standard MBIR recon.

        The method alternates between adaptive beam hardening correction (via `correct_sino_plastic_metal`)
        and reconstruction, refining the image over several iterations to suppress metal-induced artifacts.

        The method works on any geometry that provides `recon_direct` and `recon`.  For a cone
        beam model the reconstruction passes use `recon_split_sino`; for every other geometry they
        use `recon`.  It has been used mainly with cone beam models.

        Args:
            sino (numpy or tensor):  Input sinogram data to be corrected.  A tensor is converted to
                numpy at entry.  An array in sharded form is not accepted.
            weights (numpy or tensor): Transmission weights used in the reconstruction algorithm.  A
                tensor is converted to numpy at entry.  Not accepted in sharded form, like
                `sino`.
            num_BH_iterations (int, optional): Number of correction-reconstruction iterations. Defaults to 3.
            num_constraint_update_iter (int, optional): Number of iterations for updating constraints.
                At each iteration, the most violated constraints are activated and the quadratic program is re-solved via OSQP.
            stop_threshold_change_pct (float, optional): Relative change threshold (%) for early stopping in MBIR. Defaults to 0.2.
            num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
            order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
            alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
                higher-order terms more strongly. Defaults to 1.
            beta (float, optional): Regularization strength for ridge regression. Defaults to 0.002.
            gamma (float, optional): Stabilization factor used in plastic correction. Multiplies the mean of `s_p`
                to set a positive floor in the denominator, preventing division by near-zero or negative values. Defaults to 0.1.
            verbose (int, optional): Verbosity level for printing intermediate information. Defaults to 0.
            max_iterations (int, optional): Maximum MBIR iterations per reconstruction pass. Defaults to 15.
            logfile_path (str, optional): Same as in the TomographyModel.recon() method.  The BH passes'
                logs are merged into this single file, each under a section header.
            radial_margin, top_margin, bottom_margin (int or None, optional): Segmentation mask margins
                used when classifying plastic/metal; None (default) = size-relative
                (see segment_plastic_metal).

        Returns:
             (recon, recon_dict): The final corrected reconstruction after iterative beam hardening
             correction as a host NumPy array, and the reconstruction dictionary from its final
             reconstruction pass.

        Example:
            >>> recon, recon_dict = ct_model.recon_plastic_metal(
            ...     sino, weights,
            ...     num_BH_iterations=3,
            ...     stop_threshold_change_pct=0.2,
            ...     num_metal=1,
            ...     order=3,
            ...     alpha=1,
            ...     beta=0.005,
            ...     verbose=1
            ... )
            >>> mbirtorch.slice_viewer(recon)
        """
        import functools
        from .preprocess.mar import correct_sino_plastic_metal
        from .preprocess.segmentation import segment_plastic_metal
        from .utilities import merge_log_files
        from .view_utils import slice_viewer

        if num_metal < 0:
            raise ValueError("num_metal must be >= 0")

        # This driver accepts host input only.  The check comes before np.asarray, which would
        # build an object array from a sharded input rather than fail.
        if (isinstance(sino, _sharding.Shards)
                or isinstance(weights, _sharding.Shards)):
            raise ValueError(
                'recon_plastic_metal does not accept a sinogram or weights '
                'in sharded form.  Pass the host (numpy or tensor) sinogram and the '
                'host weights.')
        if isinstance(sino, torch.Tensor):
            sino = sino.detach().cpu().numpy()
        sino = np.asarray(sino)
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            weights = np.asarray(weights)

        # Cone beam uses the split sinogram recon when the model provides it, because that form
        # splits on the host and never places the full sinogram on the devices.  Other geometries
        # use the standard recon with a device form output.
        if ('cone' in self.get_params('geometry_type')
                and type(self).recon_split_sino is not TomographyModel.recon_split_sino):
            recon_function = self.recon_split_sino
        else:
            recon_function = functools.partial(self.recon, output_sharded=True)

        # The output is always a host numpy array.
        def to_output_form(r):
            return r if isinstance(r, np.ndarray) else self._gather_recon(r)

        if num_metal == 0:
            recon, recon_dict = recon_function(sino, weights=weights, max_iterations=max_iterations,
                                               stop_threshold_change_pct=stop_threshold_change_pct,
                                               logfile_path=logfile_path)
            return to_output_form(recon), recon_dict

        if verbose >= 1:
            print("\n************ Perform initial FDK reconstruction  **************")
        recon = self.recon_direct(sino, output_sharded=True)

        # Each beam hardening pass logs to its own temporary file.  The files are merged in the
        # finally block, so the logs of the passes that ran before a failure are kept.
        if logfile_path:
            log_path = os.path.expanduser(logfile_path)
            pass_log_paths = [log_path + '.pass{}'.format(i + 1) for i in range(num_BH_iterations)]
        else:
            log_path, pass_log_paths = None, [None] * num_BH_iterations
        try:
            for i in range(num_BH_iterations):
                if verbose >= 1:
                    print(f"\n************ Correct sino plastic metal {i + 1}  **************")
                corrected_sinogram = correct_sino_plastic_metal(self, sino, recon, num_metal=num_metal, order=order, alpha=alpha, beta=beta, gamma=gamma, num_constraint_update_iter=num_constraint_update_iter,
                                                                radial_margin=radial_margin, top_margin=top_margin, bottom_margin=bottom_margin)

                if verbose >= 1:
                    print(f"\n************ Perform MBIR reconstruction {i + 1} **************")
                # The recon entry points require init_recon to be a host array
                # or a tensor, so a sharded reconstruction is gathered first.
                init = (self._gather_recon(recon)
                        if isinstance(recon, _sharding.Shards) else recon)
                recon, recon_dict = recon_function(corrected_sinogram, weights=weights, init_recon=init,
                                          max_iterations=max_iterations,
                                          stop_threshold_change_pct=stop_threshold_change_pct,
                                          logfile_path=pass_log_paths[i])

                if verbose >= 2:
                    print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
                    plastic_mask, metal_masks, plastic_scale, metal_scales = segment_plastic_metal(
                        recon, num_metal, radial_margin=radial_margin, top_margin=top_margin,
                        bottom_margin=bottom_margin)
                    labels = ['Plastic Mask'] + [f'Metal {j + 1} Mask' for j in range(len(metal_masks))]
                    slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0,
                                    slice_label=labels,
                                    title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')
        finally:
            if log_path:
                # A pass that failed partway may have left its log file open,
                # and the merge below deletes the files it merges.
                self.close_log_file()
                labels = ['recon_plastic_metal: BH pass {}'.format(i + 1) for i in range(num_BH_iterations)]
                merge_log_files(log_path, zip(labels, pass_log_paths))

        return to_output_form(recon), recon_dict

    # ── projection wrappers ───────────────────────────────────────────────────
    def sparse_forward_project(self, voxel_values, pixel_indices):
        """Cylinders at ``pixel_indices`` -> full sinogram.  This is the ONE
        funnel for sparse forward projection: the recon engine, the dense
        wrappers, and external callers all route here, so the surface the
        metrics harness measures is the surface the engine runs.  The output
        matches the input form: a plain tensor in, a tensor out (single
        device); ``Shards`` in, or a multi-device placement, view shards out."""
        voxel_values = self._shard_recon(voxel_values)
        if isinstance(voxel_values, _sharding.Shards):
            return self._sparse_forward_project_sharded(voxel_values, pixel_indices)
        return self.projector_functions._sparse_forward_project_single_device(
            voxel_values, pixel_indices)

    def sparse_back_project(self, sinogram, pixel_indices, coeff_power=1):
        """Sinogram -> cylinders at ``pixel_indices``.  The one funnel for
        sparse back projection (see :meth:`sparse_forward_project`); the
        output matches the input form: tensor in, tensor out; ``Shards`` in,
        or a multi-device placement, slice shards out."""
        sinogram = self._shard_sinogram(sinogram)
        if isinstance(sinogram, _sharding.Shards):
            return self._sparse_back_project_sharded(sinogram, pixel_indices,
                                                     coeff_power=coeff_power)
        return self.projector_functions._sparse_back_project_single_device(
            sinogram, pixel_indices, coeff_power=coeff_power)

    def _band_pool(self, n):
        """Return the thread pool for a sharded projection.  This reuses the
        reconstruction loop's pool when one is active, and otherwise creates a
        pool for this call alone."""
        if self._per_device_pool is not None:
            return contextlib.nullcontext(self._per_device_pool)
        return _sharding.device_pool(n)

    @staticmethod
    def _slice_band_length(slices_per_dev, n_dev, num_pixels, fixed_band=None):
        """Return the band length for streaming the slice axis in the banded
        back projection.

        The default is one band per device, which holds the whole shard.
        Smaller bands were measured to be slower, but they use less memory.
        Setting ``back_project_slice_band`` on the model selects a smaller
        band.  The result never exceeds slices_per_dev, so a band never crosses
        a device boundary."""
        b = fixed_band if fixed_band else slices_per_dev
        return min(int(b), slices_per_dev)

    def _forward_pixel_batch(self):
        """Return the number of pixels in one transferred cylinder batch.  The
        default is :data:`FORWARD_PIXEL_BATCH`, and
        ``forward_project_pixel_batch`` on the model overrides it.  The memory
        ledger calls this method rather than compute the number itself."""
        fixed = getattr(self, 'forward_project_pixel_batch', None)
        return max(1, int(fixed)) if fixed else FORWARD_PIXEL_BATCH

    @staticmethod
    def _balanced_slice_bounds(extent, band_len):
        """Split ``[0, extent)`` into the fewest bands no longer than
        ``band_len``.  The bands do not overlap, and their lengths differ by at
        most one.  An extent that is not positive gives no bands."""
        if extent <= 0:
            # A shard with no slices also has a band length of 0, so the
            # division below would divide by zero.
            return []
        num_bands = -(-extent // band_len)            # Ceiling division.
        base, rem = divmod(extent, num_bands)
        bounds, start = [], 0
        for k in range(num_bands):
            length = base + (1 if k < rem else 0)
            bounds.append((start, start + length))
            start += length
        return bounds

    def _banded_setup(self, pixel_indices):
        """Shared setup for the sharded projectors: per-owner view spans,
        recon slice ranges, and the pixel indices placed once per device."""
        sp, rp = self.sino_placement, self.recon_placement
        if type(self)._view_batch_bodies is TomographyModel._view_batch_bodies:
            raise NotImplementedError(
                f'{type(self).__name__} has no per-view-batch projection '
                'bodies, so the multi-device drivers cannot run.')
        # These are half open view spans and slice band ranges, in device order.  A span can be
        # empty, because with more devices than views or slices the trailing devices own nothing.
        view_spans = [span for _d, span in sp.shard_ranges()]
        band_ranges = rp.shard_ranges()
        idx_per_dev = [torch.as_tensor(pixel_indices, dtype=torch.int64).to(d)
                       for d in sp.devices]
        return sp, rp, view_spans, band_ranges, idx_per_dev

    def _sparse_forward_project_sharded(self, voxel_shards, pixel_indices):
        """Run the sharded forward projection.  A single device placement calls
        the plain driver.  A multi-device placement calls
        :meth:`_sparse_forward_project_cylinders`."""
        if voxel_shards.placement.is_trivial:
            return _sharding.Shards(
                [self.projector_functions._sparse_forward_project_single_device(
                    voxel_shards.tensors[0], pixel_indices)],
                self.sino_placement)
        return self._sparse_forward_project_cylinders(voxel_shards,
                                                      pixel_indices)

    def _sparse_forward_project_cylinders(self, voxel_shards, pixel_indices):
        """Run the multi-device forward projection as a cylinder transfer in
        pixel batches.

        Each device that owns views walks the pixel axis in batches.  For each
        batch it collects the full height cylinders from every device that owns
        slices, then makes one projector call over its own views and the whole
        slice range.  The pixel batch size bounds the transfer between devices,
        and the memory ledger charges for it."""
        sp, rp, view_spans, _band_ranges, idx_per = self._banded_setup(
            pixel_indices)
        pf = self.projector_functions
        num_channels = int(self.get_params('sinogram_shape')[2])
        # This is the block height per call.  A row aligned body sizes its output by the
        # transferred cylinders, which span the whole slice axis.  A two fan body returns rows.
        num_rows = (int(rp.axis_len) if self.rows_track_slices
                    else int(self.get_params('sinogram_shape')[1]))
        num_pixels = int(idx_per[0].shape[0])
        shards = voxel_shards.tensors        # Device order is global slice order.
        pixel_batch = self._forward_pixel_batch()
        batch_bounds = [(p0, min(p0 + pixel_batch, num_pixels))
                        for p0 in range(0, num_pixels, pixel_batch)]

        def worker(i, dev):
            v0, v1 = view_spans[i]
            if v1 <= v0:
                # A device that owns no views produces an empty block.
                return torch.zeros((0, num_rows, num_channels),
                                   dtype=voxel_shards.dtype, device=dev)
            local_idx = idx_per[i]
            owned = None

            def transfer(k):
                p0, p1 = batch_bounds[k]
                return _sharding.transfer_cylinder_batch_async(
                    shards, p0, p1, dev, self.dev2dev_safe)

            # This holds the batch transferred ahead of the one being
            # projected.
            ahead = transfer(0) if batch_bounds else None
            for k, (p0, p1) in enumerate(batch_bounds):
                full_cyl, ready = ahead
                # The next batch's transfer is issued before this batch is projected, on separate
                # streams.  The wait below keeps a projection from starting before its copies land.
                ahead = transfer(k + 1) if k + 1 < len(batch_bounds) else None
                _sharding.wait_for_cylinder_batch(dev, ready)
                # The first batch allocates the block.  Later batches add into that block
                # inside the projector's view loop, which saves one allocation per batch.
                if owned is None:
                    owned = pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i)
                else:
                    pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i, accumulate_into=owned)
                # The cylinder batch is released once its projection is issued.
                full_cyl = None
            if owned is None:
                # With no pixels the device still owes a block for its views,
                # and that block is zero everywhere.
                owned = torch.zeros((v1 - v0, num_rows, num_channels),
                                    dtype=voxel_shards.dtype, device=dev)
            return owned

        # One fan out covers the whole call, so the pixel loop inside the worker issues each
        # device's transfers from the thread that consumes them.
        transfer_devices = (list(voxel_shards.placement.devices)
                            + list(sp.devices))
        _sharding.open_copy_streams(transfer_devices)
        try:
            with self._band_pool(sp.n_devices) as pool:
                tensors = _sharding.run_per_device(sp.devices, worker,
                                                   executor=pool)
        finally:
            # The streams are closed even when a worker raises, because copies
            # already issued are still running and read the shards.
            _sharding.close_copy_streams(transfer_devices)
        return _sharding.Shards(tensors, sp)

    def _sparse_back_project_sharded(self, sino_shards, pixel_indices,
                                     coeff_power=1):
        """Run the banded sharded back projection, which is the adjoint of the
        forward projection.  Every device that owns views back projects them
        onto each slice band, and the partial results are summed onto the
        device that owns that band.  A single device placement calls the plain
        driver."""
        if sino_shards.placement.is_trivial:
            return _sharding.Shards(
                [self.projector_functions._sparse_back_project_single_device(
                    sino_shards.tensors[0], pixel_indices,
                    coeff_power=coeff_power)],
                self.recon_placement)
        sp, rp, view_spans, band_ranges, idx_per = self._banded_setup(pixel_indices)
        pf = self.projector_functions
        aligned = self.rows_track_slices
        recon_tensors = []
        num_pixels = int(idx_per[0].shape[0])
        fixed_band = getattr(self, 'back_project_slice_band', None)
        with self._band_pool(sp.n_devices) as pool:
            for oi, (odev, (s0, s1)) in enumerate(band_ranges):
                # The band is streamed in sub-bands, so each partial result and
                # each gather is the size of a sub-band.
                band_len = self._slice_band_length(
                    s1 - s0, sp.n_devices, num_pixels, fixed_band)
                owner_parts = []
                for (l0, l1) in self._balanced_slice_bounds(s1 - s0, band_len):
                    # A device that owns no views contributes nothing, so its
                    # projector call is skipped and it is dropped from the sum.
                    if aligned:
                        partials = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_back_project_view_range(
                                    sino_shards.tensors[i][
                                        :, s0 + l0:s0 + l1, :],
                                    idx_per[i], view_spans[i],
                                    coeff_power=coeff_power, dev_index=i)
                                if view_spans[i][1] > view_spans[i][0]
                                else None),
                            executor=pool)
                    else:
                        partials = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_back_project_view_range(
                                    sino_shards.tensors[i],
                                    idx_per[i], view_spans[i],
                                    slice_start=s0 + l0, band_slices=l1 - l0,
                                    coeff_power=coeff_power, dev_index=i)
                                if view_spans[i][1] > view_spans[i][0]
                                else None),
                            executor=pool)
                    owner_parts.append(_sharding.sum_band_to_owner(
                        [p for p in partials if p is not None], odev,
                        self.dev2dev_safe))
                    # This release must come before the next band's run_per_device call.
                    # Without it, this band's partial result stays on every device.
                    partials = None
                if not owner_parts:
                    # A device that owns no slices produced no bands, so the
                    # dtype and the device are named explicitly here.
                    recon_tensors.append(torch.zeros(
                        (num_pixels, 0), dtype=sino_shards.dtype, device=odev))
                else:
                    recon_tensors.append(owner_parts[0] if len(owner_parts) == 1
                                         else torch.cat(owner_parts, dim=1))
        return _sharding.Shards(recon_tensors, rp)

    def _full_indices(self):
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        return vcd_utils.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)

    def full_index_count(self):
        """How many pixels the ROR mask keeps, cached per (recon_shape,
        use_ror_mask).

        The memory ledger reads this once per candidate device layout, and
        rebuilding the mask each time would repeat a full-grid numpy pass per
        candidate.  Only the COUNT is cached here; the indices themselves have
        their own device-resident cache in :meth:`full_indices_device`."""
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        key = (tuple(recon_shape),
               use_ror_mask if isinstance(use_ror_mask, bool) else None)
        cache = getattr(self, '_full_index_count_cache', None)
        if key[1] is None or cache is None or cache[0] != key:
            count = int(np.shape(self._full_indices())[0])
            if key[1] is None:
                return count
            self._full_index_count_cache = (key, count)
        return self._full_index_count_cache[1]

    def full_indices_device(self):
        """The ROR-masked full pixel indices as an int64 tensor on the model
        device, cached per (recon_shape, use_ror_mask, device) -- rebuilding
        per call was a measured cost.  A custom mask array bypasses the cache
        (unhashable).  In-memory only; freed with the model."""
        recon_shape, use_ror_mask = self.get_params(['recon_shape', 'use_ror_mask'])
        key = (tuple(recon_shape), use_ror_mask if isinstance(use_ror_mask, bool) else None,
               str(self.torch_device))
        if key[1] is None:
            return torch.as_tensor(self._full_indices(), dtype=torch.int64,
                                   device=self.torch_device)
        cache = getattr(self, '_full_indices_cache', None)
        if cache is None or cache[0] != key:
            idx = torch.as_tensor(self._full_indices(), dtype=torch.int64,
                                  device=self.torch_device)
            self._full_indices_cache = (key, idx)
        return self._full_indices_cache[1]

    def refresh_device_bindings(self):
        """Recompile hook: rebuild the placements from the CURRENT params
        (preserving the configured devices), then recreate the projectors.
        Without this a geometry-changing set_params left the placements'
        axis lengths stale, silently truncating sharded arrays."""
        devices = self.sino_placement.devices
        sinogram_shape, recon_shape = self.get_params(
            ['sinogram_shape', 'recon_shape'])
        self.sino_placement = _sharding.Placement(
            devices, axis=0, axis_len=int(sinogram_shape[0]))
        self.recon_placement = _sharding.Placement(
            devices, axis=-1, axis_len=int(recon_shape[2]))
        self._check_no_empty_shard()
        self._invalidate_device_caches()
        if self._projector_functions is not None:
            self.create_projectors()

    def _check_no_empty_shard(self):
        """Raise an error for a device layout that would leave a device with no
        views and no slices.

        A device with views but no slices, or slices but no views, still has
        work to do and is allowed.  A device owns nothing on an axis only when
        the device count exceeds that axis length, so the rule is a device
        count above both the view count and the slice count."""
        sp, rp = self.sino_placement, self.recon_placement
        if sp.axis_len is None or rp.axis_len is None:
            return
        if sp.n_devices > sp.axis_len and sp.n_devices > rp.axis_len:
            raise ValueError(
                f'{sp.n_devices} devices would leave at least one device with '
                f'no views AND no slices ({sp.axis_len} views, '
                f'{rp.axis_len} slices); use at most '
                f'{max(sp.axis_len, rp.axis_len)} devices for this '
                f'geometry.')

    def _invalidate_device_caches(self):
        """Drop every cache that depends on the device layout or the geometry,
        so that no consumer reads out of date device state."""
        self.prox_data = None
        self._dc_damping_cache = None

    # ── device configuration ──────────────────────────────────────────────────
    def configure_devices(self, num_devices=1, devices=None, like=None):
        """
        Set the compute devices the model uses.

        Specify either a CUDA device count (``num_devices=n``), an explicit
        device list (``devices=['cpu']``, ``['mps']``, or
        ``['cuda:0', 'cuda:1']``), or another model to match
        (``like=other_model``).  With more than one device, the sinogram
        is divided across the devices by view and the reconstruction by
        slice.

        ``like=`` exists for a Plug-and-Play or ADMM loop, which alternates
        :meth:`prox_map` on a reconstruction model with
        :meth:`~mbirtorch.QGGMRFDenoiser.denoise` on a denoiser over the same
        volume.  Placing the two models on the same devices lets that volume
        pass between them in its device form (``output_sharded=True``),
        instead of being gathered to the host and scattered again on every
        half-iteration::

            denoiser = QGGMRFDenoiser(ct_model.get_params('recon_shape'))
            denoiser.configure_devices(like=ct_model)

        The one limit is worth stating plainly: this makes RECON-like arrays
        interchangeable, not sinogram-like ones.  A denoiser's sinogram IS its
        image, so its sinogram placement divides an image by slice, while a
        projection model's divides a sinogram by view; they are different
        things, and nothing exchanges sinograms with a denoiser anyway.

        Without a call to this method, the model chooses its devices
        automatically: it prefers cuda, then mps, then cpu, and on CUDA it
        may spread a reconstruction across several devices (see
        :meth:`recon`).  Calling this method turns the automatic choice off
        permanently for this model, so ``configure_devices(num_devices=1)``
        pins a run to one device for reproducibility.  The
        ``MBIRTORCH_NUM_DEVICES`` environment variable pins the count for a
        whole process.  Results can differ slightly with the device count,
        and the difference decays as iterations proceed.

        The device layout is built from the current sinogram and recon
        shapes, so call this after any geometry change.

        Call this function to set the device layout before any array is placed on the devices.

        Args:
            num_devices (int, optional): number of devices to use.  1 (the
                default) uses the model's default device (cuda, mps, or
                cpu); values above 1 require that many CUDA devices.
            devices (list, optional): explicit device list.  Overrides
                num_devices.
            like (TomographyModel, optional): another model (a geometry model
                or a ``QGGMRFDenoiser``) whose device list this model copies,
                so that recon-like arrays can pass between the two in their
                device form.  The two models must agree on their recon shape,
                which is what makes a volume from one usable by the other;
                they need not agree on their sinogram shapes, and a denoiser
                paired with a geometry model never does.  What is copied is the
                layout the other model has at this moment, so configure that
                model first: one whose layout is still automatic has not
                chosen yet -- it settles on its first reconstruction -- and
                the pair would then end up on different layouts.  ``like``
                and ``devices`` cannot both be given, and ``num_devices`` is
                ignored when ``like`` is: it has a default value, so an
                explicit ``num_devices=1`` cannot be told from the default.
        """
        if like is not None and devices is not None:
            raise ValueError(
                'configure_devices takes like= or devices=, not both: like= '
                "copies another model's device list, and devices= names one "
                'directly.  Pass whichever one expresses the intent.')
        self.device_layout_is_automatic = False
        # The rejected counts and the settled record belong to the automatic search.  An
        # explicit layout did not come from that search, so they are cleared here.
        self.device_choice_rejections = []
        self._settled_shapes = None
        self._settled_workload = None
        if like is not None:
            devices = self._devices_like(like)
        if devices is None:
            if num_devices == 1:
                devices = [self.torch_device]
            else:
                if not torch.cuda.is_available() or \
                        torch.cuda.device_count() < num_devices:
                    raise ValueError(
                        f"configure_devices({num_devices}) needs {num_devices} "
                        f"CUDA devices; found "
                        f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}.")
                devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
        self._install_device_layout(devices)

    def _devices_like(self, other):
        """Return the device list of ``other``, after checking that this model
        can exchange reconstruction arrays with it.

        The check catches a second model built at the wrong shape, such as a
        denoiser built at a CT model's sinogram shape instead of its
        reconstruction shape.
        """
        placement = getattr(other, 'recon_placement', None)
        other_get_params = getattr(other, 'get_params', None)
        if not isinstance(placement, _sharding.Placement) \
                or not getattr(placement, 'devices', None) \
                or other_get_params is None:
            raise ValueError(
                'configure_devices(like=...) copies the device list from '
                'another tomography model (a geometry model or a '
                f'QGGMRFDenoiser).  Got {type(other).__name__}, which has no '
                'device placement to copy.')
        own_shape = tuple(int(s) for s in self.get_params('recon_shape'))
        other_shape = tuple(int(s) for s in other_get_params('recon_shape'))
        if own_shape != other_shape:
            raise ValueError(
                'configure_devices(like=...) needs the two models to agree on '
                'the whole recon shape, because that is what makes a volume '
                'from one usable by the other.  The slice count alone decides '
                'how the shards are cut, but a mismatch in the rows or the '
                'columns would pass this check and then fail deep inside a '
                'reconstruction as an unreadable tensor error.  This '
                f'{type(self).__name__} has recon_shape {own_shape}, and the '
                f'{type(other).__name__} given as like= has recon_shape '
                f'{other_shape}, so recon-like arrays could not pass between '
                "them.  A QGGMRFDenoiser is built at the other model's RECON "
                "shape (ct_model.get_params('recon_shape')), not its sinogram "
                'shape.  To place two models on the same devices without '
                'exchanging arrays between them, pass devices=... instead.')
        return list(placement.devices)

    def _install_device_layout(self, devices):
        """Rebuild the placements over ``devices``.  Both
        :meth:`configure_devices` and the automatic choice call this.  It does
        not change ``device_layout_is_automatic``."""
        devices = [torch.device(d) for d in devices]
        self.torch_device = devices[0]
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        self.sino_placement = _sharding.Placement(
            devices, axis=0, axis_len=int(sinogram_shape[0]))
        self.recon_placement = _sharding.Placement(
            devices, axis=-1, axis_len=int(recon_shape[2]))
        self._check_no_empty_shard()
        # This probes whether a direct copy between devices is correct on this
        # hardware.  When it is not, transfers go through host memory.
        self.dev2dev_safe = _sharding.is_dev2dev_safe(devices)
        self._invalidate_device_caches()
        if self._projector_functions is not None:
            self.create_projectors()

    # ── the memory ledger and the automatic device count ──────────────────────
    def _build_memory_ledger(self, devices=None, workload='recon',
                             **call_arrays):
        """Return the modeled peak memory per device for one candidate device
        list.  A value of None for ``devices`` means the current placement.
        The argument ``workload`` names the call, either 'recon' or 'direct'.
        """

        devices = list(self.sino_placement.devices if devices is None
                       else devices)
        return _memory_ledger.estimate_peak_device_bytes(
            _memory_ledger.plan_from_model(self, devices, workload=workload,
                                           **call_arrays))

    def _candidate_devices(self, num_devices):
        return [torch.device(f'cuda:{i}') for i in range(num_devices)]

    def _shape_pair(self):
        """Return the (sinogram_shape, recon_shape) pair.  The automatic policy
        records this pair when it settles and compares it on every later call.
        The memory ledger's plan is built from these two shapes."""
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return tuple(sinogram_shape), tuple(recon_shape)

    def _apply_device_policy(self, workload='recon', **call_arrays):
        """Settle the device layout for the reconstruction about to run, and
        return the ledger for the layout settled on.

        ``workload`` names the call in progress.  It is ``'recon'`` for a full
        reconstruction, ``'direct'`` for a direct reconstruction, and
        ``'denoise'`` for one QGGMRFDenoiser sweep.  It tells the ledger what
        is about to be allocated.

        This is the only place where the automatic device count is chosen.  The
        choice happens when a reconstruction starts, because the free memory
        reading is current only then.  The choice is made once per model, and
        the search runs again only when the sinogram or reconstruction shape
        changes.  The widening speed floors set the order in which candidate
        counts are tried, and capacity wins when no admitted count fits.
        """
        calibrating = _memory_ledger.calibration_enabled()
        # A candidate layout is sized for the largest workload this model may ever run.  That is a
        # full reconstruction, except on a QGGMRFDenoiser, which can only denoise.
        sizing = 'denoise' if workload == 'denoise' else 'recon'
        if not self.device_layout_is_automatic:
            # The ledger runs on an explicit layout only when calibration asks
            # for it.
            ledger = self._build_memory_ledger(workload=sizing,
                                               **call_arrays) \
                if calibrating else None
            return self._arm_calibration(ledger, sizing)

        if self._settled_shapes is not None:
            if self._settled_shapes == self._shape_pair():
                if _memory_ledger.workload_covers(self._settled_workload,
                                                  workload):
                    # The settled check already covers this call, so the layout
                    # is reused without a search.
                    ledger = self._build_memory_ledger(workload=sizing,
                                                       **call_arrays) \
                        if calibrating else None
                    return self._arm_calibration(ledger, sizing)
                # This call allocates more than the settled check priced, so the check runs
                # again on the settled layout, which does not move.
                ledger = self._check_settled_capacity(workload, call_arrays)
                # The check passed, so the record now names the workload the layout is known
                # to hold, and a later call of the same kind repeats no check.
                self._settled_workload = workload
                return self._arm_calibration(ledger, sizing)
            # The shapes changed, so the settled decision is dropped and the
            # layout is decided again below.
            self._settled_shapes = None
            self._settled_workload = None

        pinned = _memory_ledger.pinned_device_count()
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible < 2:
            # There is no layout to choose.  The allocator's own error covers
            # an overflow on a single device.
            return self._arm_calibration(None, sizing)

        if pinned is not None:
            # A pin that applies to the whole process is as explicit as a
            # configure_devices call, so the count is used as given.
            candidates, held = [min(pinned, visible)], {}
        else:
            candidates, held = self._speed_ordered_candidates(visible)

        rejected, best = [], None
        self._speed_floor_fallback = None
        self._speed_floor_held = held
        for count in candidates:
            if count in held:
                # Admitted counts come first, so reaching a held count means
                # capacity is about to override the speed rule.
                held_note, taken_note = held[count]
                rejected.append((count, held_note))
                self._speed_floor_fallback = (count, taken_note)
            devices = (self._candidate_devices(count) if count > 1
                       else [self.torch_device])
            if not self._layout_is_valid(devices):
                rejected.append((count, 'a device would own no real data'))
                continue
            # The price is taken at the sizing workload, whatever this call is, because the
            # count chosen here has to suit the largest workload the model may later run.
            ledger = self._build_memory_ledger(devices=devices,
                                               workload=sizing, **call_arrays)
            if ledger is None or self.skip_memory_preflight:
                # There is nothing to check against, or the caller has forced
                # the run.
                return self._settle(devices, ledger, rejected, sizing)
            fits, rows = self._layout_capacity(devices, ledger, call_arrays)
            if fits:
                return self._settle(devices, ledger, rejected, sizing)
            shortfall = max((d - b) for _dev, d, b in rows if b is not None)
            rejected.append((count, f'{shortfall / 2 ** 30:.2f} GB short'))
            if best is None or shortfall < best[0]:
                best = (shortfall, ledger, rows, count)

        if workload != sizing and best is not None:
            # No count fits the sizing workload, and this call is not running one.  The candidates
            # are tried again in the same order, priced against the call in progress instead.
            best = None
            for count in candidates:
                devices = (self._candidate_devices(count) if count > 1
                           else [self.torch_device])
                if not self._layout_is_valid(devices):
                    continue
                ledger = self._build_memory_ledger(
                    devices=devices, workload=workload, **call_arrays)
                fits, rows = self._layout_capacity(devices, ledger,
                                                   call_arrays)
                if fits:
                    # The first pass recorded this count as refused, priced for a
                    # reconstruction this call is not running.  That record is removed.
                    self._speed_floor_fallback = None
                    rejected = [(c, why) for c, why in rejected
                                if c != count]
                    rejected.append(
                        (count, f'chosen for the {workload} reconstruction '
                                'in progress: no device count fits a full '
                                'recon at this size'))
                    return self._settle(devices, ledger, rejected, workload)
                shortfall = max((d - b) for _dev, d, b in rows if b is not None)
                if best is None or shortfall < best[0]:
                    best = (shortfall, ledger, rows, count)

        # No device count fits, including a single device.  The call fails here
        # rather than start a reconstruction that is known not to fit.
        if best is None:
            raise _memory_ledger.MemoryPreflightError(
                'no device layout is valid for this geometry: '
                + '; '.join(f'{c} devices ({why})' for c, why in rejected))
        _shortfall, ledger, rows, count = best
        raise _memory_ledger.MemoryPreflightError(
            _memory_ledger.format_shortfall(
                ledger, rows, num_devices_tried=candidates,
                closest_count=count, remedies=self._memory_remedies()))

    def _speed_ordered_candidates(self, visible):
        """Return the order in which to try candidate device counts, along with
        the notes for the counts the widening speed floors hold back.

        The floors reorder the counts and never remove one.  Admitted counts
        come first, largest first, then the held counts, also largest first.
        Capacity therefore always wins.

        Returns:
            (list, dict): candidate counts in the order to try them, and
            ``{count: (held_note, taken_note)}`` for the held ones.
        """
        candidates = list(range(visible, 0, -1))
        if not _widening_floors.guard_enabled():
            return candidates, {}
        elements = _widening_floors.sinogram_elements(
            self.get_params('sinogram_shape'))
        family = self._floor_family
        note = _widening_floors.stale_note()
        if note is not None:
            self.logger.info('Note: ' + note + '.')
        if family is None and self.get_params('verbose') >= 2:
            self.logger.debug(
                f'  {type(self).__name__} names no _floor_family, so the '
                f'{_widening_floors.DEFAULT_FAMILY} widening speed floors '
                f'apply to its automatic device count.')
        admitted, held = [], {}
        for count in candidates:
            ok, why = _widening_floors.admitted(family, count, elements)
            if ok:
                admitted.append(count)
            else:
                held[count] = (why, _widening_floors.fallback_reason(
                    family, count, elements))
        # A count of 1 is always admitted, so the admitted list is never empty.
        return admitted + list(held), held

    def _memory_remedies(self):
        """Extra remedy lines for this geometry's preflight message."""
        if type(self).recon_split_sino is not TomographyModel.recon_split_sino:
            return ['  model.recon_split_sino(...)                '
                    '# reconstructs one row band at a time; raises',
                    '                                             '
                    '# the feasible size at a fixed device count']
        return []

    def _layout_is_valid(self, devices):
        """Return True when ``devices`` passes the rule that
        :meth:`_check_no_empty_shard` applies.  Nothing is changed."""
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return len(devices) <= max(int(sinogram_shape[0]), int(recon_shape[2]))

    def _layout_capacity(self, devices, ledger, call_arrays):
        """Return whether the modeled peak in ``ledger`` fits ``devices``,
        along with the rows used to report it.  This is the only place where a
        memory budget is compared with a modeled demand."""
        budgets = [_memory_ledger.device_budget_bytes(d) for d in devices]
        credits = _memory_ledger.resident_credits(
            devices, list(call_arrays.values()))
        return _memory_ledger.layout_fits(
            ledger, budgets, credits, margin=self.memory_preflight_margin)

    def _fits_available_devices(self, workload='recon', **call_arrays):
        """Return whether the modeled peak for ``workload`` fits any device
        layout this model could run on.  Nothing is changed and no memory is
        allocated, so a caller can price a model it is about to discard.

        The candidate device lists are the ones the policy would try.  The
        order does not matter to a yes or no answer, so the widening speed
        floors are not consulted.  A device with no readable memory budget,
        meaning anything other than CUDA, is treated as holding whatever it is
        asked to hold.

        Returns:
            bool: True when some candidate layout holds the modeled peak.
        """
        if self.skip_memory_preflight:
            return True
        if not self.device_layout_is_automatic:
            candidates = [list(self.sino_placement.devices)]
        else:
            pinned = _memory_ledger.pinned_device_count()
            visible = (torch.cuda.device_count() if torch.cuda.is_available()
                       else 0)
            if visible < 2:
                counts = [1]
            elif pinned is not None:
                counts = [min(pinned, visible)]
            else:
                counts = list(range(visible, 0, -1))
            candidates = [self._candidate_devices(count) if count > 1
                          else [self.torch_device] for count in counts]
        for devices in candidates:
            if not self._layout_is_valid(devices):
                continue
            ledger = self._build_memory_ledger(devices=devices,
                                               workload=workload,
                                               **call_arrays)
            if ledger is None:
                return True
            fits, _rows = self._layout_capacity(devices, ledger, call_arrays)
            if fits:
                return True
        return False

    def _check_settled_capacity(self, workload, call_arrays):
        """Run the capacity check for ``workload`` on the settled layout, and
        return the ledger it priced.

        The layout does not change, because a caller may be holding shards of
        it.  This check can refuse the call.  Without it, a model that settled
        under a direct reconstruction would reach the allocator on a later full
        reconstruction with no preflight message or remedies.
        """
        if self.skip_memory_preflight:
            # The caller has forced the run.
            return None
        devices = list(self.sino_placement.devices)
        ledger = self._build_memory_ledger(devices=devices, workload=workload,
                                           **call_arrays)
        fits, rows = self._layout_capacity(devices, ledger, call_arrays)
        if fits:
            return ledger
        raise _memory_ledger.MemoryPreflightError(
            _memory_ledger.format_shortfall(
                ledger, rows, num_devices_tried=[len(devices)],
                closest_count=len(devices),
                remedies=self._memory_remedies()))

    def _settle(self, devices, ledger, rejected, workload='recon'):
        """Install the chosen layout when it differs from the current one, log
        the choice, and enable the calibration mode.

        ``workload`` is the plan ``ledger`` was priced with, which is the
        workload the settled layout has been checked for."""
        chosen, current = len(devices), self.sino_placement.n_devices
        # When the search settled on a held count, capacity overrode the speed
        # floor, so the note recorded for that count is replaced.
        fallback = getattr(self, '_speed_floor_fallback', None)
        if fallback is not None and fallback[0] == chosen:
            rejected = [fallback if count == chosen else (count, why)
                        for count, why in rejected]
        # Every larger count the floors held back is named in the log, so that
        # idle GPUs are explained.
        held = getattr(self, '_speed_floor_held', None) or {}
        rejected = list(rejected)
        already = {count for count, _why in rejected}
        for count in sorted(held, reverse=True):
            if count > chosen and count not in already:
                rejected.append((count, held[count][0]))
        self._speed_floor_fallback = None
        self._speed_floor_held = None
        # The run log's device line reads this to explain any idle GPUs.
        self.device_choice_rejections = list(rejected)
        if chosen != current:
            self.logger.info(
                f'Using {chosen} CUDA device(s) for this reconstruction '
                f'(was {current}).  configure_devices(num_devices=n) pins it.')
            self._install_device_layout(devices)
        if rejected and self.get_params('verbose') >= 2:
            for count, why in rejected:
                self.logger.debug(f'  device count {count} rejected: {why}')
        # These are the shapes this decision came from.  While they hold, later policy calls
        # reuse the layout.  The workload recorded beside them is what the capacity check used.
        self._settled_shapes = self._shape_pair()
        self._settled_workload = workload
        return self._arm_calibration(ledger, workload)

    def _arm_calibration(self, ledger, workload='recon'):
        """Record the ledger for a test harness to read.  Under the calibration
        mode, build one when the caller had none, so that a policy return
        always carries a ledger.  The argument ``workload`` names the plan to
        build that missing ledger with."""
        if ledger is not None:
            self.last_memory_ledger = ledger
        if _memory_ledger.calibration_enabled() and ledger is None:
            ledger = self._build_memory_ledger(workload=workload)
            self.last_memory_ledger = ledger
        return ledger

    # Every sinogram array is placed by _shard_sinogram and every reconstruction array by
    # _shard_recon.  The matching gather functions bring them back to the host.
    def _shard_sinogram(self, sinogram):
        """Place a sinogram or a weights array in its device form.  The result
        is float32, its view axis is checked, and on a multi-device placement
        it is sharded by view."""
        num_views = self.get_params('sinogram_shape')[0]
        if isinstance(sinogram, _sharding.Shards):
            if sinogram.placement != self.sino_placement:
                raise ValueError(
                    'Sinogram shards belong to a different device '
                    'configuration: the shards are placed as '
                    f'{sinogram.placement}, and this model uses '
                    f'{self.sino_placement}; re-place the array.')
            return sinogram
        if self.sino_placement.is_trivial:
            sinogram = torch.as_tensor(sinogram, dtype=torch.float32,
                                       device=self.torch_device)
            if sinogram.shape[0] != num_views:
                raise ValueError(
                    'Cannot place the sinogram: its view axis has size '
                    f'{sinogram.shape[0]}, but the model expects {num_views} views.')
            return sinogram
        return self._split_to_shards(sinogram, self.sino_placement, num_views,
                                     what='sinogram (view axis)')

    # This is True when detector row r corresponds to reconstruction slice r, as it does in
    # parallel beam.  The base value is False, so a geometry must declare the row aligned path.
    rows_track_slices = False

    # This is the fewest pixels the compiled projection bodies may be called
    # with.  Narrower calls are padded outside the compiled region.
    min_compiled_pixel_width = 1

    # This names the set of measured widening floors that governs the automatic
    # device count.  None means the parallel beam floors.
    _floor_family = None

    def prepare_sino_for_devices(self, sinogram, weights=None):
        """Place a sinogram (and optionally weights) in the model's device
        form, once.

        The device form is the layout the reconstruction methods use
        internally: the sinogram is divided across the configured devices by
        view.

        Calling this is OPTIONAL: every reconstruction method applies the
        same placement automatically to a plain input.  Use this function to
        transfer just once when running several reconstructions on
        the same large sinogram.  What it returns goes straight into
        :meth:`recon` and :meth:`prox_map` in place of the sinogram (and the
        weights), so those calls do no transfer of their own.
        If the device configuration changes afterwards, the prepared array no
        longer matches, and the reconstruction methods raise an error; re-run
        this method to fix it.

        On a model whose device layout is still automatic, this call also
        decides the layout, and every later reconstruction on the model
        reuses it.  The layout is sized for a full reconstruction whenever one
        fits.  On a problem too large for any full reconstruction, the memory
        check falls back to what this call itself allocates, which is much
        smaller, the way the direct reconstructions do; preparing a sinogram
        then succeeds where a full reconstruction could not run.  A later
        :meth:`recon` on such a layout runs the memory check again and raises
        ``MemoryPreflightError``, rather than reusing a layout that was never
        checked for it.

        Args:
            sinogram (numpy or tensor): sinogram in the model's sinogram_shape.
            weights (numpy or tensor, optional): weights of the same shape.

        Returns:
            The prepared sinogram, or a (sinogram, weights) tuple when weights
            were given.
        """
        # The layout is settled before the sinogram is placed.  Placing first would put the
        # whole sinogram on the lead device and then move it again.
        self._apply_device_policy(workload='direct')
        sino = self._shard_sinogram(sinogram)
        if weights is None:
            return sino
        return sino, self._shard_sinogram(weights)

    def _shard_recon(self, recon):
        """Place a reconstruction array in its device form.  The array is
        either three dimensional or flat with shape (num_pixels, num_slices).
        The result is float32, its slice axis is checked, and on a
        multi-device placement it is sharded by slice."""
        num_slices = self.get_params('recon_shape')[2]
        if isinstance(recon, _sharding.Shards):
            # Placements compare by value, so shards made by another model on the same
            # devices with the same slice count are accepted.  No check of the whole volume shape
            # belongs here, because sparse_forward_project sends subsets of the pixels through it.
            if recon.placement != self.recon_placement:
                raise ValueError(
                    'Recon shards belong to a different device '
                    'configuration: the shards are placed as '
                    f'{recon.placement}, and this model uses '
                    f'{self.recon_placement}; re-place the array.')
            return recon
        if self.recon_placement.is_trivial:
            recon = torch.as_tensor(recon, dtype=torch.float32,
                                    device=self.torch_device)
            if recon.shape[-1] != num_slices:
                raise ValueError(
                    'Cannot place the reconstruction: its slice axis has size '
                    f'{recon.shape[-1]}, but the model expects {num_slices} slices.')
            return recon
        return self._split_to_shards(recon, self.recon_placement, num_slices,
                                     what='reconstruction (slice axis)')

    def _split_to_shards(self, x, placement, axis_len, what='array'):
        """Split an array into one shard tensor per device.  Each device gets a
        contiguous block of the sharded axis.  The blocks differ in length by
        at most one.  A device count above the axis length leaves the trailing
        devices with empty blocks."""
        x = torch.as_tensor(x, dtype=torch.float32)
        axis = placement.axis % x.ndim
        if x.shape[axis] != axis_len:
            raise ValueError(
                f'Cannot place the {what}: got shape {tuple(x.shape)}, '
                f'but the model expects size {axis_len} on axis {axis}.')
        tensors = []
        for dev, (start, end) in placement.shard_ranges(axis_len):
            idx = [slice(None)] * x.ndim
            idx[axis] = slice(start, end)
            tensors.append(x[tuple(idx)].to(dev))
        return _sharding.Shards(tensors, placement)

    def _gather_sinogram(self, sinogram):
        """Return a sinogram-like array as a host numpy array, with the
        shards concatenated on the view axis."""
        if isinstance(sinogram, _sharding.Shards):
            out = self._gather_shards(sinogram)
        else:
            out = sinogram.detach().cpu().numpy()
        return out

    def _gather_recon(self, recon):
        """Return a recon-like array as a host numpy array, with the shards
        concatenated on the slice axis."""
        if isinstance(recon, _sharding.Shards):
            out = self._gather_shards(recon)
        else:
            out = recon.detach().cpu().numpy()
        return out

    def _constant_recon(self, value):
        """Return a reconstruction of constant value in the device form.  It is
        built one shard at a time, so no full volume lands on one device."""
        recon_shape = self.get_params('recon_shape')
        if self.recon_placement.is_trivial:
            recon = torch.full(tuple(recon_shape), float(value),
                               dtype=torch.float32, device=self.torch_device)
        else:
            tensors = [
                torch.full(tuple(recon_shape[:2]) + (e - s,), float(value),
                           dtype=torch.float32, device=d)
                for d, (s, e) in self.recon_placement.shard_ranges()]
            recon = _sharding.Shards(tensors, self.recon_placement)
        return recon

    def _initial_error_state(self, sinogram, init_recon, weights,
                             constant_weights, scale_recon_to_sinogram):
        """Return the initial (error_sinogram, init_recon) pair.  The initial
        reconstruction is forward projected, the optimal scale alpha is found,
        and both arrays are scaled by it."""
        self.logger.info('Initializing error sinogram')
        fwd = self.forward_project(init_recon, output_sharded=True)
        if isinstance(fwd, _sharding.Shards):
            def dots_worker(i, d):
                # Both sums are reduced one block of views at a time, so the shard never
                # holds a weighted projection or a full array of products.
                f = fwd.tensors[i]
                w = None if constant_weights else weights.tensors[i]
                return (float(_memory_ledger.weighted_square_sum(f, w)),
                        float(_memory_ledger.weighted_dot(
                            f, sinogram.tensors[i], w)))
            dots = _sharding.run_per_device(self.sino_placement.devices,
                                            dots_worker)
            wtd_err_sino_norm = sum(a for a, _ in dots)
            if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
                alpha = sum(b for _, b in dots) / wtd_err_sino_norm
            else:
                alpha = 1
            # The error is formed in the projection's own shards.  Scaling a shard by minus alpha
            # and adding the sinogram gives sinogram - alpha * fwd, and allocates nothing.
            error_sinogram = _sharding.Shards(
                _sharding.run_per_device(
                    self.sino_placement.devices,
                    lambda i, d: fwd.tensors[i].mul_(-alpha).add_(
                        sinogram.tensors[i])),
                self.sino_placement)
            # The projection's shards now hold the error sinogram, so dropping
            # this name releases only the container.
            fwd = None
            init_recon = _sharding.Shards(
                [alpha * t for t in init_recon.tensors], self.recon_placement)
        else:
            # The reduction runs one block of views at a time, as in the
            # sharded branch above.
            w = None if constant_weights else weights
            wtd_err_sino_norm = _memory_ledger.weighted_square_sum(fwd, w)
            if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
                alpha = (_memory_ledger.weighted_dot(fwd, sinogram, w)
                         / wtd_err_sino_norm).item()
            else:
                alpha = 1
            # The error is formed in the projection's own buffer, as in the
            # sharded branch above.
            error_sinogram = fwd.mul_(-alpha).add_(sinogram)
            fwd = None
            init_recon = alpha * init_recon
        return error_sinogram, init_recon

    def _flatten_recon(self, recon):
        """Return the reconstruction in the VCD loop's flat
        (num_pixels, slices) layout, made contiguous for the in place row
        updates."""
        if isinstance(recon, _sharding.Shards):
            # The row count is named rather than inferred, because reshape
            # cannot infer it from a shard that owns no slices.
            flat = _sharding.Shards(
                [t.reshape((math.prod(t.shape[:-1]), t.shape[-1])).contiguous()
                 for t in recon.tensors], recon.placement)
        else:
            flat = self._shard_recon(
                recon.reshape((-1, recon.shape[-1]))).contiguous()
        return flat

    def _flatten_prox_shards(self, prox_input, recon_shape):
        """Bring a prox input that is already in the device form into the VCD
        loop's flat (num_pixels, local_slices) layout, without going through
        host memory.

        The shards together must cover the full pixel grid and the full slice
        axis, and that is checked here.  Each shard is either three dimensional
        with shape (rows, cols, local_slices) or already flat with shape
        (num_pixels, local_slices).  A shard that owns no slices is allowed.
        """
        rows, cols, num_slices = (int(recon_shape[0]), int(recon_shape[1]),
                                  int(recon_shape[2]))
        num_pixels = rows * cols
        tensors = prox_input.tensors
        covers_grid = True
        total_slices = 0
        for t in tensors:
            if t.ndim == 3:
                pixels = int(t.shape[0]) * int(t.shape[1])
            elif t.ndim == 2:
                pixels = int(t.shape[0])
            else:
                covers_grid = False
                break
            total_slices += int(t.shape[-1])
            covers_grid = covers_grid and pixels == num_pixels
        if not covers_grid or total_slices != num_slices:
            raise ValueError(
                'prox_input does not have the correct size. \n'
                f'Expected shards covering {tuple(recon_shape)}: each shard '
                f'({rows}, {cols}, local_slices) or ({num_pixels}, '
                'local_slices), with the local slice counts summing to '
                f'{num_slices}.  Got shapes '
                f'{[tuple(t.shape) for t in tensors]} for prox_input.')
        # The result goes through _shard_recon so that the placement check runs, and shards from
        # a model on a different device layout are refused here rather than later.
        return self._shard_recon(_sharding.Shards(
            [t.reshape(num_pixels, t.shape[-1]) for t in tensors],
            prox_input.placement))

    def _check_sinogram_shards(self, sinogram, sinogram_shape):
        """Check that a sinogram already in the device form covers the model's
        whole sinogram.

        Each shard must hold a block of views together with every detector row
        and channel, and the view counts must add up to the model's view count.
        A shard that owns no views is allowed.  Only the shapes are checked
        here, because :meth:`_shard_sinogram` checks the devices.
        """
        views, rows, channels = (int(sinogram_shape[0]), int(sinogram_shape[1]),
                                 int(sinogram_shape[2]))
        tensors = sinogram.tensors
        total_views = 0
        detector_matches = True
        for tensor in tensors:
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (rows, channels):
                detector_matches = False
                break
            total_views += int(tensor.shape[0])
        if not detector_matches or total_views != views:
            raise ValueError(
                'sinogram does not have the shape in sinogram_shape. \n'
                f'Expected shards covering {(views, rows, channels)}: each '
                f'shard (local_views, {rows}, {channels}), with the local view '
                f'counts summing to {views}.  Got shapes '
                f'{[tuple(t.shape) for t in tensors]}.')

    def _flatten_hessian(self, fm_hessian):
        """Return the Hessian diagonal in the VCD loop's flat layout.  The loop
        only reads it, so it is not made contiguous."""
        if isinstance(fm_hessian, _sharding.Shards):
            flat = _sharding.Shards(
                [t.reshape((math.prod(t.shape[:-1]), t.shape[-1]))
                 for t in fm_hessian.tensors], fm_hessian.placement)
        else:
            flat = fm_hessian.reshape((-1, fm_hessian.shape[-1]))
        return flat

    def _recon_from_flat(self, flat_recon, recon_shape):
        """Return the three dimensional reconstruction from the VCD loop's flat
        layout.  Each array keeps its own slice count."""
        if isinstance(flat_recon, _sharding.Shards):
            recon = _sharding.Shards(
                [t.reshape(tuple(recon_shape[:2]) + (t.shape[-1],))
                 for t in flat_recon.tensors], flat_recon.placement)
        else:
            recon = flat_recon.reshape(tuple(recon_shape[:2])
                                       + (flat_recon.shape[-1],))
        return recon

    def _gather_shards(self, shards):
        return shards.gather()

    def _as_shards(self, x, placement):
        """Return a device form array as a Shards container.  A plain tensor is
        wrapped in a one shard container that aliases it, so in place updates
        reach the caller's array.  A Shards container passes through.  Nothing
        is validated here."""
        if isinstance(x, _sharding.Shards):
            shards = x
        else:
            shards = _sharding.Shards([x], placement)
        return shards

    def _as_device_form(self, x):
        """Invert :meth:`_as_shards`.  A one shard container unwraps to the
        tensor it aliases.  A container that really spans devices is returned
        unchanged, because collapsing it would require a gather."""
        if isinstance(x, _sharding.Shards) and x.placement.is_trivial:
            out = x.tensors[0]
        else:
            out = x
        return out

    def _sino_ones_device_form(self, sino_like=None):
        """Return a sinogram of all ones in the device form, with one block per
        device.  ``sino_like`` supplies only the dtype."""
        dtype = torch.float32 if sino_like is None else sino_like.dtype
        if self.sino_placement.is_trivial:
            return torch.ones(tuple(self.get_params('sinogram_shape')),
                              dtype=dtype, device=self.torch_device)
        shape = list(self.get_params('sinogram_shape'))
        tensors = []
        for dev, (start, end) in self.sino_placement.shard_ranges():
            local = list(shape)
            local[0] = end - start
            tensors.append(torch.ones(local, dtype=dtype, device=dev))
        return _sharding.Shards(tensors, self.sino_placement)

    def forward_project(self, recon, output_sharded=False):
        """
        Perform a full forward projection.  With the ``use_ror_mask``
        parameter True (the default) the projection covers the pixels inside
        the region-of-reconstruction mask; with it False, every pixel.

        Args:
            recon (numpy or tensor): 3D volume with shape
                (num_recon_rows, num_recon_cols, num_recon_slices).
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the device form: a torch
                tensor on a single device, or a Shards container (one
                tensor per device) on a multi-device model.

        Returns:
            The sinogram, shape (num_views, num_det_rows, num_det_channels).
        """
        recon_shape = self.get_params('recon_shape')
        recon = self._shard_recon(recon)
        indices = self.full_indices_device()
        if isinstance(recon, _sharding.Shards):
            # The row count is named rather than inferred, because reshape
            # cannot infer it from a shard that owns no slices.
            num_pixels = int(recon_shape[0]) * int(recon_shape[1])
            flat = _sharding.Shards(
                [t.reshape(num_pixels, t.shape[-1])[indices.to(t.device)]
                 for t in recon.tensors], recon.placement)
            sinogram = self.sparse_forward_project(flat, indices)
        else:
            voxel_values = recon.reshape(-1, recon.shape[-1])[indices]
            sinogram = self.sparse_forward_project(voxel_values, indices)
        return sinogram if output_sharded else self._gather_sinogram(sinogram)

    def back_project(self, sinogram, output_sharded=False):
        """
        Perform a full back projection.  With the ``use_ror_mask`` parameter
        True (the default) the result is zero outside the
        region-of-reconstruction mask; with it False, every pixel is
        computed.

        Args:
            sinogram (numpy or tensor): 3D array with shape
                (num_views, num_det_rows, num_det_channels).
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the device form: a torch
                tensor on a single device, or a Shards container (one
                tensor per device) on a multi-device model.

        Returns:
            The back projection, shape (num_recon_rows, num_recon_cols,
            num_recon_slices).
        """
        recon_shape = self.get_params('recon_shape')
        sinogram = self._shard_sinogram(sinogram)
        indices = self.full_indices_device()
        cylinders = self.sparse_back_project(sinogram, indices)
        if isinstance(cylinders, _sharding.Shards):
            def scatter_worker(i, d):
                c = cylinders.tensors[i]
                r = torch.zeros((recon_shape[0] * recon_shape[1], c.shape[-1]),
                                dtype=torch.float32, device=d)
                r[indices.to(d)] = c
                return r.reshape(tuple(recon_shape[:2]) + (c.shape[-1],))
            recon = _sharding.Shards(
                _sharding.run_per_device(cylinders.placement.devices,
                                         scatter_worker), cylinders.placement)
        else:
            recon = torch.zeros((recon_shape[0] * recon_shape[1], cylinders.shape[-1]),
                                dtype=torch.float32, device=self.torch_device)
            recon[indices] = cylinders
            recon = recon.reshape(tuple(recon_shape[:2]) + (cylinders.shape[-1],))
        return recon if output_sharded else self._gather_recon(recon)

    def project_points(self, points_xyz, view_index):
        """Map object points to fractional detector indices, for one view or several.

        This is the geometric map the projectors implement, stated for points
        instead of voxels.  Each geometry class computes it from the same
        functions its projection bodies use, so the answer here is the answer
        the projector gives, up to the projector's float32 rounding and its
        footprint weights.

        The object frame is the right-handed (x, y, z) frame in which voxel
        (i, j, k) of the reconstruction has its center at

            x = delta_voxel * (j - (num_cols - 1) / 2)
            y = voxel_row_aspect * delta_voxel * (i - (num_rows - 1) / 2)
            z = voxel_slice_aspect * delta_voxel * (k - (num_slices - 1) / 2) + recon_slice_offset

        so the column index runs along x, the row index along y, and the slice
        index along z, the rotation axis.  ``recon_slice_offset`` is zero for a
        geometry that has no such parameter.  Points are given in this frame,
        before the view's own action on the object: a view rotates the object
        about z by its angle (parallel, cone, and multiaxis), shifts it by its
        helical z shift (cone), or translates it by minus its translation
        vector (translation).  This method applies that action itself.

        The result is a pair of fractional indices into a sinogram of shape
        ``(num_views, num_det_rows, num_det_channels)``.  Integer index m is
        the center of detector row m, and index -0.5 is the outer edge of row
        0.  A point outside the detector gets an index outside
        ``[-0.5, num - 0.5]``; nothing is clipped.

        Args:
            points_xyz (array_like): the points, (N, 3) as (x, y, z) in ALU,
                or one point as (3,).
            view_index (int or sequence of int): one view, or several.

        Returns:
            tuple of ndarray: ``(row, channel)`` as float64 arrays.  For one
            view each has shape (N,); for a sequence of views each has shape
            (num_selected_views, N).

        Raises:
            ValueError: if ``points_xyz`` is not (N, 3) or (3,), or if a view
                index is not an integer.
            IndexError: if a view index is outside ``[0, num_views)``.

        Example:
            >>> row, channel = model.project_points([[0.0, 0.0, 0.0]], 0)
        """
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim == 1 and points.shape == (3,):
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must have shape (N, 3) or (3,); '
                             f'got {points.shape}.')
        single_view = np.ndim(view_index) == 0
        indices = np.atleast_1d(np.asarray(view_index))
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('view_index must be an integer or a sequence of '
                             f'integers; got {view_index!r}.')
        num_views = int(self.get_params('sinogram_shape')[0])
        if indices.size and (indices.min() < 0 or indices.max() >= num_views):
            raise IndexError(f'view_index {view_index!r} is outside '
                             f'[0, {num_views}).')
        view_params = np.asarray(self.get_params(self.get_params('view_params_name')))[indices]
        # The computation runs in float64 on the host, because a device
        # transfer would cost more than this small amount of arithmetic.
        row, channel = self._project_points_batch(
            torch.as_tensor(points, dtype=torch.float64),
            torch.as_tensor(view_params, dtype=torch.float64))
        row = np.ascontiguousarray(row.numpy())
        channel = np.ascontiguousarray(channel.numpy())
        if single_view:
            return row[0], channel[0]
        return row, channel

    def _project_points_batch(self, points, view_params):
        """The geometry's own part of :meth:`project_points`.

        Args:
            points: (N, 3) float64 CPU tensor of object-frame points.
            view_params: the selected rows of the model's per-view parameter
                array, as a float64 CPU tensor: (V,) angles for parallel beam,
                (V, 2) for cone and multiaxis, (V, 3) for translation.

        Returns:
            (row, channel): float64 CPU tensors, each (V, N).
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not implement project_points.')

    def compute_hessian_diagonal(self, weights=None, output_sharded=False,
                                 indices=None):
        """
        Compute the diagonal of the Hessian matrix: a back projection of the
        weights using squared coefficients.

        Args:
            weights (numpy or tensor, optional): 3D positive weights with the
                same shape as the sinogram.  Defaults to all 1s.
            output_sharded (bool, optional): If False (default), return numpy;
                if True, return the device tensor.
            indices (tensor, optional): back-project at these flat pixel
                indices only, leaving entries outside the set ZERO.  None
                (the default) covers all pixels of the grid.

        Returns:
            Diagonal of the Hessian matrix with the same shape as the recon.

        Note:
            On a model whose device layout is still automatic, this call also
            decides the layout, and every later reconstruction on the model
            reuses it.  The layout is sized for a full reconstruction whenever
            one fits.  On a problem too large for any full reconstruction, the
            memory check falls back to what this call itself allocates, which
            is much smaller, the way the direct reconstructions do; the
            Hessian diagonal can then be computed where a full reconstruction
            could not run.  A later :meth:`recon` on such a layout runs the
            memory check again and raises ``MemoryPreflightError``, rather
            than reusing a layout that was never checked for it.
        """
        # The layout is settled before the full array of weights and the full volume are built.
        # Otherwise they would land whole on the lead device.
        self._apply_device_policy(workload='direct')
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        if weights is None:
            weights = self._sino_ones_device_form()
        elif (not isinstance(weights, _sharding.Shards)
              and tuple(weights.shape) != tuple(sinogram_shape)):
            raise ValueError('Weights must be constant or an array compatible with sinogram'
                             f'\nGot weights.shape = {tuple(weights.shape)}, but '
                             f'sinogram.shape = {tuple(sinogram_shape)}')
        else:
            weights = self._shard_sinogram(weights)
        num_grid = int(recon_shape[0] * recon_shape[1])
        dense = indices is None
        if dense:
            indices = torch.arange(num_grid, dtype=torch.int64,
                                   device=self.torch_device)
        else:
            indices = torch.as_tensor(indices, dtype=torch.int64,
                                      device=self.torch_device)
        hessian = self.sparse_back_project(weights, indices, coeff_power=2)

        # A dense back projection is already the flat volume, so it is reshaped.  A masked one
        # returns only its own rows, so it is scattered into a volume of zeros.  An explicit
        # index set is always scattered, because a reordered set is not the identity permutation.
        def to_volume(cylinders, device):
            if dense:
                return cylinders.reshape((recon_shape[0], recon_shape[1],
                                          cylinders.shape[-1]))
            volume = torch.zeros((num_grid, cylinders.shape[-1]),
                                 dtype=cylinders.dtype, device=device)
            volume.index_copy_(0, indices.to(device), cylinders)
            return volume.reshape((recon_shape[0], recon_shape[1],
                                   cylinders.shape[-1]))

        if isinstance(hessian, _sharding.Shards):
            hessian = _sharding.Shards(
                [to_volume(t, d) for t, d in zip(hessian.tensors,
                                                 hessian.placement.devices)],
                hessian.placement)
        else:
            hessian = to_volume(hessian, hessian.device)
        return hessian if output_sharded else self._gather_recon(hessian)

    def get_voxels_at_indices(self, recon, indices):
        """The voxel cylinders at ``indices``, as (num_indices, num_slices).

        A recon divided across devices keeps its division: each shard is
        flattened and indexed on the device that holds it, and the result is a
        container over the same placement.  Only the row and column axes are
        flattened and only rows are selected, so the slice axis -- the one the
        shards are cut on -- is untouched, and gathering the result gives
        exactly what the whole-array call gives.
        """
        if isinstance(recon, _sharding.Shards):
            recon_shape = self.get_params('recon_shape')
            # The row count is named rather than inferred, because reshape
            # cannot infer it from a shard that owns no slices.
            num_pixels = int(recon_shape[0]) * int(recon_shape[1])
            indices = torch.as_tensor(indices, dtype=torch.int64)
            return _sharding.Shards(
                [t.reshape(num_pixels, t.shape[-1])[indices.to(t.device)]
                 for t in recon.tensors], recon.placement)
        return recon.reshape((-1, recon.shape[-1]))[indices]

    # ── auto-regularization ───────────────────────────────────────────────────
    def auto_set_regularization_params(self, sinogram, weights=None):
        """
        Automatically set the regularization parameters (sigma_y, sigma_x,
        and sigma_prox) from the sinogram and optional weights, and return
        them as a dict.  The parameters change only when
        ``auto_regularize_flag`` is True.  The statistics run on the host,
        on a view subsample.
        """
        # The statistics run on the host, so a tensor on any device is copied
        # there first.
        if torch.is_tensor(sinogram):
            sinogram = sinogram.cpu().numpy()
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()
        if self.get_params('auto_regularize_flag'):
            # The statistics are estimated from a subsample of the views, which
            # costs the same at any sinogram size.
            small_sinogram = self.subsample_views(sinogram)
            small_weights = 1 if weights is None else self.subsample_views(weights)

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
        """Warn if the sinogram support (the indicator from
        :meth:`_get_sino_indicator`) reaches the detector's edge channels."""
        if np.all(sino_indicator):
            # An indicator of all ones means either that the background could not be determined,
            # which has already warned, or that the support really covers everything.
            return
        edge_frac = float(np.mean(np.logical_or(sino_indicator[:, :, 0],
                                                sino_indicator[:, :, -1])))
        if edge_frac > 0.02 and self.get_params('verbose') > 0:
            warnings.warn(
                f"Lateral FoV truncation detected: the object support reaches the detector's "
                f"edge channels in {edge_frac:.0%} of the sampled view-rows.  Consider using "
                f"scale_recon_shape(s, s) where s >= 1.1 to improve image quality.")

    def auto_set_sigma_y(self, sinogram, sino_indicator, weights=1):
        """Set sigma_y from the (typically view-subsampled) sinogram, its
        support indicator, and optional weights.

        The statistics run on the host, against a host support indicator of the
        same shape, so a divided sinogram is refused.  To reduce a divided
        sinogram to something this accepts, take a view subsample of it with
        :meth:`subsample_views`, which returns a host array.
        """
        _sharding.reject_shards('auto_set_sigma_y', sinogram=sinogram,
                                sino_indicator=sino_indicator, weights=weights)
        snr_db = self.get_params('snr_db')
        magnification = self.get_magnification()
        delta_voxel, delta_det_channel = self.get_params(['delta_voxel', 'delta_det_channel'])

        # This is the root mean square of the sinogram over its support.
        signal_rms = float(np.average(weights * np.asarray(sinogram) ** 2, None,
                                      sino_indicator) ** 0.5)

        rel_noise_std = 10 ** (-snr_db / 20)

        # The regularization is adjusted when the reconstruction resolution differs from the
        # default.  The default pixel pitch is the detector pitch scaled by the magnification.
        default_pixel_pitch = delta_det_channel / magnification
        pixel_pitch_relative_to_default = delta_voxel / default_pixel_pitch

        sigma_y = np.float32(rel_noise_std * signal_rms *
                             (pixel_pitch_relative_to_default ** 0.5))
        self.set_params(no_warning=True, sigma_y=float(sigma_y), auto_regularize_flag=True)

    def auto_set_sigma_x(self, recon_std):
        """Set sigma_x (the qGGMRF prior scale) from the estimated recon
        standard deviation."""
        sharpness = self.get_params('sharpness')
        # This is a fraction of the typical reconstruction value.  The constant
        # 0.2 was determined empirically.
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=float(sigma_x), auto_regularize_flag=True)

    def auto_set_sigma_prox(self, recon_std):
        """Set sigma_prox (the proximal map prior scale) from the estimated
        recon standard deviation."""
        sharpness = self.get_params('sharpness')
        # This is a fraction of the typical reconstruction value.  The constant
        # 0.2 was determined empirically.
        sigma_prox = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_prox=float(sigma_prox),
                        auto_regularize_flag=True)

    @staticmethod
    def subsample_views(array, max_views_to_use=20):
        """Return an evenly-spaced subsample of about ``max_views_to_use``
        views (axis 0) as a host numpy array.  The statistical sinogram
        estimates run on such a subsample.  The stride depends only on the
        view count, so a second call with the same arguments subsamples a
        companion array (e.g. weights) the same way.

        A sinogram already divided across devices (a ``Shards``) is subsampled
        without assembling it: each shard's strided block is taken on the
        device that holds it, and only those views cross to the host.

        For sharded input the result is EXACT -- the same views in the same
        order as striding the assembled sinogram -- because this is data
        movement rather than an approximation.  Shard k owns global views
        ``[start_k, end_k)``, so the sampled global positions
        ``0, step, 2 * step, ...`` that land in that block are the local
        positions ``j`` with ``(start_k + j) % step == 0`` -- they begin at
        local offset ``(-start_k) % step`` and continue by ``step``.  Taking
        each shard's block from that offset and concatenating on the view axis
        reproduces the strided sinogram view for view.  A shard that owns no
        views, or one in which no sampled position lands, contributes an empty
        block, which changes nothing.
        """
        if isinstance(array, _sharding.Shards):
            placement = array.placement
            # The sharded axis may be written as a negative number, so resolve
            # it against the rank before comparing it with the view axis.
            if placement.axis % array.tensors[0].ndim != 0:
                raise ValueError(
                    'A sinogram must be sharded on its first (view) axis; got '
                    f'a placement on axis {placement.axis}.')
            num_views = sum(int(t.shape[0]) for t in array.tensors)
            axis_len = placement.axis_len
            if axis_len is not None and int(axis_len) != num_views:
                raise ValueError(
                    f'The shards cover {num_views} views, but their placement '
                    f'says the view axis is {int(axis_len)} long.')
            max_views_to_use = min(max_views_to_use, num_views)
            step_size = max(num_views // max_views_to_use, 1)
            blocks = []
            for tensor, (_dev, (start, _end)) in zip(
                    array.tensors, placement.shard_ranges(num_views)):
                block = tensor[(-start) % step_size::step_size]
                # The block is made contiguous on the shard's own device, so
                # that the copy to the host carries only the sampled views.
                blocks.append(block.detach().contiguous().cpu().numpy())
            return np.concatenate(blocks, axis=0)
        num_views = array.shape[0]
        max_views_to_use = min(max_views_to_use, num_views)
        step_size = max(num_views // max_views_to_use, 1)
        return np.array(array[::step_size])

    @staticmethod
    def _get_sino_indicator(sinogram, verbose=1):
        """Return an int8 mask of the sinogram support, with the same shape as
        the input.  This runs several reductions on the host, so it is usually
        called on a subsample of the views."""
        # Taking the negative logarithm of a sinogram can produce complex
        # values, so the input is checked for complex values and for NaN.
        sinogram = np.asarray(sinogram)
        if np.iscomplexobj(sinogram):
            raise TypeError("sinogram must be real-valued; got complex dtype.")
        if not np.isfinite(sinogram).all():
            raise ValueError("sinogram contains NaN and/or Inf values.")

        # The initial threshold is the right boundary of the background cluster
        # plus one cluster width.
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

        # The final threshold is a fraction of the median over the object
        # region.
        object_level = 0.25
        object_median = np.median(sinogram[sinogram >= threshold])
        object_threshold = object_level * object_median
        return np.int8(sinogram >= object_threshold)

    def _get_estimate_of_recon_std(self, sinogram, sino_indicator):
        """Estimate the standard deviation of the reconstruction from the
        sinogram and its support indicator.  The result scales sigma_x and
        sigma_prox."""
        delta_det_channel = self.get_params('delta_det_channel')
        delta_voxel = self.get_params('delta_voxel')
        recon_shape = self.get_params('recon_shape')
        magnification = self.get_magnification()
        num_det_channels = sinogram.shape[-1]

        # This is the typical magnitude of a sinogram value.
        typical_sinogram_value = np.average(np.abs(sinogram), weights=sino_indicator)

        # This path length estimate comes from the reconstruction width and
        # height.
        typical_path_length_space = (2 * recon_shape[0] * recon_shape[1]) / (
                recon_shape[0] + recon_shape[1]) * delta_voxel

        # This path length estimate comes from the detector width.
        typical_path_length_sino = num_det_channels * delta_det_channel / magnification

        typical_path_length = np.minimum(typical_path_length_space, typical_path_length_sino)

        # The typical reconstruction value is the typical sinogram value
        # divided by the typical path length.
        return typical_sinogram_value / typical_path_length

    # ── direct recon (FBP) machinery ──────────────────────────────────────────
    def _apply_direct_recon_filter(self, sinogram, filter_name, filter_scale,
                                   output_sharded=False, row_weight=None):
        """Apply the filtered backprojection row filter for a direct
        reconstruction.

        The scale factor ``filter_scale * pi / num_views`` is folded into the
        small filter array rather than multiplied into the whole sinogram.  The
        factor pi divided by num_views assumes views equally spaced over the
        full angular range.

        Args:
            sinogram: (num_views, num_rows, num_channels); numpy or tensor.
            filter_name (str): filter for generate_direct_recon_filter.
            filter_scale (float): geometry-specific filter scaling.
            output_sharded (bool): True returns the device tensor.
            row_weight (tensor or None): optional (rows, channels)
                per-detector pre-weight (the FDK cosine map); None is pure FBP.

        Returns:
            The filtered sinogram.
        """
        sinogram = self._shard_sinogram(sinogram)
        num_views, _, num_channels = self.get_params('sinogram_shape')
        recon_filter = tomography_utils.generate_direct_recon_filter(
            num_channels, filter_name=filter_name)
        recon_filter = recon_filter * np.float32(filter_scale * (np.pi / num_views))
        if isinstance(sinogram, _sharding.Shards):
            # The filter acts on each detector row on its own, so each shard
            # filters its own views with no data from other devices.
            def filter_worker(i, d):
                ft = torch.as_tensor(recon_filter, device=d)
                rw = None if row_weight is None else row_weight.to(d)
                return tomography_utils.apply_row_filter(
                    sinogram.tensors[i], ft, row_weight=rw)
            filtered = _sharding.Shards(
                _sharding.run_per_device(self.sino_placement.devices,
                                         filter_worker), self.sino_placement)
        else:
            filter_t = torch.as_tensor(recon_filter, device=self.torch_device)
            filtered = tomography_utils.apply_row_filter(sinogram, filter_t,
                                                         row_weight=row_weight)
        return filtered if output_sharded else self._gather_sinogram(filtered)

    # ── loss and per-iteration statistics ─────────────────────────────────────
    @staticmethod
    def get_forward_model_loss(error_sinogram, sigma_y, weights=None, normalize=True):
        """
        Calculate the forward model loss from the error sinogram and weights,
        where error_sinogram = measured_sinogram - forward_proj(recon).

        Args:
            error_sinogram (tensor): 3D error sinogram.
            sigma_y (float): the sinogram noise standard deviation parameter.
            weights (tensor, optional): sinogram weights.  Defaults to all 1s.
            normalize (bool, optional): If True (default), return the
                weight-normalized RMSE form; otherwise the unnormalized
                weighted squared error.

        Returns:
            The loss as a device scalar tensor.
        """
        # The sums below run on one array on one device, so a sharded array is
        # refused here.
        _sharding.reject_shards('get_forward_model_loss',
                                error_sinogram=error_sinogram, weights=weights)
        if weights is None:
            weights = 1
            avg_weight = 1
        elif np.ndim(weights) == 0:
            # For a scalar the average weight is the scalar itself.
            avg_weight = weights
        else:
            # This branch takes any array, including a numpy array, which is
            # not a torch tensor.
            weights = torch.as_tensor(weights, dtype=torch.float32,
                                      device=error_sinogram.device)
            avg_weight = torch.mean(weights)
        # The sum runs in chunks, so no array of sinogram size is allocated.
        weighted_sq_sum = _memory_ledger.weighted_square_sum(error_sinogram,
                                                             weights)
        if normalize:
            loss = torch.sqrt(weighted_sq_sum
                              / (avg_weight * float(error_sinogram.numel()))) / sigma_y
        else:
            loss = (1.0 / (2 * sigma_y ** 2)) * weighted_sq_sum
        return loss

    @staticmethod
    def _vcd_iteration_stats(error_sinogram, flat_recon, sigma_y, weights=None):
        """Return the per-iteration VCD statistics (fm_loss, recon_l1,
        es_rmse).  Both statistics are normalized by the number of elements in
        the error sinogram."""
        fm_loss = TomographyModel.get_forward_model_loss(
            error_sinogram, sigma_y, weights)
        # The sums below run in chunks, so no second array of reconstruction or sinogram size is
        # allocated.  The chunked order can change the last digits of the NMAE stopping test.
        recon_l1 = _memory_ledger.image_ell1(flat_recon)
        es_rmse = torch.sqrt(_memory_ledger.weighted_square_sum(error_sinogram)
                             / float(error_sinogram.numel()))
        return fm_loss, recon_l1, es_rmse

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights,
                             fm_constant, const_weights):
        """
        Compute the two forward model line-search terms:
        ``fm_constant * sum(weighted_error_sinogram * delta_sinogram)`` and
        ``fm_constant * sum(delta_sinogram^2 * weights)``, returned as device
        scalars.

        The reconstruction loop calls this once per shard, on that shard's own
        tensors, and combines the partials itself.  A whole divided array is
        refused: summing one would take a cross-device reduction and a choice
        of where the scalars land.
        """
        _sharding.reject_shards('get_forward_lin_quad',
                                weighted_error_sinogram=weighted_error_sinogram,
                                delta_sinogram=delta_sinogram, weights=weights)
        forward_linear = fm_constant * torch.sum(weighted_error_sinogram * delta_sinogram)
        if const_weights:
            forward_quadratic = fm_constant * torch.sum(delta_sinogram * delta_sinogram)
        else:
            forward_quadratic = fm_constant * torch.sum(
                delta_sinogram * delta_sinogram * weights)
        return forward_linear, forward_quadratic

    # ── the VCD loop ────────────────────────────────────────────────────────
    def _get_update_direction(self, forward_grad, prior_grad, forward_hess,
                              prior_hess, pixel_indices, dev_index=0):
        """Return the update direction for one subset of pixels.

        The base implementation is the preconditioned gradient
        -(forward_grad + prior_grad) / (forward_hess + prior_hess).
        An override must return -M (forward_grad + prior_grad) with M positive
        definite, which preserves the minimizers of the cost.  The arguments
        are one shard's arrays, of shape (num_subset_pixels, local_slices).
        The base implementation ignores ``dev_index``."""
        fn = maybe_compile(_diagonal_update_direction, self.compile_enabled)
        return fn(forward_grad, prior_grad, forward_hess, prior_hess)

    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        """
        Create the function that updates one subset of pixels in the recon
        and error sinogram.  The updater is one body over the loop's uniform
        per-device state: each step runs per shard, and the line-search
        partials combine on device, so the subset loop has no host
        synchronization at any device count.

        Args:
            fm_hessian (tensor or Shards): (num_pixels, num_slices) Hessian
                diagonal for the forward model loss.
            weights (tensor, Shards, or 1): sinogram weights, or the
                constant 1.
            prox_input (tensor or Shards, optional): proximal-map input,
                flattened to (num_pixels, num_slices).

        Returns:
            (callable) vcd_subset_updater(flat_recon, error_sinogram,
            pixel_indices), which updates the state in place.
        """
        sino_placement, recon_placement = self.sino_placement, self.recon_placement
        devices = sino_placement.devices
        num_devices = sino_placement.n_devices
        fm_hessian = self._as_shards(fm_hessian, recon_placement)
        const_weights = not (torch.is_tensor(weights)
                             or isinstance(weights, _sharding.Shards))
        if const_weights and abs(weights - 1) > 1e-5:
            raise ValueError('Constant weights must have value 1.')
        weights = None if const_weights else self._as_shards(weights, sino_placement)
        prox_input = (None if prox_input is None
                      else self._as_shards(prox_input, recon_placement))

        positivity_flag = self.get_params('positivity_flag')
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = (b, sigma_x, p, q, T)
        sigma_prox = self.get_params('sigma_prox')
        recon_shape = self.get_params('recon_shape')
        max_alpha = self.get_params('max_alpha')

        # The compiled functions are bound once for all subsets, with one instance per device
        # thread.  A compiled function carries launcher state that must not be shared.
        def per_dev(fn):
            return [maybe_compile(fn, self.compile_enabled, instance_key=i)
                    for i in range(num_devices)]
        qggmrf_grad_hess = per_dev(_qggmrf.qggmrf_gradient_and_hessian_at_indices)
        prior_line_terms = per_dev(_prior_line_terms)
        lin_quad_const = per_dev(_forward_lin_quad_const)
        lin_quad_weighted = per_dev(_forward_lin_quad_weighted)
        apply_update = per_dev(_apply_update)

        dev0 = devices[0]

        def combine_on_lead(parts):
            """Sum the zero dimensional partial results of each shard on the
            lead device."""
            total = parts[0]
            for part in parts[1:]:
                total = total + _sharding.move_shard(part, dev0,
                                                     self.dev2dev_safe)
            return total

        # The qGGMRF boundary halos are staged once for each pass over a partition.  A halo of
        # None means a reflected boundary at a true edge of the volume.
        halos = {'left': [None] * num_devices, 'right': [None] * num_devices}

        def stage_halos(flat_shards):
            halos['left'], halos['right'] = _sharding.exchange_qggmrf_halos(
                flat_shards, self.dev2dev_safe)

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices):
            """Run one VCD update on a single subset of the partition.  The
            invariant is error_sinogram = measured_sinogram -
            forward_proj(recon).  The arguments flat_recon and error_sinogram
            are shards and are updated in place.

            Returns:
                flat_recon, error_sinogram, ell1_for_subset,
                alpha_for_subset, delta_sumsq_subset.
            """
            pixel_indices_per_device = [torch.as_tensor(pixel_indices, dtype=torch.int64).to(dev)
                       for dev in devices]

            # This computes the prior gradient and Hessian at each pixel of the index set, one
            # shard at a time.  The halos carry the term that crosses a shard boundary.
            def prior_worker(i, dev):
                if prox_input is None:
                    grad, hess = qggmrf_grad_hess[i](
                        flat_recon.tensors[i], recon_shape, pixel_indices_per_device[i],
                        qggmrf_params, left_halo=halos['left'][i],
                        right_halo=halos['right'][i])
                else:
                    # The proximal map prior acts pointwise, so its Hessian is
                    # a scalar.
                    grad = _qggmrf.prox_gradient_at_indices(
                        flat_recon.tensors[i], prox_input.tensors[i],
                        pixel_indices_per_device[i], sigma_prox)
                    hess = 1 / (sigma_prox ** 2)
                return grad, hess

            prior_terms = _sharding.run_per_device(devices, prior_worker,
                                             executor=self._per_device_pool)

            # The forward model loss is
            # 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2.
            if const_weights:
                weighted_error_sinogram = error_sinogram
            else:
                weighted_error_sinogram = _sharding.Shards(
                    _sharding.run_per_device(
                        devices, lambda i, dev: weights.tensors[i]
                        * error_sinogram.tensors[i],
                        executor=self._per_device_pool), sino_placement)

            # The back projection gives the gradient.  The value fm_constant is
            # 1/sigma_y^2.
            back_projected_error = self.sparse_back_project(weighted_error_sinogram, pixel_indices)
            if not const_weights:
                # The weighted product is no longer needed, because the line
                # search terms fuse the weights into their own reductions.
                weighted_error_sinogram = None

            # Each shard computes its update direction and its prior line
            # search terms in one worker.
            def direction_worker(i, dev):
                prior_grad, prior_hess = prior_terms[i]
                forward_grad = -fm_constant * back_projected_error.tensors[i]
                forward_hess = fm_constant * fm_hessian.tensors[i][pixel_indices_per_device[i]]
                delta_recon = self._get_update_direction(
                    forward_grad, prior_grad, forward_hess, prior_hess,
                    pixel_indices_per_device[i], dev_index=i)
                prior_hess_t = (prior_hess if torch.is_tensor(prior_hess)
                                else torch.as_tensor(prior_hess,
                                                     dtype=torch.float32,
                                                     device=delta_recon.device))
                prior_linear_part, prior_quadratic_part = prior_line_terms[i](
                    prior_grad, prior_hess_t, delta_recon)
                return delta_recon, prior_linear_part, prior_quadratic_part
            direction_results = _sharding.run_per_device(devices, direction_worker,
                                            executor=self._per_device_pool)
            delta_recon_per_device = [delta for delta, _, _ in direction_results]
            prior_linear = combine_on_lead(
                [linear for _, linear, _ in direction_results])
            prior_quadratic_approx = combine_on_lead(
                [quadratic for _, _, quadratic in direction_results])

            # These buffers are freed before the delta projection, which uses
            # much more memory.
            del prior_terms, back_projected_error, direction_results

            # This is the update direction in the sinogram domain.
            delta_sinogram = self.sparse_forward_project(
                _sharding.Shards(delta_recon_per_device, recon_placement), pixel_indices)

            def lin_quad_worker(i, dev):
                local_delta_sinogram = delta_sinogram.tensors[i]
                if const_weights:
                    return lin_quad_const[i](error_sinogram.tensors[i], local_delta_sinogram,
                                             fm_constant)
                # Fusing the weights into the reductions avoids an array of
                # sinogram size for each subset.
                return lin_quad_weighted[i](error_sinogram.tensors[i], local_delta_sinogram,
                                            weights.tensors[i], fm_constant)
            forward_line_terms = _sharding.run_per_device(devices, lin_quad_worker,
                                          executor=self._per_device_pool)
            forward_linear = combine_on_lead(
                [linear for linear, _ in forward_line_terms])
            forward_quadratic = combine_on_lead(
                [quadratic for _, quadratic in forward_line_terms])

            # The line search runs on the device.  The step alpha stays a
            # scalar tensor, so no subset costs a host synchronization.
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + _F32_EPS
            alpha = alpha_numerator / alpha_denominator
            alpha = torch.clamp(alpha, _F32_EPS, max_alpha)
            alpha_per_device = ([alpha] if num_devices == 1 else
                         [_sharding.move_shard(alpha, dev, self.dev2dev_safe)
                          for dev in devices])

            # The positivity constraint clips the update so that recon + alpha * delta is at
            # least zero, and the sinogram projection is then computed again.
            if positivity_flag is True:
                def positivity_worker(i, dev):
                    recon_at_indices = flat_recon.tensors[i][pixel_indices_per_device[i]]
                    pos_constant = 1.0 / (alpha_per_device[i] + _F32_EPS)
                    return torch.maximum(-pos_constant * recon_at_indices,
                                         delta_recon_per_device[i])
                delta_recon_per_device[:] = _sharding.run_per_device(
                    devices, positivity_worker, executor=self._per_device_pool)
                delta_sinogram = self.sparse_forward_project(
                    _sharding.Shards(delta_recon_per_device, recon_placement), pixel_indices)

            # The updates are applied in place on each shard.  The sum of
            # squared updates per slice is the convergence diagnostic.
            def apply_worker(i, dev):
                delta_scaled = alpha_per_device[i] * delta_recon_per_device[i]
                _, _, delta_sumsq_local, ell1_local = apply_update[i](
                    flat_recon.tensors[i], error_sinogram.tensors[i],
                    pixel_indices_per_device[i], delta_scaled, alpha_per_device[i],
                    delta_sinogram.tensors[i])
                return delta_sumsq_local, ell1_local
            apply_results = _sharding.run_per_device(devices, apply_worker,
                                               executor=self._per_device_pool)
            # The per-slice sums are concatenated in global slice order on the
            # lead device.
            delta_sumsq_subset = (apply_results[0][0] if num_devices == 1
                                  else torch.cat(
                [_sharding.move_shard(sumsq, dev0, self.dev2dev_safe)
                 for sumsq, _ in apply_results]))
            ell1_for_subset = combine_on_lead(
                [ell1 for _, ell1 in apply_results])
            return (flat_recon, error_sinogram, ell1_for_subset, alpha,
                    delta_sumsq_subset)

        vcd_subset_updater.stage_halos = stage_halos
        return vcd_subset_updater

    def vcd_partition_iterator(self, vcd_subset_updater, flat_recon, error_sinogram,
                               partition, rng=None):
        """
        Calculate a full iteration of the VCD algorithm by scanning over the
        subsets of the partition, updating flat_recon and error_sinogram in
        place.

        Args:
            rng (numpy.random.Generator, optional): the generator the visiting
                order is drawn from.  Defaults to None, the global np.random
                state.

        Returns:
            (flat_recon, error_sinogram, ell1_for_partition, alpha,
            delta_sumsq_partition): the updated state, the summed L1 recon
            change, alpha averaged over the subsets, and the per-slice sum
            of squared update values over the partition.
        """
        # The qGGMRF boundary halos are staged once for this whole pass over
        # the partition.
        if hasattr(vcd_subset_updater, 'stage_halos'):
            vcd_subset_updater.stage_halos(flat_recon)
        # The subsets are visited in a random order.  Do not change this draw, because
        # a different random sequence changes the iteration trace that the tests compare against.
        ell1_for_partition = 0
        alpha_sum = 0
        delta_sumsq_partition = 0
        draw = np.random if rng is None else rng
        subset_indices = draw.permutation(partition.shape[0])

        for index in subset_indices:
            subset = partition[index]
            (flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset,
             delta_sumsq_subset) = vcd_subset_updater(flat_recon, error_sinogram, subset)
            ell1_for_partition += ell1_for_subset
            alpha_sum += alpha_for_subset
            delta_sumsq_partition = delta_sumsq_partition + delta_sumsq_subset

        return (flat_recon, error_sinogram, ell1_for_partition,
                alpha_sum / partition.shape[0], delta_sumsq_partition)

    def _vcd_recon(self, sinogram, partitions, partition_sequence,
                   stop_threshold_change_pct, weights=None, init_recon=None,
                   prox_input=None, compute_prior_loss=False, first_iteration=0,
                   init_error_sinogram=None, fm_hessian=None,
                   return_checkpoint=False, rng=None):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate
        Descent algorithm for a given set of partitions and a prescribed
        partition sequence.

        This is the reconstruction engine that :meth:`recon` and
        :meth:`prox_map` run, and it is not part of the public interface.  An
        ordinary reconstruction calls one of those two, which build the
        partitions, the partition sequence and the regularization first.  This
        method is described here for advanced users who need the
        ``init_error_sinogram`` / ``return_checkpoint`` resume workflow
        below, which the public methods do not offer.

        Args:
            sinogram (numpy or tensor or Shards): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).  The device form
                as returned by :meth:`prepare_sino_for_devices` is accepted
                too, so repeated reconstructions of one large sinogram pay the
                host-to-device transfer once.
            partitions (list): K partitions, each an (N_subsets, N_indices)
                integer index tensor of voxels to update.
            partition_sequence (ndarray): which partition to use at each
                iteration.
            stop_threshold_change_pct (float): stop when the NMAE percent
                change between iterations falls below this value.
            weights (numpy or tensor or Shards, optional): 3D positive weights
                with the same shape as the sinogram, in a plain array or in the
                device form.  Defaults to all 1s.
            init_recon (array or int or None): initial reconstruction.  None
                uses recon_direct; an int gives a constant volume.
            prox_input (array or Shards, optional): input to a proximal map,
                as a full volume or in the device form (one shard per device,
                on this model's recon placement).
            compute_prior_loss (bool, optional): If True, also compute the
                prior loss (a debug path for small recons).
            first_iteration (int, optional): iteration offset for restarts.
            init_error_sinogram (array or tensor, optional): precomputed error
                sinogram to resume from, skipping the initializing forward
                projection.  Requires init_recon, and the pair is trusted as
                consistent (init_error_sinogram == sinogram - A @ init_recon).
                No defensive copy is made: both arrays become the loop's
                working buffers and are updated in place, so after the call
                they reflect the resumed state.  To keep the pre-resume state,
                copy before resuming.
            fm_hessian (array or tensor, optional): precomputed forward-model
                Hessian diagonal for the same weights and geometry; read-only
                in the loop.  None computes it internally.
            return_checkpoint (bool, optional): If True, additionally return
                {'error_sinogram': ..., 'fm_hessian': ...} for the two
                arguments above.  The dict references the loop's own final
                device tensors with no copy; copy them to snapshot or persist.
            rng (numpy.random.Generator, optional): the generator the order of
                the subsets is drawn from at each iteration.  Defaults to
                None, the global np.random state.

        Returns:
            (recon, recon_stats): the 3D reconstruction tensor and a tuple of
            per-iteration stats (fm_rmse, pm_loss, nmae_update, alpha_values,
            delta_norm_per_slice), where nmae_update is
            ||recon(i+1) - recon(i)||_1 / ||recon(i+1)||_1.
            With return_checkpoint=True: (recon, recon_stats, checkpoint).
        """
        self.verify_valid_params()
        dev = self.torch_device
        recon_shape = self.get_params('recon_shape')
        sinogram_shape = self.get_params('sinogram_shape')
        if isinstance(sinogram, _sharding.Shards):
            # The container has no shape of its own, so the shard tensors are
            # checked instead.
            self._check_sinogram_shards(sinogram, sinogram_shape)
        elif tuple(sinogram.shape) != tuple(sinogram_shape):
            raise ValueError('sinogram does not have the shape in sinogram_shape. \n'
                             f'Expected {tuple(sinogram_shape)}, got '
                             f'{tuple(sinogram.shape)}.')

        # The device layout is settled before the first large allocation.
        memory_ledger = self._apply_device_policy(
            partition_sequence=partition_sequence, weights=weights,
            init_recon=init_recon, fm_hessian=fm_hessian,
            prox_input=prox_input, init_error_sinogram=init_error_sinogram)
        if _memory_ledger.calibration_enabled():
            # The measured run begins here, so the peak memory counters are
            # reset here, once per reconstruction.
            _memory_ledger.calibration_start(self.sino_placement.devices)
        # The layout is final here, so the log can now name the devices the run
        # will use.
        self._log_device_report()

        constant_weights = weights is None
        if constant_weights:
            weights = 1
        else:
            weights = self._shard_sinogram(weights)

        if init_error_sinogram is not None and init_recon is None:
            raise ValueError('init_error_sinogram requires init_recon (the pair must be a '
                             'consistent resume state; see the docstring).')

        # When resuming, the error sinogram replaces the only use of the
        # sinogram, so the sinogram is not placed on the devices.
        if init_error_sinogram is None:
            sinogram = self._shard_sinogram(sinogram)

        scale_recon_to_sinogram = init_recon is None
        if init_recon is None:
            self.logger.info('Starting direct recon for initial reconstruction')
            init_recon = self.recon_direct(sinogram, output_sharded=True)
        elif isinstance(init_recon, int):
            init_recon = self._constant_recon(init_recon)
        else:
            if tuple(np.shape(init_recon)) != tuple(recon_shape):
                raise ValueError(f"init_recon does not have the correct shape. Expected "
                                 f"{tuple(recon_shape)}, got {tuple(np.shape(init_recon))}.")
            init_recon = self._shard_recon(init_recon)

        if init_error_sinogram is not None:
            # The pair is trusted, so the initializing forward projection is
            # skipped.  The caller's arrays become the loop's working buffers.
            self.logger.info('Resuming from init_error_sinogram')
            error_sinogram = self._shard_sinogram(init_error_sinogram)
        else:
            error_sinogram, init_recon = self._initial_error_state(
                sinogram, init_recon, weights, constant_weights,
                scale_recon_to_sinogram)

        # The sinogram is now folded into the error sinogram, so dropping this
        # reference frees any device copy made above.
        sinogram = None
        error_sinogram = self._shard_sinogram(error_sinogram)

        if prox_input is not None:
            if isinstance(prox_input, _sharding.Shards):
                # The container has no shape of its own, so the shard tensors
                # are flattened and checked instead.
                prox_input = self._flatten_prox_shards(prox_input, recon_shape)
            else:
                # The shape is checked before flattening, so that an input with the right
                # number of elements but the wrong shape raises rather than being reshaped.
                if tuple(prox_input.shape) != tuple(recon_shape):
                    raise ValueError('prox_input does not have the correct size. \n'
                                     f'Expected {tuple(recon_shape)}, got shape '
                                     f'{tuple(prox_input.shape)} for prox_input shape.')
                prox_input = self._shard_recon(
                    prox_input.reshape((-1, prox_input.shape[-1])))

        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # math.prod uses exact Python integers.  np.prod would wrap past
        # 2^31 elements.
        total_sino_size = math.prod(sinogram_shape)

        # The Hessian diagonal of the forward model is the back projection of the weights with
        # squared coefficients.  A Hessian supplied by the caller skips that back projection.
        if fm_hessian is None:
            if constant_weights:
                hess_weights = self._sino_ones_device_form(error_sinogram)
            else:
                hess_weights = weights
            self.logger.info('Computing Hessian diagonal')
            # The back projection covers only the masked pixels.  The loop reads the Hessian only
            # at partition indices from the same mask, so the values are unchanged.
            hess_indices = (None if self.get_params('use_ror_mask') is False
                            else self.full_indices_device())
            fm_hessian = self.compute_hessian_diagonal(weights=hess_weights,
                                                       output_sharded=True,
                                                       indices=hess_indices)
        else:
            self.logger.info('Using precomputed Hessian diagonal')
            fm_hessian = self._shard_recon(fm_hessian)
        fm_hessian = self._flatten_hessian(fm_hessian)

        flat_recon = self._flatten_recon(init_recon)

        # From here the loop runs one code path for any device count.  On a single device the
        # one shard container aliases its tensor, so in place updates reach the caller's array.
        flat_recon = self._as_shards(flat_recon, self.recon_placement)
        error_sinogram = self._as_shards(error_sinogram,
                                             self.sino_placement)

        vcd_subset_updater = self.create_vcd_subset_updater(
            fm_hessian, weights=weights, prox_input=prox_input)

        self.logger.info('Starting VCD iterations')
        if verbose >= 2:
            if memory_ledger is not None:
                self.logger.debug('Modeled peak device memory by phase:')
                self.logger.debug(memory_ledger.format_table())
                self.logger.debug('--------')
            output = io.StringIO()
            get_memory_stats(file=output)
            self.logger.debug(output.getvalue())
            self.logger.debug('--------')

        max_iters = partition_sequence.size
        fm_rmse = np.zeros(max_iters)
        pm_loss = np.zeros(max_iters)
        nmae_update = np.zeros(max_iters)
        alpha_values = np.zeros(max_iters)
        delta_norm_per_slice = np.zeros((max_iters, recon_shape[2]))
        num_iters = 0
        if not self.sino_placement.is_trivial:
            # One thread pool serves the whole loop.  A single device model
            # never creates one.
            self._per_device_pool = _sharding.device_pool(
                self.sino_placement.n_devices)
        try:
            for i in range(max_iters):
                partition = partitions[partition_sequence[i]]
                (flat_recon, error_sinogram, ell1_for_partition, alpha,
                 delta_sumsq_partition) = self.vcd_partition_iterator(
                    vcd_subset_updater, flat_recon, error_sinogram, partition,
                    rng=rng)

                # The element count is passed in, because a sharded error sinogram is a list
                # of tensors and the statistics normalize by the total element count.
                fm_loss_i, recon_l1, es_rmse = self._iteration_stats(
                    error_sinogram, flat_recon, sigma_y, weights,
                    constant_weights, float(total_sino_size))
                fm_rmse[i] = float(fm_loss_i)
                recon_l1_f = float(recon_l1)
                # A reconstruction of all zeros gives nan rather than raising
                # ZeroDivisionError.
                nmae_update[i] = (float(ell1_for_partition) / recon_l1_f
                                  if recon_l1_f else float('nan'))
                alpha_values[i] = float(alpha)
                delta_norm_per_slice[i] = np.sqrt(
                    delta_sumsq_partition.cpu().numpy())[:recon_shape[2]]

                if verbose >= 1:
                    iter_output = (
                        '\nAfter iteration {} of a max of {}: Pct change={:.4f}, '
                        'Forward loss={:.4f}'.format(i + first_iteration,
                                                     max_iters + first_iteration,
                                                     100 * nmae_update[i], fm_rmse[i]))
                    if compute_prior_loss:
                        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(
                            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
                        b = _qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts)
                        qggmrf_params = (b, sigma_x, p, q, T)
                        # The prior loss is evaluated on the assembled volume,
                        # so that terms between slices cross shard boundaries.
                        total_recon_size = math.prod(recon_shape)
                        loss_recon = self._gather_recon(flat_recon).reshape(
                            tuple(recon_shape))
                        pm_loss[i] = _qggmrf.qggmrf_loss(loss_recon, qggmrf_params)
                        pm_loss[i] /= total_recon_size
                        # Each loss arrives scaled by its own element count.  Both are
                        # unscaled, added, and rescaled by the average element count.
                        total_loss = ((fm_rmse[i] * total_sino_size
                                       + pm_loss[i] * total_recon_size)
                                      / (0.5 * (total_sino_size + total_recon_size)))
                        iter_output += ', Prior loss={:.4f}, Weighted total loss={:.4f}'.format(
                            pm_loss[i], total_loss)
                    self.logger.info(iter_output)
                    self.logger.info(f'Relative step size (alpha)={alpha_values[i]:.2f}, '
                                     f'Error sino RMSE={float(es_rmse):.4f}')
                    self.logger.info('Number subsets = {}'.format(partition.shape[0]))
                    if verbose >= 2:
                        output = io.StringIO()
                        get_memory_stats(file=output)
                        self.logger.debug(output.getvalue())
                        self.logger.debug('--------')
                num_iters += 1
                if nmae_update[i] < stop_threshold_change_pct / 100:
                    self.logger.warning('Change threshold stopping condition reached')
                    break
        finally:
            if self._per_device_pool is not None:
                self._per_device_pool.shutdown(wait=True)
                self._per_device_pool = None

        # The calibration comparison runs before the loop's state is released,
        # so that the measured peak still reflects the reconstruction.
        if memory_ledger is not None and _memory_ledger.calibration_enabled():
            rows = _memory_ledger.calibration_report(
                memory_ledger, self.sino_placement.devices)
            self.last_memory_calibration = rows
            if rows:
                self.logger.warning(_memory_ledger.format_calibration(rows))

        # The state returns to the device form here.  On a single device that
        # is a plain tensor, and it is the same object the loop updated.
        flat_recon = self._as_device_form(flat_recon)
        error_sinogram = self._as_device_form(error_sinogram)
        recon_3d = self._recon_from_flat(flat_recon, recon_shape)
        losses = (fm_rmse[:num_iters], pm_loss[:num_iters], nmae_update[:num_iters],
                  alpha_values[:num_iters], delta_norm_per_slice[:num_iters])
        if return_checkpoint:
            checkpoint = {'error_sinogram': error_sinogram, 'fm_hessian': fm_hessian}
            return recon_3d, losses, checkpoint
        return recon_3d, losses

    def initialize_recon(self, sinogram, weights=None, init_recon=None,
                         max_iterations=15, first_iteration=0,
                         logfile_path='~/.mbirtorch/logs/recon.log',
                         print_logs=True, rng=None):
        """
        Do the parameter initialization needed for recon: generate the set of
        voxel partitions and the partition sequence, validate the inputs, and
        run auto-regularization.

        Args:
            See :meth:`recon` for arguments.  ``rng`` is the generator the
            partitions are drawn from; None uses the global np.random state.

        Returns:
            sinogram, weights, init_recon, partitions, partition_sequence,
            granularity, regularization_params
        """
        # The run logger is set up when a run is initialized.  A Plug and Play loop that passes
        # do_initialization=False skips this method, so the whole loop writes to one log.
        self._log_run_header(first_iteration, logfile_path, print_logs)
        recon_shape, granularity, use_ror_mask = self.get_params(
            ['recon_shape', 'granularity', 'use_ror_mask'])
        partitions = vcd_utils.gen_set_of_pixel_partitions(
            recon_shape, granularity, device=self.torch_device,
            use_ror_mask=use_ror_mask, rng=rng)

        partition_sequence = self.get_params('partition_sequence')
        partition_sequence = vcd_utils.gen_partition_sequence(
            partition_sequence, max_iterations=max_iterations)
        partition_sequence = partition_sequence[first_iteration:]

        # The input checks run where the data already is.  A sharded sinogram is checked one
        # shard at a time on the device that holds it, and never brought back to the host.
        if isinstance(sinogram, _sharding.Shards):
            for tensor in sinogram.tensors:
                if tensor.is_complex():
                    raise TypeError(
                        "sinogram must be real-valued; got complex dtype.")
                low, high = _array_extremes(tensor)
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError("sinogram contains NaN and/or Inf values.")
            # The statistics below reduce this to a small subsample of the
            # views before anything reaches the host.
            sinogram_for_stats = sinogram
        else:
            sinogram_np = np.asarray(sinogram) if not torch.is_tensor(sinogram) \
                else sinogram.cpu().numpy()
            if np.iscomplexobj(sinogram_np):
                raise TypeError("sinogram must be real-valued; got complex dtype.")
            low, high = _array_extremes(sinogram_np)
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError("sinogram contains NaN and/or Inf values.")
            sinogram_for_stats = sinogram_np
        if weights is not None:
            if isinstance(weights, _sharding.Shards):
                # The weights are all zero only when every shard is all zero.
                all_zero = True
                for tensor in weights.tensors:
                    low, high = _array_extremes(tensor)
                    if not (math.isfinite(low) and math.isfinite(high)):
                        raise ValueError("weights contains NaN and/or Inf values.")
                    if low < 0:
                        raise ValueError("weights contain negative values.")
                    all_zero = all_zero and low == 0 and high == 0
                if all_zero:
                    raise ValueError("all weights are zero.")
            else:
                weights_np = np.asarray(weights) if not torch.is_tensor(weights) \
                    else weights.cpu().numpy()
                low, high = _array_extremes(weights_np)
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError("weights contains NaN and/or Inf values.")
                if low < 0:
                    raise ValueError("weights contain negative values.")
                if low == 0 and high == 0:
                    raise ValueError("all weights are zero.")

        regularization_params = self.auto_set_regularization_params(
            sinogram_for_stats, weights=weights)
        return (sinogram, weights, init_recon, partitions, partition_sequence,
                granularity, regularization_params)

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=15,
              stop_threshold_change_pct=0.2, first_iteration=0,
              logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
              output_sharded=False, rng=None):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate
        Descent algorithm.  This function takes care of generating its own
        partitions and partition sequence.

        To restart a recon using the same partition sequence, set
        first_iteration to the number of iterations completed so far and set
        init_recon to the output of the previous recon; this continues the
        partition sequence from where the previous recon left off.

        Device use: on CUDA with several devices, this chooses a device
        count automatically.  Two rules make the choice: measured speed
        thresholds decide how many devices are worth using at this problem
        size, and a memory check confirms the chosen layout fits before the
        first large allocation.  Nothing needs to change in a calling
        script.  ``configure_devices(num_devices=n)`` fixes the count
        instead, and ``configure_devices(num_devices=1)`` pins the run to
        one device for reproducibility.  The environment variable
        ``MBIRTORCH_NUM_DEVICES`` pins the count process-wide, which is
        what a test suite or a nightly should use.

        Reproducibility note: the pixel partitions and the order in which the
        subsets are visited are drawn from numpy's global random number
        generator, so reconstructions vary slightly from run to run.  For a
        reproducible result, call ``np.random.seed(seed)`` before calling this
        method, or pass ``rng``, which makes the draws independent of anything
        else the process draws.  Results also differ slightly with the device
        count, and that difference decays as iterations proceed.

        Args:
            sinogram (numpy or tensor or Shards): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).  The device form
                as returned by :meth:`prepare_sino_for_devices` is accepted
                too, so repeated reconstructions of one large sinogram pay the
                host-to-device transfer once.
            weights (numpy or tensor or Shards, optional): 3D positive weights
                with the same shape as the sinogram, in a plain array or in the
                device form.  Defaults to None (all 1s).
            init_recon (array, int, or None, optional): initial reconstruction.
                If None, recon_direct is called with default arguments.
            max_iterations (int, optional): maximum number of VCD iterations.
            stop_threshold_change_pct (float, optional): stop when
                100 * ||delta_recon||_1 / ||recon||_1 between iterations drops
                below this value.  Defaults to 0.2; set 0 to guarantee exactly
                max_iterations.
            first_iteration (int, optional): the number of iterations previously
                completed when restarting a recon.  Defaults to 0.
            logfile_path (str, optional): Path to the output log file ('~' expands to the
                user's home directory).  If None or empty, no log file is written.
                Defaults to '~/.mbirtorch/logs/recon.log'.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the device form: a torch
                tensor on a single device, or a Shards container (one
                tensor per device) on a multi-device model.
            rng (numpy.random.Generator, optional): the generator every random
                draw of the run comes from, the partitions and the order of
                the subsets.  Defaults to None, the global np.random state.

        Returns:
            (recon, recon_dict): the reconstruction volume, and a dict
            with entries 'recon_params' (per-iteration traces and settings),
            'recon_log' (the run's log text), 'notes', and
            'model_params' (a snapshot of the model parameters).
        """
        # The initial reconstruction is checked against the shape of the whole volume, which a
        # sharded array does not have, so a sharded init_recon is refused.
        _sharding.reject_shards('recon', init_recon=init_recon)
        (sinogram, weights, init_recon, partitions, partition_sequence, granularity,
         regularization_params) = self.initialize_recon(
            sinogram, weights, init_recon, max_iterations, first_iteration,
            logfile_path=logfile_path, print_logs=print_logs, rng=rng)

        # This uses no_grad rather than inference_mode, because torch.compile guards fail on
        # compiled calls with in place updates inside inference_mode.
        with torch.no_grad():
            recon, loss_vectors = self._vcd_recon(
                sinogram, partitions, partition_sequence,
                stop_threshold_change_pct, weights=weights, init_recon=init_recon,
                first_iteration=first_iteration, rng=rng)

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

        if logfile_path:
            self.logger.info('Logs written to {}'.format(
                os.path.abspath(os.path.expanduser(logfile_path))))
        for h in list(self.logger.handlers):
            h.flush()
        # This call has written its last line, so the log file is closed.  A call that continues
        # this run reopens it.
        self.close_log_file()

        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        return (recon if output_sharded else self._gather_recon(recon)), recon_dict

    def _iteration_stats(self, error_sinogram, flat_recon, sigma_y, weights,
                         constant_weights, total_sino_size):
        """Return the per-iteration statistics (forward model loss, the L1 norm
        of the reconstruction, and the error sinogram RMSE).

        A single device state calls _vcd_iteration_stats.  A state spread over
        devices combines the shard sums on the host, which is the loop's one
        host synchronization per iteration.  ``total_sino_size`` is the element
        count of the whole sinogram, which no single shard knows."""
        if (isinstance(error_sinogram, _sharding.Shards)
                and not error_sinogram.placement.is_trivial):
            error_shards, flat_shards = error_sinogram, flat_recon
            weights_shards = None if constant_weights else weights

            def sino_worker(i, d):
                # The sums run in chunks on each shard, so no array the size of
                # a shard is allocated.
                e = error_shards.tensors[i]
                sq = float(_memory_ledger.weighted_square_sum(e))
                if weights_shards is None:
                    return sq, 0.0, sq
                w = weights_shards.tensors[i]
                return (float(_memory_ledger.weighted_square_sum(e, w)),
                        float(torch.sum(w)), sq)
            parts = _sharding.run_per_device(error_shards.placement.devices,
                                             sino_worker)
            weighted_sq = sum(a for a, _, _ in parts)
            sq = sum(c for _, _, c in parts)
            if weights_shards is None:
                avg_weight = 1.0
            else:
                avg_weight = sum(b for _, b, _ in parts) / total_sino_size
            fm_loss = ((weighted_sq / (avg_weight * total_sino_size)) ** 0.5
                       / sigma_y)
            recon_l1 = sum(
                float(_memory_ledger.image_ell1(t))
                for t in flat_shards.tensors)
            es_rmse = (sq / total_sino_size) ** 0.5
        else:
            if isinstance(error_sinogram, _sharding.Shards):
                # A one shard container unwraps to the tensor it aliases.
                error_sinogram = error_sinogram.tensors[0]
                flat_recon = flat_recon.tensors[0]
            fm_loss, recon_l1, es_rmse = TomographyModel._vcd_iteration_stats(
                error_sinogram, flat_recon, sigma_y, weights)
        return fm_loss, recon_l1, es_rmse

    def initialize_prox(self, sinogram, weights=None, init_recon=None,
                        max_iterations=3, first_iteration=0,
                        logfile_path='~/.mbirtorch/logs/prox.log',
                        print_logs=True, rng=None):
        """
        Do the initialization of :meth:`prox_map` and store it in
        ``prox_data``, which later calls of ``prox_map`` reuse.

        Call this to initialize with one generator and then sweep with
        another, which is what makes a Plug-and-Play loop draw the same
        partitions whatever thread runs its first iteration.  ``prox_map``
        calls this itself on a call that asks for initialization or finds no
        cache.

        Args:
            See :meth:`prox_map` for arguments.  ``rng`` is the generator the
            partitions are drawn from; None uses the global np.random state.

        Returns:
            The tuple :meth:`initialize_recon` returns.
        """
        initialized = self.initialize_recon(
            sinogram, weights, init_recon, max_iterations, first_iteration,
            logfile_path=logfile_path, print_logs=print_logs, rng=rng)
        # The cache holds the last four entries: the partitions, the partition
        # sequence, the granularity, and the regularization parameters.
        self.prox_data = tuple(initialized[3:])
        return initialized

    def prox_map(self, prox_input, sinogram, sigma_prox=None, weights=None,
                 init_recon=None, do_initialization=True,
                 stop_threshold_change_pct=0.2, max_iterations=3,
                 first_iteration=0,
                 logfile_path='~/.mbirtorch/logs/prox.log', print_logs=True,
                 output_sharded=False, rng=None):
        """
        Proximal Map function for use in Plug-and-Play applications.  This
        function is similar to recon, but it essentially uses a prior with a
        mean of prox_input and a standard deviation of sigma_prox.

        Reproducibility note: the pixel partitions and the order in which the
        subsets are visited are drawn from numpy's global random number
        generator; call ``np.random.seed(seed)`` first for a reproducible
        result, or pass ``rng``.

        Args:
            prox_input (numpy or tensor or Shards): proximal map input with the
                same shape as the reconstruction.  The device form is accepted
                too, so a Plug-and-Play loop can feed back what a denoiser
                returned with ``output_sharded=True``, provided the two models
                share a device layout (see :meth:`configure_devices`).
            sinogram (numpy or tensor or Shards): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).  The device form
                as returned by :meth:`prepare_sino_for_devices` is accepted
                too, so a Plug-and-Play loop that prepares its sinogram once
                pays the host-to-device transfer once rather than on every
                call.
            sigma_prox (None or float, optional): standard deviation of the
                proximal map prior term.  If None, set automatically from the
                sinogram.  Defaults to None.
            weights (numpy or tensor or Shards, optional): 3D positive weights
                with the same shape as the sinogram, in a plain array or in the
                device form.  Defaults to None (all 1s).
            init_recon (numpy or tensor, optional): reconstruction used for
                initialization.  Defaults to None (determined by _vcd_recon).
            do_initialization (bool, optional): If True, initialize parameters
                (partitions and regularization) through :meth:`initialize_prox`.
                Set False if a previous prox_map call on this model, or a call
                to :meth:`initialize_prox`, already initialized this sinogram.
            stop_threshold_change_pct (float, optional): stop when the NMAE
                percent change drops below this value.  Defaults to 0.2.
            max_iterations (int, optional): maximum VCD iterations, counted
                from iteration 0: a call resuming at ``first_iteration=k``
                runs ``max_iterations - k`` iterations.  Defaults to 3.
            first_iteration (int, optional): cumulative iteration count for
                restarts.  The partition sequence is advanced by this amount
                (on the cached ``do_initialization=False`` path too), so a
                Plug-and-Play loop that passes the total number of prox
                iterations completed so far walks the sequence coarse to fine
                and, past its end, stays on its last (typically finest)
                entry.  Defaults to 0.
            logfile_path (str, optional): Path to the output log file ('~' expands to the
                user's home directory).  If None or empty, no log file is written.
                Defaults to '~/.mbirtorch/logs/prox.log'.  A Plug-and-Play loop
                that passes do_initialization=False after its first call keeps
                writing to the log that call opened, so the whole loop lands in
                one file.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the device form: a torch
                tensor on a single device, or a Shards container (one
                tensor per device) on a multi-device model.
            rng (numpy.random.Generator, optional): the generator every random
                draw of the call comes from, the partitions when this call
                initializes and the order of the subsets.  Defaults to None,
                the global np.random state.

        Returns:
            (recon, recon_dict): the reconstruction volume, and a dict
            with entries 'recon_params' (per-iteration traces and settings),
            'recon_log' (the run's log text), 'notes', and
            'model_params' (a snapshot of the model parameters).
        """
        # The initial reconstruction is checked against the shape of the whole volume, which a
        # sharded array does not have, so a sharded init_recon is refused.
        _sharding.reject_shards('prox_map', init_recon=init_recon)
        prior_loss = [0]
        if do_initialization or self.prox_data is None:
            (sinogram, weights, init_recon, partitions, partition_sequence,
             granularity, regularization_params) = self.initialize_prox(
                sinogram, weights, init_recon, max_iterations, first_iteration,
                logfile_path=logfile_path, print_logs=print_logs, rng=rng)
        else:
            (partitions, partition_sequence, granularity,
             regularization_params) = self.prox_data
            # The cache is dropped by a change of the device layout, and by nothing else: a
            # change of granularity, partition_sequence, or use_ror_mask leaves it in place.
            # The QGGMRFDenoiser cache checks those three and redraws when they move.
            # The cache holds the pixel partitions and the regularization estimates, which are
            # expensive.  The partition sequence is cheap and is recomputed here the same way.
            partition_sequence = vcd_utils.gen_partition_sequence(
                self.get_params('partition_sequence'),
                max_iterations=max_iterations)
            partition_sequence = partition_sequence[first_iteration:]
            # This pass skips the initialization, and with it the run header that reopens
            # the log file the previous pass closed, so the file is reopened here.
            self._reopen_log_file()

        # A supplied sigma_prox overrides the automatic value, and the
        # automatic value is restored at the end of the call.
        self_sigma_prox = self.get_params('sigma_prox')
        if sigma_prox is not None:
            regularization_params = dict(regularization_params,
                                         sigma_prox=sigma_prox)
            self.set_params(no_warning=True, sigma_prox=sigma_prox,
                            auto_regularize_flag=self.get_params('auto_regularize_flag'))

        with torch.no_grad():
            recon, loss_vectors = self._vcd_recon(
                sinogram, partitions, partition_sequence,
                stop_threshold_change_pct, weights=weights,
                init_recon=init_recon, prox_input=prox_input,
                first_iteration=first_iteration, rng=rng)

        partition_sequence = [int(val) for val in partition_sequence]
        fm_rmse = [float(val) for val in loss_vectors[0]]
        stop_pct = [100 * float(val) for val in loss_vectors[2]]
        alpha_values = [float(val) for val in loss_vectors[3]]
        delta_norm_per_slice = [[float(v) for v in row] for row in loss_vectors[4]]
        num_iterations = len(fm_rmse)
        recon_params = dict(zip(recon_param_names,
                                [num_iterations, granularity, partition_sequence,
                                 fm_rmse, prior_loss, regularization_params,
                                 stop_pct, alpha_values, delta_norm_per_slice]))
        self.set_params(no_warning=True, sigma_prox=self_sigma_prox)

        if logfile_path:
            self.logger.info('Logs written to {}'.format(
                os.path.abspath(os.path.expanduser(logfile_path))))
        for h in list(self.logger.handlers):
            h.flush()
        # The log file is closed here and reopened by the next pass.
        self.close_log_file()

        notes = 'Proximal map completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        return (recon if output_sharded else self._gather_recon(recon)), recon_dict

    @staticmethod
    def gen_weights(sinogram, weight_type):
        return vcd_utils.gen_weights(sinogram, weight_type)

    def scale_recon_shape(self, row_scale=1.0, col_scale=1.0, slice_scale=1.0):
        """
        Scale the reconstruction shape by the given scale factors.

        This can be used before starting a reconstruction to improve results
        when part of the object projects outside the detector.  The method
        updates the internal `recon_shape` parameter.

        For lateral field-of-view truncation (flagged by the "Lateral FoV
        truncation detected" warning), use ``scale_recon_shape(s, s)`` with
        ``s`` typically chosen as ``s >= 1.1``.

        Args:
            row_scale (float): Scale factor for the number of recon rows.
            col_scale (float): Scale factor for the number of recon columns.
            slice_scale (float): Scale factor for the number of recon slices.

        Returns:
            tuple[int, int, int]: pixels added to (rows, columns, slices).
        """
        old_rows, old_cols, old_slices = self.get_params('recon_shape')
        new_rows = int(old_rows * row_scale)
        new_cols = int(old_cols * col_scale)
        new_slices = int(old_slices * slice_scale)
        self.set_params(recon_shape=(new_rows, new_cols, new_slices))
        return new_rows - old_rows, new_cols - old_cols, new_slices - old_slices

    def recon_slice_z(self, slice_indices=None):
        """The axial coordinate, in ALU, of the center of each recon slice.

        Slices are spaced by the slice pitch ``voxel_slice_aspect * delta_voxel``
        and centered on ``recon_slice_offset``, which is zero for a geometry
        that has no such parameter.  This is the one host-side statement of
        the map the projectors use: the compiled cone and multiaxis bodies
        write the same expression in ``_cone_vertical_affine`` and
        ``_multiaxis_vertical_terms``, where a Python call cannot be traced.

        Args:
            slice_indices (int, sequence of int, or None): the slices to map.
                None (the default) maps every slice.

        Returns:
            float or ndarray: the coordinate of each requested slice.
        """
        recon_shape = self.get_params('recon_shape')
        delta_voxel, voxel_slice_aspect = self.get_params(['delta_voxel', 'voxel_slice_aspect'])
        offset = self.get_params('recon_slice_offset') if 'recon_slice_offset' in self.params else 0.0
        num_slices = int(recon_shape[2])
        k = np.arange(num_slices) if slice_indices is None else np.asarray(slice_indices)
        return voxel_slice_aspect * delta_voxel * (k - (num_slices - 1) / 2.0) + offset

    def _fractional_slice_index(self, z):
        """Return the fractional reconstruction slice index of the axial
        coordinate ``z``.  This inverts :meth:`recon_slice_z` without rounding.
        The argument may be a float, a numpy array, or a torch tensor, and the
        result has the same form."""
        recon_shape = self.get_params('recon_shape')
        delta_voxel, voxel_slice_aspect = self.get_params(['delta_voxel', 'voxel_slice_aspect'])
        offset = self.get_params('recon_slice_offset') if 'recon_slice_offset' in self.params else 0.0
        num_slices = int(recon_shape[2])
        return (z - offset) / (voxel_slice_aspect * delta_voxel) + (num_slices - 1) / 2.0

    def nearest_recon_slice(self, z):
        """The index of the recon slice whose center is nearest the axial
        coordinate ``z`` in ALU, clipped to the volume.  The inverse of
        :meth:`recon_slice_z`."""
        num_slices = int(self.get_params('recon_shape')[2])
        index = int(round(float(self._fractional_slice_index(z))))
        return min(max(index, 0), num_slices - 1)

    def reshape_recon(self, recon):
        """Reshape a recon-like array to the model's ``recon_shape``.

        The target shape names the WHOLE volume's slice count, which no single
        shard has, so a divided array is refused rather than reshaped shard by
        shard against a slice count that is not its own.
        """
        _sharding.reject_shards('reshape_recon', recon=recon)
        recon_shape = self.get_params('recon_shape')
        return recon.reshape(recon_shape)

    # ── model description and HDF5 persistence ────────────────────────────────

    def get_all_params(self):
        """
        Return this model's parameters as ``(required_params, optional_params, regularization)``.

        This is the single source of truth for reading a model's parameters back out.  The three
        dicts partition the parameters so a caller can reconstruct or serialize the model and choose
        which parts to apply:

        * **required_params** -- the geometry arguments the model constructor takes, with the
          view-dependent arguments reconstructed from storage (e.g. cone's ``angles`` and
          ``helical_z_shifts``), plus a ``geometry_type`` entry so the model class can be
          resolved.  The execution-environment constructor arguments (``view_batch_size``,
          ``compile_mode``) are not model parameters and are excluded.
        * **optional_params** -- the remaining geometry/detector parameters that are applied with
          ``set_params`` (detector pitches, offsets, ``delta_voxel``, ``recon_shape``, voxel aspects).
        * **regularization** -- the regularization parameters (``sigma_y``, ``sigma_x``,
          ``sigma_prox``, ``snr_db``, ``sharpness``, ``auto_regularize_flag``), separated so a
          consumer can drop them and let them be re-chosen at reconstruction time.

        Returns:
            tuple: ``(required_params, optional_params, regularization)`` -- three dicts of values.
        """
        import inspect

        regularization_names = _AUTO_REGULARIZATION_PARAM_NAMES + (
            'snr_db', 'sharpness', 'auto_regularize_flag')
        # These parameters are derived again when a model is constructed.
        construction_derived_names = ('geometry_type', 'view_params_name', 'file_format',
                                      'version', 'use_gpu')
        # These constructor arguments describe the execution environment rather
        # than the model.
        environment_args = ('self', 'view_batch_size', 'compile_mode')

        ctor_names = [n for n in inspect.signature(type(self).__init__).parameters
                      if n not in environment_args]
        view_params_name = self.get_params('view_params_name')
        view_array = np.asarray(self.get_params(view_params_name))

        required_params = {}
        for name in ctor_names:
            if name in self.params:
                required_params[name] = self.get_params(name)
            elif name == 'angles':
                required_params[name] = view_array[:, 0] if view_array.ndim == 2 else view_array
            elif name == 'helical_z_shifts' and view_array.ndim == 2:
                required_params[name] = view_array[:, 1]

        optional_params = {}
        for key in self.params:
            if (key in ctor_names or key == view_params_name
                    or key in construction_derived_names or key in regularization_names):
                continue
            optional_params[key] = self.get_params(key)

        required_params['geometry_type'] = str(type(self))

        regularization = {name: self.get_params(name)
                          for name in regularization_names if name in self.params}

        return required_params, optional_params, regularization

    def get_recon_dict(self, recon_params=None, notes=None, save_log=True, save_model=True, str_format=False):
        """
        Collect the recon parameters, logs, notes, and optionally all model parameters into a dict
        with entries 'recon_params', 'recon_log', 'notes', and 'model_params'.  This dict can be used with
        :func:`mbirtorch.view_utils.slice_viewer` and :meth:`TomographyModel.save_recon_hdf5`.
        By default the entries hold their original values; str_format=True serializes each top-level
        entry to a string.

        Args:
            recon_params (dict, optional): dict of reconstruction parameters. Defaults to None.
            notes (str, optional): User-supplied notes to attach to the dataset. Defaults to None.
            save_log (bool, optional): If True, saves the internal log buffer (if available). Defaults to True.
            save_model (bool, optional): If True, saves the model parameters. Defaults to True.
            str_format (bool, optional): If True, then each top level entry is serialized to a string.

        Returns:
            dict: A dict with entries
                 - 'recon_params'
                 - 'notes'
                 - 'recon_log'
                 - 'model_params'.

        Example:
            >>> recon, recon_dict = ct_model.recon(sinogram)
            >>> print(recon_dict['recon_log'])
        """
        recon_dict = dict()
        if recon_params is None:
            recon_dict['recon_params'] = "# Recon params not saved."
        else:
            recon_dict['recon_params'] = recon_params

        log_buffer = getattr(self, 'log_buffer', None)
        if log_buffer is None or not save_log:
            recon_dict['recon_log'] = "# Log info not saved."
        else:
            recon_dict['recon_log'] = log_buffer.getvalue()

        if notes is None:
            notes = '# No notes saved'
        recon_dict['notes'] = notes

        if save_model:
            recon_dict['model_params'] = {k: v.val for k, v in self.params.items()}
        else:
            recon_dict['model_params'] = '# Model not saved'

        if str_format:
            from .view_utils import convert_subdicts_to_strings
            recon_dict = convert_subdicts_to_strings(recon_dict)

        return recon_dict

    def save_recon_hdf5(self, filepath, recon, recon_dict=None):
        """
        Save the reconstruction array and optionally the recon_dict from :meth:`~mbirtorch.TomographyModel.recon`.

        This method creates a file that contains a single dataset named 'recon', with the entries in recon_dict
        serialized to strings and saved as hdf5 dataset attributes.

        The resulting file can be loaded with :meth:`load_recon_hdf5` or :func:`mbirtorch.view_utils.slice_viewer`.

        Args:
            filepath (str or Path): Path to the output HDF5 file. Should typically end with a .h5 extension.
            recon (array-like): The reconstruction volume as a NumPy array, torch tensor, or the
                sharded device form from ``recon(..., output_sharded=True)``.
            recon_dict (dict or None, optional): The dictionary of recon attributes from :meth:`get_recon_dict`

        Raises:
            Exception: If saving the file or directory creation fails.

        Example:
            >>> recon, recon_dict = ct_model.recon(sinogram)
            >>> recon_dict['notes'] += 'Test scan'
            >>> ct_model.save_recon_hdf5("output/my_recon.h5", recon, recon_dict=recon_dict)
        """
        from .utilities import save_data_hdf5, _to_host
        arr = _to_host(recon)
        save_data_hdf5(filepath, arr, 'recon', recon_dict)

        if self.logger:
            self.logger.info(f"Saved reconstruction and params to '{filepath}'")

    @staticmethod
    def load_recon_hdf5(filepath):
        """
        This function loads a numpy array stored in an HDF5 file created by :meth:`~mbirtorch.TomographyModel.save_recon_hdf5`.
        It also loads any associated attribute dict.

        Args:
            filepath (str): Path to the HDF5 file containing the reconstructed volume.

        Returns:
            (recon, recon_dict)
                - recon (ndarray): The array saved by save_recon_hdf5()
                - recon_dict (dict): A dict with the same entries as :meth:`get_recon_dict`, with
                  each value as the string it was stored as in the HDF5 attributes

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If more than one dataset is found in the file.

        Example:
            >>> recon, recon_dict = ct_model.load_recon_hdf5("output/recon_volume.h5")
            >>> recon.shape
            (64, 256, 256)
        """
        from .utilities import load_data_hdf5
        recon, recon_dict = load_data_hdf5(filepath)
        return recon, recon_dict
