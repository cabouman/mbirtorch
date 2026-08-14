"""TomographyModel: the VCD reconstruction loop, FBP, and projector wrappers.

Ported from mbirjax.tomography_model: the VCD loop (numpy partitions, subset updater, on-device line search,
positivity), FBP via torch.fft, the auto-regularization chain, the placement
placement and gather functions, the multi-device VCD loop, and the public
numpy-at-the-boundary API.  The loop's working state is uniformly per-device (:class:`_sharding.Shards`;
a single device is the trivial one-shard case, following mbirjax's
everything-is-sharded principle).  Deliberately not (yet) ported: the tile
policy, save/load, and the memory-management machinery jax needed (donation
and .delete() become plain in-place ops here; the checkpoint-resume path
mutates the caller arrays in place, advertised, instead of donating).

Value-parity intent: every formula follows the mbirjax source read 2026-08-04,
in the same order of operations, so a seeded run matches a seeded mbirjax run
iteration for iteration (the convergence-parity gate in tests/test_vs_goldens).
"""

import contextlib
import datetime
import io
import logging
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

# ── the multi-device forward's column gather ─────────────────────────────────
# Pixel columns per gathered cylinder in the multi-device forward; bounds the
# cross-device transient.  Set forward_project_pixel_batch on the model to
# override.
FORWARD_PIXEL_BATCH = 8192

# This environment variable forces the column gather on or off regardless
# of the model attribute.  It is read per call.
COLUMN_GATHER_ENV_VAR = 'MBIRTORCH_FORWARD_COLUMN_GATHER'
_COLUMN_GATHER_ON_VALUES = ('1', 'true', 'yes', 'on')
_COLUMN_GATHER_OFF_VALUES = ('0', 'false', 'no', 'off')


# ── compiled updater glue (module level, one compile per process) ─────────────
# Fused forms of the per-subset arithmetic.  _apply_update mutates its two
# state tensors in place but returns them; call sites rebind through the
# returns.
def _diagonal_update_direction(forward_grad, prior_grad, forward_hess, prior_hess):
    # The base preconditioned direction (the mbirjax module-level jitted helper's
    # torch analog), fused so the sums and divide are one kernel.
    return -((forward_grad + prior_grad) / (forward_hess + prior_hess))


def _prior_line_terms(prior_grad, prior_hess, delta):
    return (torch.sum(prior_grad * delta),
            torch.sum(prior_hess * delta * delta))


def _forward_lin_quad_const(weighted_error_sinogram, delta_sinogram, fm_constant):
    return (fm_constant * torch.sum(weighted_error_sinogram * delta_sinogram),
            fm_constant * torch.sum(delta_sinogram * delta_sinogram))


def _forward_lin_quad_weighted(error_sinogram, delta_sinogram, weights, fm_constant):
    # weighted_error = weights * error is fused into the reductions here, so no
    # sinogram-sized weighted-product transient is materialized per subset.
    return (fm_constant * torch.sum(weights * error_sinogram * delta_sinogram),
            fm_constant * torch.sum(delta_sinogram * delta_sinogram * weights))


def _apply_update(flat_recon, error_sinogram, pixel_indices, delta_scaled,
                  alpha, delta_sinogram):
    # In-place state application: recon scatter-add, per-slice sumsq, the error
    # sinogram FMA, and the ell1 reduction, in one compiled region.  The state
    # tensors are returned (same storage, no copy) so callers rebind
    # functionally rather than relying on the side effect.
    flat_recon.index_add_(0, pixel_indices, delta_scaled)
    delta_sumsq = torch.sum(delta_scaled * delta_scaled, dim=0)
    error_sinogram.sub_(alpha * delta_sinogram)
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


class TomographyModel(ParameterHandler):
    """Base class: geometry classes supply their per-view-batch projection
    bodies (``_view_batch_bodies`` / ``_view_batch_args``), the psf radius,
    and the auto recon geometry; this class owns the projection wrappers and
    the VCD loop."""

    def __init__(self, sinogram_shape, view_batch_size=None,
                 compile_mode='auto', **kwargs):
        super().__init__()
        # Device state resolves lazily on first use, so inspecting a model or
        # calling configure_devices first never touches an unchosen device.
        self._torch_device = None
        self._sino_placement = None
        self._recon_placement = None
        self._projector_functions = None
        # Views per body call in the batched drivers.  None means the
        # per-body default, and the driver's transient budget may cap the
        # realized batch below either.
        self.view_batch_size = view_batch_size
        # torch.compile of the hot chains: 'auto' compiles, 'off' is pure eager.
        self.compile_mode = compile_mode
        # Cached prox initialization, so a Plug-and-Play loop pays
        # initialize_recon once.
        self.prox_data = None
        # Device-layout caches (see _invalidate_device_caches).
        self._qggmrf_interface_masks_cache = None
        self._dc_damping_cache = None
        self.dev2dev_safe = True     # probed for real in configure_devices
        # False once configure_devices is called: an explicit layout is the
        # caller's, permanently.
        self.device_layout_is_automatic = True
        # Device counts the automatic choice turned down, for the run log.
        self.device_choice_rejections = []
        # Speed-floor bookkeeping for the run log (see
        # _speed_ordered_candidates).
        self._speed_floor_fallback = None
        self._speed_floor_held = None
        # Memory-preflight knobs; the margin covers what the ledger cannot
        # see (fragmentation, non-torch CUDA workspaces).
        self.skip_memory_preflight = False
        self.memory_preflight_margin = 0.15
        # These two exist for a harness to read.  Nothing in the library
        # reads them back.
        self.last_memory_ledger = None
        self.last_memory_calibration = None
        # The per-device thread pool is owned by vcd_recon and is None
        # outside a recon.
        self._per_device_pool = None
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
        self.verify_valid_params()

    @property
    def compile_enabled(self):
        return self.compile_mode != 'off'

    # ── lazily resolved device state ──────────────────────────────────────────
    # Each of these resolves on first read and is plain-assignable, so
    # configure_devices and the automatic widening keep setting them directly.
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

    # Placement source of truth: sino-like arrays shard by VIEW (axis 0),
    # recon-like arrays by SLICE (the last axis).
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
        """The geometry's per-view-batch projection bodies, (forward, back):
        MODULE-LEVEL pure functions taking parameter VALUES (built by
        :meth:`_view_batch_args`), never bound methods -- bound methods would
        pin the model in the module-level compile cache, and parameter reads
        inside the compiled region would make dynamo trace the parameter
        machinery.  The driver compiles one instance per device."""
        raise NotImplementedError(
            f'{type(self).__name__} defines no per-view-batch projection '
            'bodies.')

    def _view_batch_args(self):
        """The eager argument dict for this geometry's bodies: every
        parameter read happens HERE, outside the traced region, per call --
        never frozen at build time (the stale-bind lesson)."""
        raise NotImplementedError

    def _transient_cols(self, band_cols):
        """The column count of this geometry's dominant per-view transient,
        for the driver's view-batch budget.  The base tracks the runtime
        band length (single-fan geometries); a two-fan geometry -- one
        whose slice band projects onto many detector rows, like cone --
        overrides with its params-derived width.  Geometry-owned because the value is
        calibrated: changing it silently changes batch sizes, float
        summation order, and measured peaks."""
        return band_cols

    def direct_recon(self, sinogram, filter_name=None, output_sharded=False):
        """
        Do a direct (non-iterative) reconstruction, typically using a form of
        filtered backprojection.  The implementation details are geometry
        specific, and direct_recon may not be available for all geometries.

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            filter_name (string or None, optional): The name of the filter to
                use.  Defaults to None, in which case the geometry-specific
                method chooses a default, typically 'ramp'.
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the internal device form
                (slice shards on a multi-device model; on a single-device
                model the output is the same tensor either way).

        Returns:
            recon (numpy or tensor): The reconstructed volume.
        """
        raise NotImplementedError

    def split_sino_recon(self, sino, weights=None, half_overlap=5, init_recon=None,
                         max_iterations=15, stop_threshold_change_pct=0.2,
                         first_iteration=0, compute_prior_loss=False,
                         logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
                         align_split_grid=False):
        """
        Perform MBIR reconstruction with about half the memory of :meth:`recon`
        by splitting the detector rows into two overlapping halves,
        reconstructing each half separately, and stitching the results.  The
        output is approximately equal to the output of :meth:`recon`.

        The split arithmetic is geometry specific, and split_sino_recon may
        not be available for all geometries; geometries without an
        implementation raise ``NotImplementedError``.

        Args:
            sino (ndarray): Full sinogram of shape (num_views, num_rows, num_cols).
            weights (ndarray, optional): Sinogram weights with the same shape
                as ``sino``.
            half_overlap (int, optional): Number of overlapping detector rows
                kept past the split in each half.  Defaults to 5.
            init_recon (optional): Same as in :meth:`recon`.
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
        """
        raise NotImplementedError(
            f'split_sino_recon is not implemented for {type(self).__name__}.')

    # ── projection wrappers (numpy at the public boundary) ────────────────────
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
        """The thread pool for a banded projection's per-band fan-outs:
        reuse the recon-loop pool when one is active (vcd_recon creates it
        once for the whole loop), else a private pool for this call."""
        if self._per_device_pool is not None:
            return contextlib.nullcontext(self._per_device_pool)
        return _sharding.device_pool(n)

    @staticmethod
    def _slice_band_length(slices_per_dev, n_dev, num_pixels, fixed_band=None):
        """Band length B for streaming the slice axis in the banded drivers.

        The default is one band per slice-owner (the whole shard): sub-bands
        were measured to be slower, and a single device never runs the banded
        drivers.  A smaller B reduces per-band memory; set
        ``forward_project_slice_band`` / ``back_project_slice_band`` on the
        model to opt in.  The result is capped at slices_per_dev so a band
        never crosses a slice-owner boundary."""
        b = fixed_band if fixed_band else slices_per_dev
        return min(int(b), slices_per_dev)

    def _column_gather_forward(self):
        """Whether the multi-device forward gathers pixel COLUMNS instead of
        walking slice bands.

        The gather runs when the geometry declares ``column_gather_geometry``
        (cone and parallel do, on measurement; translation and multiaxis are
        untimed) and the ``forward_column_gather`` switch is not False.  The
        environment variable overrides the switch either way."""
        if not self.column_gather_geometry:
            return False
        override = os.environ.get(COLUMN_GATHER_ENV_VAR, '').strip().lower()
        if override in _COLUMN_GATHER_ON_VALUES:
            return True
        if override in _COLUMN_GATHER_OFF_VALUES:
            return False
        switch = getattr(self, 'forward_column_gather', None)
        return True if switch is None else bool(switch)

    def _forward_pixel_batch(self):
        """How many pixel columns one gathered cylinder covers.

        :data:`FORWARD_PIXEL_BATCH` carries the value and its provenance.
        ``forward_project_pixel_batch`` on the model overrides it, the same
        way ``forward_project_slice_band`` overrides the band rule.  The
        memory ledger calls THIS method rather than re-deriving the number,
        so a changed default cannot leave the charge behind."""
        fixed = getattr(self, 'forward_project_pixel_batch', None)
        return max(1, int(fixed)) if fixed else FORWARD_PIXEL_BATCH

    @staticmethod
    def _balanced_slice_bounds(extent, band_len):
        """Tile ``[0, extent)`` into balanced bands no longer than
        ``band_len``: the fewest bands, lengths as equal as possible
        (differing by at most 1), non-overlapping -- no slice is ever
        recomputed."""
        num_bands = -(-extent // band_len)            # ceil division
        base, rem = divmod(extent, num_bands)
        bounds, start = [], 0
        for k in range(num_bands):
            length = base + (1 if k < rem else 0)
            bounds.append((start, start + length))
            start += length
        return bounds

    def _banded_setup(self, pixel_indices):
        """Shared setup for the banded sharded projectors: per-owner view
        spans (real views only -- a padded view has no angle), recon band
        ranges, and the pixel indices placed once per device."""
        sp, rp = self.sino_placement, self.recon_placement
        if type(self)._view_batch_bodies is TomographyModel._view_batch_bodies:
            raise NotImplementedError(
                f'{type(self).__name__} has no per-view-batch projection '
                'bodies, so the banded multi-device drivers cannot run.')
        # (start, start + n_valid) real-view spans plus each owner's full
        # (padded) block length, in device order; band_ranges carry each
        # slice-owner's real count so the drivers can skip all-padding bands
        # (an owner may own NO real slices -- the thin-volume extension).
        view_spans = [(v0, v0 + n_valid, v1 - v0)
                      for _, (v0, v1), n_valid in sp.padded_shard_ranges()]
        band_ranges = rp.padded_shard_ranges()
        idx_per_dev = [torch.as_tensor(pixel_indices, dtype=torch.int64).to(d)
                       for d in sp.devices]
        return sp, rp, view_spans, band_ranges, idx_per_dev

    def _sparse_forward_project_sharded(self, voxel_shards, pixel_indices):
        """The banded sharded forward: visit slice-owners in global slice
        order, broadcast each band to every view-owner, forward-project each
        owner's OWN views (one producer per detector row -- no reduce), and
        concatenate the row-bands into per-view-owner sinogram shards.  A
        trivial placement is the plain driver, wrapped.  Padded views are
        never projected; their tails are zero-filled after assembly."""
        if voxel_shards.placement.is_trivial:
            return _sharding.Shards(
                [self.projector_functions._sparse_forward_project_single_device(
                    voxel_shards.tensors[0], pixel_indices)],
                self.sino_placement)
        if self._column_gather_forward():
            return self._sparse_forward_project_columns(voxel_shards,
                                                        pixel_indices)
        sp, rp, view_spans, band_ranges, idx_per = self._banded_setup(pixel_indices)
        pf = self.projector_functions
        aligned = self.rows_track_slices
        view_bands = [[] for _ in sp.devices]
        partial_shards = None
        num_rows = int(self.get_params('sinogram_shape')[1])
        num_channels = int(self.get_params('sinogram_shape')[2])
        num_pixels = int(idx_per[0].shape[0])
        fixed_band = getattr(self, 'forward_project_slice_band', None)
        # Bands go only to view-owners that project something: an owner with
        # no real views (the sparse-view extension) receives no copies and
        # produces empty row-bands, so its block assembles as pure zeros.
        proj_devs = [d for i, d in enumerate(sp.devices)
                     if view_spans[i][1] > view_spans[i][0]]
        with self._band_pool(sp.n_devices) as pool:
            for oi, (odev, (s0, s1), band_valid) in enumerate(band_ranges):  # oi = owner index, odev = owner device
                # Stream the owner's shard in sub-bands: each broadcast copy
                # and per-band transient is band-sized, not shard-sized.
                band_len = self._slice_band_length(
                    s1 - s0, sp.n_devices, num_pixels, fixed_band)
                for (l0, l1) in self._balanced_slice_bounds(s1 - s0, band_len):
                    if l0 >= band_valid:
                        # An all-padding sub-band (zero voxels) contributes
                        # nothing: skip the broadcast and the projector call.
                        # The aligned form still owes its detector rows --
                        # append them as zeros so the row tiling stays exact.
                        if aligned:
                            for i in range(sp.n_devices):
                                nv = view_spans[i][1] - view_spans[i][0]
                                view_bands[i].append(torch.zeros(
                                    (nv, l1 - l0, num_channels),
                                    dtype=voxel_shards.dtype,
                                    device=sp.devices[i]))
                        continue
                    band = voxel_shards.tensors[oi][:, l0:l1]
                    copies = _sharding.broadcast_band_to_views(
                        band, proj_devs, self.dev2dev_safe)
                    if aligned:
                        # Rows track slices 1:1 (parallel): each band yields
                        # the matching ROW-band, a single producer per row --
                        # concat (owners in slice order, sub-bands in order).
                        # A no-view owner yields an empty row-band.
                        row_bands = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_forward_project_view_range(
                                    copies[d], idx_per[i], view_spans[i][:2],
                                    dev_index=i)
                                if view_spans[i][1] > view_spans[i][0] else
                                torch.zeros((0, l1 - l0, num_channels),
                                            dtype=voxel_shards.dtype,
                                            device=d)),
                            executor=pool)
                        for i in range(sp.n_devices):
                            view_bands[i].append(row_bands[i])
                    else:
                        # A slice band spreads over MANY rows (cone): each
                        # band yields a full-row PARTIAL shard -- accumulate.
                        # A no-view owner yields an empty partial.
                        partials = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_forward_project_view_range(
                                    copies[d], idx_per[i], view_spans[i][:2],
                                    slice_start=s0 + l0, dev_index=i)
                                if view_spans[i][1] > view_spans[i][0] else
                                torch.zeros((0, num_rows, num_channels),
                                            dtype=voxel_shards.dtype,
                                            device=d)),
                            executor=pool)
                        if partial_shards is None:
                            partial_shards = list(partials)
                        else:
                            for i in range(sp.n_devices):
                                partial_shards[i].add_(partials[i])
                        # This release must come before the next band's
                        # run_per_device call.  Without it, this band's
                        # partial stays live on every device through the next
                        # projection.
                        partials = None
        if aligned:
            tensors = [b[0] if len(b) == 1 else torch.cat(b, dim=1)
                       for b in view_bands]
        else:
            tensors = partial_shards
        if sp.is_padded:
            # Zero-fill each owner's padded view tail up to its block length
            # (built on the owner device; only the last owner has a tail).
            tensors = [
                t if t.shape[0] == block else torch.cat(
                    [t, torch.zeros((block - t.shape[0],) + tuple(t.shape[1:]),
                                    dtype=t.dtype, device=t.device)])
                for t, (_v0, _v1, block) in zip(tensors, view_spans)]
        return _sharding.Shards(tensors, sp)

    def _sparse_forward_project_columns(self, voxel_shards, pixel_indices):
        """The multi-device forward as a pixel-batched column gather: each
        view-owner walks the pixel axis in batches, gathers each batch's
        full-height cylinder from every slice-owner, and makes ONE projector
        call per batch over its own views and the whole slice range.  The
        alternative to the banded walk, for the geometries
        :meth:`_column_gather_forward` admits: banding costs a two-fan
        geometry a full call per band, and a row-aligned kernel runs about
        twice as fast on full-width blocks.  The operator is unchanged, so
        the forward stays the adjoint of the sharded back, and the back
        driver is untouched.  ``forward_project_slice_band`` has no effect
        here.  The pixel batch bounds the transfer instead, and the memory
        ledger charges it (COLUMN_GATHER_RESIDENTS)."""
        sp, rp, view_spans, _band_ranges, idx_per = self._banded_setup(
            pixel_indices)
        pf = self.projector_functions
        num_channels = int(self.get_params('sinogram_shape')[2])
        # Block height per call.  A row-aligned body sizes its output by the
        # gathered cylinder (the padded device-form slice axis); a two-fan
        # body returns the real detector rows.
        num_rows = (int(rp.padded_size) if self.rows_track_slices
                    else int(self.get_params('sinogram_shape')[1]))
        num_pixels = int(idx_per[0].shape[0])
        shards = voxel_shards.tensors        # in device = global slice order
        pixel_batch = self._forward_pixel_batch()
        batch_bounds = [(p0, min(p0 + pixel_batch, num_pixels))
                        for p0 in range(0, num_pixels, pixel_batch)]

        def worker(i, dev):
            v0, v1, _block = view_spans[i]
            if v1 <= v0:
                # A view-owner with no real views (the sparse-view extension)
                # produces an empty block, which assembles as pure zeros.
                return torch.zeros((0, num_rows, num_channels),
                                   dtype=voxel_shards.dtype, device=dev)
            local_idx = idx_per[i]
            owned = None

            def gather(k):
                p0, p1 = batch_bounds[k]
                return _sharding.gather_column_band_async(
                    shards, p0, p1, dev, self.dev2dev_safe)

            # The batch after the one being projected, gathered ahead of it.
            # A pass of one batch has nothing to gather ahead, and no pixels
            # at all leaves this empty.
            ahead = gather(0) if batch_bounds else None
            for k, (p0, p1) in enumerate(batch_bounds):
                full_cyl, ready = ahead
                # The next batch's gather is issued before this batch is
                # projected.  The copies run on separate per-device streams,
                # so they move while this projection runs.  Each batch
                # carries an event, and the wait below keeps a projection
                # from starting before its batch's copies finish.  Off CUDA
                # the gather is synchronous and the wait does nothing.
                ahead = gather(k + 1) if k + 1 < len(batch_bounds) else None
                _sharding.wait_for_column_band(dev, ready)
                # The first batch's projection allocates the owner's block.
                # Later batches add into that block inside the projector's
                # view loop, which saves a full-block allocation and pass per
                # batch.  The summands and their order are unchanged, so the
                # result is bit for bit the same.
                if owned is None:
                    owned = pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i)
                else:
                    pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i, accumulate_into=owned)
                # The cylinder is released once its projection is issued.
                # The next batch is already resident, which is the third
                # cylinder the memory ledger charges.
                full_cyl = None
            if owned is None:
                # No pixels at all: the owner still owes its views' block,
                # and the banded form would have produced it as zeros too.
                owned = torch.zeros((v1 - v0, num_rows, num_channels),
                                    dtype=voxel_shards.dtype, device=dev)
            return owned

        # One fan-out covers the whole call: the pixel loop inside the worker
        # issues each device's gathers from the thread that consumes them.
        # Both slice-owners and view-owners get their copy streams ordered.
        gather_devices = (list(voxel_shards.placement.devices)
                          + list(sp.devices))
        _sharding.open_copy_streams(gather_devices)
        try:
            with self._band_pool(sp.n_devices) as pool:
                tensors = _sharding.run_per_device(sp.devices, worker,
                                                   executor=pool)
        finally:
            # Closed even if a worker raised: copies that were already issued
            # are still in flight, and the shards they read must not be
            # overwritten under them.
            _sharding.close_copy_streams(gather_devices)
        if sp.is_padded:
            # The banded form's own tail fill: zero-fill each owner's padded
            # view tail up to its block length.
            tensors = [
                t if t.shape[0] == block else torch.cat(
                    [t, torch.zeros((block - t.shape[0],) + tuple(t.shape[1:]),
                                    dtype=t.dtype, device=t.device)])
                for t, (_v0, _v1, block) in zip(tensors, view_spans)]
        return _sharding.Shards(tensors, sp)

    def _sparse_back_project_sharded(self, sino_shards, pixel_indices,
                                     coeff_power=1):
        """The banded sharded back (the forward's adjoint): every view-owner
        back-projects its views onto each slice band (a PARTIAL (P, L) each),
        and the partials sum onto the band's slice-owner.  A trivial
        placement is the plain driver, wrapped.  Only real views are
        projected, and the padded slice tail is re-zeroed after the reduce
        (a gather DOES land real data in padded positions)."""
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
            for oi, (odev, (s0, s1), band_valid) in enumerate(band_ranges):
                # Stream the owner's band in sub-bands: each view-owner
                # partial and the owner's reduce gather are band-sized.
                band_len = self._slice_band_length(
                    s1 - s0, sp.n_devices, num_pixels, fixed_band)
                owner_parts = []
                for (l0, l1) in self._balanced_slice_bounds(s1 - s0, band_len):
                    if l0 >= band_valid:
                        # All-padding sub-band: its result is forced to zero
                        # after the reduce anyway, so produce the zeros
                        # directly on the owner and skip the projector pass.
                        owner_parts.append(torch.zeros(
                            (num_pixels, l1 - l0),
                            dtype=sino_shards.dtype, device=odev))
                        continue
                    # A view-owner with no real views (sparse-view extension)
                    # contributes nothing: skip its projector call and drop
                    # it from the band reduce.
                    if aligned:
                        partials = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_back_project_view_range(
                                    sino_shards.tensors[i][
                                        :view_spans[i][1] - view_spans[i][0],
                                        s0 + l0:s0 + l1, :],
                                    idx_per[i], view_spans[i][:2],
                                    coeff_power=coeff_power, dev_index=i)
                                if view_spans[i][1] > view_spans[i][0]
                                else None),
                            executor=pool)
                    else:
                        partials = _sharding.run_per_device(
                            sp.devices,
                            lambda i, d: (
                                pf.sparse_back_project_view_range(
                                    sino_shards.tensors[i][
                                        :view_spans[i][1] - view_spans[i][0]],
                                    idx_per[i], view_spans[i][:2],
                                    slice_start=s0 + l0, band_slices=l1 - l0,
                                    coeff_power=coeff_power, dev_index=i)
                                if view_spans[i][1] > view_spans[i][0]
                                else None),
                            executor=pool)
                    owner_parts.append(_sharding.sum_band_to_owner(
                        [p for p in partials if p is not None], odev,
                        self.dev2dev_safe))
                    # This release must come before the next band's
                    # run_per_device call.  Without it, this band's partial
                    # stays live on every device through the next projection.
                    partials = None
                recon_tensors.append(owner_parts[0] if len(owner_parts) == 1
                                     else torch.cat(owner_parts, dim=1))
        if rp.is_padded:
            for oi, (_dev, (s0, s1), n_valid) in enumerate(
                    rp.padded_shard_ranges()):
                if n_valid < s1 - s0:
                    recon_tensors[oi][:, n_valid:] = 0
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
        real sizes stale, silently truncating sharded arrays."""
        devices = self.sino_placement.devices
        sinogram_shape, recon_shape = self.get_params(
            ['sinogram_shape', 'recon_shape'])
        self.sino_placement = _sharding.Placement(
            devices, axis=0, real_size=int(sinogram_shape[0]))
        self.recon_placement = _sharding.Placement(
            devices, axis=-1, real_size=int(recon_shape[2]))
        self._check_no_empty_shard()
        self._invalidate_device_caches()
        if self._projector_functions is not None:
            self.create_projectors()

    def _check_no_empty_shard(self):
        """Refuse a device layout that would leave a device idle on BOTH
        axes.  A device that is all-padding on ONE axis is legal: thin
        volumes and sparse views still give it work, and the padding
        invariants keep the empty axis inert.  A device that is all-padding
        on both axes would do nothing, so that layout is refused."""
        sp, rp = self.sino_placement, self.recon_placement
        if sp.padded_size is None or rp.padded_size is None:
            return
        sino_valid = [n for _, _, n in sp.padded_shard_ranges()]
        recon_valid = [n for _, _, n in rp.padded_shard_ranges()]
        for i, (sv, rv) in enumerate(zip(sino_valid, recon_valid)):
            if sv <= 0 and rv <= 0:
                raise ValueError(
                    f'{sp.n_devices} devices would leave device {i} with no '
                    f'real views AND no real slices ({sp.real_size} views, '
                    f'{rp.real_size} slices); use at most '
                    f'{max(sp.real_size, rp.real_size)} devices for this '
                    f'geometry.')

    def _invalidate_device_caches(self):
        """Drop every cache keyed to the device layout or geometry, so no
        consumer can bind stale device-resident state."""
        self.prox_data = None
        self._qggmrf_interface_masks_cache = None
        self._dc_damping_cache = None

    def _qggmrf_interface_masks(self):
        """Per-device qGGMRF interface masks for a padded slice axis, or None.

        Interface j of a shard starting at global slice g0 is valid iff
        ``g0 + j < num_real_slices``; masking reproduces the reflected
        boundary at the last REAL slice even mid-shard.  Built once and
        cached: per-subset builds would re-pay a host->device transfer
        thousands of times.

        Returns:
            list ((local_slices+1,) float32 tensor per device), or None.
        """
        if not self.recon_placement.is_padded:
            return None
        if self._qggmrf_interface_masks_cache is None:
            real = self.recon_placement.real_size
            masks = []
            for dev, (s0, s1), _n_valid in \
                    self.recon_placement.padded_shard_ranges():
                mask = ((s0 + np.arange(s1 - s0 + 1)) < real)
                masks.append(torch.as_tensor(mask.astype(np.float32),
                                             device=dev))
            self._qggmrf_interface_masks_cache = masks
        return self._qggmrf_interface_masks_cache

    # ── device configuration (the mbirjax configure_devices seam) ─────────────
    def configure_devices(self, num_devices=1, devices=None):
        """
        Set the compute devices the model uses.

        Specify either a CUDA device count (``num_devices=n``) or an explicit
        device list (``devices=['cpu']``, ``['mps']``, or
        ``['cuda:0', 'cuda:1']``).  With more than one device, the sinogram
        is divided across the devices by view and the reconstruction by
        slice.

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

        Args:
            num_devices (int, optional): number of CUDA devices to use.
                Defaults to 1.
            devices (list, optional): explicit device list.  Overrides
                num_devices.
        """
        self.device_layout_is_automatic = False
        # An earlier automatic settle may have left rejected counts behind.
        # They explain a search this layout did not come from, so the run log
        # must not carry them into a run the caller placed by hand.
        self.device_choice_rejections = []
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

    def _install_device_layout(self, devices):
        """Rebuild the placements over ``devices``: the shared body of
        :meth:`configure_devices` and the automatic choice.  Carries no
        policy; does not touch ``device_layout_is_automatic``."""
        devices = [torch.device(d) for d in devices]
        self.torch_device = devices[0]
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        self.sino_placement = _sharding.Placement(
            devices, axis=0, real_size=int(sinogram_shape[0]))
        self.recon_placement = _sharding.Placement(
            devices, axis=-1, real_size=int(recon_shape[2]))
        self._check_no_empty_shard()
        # One empirical probe per configuration (the L40S device_put lesson):
        # route transfers through host memory if a direct copy ever corrupts.
        self.dev2dev_safe = _sharding.is_dev2dev_safe(devices)
        self._invalidate_device_caches()
        if self._projector_functions is not None:
            self.create_projectors()

    # ── the memory ledger and the automatic device count ──────────────────────
    def _build_memory_ledger(self, devices=None, **call_arrays):
        """The modeled per-device peak for one candidate device list; None
        means the current placement.  Device-agnostic, so the rule can be
        tested on CPU."""
        devices = list(self.sino_placement.devices if devices is None
                       else devices)
        return _memory_ledger.estimate_peak_device_bytes(
            _memory_ledger.plan_from_model(self, devices, **call_arrays))

    def _candidate_devices(self, num_devices):
        return [torch.device(f'cuda:{i}') for i in range(num_devices)]

    def _apply_device_policy(self, **call_arrays):
        """Settle the device layout for the reconstruction about to run, and
        return the ledger for the layout settled on.

        This is the one site where the automatic device count is chosen.
        The choice happens at recon time, not construction, because the
        free-memory reading is only current here.  It is re-evaluated on
        every entry while the layout is automatic, and the placements are
        rebuilt only when the chosen count changes.  On the unpinned branch
        the candidate ORDER comes from the widening speed floors
        (:meth:`_speed_ordered_candidates`), and capacity still wins when
        nothing admitted fits.  Explicit layouts and process-wide pins skip
        the floors.  split_sino_recon's halves arrive here and choose for
        themselves at their own sinogram size."""
        calibrating = _memory_ledger.calibration_enabled()
        if not self.device_layout_is_automatic:
            # An explicit layout is the caller's; the ledger runs only when
            # the calibration mode asks for it.
            ledger = self._build_memory_ledger(**call_arrays) if calibrating \
                else None
            return self._arm_calibration(ledger)

        pinned = _memory_ledger.pinned_device_count()
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible < 2:
            # No layout to choose; the allocator's own error covers a
            # single-device overflow, and the n=1 path stays free of new cost.
            return self._arm_calibration(None)

        if pinned is not None:
            # A process-wide pin is as explicit as a configure_devices call.
            # The pinned count is not searched, not reduced, and not subject
            # to the speed floors.
            candidates, held = [min(pinned, visible)], {}
        else:
            candidates, held = self._speed_ordered_candidates(visible)

        rejected, best = [], None
        self._speed_floor_fallback = None
        # _settle names every wider held count in the run log once the
        # chosen count is known.
        self._speed_floor_held = held
        for count in candidates:
            if count in held:
                # Admitted counts come first, so reaching a held count means
                # capacity is about to override the speed rule.  If the loop
                # settles on this count, _settle rewrites the note.
                held_note, taken_note = held[count]
                rejected.append((count, held_note))
                self._speed_floor_fallback = (count, taken_note)
            devices = (self._candidate_devices(count) if count > 1
                       else [self.torch_device])
            if not self._layout_is_valid(devices):
                rejected.append((count, 'a device would own no real data'))
                continue
            ledger = self._build_memory_ledger(devices=devices, **call_arrays)
            if ledger is None or self.skip_memory_preflight:
                # Nothing to check against, or the caller has forced the run.
                return self._settle(devices, ledger, rejected)
            budgets = [_memory_ledger.device_budget_bytes(d) for d in devices]
            credits = _memory_ledger.resident_credits(
                devices, list(call_arrays.values()))
            fits, rows = _memory_ledger.layout_fits(
                ledger, budgets, credits, margin=self.memory_preflight_margin)
            if fits:
                return self._settle(devices, ledger, rejected)
            shortfall = max((d - b) for _dev, d, b in rows if b is not None)
            rejected.append((count, f'{shortfall / 2 ** 30:.2f} GB short'))
            if best is None or shortfall < best[0]:
                best = (shortfall, ledger, rows, count)

        # Nothing fits, including a single device.  The answer to "which
        # count" is "none", so fail here with the dominant phase named rather
        # than launch a reconstruction that is known not to fit.
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
        """The unpinned automatic branch's candidate order, and the notes for
        the counts the widening speed floors hold back.

        The floors REORDER, never remove: admitted counts largest-first,
        then held counts largest-first.  Capacity therefore always wins, and
        ``skip_memory_preflight`` (which settles on the first candidate)
        leaves the floors in force.

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
        # Debt and substitution are both said out loud rather than inferred
        # from a count that came out smaller than expected.
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
        # A count of 1 is always admitted, so `admitted` is never empty and a
        # held count always has something ahead of it.
        return admitted + list(held), held

    def _memory_remedies(self):
        """Extra remedy lines for this geometry's preflight message."""
        if type(self).split_sino_recon is not TomographyModel.split_sino_recon:
            return ['  model.split_sino_recon(...)                '
                    '# reconstructs in halves; nearly doubles the',
                    '                                             '
                    '# feasible size at a fixed device count']
        return []

    def _layout_is_valid(self, devices):
        """Whether ``devices`` passes the empty-shard rules, without mutating
        anything: the same predicate :meth:`_check_no_empty_shard` applies,
        evaluated on candidate placements."""
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        sino = _sharding.Placement(devices, axis=0,
                                   real_size=int(sinogram_shape[0]))
        recon = _sharding.Placement(devices, axis=-1,
                                    real_size=int(recon_shape[2]))
        return not any(sv <= 0 and rv <= 0 for sv, rv in zip(
            [n for _d, _r, n in sino.padded_shard_ranges()],
            [n for _d, _r, n in recon.padded_shard_ranges()]))

    def _settle(self, devices, ledger, rejected):
        """Install the chosen layout when it differs from the current one, log
        the choice, and enable the calibration mode."""
        chosen, current = len(devices), self.sino_placement.n_devices
        # A note recorded when the search reached a held count is replaced
        # if that count was settled on: capacity went past the floor.
        fallback = getattr(self, '_speed_floor_fallback', None)
        if fallback is not None and fallback[0] == chosen:
            rejected = [fallback if count == chosen else (count, why)
                        for count, why in rejected]
        # Every wider count the floors held back is named here, because the
        # loop usually never reaches them and idle GPUs need explaining.  A
        # held count smaller than the chosen one was outranked, not
        # excluded, so it carries no entry.
        held = getattr(self, '_speed_floor_held', None) or {}
        rejected = list(rejected)
        already = {count for count, _why in rejected}
        for count in sorted(held, reverse=True):
            if count > chosen and count not in already:
                rejected.append((count, held[count][0]))
        self._speed_floor_fallback = None
        self._speed_floor_held = None
        # Kept for the run log's device line, which explains any GPUs the
        # automatic choice left idle (see ParameterHandler._device_report).
        self.device_choice_rejections = list(rejected)
        if chosen != current:
            self.logger.info(
                f'Using {chosen} CUDA device(s) for this reconstruction '
                f'(was {current}).  configure_devices(num_devices=n) pins it.')
            self._install_device_layout(devices)
            # The ledger priced the layout we just installed, but the view
            # batch it charged came from the same candidate count, so it
            # stays valid.
        if rejected and self.get_params('verbose') >= 2:
            for count, why in rejected:
                self.logger.debug(f'  device count {count} rejected: {why}')
        return self._arm_calibration(ledger)

    def _arm_calibration(self, ledger):
        """Record the ledger and, under the calibration mode, take ownership
        of the peak counter it is about to compare against."""
        if ledger is not None:
            self.last_memory_ledger = ledger
        if _memory_ledger.calibration_enabled():
            if ledger is None:
                ledger = self._build_memory_ledger()
                self.last_memory_ledger = ledger
            _memory_ledger.calibration_start(self.sino_placement.devices)
        return ledger

    # ── array placement (entry) and gathering (exit) ──────────────────────────
    # Every sinogram-like placement routes through _shard_sinogram, every
    # recon-like one through _shard_recon; the exits route through the
    # matching gathers.  Multi-device support changes these four functions
    # alone.
    def _shard_sinogram(self, sinogram):
        """Place a sinogram-like array (sinogram or weights) in its device
        form: float32 on the model device, view axis checked, zero-padded
        and view-sharded on a multi-device placement."""
        num_views = self.get_params('sinogram_shape')[0]
        if isinstance(sinogram, _sharding.Shards):
            if sinogram.placement is not self.sino_placement:
                raise ValueError('Sinogram shards belong to a different '
                                 'device configuration; re-place the array.')
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
                                     what='sinogram (view axis)',
                                     row_pad=self._sino_row_padding())

    # Whether detector row r ties to recon slice r 1:1 (parallel beam:
    # True).  False is the base so a geometry that forgets to declare
    # itself is never mis-assembled by the row-aligned fast path.
    rows_track_slices = False

    # Whether the multi-device forward MAY gather pixel columns (see
    # _column_gather_forward).  Measured on cone and parallel only, so
    # False is the base; a geometry switches on its own measurement.
    column_gather_geometry = False

    # The fewest pixels this geometry's COMPILED bodies may be called with;
    # narrower calls are padded up outside the compiled region (see
    # projectors.forward_at_min_pixel_width).
    min_compiled_pixel_width = 1

    # Which measured widening-floor set governs the automatic device count
    # (see _widening_floors).  None means the parallel floors, the more
    # permissive measured set.
    _floor_family = None

    def _sino_row_padding(self):
        """Detector-row padding spec for the sinogram device form, or None.

        A geometry whose kernels tie rows to slices 1:1 must present the
        same padded length on both axes, so the row axis pads with the
        sharded slice axis (zero-filled, inert).

        Returns:
            (row_axis, real_rows, padded_rows) when active, else None.
        """
        if not (self.rows_track_slices and self.recon_placement.is_padded):
            return None
        real_rows = int(self.get_params('sinogram_shape')[1])
        return 1, real_rows, self.recon_placement.padded_size

    def prepare_sino_for_devices(self, sinogram, weights=None):
        """Place a sinogram (and optionally weights) in the model's device
        form, once.

        The device form is the view-sharded layout the reconstruction methods
        use internally: the sinogram is distributed across the configured
        devices, and when the view count (or, for parallel beam, the
        row-tracking slice count) does not divide the device count it is
        zero-padded to the next multiple.  The padding is exactly inert -- it
        cannot affect the results.  Each device receives only its own block,
        with zero tails built on the device, so no padded host copy is ever
        created.

        Calling this is OPTIONAL: every reconstruction method applies the same
        placement automatically to a plain input.  Use it to pay the
        host-to-device transfer once when running several reconstructions on
        the same large sinogram -- a prepared array passes through the entry
        placement untouched.  If the device configuration changes afterwards,
        the prepared array no longer matches and the entry placement raises
        with instructions to re-run this method.

        Args:
            sinogram (numpy or tensor): sinogram in the model's sinogram_shape.
            weights (numpy or tensor, optional): weights of the same shape;
                the zero-filled padding makes padded entries weightless as
                well.

        Returns:
            The prepared sinogram, or a (sinogram, weights) tuple when weights
            were given.
        """
        sino = self._shard_sinogram(sinogram)
        if weights is None:
            return sino
        return sino, self._shard_sinogram(weights)

    def _shard_recon(self, recon):
        """Place a recon-like array (3-D, or flat (num_pixels, num_slices))
        in its device form: float32 on the model device, slice axis (the
        LAST axis) checked, slice-sharded on a multi-device placement."""
        num_slices = self.get_params('recon_shape')[2]
        if isinstance(recon, _sharding.Shards):
            if recon.placement is not self.recon_placement:
                raise ValueError('Recon shards belong to a different device '
                                 'configuration; re-place the array.')
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

    def _split_to_shards(self, x, placement, real_size, what='array',
                         row_pad=None):
        """Split an array into per-device shard tensors (the n>1 body of
        _shard_sinogram / _shard_recon): each device gets its contiguous
        block of the sharded axis; a non-dividing axis zero-pads on the LAST
        shard.  Accepts the real length (pads) or the already-padded
        device-form length; a mixed shape is refused as stale.  Zero tails
        are built ON the receiving device, never as a padded host copy.
        ``row_pad`` = (row_axis, real_rows, padded_rows) serves the
        rows-track-slices sinogram only; None everywhere else."""
        x = torch.as_tensor(x, dtype=torch.float32)
        axis = placement.axis % x.ndim
        padded_size = (placement.padded_size if placement.padded_size
                       is not None else real_size)
        if row_pad is not None:
            row_axis, real_rows, padded_rows = row_pad
            row_axis %= x.ndim
            rows_are_padded = x.shape[row_axis] == padded_rows
            rows_are_real = x.shape[row_axis] == real_rows
        else:
            rows_are_padded = rows_are_real = True
        if x.shape[axis] == padded_size and rows_are_padded:
            # Already the device form (e.g. a prepare_sino_for_devices
            # output): split the equal blocks, nothing further to pad.
            tensors = []
            for dev, (start, end) in placement.shard_ranges(padded_size):
                idx = [slice(None)] * x.ndim
                idx[axis] = slice(start, end)
                tensors.append(x[tuple(idx)].to(dev))
        else:
            if x.shape[axis] != real_size or not rows_are_real:
                raise ValueError(
                    f'Cannot place the {what}: got shape {tuple(x.shape)}, '
                    f'but the model expects size {real_size} on axis {axis} '
                    f'(or the prepared device-form size {padded_size}).  If '
                    'the device configuration changed since '
                    'prepare_sino_for_devices, re-run it.')
            tensors = []
            for dev, (start, end), n_valid in placement.padded_shard_ranges():
                parts = []
                if n_valid > 0:
                    idx = [slice(None)] * x.ndim
                    idx[axis] = slice(start, start + n_valid)
                    piece = x[tuple(idx)].to(dev)
                    if row_pad is not None and padded_rows > real_rows:
                        tail_shape = list(piece.shape)
                        tail_shape[row_axis] = padded_rows - real_rows
                        piece = torch.cat(
                            [piece, torch.zeros(tail_shape, dtype=piece.dtype,
                                                device=dev)], dim=row_axis)
                    parts.append(piece)
                if end - start > n_valid:
                    pad_shape = list(x.shape)
                    pad_shape[axis] = (end - start) - n_valid
                    if row_pad is not None:
                        pad_shape[row_axis] = padded_rows
                    parts.append(torch.zeros(pad_shape, dtype=torch.float32,
                                             device=dev))
                tensors.append(parts[0] if len(parts) == 1
                               else torch.cat(parts, dim=axis))
        return _sharding.Shards(tensors, placement)

    def _gather_sinogram(self, sinogram):
        """Return a sinogram-like array as a host numpy array.  Shards are
        concatenated on the view axis and any zero-filled padded views (and,
        for parallel beam, padded detector rows) are CROPPED back to the real
        counts, so padded entries never leak into user-facing arrays (the
        mbirjax gather contract)."""
        if isinstance(sinogram, _sharding.Shards):
            out = self._gather_shards(sinogram)
            row_pad = self._sino_row_padding()
            if row_pad is not None:
                row_axis, real_rows, padded_rows = row_pad
                row_axis %= out.ndim
                if out.shape[row_axis] == padded_rows:
                    idx = [slice(None)] * out.ndim
                    idx[row_axis] = slice(0, real_rows)
                    out = out[tuple(idx)]
        else:
            out = sinogram.detach().cpu().numpy()
        return out

    def _gather_recon(self, recon):
        """Return a recon-like array as a host numpy array; the padded-slice
        crop of :meth:`_gather_sinogram` applies on the slice axis."""
        if isinstance(recon, _sharding.Shards):
            out = self._gather_shards(recon)
        else:
            out = recon.detach().cpu().numpy()
        return out

    def _constant_recon(self, value):
        """A constant-valued recon in the device form, for either
        state layout (built per shard so no full volume lands on one
        device)."""
        recon_shape = self.get_params('recon_shape')
        if self.recon_placement.is_trivial:
            recon = torch.full(tuple(recon_shape), float(value),
                               dtype=torch.float32, device=self.torch_device)
        else:
            tensors = []
            for d, (s, e), n_valid in \
                    self.recon_placement.padded_shard_ranges():
                t = torch.zeros(tuple(recon_shape[:2]) + (e - s,),
                                dtype=torch.float32, device=d)
                t[..., :n_valid] = float(value)   # padded slices stay zero
                tensors.append(t)
            recon = _sharding.Shards(tensors, self.recon_placement)
        return recon

    def _initial_error_state(self, sinogram, init_recon, weights,
                             constant_weights, scale_recon_to_sinogram):
        """The initial (error_sinogram, init_recon) pair: forward-project the
        init, find the optimal scale alpha (applied only to the default
        direct-recon init), and scale both -- for either state layout."""
        self.logger.info('Initializing error sinogram')
        fwd = self.forward_project(init_recon, output_sharded=True)
        if isinstance(fwd, _sharding.Shards):
            def dots_worker(i, d):
                f = fwd.tensors[i]
                w = 1 if constant_weights else weights.tensors[i]
                wf = f if constant_weights else w * f
                return (float(torch.sum(wf * f)),
                        float(torch.sum(wf * sinogram.tensors[i])))
            dots = _sharding.run_per_device(self.sino_placement.devices,
                                            dots_worker)
            wtd_err_sino_norm = sum(a for a, _ in dots)
            if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
                alpha = sum(b for _, b in dots) / wtd_err_sino_norm
            else:
                alpha = 1
            error_sinogram = _sharding.Shards(
                _sharding.run_per_device(
                    self.sino_placement.devices,
                    lambda i, d: sinogram.tensors[i] - alpha * fwd.tensors[i]),
                self.sino_placement)
            # The init projection is folded into the error; free its
            # sino-sized shards before the Hessian and the loop.
            fwd = None
            init_recon = _sharding.Shards(
                [alpha * t for t in init_recon.tensors], self.recon_placement)
        else:
            weighted_fwd = fwd if constant_weights else weights * fwd
            wtd_err_sino_norm = torch.sum(weighted_fwd * fwd)
            if wtd_err_sino_norm > 0 and scale_recon_to_sinogram:
                alpha = (torch.sum(weighted_fwd * sinogram)
                         / wtd_err_sino_norm).item()
            else:
                alpha = 1
            # Drop the weights product before the two sinogram-sized
            # allocations below: holding it made this function the measured
            # peak of a weighted reconstruction.
            weighted_fwd = None
            error_sinogram = sinogram - alpha * fwd
            fwd = None
            init_recon = alpha * init_recon
        return error_sinogram, init_recon

    def _flatten_recon(self, recon):
        """The VCD loop's flat (num_pixels, slices) recon layout, placed via
        _shard_recon and made contiguous for the in-place row updates --
        for either state layout."""
        if isinstance(recon, _sharding.Shards):
            flat = _sharding.Shards(
                [t.reshape((-1, t.shape[-1])).contiguous()
                 for t in recon.tensors], recon.placement)
        else:
            flat = self._shard_recon(
                recon.reshape((-1, recon.shape[-1]))).contiguous()
        return flat

    def _flatten_hessian(self, fm_hessian):
        """The Hessian diagonal in the VCD loop's flat layout, for either
        state layout (read-only in the loop, so no contiguity forcing)."""
        if isinstance(fm_hessian, _sharding.Shards):
            flat = _sharding.Shards(
                [t.reshape((-1, t.shape[-1])) for t in fm_hessian.tensors],
                fm_hessian.placement)
        else:
            flat = fm_hessian.reshape((-1, fm_hessian.shape[-1]))
        return flat

    def _recon_from_flat(self, flat_recon, recon_shape):
        """The 3-D recon from the VCD loop's flat layout, keeping each
        array's OWN slice count, for either state layout."""
        if isinstance(flat_recon, _sharding.Shards):
            recon = _sharding.Shards(
                [t.reshape(tuple(recon_shape[:2]) + (t.shape[-1],))
                 for t in flat_recon.tensors], flat_recon.placement)
        else:
            recon = flat_recon.reshape(tuple(recon_shape[:2])
                                       + (flat_recon.shape[-1],))
        return recon

    def _gather_shards(self, shards):
        out = shards.gather()
        p = shards.placement
        if p.is_padded:
            axis = p.axis % out.ndim
            idx = [slice(None)] * out.ndim
            idx[axis] = slice(0, p.real_size)
            out = out[tuple(idx)]
        return out

    def _as_shards(self, x, placement):
        """The uniform per-device container view of a device-form array: a
        plain tensor wraps as a one-shard container ALIASING it (in-place
        updates reach the caller's array); Shards pass through.
        Representation only -- the placement functions own validation."""
        if isinstance(x, _sharding.Shards):
            shards = x
        else:
            shards = _sharding.Shards([x], placement)
        return shards

    def _as_device_form(self, x):
        """The inverse of :meth:`_as_shards`: a trivial one-shard container
        unwraps to its (aliased) tensor; a genuinely per-device state stays
        Shards (collapsing it would take a gather)."""
        if isinstance(x, _sharding.Shards) and x.placement.is_trivial:
            out = x.tensors[0]
        else:
            out = x
        return out

    def _sino_ones_device_form(self, sino_like=None):
        """All-ones sinogram in the device form, with padded entries ZERO --
        a bare torch.ones would silently add padded mass to the
        constant-weights Hessian.  ``sino_like`` supplies only the dtype."""
        dtype = torch.float32 if sino_like is None else sino_like.dtype
        if self.sino_placement.is_trivial:
            return torch.ones(tuple(self.get_params('sinogram_shape')),
                              dtype=dtype, device=self.torch_device)
        shape = list(self.get_params('sinogram_shape'))
        row_pad = self._sino_row_padding()
        tensors = []
        for dev, (start, end), n_valid in \
                self.sino_placement.padded_shard_ranges():
            local = list(shape)
            local[0] = end - start
            if row_pad is not None:
                row_axis, real_rows, padded_rows = row_pad
                local[row_axis % len(local)] = padded_rows
            t = torch.ones(local, dtype=dtype, device=dev)
            if n_valid < end - start:
                t[n_valid:] = 0
            if row_pad is not None and padded_rows > real_rows:
                idx = [slice(None)] * len(local)
                idx[row_axis % len(local)] = slice(real_rows, None)
                t[tuple(idx)] = 0
            tensors.append(t)
        return _sharding.Shards(tensors, self.sino_placement)

    def forward_project(self, recon, output_sharded=False):
        """
        Perform a full forward projection at all voxels in the region of
        reconstruction (the pixels selected by the ROR mask; see
        ``vcd_utils.get_2d_ror_mask``).

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
        recon = self._shard_recon(recon)
        indices = self.full_indices_device()
        if isinstance(recon, _sharding.Shards):
            flat = _sharding.Shards(
                [t.reshape(-1, t.shape[-1])[indices.to(t.device)]
                 for t in recon.tensors], recon.placement)
            sinogram = self.sparse_forward_project(flat, indices)
        else:
            voxel_values = recon.reshape(-1, recon.shape[-1])[indices]
            sinogram = self.sparse_forward_project(voxel_values, indices)
        return sinogram if output_sharded else self._gather_sinogram(sinogram)

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
        """
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape', 'recon_shape'])
        if weights is None:
            # Unit weights built through the device-form seam (ones in the
            # real views/rows, zero in any inert padding), for either layout.
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

        # The dense back projection IS the flat volume, so it reshapes.  A
        # masked one returns only its own rows, so it scatters into a
        # zero-filled volume first.  An explicit index set always scatters,
        # even at full length: assuming a given set is the identity
        # permutation would silently mis-place a reordered one.
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
        return recon.reshape((-1, recon.shape[-1]))[indices]

    # ── auto-regularization (verbatim-math numpy ports) ───────────────────────
    def auto_set_regularization_params(self, sinogram, weights=None):
        """
        Automatically set the regularization parameters (sigma_y, sigma_x,
        and sigma_prox) from the sinogram and optional weights, and return
        them as a dict.  The parameters change only when
        ``auto_regularize_flag`` is True.  The statistics run on the host,
        on a view subsample.
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

            # Likewise crop padded detector ROWS (a device-form input whose row
            # axis pads with the recon slices) -- the zero rows would bias the
            # indicator/sigma stats.  A no-op until a sharding port pads arrays;
            # kept now so padded inputs can never silently bias the estimates.
            num_real_rows = self.get_params('sinogram_shape')[1]
            if small_sinogram.shape[1] != num_real_rows:
                small_sinogram = small_sinogram[:, :num_real_rows]
                if weights is not None:
                    small_weights = small_weights[:, :num_real_rows]

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
        """Set sigma_y from the (typically view-subsampled) sinogram, its
        support indicator, and optional weights."""
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
        """Set sigma_x (the qGGMRF prior scale) from the estimated recon
        standard deviation."""
        sharpness = self.get_params('sharpness')
        # Compute sigma_x as a fraction of the typical recon value.
        # 0.2 is an empirically determined constant.
        sigma_x = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_x=float(sigma_x), auto_regularize_flag=True)

    def auto_set_sigma_prox(self, recon_std):
        """Set sigma_prox (the proximal map prior scale) from the estimated
        recon standard deviation."""
        sharpness = self.get_params('sharpness')
        # Compute sigma_prox as a fraction of the typical recon value.
        # 0.2 is an empirically determined constant.
        sigma_prox = np.float32(0.2 * (2 ** sharpness) * recon_std)
        self.set_params(no_warning=True, sigma_prox=float(sigma_prox),
                        auto_regularize_flag=True)

    @staticmethod
    def subsample_views(array, max_views_to_use=20, num_real_views=None):
        """Return an evenly-spaced subsample of about ``max_views_to_use``
        views (axis 0) as a host numpy array.  The statistical sinogram
        estimates run on such a subsample.  The stride depends only on the
        view count, so a second call with the same arguments subsamples a
        companion array (e.g. weights) the same way.  ``num_real_views``
        restricts the sampling to ``array[:num_real_views]``."""
        num_views = array.shape[0] if num_real_views is None else num_real_views
        max_views_to_use = min(max_views_to_use, num_views)
        step_size = max(num_views // max_views_to_use, 1)
        return np.array(array[:num_views][::step_size])

    @staticmethod
    def _get_sino_indicator(sinogram, verbose=1):
        """Compute an int8 mask marking the region of sinogram support, the
        same shape as the input.  This runs several host-side reductions, so
        it is typically called on a view subsample."""
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
        """Estimate the standard deviation of the reconstruction from the
        (typically view-subsampled) sinogram and its support indicator.  The
        estimate scales sigma_x and sigma_prox."""
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
                                   output_sharded=False, row_weight=None):
        """Shared FBP row-filter for direct reconstruction.

        The filter is scaled by ``filter_scale * pi / num_views``, folded into
        the (tiny) filter array rather than applied as a full-sinogram
        multiply (which would promote f32 -> f64 and about double peak
        memory).  The pi / num_views factor assumes equally spaced views over
        the full angular range; for nonuniform, limited-angle, or short scans
        a standalone direct recon is only approximate -- prefer ``recon()``.

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
            # The row filter is per detector row, so each view-shard filters
            # locally (one thread per device; no cross-device data).
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

    # ── loss / stats (mirrors get_forward_model_loss + _vcd_iteration_stats) ──
    @staticmethod
    def get_forward_model_loss(error_sinogram, sigma_y, weights=None, normalize=True,
                               num_real_elements=None):
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
            num_real_elements (int, optional): the REAL element count when
                error_sinogram carries zero-filled padding.  Default None
                uses error_sinogram.numel().

        Returns:
            The loss as a device scalar tensor.
        """
        if weights is None:
            weights = 1
            avg_weight = 1
        elif np.ndim(weights) == 0:
            # A true scalar (python or 0-d): the average weight is itself,
            # independent of the element count -- so also exact on padded runs.
            avg_weight = weights
        elif num_real_elements is None:
            # Array-likes (numpy included -- a numpy array is not a torch
            # tensor, and a tensor-only test would route it to the scalar
            # branch, returning a sinogram-shaped 'loss').
            weights = torch.as_tensor(weights, dtype=torch.float32,
                                      device=error_sinogram.device)
            avg_weight = torch.mean(weights)
        else:
            # Weights ARRAY in a padded device form: the padded entries are
            # identically zero, so summing and dividing by the REAL count gives
            # exactly the average over the real elements.
            weights = torch.as_tensor(weights, dtype=torch.float32,
                                      device=error_sinogram.device)
            avg_weight = torch.sum(weights) / float(num_real_elements)
        if normalize:
            weighted_sq_sum = torch.sum(error_sinogram * error_sinogram * weights)
            denom = (float(error_sinogram.numel()) if num_real_elements is None
                     else float(num_real_elements))
            loss = torch.sqrt(weighted_sq_sum / (avg_weight * denom)) / sigma_y
        else:
            loss = (1.0 / (2 * sigma_y ** 2)) * torch.sum(
                (error_sinogram * error_sinogram) * weights)
        return loss

    @staticmethod
    def _vcd_iteration_stats(error_sinogram, flat_recon, sigma_y, weights=None,
                             num_real_elements=None, real_sino_size=None):
        """Per-iteration VCD logging stats: (fm_loss, recon_l1, es_rmse).

        ``num_real_elements``/``real_sino_size`` are the REAL element count when
        the error sinogram carries zero-filled padding (see
        get_forward_model_loss); padded entries must not dilute the RMSE.
        Both default to None (unpadded), which uses the array's own size."""
        fm_loss = TomographyModel.get_forward_model_loss(
            error_sinogram, sigma_y, weights, num_real_elements=num_real_elements)
        recon_l1 = torch.sum(torch.abs(flat_recon))
        denom = (float(error_sinogram.numel()) if real_sino_size is None
                 else float(real_sino_size))
        es_rmse = torch.sqrt(torch.sum(error_sinogram * error_sinogram) / denom)
        return fm_loss, recon_l1, es_rmse

    def get_forward_lin_quad(self, weighted_error_sinogram, delta_sinogram, weights,
                             fm_constant, const_weights):
        """
        Compute the two forward model line-search terms:
        ``fm_constant * sum(weighted_error_sinogram * delta_sinogram)`` and
        ``fm_constant * sum(delta_sinogram^2 * weights)``, returned as device
        scalars.
        """
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
        Overrides must return -M (forward_grad + prior_grad) with M positive
        definite, which preserves the cost's minimizers.  Arguments are one
        shard's (num_subset_pixels, local_slices) arrays; ``dev_index``
        selects a slice-profile override's shard (cone DC damping) and is
        ignored here."""
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

        # Bind the compiled forms once for all subsets, one instance per
        # device thread: compiled artifacts carry launcher state that must
        # not be shared across threads (see maybe_compile).
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
            """Sum per-shard 0-d tensor partials on the lead device: the
            identity on one device, scalar-sized device moves otherwise."""
            total = parts[0]
            for part in parts[1:]:
                total = total + _sharding.move_shard(part, dev0,
                                                     self.dev2dev_safe)
            return total

        # The qGGMRF boundary halos are staged once per PARTITION pass.  A
        # None halo means the reflected boundary at a true volume edge,
        # which is every entry on a single device.
        halos = {'left': [None] * num_devices, 'right': [None] * num_devices}
        # Interface masks keep a padded slice tail out of the prior.
        interface_masks = self._qggmrf_interface_masks()

        def stage_halos(flat_shards):
            halos['left'], halos['right'] = _sharding.exchange_qggmrf_halos(
                flat_shards, self.dev2dev_safe)

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices):
            """One VCD iteration on a single subset of the partition, under
            the invariant error_sinogram = measured_sinogram -
            forward_proj(recon).  flat_recon and error_sinogram are
            per-device shards, updated IN PLACE.

            Returns:
                flat_recon, error_sinogram, ell1_for_subset,
                alpha_for_subset, delta_sumsq_subset.
            """
            pixel_indices_per_device = [torch.as_tensor(pixel_indices, dtype=torch.int64).to(dev)
                       for dev in devices]

            # Compute the prior model gradient and Hessian (i.e., second
            # derivative) terms at each pixel in the index set, per slice-shard
            # (halos carry the cross-boundary term; a true edge is reflected).
            def prior_worker(i, dev):
                if prox_input is None:
                    # qGGMRF prior.
                    grad, hess = qggmrf_grad_hess[i](
                        flat_recon.tensors[i], recon_shape, pixel_indices_per_device[i],
                        qggmrf_params, left_halo=halos['left'][i],
                        right_halo=halos['right'][i],
                        interface_mask=(None if interface_masks is None
                                        else interface_masks[i]))
                else:
                    # Proximal map prior: pointwise, so the Hessian is a scalar.
                    grad = _qggmrf.prox_gradient_at_indices(
                        flat_recon.tensors[i], prox_input.tensors[i],
                        pixel_indices_per_device[i], sigma_prox)
                    hess = 1 / (sigma_prox ** 2)
                return grad, hess

            prior_terms = _sharding.run_per_device(devices, prior_worker,
                                             executor=self._per_device_pool)

            # Compute the forward model gradient and Hessian at each pixel in
            # the index set.  Assumes Loss(delta) =
            # 1/(2 sigma_y^2) || error_sinogram - A delta ||_weights^2.
            if const_weights:
                weighted_error_sinogram = error_sinogram
            else:
                weighted_error_sinogram = _sharding.Shards(
                    _sharding.run_per_device(
                        devices, lambda i, dev: weights.tensors[i]
                        * error_sinogram.tensors[i],
                        executor=self._per_device_pool), sino_placement)

            # Back project to get the gradient; note fm_constant = 1/sigma_y^2.
            back_projected_error = self.sparse_back_project(weighted_error_sinogram, pixel_indices)
            if not const_weights:
                # The weighted product is dead here, because the line-search
                # terms fuse the weights into their reductions.  Freeing it
                # now drops a full sinogram before the delta projection.
                weighted_error_sinogram = None

            # Each shard computes its update direction and its prior
            # line-search partials in one worker.
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

            # This frees the dead gradient and Hessian buffers before the
            # memory-heavy delta projection.  The stream-aware allocator
            # needs no synchronization.
            del prior_terms, back_projected_error, direction_results

            # Compute the update direction in the sinogram domain.
            delta_sinogram = self.sparse_forward_project(
                _sharding.Shards(delta_recon_per_device, recon_placement), pixel_indices)

            # Forward line-search reductions per view-shard.
            def lin_quad_worker(i, dev):
                local_delta_sinogram = delta_sinogram.tensors[i]
                if const_weights:
                    return lin_quad_const[i](error_sinogram.tensors[i], local_delta_sinogram,
                                             fm_constant)
                # Fusing the weights product into the reductions avoids the
                # per-subset sinogram-sized weighted transient.
                return lin_quad_weighted[i](error_sinogram.tensors[i], local_delta_sinogram,
                                            weights.tensors[i], fm_constant)
            forward_line_terms = _sharding.run_per_device(devices, lin_quad_worker,
                                          executor=self._per_device_pool)
            forward_linear = combine_on_lead(
                [linear for linear, _ in forward_line_terms])
            forward_quadratic = combine_on_lead(
                [quadratic for _, quadratic in forward_line_terms])

            # Compute the optimal update step.  The line search stays ON DEVICE
            # (alpha is a scalar tensor; no host synchronization per subset).
            alpha_numerator = forward_linear - prior_linear
            alpha_denominator = forward_quadratic + prior_quadratic_approx + _F32_EPS
            alpha = alpha_numerator / alpha_denominator
            alpha = torch.clamp(alpha, _F32_EPS, max_alpha)
            alpha_per_device = ([alpha] if num_devices == 1 else
                         [_sharding.move_shard(alpha, dev, self.dev2dev_safe)
                          for dev in devices])

            # Enforce the positivity constraint if desired: clip updates so that
            # recon + alpha * delta >= 0, then recompute the sinogram projection.
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

            # The sparse updates are applied IN PLACE, each shard locally.
            # The per-slice sum of squared updates is the convergence
            # diagnostic.
            def apply_worker(i, dev):
                delta_scaled = alpha_per_device[i] * delta_recon_per_device[i]
                _, _, delta_sumsq_local, ell1_local = apply_update[i](
                    flat_recon.tensors[i], error_sinogram.tensors[i],
                    pixel_indices_per_device[i], delta_scaled, alpha_per_device[i],
                    delta_sinogram.tensors[i])
                return delta_sumsq_local, ell1_local
            apply_results = _sharding.run_per_device(devices, apply_worker,
                                               executor=self._per_device_pool)
            # Per-slice sums concatenate in global slice order on the lead
            # device (the identity on one device; (local_slices,)-sized moves
            # otherwise); the L1 partials combine like the line-search terms.
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
                               partition):
        """
        Calculate a full iteration of the VCD algorithm by scanning over the
        subsets of the partition, updating flat_recon and error_sinogram in
        place.

        Returns:
            (flat_recon, error_sinogram, ell1_for_partition, alpha,
            delta_sumsq_partition): the updated state, the summed L1 recon
            change, alpha averaged over the subsets, and the per-slice sum
            of squared update values over the partition.
        """
        # The qGGMRF boundary halos are staged once for this whole partition
        # pass.  A single device has no shard boundaries, so this costs it
        # nothing.
        if hasattr(vcd_subset_updater, 'stage_halos'):
            vcd_subset_updater.stage_halos(flat_recon)
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
                  prox_input=None, compute_prior_loss=False, first_iteration=0,
                  init_error_sinogram=None, fm_hessian=None,
                  return_checkpoint=False):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate
        Descent algorithm for a given set of partitions and a prescribed
        partition sequence.

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            partitions (list): K partitions, each an (N_subsets, N_indices)
                integer index tensor of voxels to update.
            partition_sequence (ndarray): which partition to use at each
                iteration.
            stop_threshold_change_pct (float): stop when the NMAE percent
                change between iterations falls below this value.
            weights (numpy or tensor, optional): 3D positive weights with the
                same shape as the sinogram.  Defaults to all 1s.
            init_recon (array or int or None): initial reconstruction.  None
                uses direct_recon; an int gives a constant volume.
            prox_input (array, optional): input to a proximal map.
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
        if tuple(sinogram.shape) != tuple(sinogram_shape):
            raise ValueError('sinogram does not have the shape in sinogram_shape. \n'
                             f'Expected {tuple(sinogram_shape)}, got '
                             f'{tuple(sinogram.shape)}.')

        # Settle the device layout BEFORE the first large allocation; the
        # returned ledger is what the calibration mode compares against.
        memory_ledger = self._apply_device_policy(
            partition_sequence=partition_sequence, weights=weights,
            init_recon=init_recon, fm_hessian=fm_hessian,
            prox_input=prox_input, init_error_sinogram=init_error_sinogram)
        # The layout is final here, so this is where the log can name the
        # devices the run will actually use.
        self._log_device_report()

        # Placement: recon-like arrays route through _shard_recon and
        # sino-like arrays through _shard_sinogram (a single device is the trivial
        # 1-shard case), keeping the rest of the loop placement-agnostic.
        constant_weights = weights is None
        if constant_weights:
            weights = 1
        else:
            weights = self._shard_sinogram(weights)

        if init_error_sinogram is not None and init_recon is None:
            raise ValueError('init_error_sinogram requires init_recon (the pair must be a '
                             'consistent resume state; see the docstring).')

        # On the resume path the error sinogram replaces the sinogram's only
        # use, so no device copy is made.
        if init_error_sinogram is None:
            sinogram = self._shard_sinogram(sinogram)

        scale_recon_to_sinogram = init_recon is None
        if init_recon is None:
            self.logger.info('Starting direct recon for initial reconstruction')
            init_recon = self.direct_recon(sinogram, output_sharded=True)
        elif isinstance(init_recon, int):
            init_recon = self._constant_recon(init_recon)
        else:
            if tuple(np.shape(init_recon)) != tuple(recon_shape):
                raise ValueError(f"init_recon does not have the correct shape. Expected "
                                 f"{tuple(recon_shape)}, got {tuple(np.shape(init_recon))}.")
            init_recon = self._shard_recon(init_recon)

        if init_error_sinogram is not None:
            # Resume fast path: trust the pair and skip the initializing
            # forward projection.  No defensive copies -- the caller's arrays
            # become the loop's working buffers (see the docstring).
            self.logger.info('Resuming from init_error_sinogram')
            error_sinogram = self._shard_sinogram(init_error_sinogram)
        else:
            error_sinogram, init_recon = self._initial_error_state(
                sinogram, init_recon, weights, constant_weights,
                scale_recon_to_sinogram)

        # The sinogram is fully folded into error_sinogram.  Dropping the
        # reference frees any device copy made here before the loop.
        sinogram = None
        # Placement invariant at the loop boundary (mirrors mbirjax): the
        # error sinogram is in the sino device form -- a no-op re-placement
        # on a single device.
        error_sinogram = self._shard_sinogram(error_sinogram)

        if prox_input is not None:
            # mbirjax validates the prox input's shape before flattening; a
            # size-compatible but mis-shaped input (e.g. a transposed volume)
            # must fail loudly rather than silently reshape.
            if tuple(prox_input.shape) != tuple(recon_shape):
                raise ValueError('prox_input does not have the correct size. \n'
                                 f'Expected {tuple(recon_shape)}, got shape '
                                 f'{tuple(prox_input.shape)} for prox_input shape.')
            # Flatten first, then place: the flat form is the slice-sharded
            # device form.
            prox_input = self._shard_recon(
                prox_input.reshape((-1, prox_input.shape[-1])))

        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # math.prod uses exact Python integers; np.prod would silently wrap
        # past 2^31 elements.
        real_sino_size = math.prod(sinogram_shape)
        loss_num_real = None

        # Initialize the diagonal of the Hessian of the forward model: the back
        # projection of the weights with squared coefficients (constant weights
        # use an all-ones sinogram).  A precomputed fm_hessian (the checkpoint
        # fast path) skips the back projection; it is read-only in the loop.
        if fm_hessian is None:
            if constant_weights:
                # The ones array is zero over any padding, so padding
                # contributes nothing to the Hessian back projection.
                hess_weights = self._sino_ones_device_form(error_sinogram)
            else:
                hess_weights = weights
            self.logger.info('Computing Hessian diagonal')
            # Back-project only at the ROR-masked pixels: the loop reads the
            # Hessian only at partition indices from the same mask, so the
            # values are unchanged while the transients shrink.
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

        # From here the loop runs one code path for any device count.  On a
        # single device the one-shard container aliases its tensor, so the
        # in-place updates still reach the caller-visible array.
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
            # One per-device thread pool serves the whole loop.  A single
            # device never creates it.
            self._per_device_pool = _sharding.device_pool(
                self.sino_placement.n_devices)
        try:
            for i in range(max_iters):
                partition = partitions[partition_sequence[i]]
                (flat_recon, error_sinogram, ell1_for_partition, alpha,
                 delta_sumsq_partition) = self.vcd_partition_iterator(
                    vcd_subset_updater, flat_recon, error_sinogram, partition)

                # real_sino_size == error_sinogram.numel() except under
                # padding, where the padded entries are identically zero and
                # must not dilute the RMSE.
                fm_loss_i, recon_l1, es_rmse = self._iteration_stats(
                    error_sinogram, flat_recon, sigma_y, weights,
                    constant_weights, num_real_elements=loss_num_real,
                    real_sino_size=float(real_sino_size))
                fm_rmse[i] = float(fm_loss_i)
                recon_l1_f = float(recon_l1)
                # A zero recon gives nan (matching mbirjax) rather than
                # raising ZeroDivisionError.
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
                        # Evaluate the prior loss on the REAL volume (padded
                        # slices would add spurious boundary terms).
                        real_recon_size = math.prod(recon_shape)
                        loss_recon = self._gather_recon(flat_recon).reshape(
                            tuple(recon_shape))
                        pm_loss[i] = _qggmrf.qggmrf_loss(loss_recon, qggmrf_params)
                        pm_loss[i] /= real_recon_size
                        # Each loss is scaled by its element count, but the
                        # optimization uses unscaled values.  Remove the
                        # scaling, add, then scale by the average element
                        # count of the two.
                        total_loss = ((fm_rmse[i] * real_sino_size
                                       + pm_loss[i] * real_recon_size)
                                      / (0.5 * (real_sino_size + real_recon_size)))
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

        # The calibration comparison, before the loop's state is released, so
        # the measured high-water mark still reflects the reconstruction.
        if memory_ledger is not None and _memory_ledger.calibration_enabled():
            rows = _memory_ledger.calibration_report(
                memory_ledger, self.sino_placement.devices)
            self.last_memory_calibration = rows
            if rows:
                self.logger.warning(_memory_ledger.format_calibration(rows))

        # Loop exit: back to the public/device forms -- plain tensors on a
        # single device (the same objects the loop mutated, so the checkpoint
        # aliasing contract is unchanged), per-device shards otherwise.
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
                         print_logs=True):
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
        # The run logger is set up exactly when a run is initialized.  A
        # Plug-and-Play loop passing do_initialization=False skips this
        # method, so the whole loop writes to one log.
        self._log_run_header(first_iteration, logfile_path, print_logs)
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
              stop_threshold_change_pct=0.2, first_iteration=0,
              logfile_path='~/.mbirtorch/logs/recon.log', print_logs=True,
              output_sharded=False):
        """
        Perform MBIR reconstruction using the Multi-Granular Vector Coordinate
        Descent algorithm.  This function takes care of generating its own
        partitions and partition sequence.

        To restart a recon using the same partition sequence, set
        first_iteration to the number of iterations completed so far and set
        init_recon to the output of the previous recon; this continues the
        partition sequence from where the previous recon left off.

        Device use: on CUDA this spreads the reconstruction across the
        available devices, using every device that can hold its share.  The
        share is judged by a memory check that runs before the first large
        allocation, so a layout that cannot fit is refused in seconds rather
        than part way through.  Nothing needs to change in a calling script.
        ``configure_devices(num_devices=n)`` fixes the count instead, and
        ``configure_devices(num_devices=1)`` is the reproducibility pin.  The
        environment variable ``MBIRTORCH_NUM_DEVICES`` pins it process-wide,
        which is what a test suite or a nightly should use.

        Reproducibility note: the pixel partitions are drawn from numpy's
        global random number generator, so reconstructions vary slightly from
        run to run.  For a reproducible result, call ``np.random.seed(seed)``
        before calling this method.  Results also differ slightly with the
        device count, and that difference decays as iterations proceed.

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
            logfile_path (str, optional): Path to the output log file ('~' expands to the
                user's home directory).  If None or empty, no log file is written.
                Defaults to '~/.mbirtorch/logs/recon.log'.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a numpy
                array; if True, return the device tensor (the mbirjax argument
                name, kept for API compatibility).

        Returns:
            (recon, recon_dict): the reconstruction volume, and a dict
            with entries 'recon_params' (per-iteration traces and settings),
            'recon_log' (the run's log text), 'notes', and
            'model_params' (a snapshot of the model parameters).
        """
        (sinogram, weights, init_recon, partitions, partition_sequence, granularity,
         regularization_params) = self.initialize_recon(
            sinogram, weights, init_recon, max_iterations, first_iteration,
            logfile_path=logfile_path, print_logs=print_logs)

        # no_grad, not inference_mode: torch.compile's guards crash on
        # compiled calls inside inference_mode with in-place updates.
        with torch.no_grad():
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

        if logfile_path:
            self.logger.info('Logs written to {}'.format(
                os.path.abspath(os.path.expanduser(logfile_path))))
        for h in list(self.logger.handlers):  # Make sure the log files are up to date
            h.flush()

        notes = 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        # output_sharded keeps the device tensor (the mbirjax parameter; here
        # it means "skip the numpy exit").
        return (recon if output_sharded else self._gather_recon(recon)), recon_dict

    def _iteration_stats(self, error_sinogram, flat_recon, sigma_y, weights,
                         constant_weights, num_real_elements=None,
                         real_sino_size=None):
        """Per-iteration logging stats (fm loss, recon L1, error-sino RMSE).
        A single-device state delegates to the fused _vcd_iteration_stats
        with bit-identical results.  A per-device state combines per-shard
        sums on the host, which is the loop's one host synchronization point
        per iteration."""
        if (isinstance(error_sinogram, _sharding.Shards)
                and not error_sinogram.placement.is_trivial):
            error_shards, flat_shards = error_sinogram, flat_recon
            weights_shards = None if constant_weights else weights

            def sino_worker(i, d):
                e = error_shards.tensors[i]
                if weights_shards is None:
                    return (float(torch.sum(e * e)), 0.0,
                            float(torch.sum(e * e)))
                w = weights_shards.tensors[i]
                return (float(torch.sum(e * e * w)), float(torch.sum(w)),
                        float(torch.sum(e * e)))
            parts = _sharding.run_per_device(error_shards.placement.devices,
                                             sino_worker)
            weighted_sq = sum(a for a, _, _ in parts)
            sq = sum(c for _, _, c in parts)
            if weights_shards is None:
                avg_weight = 1.0
            else:
                avg_weight = sum(b for _, b, _ in parts) / real_sino_size
            fm_loss = ((weighted_sq / (avg_weight * real_sino_size)) ** 0.5
                       / sigma_y)
            recon_l1 = sum(
                float(torch.sum(torch.abs(t))) for t in flat_shards.tensors)
            es_rmse = (sq / real_sino_size) ** 0.5
        else:
            if isinstance(error_sinogram, _sharding.Shards):
                # The trivial one-shard container unwraps (aliasing) to the
                # fused single-tensor kernel below, bit-identically.
                error_sinogram = error_sinogram.tensors[0]
                flat_recon = flat_recon.tensors[0]
            fm_loss, recon_l1, es_rmse = TomographyModel._vcd_iteration_stats(
                error_sinogram, flat_recon, sigma_y, weights,
                num_real_elements=num_real_elements,
                real_sino_size=real_sino_size)
        return fm_loss, recon_l1, es_rmse

    def prox_map(self, prox_input, sinogram, sigma_prox=None, weights=None,
                 init_recon=None, do_initialization=True,
                 stop_threshold_change_pct=0.2, max_iterations=3,
                 first_iteration=0,
                 logfile_path='~/.mbirtorch/logs/prox.log', print_logs=True,
                 output_sharded=False):
        """
        Proximal Map function for use in Plug-and-Play applications.  This
        function is similar to recon, but it essentially uses a prior with a
        mean of prox_input and a standard deviation of sigma_prox.

        Reproducibility note: the pixel partitions are drawn from numpy's global
        random number generator; call ``np.random.seed(seed)`` first for a
        reproducible result.

        Args:
            prox_input (numpy or tensor): proximal map input with the same shape
                as the reconstruction.
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            sigma_prox (None or float, optional): standard deviation of the
                proximal map prior term.  If None, set automatically from the
                sinogram.  Defaults to None.
            weights (numpy or tensor, optional): 3D positive weights with the
                same shape as the sinogram.  Defaults to None (all 1s).
            init_recon (numpy or tensor, optional): reconstruction used for
                initialization.  Defaults to None (determined by vcd_recon).
            do_initialization (bool, optional): If True, initialize parameters
                (partitions and regularization).  Set False if a previous
                prox_map call on this model already initialized this sinogram.
            stop_threshold_change_pct (float, optional): stop when the NMAE
                percent change drops below this value.  Defaults to 0.2.
            max_iterations (int, optional): maximum VCD iterations.  Defaults to 3.
            first_iteration (int, optional): partition-sequence offset for
                restarts.  Defaults to 0.
            logfile_path (str, optional): Path to the output log file ('~' expands to the
                user's home directory).  If None or empty, no log file is written.
                Defaults to '~/.mbirtorch/logs/prox.log'.  A Plug-and-Play loop
                that passes do_initialization=False after its first call keeps
                writing to the log that call opened, so the whole loop lands in
                one file.
            print_logs (bool, optional): If true then print logs to console.  Defaults to True.
            output_sharded (bool, optional): If False (default), return a numpy
                array; if True, return the device tensor (the mbirjax argument
                name, kept for API compatibility).

        Returns:
            (recon, recon_dict): the reconstruction volume, and a dict
            with entries 'recon_params' (per-iteration traces and settings),
            'recon_log' (the run's log text), 'notes', and
            'model_params' (a snapshot of the model parameters).
        """
        prior_loss = [0]
        if do_initialization or self.prox_data is None:
            (sinogram, weights, init_recon, partitions, partition_sequence,
             granularity, regularization_params) = self.initialize_recon(
                sinogram, weights, init_recon, max_iterations, first_iteration,
                logfile_path=logfile_path, print_logs=print_logs)
            self.prox_data = (partitions, partition_sequence, granularity,
                              regularization_params)
        else:
            (partitions, partition_sequence, granularity,
             regularization_params) = self.prox_data

        # Override the auto sigma_prox if requested, restoring it afterward.
        self_sigma_prox = self.get_params('sigma_prox')
        if sigma_prox is not None:
            regularization_params = dict(regularization_params,
                                         sigma_prox=sigma_prox)
            self.set_params(no_warning=True, sigma_prox=sigma_prox,
                            auto_regularize_flag=self.get_params('auto_regularize_flag'))

        with torch.no_grad():
            recon, loss_vectors = self.vcd_recon(
                sinogram, partitions, partition_sequence,
                stop_threshold_change_pct, weights=weights,
                init_recon=init_recon, prox_input=prox_input,
                first_iteration=first_iteration)

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
        for h in list(self.logger.handlers):  # Make sure the log files are up to date
            h.flush()

        notes = 'Proximal map completed: {}\n\n'.format(datetime.datetime.now())
        recon_dict = self.get_recon_dict(recon_params, notes=notes)
        # output_sharded keeps the device tensor (the mbirjax parameter; here
        # it means "skip the numpy exit").
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

    def reshape_recon(self, recon):
        recon_shape = self.get_params('recon_shape')
        return recon.reshape(recon_shape)

    # ── model description and HDF5 persistence ────────────────────────────────

    def get_all_params(self):
        """
        Return this model's parameters as ``(required_params, optional_params, regularization)``.

        This is the single source of truth for reading a model's parameters back out.  The three
        dicts partition the parameters so a caller can reconstruct or serialize the model and choose
        which parts to apply:

        * **required_params** -- the arguments the model constructor takes (from its ``__init__``
          signature), with the view-dependent arguments reconstructed from storage (e.g. cone's
          ``angles`` and ``helical_z_shifts`` are unpacked from the stored ``view_params_array``),
          plus a ``geometry_type`` entry so the model class can be resolved.
        * **optional_params** -- the remaining geometry/detector parameters that are applied with
          ``set_params`` (detector pitches, offsets, ``delta_voxel``, ``recon_shape``, voxel aspects).
        * **regularization** -- the recon-time regularization knobs (``sigma_y``, ``sigma_x``,
          ``sigma_prox``, ``snr_db``, ``sharpness``, ``auto_regularize_flag``), separated so a
          consumer such as ``save_cone_preprocessing`` can drop them and let them be re-chosen at
          reconstruction time.

        Returns:
            tuple: ``(required_params, optional_params, regularization)`` -- three dicts of values.
        """
        import inspect

        regularization_names = _AUTO_REGULARIZATION_PARAM_NAMES + (
            'snr_db', 'sharpness', 'auto_regularize_flag')
        # Bookkeeping params re-derived at construction.
        construction_derived_names = ('geometry_type', 'view_params_name', 'file_format',
                                      'version', 'use_gpu')
        # Execution-environment constructor arguments; not model parameters.
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
        Encapsulate the recon parameters, logs, notes, and optionally all model parameters to a text-based dict
        with entries 'recon_params', 'recon_log', 'notes', and optionally 'model_params'.  This dict can be used with
        :func:`mbirtorch.view_utils.slice_viewer` and :meth:`TomographyModel.save_recon_hdf5`.

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
            recon (array-like): The reconstruction volume as a NumPy array or torch tensor.
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

        # Log the save
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
                - recon_dict (dict): A dict with the attributes for the data array as in :meth:`get_recon_dict`

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
