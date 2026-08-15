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
# How many pixel columns one gathered cylinder covers (see
# TomographyModel._forward_pixel_batch and _sparse_forward_project_columns).
# The cylinder is this many columns by the whole slice axis, so this is the
# knob that bounds the cross-device transient on that path.  Measured
# 2026-08-10 on four H100s, job mg10: per-device forward time fell at every
# batch tried -- 2048, 4096, 8192 -- and was still falling at the largest.
# The sweep above it ran the next night (job mg11, same machines, 1K cells):
# 16384 and 32768 kept improving the composed wall by a further 4 to 15
# percent depending on geometry and device count, so the knee is still not
# bracketed.  8192 stays the default anyway, because those readings come from
# a 1K harness and production runs at 2K and above, where the batch's
# transient grows with the slice axis and the sweep has not been run.  Set
# forward_project_pixel_batch on the model to override.
FORWARD_PIXEL_BATCH = 8192

# Forces the column gather on ('1', 'true', 'yes', 'on') or off ('0',
# 'false', 'no', 'off') whatever the model attribute says.  Read per call,
# like the other environment knobs, so one session can run both shapes -- the
# comparison the value gate for this path is read from.
COLUMN_GATHER_ENV_VAR = 'MBIRTORCH_FORWARD_COLUMN_GATHER'
_COLUMN_GATHER_ON_VALUES = ('1', 'true', 'yes', 'on')
_COLUMN_GATHER_OFF_VALUES = ('0', 'false', 'no', 'off')


# ── compiled updater glue (module level, one compile per process) ─────────────
# Eagerly there were ~20 kernel launches per subset between the projector
# calls; these fused forms remove most of them.  Each is pure except
# _apply_update, which updates the two state tensors in place (supported by
# torch.compile) while RETURNING them for a functional interface -- the
# in-place ops are the memory mechanism, the returns are the contract the
# call sites rebind through.  All are value-equal to the eager forms up to
# float summation order.
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
        # The device and the placements resolve LAZILY, on first use.  Two
        # things follow.  A caller who only inspects a model, or who is about
        # to call configure_devices, never pays CUDA context initialization
        # for a device they did not choose.  And the projectors are built
        # once, against the layout actually in force, instead of being built
        # at construction and rebuilt by the first configure_devices call.
        self._torch_device = None
        self._sino_placement = None
        self._recon_placement = None
        self._projector_functions = None
        # Views per body call in the batched drivers.  None means automatic:
        # a torch body uses the long-standing default of 64, and a
        # hand-written kernel body uses its own swept view chunk.  An
        # explicit integer is the nominal for every body; the driver's
        # transient budget may cap the realized batch below it either way
        # (see Projectors._effective_view_batch).
        self.view_batch_size = view_batch_size
        # torch.compile of the hot chains.  'auto' compiles on every backend
        # (torch 2.13 supports CPU, CUDA, and MPS); 'off' keeps pure eager (debugging, or a backend where compile
        # misbehaves).  An execution-environment choice like `device`, so a
        # plain attribute rather than a saved model parameter.
        self.compile_mode = compile_mode
        # Cached prox initialization (partitions/sequence/regularization), so a
        # Plug-and-Play loop pays initialize_recon once (do_initialization=False
        # on subsequent prox_map calls) -- the mbirjax mechanism.
        self.prox_data = None
        # Device-layout caches (see _invalidate_device_caches).
        self._qggmrf_interface_masks_cache = None
        self._dc_damping_cache = None
        self.dev2dev_safe = True     # probed for real in configure_devices
        # Whether the device layout is still the library's to choose.  It is
        # one bit: an explicit configure_devices call sets it False
        # permanently, because a layout the caller chose is the caller's.
        # With no such call the model resolves 'auto' and, on CUDA with two
        # or more visible devices, takes the automatic widening path.
        self.device_layout_is_automatic = True
        # The (sinogram_shape, recon_shape) pair the current automatic layout
        # was decided from, or None while no automatic decision is in force.
        # _apply_device_policy compares it against the current shapes on
        # every call: equal shapes reuse the settled layout without a new
        # search, and different shapes clear this record and re-decide.
        # configure_devices clears it too, so a pinned model carries no
        # settled record.
        self._settled_shapes = None
        # The workload the settled layout's capacity check was made against
        # ('recon' or 'direct').  A call that allocates more than that check
        # priced re-runs the check on the settled layout, so a reconstruction
        # can never reach the allocator with no preflight behind it.
        self._settled_workload = None
        # Device counts the automatic choice turned down, and why, for the run
        # log's device line.  Empty when the layout was never searched.
        self.device_choice_rejections = []
        # Set by the automatic search when it reaches a count the speed floors
        # hold back, so _settle can say whether that count was merely tried or
        # actually taken (see _speed_ordered_candidates).
        self._speed_floor_fallback = None
        # Every count the floors held back on this pass, so _settle can name
        # the wider ones in the run log once the chosen count is known.
        self._speed_floor_held = None
        # Memory-preflight knobs, beside the other memory knobs
        # (view_batch_size above, and the slice-band attributes the banded
        # drivers read).  The margin covers what the closed-form ledger cannot
        # see: allocator fragmentation, and library workspaces CUDA allocates
        # outside torch's allocator.  The preflight runs when the automatic
        # layout is decided, so setting skip_memory_preflight after a model
        # has settled changes nothing until a shape change re-decides.
        self.skip_memory_preflight = False
        self.memory_preflight_margin = 0.15
        # The last ledger built, and the last calibration comparison, for a
        # harness to read.  Inspection surfaces only: nothing in the library
        # reads them back, and the calibration entry stays None unless the
        # calibration mode is on.
        self.last_memory_ledger = None
        self.last_memory_calibration = None
        # A per-device thread pool reused across the recon loop's many
        # fan-outs (vcd_recon owns its lifetime); None outside a recon, where
        # each fan-out makes a private pool.  Never created on one device.
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

    # recon_placement / sino_placement are the single source of truth for how
    # the two array types are distributed (the mbirjax structure): sino-like
    # arrays shard by VIEW (axis 0), recon-like arrays by SLICE (the last
    # axis).  Unset, they are the trivial single-device placements;
    # configure_devices and the automatic widening replace them.
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

        DEFAULT = one band per slice-owner (the whole shard).  This differs
        from mbirjax deliberately, on measurement: mbirjax's sweeps found
        time flat across B, so it streams by default for the memory win, but
        the torch banded pass pays a fixed orchestration cost per band.
        With the compiled kernels in place, splitting the shard into
        sub-bands was measured on four H100s at 2 to 23 percent more busy
        time at parallel 1024 with two devices, depending on the walk (job
        mg10, 2026-08-10; an earlier pre-kernel reading of 47 to 66 percent
        overstated the cost).  The one exception, a 9.5 percent win at the
        63-slice walk, is non-monotonic and unexplained, and is not a basis
        for a default.
        Time buys nothing back here because a single torch device never runs
        the banded drivers at all (the trivial fast path uses the plain
        projectors), so mbirjax's stream-even-at-n=1 rationale is void.

        A smaller B remains a real MEMORY lever (the per-band broadcast
        copy, the per-band partial, and the running total each slice-owner
        reduces into all scale with B; the same mg10 sweep read per-device
        peaks of 11.84 to 11.97 GB across the sub-band walks against 12.48 GB
        at the default, with total copied bytes unchanged).  That sweep
        predates the streamed reduce, which took the default-B reduce from n
        whole bands down to two plus a bounded slab, so expect a narrower gap
        than those peaks show.  Set
        ``forward_project_slice_band`` / ``back_project_slice_band`` on the
        model to opt in with a fixed B when a run is memory-constrained.
        Every result is capped at slices_per_dev so a band never crosses a
        slice-owner boundary."""
        b = fixed_band if fixed_band else slices_per_dev
        return min(int(b), slices_per_dev)

    def _column_gather_forward(self):
        """Whether the multi-device forward gathers pixel COLUMNS instead of
        walking slice bands.

        Three things have to agree, and each guards a different mistake.

        The GEOMETRY must be one the column gather has been measured for.
        ``column_gather_geometry`` is set by cone beam and by parallel beam,
        which want the same full slice range for two different measured
        reasons.  Cone NEEDS it: one slice projects onto a range of detector
        rows, so a band-sized call still writes every row and costs what a
        full call costs.  Parallel merely wants it: its forward kernel runs
        about twice as efficiently per slice on a full-width block of values
        as on the shard-width blocks the banded walk hands it at more than
        one device.  Translation and multiaxis share cone's banded
        branch and its band-independent per-call cost, so the shape should
        help them too, but neither has ever been timed on it and neither
        should be switched over on an argument.

        The SWITCH must not be off.  ``forward_column_gather`` unset means
        the gather runs: it is the shipped behaviour for the geometries that
        declare the capability, gated on measured speed, value, and memory
        (2026-08-11, four H100s, both geometries).  Setting it to False
        selects the banded walk, which stays in place as the rollback.

        The ENVIRONMENT may override the switch either way, which is what
        lets one session run both shapes over the same inputs and compare
        their values.
        """
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
        """Shared setup for the banded sharded projectors: the per-owner view
        spans, the recon band ranges (global slice order, over the padded
        device-form axis), and the pixel indices placed once per device.

        Padding semantics: each view-owner's span covers only its REAL views
        (``n_valid``), so no projector call ever runs at a padded view -- a
        padded view has no angle.  The forward driver zero-fills each owner's
        padded view tail after assembly; the back driver reads only the real
        views and masks the padded slice tail after the band reduce.  The
        geometry's projectors must define the view-range seams (parallel beam
        and cone do)."""
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
        """The banded sharded forward (mbirjax's _forward_project_all_bands):
        visit slice-owners in global slice order, broadcast each band to every
        view-owner, forward-project each owner's OWN views from the band (a
        single producer per detector row -- no reduce), and concatenate the
        row-bands into per-view-owner sinogram shards.

        A trivial (one-shard) placement is the whole-volume band on the whole
        view range: the plain driver, wrapped -- so the uniform VCD loop costs
        a single device nothing and needs no view-range seams there.

        Under padding each owner projects only its REAL views (padded views
        have no angles), and its padded view tail is zero-filled after
        assembly, keeping the device form inert end to end.

        A geometry whose slices spread over a range of detector rows can take
        :meth:`_sparse_forward_project_columns` instead, which cuts the
        cylinder the other way; :meth:`_column_gather_forward` says when."""
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
                        # Accumulated -- release the name, the same release
                        # the back driver makes.  The next band's
                        # `partials = run_per_device(...)`
                        # evaluates BEFORE rebinding, so an un-released list
                        # keeps the previous band's full-row partial live on
                        # every device through the next band's projection: one
                        # sinogram shard per device, uncharged.  The aligned
                        # branch above has no such release to make -- its
                        # `row_bands` tensors are appended to `view_bands`, so
                        # they are the OUTPUT and stay live either way.
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
        cylinder at every slice from every slice-owner, and makes ONE
        projector call per batch over its own views and the whole slice
        range.  The alternative to the banded walk in
        :meth:`_sparse_forward_project_sharded`, for the geometries
        :meth:`_column_gather_forward` admits.

        WHY the shape exists, for the two geometries that take it.  A
        geometry whose slices spread over a range of detector rows pays per
        projector call whatever the slice band contains, because the call's
        output spans the whole detector either way.  Walking one band per
        slice-owner therefore costs that owner count times one full call, and
        the forward stops falling when devices are added -- measured flat at
        32.2, 30.6 and 30.5 s over one, two and four devices.  A full-height
        call per pixel batch is the shape the single-device path already
        runs, so the work divides with the view split the way it was meant
        to.  Measured 2026-08-10 on four H100s, job mg10: cone's per-device
        forward fell from 29.7 to 19.4 s at two devices and from 29.3 to
        15.3 s at four, with a lower peak.

        A ROW-ALIGNED geometry's banded walk does divide the work, so its
        reason is the other one: the forward kernel is about twice as
        efficient per slice on a full-width block of values as on the
        shard-width blocks the banded walk hands it, and this shape hands it
        full width at every device count.  Measured 2026-08-10 on one H100,
        at 0.0411 ms per slice on a 1008-wide block against 0.0823 on a
        504-wide one with the device count held at one.

        WHAT DOES NOT MOVE.  Every view-owner still produces its own views'
        whole sinogram block, from the same voxels, through the same body, so
        the operator is unchanged and the sharded forward stays the adjoint
        of the sharded back.  Only which device assembles which voxels
        changes, and the back driver is untouched.  Two summation orders do
        change for a two-fan geometry: the vertical sum moves from a host-side
        sum across bands into the body, and the pixel sum moves the other way,
        from the body into the host-side sum across pixel batches.  Both sit
        inside the value class the forward already has.  A row-aligned
        geometry has no vertical sum to move, its rows being concatenated
        rather than added, so the pixel sum is the whole of what changes
        there, and nothing changes at all when one batch covers the pass.

        The two skips of the banded form are kept or dropped deliberately.  A
        view-owner with no real views receives no gathers and produces an
        empty block, as before.  The banded form's all-padding sub-band skip
        has no counterpart here, because a gathered cylinder spans every
        slice-owner at once; the padding it carries is inert (see
        :func:`_sharding.gather_column_band`).

        Each batch's gather is issued ONE BATCH AHEAD of the projection that
        reads it, and on CUDA its copies run on a stream of their own, so a
        device projects one batch while the next batch's values are still
        moving to it.  Issuing early is what makes that possible; the separate
        stream is what makes it happen, because a stream runs its work one
        item at a time and copies sharing the projection's stream could only
        take turns with it.  The batches are summed in the same order they
        always were, so the values do not move; what changes is that a device
        holds one more cylinder at once, which the memory ledger charges
        (COLUMN_GATHER_RESIDENTS).  The comment at the gather gives the full
        ordering argument, and off CUDA the gather stays the synchronous one it
        has always been.

        Each batch after the first adds into the owner's block from INSIDE the
        projector's view loop (``accumulate_into``), rather than receiving its
        own block for the driver to add.  That drops a full-block pass and a
        full-block allocation per batch -- a cost that does not shrink with the
        batch, so bigger batches only hide it -- and it lowers the widest
        instant by the block it no longer allocates.  The summation order is
        unchanged, element for element.  The comment at the accumulation carries
        the argument, including why a preallocated zeroed buffer would be worse
        rather than better.

        ``forward_project_slice_band`` has nothing to act on here, because
        this shape does not band the slice axis at all; what bounds the
        transfer instead is the pixel batch.  The memory ledger stops
        charging the band copy to match.  ``back_project_slice_band`` is
        unaffected, the back driver being untouched."""
        sp, rp, view_spans, _band_ranges, idx_per = self._banded_setup(
            pixel_indices)
        pf = self.projector_functions
        num_channels = int(self.get_params('sinogram_shape')[2])
        # How tall a block one call returns, which is what the empty blocks
        # below have to match.  A row-aligned geometry's body sizes its output
        # by the values it was handed, and the gathered cylinder is the whole
        # DEVICE-form slice axis -- padded tail included, which is exactly the
        # length that geometry's sinogram pads its detector rows to.  A
        # geometry whose slices spread over a range of rows returns the real
        # detector rows whatever it is handed.
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
                # Issue the NEXT batch's gather before this batch is
                # projected, rather than after, so its copies are already
                # moving while this projection runs.  Nothing here waits for a
                # value: run_per_device performs no synchronization, and the
                # gather returns once its copies are issued.
                #
                # THE ORDERING, end to end.  Four things arrange it, and each
                # covers a different way the copies and the projections could
                # get in each other's way.
                #
                # The copies run on a stream of their own, one per device
                # (:func:`_sharding.copy_stream`).  A stream runs its work in
                # the order it was given, one item at a time, so copies left
                # on the stream a device projects on could only take turns
                # with the projections, however early they were issued.  On
                # their own stream the two run at once.
                #
                # Before any copy starts, each copy stream waits for its
                # device's compute stream, so a copy cannot read a shard
                # before the kernel that wrote it has finished
                # (``open_copy_streams``, called once below).
                #
                # Every batch carries its OWN event, recorded on the copy
                # stream once that batch's copies and their concatenation are
                # queued.  The compute stream waits for that one event just
                # before the projection that reads that batch, so a projection
                # never starts on a cylinder that has not arrived -- and never
                # waits for the batch gathered ahead of it, which is the work
                # meant to be moving right now.
                #
                # After the pass, each compute stream waits for its copy
                # stream (``close_copy_streams``), so a later update cannot
                # overwrite a shard while a copy is still reading it.
                #
                # Off CUDA none of this applies: the gather copies
                # synchronously, returns no event, the wait below does
                # nothing, and the values are the ones the plain path has
                # always produced.
                ahead = gather(k + 1) if k + 1 < len(batch_bounds) else None
                _sharding.wait_for_column_band(dev, ready)
                # THE ACCUMULATION.  The first batch's projection allocates the
                # owner's block and fills it; every later batch adds into that
                # same block from inside the projector's own view loop, which
                # is where the block was going to be written anyway.
                #
                # What this removes, per batch after the first: the projector
                # allocated a fresh full block, copied its view batches into
                # it, and handed it back for the driver to add -- two full-block
                # passes and one full-block allocation where a single pass does
                # the same work.  The cost is the same at every batch size, so
                # it is one the bigger batches HIDE rather than remove, and a
                # 1024-class pass at the default batch runs on the order of a
                # hundred of them.
                #
                # NOT a preallocated zeroed buffer, which is the shape this
                # looks like from a distance and is strictly worse: adopting
                # the first batch's block, as below, costs no zero-fill and no
                # add, while a zeroed buffer pays both.
                #
                # THE VALUES DO NOT MOVE.  Per element the sequence is still
                # batch 0's contribution, then batch 1's added to it, then batch
                # 2's -- the same summands added in the same order as the
                # driver-side add did.  Only where the addition happens changes,
                # so the result is bit for bit what it was.
                #
                # STREAM LIFETIMES are untouched, and the persistent block needs
                # no record_stream.  It is allocated, written and added into
                # ONLY by this device's compute stream, in program order, and a
                # stream runs its work in order; the copy streams read the
                # slice-owners' shards and write the gathered cylinders, and
                # never touch this block.  Holding it across batches instead of
                # freeing it each time also keeps its memory out of the caching
                # allocator between batches, so it can never be handed to a
                # copy stream mid-pass.
                if owned is None:
                    owned = pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i)
                else:
                    pf.sparse_forward_project_view_range(
                        full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                        dev_index=i, accumulate_into=owned)
                # Released once the projection that reads it has been issued, so
                # a device carries this batch's cylinder no further.  With the
                # gather ahead of it, the batch after this one is already
                # resident by now, which is the third cylinder the memory ledger
                # charges (COLUMN_GATHER_RESIDENTS).  The release moved after
                # the accumulation because the accumulation moved INTO the
                # projection; the widest instant is narrower than it was, the
                # separate incoming block having gone.
                full_cyl = None
            if owned is None:
                # No pixels at all: the owner still owes its views' block,
                # and the banded form would have produced it as zeros too.
                owned = torch.zeros((v1 - v0, num_rows, num_channels),
                                    dtype=voxel_shards.dtype, device=dev)
            return owned

        # ONE fan-out for the whole call, with the pixel loop inside the
        # worker: a fan-out per pixel batch would issue a thread dispatch per
        # (batch, device), and putting the loop inside also issues each
        # device's gathers from the thread that consumes them.
        #
        # The copies read the slice-owners' shards and land on the
        # view-owners, so both sets of devices have a copy stream to order
        # (see the comment at the gather above).  Off CUDA both calls do
        # nothing.
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
        and the partials sum onto the band's slice-owner.

        A trivial (one-shard) placement is the plain driver, wrapped (see
        :meth:`_sparse_forward_project_sharded`).

        Under padding each view-owner back-projects only its REAL views, and
        the padded slice tail of the last slice-owner is re-zeroed after the
        band reduce: back projection is a gather, so real detector data DOES
        land in padded slice positions (unlike padded views, which are never
        computed), and leaving it there would break the forced-zero padding
        invariant the prior and the stats rely on."""
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
                    # The partials are consumed by the reduce.  Release the
                    # name here: the next band's `partials = run_per_device(...)`
                    # evaluates its call BEFORE rebinding, so without this the
                    # previous band's partial stays live on every device for the
                    # whole of the next band's projection -- one cylinder per
                    # device, uncharged.  (The `weighted_fwd` treatment.)
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
        device, cached per (recon_shape, use_ror_mask) -- the hot consumers
        (forward/back_project, the differentiable wrappers) call this per
        invocation, and rebuilding + re-uploading the indices each time was a
        measured per-call cost.  A custom mask array bypasses
        the cache (unhashable).

        The cache is IN-MEMORY only: a single (key, tensor) entry held on the
        model instance, replaced in place when the key changes and freed with
        the model.  Nothing is written to disk (the on-disk state under
        ``~/.mbirtorch`` is the torch.compile cache; see ``clear_cache``)."""
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
        Without this, a geometry-changing set_params after configure_devices
        left the placements' real sizes stale, and the placement functions
        sliced with stale ranges -- silently truncating sharded arrays (the
        mbirjax counterpart re-runs its device setup on every recompile).
        Single-device placements are rebuilt too: their real sizes feed the
        same consumers (e.g. the cone DC-damping profile)."""
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
        axes.

        Padding rounds each non-dividing axis UP to a multiple of the device
        count, and for some (size, count) pairs the rounding leaves trailing
        devices entirely padded on an axis.  mbirjax refuses any such
        layout; here a device that is all-padding on ONE axis is legal and
        useful -- two deliberate extensions beyond mbirjax (Greg,
        2026-08-06).  With fewer SLICES than devices (thin volumes, many
        views) the extra devices still project their views, which is the
        dominant compute and memory there.  With fewer VIEWS than devices
        (sparse-view, large volumes) the extra devices still hold slice
        shards and run the prior and updates, which dominate there.  Either
        way the padding invariants keep the empty axis exactly inert: the
        drivers skip its projector calls, its reductions see only zeros,
        and the interface masks neutralize any halo sourced from padding.
        A device with no real data on EITHER axis would do nothing at all,
        so that layout is refused."""
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
        """Drop every cache keyed to the device layout or geometry: the prox
        initialization (its partitions are device tensors of the old layout),
        the per-device qGGMRF interface masks, and a geometry's update-
        direction profile cache (cone DC damping).  Called on every
        configure_devices and recompile, so no consumer can bind stale
        device-resident state (the mbirjax stale-bind lesson)."""
        self.prox_data = None
        self._qggmrf_interface_masks_cache = None
        self._dc_damping_cache = None

    def _qggmrf_interface_masks(self):
        """Per-device qGGMRF interface masks for a padded slice axis, or None.

        The qGGMRF kernel masks the inter-slice DIFFERENCES: interface j of a
        shard starting at global slice g0 is valid iff its higher-index slice
        is real (``g0 + j < num_real_slices`` -- the one padding predicate).
        Masking an interface reproduces the reflected boundary condition
        there, so the prior sees the recon as ending at the last REAL slice
        even mid-shard.  When the slice axis is padded EVERY shard gets a
        mask (all-ones for fully real shards); when nothing is padded this
        returns None and the kernel call carries zero new work.

        The masks depend only on the device layout, so they are built once
        and cached (``_invalidate_device_caches`` drops them on every
        reconfigure/recompile).  Building them per VCD subset would re-pay a
        host->device transfer thousands of times (the staged-halos lesson).

        Returns:
            list ((local_slices+1,) float32 tensor per device, in device
            order), or None.
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
        """Set the device layout, and take it out of the library's hands.

        This is the ONE place a device choice is expressed.  The model
        constructors take no device argument, so every explicit choice comes
        through here: a count (``num_devices=n``), or a list
        (``devices=['cpu']``, ``['mps']``, ``['cuda:1']``, or several CUDA
        devices).

        Without a call to this method, the model resolves its device lazily,
        preferring cuda, then mps, then cpu.  On CUDA with two or more visible
        devices it then spreads a reconstruction across the devices that can
        hold their share, judged by a memory check that runs before the first
        large allocation.  ``configure_devices(num_devices=1)`` is the
        reproducibility pin, and the ``MBIRTORCH_NUM_DEVICES`` environment
        variable pins the count process-wide for a suite or a nightly.
        Results can differ slightly with the device count, and the difference
        decays as iterations proceed.

        It rebuilds the sino (view-axis) and recon (slice-axis) placements
        over ``num_devices`` CUDA devices, or over the explicit device list.

        The placements' real sizes come from the CURRENT params
        (sinogram_shape / recon_shape), so call this after any geometry
        change -- and note the mbirjax stale-bind lesson: this RECREATES the
        projectors so nothing keeps a stale single-device binding.

        A single device (the default) restores the trivial placements and
        the unchanged n=1 path.

        Calling this at all takes the layout out of the library's hands: the
        count given here is the count used, the memory preflight no longer
        second-guesses it, and the automatic device-count choice never runs
        again on this model.  ``num_devices=1`` is therefore the way to pin a
        run to one device for reproducibility.

        Without such a call, a CUDA model spreads a reconstruction across the
        devices that can hold their share; see :meth:`recon`.
        """
        self.device_layout_is_automatic = False
        # An earlier automatic settle may have left rejected counts behind.
        # They explain a search this layout did not come from, so the run log
        # must not carry them into a run the caller placed by hand.
        self.device_choice_rejections = []
        # The settled record is likewise the automatic path's; a pinned model
        # must not carry one.  The explicit branch never reads it, so this
        # keeps the two states consistent rather than changing behavior.
        self._settled_shapes = None
        self._settled_workload = None
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
        :meth:`configure_devices` and of the automatic device-count choice.

        This carries no policy.  It does not touch
        ``device_layout_is_automatic``, so the automatic path can install a
        layout without pretending the caller asked for one.
        """
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
    def _build_memory_ledger(self, devices=None, workload='recon',
                             **call_arrays):
        """The modeled per-device peak for one candidate device list.

        ``devices`` prices a CANDIDATE list rather than the model's current
        one, which is what lets the automatic choice evaluate a layout the
        model is not in.  None means the current placement.  ``workload``
        names the call to price; see :func:`_memory_ledger.plan_from_model`.

        The ledger math is device-agnostic, so this builds one for any
        backend.  WHETHER to consult one is the policy's decision, not this
        function's, which is also what lets the rule be tested on CPU.
        """
        devices = list(self.sino_placement.devices if devices is None
                       else devices)
        return _memory_ledger.estimate_peak_device_bytes(
            _memory_ledger.plan_from_model(self, devices, workload=workload,
                                           **call_arrays))

    def _candidate_devices(self, num_devices):
        return [torch.device(f'cuda:{i}') for i in range(num_devices)]

    def _shape_pair(self):
        """The (sinogram_shape, recon_shape) tuple pair the automatic policy
        records at settle time and compares on every later call.  These two
        shapes are what the memory ledger's plan is built from, so a change
        in either invalidates a settled decision."""
        sinogram_shape, recon_shape = self.get_params(['sinogram_shape',
                                                       'recon_shape'])
        return tuple(sinogram_shape), tuple(recon_shape)

    def _apply_device_policy(self, workload='recon', **call_arrays):
        """Settle the device layout for the reconstruction about to run, and
        return the ledger for the layout settled on.

        ``workload`` names the call in progress: ``'recon'`` (the default) for
        a full reconstruction, ``'direct'`` for a direct reconstruction.  It
        tells the ledger what is about to be allocated; it is not a way for a
        caller to overrule the policy.

        This is the one site where the automatic device count is chosen, and
        it is deliberately not model construction.  Two reasons carry that.
        The ledger needs a free memory reading, and that is only knowable when
        the reconstruction is about to start; a reading taken at construction
        would be trusted while stale.  And a developer calling a projector
        directly has not asked for a layout change, so the reconstruction
        entries are the right scope.

        A third reason used to be stated here and no longer holds:
        ``QGGMRFDenoiser`` once had no multi-device loop, so a
        construction-time choice could have handed it a layout it could not
        run.  The 2026-08 prerelease gave the denoiser a real sharded loop, so
        that constraint is gone.  What is true today is narrower: the denoiser
        never enters the reconstruction entries that call this method, so
        it reaches several devices only through an explicit
        :meth:`configure_devices` call -- and the first branch below already
        treats an explicit layout as the caller's, neither searched nor
        reduced.  The denoiser is to join the automatic policy once it has a
        ledger and floors of its own; until then, nothing in this method
        assumes either state.

        The choice is made ONCE per model and kept.  Settling records the
        (sinogram_shape, recon_shape) pair it was decided from, and while
        those shapes hold, every later call returns the settled layout
        without a search.  The layout therefore never moves mid-pipeline, so
        sharded arrays a caller still holds -- a prepared sinogram, a
        precomputed Hessian, a Plug-and-Play loop's previous output -- stay
        valid.  A shape change invalidates the ledger inputs the decision
        came from, so it clears the record and the next call re-decides.  A
        change in free device memory never re-decides; the remedy for changed
        conditions is :meth:`configure_devices` or a new model.  The
        ``MBIRTORCH_NUM_DEVICES`` pin is read when the model settles, so
        changing it later moves only models that have not settled yet.

        Capacity is not the only rule here.  On the unpinned automatic branch
        the candidate ORDER comes from the widening speed floors
        (:meth:`_speed_ordered_candidates`), which put the counts worth using
        at this problem size ahead of the counts that would only run it
        slower.  Nothing is removed, so capacity still wins whenever nothing
        admitted fits.  Both pin branches above skip the floors by
        construction: a count the caller named is not the library's to
        second-guess.

        Two decisions are made from two different plans.  The COUNT is chosen
        with the full recon plan, because the settled layout serves the
        model's whole life and the most demanding call it may later carry is a
        full ``recon``.  The capacity check that can REFUSE is made against
        the plan for the call in progress, so a direct reconstruction is not
        turned away for a reconstruction it is not running: when no count fits
        a full recon, the search settles on the first candidate that fits the
        work in progress and refuses only when nothing fits even that.  The
        workload whose check the layout passed is recorded beside the shapes,
        and a later call that allocates more than it re-runs the check on the
        settled layout -- the layout does not move, only the check runs.

        Note that the halves of ``split_sino_recon`` arrive HERE.  Since the
        2026-08 prerelease change they inherit no explicit layout from the
        parent unless the parent had one, so each half chooses for itself --
        at its own, smaller sinogram, which is exactly the size the floors are
        asked about.
        """
        calibrating = _memory_ledger.calibration_enabled()
        if not self.device_layout_is_automatic:
            # An explicit layout is the caller's; the ledger runs only when
            # the calibration mode asks for it.
            ledger = self._build_memory_ledger(**call_arrays) if calibrating \
                else None
            return self._arm_calibration(ledger)

        if self._settled_shapes is not None:
            if self._settled_shapes == self._shape_pair():
                if _memory_ledger.workload_covers(self._settled_workload,
                                                  workload):
                    # The automatic choice for these shapes is settled and the
                    # settled check already covers this call: reuse the layout
                    # without a search, as the pinned branch above reuses an
                    # explicit one.  The ledger runs only for the calibration
                    # mode, and it prices a full reconstruction because that
                    # is the scope the measured peak covers.
                    ledger = self._build_memory_ledger(**call_arrays) \
                        if calibrating else None
                    return self._arm_calibration(ledger)
                # This call allocates more than the settled check priced, so
                # the check runs again -- on the settled layout, which does
                # not move.
                ledger = self._check_settled_capacity(workload, call_arrays)
                # It passed, so the record now names the workload the layout
                # is known to hold, and a later call of the same kind repeats
                # no check: the preflight stays a once-per-model cost.
                self._settled_workload = workload
                return self._arm_calibration(ledger)
            # The shapes changed, so the settled decision's inputs are gone:
            # drop the record and re-decide below.
            self._settled_shapes = None
            self._settled_workload = None

        pinned = _memory_ledger.pinned_device_count()
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible < 2:
            # A single visible device has no layout to choose, so there is
            # nothing for the ledger to decide and it does not run.  Torch's
            # caching allocator already raises a fast, readable error on a
            # single-device overflow, which is the job the preflight exists to
            # do where the allocator cannot.  This also keeps the n=1 path free
            # of any new per-reconstruction cost.
            return self._arm_calibration(None)

        if pinned is not None:
            # A process-wide pin is as explicit as a configure_devices call:
            # the count is not searched and never reduced.  The empty-shard
            # validation and the preflight still apply to it.  The speed
            # floors do not: the pin named the count.
            candidates, held = [min(pinned, visible)], {}
        else:
            candidates, held = self._speed_ordered_candidates(visible)

        rejected, best = [], None
        self._speed_floor_fallback = None
        # Handed to _settle, which names every WIDER held count once the
        # chosen count is known.  Holding a small problem at one device is the
        # guard's commonest action, and the loop never reaches the counts it
        # held, so without this the idle devices would go unexplained.
        self._speed_floor_held = held
        for count in candidates:
            if count in held:
                # Every admitted count comes first, so reaching a held one
                # means none of them was usable and capacity is about to
                # override the speed rule.  Record the floor now, while the
                # outcome is still unknown; if this count is then SETTLED on,
                # _settle rewrites the note to say the floor was overridden.
                held_note, taken_note = held[count]
                rejected.append((count, held_note))
                self._speed_floor_fallback = (count, taken_note)
            devices = (self._candidate_devices(count) if count > 1
                       else [self.torch_device])
            if not self._layout_is_valid(devices):
                rejected.append((count, 'a device would own no real data'))
                continue
            # Priced as a full recon whatever this call is: the count chosen
            # here has to suit the largest workload the model may later run.
            ledger = self._build_memory_ledger(devices=devices, **call_arrays)
            if ledger is None or self.skip_memory_preflight:
                # Nothing to check against, or the caller has forced the run.
                return self._settle(devices, ledger, rejected)
            fits, rows = self._layout_capacity(devices, ledger, call_arrays)
            if fits:
                return self._settle(devices, ledger, rejected)
            shortfall = max((d - b) for _dev, d, b in rows if b is not None)
            rejected.append((count, f'{shortfall / 2 ** 30:.2f} GB short'))
            if best is None or shortfall < best[0]:
                best = (shortfall, ledger, rows, count)

        if workload != 'recon' and best is not None:
            # No count fits a full recon, and this call is not running one.
            # The check that can refuse is made against the work in progress,
            # in the same candidate order, so the count is still the one the
            # floors and capacity prefer.  Only what it is checked against
            # changes.  The shortfall reported below then describes the check
            # that actually refused, so the search starts its record over.
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
                    # The first pass recorded this count as refused, priced
                    # for a recon this call is not running, and the device
                    # line must not call the count in use rejected.  A
                    # speed-floor note pending for the same count goes with
                    # it: both explain this choice, and the note below names
                    # the rule that admitted the count.
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

        The floors REORDER, they never remove: admitted counts largest-first,
        then held counts largest-first.  Two consequences are the point of
        doing it this way.  Capacity always wins -- a held count is reached
        only after every admitted count has been refused, so a problem that
        genuinely needs four devices still gets them.  And
        ``skip_memory_preflight`` does not disable the guard: that flag makes
        the loop settle on its FIRST candidate, which this ordering has
        already made the first ADMITTED count rather than the widest one.
        The floors are a speed rule, so forcing past the capacity check
        leaves them in force.

        Every held count WIDER than the one finally chosen is named in the
        run log by :meth:`_settle`, whether or not the loop reached it.  That
        matters because the guard's commonest action -- holding a small
        problem at one device -- reaches none of the counts it excluded.

        Returns:
            (list, dict): the candidate counts in the order to try them, and
            ``{count: (held_note, taken_note)}`` for the held ones -- the
            note if the count is passed over, and the note if capacity ends
            up settling on it anyway.
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
        if hasattr(self, 'split_sino_recon'):
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

    def _layout_capacity(self, devices, ledger, call_arrays):
        """Whether ``ledger``'s modeled peak fits ``devices``, and the rows to
        report it with.  The one place a budget reading is compared with a
        modeled demand, so the search, the narrower second pass and the
        settled re-check all ask the question the same way."""
        budgets = [_memory_ledger.device_budget_bytes(d) for d in devices]
        credits = _memory_ledger.resident_credits(
            devices, list(call_arrays.values()))
        return _memory_ledger.layout_fits(
            ledger, budgets, credits, margin=self.memory_preflight_margin)

    def _check_settled_capacity(self, workload, call_arrays):
        """Run the capacity check for ``workload`` on the layout already
        settled, and return the ledger it priced.

        The layout does not move here: the model has settled and a caller may
        be holding shards of it.  What this can do is refuse, which is the
        point.  A model that settled under a direct reconstruction was checked
        against what that reconstruction allocates, so without this a later
        ``recon`` would reach the allocator with none of the preflight's
        message and remedies behind it.
        """
        if self.skip_memory_preflight:
            # The caller has forced the run, here as in the search.
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
        # The search records a speed-floor note the moment it REACHES a held
        # count, before the outcome is known.  A count it then settles on was
        # not turned down, so that note is replaced by what actually
        # happened: capacity found nothing admitted and went past the floor.
        fallback = getattr(self, '_speed_floor_fallback', None)
        if fallback is not None and fallback[0] == chosen:
            rejected = [fallback if count == chosen else (count, why)
                        for count, why in rejected]
        # Now that the count is known, name every WIDER count the floors held
        # back.  Most of them the loop never reached -- it settled on an
        # admitted count first -- so this is the only place they can be
        # explained, and holding a small problem down is exactly the case a
        # user needs explained.  A held count SMALLER than the chosen one was
        # outranked rather than excluded, so it carries no entry.
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
        # Record the shapes this decision came from.  While they hold, later
        # policy calls reuse the layout instead of re-deciding; a shape
        # change clears the record (see _apply_device_policy).  The workload
        # beside them is the one the capacity check was made against, so a
        # later call that allocates more re-runs that check.
        self._settled_shapes = self._shape_pair()
        self._settled_workload = workload
        return self._arm_calibration(ledger)

    def _arm_calibration(self, ledger):
        """Record the ledger for a harness to read; under the calibration
        mode, build one when the caller had none, so a policy return always
        carries a ledger to compare against.

        The peak-counter reset the calibration mode compares against lives in
        :meth:`vcd_recon`, beside the report that reads the counters.  A
        reset here would run on every policy return, and the nested return
        inside a reconstruction (vcd_recon -> direct_recon -> policy) would
        clear the peak after the sinogram and weights were already placed,
        under-measuring the run."""
        if ledger is not None:
            self.last_memory_ledger = ledger
        if _memory_ledger.calibration_enabled() and ledger is None:
            ledger = self._build_memory_ledger()
            self.last_memory_ledger = ledger
        return ledger

    # ── array placement (entry) and gathering (exit) ──────────────────────────
    # Every sinogram-like placement routes through _shard_sinogram and every
    # recon-like placement through _shard_recon; the exits route through the
    # matching gathers.  Multi-device support (validation, movement, padding,
    # cropping) therefore changes these four functions alone instead of every
    # call site (mbirjax calls the same functions its 'chokepoints').
    def _shard_sinogram(self, sinogram):
        """Place a sinogram-like array (sinogram or weights) in its device
        form: float32 on the model device, with the view axis checked against
        the model.

        In mbirjax this is the pad-aware VIEW-SHARDING placement: a sharding
        port zero-pads and distributes here (and crops in
        :meth:`_gather_sinogram`), so routing every sinogram-like placement
        through these two functions means multi-device support changes them
        alone instead of every entry point.
        """
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

    # Whether this geometry's projection ties detector row r to recon slice
    # r 1:1 (parallel beam: True).  False is correct for every geometry
    # whose slices spread over many rows (cone, translation, multiaxis) --
    # and is deliberately the base value, because the banded multi-device
    # drivers take the row-aligned fast path when this is True and would
    # silently mis-assemble a geometry that forgot to declare itself.
    rows_track_slices = False

    # Whether this geometry's multi-device forward MAY gather pixel columns
    # instead of walking slice bands (see _column_gather_forward).  False is
    # the base value because the shape has been measured on cone beam and
    # parallel beam only: translation and multiaxis have the same
    # band-independent per-call cost as cone and should gain from it too, but
    # a geometry is switched over on its own measurement rather than on the
    # argument.  Declaring this True is what lets the gather run, and it runs
    # by default -- forward_column_gather = False selects the banded walk.
    column_gather_geometry = False

    # The fewest pixels this geometry's COMPILED bodies may be called with.
    # 1 -- the base value -- means any width, which is what a geometry whose
    # compiled bodies are all correct wants.  A geometry that declares more
    # gets narrow calls padded up to that width and unpadded again outside the
    # compiled region (see projectors.forward_at_min_pixel_width, which also
    # carries the measured reason parallel beam declares 2).  It is a property
    # of the geometry's bodies rather than a user setting, so it is a class
    # attribute and not a parameter.
    min_compiled_pixel_width = 1

    # Which measured set of widening speed floors governs this geometry's
    # automatic device count (see _widening_floors).  None -- the base value
    # -- means the parallel floors, which are the more permissive measured
    # set, so a geometry that has never been measured is slowed by the guard
    # in no case where parallel beam would not be, and the reason string says
    # the substitution happened.
    _floor_family = None

    def _sino_row_padding(self):
        """Detector-row padding spec for the sinogram device form, or None.

        Derived from :attr:`rows_track_slices`: a geometry whose kernels tie
        detector rows to recon slices must present the SAME padded length on
        both axes, so when the recon slice axis is padded for sharding the
        sinogram's (unsharded) row axis pads with it -- zero-filled at entry
        and inert, exactly like the padded views.  Geometries without that
        tie keep their real rows and return None.

        Returns:
            (row_axis, real_rows, padded_rows) when row padding is active,
            else None.
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
        """Place a recon-like array (3-D, or flat (num_pixels, num_slices)) in
        its device form: float32 on the model device, with the slice axis (the
        LAST axis in both forms) checked against the model.

        The mbirjax counterpart is the pad-aware SLICE-SHARDING placement; see
        :meth:`_shard_sinogram` for the single-route rationale.
        """
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
        _shard_sinogram / _shard_recon): each device gets its contiguous block of the
        sharded axis, and a non-dividing axis is zero-padded on the LAST
        shard so the padding stays inert.

        Accepts either the problem's REAL length (pads) or the device-form
        PADDED length (an already-prepared array; re-split with no further
        padding); a mixed shape is refused as stale.  All zero tails are
        built ON the receiving device, so the only transient is one shard on
        one device, never a padded host copy (the mbirjax
        _pad_shard_on_axis semantics).

        ``row_pad`` is used in exactly one case -- the ParallelBeam sinogram,
        whose detector rows track the recon slices 1:1 and so must pad to the
        SAME device-form length as the (sharded, padded) slice axis even
        though rows are not the sinogram's sharded axis.  It carries
        ``(row_axis, real_rows, padded_rows)``; None everywhere else.
        """
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
        init, find the optimal alpha minimizing (1/2)||y - alpha A x0||_w^2
        (applied only to the default direct-recon init; a user-supplied init
        is used as-is), and scale both.  One conceptual operation for either
        state layout; a per-device state combines the alpha dot products from
        per-shard partial sums on the host and forms the error per view-shard
        locally."""
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
            # The weights product is dead once alpha is known, and holding it
            # through the error formation below made THIS function the peak of
            # a weighted reconstruction (the memory ledger's calibration).
            # Dropping the reference frees a full sinogram before the two
            # sinogram-sized allocations of the next line.  Under constant
            # weights the name aliases fwd, so this frees nothing and costs
            # nothing.  The sharded branch above never had the array: it fuses
            # the weights into per-shard dot products whose locals die on
            # worker return.
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
        """The uniform per-device container view of a device-form array: one
        code path for any device count (mbirjax's everything-is-sharded
        principle).  A plain tensor (the trivial single-device form) wraps
        as a one-shard container ALIASING the tensor -- no copy, so
        in-place updates through the container reach the caller's array --
        and an already-per-device state passes through.  Representation
        only: the input must already be in the device form (the placement
        placement functions own validation, movement, and padding)."""
        if isinstance(x, _sharding.Shards):
            shards = x
        else:
            shards = _sharding.Shards([x], placement)
        return shards

    def _as_device_form(self, x):
        """The inverse of :meth:`_as_shards`: back to the device form the
        placement functions produce.  A trivial one-shard container unwraps to its
        (aliased) tensor, so single-device callers and the checkpoint
        contract see plain tensors; a genuinely per-device state stays a
        :class:`_sharding.Shards` (collapsing it to one tensor would take a
        gather)."""
        if isinstance(x, _sharding.Shards) and x.placement.is_trivial:
            out = x.tensors[0]
        else:
            out = x
        return out

    def _sino_ones_device_form(self, sino_like=None):
        """All-ones sinogram in the device form, with any padded entries ZERO.

        The constant-weights Hessian path back-projects a ones sinogram, and
        padded views and padded detector rows must contribute nothing to it --
        a bare ``torch.ones`` at whatever shape the device arrays happen to
        have would silently add padded mass to the Hessian.  On a single
        device nothing pads, so this is a plain ones tensor at the params
        sinogram_shape; under a multi-device placement it is built per shard
        on each owner device, ones over the real views and rows and zero over
        every padded tail (mbirjax: ``sharded_full`` with the row-pad spec).
        ``sino_like`` supplies only the dtype; None defaults to float32.
        """
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
        Computes the diagonal of the Hessian matrix, which is computed by doing
        a backprojection of the weight matrix except using the square of the
        coefficients in the backprojection to a given voxel.  If weights is not
        None, it must be an array with the same shape as the sinogram; if None,
        constant weights of 1 are used.

        By default the indices cover ALL pixels of the grid (matching mbirjax's
        arange over the full grid, not the ROR-masked set).

        Args:
            weights (numpy or tensor, optional): 3D positive weights with the
                same shape as the sinogram.  Defaults to all 1s.
            output_sharded (bool, optional): If False (default), return numpy;
                if True, return the device tensor.
            indices (tensor, optional): back-project at these flat pixel
                indices only, scattering the result into a zero-filled volume.
                The entries outside the index set are ZERO rather than
                computed.  None (the default) keeps the full-grid behavior
                exactly, so the public contract is unchanged.

                The reconstruction loop passes its ROR-masked index set here.
                Every index the loop ever reads is inside that set, and the
                back projection is independent per pixel, so the values it
                reads are bitwise identical either way.  The saving is the
                masked set's smaller cylinder arrays; a square grid's mask
                drops about 21 percent of the pixels.

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
                                   output_sharded=False, row_weight=None):
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
                (FBP: 1/(delta_voxel * delta_voxel_row); FDK: alpha =
                delta_det_row / (voxel_volume * M_0)).
            output_sharded (bool): True returns the device tensor; False numpy.
            row_weight (tensor or None): optional (rows, channels) per-detector
                pre-weight (the FDK cosine map), broadcast over views.  None
                (default) is pure FBP.

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
            num_real_elements (int, optional): the number of REAL sinogram
                elements, when error_sinogram carries extra zero-filled padding
                (e.g. a padded view or row axis under a future sharding port).
                The padded entries contribute nothing to the sums, so
                normalizing by the real count gives exactly the unpadded loss.
                Default None uses error_sinogram.numel() (the unpadded case).

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

    # ── the VCD loop ────────────────────────────────────────────────────────
    def _get_update_direction(self, forward_grad, prior_grad, forward_hess,
                              prior_hess, pixel_indices, dev_index=0):
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
            forward_grad: data-term gradient, shape
                (num_subset_pixels, local_slices) -- one shard's slice band.
            prior_grad: prior gradient, same shape.
            forward_hess: data-term Hessian diagonal, same shape.
            prior_hess: prior curvature, same shape (a scalar on the
                proximal-map path).
            pixel_indices: flat in-plane indices of this subset.  Unused by the
                base implementation; spatially-aware overrides may use it.
            dev_index (int): which shard of the loop's per-device state these
                arguments belong to (0 on a single device).  Slice-profile
                overrides (cone DC damping) use it to select their shard-local
                profile; the base implementation is pointwise and ignores it.

        Returns:
            The update direction, same shape as forward_grad.
        """
        fn = maybe_compile(_diagonal_update_direction, self.compile_enabled)
        return fn(forward_grad, prior_grad, forward_hess, prior_hess)

    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        """
        Create a function to update a subset of pixels in the recon and error
        sinogram (mirrors mbirjax's create_vcd_subset_updater, with in-place
        torch state updates replacing jax's buffer donation).

        The updater is ONE body over the loop's uniform per-device state:
        each step runs per shard through :func:`_sharding.run_per_device` (a
        direct call on one device, one thread per device otherwise), the
        projections route through the banded drivers (the plain drivers on
        one device), the qGGMRF prior sees its neighbor slices across shard
        boundaries through halos staged once per partition pass (``None`` --
        the reflected boundary -- at true volume edges, so one shard
        reproduces the single-device prior exactly), and the line-search
        partials combine ON DEVICE as 0-d tensors -- no host synchronization
        inside the subset loop for any device count.

        Args:
            fm_hessian (tensor or Shards): (num_pixels, num_slices) diagonal
                of the Hessian for the forward model loss.
            weights (tensor, Shards, or 1): 3D positive weights with the same
                shape as the sinogram, or the constant 1.
            prox_input (tensor or Shards, optional): input for the proximal
                map, flattened to (num_pixels, num_slices).

        Returns:
            (callable) vcd_subset_updater(flat_recon, error_sinogram,
            pixel_indices) that updates the recon and error sinogram in place.
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

        # The qGGMRF chain is the updater's memory attention point: bind the
        # compiled forms once for all subsets.  The line-search and
        # state-application glue is compiled too (the ~20 eager launches
        # between projector calls were the interactive-VCD dispatch cost).
        # One compiled instance PER DEVICE THREAD: compiled artifacts carry
        # launcher state that must not be shared across concurrently
        # executing threads, and the process-wide lock serializes the
        # compile events (see maybe_compile).  Instances are cached per
        # (function, device index), so rebuilds re-use them.
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

        # qGGMRF boundary halos, staged once per PARTITION pass through the
        # ``stage_halos`` attribute (the mbirjax structure -- the partition
        # iterator calls it before the subset loop); the seeded n=2-vs-n=1
        # parity test bounds the staleness this admits within a pass.  On one
        # device there are no boundaries: both stay None (reflected edges).
        halos = {'left': [None] * num_devices, 'right': [None] * num_devices}
        # Per-device interface masks for a padded slice axis (None otherwise):
        # they reproduce the reflected boundary at the last REAL slice even
        # mid-shard, so the padded tail never enters the prior.
        interface_masks = self._qggmrf_interface_masks()

        def stage_halos(flat_shards):
            halos['left'], halos['right'] = _sharding.exchange_qggmrf_halos(
                flat_shards, self.dev2dev_safe)

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices):
            """
            Calculate an iteration of the VCD algorithm on a single subset of the
            partition.  Each application should return a better reconstruction.

            The combination (error_sinogram, recon) forms an overcomplete state
            that makes computation efficient, maintained under the invariant
                error_sinogram = measured_sinogram - forward_proj(recon).

            Args:
                flat_recon (Shards): per-device (num_pixels, local_slices)
                    slice shards of the flat recon; updated IN PLACE.
                error_sinogram (Shards): per-device (local_views, num_det_rows,
                    num_det_channels) view shards; updated IN PLACE.
                pixel_indices (int tensor): 1D array of pixel indices.

            Returns:
                flat_recon, error_sinogram, ell1_for_subset, alpha_for_subset,
                delta_sumsq_subset: the state (updated to reduce the overall
                loss), the L1 norm of this subset's recon change (0-d tensor),
                the relative step size (0-d tensor), and the per-slice sum of
                squared update values (on the lead device).
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
                # The weighted product (a sinogram-sized transient) is dead
                # here: the non-constant line-search terms fuse the weights
                # product into their reductions instead of re-reading it, so
                # free it before the delta projection below (mbirjax must
                # hold its product through the line search and delete after).
                weighted_error_sinogram = None

            # Per-shard update direction in the recon domain -- the per-subset
            # preconditioning seam (base: the diagonally-preconditioned
            # direction; geometry models may override _get_update_direction) --
            # with the prior line-search partials fused behind it:
            # delta^T \nabla Q(x_hat; x'=x_hat) and the prior quadratic bound.
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

            # Free the (now-dead) gradient/Hessian buffers BEFORE the
            # memory-heavy forward projection of the delta -- several
            # subset-sized buffers at the coarse partitions (mirrors mbirjax's
            # del at the same point; the per-worker locals died on worker
            # return).  No sync is needed: torch's stream-aware caching
            # allocator keeps a freed block from being reused until queued
            # reads of it complete (jax needed a block_until_ready).
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

            # Perform sparse updates at the index locations, IN PLACE, each
            # shard locally.  In jax this required buffer donation
            # (out-of-place per-subset updates leak via sharded-array
            # reference cycles); in torch a plain index_add_ / sub_ is the
            # whole mechanism.  The per-slice sum of squared updates is the
            # per-slice convergence diagnostic (delta_norm_per_slice in the
            # recon dict).
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
        subsets of the partition.  Each iteration should return a better
        reconstruction.  The error_sinogram should always satisfy
        error_sinogram = measured_sinogram - forward_proj(recon).

        Args:
            vcd_subset_updater (callable): function to apply to each subset.
            flat_recon (Shards): the per-device flat recon (one shard on a
                single device); updated in place across subsets.
            error_sinogram (Shards): the per-device error sinogram; updated
                in place across subsets.
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
        # Stage the qGGMRF boundary halos ONCE for this whole partition pass
        # and reuse them across its subsets (the mbirjax structure; the seeded
        # n=2-vs-n=1 parity test bounds the within-pass staleness).  On a
        # single device there are no shard boundaries and this is free.
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
        partition sequence (single device; mirrors mbirjax.vcd_recon minus
        sharding).

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
            compute_prior_loss (bool, optional): Set True to calculate and
                return the prior model loss (recorded at verbose >= 1 only,
                as in mbirjax; a debug/demo path for relatively small recons).
            first_iteration (int, optional): iteration offset for restarts (used
                only in the printed iteration labels here).
            init_error_sinogram (array or tensor, optional): Precomputed error
                sinogram to resume from, skipping the initializing forward
                projection.  Must be supplied together with init_recon, and the
                pair is TRUSTED as consistent (init_error_sinogram ==
                sinogram - A @ init_recon for the SAME sinogram and geometry) --
                verifying would cost the forward projection this argument
                exists to avoid.  The array returned via return_checkpoint
                satisfies this by construction.  NO defensive copy is made (at
                large sizes a clone would defeat the purpose of this path):
                both this array and init_recon become the loop's working
                buffers, updated IN PLACE where memory-compatible (a device
                tensor, or a CPU-model input of matching dtype), so after the
                call they -- and any checkpoint dict referencing them --
                reflect the RESUMED state.  This is the no-copy analog of
                mbirjax's buffer donation.  To keep or branch from the
                pre-resume state, copy BEFORE resuming; pairing a pre-resume
                recon copy with a post-resume error sinogram is an
                inconsistent pair and resumes silently wrong.
            fm_hessian (array or tensor, optional): Precomputed forward-model
                Hessian diagonal (as returned via return_checkpoint, or
                compute_hessian_diagonal(weights=weights) in either the 3-D or
                the flat (num_pixels, num_slices) form).  Must correspond to
                the SAME weights and geometry.  Read-only in the loop.  When
                None (default), it is computed internally.
            return_checkpoint (bool, optional): If True, additionally return
                the resume state -- a dict {'error_sinogram': <device tensor>,
                'fm_hessian': <device tensor>} suitable for the two arguments
                above -- so a chunked/checkpointed run continues with no
                re-initialization cost.  Zero-copy: the dict references the
                loop's own final device tensors.  A later resume that consumes
                these arrays updates them in place, so the dict then reflects
                the resumed state -- chaining resumes through the same dict
                stays consistent; copy first (e.g. .cpu().numpy()) to
                snapshot or persist.  Defaults to False.

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

        # Settle the device layout BEFORE the first large allocation.  On a
        # CUDA model whose layout the caller has not fixed, this is where the
        # reconstruction spreads across the devices that can hold their
        # share; everywhere else it returns without changing the layout.  The
        # ledger it returns is the model of the layout settled on, and it is
        # what the calibration mode compares against the measured peak.
        memory_ledger = self._apply_device_policy(
            partition_sequence=partition_sequence, weights=weights,
            init_recon=init_recon, fm_hessian=fm_hessian,
            prox_input=prox_input, init_error_sinogram=init_error_sinogram)
        if _memory_ledger.calibration_enabled():
            # The measured run begins here, so this is where the peak
            # counters reset -- one reset per reconstruction, owned by the
            # same function that reads the counters at the end.  A reset
            # inside the policy would also run on the nested direct_recon
            # call below and clear the peak mid-run.
            _memory_ledger.calibration_start(self.sino_placement.devices)
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

        # Place the sinogram only when it is needed -- to compute the error
        # sinogram.  On the RESUME path the error sinogram replaces its only
        # use, so no device copy is made (mbirjax likewise never places it
        # there, and frees its own copy after the fold below).
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
            # Resume fast path: trust the (init_recon, init_error_sinogram)
            # pair and skip the initializing forward projection.  NO defensive
            # copies (Greg, 2026-08-05): at large sizes cloning the state
            # arrays would defeat the purpose of the checkpoint path, so the
            # caller's arrays become the loop's working buffers and are
            # updated IN PLACE where memory-compatible -- the no-copy analog
            # of mbirjax's buffer donation.  Callers who need the pre-resume
            # state copy it before resuming; see the docstring.
            self.logger.info('Resuming from init_error_sinogram')
            error_sinogram = self._shard_sinogram(init_error_sinogram)
        else:
            error_sinogram, init_recon = self._initial_error_state(
                sinogram, init_recon, weights, constant_weights,
                scale_recon_to_sinogram)

        # The sinogram's contents are now fully folded into error_sinogram;
        # its remaining dtype read (the constant-weights ones array) is served
        # by error_sinogram.  Drop the reference so a device copy this
        # function made (a numpy or cross-device input) is freed before the
        # Hessian and the loop -- refcounting replaces mbirjax's explicit
        # own-and-delete bookkeeping, and a caller-owned tensor is unaffected.
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
            # Flatten first, then place: under sharding the flat
            # (num_pixels, slices) form is the slice-sharded device form (the
            # same order as the flat_recon placement below; mbirjax's
            # to_recon(prox_input.reshape(...))).
            prox_input = self._shard_recon(
                prox_input.reshape((-1, prox_input.shape[-1])))

        verbose, sigma_y = self.get_params(['verbose', 'sigma_y'])

        # The REAL sinogram element count, from the params (which always hold
        # the problem's shapes).  Equals the device arrays' size until a
        # sharding port pads the view/row axes; normalizing the reported
        # losses by the real count keeps them independent of inert padding.
        # math.prod (exact Python ints), NOT np.prod: numpy accumulates in the
        # platform default integer and a >2^31-element sinogram would silently
        # wrap.  num_real_elements stays None until padding can exist (the
        # sharding port gates it on its pad-active state, as mbirjax does).
        real_sino_size = math.prod(sinogram_shape)
        loss_num_real = None

        # Initialize the diagonal of the Hessian of the forward model: the back
        # projection of the weights with squared coefficients (constant weights
        # use an all-ones sinogram).  A precomputed fm_hessian (the checkpoint
        # fast path) skips the back projection; it is read-only in the loop.
        if fm_hessian is None:
            if constant_weights:
                # Ones over the real views and rows, ZEROS over any padded
                # entries (device form): padding must not contribute to the
                # Hessian back projection.  One seam for either layout (dtype
                # from error_sinogram, same device form).
                hess_weights = self._sino_ones_device_form(error_sinogram)
            else:
                hess_weights = weights
            self.logger.info('Computing Hessian diagonal')
            # Back-project only at the ROR-masked pixels.  The loop reads the
            # Hessian exclusively at partition indices, and those come from
            # the same mask, so every value it reads is bitwise unchanged
            # while the transient cylinder arrays shrink with the mask.  An
            # unmasked model has nothing to gain, so it keeps the dense path
            # and its single reshape.
            hess_indices = (None if self.get_params('use_ror_mask') is False
                            else self.full_indices_device())
            fm_hessian = self.compute_hessian_diagonal(weights=hess_weights,
                                                       output_sharded=True,
                                                       indices=hess_indices)
        else:
            self.logger.info('Using precomputed Hessian diagonal')
            fm_hessian = self._shard_recon(fm_hessian)
        fm_hessian = self._flatten_hessian(fm_hessian)

        # Flat recon layout, placed via _shard_recon (mbirjax: to_recon) --
        # under sharding the flat (num_pixels, slices) form is the
        # slice-sharded device form; on a single device the placement is a
        # no-op and contiguous() gives the VCD loop its packed row layout.
        flat_recon = self._flatten_recon(init_recon)

        # The loop's uniform per-device state: from here to the loop exit
        # there is ONE code path for any device count (a single device is the
        # trivial one-shard container, ALIASING its tensor, so the in-place
        # subset updates keep reaching the caller-visible arrays).
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
            # ONE per-device thread pool for the whole loop: every fan-out
            # (subset steps, band transfers, stats) reuses it instead of
            # creating and tearing down a pool per call.  A single device
            # never creates it (run_per_device short-circuits to a direct
            # call there).
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
                # An identically-zero recon gives recon_l1 == 0; mbirjax's jnp
                # division produces nan (the stop test is then False and the
                # loop continues) -- match that rather than raising
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
                        # Evaluate the prior loss on the REAL volume:
                        # _gather_recon crops any padded slices (whose zero
                        # values would otherwise add spurious
                        # boundary-difference terms).  Debug/verbose path only.
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
        # The run logger is set up here, as in mbirjax, so that it is set up
        # exactly when a run is initialized.  A Plug-and-Play loop calls
        # prox_map with do_initialization=False after the first pass, which
        # skips this method, and so keeps writing to the one log rather than
        # starting a new one on every pass.
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

        # no_grad, not inference_mode: the VCD loop needs autograd off (it never
        # differentiates), but torch.compile's guard machinery (torch 2.13)
        # crashes on compiled calls inside inference_mode with in-place state
        # updates, while no_grad composes cleanly.  The remaining
        # inference_mode benefit (skipping version-counter bookkeeping) is
        # noise next to the kernels.
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
        """Per-iteration logging stats -- (fm loss in the weight-normalized
        RMSE form, recon L1, error-sino RMSE).  A single-device state (a
        plain tensor, or the trivial one-shard container) delegates
        to the fused :meth:`_vcd_iteration_stats` kernel bit-identically; a
        genuinely per-device state sums each reduction per shard first and
        combines on the host (once per iteration -- this is the loop's one
        host sync point)."""
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
        # Internal bookkeeping params that are re-derived at construction: geometry_type is moved
        # into required_params (as the class identity); the rest are set by the geometry
        # constructors.
        construction_derived_names = ('geometry_type', 'view_params_name', 'file_format',
                                      'version', 'use_gpu')
        # Execution-environment constructor arguments; not model parameters.
        # 'device' was one until the constructors dropped it; the device
        # layout now travels through configure_devices alone and is never a
        # saved parameter.
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
