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
import io
import logging
import math
import warnings

import numpy as np
import torch

from . import _sharding
from . import qggmrf as _qggmrf
from . import tomography_utils, vcd_utils
from .memory_stats import get_memory_stats
from ._utils import _AUTO_REGULARIZATION_PARAM_NAMES, recon_param_names
from .parameter_handler import ParameterHandler
from .projectors import Projectors, maybe_compile

_F32_EPS = float(np.finfo(np.float32).eps)


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

    def __init__(self, sinogram_shape, device='auto', view_batch_size=64,
                 compile_mode='auto', **kwargs):
        super().__init__()
        self.torch_device = _resolve_device(device)
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
        # Device layout.  recon_placement / sino_placement are the single
        # source of truth for how the two array types are distributed (the
        # mbirjax structure): sino-like arrays shard by VIEW (axis 0),
        # recon-like arrays by SLICE (the last axis).  Construction gives the
        # trivial single-device placements; configure_devices() widens them.
        self.sino_placement = _sharding.Placement([self.torch_device], axis=0)
        self.recon_placement = _sharding.Placement([self.torch_device], axis=-1)
        self.dev2dev_safe = True     # probed for real in configure_devices
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
        self.create_projectors()
        self.verify_valid_params()

    @property
    def compile_enabled(self):
        return self.compile_mode != 'off'

    # ── hooks for geometry subclasses ─────────────────────────────────────────
    def create_projectors(self):
        self.projector_functions = Projectors(self)

    def get_magnification(self):
        raise NotImplementedError

    def get_psf_radius(self):
        raise NotImplementedError

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
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
        band length (single-fan geometries); a two-fan geometry overrides
        with its params-derived width.  Geometry-owned because the value is
        calibrated: changing it silently changes batch sizes, float
        summation order, and measured peaks."""
        return band_cols

    def direct_recon(self, sinogram, filter_name=None, output_sharded=False):
        raise NotImplementedError

    # ── projection wrappers (numpy at the public boundary) ────────────────────
    def sparse_forward_project(self, voxel_values, pixel_indices):
        """Cylinders at ``pixel_indices`` -> full sinogram (tensor, or view
        shards under a multi-device configuration)."""
        voxel_values = self._shard_recon(voxel_values)
        if isinstance(voxel_values, _sharding.Shards):
            return self._sparse_forward_project_sharded(voxel_values, pixel_indices)
        return self.projector_functions.sparse_forward_project(voxel_values, pixel_indices)

    def sparse_back_project(self, sinogram, pixel_indices, coeff_power=1):
        """Sinogram -> cylinders at ``pixel_indices`` (tensor, or slice shards
        under a multi-device configuration)."""
        sinogram = self._shard_sinogram(sinogram)
        if isinstance(sinogram, _sharding.Shards):
            return self._sparse_back_project_sharded(sinogram, pixel_indices,
                                                     coeff_power=coeff_power)
        return self.projector_functions.sparse_back_project(sinogram, pixel_indices,
                                                            coeff_power=coeff_power)

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
        the torch banded pass is orchestration-bound (eager fan-out per
        band), and the H100 gate matrix priced its sub-band default at +47
        to +66 percent warm-vcd time at the n=2 cells -- past the
        replacement-rule ceiling -- for peak savings of 0 to 61 percent.
        Time buys nothing back here because a single torch device never runs
        the banded drivers at all (the trivial fast path uses the plain
        projectors), so mbirjax's stream-even-at-n=1 rationale is void.

        A smaller B remains a real MEMORY lever (the per-band broadcast
        copy, the per-band partial, and each slice-owner's reduce gather all
        scale with B; measured n=4 @512: 6.6 to 2.6 GiB for +8 percent
        time).  Set ``forward_project_slice_band`` /
        ``back_project_slice_band`` on the model to opt in with a fixed B
        when a run is memory-constrained.  Every result is capped at
        slices_per_dev so a band never crosses a slice-owner boundary."""
        b = fixed_band if fixed_band else slices_per_dev
        return min(int(b), slices_per_dev)

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
        assembly, keeping the device form inert end to end."""
        if voxel_shards.placement.is_trivial:
            return _sharding.Shards(
                [self.projector_functions.sparse_forward_project(
                    voxel_shards.tensors[0], pixel_indices)],
                self.sino_placement)
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
                [self.projector_functions.sparse_back_project(
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
        mbirjax counterpart re-runs its device setup on every recompile)."""
        if not self.sino_placement.is_trivial:
            devices = self.sino_placement.devices
            sinogram_shape, recon_shape = self.get_params(
                ['sinogram_shape', 'recon_shape'])
            self.sino_placement = _sharding.Placement(
                devices, axis=0, real_size=int(sinogram_shape[0]))
            self.recon_placement = _sharding.Placement(
                devices, axis=-1, real_size=int(recon_shape[2]))
            self._check_no_empty_shard()
        self._invalidate_device_caches()
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
        """Set the device layout: rebuild the sino (view-axis) and recon
        (slice-axis) placements over ``num_devices`` CUDA devices, or an
        explicit device list.

        The placements' real sizes come from the CURRENT params
        (sinogram_shape / recon_shape), so call this after any geometry
        change -- and note the mbirjax stale-bind lesson: this RECREATES the
        projectors so nothing keeps a stale single-device binding.

        A single device (the default) restores the trivial placements and
        the unchanged n=1 path.
        """
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
        self.create_projectors()

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
        recon = self._shard_recon(recon)
        indices = self.full_indices_device()
        if isinstance(recon, _sharding.Shards):
            flat = _sharding.Shards(
                [t.reshape(-1, t.shape[-1])[indices.to(t.device)]
                 for t in recon.tensors], recon.placement)
            sinogram = self._sparse_forward_project_sharded(flat, indices)
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
        indices = torch.arange(recon_shape[0] * recon_shape[1], dtype=torch.int64,
                               device=self.torch_device)
        hessian = self.sparse_back_project(weights, indices, coeff_power=2)
        if isinstance(hessian, _sharding.Shards):
            hessian = _sharding.Shards(
                [t.reshape((recon_shape[0], recon_shape[1], t.shape[-1]))
                 for t in hessian.tensors], hessian.placement)
        else:
            hessian = hessian.reshape((recon_shape[0], recon_shape[1],
                                       hessian.shape[-1]))
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
            back_projected_error = self._sparse_back_project_sharded(weighted_error_sinogram, pixel_indices)
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
            delta_sinogram = self._sparse_forward_project_sharded(
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
                delta_sinogram = self._sparse_forward_project_sharded(
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
            fm_hessian = self.compute_hessian_diagonal(weights=hess_weights,
                                                       output_sharded=True)
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
              stop_threshold_change_pct=0.2, first_iteration=0,
              output_sharded=False):
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
            output_sharded (bool, optional): If False (default), return a numpy
                array; if True, return the device tensor (the mbirjax argument
                name, kept for API compatibility).

        Returns:
            (recon, recon_dict): the reconstruction volume, and a dict
            with entries 'recon_params' (per-iteration traces and settings) and
            'model_params' (a snapshot of the model parameters).
        """
        (sinogram, weights, init_recon, partitions, partition_sequence, granularity,
         regularization_params) = self.initialize_recon(
            sinogram, weights, init_recon, max_iterations, first_iteration)

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
        recon_dict = {'recon_params': recon_params,
                      'model_params': {k: v.val for k, v in self.params.items()}}
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
                 first_iteration=0, output_sharded=False):
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

        Returns:
            (recon, recon_dict): the numpy reconstruction volume and the recon
            parameters dict.
        """
        prior_loss = [0]
        if do_initialization or self.prox_data is None:
            (sinogram, weights, init_recon, partitions, partition_sequence,
             granularity, regularization_params) = self.initialize_recon(
                sinogram, weights, init_recon, max_iterations, first_iteration)
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
        recon_dict = {'recon_params': recon_params,
                      'model_params': {k: v.val for k, v in self.params.items()}}
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
