"""The geometry-agnostic projection driver and the torch.compile plumbing.

The division of labor (panel-reviewed design, plans repo
projector_layer_design.md): this module owns ITERATION and MEMORY -- the
view-batch loops, the transient budget, per-device compiled-instance
management, and the compile lock.  The shared horizontal-fan math and the
hfan data contract live in horizontal_fan.py; each geometry model owns its
geometry chains and its per-view-batch bodies.  No sorted channel
reduction, no stacked gather, no tile policy here (the jax perf layer;
their torch analogs belong with the future Triton kernel work).

The drivers batch over VIEWS with a plain python loop.  For the torch
bodies the eager transient is (view_batch, num_pixels, cols) floats and
``view_batch_size`` is the single memory/speed knob; a hand-written kernel
body declares its own, much smaller, per-view cost through its
``_view_batch_cost`` attribute (see _effective_view_batch).  The torch
bodies are torch.compiled (see maybe_compile below).
"""

import os
import threading

import torch

from . import _sharding

_F32 = torch.float32

# ── torch.compile plumbing ────────────────────────────────────────────────────
# A compile failure falls back to eager and is recorded in _COMPILE_ERRORS.
_COMPILE_CACHE = {}
_COMPILE_ERRORS = {}
# Triton and inductor compilation is not thread safe, so this lock serializes compiles.
# Two threads compiling at once crash in static_triton_launcher on an A100 with torch 2.13.
_GLOBAL_COMPILE_LOCK = threading.Lock()

#: The per-function recompile budget this module raises before it compiles anything.  Torch's
#: cap sits on the code object, so every per-device instance of one body shares one budget, and
#: a run on n devices needs about n times the one-device variant count.  When the budget fills,
#: the remaining calls run eagerly, which measured 5 to 11 times slower on two H100s.
_RECOMPILE_LIMIT_FLOOR = 64


def _raise_recompile_budget():
    """Raise torch's per-function recompile budget to at least
    ``_RECOMPILE_LIMIT_FLOOR`` on the calling thread.

    Dynamo consults a per-thread view of this config, so the budget must be
    raised on every thread that can trigger a compilation.  The per-device
    fan-outs run the compiled bodies on pool threads, so each of those
    threads raises the budget itself.

    ``MBIRTORCH_RECOMPILE_LIMIT`` overrides the floor exactly, including
    downward, which is useful for debugging.  Both config names are set
    together, because ``recompile_limit`` is the current name and
    ``cache_size_limit`` is its older spelling.
    """
    import torch._dynamo.config as dynamo_config

    override = os.environ.get('MBIRTORCH_RECOMPILE_LIMIT')
    if override:
        limit = int(override)
    else:
        limit = max(int(dynamo_config.recompile_limit),
                    _RECOMPILE_LIMIT_FLOOR)
    dynamo_config.recompile_limit = limit
    dynamo_config.cache_size_limit = limit


def _shape_key(args, kwargs):
    parts = []
    for a in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if torch.is_tensor(a):
            parts.append(('T', str(a.device)) + tuple(a.shape))
        elif isinstance(a, tuple):
            for t in a:
                parts.append(((('T', str(t.device)) + tuple(t.shape))
                              if torch.is_tensor(t) else ('o', str(t)[:32])))
        else:
            parts.append(('o', str(a)[:32]))
    return tuple(parts)


def maybe_compile(fn, enabled, instance_key=None):
    """Return a compiled form of ``fn`` (cached per (function, instance_key))
    when enabled, else ``fn`` itself.

    A function carrying ``_mbirtorch_no_compile`` is returned as it is,
    whatever ``enabled`` says: that marker is how a HAND-WRITTEN kernel body
    (triton_cone.py) declares that it launches its own kernel and must stay
    eager.  The marker is needed because ``torch.compiler.disable`` alone does
    not survive an explicit compile -- ``torch.compile`` unwraps the disable
    decorator (``innermost_fn``) and traces the original function, launch and
    all -- so the opt-out has to be honored at the call site that compiles.

    ``instance_key`` names a distinct
    compiled instance -- the per-device threads of the VCD loop pass their
    device index, because compiled artifacts carry triton-launcher state
    that must not be shared across concurrently executing threads (a shared
    instance crashed under concurrent cold compiles; the process-wide lock
    below serializes the COMPILE events, the per-instance split isolates the
    launcher state).  Instances are cached at module level, so rebuilding a
    model's projectors reuses them instead of re-tracing devices 1..n-1.

    torch.compile is LAZY: the wrapper it returns compiles at the first
    invocation, so a broken backend (no C++ toolchain, a broken triton) would
    surface there, not at torch.compile() time.  The returned
    callable therefore guards the FIRST call: on any exception it retries the
    call EAGERLY -- the kernels here are pure, so the retry is safe -- and, if
    eager succeeds, records the compile error in ``_COMPILE_ERRORS`` and
    permanently rebinds to eager (the compile failure was environmental).  If
    eager also raises, that error is the real one and propagates.  After one
    successful compiled call the guard collapses to a direct dispatch.
    (A LATER per-shape recompile could still fail on a broken toolchain; in
    practice the first call exercises the backend end to end.)
    """
    if not enabled or getattr(fn, '_mbirtorch_no_compile', False):
        return fn
    cache_key = fn if instance_key is None else (fn, instance_key)
    if cache_key in _COMPILE_CACHE:
        return _COMPILE_CACHE[cache_key]
    _raise_recompile_budget()
    compiled = torch.compile(fn)
    state = {"impl": compiled}
    seen_keys = set()

    def guarded(*args, **kwargs):
        key = _shape_key(args, kwargs)
        if key in seen_keys:
            return state["impl"](*args, **kwargs)
        # This is the first call at this shape, so it may compile.  The lock keeps it
        # from compiling at the same time as another thread.
        with _GLOBAL_COMPILE_LOCK:
            _raise_recompile_budget()
            try:
                out = state["impl"](*args, **kwargs)
            except Exception as e:                            # noqa: BLE001
                # A compile backend failure succeeds on this eager retry and
                # falls back for good.  A real input error raises again.
                out = fn(*args, **kwargs)
                _COMPILE_ERRORS[f"{fn.__module__}.{fn.__name__}"] = \
                    f"{type(e).__name__}: {e}"[:400]
                state["impl"] = fn
            seen_keys.add(key)
            return out

    guarded.__name__ = f"compiled_{fn.__name__}"
    _COMPILE_CACHE[cache_key] = guarded
    return guarded


# The wrappers below pad a call narrower than ``min_compiled_pixel_width`` and undo the padding.
# Inductor miscompiles the one-pixel specialization of both fused parallel-beam bodies, with a
# relative error of about 5e-02.  Eager is correct at one pixel, and every width of two or more
# is correct compiled.  The cone bodies do not have the defect.


def _callable_name(fn, fallback):
    """Return a readable name for a wrapped callable."""
    return getattr(fn, '__name__', fallback)


def forward_at_min_pixel_width(compiled, min_width):
    """The forward body with narrow pixel batches padded to ``min_width``.

    The padded pixels carry zero values at a repeated -- hence in-range --
    pixel index.  The forward output has no pixel axis: the fan bins each
    pixel's weighted row into the detector channels with index_add_, so a
    zero-valued pixel adds exactly 0.0 wherever it lands and the padded call
    returns bit-identical values with nothing to slice off (verified against
    the eager body).
    """
    def forward_padded(values, pixel_indices, *args, **kwargs):
        width = int(pixel_indices.shape[0])
        if width == 0 or width >= min_width:
            return compiled(values, pixel_indices, *args, **kwargs)
        pad = min_width - width
        wide_values = torch.cat(
            [values, values.new_zeros((pad,) + tuple(values.shape[1:]))])
        wide_indices = torch.cat([pixel_indices,
                                  pixel_indices[-1:].repeat(pad)])
        return compiled(wide_values, wide_indices, *args, **kwargs)

    forward_padded.__name__ = f'padded_{_callable_name(compiled, "forward")}'
    return forward_padded


def back_at_min_pixel_width(compiled, min_width):
    """The back body with narrow pixel batches padded to ``min_width``.

    The back output DOES carry the pixel axis, so here the padding repeats the
    last real pixel index and the extra rows are sliced off again.  Every
    output row is computed from its own pixel alone (the fan gathers per pixel
    and sums over views), so the rows that stay are the rows the narrow call
    would have produced -- exactly, not to a tolerance (verified against the
    eager body, at coeff_power 1 and 2).
    """
    def back_padded(sino_batch, pixel_indices, *args, **kwargs):
        width = int(pixel_indices.shape[0])
        if width == 0 or width >= min_width:
            return compiled(sino_batch, pixel_indices, *args, **kwargs)
        pad = min_width - width
        wide_indices = torch.cat([pixel_indices,
                                  pixel_indices[-1:].repeat(pad)])
        block = compiled(sino_batch, wide_indices, *args, **kwargs)
        # Cloned rather than returned as a view, so the caller's output owns
        # its memory and does not keep the padded block alive.
        return block[:width].clone()

    back_padded.__name__ = f'padded_{_callable_name(compiled, "back")}'
    return back_padded


def compile_serialized():
    """The process-wide compile lock, as a context manager -- for HAND-WRITTEN
    kernel paths only::

        with compile_serialized():
            my_triton_kernel[grid](...)     # first launch: jit / autotune

    torch.compile paths need this nowhere: ``maybe_compile``'s wrapper already
    takes the same lock around every call that can trigger a compile.  A
    triton.jit or triton.autotune path compiles OUTSIDE torch.compile, at its
    first launch (and again per autotune configuration), and races the same
    launcher/compiler state the lock exists for, so it must borrow the lock
    rather than introduce a second one.

    Wrap only the compiling launches: as a decorator on the launching function
    this would take the lock on EVERY call and serialize execution, not just
    compilation.
    """
    return _GLOBAL_COMPILE_LOCK




class Projectors:
    """The batched sparse projection driver for one model: geometry-agnostic
    iteration and memory.

    The geometry enters ONLY through the model's per-view-batch bodies
    (``_view_batch_bodies`` / ``_view_batch_args``): the driver slices view
    parameters, applies the transient budget, calls the compiled body, and
    assembles outputs sized lazily from the first block.  One geometry class
    therefore never subclasses this driver.

    Center-consistency contract: forward and back consume the SAME
    deterministic center computation for each (view, pixel), so the pair stays
    exactly adjoint even at rounding ties.  The centers are recomputed by that
    same chain on every call rather than cached, which preserves the property.
    """

    # Cap on the per-batch transient bytes for the fan kernels' (Vb, P, cols) arrays, since the
    # back fan's gather is materialized.  The accounting charges one nominal slab per view while
    # the kernels hold two to five at once, so any retuning must measure the real per-view
    # transient.  Past about 1400 cubed one view's slab alone exceeds the cap.
    VIEW_BATCH_TRANSIENT_BUDGET_BYTES = 2 * 2**30
    VIEW_BATCH_TRANSIENT_FLOOR_BYTES = 256 * 2**20
    VIEW_BATCH_SINO_MULTIPLE = 8
    # The torch bodies' nominal view batch when model.view_batch_size is None.
    # A kernel body's nominal is its own swept view chunk instead.
    VIEW_BATCH_BODY_DEFAULT = 64

    def _transient_budget_bytes(self, n_devices=None):
        if self.model.torch_device.type == 'cpu':
            return self.VIEW_BATCH_TRANSIENT_BUDGET_BYTES
        num_views, num_rows, num_channels = self.model.get_params('sinogram_shape')
        # Under view sharding each device projects only its share of the views, so the budget
        # scales with the per-device shard.  ``n_devices`` prices a candidate layout.
        n_dev = (self.model.sino_placement.n_devices if n_devices is None
                 else int(n_devices))
        local_views = -(-int(num_views) // n_dev)
        sino_bytes = local_views * num_rows * num_channels * 4
        return max(self.VIEW_BATCH_TRANSIENT_FLOOR_BYTES,
                   min(self.VIEW_BATCH_TRANSIENT_BUDGET_BYTES,
                       self.VIEW_BATCH_SINO_MULTIPLE * sino_bytes))

    def __init__(self, model):
        # The geometry supplies its per-view-batch bodies as module-level functions.  A bound
        # method would pin the model in the module-level compile cache.
        self.model = model
        fwd_body, back_body = model._view_batch_bodies()
        use_compile = model.compile_enabled
        n_dev = model.sino_placement.n_devices
        min_width = int(getattr(model, 'min_compiled_pixel_width', 1))

        def bind(body, pad_narrow, i):
            """Return one device's bound body.  It is compiled, then wrapped
            when the model declares a minimum pixel width and the binding
            really did compile.

            maybe_compile hands back the function itself when compilation is
            off and when the body is a hand-written kernel.  Neither can be
            miscompiled, so neither is wrapped."""
            bound = maybe_compile(body, use_compile, instance_key=i)
            if min_width > 1 and bound is not body:
                bound = pad_narrow(bound, min_width)
            return bound

        self._fwd_body_per_dev = [
            bind(fwd_body, forward_at_min_pixel_width, i)
            for i in range(n_dev)]
        self._back_body_per_dev = [
            bind(back_body, back_at_min_pixel_width, i)
            for i in range(n_dev)]
        # The view parameters are read from the current params at every projector build.
        # Each device gets its copy through the probed transfer primitive.
        view_params_name = model.get_params('view_params_name')
        view_params = torch.as_tensor(model.get_params(view_params_name),
                                      dtype=_F32, device=model.torch_device)
        self._view_params_per_dev = [
            _sharding.move_shard(view_params, dev, model.dev2dev_safe)
            for dev in model.sino_placement.devices]
        self.view_params_array = self._view_params_per_dev[0]

    def _effective_view_batch(self, body, num_pixels, band_cols, args):
        """Return the view batch for one call of ``body``.  It is the nominal
        batch, capped so that one batch's transient stays within the budget.

        The batching rule follows the body that is bound, not the geometry.
        A hand-written kernel body carries a ``_view_batch_cost`` attribute
        giving its own per-view bytes and its swept view chunk.  The forward
        and back bodies are consulted separately, so a model that binds a
        kernel one way and a torch body the other batches each direction by
        its own rule."""
        return self.view_batch_charge(body, num_pixels, band_cols, args)[0]

    def view_batch_charge(self, body, num_pixels, band_cols, args,
                          n_devices=None):
        """``(view_batch, bytes_per_view)`` for one call of ``body``: the
        single per-view cost model, with TWO consumers.

        The driver consumes the batch (through :meth:`_effective_view_batch`)
        to decide how many views one body call takes.  The memory ledger
        (``_memory_ledger``) consumes both numbers, because the batch alone
        does not say how many bytes that batch holds.  Keeping one function
        means the batch chooser and the ledger cannot drift apart.

        The charge EXCLUDES the call-fixed tensors -- the assembled output
        and the accumulated partial -- exactly as each body's own
        ``_view_batch_cost`` docstring states.  Those exist at any batch
        size, so the batch choice cannot control them.  The ledger adds them
        itself, per phase.

        Args:
            body: the projection body actually bound (kernel or torch).
            num_pixels (int): P, the pixel subset this call projects.
            band_cols (int): the call's column count -- the forward's slice
                extent, or the back's local sinogram row count.
            args (dict): the geometry's per-call argument dict.
            n_devices (int, optional): price the budget for a HYPOTHETICAL
                device count instead of the model's current placement.  Used
                only by the ledger when it evaluates a candidate layout;
                None (the default) is every production call site.
        """
        nominal = self.model.view_batch_size
        cost = getattr(body, '_view_batch_cost', None)
        if cost is None:
            cols = self.model._transient_cols(band_cols)
            bytes_per_view = num_pixels * cols * 4
            if nominal is None:
                nominal = self.VIEW_BATCH_BODY_DEFAULT
        else:
            bytes_per_view, view_chunk = cost(num_pixels, band_cols, args)
            if nominal is None:
                nominal = view_chunk
        budget = self._transient_budget_bytes(n_devices=n_devices)
        cap = budget // max(1, int(bytes_per_view))
        return max(1, min(int(nominal), int(cap))), int(bytes_per_view)

    def sparse_forward_project_view_range(self, band_values, pixel_indices,
                                          view_range, slice_start=0,
                                          dev_index=0, plan=None,
                                          accumulate_into=None):
        """Forward-project voxel values into ONE view-owner's sinogram block:
        the single forward loop -- the single-device full-range form is the
        adapter below over (0, num_views).  The geometry body owns all geometry,
        layout, and output orientation; this loop owns view slicing, the
        transient budget, and assembly (output sized lazily from the first
        block, so the driver never derives geometry-specific shapes).

        ``accumulate_into`` lets a caller that runs this loop repeatedly --
        the cylinder-transfer forward, once per pixel batch -- add straight
        into the block it is building instead of receiving a fresh one to add
        itself.  That merges two full-block passes into one and drops one
        full-block allocation per call; see the accumulation comment in
        ``TomographyModel._sparse_forward_project_cylinders`` for why it is
        worth doing and why the values do not move.  The parameter is added
        HERE, on a plain python method, and not to the geometry body: the
        bodies are torch.compile'd per device with shape-keyed caches, so a
        new argument there would recompile every one of them.

        Args:
            band_values: (P, cols) voxel cylinders (or a slice band), on this
                owner's device.
            pixel_indices: (P,) int64 on the same device.
            view_range: (v0, v1) half-open GLOBAL view range this owner owns
                (the banded drivers' contract: one contiguous real-view span
                per owner).
            slice_start (int): global slice anchor of a slice BAND (two-fan
                geometries); a row-aligned geometry's body asserts 0.
            dev_index (int): which per-device compiled instance to use.
            plan: the memoization slot for a future sorted/CSR stream variant
                (per pixel-subset x view-range); unused today.
            accumulate_into: an existing block of the shape this call returns.
                Given one, the loop ADDS into it and returns it; given None
                (every other caller), it allocates the block and writes.

        Returns:
            (v1 - v0, rows_or_band, num_channels) on the input's device.
        """
        m = self.model
        v0, v1 = view_range
        args = m._view_batch_args()
        fwd_body = self._fwd_body_per_dev[dev_index]
        vb_size = self._effective_view_batch(fwd_body, pixel_indices.shape[0],
                                             band_values.shape[-1], args)
        view_params = self._view_params_per_dev[dev_index]
        out = accumulate_into
        # An accumulating call adds every batch, including the first, because
        # the block it was handed already holds earlier calls' work.
        adding = out is not None
        for v in range(v0, v1, vb_size):
            view_params_batch = view_params[v:min(v + vb_size, v1)]
            block = fwd_body(
                band_values, pixel_indices, view_params_batch,
                slice_start=slice_start, plan=plan, **args)
            if out is None:
                out = torch.empty((v1 - v0,) + tuple(block.shape[1:]),
                                  dtype=block.dtype, device=block.device)
            rows = slice(v - v0, v - v0 + block.shape[0])
            # View batches cover disjoint rows of the block, so each row is
            # touched exactly once whichever branch runs.
            if adding:
                out[rows].add_(block)
            else:
                out[rows] = block
        return out

    def sparse_back_project_view_range(self, local_sino, pixel_indices,
                                       view_range, coeff_power=1,
                                       slice_start=0, band_slices=None,
                                       dev_index=0, plan=None):
        """Back-project ONE view-owner's local sinogram onto voxel cylinders
        (the adjoint of :meth:`sparse_forward_project_view_range`): the
        single back loop, accumulating lazily from the first block so the
        output shape comes from the geometry body, not the driver.

        Args:
            local_sino: this owner's views -- (v1 - v0, rows, channels) (a
                row band for a row-aligned geometry; the full local block for
                a two-fan geometry).
            pixel_indices: (P,) int64 on the same device.
            view_range: (v0, v1) half-open GLOBAL view range.
            coeff_power (int): 1, or 2 for the Hessian diagonal.
            slice_start (int) / band_slices (int or None): the slice-band
                request for two-fan geometries; a row-aligned geometry's body
                asserts the defaults.
            dev_index (int): which per-device compiled instance to use.
            plan: the memoization slot for a future sorted/CSR stream variant
                (per pixel-subset x view-range); unused today.

        Returns:
            (P, slices_or_band) on the input's device.
        """
        m = self.model
        v0, v1 = view_range
        args = m._view_batch_args()
        back_body = self._back_body_per_dev[dev_index]
        vb_size = self._effective_view_batch(back_body, pixel_indices.shape[0],
                                             local_sino.shape[1], args)
        view_params = self._view_params_per_dev[dev_index]
        out = None
        for v in range(v0, v1, vb_size):
            view_params_batch = view_params[v:min(v + vb_size, v1)]
            block = back_body(
                local_sino[v - v0:v - v0 + view_params_batch.shape[0]],
                pixel_indices, view_params_batch, coeff_power=coeff_power,
                slice_start=slice_start, band_slices=band_slices, plan=plan,
                **args)
            if out is None:
                out = block
            else:
                out.add_(block)
            # Releasing the block here keeps the loop to two cylinder shards instead of
            # three.  The summation order is unchanged.
            block = None
        return out

    def _sparse_forward_project_single_device(self, voxel_values, pixel_indices):
        """Forward project voxel cylinders into a full sinogram on one device.

        This runs the view-range loop at (0, num_views) on device 0.  Every
        external call goes through TomographyModel.sparse_forward_project.
        """
        m = self.model
        num_views = int(m.get_params('sinogram_shape')[0])
        voxel_values = torch.as_tensor(voxel_values, dtype=_F32,
                                       device=m.torch_device)
        pixel_indices = torch.as_tensor(pixel_indices, dtype=torch.int64,
                                        device=m.torch_device)
        return self.sparse_forward_project_view_range(
            voxel_values, pixel_indices, (0, num_views), dev_index=0)

    def _sparse_back_project_single_device(self, sinogram, pixel_indices,
                                           coeff_power=1):
        """Back project a full sinogram onto the voxel cylinders at
        ``pixel_indices`` on one device.

        This runs the view-range loop at (0, num_views) on device 0.  Every
        external call goes through TomographyModel.sparse_back_project.

        Args:
            sinogram: (num_views, num_det_rows, num_det_channels).
            pixel_indices: (P,) indices into the flattened (rows, cols) grid.
            coeff_power (int): backproject (A_ij ** coeff_power); 2 for the
                Hessian diagonal.

        Returns:
            (P, num_slices) tensor of per-pixel cylinders.
        """
        m = self.model
        num_views = int(m.get_params('sinogram_shape')[0])
        sinogram = torch.as_tensor(sinogram, dtype=_F32, device=m.torch_device)
        pixel_indices = torch.as_tensor(pixel_indices, dtype=torch.int64,
                                        device=m.torch_device)
        return self.sparse_back_project_view_range(
            sinogram, pixel_indices, (0, num_views), coeff_power=coeff_power,
            dev_index=0)
