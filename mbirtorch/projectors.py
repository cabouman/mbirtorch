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

import threading

import torch

from . import _sharding

_F32 = torch.float32

# ── torch.compile plumbing ────────────────────────────────────────────────────
# Measured chain-level compile wins: 1.7-3.6x (CPU), 5-17x (MPS), and 2.6-22x
# (CUDA), with the fan chain's peak-memory transients collapsing 6-41x.  The
# compiled callables are cached per FUNCTION at
# module level: torch.compile handles multiple input shapes itself (one
# specialization per shape guard), and the VCD loop's shape set is small (one
# subset size per partition granularity, plus the full-index size).  A compile
# failure falls back to eager silently-but-recorded, so exotic
# backends/toolchains keep working.
_COMPILE_CACHE = {}
_COMPILE_ERRORS = {}
# Serializes COMPILE EVENTS process-wide: triton/inductor compilation is not
# thread-safe (measured on A100, torch 2.13: two per-device threads cold-compiling
# concurrently crash in static_triton_launcher EVEN WITH separate compiled
# instances).  Each wrapper takes this lock only for input-shape keys it has
# not completed before, so steady-state threaded execution stays lock-free.
_GLOBAL_COMPILE_LOCK = threading.Lock()


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
    compiled = torch.compile(fn)
    state = {"impl": compiled}
    seen_keys = set()

    def guarded(*args, **kwargs):
        key = _shape_key(args, kwargs)
        if key in seen_keys:
            return state["impl"](*args, **kwargs)
        # First sight of this shape: the call may trigger dynamo/inductor
        # compilation, which must not run concurrently with any other
        # compile in the process (see _GLOBAL_COMPILE_LOCK).
        with _GLOBAL_COMPILE_LOCK:
            try:
                out = state["impl"](*args, **kwargs)
            except Exception as e:                            # noqa: BLE001
                # Retry eagerly: if the failure was the compile backend, this
                # succeeds and we fall back for good; a real input error
                # re-raises.
                out = fn(*args, **kwargs)
                _COMPILE_ERRORS[f"{fn.__module__}.{fn.__name__}"] = \
                    f"{type(e).__name__}: {e}"[:400]
                state["impl"] = fn
            seen_keys.add(key)
            return out

    guarded.__name__ = f"compiled_{fn.__name__}"
    _COMPILE_CACHE[cache_key] = guarded
    return guarded


# ── the minimum pixel width a compiled body is called at ─────────────────────
# A model may declare that its compiled bodies must not be called with fewer
# than N pixels (``min_compiled_pixel_width``, see TomographyModel).  The two
# wrappers below pad a narrower call up to N and undo the padding on the way
# out.  They are applied OUTSIDE torch.compile, around the callable
# maybe_compile returns, so the padding is ordinary python that dynamo never
# traces -- padding inside the body would be traced and specialized with it,
# which is exactly what has to be avoided.
#
# Why they exist (measured 2026-08-11, linux CPU, torch 2.13.0): inductor
# miscompiles the one-pixel specialization of both fused parallel-beam bodies.
# A one-pixel call puts that pixel's horizontal-fan footprint one whole
# detector channel away from where the same pixel lands in a call with more
# pixels -- 6.56e-02 relative error on the forward (the pixel's mass at
# channels {4, 5} instead of {3, 4}) and 5.04e-02 on the back, on a seeded
# 8x6x8 test cell.  Eager is correct at one pixel (1.05e-07), every width of
# two or more is correct compiled, and a one-pixel call is correct once the
# process has compiled the body at a larger width, so what is wrong is the
# one-pixel compile itself.  The cone bodies do not have the defect, and macOS
# inductor compiles the same one-pixel body correctly.  One-pixel calls are
# ordinary: sparse_forward_project with a single index makes one, and the
# column gather's pixel batching makes one whenever a batch, or the remainder
# of a batch, is a single pixel.
#
# The driver's view-batch charge is computed from the REAL pixel count, before
# the padding: it prices the transient of a call this small at a batch far
# below any cap, so the one padded column cannot move it.


def _callable_name(fn, fallback):
    """A readable name for a wrapped callable, for the wrapper's own name."""
    return getattr(fn, '__name__', fallback)


def forward_at_min_pixel_width(compiled, min_width):
    """The forward body with narrow pixel batches padded to ``min_width``.

    The padded columns carry zero values at a repeated -- hence in-range --
    pixel index.  The forward output has no pixel axis: the fan bins each
    pixel's weighted row into the detector channels with index_add_, so a
    zero-valued column adds exactly 0.0 wherever it lands and the padded call
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

    # Rough per-batch transient budget for the fan kernels' (Vb, P, cols)
    # arrays.  The back fan's gather output is a REAL materialized tensor even
    # under torch.compile (a gather cannot fuse away), so an unbounded view
    # batch at large cells allocates tens of GB (the 512-cell at the default
    # batch of 64 wants ~13 GB).  The batch size never changes values beyond
    # float summation order, so capping it is a pure memory knob.
    #
    # The BUDGET below applies to every body; WHAT one view is charged is the
    # body's own model (see _effective_view_batch): the torch bodies' gather
    # slab here, or a hand-written kernel body's _view_batch_cost.
    #
    # On the DEVICE backends the budget also scales DOWN with the problem: a
    # flat 2 GiB let a 200-class cell hold a gather transient ~12x jax's whole
    # peak (the CUDA gate readout's back/vcd memory breaches).  Scaling by the
    # sinogram size (8x, floored at 256 MiB for batch efficiency) keeps small
    # cells lean while leaving the large cells -- where torch already beat jax
    # on memory -- at the 2 GiB cap.  CPU keeps the flat cap: host RSS was
    # already 0.4-0.6x of jax's, and the small batches the scaled budget
    # implies were measured slower there (the measured CPU optimum is a large
    # batch).
    # TODO(tuning): known limits of the current form (measured 2026-08-05,
    # MPS 256^3), to be resolved with the fused-kernel work:
    #   - The accounting is per-SLAB nominal, and the kernels hold several
    #     slab-scale tensors at once (forward: product + accumulator; back:
    #     transpose copy + gather), so the actual per-view transient is a small
    #     multiple (~2-5x) of the nominal slab; the 8x sinogram multiple was
    #     calibrated empirically with that multiplier baked in.  Recalibrate
    #     against a measured per-view delta, not the nominal slab, if retuned.
    #   - Below the 2 GiB cap the formula gives vb ~ 10 for ROR-masked cubes at
    #     ANY size (8*sino / slab ~ 10.2 by construction); raising vb past that
    #     bought only ~5-7% speed for ~6x more device memory, so the small vb
    #     is deliberate, not accidental.
    #   - Above ~810^3 the cap forces vb=1 and the SINGLE-view slab keeps
    #     growing as N^3 (3.2 GB at 1024^3, 26 GB at 2048^3): past ~1400^3 the
    #     knob no longer protects at all -- needs pixel-axis chunking or the
    #     planned fused (Triton) kernels that never materialize the gather.
    #     Detector growth makes this NEAR-TERM, not hypothetical: panels are
    #     heading to ~6K x 10K (2026 estimate), where ONE view's slab against
    #     a 512-class pixel set is ~6 GB -- the view axis alone cannot bound
    #     it.  The driver loop is shaped as a two-axis tile walk with an
    #     accumulating forward precisely so the pixel loop drops in without
    #     touching the geometry contract.
    #     Pixel chunking here means TWO-axis tiling: forward sums partial
    #     sinograms over PIXEL batches around the view loop, back concatenates
    #     per-PIXEL-batch outputs inside the view-sum loop, with a jointly
    #     chosen (view_batch, pixel_batch) tile.  These drivers tile views
    #     only; the joint tile choice needs a 2-D budget rule and its own
    #     measurements (tile shape has changed kernel run time several-fold in
    #     earlier work), so it belongs with the kernel work.
    VIEW_BATCH_TRANSIENT_BUDGET_BYTES = 2 * 2**30
    VIEW_BATCH_TRANSIENT_FLOOR_BYTES = 256 * 2**20
    VIEW_BATCH_SINO_MULTIPLE = 8
    # The torch bodies' nominal view batch when the model's view_batch_size
    # is None (automatic) -- the value the constructor default has always
    # been.  A kernel body's automatic nominal is its own swept view chunk,
    # returned by its _view_batch_cost (see _effective_view_batch).
    VIEW_BATCH_BODY_DEFAULT = 64

    def _transient_budget_bytes(self, n_devices=None):
        if self.model.torch_device.type == 'cpu':
            return self.VIEW_BATCH_TRANSIENT_BUDGET_BYTES
        num_views, num_rows, num_channels = self.model.get_params('sinogram_shape')
        # Under view sharding each device projects only its share of the
        # views, so the size-scaled budget derives from the PER-DEVICE shard
        # (the global sinogram would overshoot each device's transient by the
        # device count).  Derived per call from the current params and
        # placement -- never frozen at construction (the stale-bind lesson).
        # ``n_devices`` overrides the live placement for a HYPOTHETICAL layout
        # (the memory ledger pricing a candidate device count); None means the
        # model's current placement, which is every production call site.
        n_dev = (self.model.sino_placement.n_devices if n_devices is None
                 else int(n_devices))
        local_views = -(-int(num_views) // n_dev)
        sino_bytes = local_views * num_rows * num_channels * 4
        return max(self.VIEW_BATCH_TRANSIENT_FLOOR_BYTES,
                   min(self.VIEW_BATCH_TRANSIENT_BUDGET_BYTES,
                       self.VIEW_BATCH_SINO_MULTIPLE * sino_bytes))

    def __init__(self, model):
        # The geometry supplies its per-view-batch bodies (module-level pure
        # functions -- never bound methods, which would pin the model in the
        # module-level compile cache) and the driver binds one compiled
        # instance per device (see maybe_compile).  Index 0 serves the
        # single-device path.
        self.model = model
        fwd_body, back_body = model._view_batch_bodies()
        use_compile = model.compile_enabled
        n_dev = model.sino_placement.n_devices
        min_width = int(getattr(model, 'min_compiled_pixel_width', 1))

        def bind(body, pad_narrow, i):
            """One device's bound body: compiled, then wrapped when the model
            declares a minimum pixel width AND the binding really did compile.

            The identity test is the whole gate.  maybe_compile hands back the
            function itself when compilation is off and when the body is a
            hand-written kernel (``_mbirtorch_no_compile``); neither can be
            miscompiled, so neither needs the workaround, and leaving them
            alone keeps the two things callers read off a bound body -- its
            identity and its ``_view_batch_cost`` attribute -- exactly as they
            were.  Every driver, plain and sharded, reads its body from these
            two lists, so this is the one place per direction to wrap."""
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
        # View parameters, read from the CURRENT params at every projector
        # build (create_projectors re-runs on reconfigure/recompile, closing
        # the stale-bind class) and PRE-PLACED once per device through the
        # probed transfer primitive -- the per-band `.to(dev)` copies this
        # replaces bypassed the dev2dev-safe policy.
        view_params_name = model.get_params('view_params_name')
        view_params = torch.as_tensor(model.get_params(view_params_name),
                                      dtype=_F32, device=model.torch_device)
        self._view_params_per_dev = [
            _sharding.move_shard(view_params, dev, model.dev2dev_safe)
            for dev in model.sino_placement.devices]
        self.view_params_array = self._view_params_per_dev[0]

    def _effective_view_batch(self, body, num_pixels, band_cols, args):
        """The view batch for one call of ``body``: the nominal batch, capped
        so one batch's transient stays within the budget above.

        The batching rule follows the BODY actually bound, never the
        geometry.  A hand-written kernel body carries a ``_view_batch_cost``
        attribute (attached in its own module, e.g. triton_cone.py) returning
        ``(bytes_per_view, view_chunk)`` from ``(num_pixels, band_cols,
        args)``: its real charged residency and its swept nominal chunk.  The
        torch bodies' gather-transient model below would charge such a kernel
        a transient it never materializes -- at large cells that mischarge
        forces view batch 1 and a per-view contract rebuild per launch.  The
        forward and back bodies are consulted separately, so a mixed
        selection (kernel one way, torch body the other, including a
        self-check fallback at create_projectors time) batches each direction
        by its own model.

        Torch-body path (any body without the attribute, compiled or eager):
        unchanged.  The column count is a GEOMETRY hook (_transient_cols):
        parallel's transient tracks the runtime band length, cone's tracks
        max(num_slices, num_rows) from the params -- unifying them naively
        would silently change each geometry's batch size, float summation
        order, and calibrated peak memory.

        ``model.view_batch_size`` is the user's nominal for every body; None
        means automatic (VIEW_BATCH_BODY_DEFAULT for a torch body, the
        kernel's own chunk for a kernel body)."""
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
            band_cols (int): the call's column count -- the forward's voxel
                columns, or the back's local sinogram row count.
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

        ``accumulate_into`` lets a caller that runs this loop repeatedly -- the
        column-gather forward, once per pixel batch -- add straight into the
        block it is building instead of receiving a fresh one to add itself.
        That merges two full-block passes into one and drops one full-block
        allocation per call; see the accumulation comment in
        ``TomographyModel._sparse_forward_project_columns`` for why it is worth
        doing and why the values do not move.  The parameter is added HERE, on
        a plain python method, and not to the geometry body: the bodies are
        torch.compile'd per device with shape-keyed caches, so a new argument
        there would recompile every one of them.

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
        # Whether this call adds into a block it was handed or fills one of its
        # own, decided ONCE here rather than per view batch: an accumulating
        # call adds every batch, including the first, because the block already
        # holds earlier calls' work.
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
            # View batches cover DISJOINT rows of the block, so neither arm
            # sums anything across this loop -- assignment and addition touch
            # each row exactly once either way.
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
            # Released AFTER the accumulation, so the summation order is
            # untouched -- this changes only how much memory is held.
            # Without it the next iteration's `back_body(...)` evaluates
            # before rebinding `block`, holding accumulator + outgoing +
            # incoming: min(3, nb) cylinder-shards where min(2, nb) is what
            # the loop needs.  On the first iteration `out` IS `block`, so
            # dropping the name costs nothing there.  The saving measures
            # smaller than the counts suggest, because the outgoing block is
            # often freed partway through the next kernel anyway: expect
            # about half a cylinder-shard, not a whole one.
            block = None
        return out

    def _sparse_forward_project_single_device(self, voxel_values, pixel_indices):
        """Forward project voxel cylinders into a full sinogram on ONE device:
        the view-range loop at (0, num_views) on device 0, coercing
        array-likes to placed tensors first.  This is the trivial-placement
        form only -- both of its callers are the trivial branches in
        TomographyModel, and every external call funnels through
        TomographyModel.sparse_forward_project.  For a row-aligned geometry
        the output row count equals the input column count -- the
        rows==slices invariant its verify_valid_params enforces."""
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
        ``pixel_indices`` on ONE device: the view-range loop at
        (0, num_views) on device 0.  The trivial-placement form only (see
        :meth:`_sparse_forward_project_single_device`); every external call
        funnels through TomographyModel.sparse_back_project.

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
