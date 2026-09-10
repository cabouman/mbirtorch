# Comparison of mbirjax and mbirtorch

This document compares the two packages and explains how the code was migrated
from mbirjax to mbirtorch.  It was written on 2026-09-09 from
`mbirjax/mbirjax/` and `mbirtorch_ziyun_dev/mbirtorch/`.

## Summary

The migration kept the algorithms and replaced the framework mechanisms.  The
science is the same in both packages.  This includes the VCD algorithm, the
qGGMRF surrogate formulas, the projection weights, the automatic regularization
rules, the arbitrary length unit conventions, and the whole `preprocess`
subpackage.  Section 1.7 defines the projection weights, because the rest of
this document refers to them repeatedly.  What changed is every place where the
code used a JAX feature to organize the computation.  The two packages are close
in size, at 25,570 Python lines in `mbirjax/` and 26,868 in `mbirtorch/`.

The migration was not a substitution of `torch` calls for `jnp` calls.  Six
larger changes describe it.  Each is discussed in Section 1.

1. Compilation moved from the whole projector driver to small individual
   functions.
2. The interface a geometry model must implement became four methods and three
   class attributes.
3. Multiple-device support was rewritten around a list of per-device tensors
   instead of one array spread over a mesh.
4. Memory management changed from handling out-of-memory errors to predicting
   memory use before allocating.
5. The custom GPU kernels changed from Pallas to Triton, and the rule for using
   them changed from a list of approved GPUs to a test run on the actual GPU.
6. The projectors became differentiable, which mbirjax did not support.

Two smaller changes run through the whole package.  The public functions still
accept and return NumPy arrays, and still offer `output_sharded=True` to get the
device form instead.  Several function names were changed to put the verb first.
For example, `fbp_recon` became `recon_fbp`, `fdk_recon` became `recon_fdk`,
`direct_recon` became `recon_direct`, and `split_sino_recon` became
`recon_split_sino`.  The name `vcd_recon` became the private `_vcd_recon`.

## 1. The six larger changes

### 1.1 Compilation moved to individual functions

In mbirjax the entire projector driver is one compiled program.  A single
`jax.jit` covers the driver, and the loops over pixel batches and view batches
are written inside that compiled program using `jax.lax.map` and `jax.vmap`.
Any value that must be a compile-time constant is passed as a named tuple in
`static_argnames`.  This is why mbirjax takes care that the named tuple classes
stay stable, so that their tree definitions hash equal and the compilation cache
is shared.  See `mbirjax/projectors.py:9-30` and
`mbirjax/parameter_handler.py:20-25`.

In mbirtorch the driver is ordinary Python.  The loops over view batches are
written as `for` loops.  Only small pure functions at module level are passed to
`torch.compile`.  Parameters are read at each call, outside the compiled region.
The reason is stated in the code.  Bodies must be module-level pure functions
and never bound methods.  A bound method would keep the model alive in the
module-level compilation cache.  A parameter read inside the compiled region
would make Dynamo trace the parameter machinery.  See
`mbirtorch/tomography_model.py:286-296`.

### 1.2 The geometry model interface became smaller

In mbirjax a geometry model supplies per-view projection functions as static
methods.  Each is decorated with `@partial(jax.jit,
static_argnames='projector_params')`.  The cone beam model also supplies a
separate banded back projector, `back_project_one_view_to_band`.  The model
supplies `get_geometry_parameters()`, which returns a named tuple that JAX can
treat as a compile-time constant.

In mbirtorch a geometry model supplies four methods.  `_view_batch_bodies()`
returns the forward and back projection functions.  `_view_batch_args()` builds
the keyword-argument dictionary for them.  `_transient_cols(band_cols)` reports
the largest temporary array width, which the memory model needs.  The model also
sets three class attributes: `rows_track_slices`, `_floor_family`, and
`min_compiled_pixel_width`.  The named-tuple machinery still exists as
`ParameterHandler.make_geometry_params` at `mbirtorch/parameter_handler.py:396`.
Its only remaining purpose is convenient equality and printing.

### 1.3 Multiple-device support was rewritten

In mbirjax the reconstruction is one `jax.Array` spread over a device mesh.
That array is built with `make_array_from_single_device_arrays`.  A mesh-backed
array needs equal-length pieces on every device.  Therefore mbirjax pads the
divided axis with zeros up to a multiple of the device count.  Considerable code
then keeps those padded entries from affecting results.  That code includes
`Placement.real_mask`, `padded_shard_ranges`, `_pad_shard_on_axis`,
`_mask_padded_views`, `_mask_padded_slices`, and the `interface_mask` argument in
the qGGMRF prior.

In mbirtorch the reconstruction is a `Shards` object, which holds one ordinary
tensor per device.  The division follows `numpy.array_split`, so the pieces may
have unequal lengths and the axis is never padded.  See
`mbirtorch/_sharding.py:43-70`.  DTensor was considered and not used, because it
was judged immature for these index-heavy kernels.  See
`mbirtorch/_sharding.py:7-14`.

This one decision removed all of the padding-related code listed above.  It
introduced a new requirement in its place.  Any operation on a block of slices
or views must use the length of the block it was given, and must accept a block
of length zero.  That requirement is stated in `docs/source/dev_api.rst`.

### 1.4 Memory management became a prediction

In mbirjax an out-of-memory error is handled after it happens.  The helper
`_utils.is_oom` matches the error text, `log_oom_guidance` prints advice, and
`TomographyModel._handle_jax_error` reformats the error.  See
`mbirjax/tomography_model.py:3017`.  Performance is tuned by a `TilePolicy`
named tuple that holds about eight batching settings.  See
`mbirjax/tomography_model.py:44-63`.

In mbirtorch both of those are gone.  The module `_memory_ledger.py` computes
the peak memory of each phase of the computation before any large array is
allocated, and uses that to choose how many devices to use.  The module is 1,646
lines.  It shares `Projectors.view_batch_charge` with the driver, so that the
batch chooser and the memory model cannot disagree.  It also shares
`padded_kernel_width` and `reduce_slab_rows` with the code that allocates.  If
the problem does not fit, the code raises `MemoryPreflightError` naming the
phase that dominates.

A second module, `_widening_floors.py`, holds measured problem sizes below which
using more devices is slower rather than faster.  On parallel beam at small
sizes the measured penalty reached a factor of 13.  The memory model chooses a
device count that both fits and is not below the measured floor.

The users' tuning settings were reduced to one, `view_batch_size`.  The mbirjax
settings `fwd_view_batch`, `back_view_batch`, `fwd_pixel_batch`,
`back_pixel_batch`, the slice-band settings, `sort_by_channel`, and
`back_stacked_gather` were all removed.

### 1.5 Custom kernels changed from Pallas to Triton

In mbirjax the custom kernels are written in Pallas, in
`mbirjax/_pallas_kernels.py`.  They are used only on GPUs named in
`_ARCH_ALLOWLIST`, which contains `H100`, `H200`, and `A100`.  The check is a
substring match on the reported device kind.

In mbirtorch the custom kernels are written in Triton, in `triton_cone.py` and
`triton_parallel.py`.  The rule for using them is in `kernel_availability.py`.
That module first compiles and runs a small Triton kernel once per process to
see whether Triton works at all.  It then checks each kernel on each device by
comparing its output against the plain PyTorch version, with a relative
tolerance of 1e-4.  Because the check measures the actual machine, the kernels
are enabled by default on any GPU architecture.

### 1.6 The projectors became differentiable

mbirjax has no `jax.custom_vjp` or `jax.custom_jvp` anywhere.  Its only
transpose facility is `get_transpose`, which calls `jax.linear_transpose`.  See
`mbirjax/tomography_model.py:4095`.

mbirtorch adds `autograd.py`, which wraps the projectors as
`torch.autograd.Function` subclasses and provides a `torch.nn.Module` wrapper.
The backward pass of each is simply the other projector.  This is correct
because the forward and back projectors are an exact adjoint pair, which
`tests/test_adjoint.py` checks.  No differentiation through the kernel internals
is needed.  The wrapper calls `_require_single_device(model)` and raises
`NotImplementedError` for a multiple-device layout, because the projector
outputs would be `Shards` objects and the gradients would be silently lost.

### 1.7 The projection weights, which did not change

Both packages compute the same projection weights.  This section defines them,
because later sections refer to them and to the functions that compute them.

A projection weight is the overlap of one voxel with one detector cell.  A voxel
projects onto the detector as a footprint of some width, centered at some
position.  A detector cell is an interval of width one, in units of the cell
spacing.  The weight is the length of the overlap between the two.  Three quantities enter the
expression for it:

* `W`, the projected width of the voxel, in units of the cell spacing;
* `x_p`, the continuous projected coordinate of the voxel center;
* `n`, the integer index of the detector cell.

The weight is then the following expression.

```
A = weight_scale * clip((W + 1) / 2 - |x_p - n|, 0, min(1, W))
```

Considered as a function of the offset `x_p - n`, this overlap has the shape of
a trapezoid.  Its base has width `W + 1`, its flat top has width `|W - 1|`, and
its height is `min(W, 1)`.  The code therefore calls it the trapezoid weight.
The name refers to that shape.  It does not refer to the trapezoidal rule for
numerical integration, and the expression is an exact overlap rather than an
approximation.  In mbirtorch this expression is the function `tap_weights` at
`mbirtorch/horizontal_fan.py:52`.

The projection is computed as two separate one-dimensional operations, which the
code calls fans.  The horizontal fan maps a voxel to a range of detector
channels.  The vertical fan maps each slice of a column of voxels to a range of
detector rows, which is needed because the cone angle spreads one slice over
several rows.  Both fans use the weight expression above.  The vertical fan then divides by
`cos(phi)`, where `phi` is the vertical cone angle of the voxel.  That division
accounts for the longer path a ray takes through a voxel away from the central
plane.  See the module docstring at `mbirtorch/cone_beam.py:13-16`.

Parallel beam has no vertical fan.  There, detector row `r` corresponds to
reconstruction slice `r`, which the class records by setting
`rows_track_slices = True` at `mbirtorch/parallel_beam.py:155`.

Only one part of this arrangement changed in the migration.  In mbirjax the
geometry supplies the horizontal fan data through
`get_geometry_parameters()`.  In mbirtorch it supplies the same four values as
the tuple `(n_p, centers, W_p_c, weight_scale)`, computed for each view and
pixel.  Those four values are the continuous projected coordinate, its rounded
integer center, the projected voxel width, and the geometry weight scale.  The
Triton kernels read that same tuple.  Neither package ever allocates an array
indexed by tap, because the taps are expanded inside the kernel loops.

## 2. Module correspondence

The table lists each module, its line count, and what changed.  Line counts are
of the Python files as of 2026-09-09.

| mbirjax module | Lines | mbirtorch module | Lines | What changed |
|---|---|---|---|---|
| `tomography_model.py` | 4116 | `tomography_model.py` | 3588 | Ported and reorganized.  Removed `TilePolicy` and `_select_tile_policy`, the `*_for_vmap` properties, `set_devices`, `get_compute_config`, `set_view_parameters`, `_handle_jax_error`, the padding and masking helpers, the Pallas dispatch, and `get_transpose`.  Added `_apply_device_policy`, `_settle`, `_fits_available_devices`, `_arm_calibration`, a wider `configure_devices`, the three geometry hooks, `recon_plastic_metal`, the `full_indices_device()` cache, and `refresh_device_bindings`. |
| `projectors.py` | 1001 | `projectors.py` | 690 | Rewritten.  The fan arithmetic moved to `horizontal_fan.py`.  The compiled drivers became the Python loops in `sparse_*_view_range`.  Removed `channel_scatter_reduce`, `_channel_reduce_sort_segsum`, `concatenate_function_in_batches`, `sum_function_in_batches`, `unbatch`, and `ensure_tuple`.  Added `maybe_compile`, `_raise_recompile_budget`, `compile_serialized`, `view_batch_charge`, and the minimum-pixel-width padding wrappers. |
| `parameter_handler.py` | 642 | `parameter_handler.py` | 401 | Reduced.  Removed the YAML save and load functions, which are `save_params`, `load_param_dict`, `serialize_parameter`, `deserialize_parameter`, `convert_arrays_to_strings`, `compare_parameter_handlers`, and `get_required_params_from_dict`.  Also removed the `ParamNames` type annotations.  Added per-instance logging keyed on a counter, the log-file handlers, `_device_report`, `_log_device_report`, and `refresh_device_bindings`. |
| `parallel_beam.py` | 596 | `parallel_beam.py` | 722 | Ported, and larger.  It gained the kernel selection code, `recon_split_sino`, and `recon_simple_parallel`. |
| `cone_beam.py` | 1674 | `cone_beam.py` | 1190 | Ported.  The per-pixel functions that JAX applied with `vmap` were rewritten as batched arithmetic. |
| `multiaxis_parallel.py` | 808 | `multiaxis_parallel.py` | 449 | Ported, with the same rewrite of the per-pixel functions. |
| `translation_model.py` | 933 | `translation_model.py` | 482 | Ported. |
| `qggmrf.py` | 573 | `qggmrf.py` | 252 | Ported and divided.  The multiple-device halo exchange moved to `_sharding.exchange_qggmrf_halos`.  The denoiser support functions `compute_surrogate_and_grad` and `compute_qggmrf_grad_and_hessian` moved to `denoising.py`.  The two nested `vmap` calls became one batched function.  The `interface_mask` argument was removed. |
| `_sharding/` (4 files) | 561 | `_sharding.py` | 726 | Combined into one module.  Removed `assemble_sharded` and `sharded_full`.  Added `Shards`, `reject_shards`, `reduce_slab_rows`, `transfer_cylinder_batch`, the copy-stream helpers, and `exchange_qggmrf_halos`. |
| `_device_setup.py` | 250 | none | | Not ported.  Its purpose was to set `XLA_FLAGS` for virtual CPU devices before JAX was imported, and PyTorch needs nothing equivalent.  Device selection moved into `TomographyModel._resolve_device` and `configure_devices`.  The compilation-cache environment setup at the top of `mbirtorch/__init__.py` is the only remaining part. |
| `_pallas_kernels.py` | 885 | `triton_cone.py`, `triton_parallel.py`, `kernel_availability.py` | 770, 776, 469 | Replaced.  See Section 1.5. |
| `_utils.py` | 242 | `_utils.py` | 115 | Reduced.  Removed `is_oom`, `log_oom_guidance`, `_called_by`, `update_param_literal`, and `highlight_differences`.  Added `padded_kernel_width` and `KERNEL_WIDTH_MULTIPLE`.  The parameter defaults are identical except that `use_gpu` was removed. |
| `memory_stats.py` | 108 | `memory_stats.py` | 88 | Ported to `torch.cuda`, `torch.mps`, and psutil.  It does not list the live arrays, because PyTorch keeps no registry of them. |
| `denoising.py` | 663 | `denoising.py` | 684 | Ported.  It gained `_denoise_sharded`, `_subsample_to_host`, and a chunked `image_ell1`. |
| `vcd_utils.py` | 496 | `vcd_utils.py` | 389 | Ported without `gen_pixel_partition_grid` and `gen_pixel_partition_blue_noise`.  It gained `estimate_background_cluster_boundaries`. |
| `tomography_utils.py` | 131 | `tomography_utils.py` | 78 | Ported.  The `@jax.jit` decorator on `apply_row_filter` was dropped. |
| `utilities.py` | 2140 | `utilities.py` | 1988 | Ported.  It gained `clear_cache` and the multiple-device phantom helpers `_shepp_logan_band`, `_phantom_devices`, and `_sharded_slab_source`.  It lost the Matplotlib plotting helpers. |
| `viewer.py` | 1136 | `viewer.py` | 2415 | Rewritten rather than ported.  It is now a NumPy-only `VolumeStack` data model plus a Matplotlib `SliceViewer`, and it does not import the rest of the package. |
| none | | `view_utils.py` | 135 | New.  It converts tensors to NumPy arrays and dictionaries to display strings for the viewer. |
| none | | `autograd.py` | 119 | New.  See Section 1.6. |
| none | | `horizontal_fan.py` | 154 | New.  It holds `horizontal_fan_project` and `horizontal_fan_back`, which were in `projectors.py`. |
| none | | `_memory_ledger.py` | 1646 | New.  See Section 1.4. |
| none | | `_widening_floors.py` | 711 | New.  See Section 1.4. |
| none | | `kernel_availability.py` | 469 | New.  See Section 1.5. |
| `hsnt.py` | 663 | `hsnt.py` | 664 | Ported with almost no change. |
| `vcls.py` | 634 | `vcls.py` | 614 | Ported. |
| `bn256.py` | 259 | `bn256.py` | 261 | Ported. |
| `preprocess/` (9 files) | 5598 | `preprocess/` (9 files) | 5808 | All ported with almost no change.  `_xradia_ole.py` and `zeiss_tct.py` have identical line counts in the two packages. |
| `mace4d.py` | 1039 | none | | Not ported.  See `mace4d_migration_plan.md`. |

## 3. How individual JAX constructs were translated

Each subsection below gives one JAX construct and the mbirtorch replacement.

### 3.1 A compiled static method becomes a module-level function

In mbirjax the projection function is a static method with a decorator.  See
`mbirjax/parallel_beam.py:294`.

```python
@staticmethod
@partial(jax.jit, static_argnames='projector_params')
def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle,
                                            n_p_centers, projector_params):
```

In mbirtorch it is a module-level function whose arguments are plain values.
See `mbirtorch/parallel_beam.py:56`.

```python
def _parallel_forward_view_batch(values, pixel_indices, view_params_batch,
                                 num_rows, num_cols, num_channels,
                                 delta_det_channel, det_channel_offset,
                                 delta_voxel, delta_voxel_row, psf_radius,
                                 slice_start=0, plan=None):
```

Compilation is decided by the driver, not by the function.  The driver calls
`maybe_compile(fn, enabled, instance_key=None)` at
`mbirtorch/projectors.py:123`.  That function caches the compiled result under
the pair of the function and the `instance_key`, and the `instance_key` is the
device index.  A separate compiled copy per device is required because the
compiled artifact holds Triton launcher state that must not be shared between
threads running at the same time.

### 3.2 A static named tuple becomes a dictionary built at each call

mbirjax passes `projector_params` and `geometry_params` as named tuples in
`static_argnames`.  Keeping the tree definitions hashable required care about
the stability of the named tuple classes.

mbirtorch builds a keyword-argument dictionary in `_view_batch_args()` instead.
The docstring of that method states the requirement.  Every parameter is read
inside `_view_batch_args`, outside the compiled region, at each call, and never
frozen when the projector is built.  See `mbirtorch/tomography_model.py:297`.
Scalar arguments then become compile-time constants inside `torch.compile`
without any extra work.

### 3.3 `jax.vmap` becomes arithmetic written on batched arrays

There is no use of `torch.vmap` or `torch.func.vmap` anywhere in mbirtorch.
The functions that JAX applied with `vmap` were rewritten to operate on the
batched arrays directly.

The qGGMRF prior is one example.  mbirjax nests two `vmap` calls at
`mbirjax/qggmrf.py:126-132`.

```python
cylinder_map = jax.vmap(qggmrf_grad_and_hessian_per_cylinder, in_axes=(0, None, 0, 0, None))
slice_map    = jax.vmap(qggmrf_grad_and_hessian_per_slice,
                        in_axes=(1, None, None, None, 1, 1), out_axes=1)
```

mbirtorch writes one function that takes arrays of shape `(N, S)`.  Its
docstring gives the reason this is valid.  The operations are elementwise on the
same operands, so the batched form computes the same values as a loop over
cylinders.  See `mbirtorch/qggmrf.py:7-12`.  Three functions in mbirjax became
the single function `qggmrf_gradient_and_hessian_at_indices`, which is about 100
lines.

The cone beam vertical fan is a second example.  mbirjax applies `vmap` to
`ConeBeamModel.forward_vertical_fan_one_pixel_to_one_view` at
`mbirjax/cone_beam.py:518`.  mbirtorch computes the same result with explicit
arithmetic on arrays of shape `(Vb, P, L)` inside `_cone_forward_view_batch`.

### 3.4 `jax.lax.map` over reshaped batches becomes a `for` loop

mbirjax uses `concatenate_function_in_batches` at `mbirjax/projectors.py:662`.
That function runs one initial batch of size `num % batch`, reshapes the rest to
`(num_batches, batch, ...)`, and calls `jax.lax.map`.

mbirtorch uses a `for` loop at `mbirtorch/projectors.py:518-588`.

```python
for v in range(v0, v1, vb_size):
    view_params_batch = view_params[v:min(v + vb_size, v1)]
    block = fwd_body(band_values, pixel_indices, view_params_batch,
                     slice_start=slice_start, plan=plan, **args)
    if out is None:
        out = torch.empty((v1 - v0,) + tuple(block.shape[1:]), ...)
    out[rows] = block
```

The output array is allocated from the shape of the first block.  The driver
therefore never computes any geometry-specific shape itself.  This plays the
same role that shape-polymorphic tracing plays in JAX.

### 3.5 `donate_argnames` becomes in-place mutation in one compiled function

mbirjax marks buffers for reuse with `donate_argnames` on two separate compiled
functions.  See `mbirjax/tomography_model.py:4040-4073`.

```python
@partial(jax.jit, donate_argnames='cur_flat_recon')
def update_recon(cur_flat_recon, cur_indices, cur_delta): ...

@partial(jax.jit, donate_argnames='error_sinogram')
def update_error_sinogram(error_sinogram, alpha, delta_sinogram): ...
```

mbirtorch combines both into one compiled function, `_apply_update`.  That
function modifies its two state tensors in place and also returns them.  Callers
reassign their variables from the return values.  See
`mbirtorch/tomography_model.py:50-88`.

### 3.6 Index updates require zeroing the weight and clamping the index

This is the semantic difference that affects the most code.  JAX index
operations drop or clip out-of-range indices.  PyTorch index operations raise an
error instead.  Every place that adds into an array at computed indices
therefore sets the weight to zero and clamps the index into range.  See
`mbirtorch/horizontal_fan.py:31-38`.

```python
A = A * ((n >= 0) & (n < num_channels)).to(_F32)
return A, n.clamp(0, num_channels - 1)
```

The same pattern replaces `jnp.ravel_multi_index(..., mode='clip')` in the
prior.  The row index becomes `r = (row_index + dr).clamp(0, num_rows - 1)`, and
the value is read as `flat_recon[r * num_cols + c]`.  See
`mbirtorch/qggmrf.py:165-176`.

### 3.7 `jax.device_put` becomes `move_shard`

All movement of data between devices goes through
`_sharding.move_shard(x, target, dev2dev_safe)`.  A bare call to `tensor.to(dev)`
should not appear in code that runs often.

The function keeps a test that mbirjax needed for a different reason.  In
mbirjax, `device_put` silently corrupted device-to-device transfers on some
GPUs, including the L40S.  No such fault is known for `tensor.to()` in PyTorch.
The test was kept anyway because it costs almost nothing.  See
`mbirtorch/_sharding.py:308-312`.  If the test fails, the transfer goes through
host memory and a warning is printed once per process.

### 3.8 `jax.default_device` is not needed

mbirjax runs each per-device worker inside `with jax.default_device(devices[i])`,
then assembles the results into one mesh-backed array.  mbirtorch needs neither
step, because each tensor records its own device.  See
`mbirtorch/_sharding.py:644`.  The results stay a `Shards` list.  Both packages
deliberately omit a synchronization at the end, so that the caller can overlap
the transfer of the next block with the computation on the current one.

### 3.9 Compilation cache misses need an explicit budget

PyTorch recompiles a function when its guards fail, and stops after a fixed
number of recompilations.  JAX has no equivalent setting.  mbirtorch raises
`torch._dynamo.config.recompile_limit` to at least 64.  See
`mbirtorch/projectors.py:52-107`.

The budget must be raised on the thread that will call the compiled function.
Dynamo reads a per-thread view of this setting, so an assignment on one thread
does not reach another.  This was measured on torch 2.13.  Exceeding the budget
produces no message, and it was measured to cost a factor of 5 to 11 on two
H100 GPUs on 2026-08-19.

### 3.10 A Pallas kernel becomes a Triton kernel with the same signature

In mbirjax the Pallas kernels are built as `pl.pallas_call` objects under
`@functools.cache`, keyed on the static shapes.  The driver selects them through
flags such as `tiles.fwd_pallas`.

In mbirtorch the Triton kernels are wrapped in functions decorated with
`@torch.compiler.disable`.  Each wrapper has the same signature as the plain
PyTorch function it replaces, and `_view_batch_bodies` returns one or the other.
The wrapper also carries the attribute `_mbirtorch_no_compile`, which makes
`maybe_compile` return it unchanged.  That attribute is necessary because
`torch.compile` otherwise removes the `disable` decorator with `innermost_fn`
and traces the original function including the kernel launch.  See
`mbirtorch/projectors.py:128-134`.

Two further requirements apply only to PyTorch.  A Triton launch uses the
current device of the launching thread, so each wrapper sets the device from the
tensors it was given.  The test `tests/test_kernels_sharded.py` checks this.
Every width argument passes through `_utils.padded_kernel_width`, which rounds
up to a multiple of 16.  The measured cost of an unrounded width was a factor of
2.44, and 1.06 after rounding.

### 3.11 Gradients are disabled with `no_grad`

The reconstruction loop runs inside `with torch.no_grad():`.  It does not use
`torch.inference_mode()`.  The reason is recorded in the code.  The guards in
`torch.compile` fail on compiled calls inside `inference_mode` when those calls
update tensors in place.  See `mbirtorch/tomography_model.py:3170`.

### 3.12 Rematerialization appears in neither package

The word `checkpoint` appears 23 times in `mbirjax/tomography_model.py`.  Those
occurrences are the `return_checkpoint` interface for resuming a reconstruction,
not rematerialization of intermediate values.  That interface ported directly,
with one change in meaning.  In mbirjax the checkpoint can be used only once,
because buffer donation deletes the caller's copy, and a second use raises an
error.  In mbirtorch the checkpoint holds references to the loop's own final
tensors with no copy, and the documentation says to copy them if a snapshot is
wanted.

## 4. Modules that exist only in mbirtorch

Each new module is listed with its purpose.

`autograd.py`, 119 lines.  It presents the projectors as differentiable
operations for use with learned priors.  Section 1.6 explains why the
implementation is short.

`horizontal_fan.py`, 154 lines.  It holds the fan arithmetic that was in
`projectors.py`, so that the driver contains only iteration and memory
management.  It defines the data that a horizontal fan produces for each view
and pixel, which is the tuple `(n_p, centers, W_p_c, weight_scale)`.  The Triton
kernels consume that same data.  The array of taps is never allocated in full.
The `centers` array is `int32`, because 32-bit indices halve the size of this
data and are what GPU kernels prefer.

`kernel_availability.py`, 469 lines.  It decides whether to use the Triton
kernels, as described in Section 1.5.  It sets a flag while a check is running,
because each check builds a small model whose `create_projectors` call asks this
same module which functions to use.  It also reads the environment variables
`MBIRTORCH_DISABLE_TRITON` and `MBIRTORCH_SORTED_FORWARD`.  Three environment
variables that are no longer used are still defined, so that older scripts that
set them still run.

`triton_parallel.py` and `triton_cone.py`, 776 and 770 lines.  These hold the
Triton kernels.  `triton_parallel` imports its Triton compatibility helpers from
`triton_cone`, which keeps those helpers in one place and keeps the import graph
free of cycles.  Both modules import successfully when Triton is not installed,
so that the tests and the availability check can import them anywhere.  The
parallel-beam forward projector has two kernels behind one wrapper.  One works
tap by tap.  The other sorts each view's pixels by detector channel and finishes
each tile with a small full-precision matrix multiply.

`_memory_ledger.py`, 1646 lines.  It predicts peak memory, as described in
Section 1.4.  It is necessary because of the change described in Section 1.1.
The mbirtorch update loop is ordinary Python, so no single compiled program ever
sees which tensors are alive across calls.  In mbirjax, XLA had that view of the
whole program.  Setting `MBIRTORCH_MEMORY_CALIBRATION=1` compares the predicted
peak against `torch.cuda.max_memory_allocated`.

`_widening_floors.py`, 711 lines.  It holds the measured problem sizes below
which using more devices is slower, as described in Section 1.4.  It also
protects itself against becoming inconsistent, through four members:

* `BLESSED_COST_HASHES` records hashes of the inputs to the cost model;
* `stale_note()` reports when the table was last measured;
* `monotone_violations()` checks the table for inconsistent entries;
* `TABLE_CHECKSUM` covers the floors, the hashes, and `STALE_SINCE` together.

The checksum covers the hashes, so a hand-edited hash is detected.

`view_utils.py`, 135 lines.  It supplies the two package-specific services that
`viewer.py` needs.  The first converts a tensor to a NumPy array with
`.detach().cpu().numpy()`, which is needed because `np.asarray` cannot convert a
tensor that is on a device.  The second serializes a dictionary to a display
string.  mbirjax has that second function as a static method on
`TomographyModel` at `mbirjax/tomography_model.py:1193`.

Several functions were also added to `_sharding.py`.  `Shards` holds one tensor
per device.  `reject_shards` refuses a divided array at the start of a function
that cannot accept one, and its message names the function, the argument, and
the fix.  `reduce_slab_rows` performs a sum in slabs of at most `REDUCE_SLAB_BYTES`,
which is 256 MiB.  The extra memory needed above the running total is therefore
a fixed number of bytes rather than a fraction of the volume.
`transfer_cylinder_batch` and its asynchronous form transfer pixels for the
forward projector, with a default batch of `FORWARD_PIXEL_BATCH`, which is
32768.  `exchange_qggmrf_halos` exchanges the boundary slices that the prior
needs.

## 5. What mbirtorch does not have

### 5.1 Not yet ported

The following exist in mbirjax and are intended to work in mbirtorch, but have
not been written yet.

* `mace4d.py`, 1039 lines.  This is the only complete feature module that is
  absent.  Its helper `_construct_time_frame_models` in `utilities.py` and its
  documentation page `docs/source/usr_mace4d.rst` are also absent.  A plan for
  this port is in `mace4d_migration_plan.md`.
* `set_view_parameters()`.  In mbirjax this changes the view angles without
  triggering a recompilation.  See `mbirjax/tomography_model.py:2428` and
  `mbirjax/tests/test_view_params.py`.  In mbirtorch the view parameters are
  read and placed on the devices at each call to `create_projectors`.
* `get_compute_config()`.  Some of what it reported is now in `_device_report`,
  `_log_device_report`, and the run log.
* The functions that read and write parameters as YAML.  These are
  `save_params`, `load_param_dict`, `serialize_parameter`,
  `deserialize_parameter`, `convert_arrays_to_strings`,
  `convert_strings_to_arrays`, and `compare_parameter_handlers`.  Note that
  `save_recon_hdf5` and `load_recon_hdf5` were ported, so saving a
  reconstruction works.  `load_recon_hdf5` lost the argument
  `recreate_model=False`.
* `gen_pixel_partition_grid` and `gen_pixel_partition_blue_noise`.  The module
  `bn256.py` was ported, and `vcls.get_2d_subsampling_indices(blue_noise=True)`
  uses it.
* The Matplotlib helpers in `utilities.py`.  These are `debug_plot_indices`,
  `debug_plot_partitions`, `plot_granularity_and_loss`, `save_volume_as_gif`,
  `make_figure_folder`, `display_translation_vectors`, and
  `download_and_extract_tar`.

### 5.2 Removed on purpose

The following were removed and are not expected to return.

* The `ParamNames` type annotations, which used `Literal` and `@overload` on
  `get_params` in every geometry class.  The script `_utils.update_param_literal`
  that kept them consistent was removed with them.
* The parameter `use_gpu`.  Its replacement is `configure_devices`.
* The property `device_summary`.  Its replacement is `get_memory_stats`.
* `get_transpose`, which used `jax.linear_transpose`.  Its replacement is
  `autograd.py`.
* The mbirjax performance layer.  This is the sorted channel reduction
  `channel_scatter_reduce` with its four measured crossover constants, the
  stacked-gather back projector, and `TilePolicy`.  The module docstring at
  `mbirtorch/projectors.py:8-11` states that their PyTorch equivalents belong
  with future Triton kernel work.

### 5.3 A common mistake about the module list

`translation_model.py` and `multiaxis_parallel.py` were ported.  They are 482
and 449 lines, both implement `_view_batch_bodies` and `_transient_cols`, and
both are exported from `__init__.py`.  The name `MultiAxisParallelBeamModel` is
kept as an alias of `MultiAxisParallelModel`.

### 5.4 Known limitations recorded in the code

The view-batch memory budget cannot bound the arrays for a single view beyond a
reconstruction of about 1400 voxels on a side, or on a detector of about 6000 by
10000 pixels.  The driver loop was written as a two-axis walk over tiles with an
accumulating forward projector, so that a loop over pixel batches can be added
without changing the geometry model interface.  See
`mbirtorch/projectors.py:340-374`.

Two rows of `_widening_floors.py` are placeholders.  They belong to the
denoiser, for which splitting across devices was slower at every size measured.

The directory `docs/source/_pending/` currently holds only its README file, and
that README lists no pending pages.  Every ported module therefore has a live
documentation page.

## 6. Rules for writing a new mbirtorch module

The example to copy is `docs/source/_static/new_model_template.py`, which
`dev_api.rst` includes in full.

### 6.1 Structure of a geometry model

The projection functions are module-level functions, not methods.  Name them
`_<geom>_forward_view_batch` and `_<geom>_back_view_batch`.  Name the geometry
helpers `_<geom>_horizontal_data` and `_<geom>_vertical_affine`.  Their
signatures are fixed.

```python
forward(values, pixel_indices, view_params_batch, <geometry keywords>,
        slice_start=0, plan=None)                              -> (Vb, rows, channels)

back(sino_batch, pixel_indices, view_params_batch, <geometry keywords>,
     coeff_power=1, slice_start=0, band_slices=None, plan=None) -> (P, slices)
```

The argument `plan` is reserved for future memoization and is not used today.
Keep it in the signature.  The argument `coeff_power` is 1 for the gradient and
2 for the diagonal of the Hessian.

The class overrides a fixed set of members.  These are `__init__`,
`get_magnification`, `get_psf_radius`, `auto_set_recon_geometry`,
`verify_valid_params`, `_view_batch_bodies`, `_view_batch_args`, and
`_transient_cols` if the geometry has two fans.  `__init__` must not accept
`**kwargs`, so that `set_params` can continue to reject misspelled parameter
names.  `verify_valid_params` must call the base class version first and must
raise `ValueError`.

The class also sets three attributes.  Set `rows_track_slices` to True only when
detector row `r` corresponds to reconstruction slice `r`.  Set `_floor_family`
to one of the families in `_widening_floors.py`, because omitting it silently
uses the parallel-beam floors.  Set `min_compiled_pixel_width` only when a
measured defect in the compiler justifies it.

Helper functions that are pure coordinate arithmetic remain static methods.  One
example is `ConeBeamModel.detector_mn_to_uv`.

### 6.2 Devices and data types

All floating-point arrays are `float32`.  Define `_F32 = torch.float32` at the
top of the module.  Index arrays are `int64`.  The `centers` array in the
horizontal fan data is `int32`.

Do not touch a device while constructing a model.  The device state resolves
when it is first needed, through the lazy properties `torch_device`,
`sino_placement`, `recon_placement`, and `projector_functions`.  The order of
preference is CUDA, then MPS, then CPU.

`Placement` is the only record of how arrays are divided across devices.  There
is no separate flag for a main device or for whether the model is divided.
Sinogram-shaped arrays are divided on axis 0, which is the view axis.
Reconstruction-shaped arrays are divided on the last axis, which is the slice
axis.

Move data between devices only with
`_sharding.move_shard(x, target, model.dev2dev_safe)`.

A block of slices or views may have a different length on each device, and may
have length zero.  Write per-slice and per-view code against the length of the
block that was passed in.

Public methods accept NumPy arrays or tensors and return NumPy arrays.  Passing
`output_sharded=True` returns the device form instead, which is a tensor for one
device and a `Shards` object for more than one.  Any function that requires a
whole array should call `_sharding.reject_shards('<function name>', arg=arg)` at
its start.

Do not allocate an array from a shape inferred from an empty tensor.  The
comment in `forward_project` about naming the row count rather than inferring it
records this.

### 6.3 Parameters

Parameter defaults are defined only in `_utils.py`, as `Param(value,
recompile_flag)`.  That file states that the names, values, and flags in it are
fixed and should not be changed there.

Read parameters with `get_params('name')` or `get_params([...])`.  Read them
inside `_view_batch_args`, at each call.  Do not read them when the projector is
built.

Put validation in `verify_valid_params`, which runs when a reconstruction
starts.  Do not put it in `set_params`.

Calling `set_params(no_warning=True, **kwargs)` with an unrecognized name adds
that name as a new parameter with a recompile flag.  See
`mbirtorch/parameter_handler.py:277-280`.  A subsystem uses this to register its
own parameters.

### 6.4 Compilation

Compile only through
`projectors.maybe_compile(fn, model.compile_enabled, instance_key=<device index>)`.
Use one compiled instance per device.

A hand-written kernel opts out of compilation with the attribute
`_mbirtorch_no_compile`.  Wrap only its first launch in `compile_serialized()`,
because that launch is where compilation and autotuning happen.

Pass kernel width arguments through `_utils.padded_kernel_width`.

Select kernels in `_view_batch_bodies`, guarded by
`kernel_availability.*_usable(model)[0]`.  Removing a kernel from use means
deleting the line that selects it.

### 6.5 Docstrings and comments

Use Google-style docstrings with `Args:`, `Returns:`, and `Raises:` sections,
and give types in parentheses.  Examples are `(tensor)`, `(int, optional)`, and
`(int, static)`.  A module docstring should say what the module is responsible
for, and also what it deliberately does not contain.

Record a measurement and a date next to any code whose form is not obvious.
Examples in the package are "Measured 2026-08-19 on two H100s (job 15391547)"
and "measured 2.44x penalty across the divisibility boundary before the padding
and 1.06x after".  Also record an alternative that was tried and rejected,
together with the reason.

Mark differences from mbirjax with one of three comment markers.  Use
`DIVERGENCE(<topic>)` for an intentional difference in behavior.  Use
`REPLACED(<topic>)` for a permanent substitution.  Use `PENDING(<topic>)` for
something that is still to be ported.

### 6.6 Logging

Each instance gets a logger named `mbirtorch.<Class>.<counter>`.  The counter
increases and is never the value of `id(self)`, because Python reuses id values.
A console handler is added when the object is constructed.  A file handler is
added by `setup_logger(logfile_path='~/.mbirtorch/logs/recon.log')`.  The
parameter `verbose` maps to logging levels, with 0 giving WARNING, 1 giving
INFO, and 2 or more giving DEBUG.

### 6.7 Tests

The file `tests/conftest.py` provides a `device` fixture that runs a test on
CPU, MPS, and CUDA.  It also provides two fixtures that apply to every test.
`redirect_default_log_location` patches `os.path.expanduser` so that
`~/.mbirtorch` is inside the pytest temporary directory.  `pin_device_count`
sets `MBIRTORCH_NUM_DEVICES=1`, and the comment explains that an environment
variable is easier to audit than a monkeypatch.

Agreement with mbirjax is checked against stored arrays.  Those arrays are
produced in the mbirjax environment by `tests/generate_goldens.py` and
`tests/generate_preprocess_goldens.py`, and are written to `tests/goldens/*.npz`,
which is not tracked by git.  `tests/test_vs_goldens.py` reads them under
`pytest.mark.goldens`, and skips when they are absent.  Setting `RUN_GOLDENS=1`
is therefore required to make these tests a real check.  The tolerances are a
relative maximum of 1e-4 for a single operation and 1e-3 for an iterated
computation.

A new geometry model needs four additions to the tests.  It needs an entry in
the stored arrays.  It needs a case in `test_sharding.py` that covers a
single-device reference result, unequal block lengths, adjointness, and more
devices than slices or views.  It needs an entry in `test_kernels_sharded.py` if
it adds kernels.  It needs a row in `test_memory_ledger.py` and
`test_widening_floors.py` if it adds a floor family.  Run the tests with
`python -m pytest -n 4 tests ci`.

### 6.8 Registering a module in the documentation

Three steps are required.  First, add the public names to `__all__` in
`mbirtorch/__init__.py`, because the package documents exactly the names
declared there.  A module that is imported lazily also needs an entry in
`_LAZY_MODULES` and `_LAZY_NAMES`.  Second, add a `usr_<topic>.rst` page, and
reference it both in the parent page's list of links and in the parent page's
`toctree`.  An example parent page is `usr_geometry_models.rst`.  Third, if the
code is not ready, put the page in `docs/source/_pending/`, which the build
excludes, and add a row to that directory's README naming the parent page it
will join.

One constraint on `usr_api.rst` should be preserved.  The `automodule` options
`:members:`, `:undoc-members:`, and `:show-inheritance:` are omitted on purpose.
Restoring them grows the page from 16.7 KB to 140 KB and produces eight warnings
about references to private names.
