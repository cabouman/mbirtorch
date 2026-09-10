# Plan for porting MACE4D to mbirtorch

This document describes what the MACE4D code in mbirjax does, which of its
dependencies already exist in mbirtorch, and a plan for writing the mbirtorch
version.  It was written on 2026-09-09.  The general porting conventions it
assumes are in `mbirjax_to_mbirtorch_migration.md`.

The source files are `mbirjax/mbirjax/mace4d.py` at 1039 lines,
`mbirjax/tests/test_mace4d.py` at 305 lines, and
`mbirjax/docs/source/usr_mace4d.rst` at 55 lines.  The files to be written are
`mbirtorch/mbirtorch/mace4d.py`, `mbirtorch/tests/test_mace4d.py`, and
`mbirtorch/docs/source/usr_mace4d.rst`.

## Summary

Most of MACE4D ports easily, and one part does not.

The parts that port easily are the outer algorithm and its bookkeeping.  These
run on the host in NumPy and use almost no JAX.  They include the division of
the scan into time frames, the consensus iteration, the temporal filter, the
task assignment across devices, and the logging.  Together they are roughly 700
of the 1039 lines.

The part that does not port easily is the batched denoiser.  MACE4D gets its
speed by denoising hundreds of two-dimensional planes as one compiled program,
written in JAX as `jax.jit(jax.vmap(sweep))`.  Two facts block a direct
translation.  First, `mbirtorch.denoising.vcd_subset_denoiser` modifies its
inputs in place, and `torch.func.vmap` does not allow that.  Second, JAX gives
each vmapped plane its own data-dependent stopping test through
`jax.lax.while_loop`, and PyTorch has no batched equivalent.  Section 4 gives
three options and recommends one.

One refactoring in mbirtorch is needed before the port can start.  The method
`_denoise_single_device`, which MACE4D calls directly, does not exist in
mbirtorch.  Its code was written inline inside `denoise()` instead.  Extracting
it is a small change that `tests/test_denoiser.py` can verify.

The estimated size of the port is 1200 to 1400 lines.  Section 7 gives the
suggested order of work.

## 1. What MACE4D computes

MACE4D reconstructs a time sequence of three-dimensional volumes from one
continuous scan of a moving object.  A conventional reconstruction of the same
scan gives one volume in which the motion appears as blur.  See
`mbirjax/mace4d.py:63-92`.

### 1.1 Division of the scan into time frames

The views are assumed to be recorded in time order at a uniform rate.  The
angular sweep is divided into overlapping windows of consecutive views, one
window per time frame.  Two parameters control the division.  The parameter
`frames_per_rotation`, whose default is 6, sets the angular step between window
starts to `2 pi / frames_per_rotation`.  The parameter
`frame_overlap_factor`, whose default is 2.0, sets each window to span
`frame_overlap_factor` times that step.  With the defaults each window spans 120
degrees and each view belongs to two windows.

The code is `_construct_time_frame_models` at `mbirjax/utilities.py:1915-1985`.

```python
angle_step      = float(np.median(np.abs(np.diff(angles))))
views_per_frame = int(round(angle_span_per_frame / angle_step))
stride          = int(round(angle_stride / angle_step))
for start in range(0, num_views - views_per_frame + 1, stride):
    view_slice = slice(start, start + views_per_frame)
    frame = copy_ct_model(model, new_angles=angles[view_slice])
```

Views at the end that cannot fill a whole window are discarded.  The function
returns a list of models and a list of view slices.  Each model is a complete
`TomographyModel` for one frame.  The function sets `verbose=0` on every model
after the first, so that the geometry report is printed once.

The test geometry has 24 views over 360 degrees.  With the default parameters it
produces 5 frames, covering views 0 to 7, 4 to 11, 8 to 15, 12 to 19, and 16 to
23.

### 1.2 The four agents

MACE4D is an instance of Multi-Agent Consensus Equilibrium.  It uses four
agents.  The reconstruction array has axis order `(t, x, y, z)`, where `t` is
the frame index.  The three prior agents are defined by `_PRIOR_ORIENTATIONS` at
`mbirjax/mace4d.py:37-40`.

| Index | Agent | What it is | Planes it denoises |
|---|---|---|---|
| 0 | Data fit | One `TomographyModel.prox_map` call per frame | Each frame separately |
| 1 | XY-t prior | qGGMRF denoiser, permutation `(3,0,1,2)` | `nz` planes of shape `(T, nx, ny)` |
| 2 | YZ-t prior | qGGMRF denoiser, permutation `(1,0,2,3)` | `nx` planes of shape `(T, ny, nz)` |
| 3 | XZ-t prior | qGGMRF denoiser, permutation `(2,0,1,3)` | `ny` planes of shape `(T, nx, nz)` |

The three prior agents are what makes the method four-dimensional.  Each one
treats the frame index as one of the three axes of a three-dimensional qGGMRF
prior.  The prior therefore couples neighboring frames as well as neighboring
voxels.

### 1.3 The agent weights

The four agents are combined with weights that sum to one.  The function
`_normalize_prior_weights` computes them at `mbirjax/mace4d.py:820-840`.  A
scalar `w` gives the weights `[1-w, w/3, w/3, w/3]`.  A list `[w1, w2, w3]`
gives the weights `[1-(w1+w2+w3), w1, w2, w3]`.  The order is data fit, XY-t,
YZ-t, XZ-t.  A negative weight, or weights that sum above one, raises
`ValueError`.  The default `mace_prior_weight=0.5` gives the weights
`[0.5, 1/6, 1/6, 1/6]`.

### 1.4 The consensus iteration

The algorithm holds two lists of four arrays each.  The arrays are `W[k]` and
`X[k]` for `k` from 0 to 3.  Each has shape `(T, nx, ny, nz)`.  All of them are
`float32` NumPy arrays in host memory.  See `mbirjax/mace4d.py:322-326`.

```python
W = [np.copy(init_recon) for _ in range(4)]
X = [np.copy(init_recon) for _ in range(4)]
_consensus_scratch = np.empty_like(init_recon)
```

Each iteration performs four steps.  The code is at
`mbirjax/mace4d.py:340-420`.

Step one applies the agents.  For each frame `t` it computes
`X_new[0][t] = prox_map(prox_input=W[0][t], init_recon=X[0][t], ...)`.  For each
orientation `k` it computes
`X_new[k+1] = denoiser(dejitter(W[k+1]), perm_k, global_sigma)`.  These
`T + 3` computations are independent of each other.

Step two reassembles the data-fit result and filters it in time.  The code is
`X[0] = self._dejitter(np.stack([results[("prox", t)] for t in range(T)]))`.
The temporal filter therefore runs four times per iteration, once here and once
inside each of the three denoiser tasks.

Step three is the consensus update.  Let `beta[k]` be the agent weights and let
`rho` be `rho_mann`.  The update computes an average and then moves each `W[k]`
toward it.

> z = sum over k of beta[k] * (2 * X[k] - W[k])
> W[k] = W[k] + 2 * rho * (z - X[k])

This is the standard MACE fixed-point iteration.  Writing `A` for the stacked
agents and `G` for the weighted averaging operator, the iteration is
`W = W + 2 * rho * (G A - I) W`.  The default `rho_mann=0.5` gives exact
consensus equilibrium, which is also Douglas-Rachford splitting.

The code computes this in place with one reused temporary array.

```python
scratch = _consensus_scratch
z = np.zeros_like(X[0])
for k in range(4):
    np.multiply(X[k], 2.0, out=scratch); scratch -= W[k]; scratch *= beta[k]; z += scratch
for k in range(4):
    np.subtract(z, X[k], out=scratch); scratch *= (2.0 * rho_mann); W[k] += scratch
```

The comment at `mbirjax/mace4d.py:365-369` gives the reason for the in-place
form.  Writing the same arithmetic as expressions allocates about 28 full-size
arrays per iteration.  That form was measured to be 7.1 times slower on the
full-resolution volume.

Step four computes the output and tests for convergence.  The output is
`xbar = sum over k of beta[k] * X[k]`.  The convergence measure is
`change_pct = 100 * norm(xbar - xbar_prev) / norm(xbar_prev)`, using the
two-norm.  The iteration stops when `change_pct` falls below
`stop_threshold_change_pct`.  The array `xbar` is what `recon` returns.  The
defaults are `max_iterations=10` and `stop_threshold_change_pct=0.2`.

### 1.5 Initialization

The initial reconstruction comes from one of three sources, in order of
preference.  See `mbirjax/mace4d.py:296-313`.  The caller may pass `init_recon`.
Otherwise the code loads `init_recon.npy` from `init_dir` if that file exists and
has the right shape.  Otherwise the code reconstructs each frame separately with
`_INIT_MBIR_ITERATIONS` iterations, which is 15.  A computed initial
reconstruction is written to `init_dir`.

### 1.6 The denoiser noise level

One noise level is estimated and used for all three orientations.  The function
is `_estimate_global_sigma` at `mbirjax/mace4d.py:545-552`.  It reshapes the
initial reconstruction to `(T * nx, ny, nz)` and calls
`QGGMRFDenoiser.estimate_image_noise_std`.  A value that is not finite, or is
not positive, raises an error.  The reason is that qGGMRF later divides by this
value.

### 1.7 The temporal filter

The overlapping windows add a periodic modulation along the frame axis, with a
period of `frames_per_rotation` frames.  The function `_dejitter_4d_dct` removes
it.  See `mbirjax/mace4d.py:736-812`.

The function works on the type-I discrete cosine transform along the frame axis,
computed as `scipy.fft.dct(..., type=1, norm="ortho", axis=0)`.  For a period
`p` and `N` frames it computes the coefficient index `k_center = 2 * (N - 1) / p`
and rounds it to `k0`.  It then sets the coefficients from `k0 - band_width` to
`k0 + band_width` to zero, and applies the inverse transform.  With
`harmonics=True` it repeats this for the periods `p / h` for `h` from 1 to
`floor(p / 2)`.  The argument `chunk_size` processes the last spatial axis in
blocks, which limits peak memory.

The signature is
`_dejitter_4d_dct(recon_4d, period, harmonics=True, band_width=1,
dtype=np.float32, chunk_size=None, verbose=False)`.  The method
`MACE4DModel._dejitter` calls it with `period=frames_per_rotation`,
`harmonics=True`, `band_width=1`, and `dtype=np.float32`.  The printing is
controlled by `dejitter_verbose` rather than `verbose`, because the filter runs
four times per iteration.

### 1.8 Task assignment across devices

Each iteration contains `T + 3` independent tasks.  The assignment of tasks to
devices is computed once per run by `_assign_tasks` at
`mbirjax/mace4d.py:697-718`.  The rule is least-loaded assignment.  The three
denoiser tasks are placed first, in order of decreasing plane count, with an
estimated cost of `_DENOISE_COST_PER_PLANE` per plane.  That constant is 0.015
and was measured on an H100.  The `T` data-fit tasks are then placed with an
estimated cost of one each.

Each device gets its own `ThreadPoolExecutor` with exactly one worker thread.
Every task for a device therefore runs on the same thread.  This is required for
two reasons.  The denoiser objects are cached in thread-local storage, and each
frame model must be used by only one thread.

### 1.9 Logging

When `log_dir` is given, `recon` writes three files.  The file `run_info.txt`
records the run settings, and is rewritten at the end to add the number of
iterations completed.  The file `timing_log.csv` has the fields listed in
`_TIMING_FIELDS`.  The file `task_log.csv` has the fields listed in
`_TASK_FIELDS`.

### 1.10 The batched denoiser

This is the part of MACE4D that does not translate directly.  The two functions
involved are `_denoiser_wrapper` at `mbirjax/mace4d.py:1002-1039` and
`_batched_hyperplane_denoise` at `mbirjax/mace4d.py:938-1000`.

`_denoiser_wrapper` performs four steps.  It permutes the array so that the
plane index is first.  It gets a denoiser from the thread-local cache, and
configures that denoiser if its configuration token has changed.  It calls
`_batched_hyperplane_denoise`.  It then permutes the result back.

`_batched_hyperplane_denoise` does not call `QGGMRFDenoiser.denoise()`.  It
calls the internal method `_denoise_single_device` instead, and applies `vmap`
and `jit` to it.

```python
def denoise_one(flat_vol):
    out, _, _, _ = denoiser._denoise_single_device(
        flat_vol, jnp.zeros_like(flat_vol), partition, fm_constant,
        qggmrf_params, image_shape, _DENOISE_MAX_ITERATIONS, stop_thresh, 0)
    return out
denoiser._mace4d_batched_fn = jax.jit(jax.vmap(denoise_one))
```

The batch size is fixed and the last block is padded with zeros, so that one
compiled program serves every block.  If the batch does not fit in memory, the
code halves the batch size and recompiles.

## 2. The interface to reproduce

The documentation page describes four members.  The property `devices` is public
but not documented.

```python
MACE4DModel(ct_model, frames_per_rotation=6, frame_overlap_factor=2.0, num_frames=None)
.set_params(no_warning=False, no_compile=False, **kwargs) -> bool
.recon(sinogram, weights=None, init_recon=None, max_iterations=10,
       stop_threshold_change_pct=0.2, init_dir=None, log_dir=None) -> (recon, recon_dict)
.set_device_pool(devices=None)
```

Everything else in the module is private and may be designed differently for
PyTorch.  One caution applies.  The mbirjax test file imports nine private
names, so the mbirtorch tests must either use the same names or be rewritten.
This plan rewrites them.

### 2.1 Parameters

| Name | Default | Where it is set | Meaning |
|---|---|---|---|
| `frames_per_rotation` | 6 | Constructor, then fixed | Frames per 360 degrees.  Also the period of the temporal filter. |
| `frame_overlap_factor` | 2.0 | Constructor, then fixed | Number of frames that share a view. |
| `num_frames` | None, meaning all | Constructor.  A value below 1 raises. | Use only the first `num_frames` frames. |
| `mace_prior_weight` | 0.5 | `set_params`, validated when set | A scalar or a list of three values. |
| `rho_mann` | 0.5 | `set_params` | The step size in the consensus update. |
| `prox_num_iterations` | 3 | `set_params` | The `max_iterations` argument of each `prox_map` call. |
| `prox_stop_threshold` | 0.02 percent | `set_params` | The stopping threshold of each `prox_map` call and of the initial reconstruction. |
| `sigma_prox` | None, meaning automatic | `set_params`, handled specially | Passed to every `prox_map` call. |
| `dejitter` | True | `set_params` | Whether to apply the temporal filter. |
| `dejitter_verbose` | 0 | `set_params` | Printing for the temporal filter only. |
| `verbose` | 1, inherited | `set_params` | 0 prints nothing, 1 prints progress, 2 prints debugging output. |
| `max_iterations` | 10 | Argument of `recon` | Number of consensus iterations. |
| `stop_threshold_change_pct` | 0.2 | Argument of `recon` | A value of 0 runs all iterations. |
| `init_dir` | None | Argument of `recon` | Directory for the cached `init_recon.npy`. |
| `log_dir` | None | Argument of `recon` | Directory for the three log files. |

Several values are fixed in the code rather than exposed as parameters:

* `_INIT_MBIR_ITERATIONS` at 15;
* `_DENOISE_MAX_ITERATIONS` at 15;
* `_DENOISE_STOP_THRESHOLD_PCT` at 0.2;
* `_DENOISE_BUFFER_MULTIPLIER` at 16;
* `_DENOISE_BATCH_CAP` at 128;
* `_SIGMA_X_FLOOR` at 1e-6;
* `_DENOISE_COST_PER_PLANE` at 0.015;
* the filter arguments `band_width=1` and `harmonics=True`;
* the subset floor of `pixels // 64`.

Two behaviors of `set_params` are worth noting.  The parameter `sigma_prox` is
removed from the keyword arguments and set again with `no_warning=True`, which
suppresses the base class warning about disabling automatic regularization.  See
`mbirjax/mace4d.py:171-178`.  The parameter `mace_prior_weight` is validated when
it is set, not when it is used.

### 2.2 Array shapes

| Array | Shape | Type and location |
|---|---|---|
| `sinogram` | `(num_views, num_det_rows, num_det_channels)` | NumPy, host |
| `weights` | Same as `sinogram`, all positive | NumPy on the host, or None |
| Per-frame sinogram | `(views_per_frame, rows, channels)` | On the frame's device for the whole run |
| `recon_shape` | `(nx, ny, nz)`, taken from `model_list[0]` | May differ from the shape in `ct_model` |
| `init_recon`, `W[k]`, `X[k]`, `xbar` | `(T, nx, ny, nz)` | `float32` NumPy, host |
| XY-t permuted array | `(nz, T, nx, ny)` | Contiguous NumPy |
| YZ-t permuted array | `(nx, T, ny, nz)` | Contiguous NumPy |
| XZ-t permuted array | `(ny, T, nx, nz)` | Contiguous NumPy |
| Flattened input to the denoiser | `(num_planes, d0 * d1, d2)` | On a device, batched |
| `plane_counts` | `[recon_shape[2], recon_shape[0], recon_shape[1]]` | Python list |
| `partition` | `(num_subsets, pixels_per_subset)` | On the denoiser's device |

The dictionary returned as the second value of `recon` has four keys.  These are
`recon_params`, `timing`, `notes`, and `model_params`.

## 3. Which dependencies already exist in mbirtorch

### 3.1 Available with no change

Twelve dependencies are available and have compatible signatures.

`TomographyModel.prox_map` at `mbirtorch/tomography_model.py:3251` has the same
signature as the mbirjax version, except for the default log file path.  It also
accepts tensors and `Shards` objects, which the mbirjax version does not.

`TomographyModel.recon` at `mbirtorch/tomography_model.py:3097` differs only in
lacking the argument `compute_prior_loss`, which MACE4D does not use.

`QGGMRFDenoiser` at `mbirtorch/denoising.py:153` has one additional optional
argument, `compile_mode='auto'`.

`QGGMRFDenoiser.estimate_image_noise_std` at `mbirtorch/denoising.py:233` uses
the same two-pass algorithm and the same stride of 5 million points.  It also
accepts `Shards` objects.

`QGGMRFDenoiser._get_sino_indicator` at `mbirtorch/denoising.py:268` is
unchanged.

`QGGMRFDenoiser._get_estimate_of_recon_std` at `mbirtorch/denoising.py:258` uses
the same formula.

`QGGMRFDenoiser.auto_set_sigma_y` at `mbirtorch/denoising.py:223` is unchanged.

`auto_set_sigma_x` and `auto_set_sigma_prox` are at
`mbirtorch/tomography_model.py:2140` and `:2149`.

`get_b_from_nbr_wts` is at `mbirtorch/qggmrf.py:22`.

`qggmrf_gradient_and_hessian_at_indices` at `mbirtorch/qggmrf.py:77` has two
additional optional arguments, `left_halo` and `right_halo`.

`copy_ct_model` at `mbirtorch/utilities.py:1001` accepts everything the mbirjax
version accepts, plus `new_translation_vectors`.  It also supports the
multiaxis and translation models.

The mechanism for registering new parameters works.  `set_params` at
`mbirtorch/parameter_handler.py:260` adds an unrecognized name as a new
parameter when `no_warning=True`, and its docstring at `:277-280` documents this.

### 3.2 Available with a small change to the call

Four dependencies need the call site adjusted.

`configure_devices` in mbirtorch has the signature
`(num_devices=1, devices=None, like=None)`.  In mbirjax it is `(devices=None)`.
Every call of the form `configure_devices([dev])` becomes
`configure_devices(devices=[dev])`.  The argument `like=` is a better way to pin
a denoiser to the same layout as a frame model.

`gen_set_of_pixel_partitions` at `mbirtorch/vcd_utils.py:159` names its third
argument `device` rather than `output_device`.  MACE4D calls it as
`(image_shape, [n], use_ror_mask=False)`, so that call works unchanged.

`QGGMRFDenoiser.__init__` has one extra optional argument, which is backward
compatible.

`recon` lacks `compute_prior_loss`, which MACE4D does not pass.

### 3.3 Requires new work

Five pieces do not exist in mbirtorch.

The method `_denoise_single_device` is missing.  In mbirtorch the single-device
sweep is written inline inside `denoise()` at `mbirtorch/denoising.py:406-436`.
MACE4D's batching depends on being able to call that sweep as a function.  This
is the largest gap.  Section 4 discusses it.

The function `_construct_time_frame_models` is missing.  Searching mbirtorch for
`time_frame`, `frames_per_rotation`, and `frame_overlap` returns nothing.  The
function uses only NumPy, `copy_ct_model`, `get_all_params`, and `set_params`,
all of which exist.  The port is straightforward.

The device-pool helpers are missing.  mbirtorch has no `gpu_devices`,
`cpu_devices`, `default_devices`, or `get_platform`.  The function
`_resolve_devices` must be rewritten using `torch.cuda.device_count()` and
`torch.backends.mps.is_available()`, and must return `torch.device` objects.
One behavior changes here.  JAX can present several virtual CPU devices, and
PyTorch presents one.  A request for a pool of CPU devices therefore returns one
device in mbirtorch, and the tests that exercise multiple devices on CPU cannot
work the same way.

The batch-size choice and the out-of-memory retry are missing.  The mbirjax code
reads `device.memory_stats()['bytes_limit']` and `['bytes_in_use']`.  The
PyTorch equivalent is `torch.cuda.mem_get_info(idx)`.  The mbirjax retry matches
the string `RESOURCE_EXHAUSTED` in the error text.  The PyTorch version should
catch `torch.cuda.OutOfMemoryError`, and also a `RuntimeError` whose text
contains "out of memory", and should call `torch.cuda.empty_cache()` before
retrying.  CPU and MPS report no memory limit, so the fixed fallback batch size
of 4 should be kept for them.  A better option is to compute the batch size with
`_memory_ledger.py`, which already models memory for the rest of the package.

The temporal filter has no PyTorch equivalent.  `torch.fft` does not provide the
type-I discrete cosine transform.  The recommendation is to keep this filter in
SciPy on the host, which is where MACE4D already runs it.  The alternative is to
build it from a fast Fourier transform of the even-symmetric extension.  No use
of `dct` appears anywhere in mbirtorch today.

### 3.4 Absent entirely

The module `mace4d.py`, the class `MACE4DModel`, the type alias
`MACE4DParamNames`, the page `usr_mace4d.rst`, and the file `test_mace4d.py` do
not exist in mbirtorch.  Searching mbirtorch for `mace`, ignoring case, returns
nothing.

### 3.5 Differences between the two `denoising.py` files

| Item | mbirjax, 663 lines | mbirtorch, 684 lines |
|---|---|---|
| Single-device sweep | `_denoise_single_device` at `:309`, using two nested `@jax.jit` closures, `lax.fori_loop`, `lax.while_loop`, and history arrays on the device | Written inline in `denoise()` at `:406-436`, using Python `for` loops, `maybe_compile(vcd_subset_denoiser)`, NumPy history arrays, and `torch.no_grad()` |
| Multiple-device sweep | `_denoise_sharded` at `:427` | `_denoise_sharded` at `:452`, with one thread pool per device |
| Regularization setup | Uses `auto_set_regularization_params` | The same, plus `_subsample_to_host` at `:100-152` and `:365-374`, which subsamples rows to avoid transferring the whole volume |
| Additional functions | None | `_volume_shape`, `_subsample_to_host`, a chunked `image_ell1`, `compile_mode`, and `_apply_device_policy(workload='denoise')` |
| `median_filter3d` | At `:559` | At `:598`, with the same signature |
| `_log_denoise_progress` | A method at `:303`, which MACE4D replaces with a function that does nothing | Not a method.  Logging is written inline at `:429-431`.  The equivalent is `set_params(verbose=0)`. |
| `vcd_subset_denoiser` | At `:511`, and functional | At `:35`, and it modifies its inputs in place with `flat_image.index_add_` and `flat_error_image.index_copy_` at `:77-81`.  It returns `(flat_image, flat_error_image, ell1, alpha)`.  The in-place updates prevent the use of `torch.func.vmap`. |

### 3.6 The existing test of `prox_map`

The file `mbirtorch/tests/test_prox_map.py` has 44 lines and one test,
`test_prox_map_pulls_toward_input(device)`.  It already exercises most of what
MACE4D needs.  It calls `configure_devices(devices=[device])`.  It seeds NumPy
with `np.random.seed(0)` before each call, so that the pixel partition is
reproducible.  It calls `prox_map` with `sigma_prox=1e-4`, `max_iterations=3`,
and `stop_threshold_change_pct=0.0`.  It then calls `prox_map` again with
`sigma_prox=1e3` and `do_initialization=False`, which checks that the cached
`prox_data` is reused correctly.

Three things it does not exercise are relevant to MACE4D.  It does not use more
than one frame.  It does not reuse a sinogram that is already on a device across
several calls.  It does not pass `logfile_path=None` or `print_logs=False`,
which MACE4D passes.

## 4. The batched denoiser

The remaining design decision is how to denoise many planes at once in PyTorch.
Section 1.10 describes what the JAX code does.  Two facts prevent a direct
translation.

The first fact is that `vcd_subset_denoiser` in mbirtorch modifies its inputs in
place, at `mbirtorch/denoising.py:77-81`.  `torch.func.vmap` does not allow a
function to modify its inputs.

The second fact is that each vmapped plane in JAX has its own stopping test.
That comes from `lax.while_loop`, which allows the number of iterations to
depend on the data.  PyTorch has no batched equivalent.  The mbirjax test
`test_batched_denoise_equals_serial` states this explicitly, asserting agreement
including planes that converge at different iterations.  See
`mbirjax/tests/test_mace4d.py:281-283`.

### 4.1 Option A: treat the planes as additional rows

Combine the plane index into the leading spatial dimension, so that all the
planes form one large image, and run the existing sweep once on that image.
This requires no use of `vmap` and no change to `vcd_subset_denoiser`.  It reuses
`maybe_compile` and every existing kernel.  This is the recommended option.

The correctness of Option A is not automatic.  The in-slice term of the qGGMRF
prior uses `recon_shape[:2]`, so a voxel at the edge of one plane must not take
a neighbor from the next plane.  See `mbirjax/qggmrf.py:90-93`.  The clamping
convention described in Section 3.6 of the comparison document makes an edge row
its own neighbor rather than wrapping to the next plane.  That changes the
computed values at exactly two rows per plane.

Two ways to handle this are available.  The first is to insert one row of zeros
between planes and exclude those rows from the pixel partition.  The second is to
measure the difference and confirm that it is below the iterated tolerance of
1e-3.  This measurement should be made before any other work, because the answer
determines the structure of the port.

### 4.2 Option B: rewrite the sweep as a pure function and use `torch.func.vmap`

Rewrite `vcd_subset_denoiser` to return new tensors instead of modifying its
inputs, then apply `torch.compile(torch.func.vmap(sweep))`.  This is the closest
match to the JAX code.

Option B has two costs.  It changes shared production code that both the
single-image denoiser and `_denoise_sharded` use, and it adds allocations to a
path that runs often for every caller.  It also still needs a scheme of
per-plane masks to reproduce the separate stopping tests.

### 4.3 Option C: loop over the planes in Python

Loop over the planes and denoise each one separately.  This is simple and
certainly correct.  It also gives up the speed that motivates the batching, and
the cost of launching kernels for each plane separately is large when `nz` is a
few hundred.

Option C is not a candidate for the final code.  It should nevertheless be
written first, as the reference against which Option A or Option B is checked.
That is exactly the role that `test_batched_denoise_equals_serial` plays in
mbirjax.

### 4.4 The choice about stopping tests

Whichever option is chosen, the treatment of the per-plane stopping test must be
decided and written down.  Two choices are available.  The first is to run every
plane for `_DENOISE_MAX_ITERATIONS` iterations and freeze each plane once it
converges, using a mask.  This gives the same values as mbirjax and does
unnecessary arithmetic on planes that converge early.  The second is to accept
that the result differs from mbirjax and to relax the corresponding test.
Record the choice with a `DIVERGENCE(mace4d batched convergence)` comment.

## 5. The plan

The plan has eight phases.  Each phase ends at a stated condition.

### Phase 0: refactor `_denoise_single_device` out of `denoise()`

Extract the code at `mbirtorch/denoising.py:406-436` into a method with this
signature.

```python
_denoise_single_device(flat_image, flat_error_image, partition, fm_constant,
                       qggmrf_params, image_shape, max_iterations,
                       stop_threshold_change_pct, first_iteration)
    -> (flat_image, nmae_hist, alpha_hist, num_iters)
```

Then have `denoise()` call it.  This is a refactoring with no change in
behavior.  It is worth doing regardless of which option in Section 4 is chosen,
because it gives MACE4D a function to call.

Phase 0 ends when `tests/test_denoiser.py` passes without modification.

### Phase 1: measure the batched denoiser options

Write the Option C loop as the reference result, using the same denoiser
configuration that MACE4D uses.  Write the Option A prototype, both with and
without rows of zeros between planes.  Measure two quantities for Option A
against the reference.  The first is the maximum relative difference.  The
second is the elapsed time, on a realistic number of planes.

If Option A meets the iterated tolerance of 1e-3, adopt it.  If it does not,
plan Option B as a separate change to `denoising.py` before starting Phase 3.
Record the measurement and its date in the module docstring.

Phase 1 ends when one option is chosen and its measured difference and speed are
recorded.  Phase 1 blocks Phase 3.

### Phase 2: the supporting functions

This phase has two parts, which are independent of Phase 1 and of each other.

Port `_construct_time_frame_models` into `mbirtorch/utilities.py`.  Keep the
angular step computed as `np.median(np.abs(np.diff(angles)))`, the discarding of
trailing views, and the suppression of printing after the first frame.  Test it
with the 24-view case from Section 1.1, checking for 5 frames and
`view_slices[1] == slice(4, 12)`.

Write the device-pool helper.  It may be private to `mace4d.py`, or it may sit
next to `TomographyModel._resolve_device`, which is preferable.  It must accept
`None`, the string `cpu`, the strings `gpu` and `cuda`, an integer count, a
sequence of integers, and a sequence of `torch.device` objects.  It must raise
for a count larger than the number of available devices, and for an unknown
platform string.  Mark the change in CPU behavior described in Section 3.3 with
a `DIVERGENCE(cpu device pool)` comment.

### Phase 3: the module and the consensus loop

Write `mbirtorch/mace4d.py` with the class `MACE4DModel(ParameterHandler)`.
This phase covers six items.

The constructor and parameter registration, including the special handling of
`sigma_prox` and the validation of `mace_prior_weight` when it is set.

The host-side helper functions.  These are `_normalize_prior_weights`,
`_assign_tasks`, `_validate_sinogram`, `_expected_init_shape`,
`_validate_init_recon`, `_load_cached_init`, `_run_settings`, and
`_write_run_info`.  All are pure NumPy and Python, and port with almost no
change.

The methods `set_device_pool` and `devices`, built on the Phase 2 helper.

The method `recon`.  This covers the choice of initial reconstruction, the host
arrays `W` and `X`, the in-place consensus update with one reused temporary
array, the stopping test, and the assembly of the returned dictionary.  Port the
in-place update as written.  The measurement behind it still applies, because
that code is pure NumPy.

The methods `_run_prox_task` and `_init_frame_task`, built on
`TomographyModel.prox_map` and `TomographyModel.recon`, calling
`configure_devices(devices=[dev])`.

The placement of the per-frame sinogram and weights.  Upload each frame's data
once and leave it on that device for the whole run, as mbirjax does at
`mbirjax/mace4d.py:455-462`.  In PyTorch this is
`torch.as_tensor(sinogram[view_slices[t]]).to(dev)`.  Consider passing the
device form directly to `prox_map`, which accepts it.  That would avoid a
transfer through host memory that the mbirjax code cannot avoid.

Phase 3 ends when a reconstruction with 3 frames, one device, and
`dejitter=False` runs, produces only finite values, and has shape
`(3,) + recon_shape`.

### Phase 4: the three prior agents

This phase has six items.

Write `_get_qggmrf_denoiser(shape, device)`, which caches one denoiser per pair
of shape and device in thread-local storage.  Pin each denoiser to its device
with `configure_devices(devices=[dev])` or with `like=`.  The mbirjax comment at
`mbirjax/mace4d.py:876-882` explains why pinning is necessary there.  Without
pinning, mbirjax spreads each denoiser over every visible GPU, and several such
denoisers running at once deadlock in NCCL.  The same class of problem can occur
in PyTorch, so keep the comment.

Port `_configure_denoiser` from `mbirjax/mace4d.py:874-910`.  Keep the reason
that `auto_set_regularization_params()` is not called.  That function would
compute its statistics from only the first plane, so the individual functions
are called on the whole stack instead.  Keep the `_SIGMA_X_FLOOR` guard.
Replace the replacement of `_log_denoise_progress` with `set_params(verbose=0)`.

Port `_denoise_constants`.  Keep the floor on the number of subsets, which is
`num_subsets = max(1, min(granularity[0], num_pixels // 64))`.  The reason is
that a subset with fewer than 64 pixels makes the line search compute zero
divided by zero.  Also keep the requirement that the partition be computed once
and cached.  The partition is drawn from the global NumPy random number
generator, so computing it again would change the order of the VCD subsets
between calls.

Write `_batched_hyperplane_denoise` according to the choice made in Phase 1.
Rewrite the out-of-memory retry for `torch.cuda.OutOfMemoryError`, and call
`torch.cuda.empty_cache()` before retrying.

Write `_denoiser_wrapper(x, permute_vector, sigma, device, config_token=None)`.
It permutes with `np.ascontiguousarray(np.transpose(...))`, reconfigures the
denoiser only when the token changes, and permutes back with
`np.argsort(permute_vector)`.

Write `_estimate_global_sigma`.  Reshape to `(T * nx, ny, nz)` and keep the
error for a value that is not finite or not positive.

Phase 4 ends when the adapted test of the batched denoiser passes and a
reconstruction using all four agents runs.

### Phase 5: the temporal filter

Port `_dejitter_4d_dct` unchanged, using SciPy.  It runs on the host and does
not involve PyTorch.  Keep the block processing controlled by `chunk_size`,
including the explicit `del block, C`.

Write the wrapper `MACE4DModel._dejitter` with `period=frames_per_rotation`,
`harmonics=True`, `band_width=1`, and `dtype=np.float32`, and with printing
controlled by `dejitter_verbose`.

Check whether SciPy is already a required dependency.  If it is not, add it to
`environment.yml` and `pyproject.toml`.

### Phase 6: task parallelism and logging

Create one `ThreadPoolExecutor` per device, each with one worker thread.
Section 1.8 gives the two reasons.

Write `_run_task_set(executors, tasks, t0)`, which returns the results and the
log rows.

Write the three log files described in Section 1.9.

Check one requirement that is specific to PyTorch.  `maybe_compile` caches
compiled functions per device, and Dynamo reads its recompilation limit per
thread.  See `mbirtorch/projectors.py:52-107`.  Every MACE4D worker thread must
therefore raise its own recompilation budget.  Verify that no thread exceeds the
budget.  Exceeding it produces no message and was measured to cost a factor of 5
to 11.

Phase 6 ends when the multiple-device test finds log rows from both devices in
`task_log.csv`.

### Phase 7: tests

Rewrite `tests/test_mace4d.py` in pytest style, since the mbirjax version uses
`unittest`.  Use the existing `device` fixture.

Keep the smooth sinogram used as test data.  The mbirjax comment at
`mbirjax/tests/test_mace4d.py:43-46` gives the reason.  A random sinogram
reconstructs to a volume with extreme values, on which the qGGMRF line search
computes zero divided by zero.

| Group | What it checks |
|---|---|
| Prior weights | `0.5` gives `[0.5, 1/6, 1/6, 1/6]`.  `[0.1,0.2,0.3]` gives `[0.4,0.1,0.2,0.3]`.  The values `1.5`, `-0.1`, `[0.5,0.5,0.5]`, and `[0.1,0.2]` each raise. |
| Task assignment | `_assign_tasks(25, [192,65,65], 4)` returns 25 frame assignments and 3 orientation assignments, all in the range 0 to 3, and places the three denoiser tasks on three different devices.  `_assign_tasks(5, [8,6,7], 1)` returns all zeros. |
| Device resolution | The inputs `None`, `1`, an explicit list, `[0]`, and `cpu` all work.  A count above the number of devices raises, and `tpu` raises.  `set_device_pool(1)` gives a pool of length 1.  Adjust for the CPU behavior in Section 3.3. |
| Construction | `num_frames` is 5, `len(model_list)` is 5, and `view_slices[1]` is `slice(4, 12)`.  A sinogram of the wrong shape raises, and weights of the wrong shape raise, both before any computation. |
| Parameters | `dejitter` defaults to True.  `set_params(rho_mann=0.25, dejitter=False)` is readable afterward.  `mace_prior_weight=1.5` raises when set. |
| A complete reconstruction | Use `np.random.seed(0)`, `num_frames=3`, `dejitter=False`, and one device.  Use `dejitter=False` because a filter with period 6 applied to 3 frames removes the entire temporal spectrum.  Check the shape, that all values are finite, that the three log files exist, that the keys of the returned dictionary are `model_params`, `notes`, `recon_params`, and `timing`, that `timing` has one entry, that the recorded iteration count is 1, that `weights` is recorded as `unit (weights=None)`, and that `init_recon.npy` is written and then reused.  Also check that `stop_threshold_change_pct=1e9` stops after one iteration, that a supplied `init_recon` is recorded as provided by the caller, and that an `init_recon` of the wrong shape raises. |
| Multiple devices | Skip if fewer than 2 devices are available.  Use `num_frames=4`.  Check that `task_log.csv` contains rows from device 0 and from device 1.  Without that check the test does not exercise concurrency at all. |
| The initial-reconstruction cache | `_load_cached_init` returns None without a message when the file is absent.  It returns None with exactly one warning containing the word `invalid` when the file has the wrong shape.  It returns a `float32` array when the file is valid. |
| The batched denoiser | Use 6 planes of shape `(8,10,12)`.  Call `_configure_denoiser` with `sigma=0.05` and `image_for_stats=x.reshape(-1,10,12)`.  Check that the batched result and the loop over planes agree to `atol=1e-5`, and that the result differs from the input. |

Add one stored-array comparison against mbirjax.  Produce a small MACE4D
reconstruction in the mbirjax environment through `tests/generate_goldens.py`,
and compare under `pytest.mark.goldens` at the iterated tolerance of 1e-3.  This
is the strongest available check that the port computes the same thing.

Note that `pin_device_count` sets `MBIRTORCH_NUM_DEVICES=1` for the whole test
session.  The multiple-device test must override it.

### Phase 8: documentation and registration

Set `__all__ = ['MACE4DModel']` in `mbirtorch/mace4d.py`.  Do not port
`MACE4DParamNames`, because the `Literal` type annotations were not ported to
mbirtorch.  Mark this with a `REPLACED(ParamNames)` comment.

Add `MACE4DModel` to `__all__` in `mbirtorch/__init__.py`, and add entries to
`_LAZY_MODULES` and `_LAZY_NAMES`.  Lazy loading is worthwhile here because the
module imports SciPy.

Port `docs/source/usr_mace4d.rst`.  Keep the description of the time windows and
the trade between signal-to-noise ratio and temporal resolution.  Keep the
explanation of the temporal filter.  Keep the citation, and add the corresponding
entry to `refs.bib`.  Then add `autoclass:: mbirtorch.MACE4DModel
:show-inheritance:` and `automethod::` directives for `set_params`, `recon`, and
`set_device_pool`.

Add the page to `usr_api.rst` in two places, which are the list of links and the
hidden `toctree`, placing it between `usr_denoising` and `usr_preprocess`.  Add a
section to `usr_api_overview.rst` with an autosummary entry for
`MACE4DModel.recon`.

If Phase 3 is complete but Phase 4 is not, put the page in
`docs/source/_pending/` and add a row to that directory's README.

Consider writing a demonstration script, `demo/demo_10_mace4d.py`.  There is
nothing to port for this.  The directory `mbirjax/experiments/MACE_4D_CT/`
contains only a stale `__pycache__` directory.

## 6. Risks

| Risk | Severity | How to handle it |
|---|---|---|
| The batched denoiser has no direct PyTorch equivalent | High | Phase 1 measures the options before any other work.  Option C is the reference result. |
| Per-plane stopping tests cannot be expressed on a batch | Medium | Run every plane to the iteration limit and freeze converged planes with a mask.  This costs arithmetic, not accuracy.  Record it as a divergence. |
| Extracting `_denoise_single_device` changes the single-image denoiser | Medium | The change is a refactoring.  Use `tests/test_denoiser.py` as the check, and make this change on its own. |
| A MACE4D worker thread exceeds the recompilation budget | Medium | Raise the budget in each worker thread.  Check for recompilation warnings in the multiple-device test.  The cost of missing this is a factor of 5 to 11, with no message. |
| The CPU device pool has one device in PyTorch | Low | Record it as a divergence.  The multiple-device test should skip rather than simulate a second CPU device. |
| Host memory holds eight full four-dimensional arrays plus two more | Medium | This is also the mbirjax behavior.  Keep every array `float32` and keep the in-place consensus update.  Consider predicting this peak with `_memory_ledger.py`, so that an over-large problem raises an error rather than being killed by the operating system. |
| Regenerating the pixel partition changes the VCD subset order | Low | Compute the partition once per configuration and cache it, as mbirjax does.  Seed the random number generator in the tests. |
| PyTorch has no type-I discrete cosine transform | Low | Keep the temporal filter in SciPy on the host.  Confirm SciPy is a declared dependency. |
| Numerical differences from mbirjax accumulate over 10 iterations | Medium | Compare against stored arrays at the iterated tolerance of 1e-3.  If the comparison fails, compare one agent at a time rather than the whole reconstruction. |

## 7. Size and order of the work

| Phase | Content | Approximate size |
|---|---|---|
| 0 | Extract `_denoise_single_device` | 40 lines, a refactoring |
| 1 | Measure the batched denoiser options | Small, and the main source of uncertainty |
| 2 | `_construct_time_frame_models` and the device-pool helper | 160 lines |
| 3 | The module, the consensus loop, and the data-fit agent | 450 lines |
| 4 | The three prior agents | 250 lines |
| 5 | The temporal filter | 90 lines, almost unchanged |
| 6 | Task parallelism and logging | 120 lines |
| 7 | Tests | 320 lines |
| 8 | Documentation and registration | 80 lines of reStructuredText |

The total is 1200 to 1400 lines.  For comparison, the mbirjax code is 1039 lines
of module, 305 lines of tests, and 55 lines of documentation.

Do the phases in numerical order.  Phase 0 comes first because Phase 1 needs the
function it creates.  Phase 2 can be done at any time, including in parallel
with Phase 1.  Phase 1 governs the schedule, because its result determines the
structure of Phase 4.
