"""Once-per-process availability probe for the hand-written (Triton) kernel
paths, and the per-kernel value self-checks.

The kernel paths are an OPTIMIZATION, so an environment that cannot run them
must fall back to the torch.compile paths silently -- but not
silently-unexplained.  The probe returns (usable, reason), and the reason
string is the record of WHY a node is not using the custom kernels (the same
question a benchmark asks first).

The probe compiles and runs a trivial kernel end to end rather than trusting
version checks: it is the toolchain, not the version number, that breaks.  It
is exception-safe by construction -- every failure becomes a (False, reason)
pair -- because a broken optional path must never break a caller who only
wanted to know whether it was there.

The probe answers "can this node run a triton kernel at all".  The second gate
is per KERNEL and per DEVICE: :func:`cone_back_kernel_usable`,
:func:`cone_forward_kernel_usable`, :func:`parallel_back_kernel_usable` and
:func:`parallel_forward_kernel_usable` each run one kernel-vs-torch-body
comparison at a tiny shape on the device that will run it (milliseconds) and
fall back on a tolerance breach.  This is the guard the kernel design puts on
the correct axis -- probe the hardware you are on, never trust a vendor list
-- so the kernels may default on wherever both gates pass, on any
architecture, and a miscompiling toolchain is caught even on a swept one.

Capability is not policy.  Every gate here answers only whether a kernel
REPRODUCES its torch body on this device; whether it is actually selected is
decided in the geometry's ``_view_batch_bodies``, where a kernel awaiting its
composed performance gate stays behind an opt-in environment variable (the
names live here, beside the kill switch, so one module carries the whole
switchboard).
"""

import os

import torch

# Kill switch: set to 1 to force the fallback paths (a bisection handle, and
# the escape hatch when a toolchain compiles the probe but miscompiles the
# real kernels).
DISABLE_ENV_VAR = 'MBIRTORCH_DISABLE_TRITON'

# RETIRED opt-in switches.  The selection protocol is that each kernel is
# opt-in through its own environment variable until ITS OWN composed
# performance gate passes, and then defaults on wherever the availability
# gates pass.  All four kernels have now passed their gates (cone 2026-08-07,
# parallel later the same day), so every switch below is retired and no
# selection reads them; they stay defined so any script still exporting one
# is harmless.  MBIRTORCH_DISABLE_TRITON is the kill switch for all kernels.
ENABLE_FWD_ENV_VAR = 'MBIRTORCH_ENABLE_TRITON_FWD'
ENABLE_PBACK_ENV_VAR = 'MBIRTORCH_ENABLE_TRITON_PBACK'
ENABLE_PFWD_ENV_VAR = 'MBIRTORCH_ENABLE_TRITON_PFWD'

# Relative tolerance of the value self-checks, at the design's Hessian-path
# figure: the cone kernels reproduce the torch bodies only up to the documented
# rounding carve-out (the sqrt-vs-atan2 divisor and the floor-vs-round tie),
# and the self-check runs coeff_power 2, where that gap is squared.  The
# parallel kernels carry no such carve-out (no vertical fan, so no divisor and
# no center rounding) and differ from their bodies by float summation order
# alone; they share the figure rather than a tighter one of their own, because
# what a self-check must catch is a miscompile, not a ULP.
SELF_CHECK_REL_TOL = 1e-4

# Module-level cache of the single probe result: compiling even a trivial
# kernel costs real time, and the answer cannot change within a process.
_PROBE_RESULT = None

# Per-device caches of the four self-checks, keyed by device string: each
# check builds a model, compiles a kernel, and runs both bodies, so it must
# happen once per process -- and its answer is a property of the DEVICE and
# its toolchain, not of the calling model.
_CONE_BACK_RESULTS = {}
_CONE_FWD_RESULTS = {}
_PARALLEL_BACK_RESULTS = {}
_PARALLEL_FWD_RESULTS = {}

# Re-entrancy flag: a self-check builds its own tiny model, whose
# create_projectors asks this same module which bodies to use.  While ANY
# check runs, every answer must be "the torch body" -- otherwise the probe
# model would recurse into the check it is part of.  One flag covers all four
# checks because the recursion it blocks is not per geometry: the flag is read
# by every gate, so a parallel check's model cannot trip a cone gate either.
_SELF_CHECK_ACTIVE = False


def triton_available():
    """(usable, reason): whether the hand-written Triton kernel paths may be
    used, and why not when they cannot.  Cached per process.

    Returns:
        (bool, str): True with a short description of what was probed, or
        False with the reason -- the kill switch, no CUDA platform, no triton
        import, or a trivial kernel that failed to compile, failed to run, or
        returned the wrong value.
    """
    global _PROBE_RESULT
    if _PROBE_RESULT is None:
        _PROBE_RESULT = _probe_triton()
    return _PROBE_RESULT


def _probe_triton():
    """Run the probe once (see :func:`triton_available`); never raises."""
    if os.environ.get(DISABLE_ENV_VAR, '0') == '1':
        result = (False, f'disabled by {DISABLE_ENV_VAR}=1')
    else:
        try:
            if not torch.cuda.is_available():
                result = (False, 'not a CUDA platform (torch.cuda is '
                                 'unavailable)')
            else:
                import triton
                import triton.language as tl

                @triton.jit
                def probe_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
                    offsets = tl.arange(0, BLOCK)
                    tl.store(out_ptr + offsets,
                             tl.load(x_ptr + offsets) + tl.load(y_ptr + offsets))

                block = 32
                x = torch.ones(block, dtype=torch.float32, device='cuda')
                y = torch.full((block,), 2.0, dtype=torch.float32, device='cuda')
                out = torch.empty_like(x)
                probe_kernel[(1,)](x, y, out, BLOCK=block)
                if bool(torch.all(out == 3.0)):
                    result = (True, f'available (triton {triton.__version__}, '
                                    f'{torch.cuda.get_device_name(0)})')
                else:
                    result = (False, 'probe kernel returned a wrong value')
        except Exception as e:                                    # noqa: BLE001
            result = (False, f'probe kernel failed to compile/run: {e}')
    return result


def cone_back_kernel_usable(model):
    """(usable, reason): whether the Triton cone back body may replace the
    torch one for ``model``, on ``model.torch_device``.

    Two gates in order: the process-wide triton probe above, then a first-use
    VALUE self-check on this device -- a tiny cone problem projected through
    both bodies at coeff_power 1 and 2, compared at
    :data:`SELF_CHECK_REL_TOL`.  Cached per device string and exception-safe:
    any failure is a (False, reason) pair, never a raise into a caller who
    only asked whether the fast path was available.

    Under a multi-device configuration the check runs on device 0 alone (the
    bodies are shared across devices, and a node's devices share an
    architecture and a toolchain).
    """
    return _kernel_usable(model, _CONE_BACK_RESULTS, _cone_back_self_check)


def cone_forward_kernel_usable(model):
    """(usable, reason): whether the Triton cone forward body may replace the
    torch one for ``model``, on ``model.torch_device``.

    The same two gates in the same order as :func:`cone_back_kernel_usable`,
    with the forward body's own value comparison -- unbanded and banded, since
    the forward carries its slice band in the VALUES argument rather than in a
    band_slices keyword.  Cached per device string and exception-safe.

    This answers CAPABILITY only.  Whether the forward kernel is actually
    selected is policy, and until its composed performance gate passes that
    selection applies this gate alone (the opt-in era ended with the
    composed-gate pass).
    process never pays for the check).
    """
    return _kernel_usable(model, _CONE_FWD_RESULTS, _cone_forward_self_check)


def parallel_back_kernel_usable(model):
    """(usable, reason): whether the Triton parallel back body may replace the
    torch one for ``model``, on ``model.torch_device``.

    The same two gates in the same order as :func:`cone_back_kernel_usable`,
    with the parallel back body's own value comparison at coefficient powers 1
    and 2 -- run both unbanded and on an interior ROW band, which is how a
    row-aligned geometry bands (the band rides in the sinogram's row axis, not
    in a band keyword).  Cached per device string and exception-safe.

    This answers CAPABILITY only; selection is policy, and while the parallel
    kernels are opt-in (see :data:`ENABLE_PBACK_ENV_VAR`) a process that has
    not opted in never calls this and never pays for the check.
    """
    return _kernel_usable(model, _PARALLEL_BACK_RESULTS,
                          _parallel_back_self_check)


def parallel_forward_kernel_usable(model):
    """(usable, reason): whether the Triton parallel forward body may replace
    the torch one for ``model``, on ``model.torch_device``.

    The same two gates in the same order, with the parallel forward body's own
    value comparison -- unbanded and on a row band, which the forward carries
    in the COLUMN count of its values (rows track slices, so a slice band is a
    row band).  Cached per device string and exception-safe; capability only,
    as above.
    """
    return _kernel_usable(model, _PARALLEL_FWD_RESULTS,
                          _parallel_forward_self_check)


def _kernel_usable(model, cache, self_check):
    """The shape all four gates above share: the re-entrancy guard, the
    process-wide probe, then this device's cached value self-check."""
    if _SELF_CHECK_ACTIVE:
        result = (False, 'kernel self-check in progress (its own probe model '
                         'uses the torch bodies)')
    else:
        probe_usable, probe_reason = triton_available()
        if not probe_usable:
            result = (False, probe_reason)
        else:
            device_key = str(model.torch_device)
            if device_key not in cache:
                cache[device_key] = self_check(device_key)
            result = cache[device_key]
    return result


def _cone_self_check_cell(device_key):
    """The tiny cone problem both cone self-checks run on: (model,
    pixel_indices, view_params, body kwargs).

    Small enough to cost milliseconds and shaped to reach every branch of the
    kernels: four views (so the back kernel's view reduction runs), a real cone
    angle, and one pixel dropped from the full index set so the pixel count is
    not a multiple of any tile size (the last block is padded).
    """
    import numpy as np

    from .cone_beam import ConeBeamModel
    from .vcd_utils import gen_full_indices

    cell = (4, 10, 10)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                          source_iso_dist=2 * cell[2], 
                          compile_mode='off')
    model.configure_devices(devices=[device_key])
    device = model.torch_device
    pixel_indices = torch.as_tensor(
        gen_full_indices(model.get_params('recon_shape')),
        dtype=torch.int64, device=device)[:-1]
    view_params = torch.as_tensor(model.get_params('view_params_array'),
                                  dtype=torch.float32, device=device)
    return model, pixel_indices, view_params, model._view_batch_args()


def _parallel_self_check_cell(device_key):
    """The tiny parallel problem both parallel self-checks run on: (model,
    pixel_indices, view_params, body kwargs) -- the cone cell's twin, at the
    same size and with the same padded last pixel block, over a half turn of
    real angles so the projected footprint varies from view to view (W_p_c and
    weight_scale are per-view under parallel beam, so a single angle would
    exercise one value of each)."""
    import numpy as np

    from .parallel_beam import ParallelBeamModel
    from .vcd_utils import gen_full_indices

    cell = (4, 10, 10)
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = ParallelBeamModel(cell, angles, 
                              compile_mode='off')
    model.configure_devices(devices=[device_key])
    device = model.torch_device
    pixel_indices = torch.as_tensor(
        gen_full_indices(model.get_params('recon_shape')),
        dtype=torch.int64, device=device)[:-1]
    view_params = torch.as_tensor(model.get_params('angles'),
                                  dtype=torch.float32, device=device)
    return model, pixel_indices, view_params, model._view_batch_args()


def _rel_diff(kernel_out, reference):
    """Max relative difference of a kernel output against its torch body."""
    return (float((kernel_out - reference).abs().max())
            / max(float(reference.abs().max()), 1e-12))


def _cone_back_self_check(device_key):
    """Run the cone back kernel-vs-torch-body comparison once on one device
    (see :func:`cone_back_kernel_usable`); never raises."""
    global _SELF_CHECK_ACTIVE
    _SELF_CHECK_ACTIVE = True
    try:
        from .cone_beam import _cone_back_view_batch
        from .triton_cone import _cone_back_view_batch_triton

        model, pixel_indices, view_params, args = _cone_self_check_cell(
            device_key)
        # A private generator: the seeded recon gates depend on the global RNG
        # streams, and an availability check must not advance them.
        generator = torch.Generator().manual_seed(0)
        sinogram = torch.rand(tuple(model.get_params('sinogram_shape')),
                              generator=generator).to(model.torch_device)

        worst_rel = 0.0
        for coeff_power in (1, 2):
            reference = _cone_back_view_batch(sinogram, pixel_indices,
                                              view_params,
                                              coeff_power=coeff_power, **args)
            kernel_out = _cone_back_view_batch_triton(
                sinogram, pixel_indices, view_params,
                coeff_power=coeff_power, **args)
            worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('cone back', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'cone back self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_ACTIVE = False
    return result


def _cone_forward_self_check(device_key):
    """Run the cone forward kernel-vs-torch-body comparison once on one device
    (see :func:`cone_forward_kernel_usable`); never raises."""
    global _SELF_CHECK_ACTIVE
    _SELF_CHECK_ACTIVE = True
    try:
        from .cone_beam import _cone_forward_view_batch
        from .triton_cone import _cone_forward_view_batch_triton

        model, pixel_indices, view_params, args = _cone_self_check_cell(
            device_key)
        num_slices = int(args['num_slices'])
        generator = torch.Generator().manual_seed(0)
        values = torch.rand((int(pixel_indices.shape[0]), num_slices),
                            generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # The whole volume, then an interior band: the band exercises the
        # slice_start seam, whose z anchor stays on the full slice count (the
        # forward carries its band in the VALUES, so nothing else says which
        # slices these are).
        interior = max(1, num_slices // 3)
        bands = ((0, num_slices), (interior, num_slices - 2 * interior))
        for slice_start, band_len in bands:
            band_values = values[:, slice_start:slice_start + band_len]
            reference = _cone_forward_view_batch(band_values, pixel_indices,
                                                 view_params,
                                                 slice_start=slice_start, **args)
            kernel_out = _cone_forward_view_batch_triton(
                band_values, pixel_indices, view_params,
                slice_start=slice_start, **args)
            worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('cone forward', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'cone forward self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_ACTIVE = False
    return result


def _parallel_back_self_check(device_key):
    """Run the parallel back kernel-vs-torch-body comparison once on one device
    (see :func:`parallel_back_kernel_usable`); never raises."""
    global _SELF_CHECK_ACTIVE
    _SELF_CHECK_ACTIVE = True
    try:
        from .parallel_beam import _parallel_back_view_batch
        from .triton_parallel import _parallel_back_view_batch_triton

        model, pixel_indices, view_params, args = _parallel_self_check_cell(
            device_key)
        # A private generator: the seeded recon gates depend on the global RNG
        # streams, and an availability check must not advance them.
        generator = torch.Generator().manual_seed(0)
        sinogram = torch.rand(tuple(model.get_params('sinogram_shape')),
                              generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # Every row, then an interior row band: a row-aligned geometry bands in
        # the SINOGRAM's row axis (rows track slices), so slicing the input is
        # the whole of the banded seam and the output band comes back with it.
        num_rows = int(sinogram.shape[1])
        interior = max(1, num_rows // 3)
        bands = ((0, num_rows), (interior, num_rows - 2 * interior))
        for coeff_power in (1, 2):
            for row_start, band_rows in bands:
                band = sinogram[:, row_start:row_start + band_rows]
                reference = _parallel_back_view_batch(
                    band, pixel_indices, view_params,
                    coeff_power=coeff_power, **args)
                kernel_out = _parallel_back_view_batch_triton(
                    band, pixel_indices, view_params,
                    coeff_power=coeff_power, **args)
                worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('parallel back', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'parallel back self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_ACTIVE = False
    return result


def _parallel_forward_self_check(device_key):
    """Run the parallel forward kernel-vs-torch-body comparison once on one
    device (see :func:`parallel_forward_kernel_usable`); never raises."""
    global _SELF_CHECK_ACTIVE
    _SELF_CHECK_ACTIVE = True
    try:
        from .parallel_beam import _parallel_forward_view_batch
        from .triton_parallel import _parallel_forward_view_batch_triton

        model, pixel_indices, view_params, args = _parallel_self_check_cell(
            device_key)
        num_slices = int(model.get_params('recon_shape')[2])
        generator = torch.Generator().manual_seed(0)
        values = torch.rand((int(pixel_indices.shape[0]), num_slices),
                            generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # The whole volume, then an interior band: the forward carries its band
        # in the COLUMN count of the values, and each band produces the
        # matching detector rows rather than a partial of the whole sinogram
        # (the row-aligned form's difference from the cone forward's).
        interior = max(1, num_slices // 3)
        bands = ((0, num_slices), (interior, num_slices - 2 * interior))
        for slice_start, band_len in bands:
            band_values = values[:, slice_start:slice_start + band_len]
            reference = _parallel_forward_view_batch(band_values,
                                                     pixel_indices,
                                                     view_params, **args)
            kernel_out = _parallel_forward_view_batch_triton(
                band_values, pixel_indices, view_params, **args)
            worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('parallel forward', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'parallel forward self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_ACTIVE = False
    return result


def _self_check_verdict(name, device_key, worst_rel):
    """(usable, reason) from a self-check's worst relative difference."""
    if worst_rel <= SELF_CHECK_REL_TOL:
        result = (True, f'{name} kernel matches the torch body on '
                        f'{device_key} (rel {worst_rel:.1e} <= '
                        f'{SELF_CHECK_REL_TOL:.0e})')
    else:
        result = (False, f'{name} kernel differs from the torch body on '
                         f'{device_key} (rel {worst_rel:.1e} > '
                         f'{SELF_CHECK_REL_TOL:.0e})')
    return result


def _reset_probe_cache():
    """Drop the cached probe result so the next call probes again (tests:
    the kill switch is read INSIDE the probe, so flipping it takes effect
    only across a reset)."""
    global _PROBE_RESULT
    _PROBE_RESULT = None


def _reset_self_check_cache():
    """Drop the cached per-device self-check results (tests: the kill switch
    and the probe cache are read inside the gates above)."""
    _CONE_BACK_RESULTS.clear()
    _CONE_FWD_RESULTS.clear()
    _PARALLEL_BACK_RESULTS.clear()
    _PARALLEL_FWD_RESULTS.clear()
