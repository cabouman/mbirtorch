"""Shared driver for the scan -> sinogram preprocessing pipeline.

The per-stage transforms in :mod:`mbirtorch.preprocess.utilities` are pure device-tensor *kernels* (the
math only).  This module owns the **single** copy of the batching + host<->device transfer +
in-place-fill scaffolding used by ``compute_sino_transmission`` / ``downsample_view_data`` /
``correct_det_rotation`` and the fused ``scan_to_sino``.

The driver supports two modes: a single-device sequential view-batch loop, and a multi-device
view-sharded mode where contiguous view shards run concurrently, one per device.  The mode follows
the device list the caller passes in.  A public preprocessing function gets that list from
``permitted_devices`` below, which by default names every visible CUDA device.  In both modes the
host output is pre-allocated once and each batch's result is written directly into its view-slice,
so the host footprint is the input + the single output (~2x) rather than input + per-batch gather +
concatenate destination (~3x).
"""
import numpy as np
import torch

from .. import _memory_ledger
from .. import _sharding
from ..tomography_model import _resolve_device


def permitted_devices(devices=None):
    """The devices a preprocessing function spreads its views over.

    The default is every visible CUDA device, capped by ``MBIRTORCH_NUM_DEVICES`` when that variable
    is set.  A machine with no visible CUDA device gets its default device instead, so the returned
    list always names at least one device.  The size of the data does not affect the choice.  One
    view batch is bounded work at any problem size, so no memory estimate is needed here.

    The environment variable is process-wide.  It is how a test suite or a nightly run fixes the
    device count, so that results and memory use do not depend on the host.  Preprocessing chooses
    its own default count, but a variable the caller set is still the caller's, so it caps that
    default here.

    Args:
        devices (sequence or None): an explicit list of devices, or None for the default described
            above.  An explicit list is returned unchanged, and the environment variable does not
            cap it.

    Returns:
        list: the devices to pass to :func:`map_view_batches`.
    """
    if devices is not None:
        return list(devices)
    num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    pinned = _memory_ledger.pinned_device_count()
    if pinned is not None:
        num_visible = min(num_visible, pinned)
    if num_visible == 0:
        return [_resolve_device('auto')]
    return [f'cuda:{i}' for i in range(num_visible)]


def _stage_batch(batch, device):
    """Stage one host batch onto the device: contiguous (a flipped/strided source view, e.g. the
    NSI reader's np.flip, cannot be wrapped as a tensor) and floating point (integer scans, e.g.
    the Zeiss reader's uint16, promote to float32 as the kernels' arithmetic expects).  The copy
    is one batch, not the full array."""
    batch = np.ascontiguousarray(batch)
    if not np.issubdtype(batch.dtype, np.floating):
        batch = batch.astype(np.float32)
    return torch.as_tensor(batch, device=device)


def _fill_view_batches(array, kernel, output, batch_size, device, lo, hi, desc=None):
    """Run ``kernel`` over views ``[lo, hi)`` of ``array`` in ``batch_size`` chunks on ``device``,
    writing each batch's host result directly into ``output[j:...]`` (a pre-allocated host array).

    Host->device per batch, device->host per batch.  Writing in place (rather than collecting
    per-batch results and concatenating) keeps the host footprint at the input + the single output
    array, with only one batch's result transiently live.
    """
    import tqdm
    steps = range(lo, hi, batch_size)
    if desc is not None:
        steps = tqdm.tqdm(steps, desc=desc)
    with torch.no_grad():
        for j in steps:
            end = min(j + batch_size, hi)
            batch = _stage_batch(array[j:end], device)
            output[j:end] = kernel(batch).cpu().numpy()


def map_view_batches(array, kernel, batch_size, desc=None, devices=None):
    """Apply a per-batch device kernel across the leading (view) axis.

    Single device (``devices`` is None or length 1): a sequential view-batch loop -- each contiguous
    batch of ``batch_size`` views is moved to the device, passed through ``kernel`` (a pure
    device-tensor -> device-tensor transform), and written back into the host output.  This bounds
    device memory to ``batch_size`` views.

    Multiple devices: the views are split into contiguous, in-order shards (one per device) and each
    shard is processed in its own thread on its own device.  Per-view kernels (no cross-view
    reduction) make this embarrassingly parallel with no cross-device communication.  Each worker
    writes its disjoint view-slice of the shared host output.

    ``kernel`` should close over HOST constants (NumPy) and move them to each batch's device
    (``torch.as_tensor(const, device=batch.device)``), NOT tensors already committed to one device.

    The host output is pre-allocated once (its shape/dtype probed from the first batch, since a kernel
    may change the trailing detector dims, e.g. downsampling) and filled in place, so the host footprint
    is input + output (~2x) regardless of device count.

    Args:
        array (numpy array or tensor): data batched along axis 0 (views).
        kernel (callable): ``device_batch -> device_batch``; per-view, no host transfer inside.
        batch_size (int): number of views per on-device batch.
        desc (str or None, optional): tqdm label (single-device path only).
        devices (sequence or None): devices to spread the views over.  None means a single device
            (the default device) -- sharding is opt-in by passing several devices.

    Returns:
        numpy.ndarray: the per-batch kernel outputs assembled along axis 0 (view order).
    """
    if devices is None or len(list(devices)) == 0:
        devices = [_resolve_device('auto')]
    else:
        devices = [torch.device(d) for d in devices]
    num_views = array.shape[0]

    # Probe the kernel's output shape/dtype on the first batch so a SINGLE host output array can be
    # pre-allocated and every batch can write its view-slice in place.  This avoids the per-batch
    # result list + the final concatenate, which together hold extra full-size copies of the result
    # on the host.  Writing in place bounds the host footprint to input + output (~2x).
    probe_hi = min(batch_size, num_views)
    with torch.no_grad():
        probe = kernel(_stage_batch(array[0:probe_hi], devices[0])).cpu().numpy()
    output = np.empty((num_views,) + probe.shape[1:], dtype=probe.dtype)
    output[0:probe_hi] = probe
    del probe

    if len(devices) <= 1:
        _fill_view_batches(array, kernel, output, batch_size, devices[0], probe_hi, num_views,
                           desc=desc)
        return output

    # Multiple devices: contiguous, in-order view shards, one per device, each filled by its own
    # thread (torch releases the GIL during CUDA work and transfers, so the devices genuinely
    # overlap).  Workers write disjoint view-slices of the shared host output.
    view_ranges = np.array_split(np.arange(num_views), len(devices))

    def worker(i, device):
        rng = view_ranges[i]
        if len(rng) == 0:
            return
        lo = max(int(rng[0]), probe_hi)  # views [0:probe_hi] are already filled by the probe
        hi = int(rng[-1]) + 1
        if lo >= hi:
            return
        _fill_view_batches(array, kernel, output, batch_size, device, lo, hi)

    _sharding.run_per_device(devices, worker)
    return output
