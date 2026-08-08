"""Shared driver for the scan -> sinogram preprocessing pipeline.

The per-stage transforms in :mod:`mbirtorch.preprocess.utilities` are pure device-tensor *kernels* (the
math only).  This module owns the **single** copy of the batching + host<->device transfer +
in-place-fill scaffolding used by ``compute_sino_transmission`` / ``downsample_view_data`` /
``correct_det_rotation`` and the fused ``scan_to_sino``.

The host output is pre-allocated once and each batch's result is written directly into its view-slice,
so the host footprint is the input + the single output (~2x) rather than input + per-batch gather +
concatenate destination (~3x).
"""
import numpy as np
import torch

from ..tomography_model import _resolve_device


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
            # ascontiguousarray: a flipped/strided source view (e.g. the NSI reader's np.flip)
            # cannot be wrapped as a tensor; the copy is one batch, not the full array.
            batch = torch.as_tensor(np.ascontiguousarray(array[j:end]), device=device)
            output[j:end] = kernel(batch).cpu().numpy()


def map_view_batches(array, kernel, batch_size, desc=None, devices=None):
    """Apply a per-batch device kernel across the leading (view) axis.

    A sequential view-batch loop: each contiguous batch of ``batch_size`` views is moved to the
    device, passed through ``kernel`` (a pure device-tensor -> device-tensor transform), and written
    back into the host output.  This bounds device memory to ``batch_size`` views.

    ``kernel`` should close over HOST constants (NumPy) and move them to each batch's device
    (``torch.as_tensor(const, device=batch.device)``), NOT tensors already committed to one device.

    The host output is pre-allocated once (its shape/dtype probed from the first batch, since a kernel
    may change the trailing detector dims, e.g. downsampling) and filled in place, so the host footprint
    is input + output (~2x).

    Args:
        array (numpy array or tensor): data batched along axis 0 (views).
        kernel (callable): ``device_batch -> device_batch``; per-view, no host transfer inside.
        batch_size (int): number of views per on-device batch.
        desc (str or None, optional): tqdm label.
        devices (sequence or None): accepted for interface compatibility; the views run on a single
            device (the first entry, or the default device when None).

    Returns:
        numpy.ndarray: the per-batch kernel outputs assembled along axis 0 (view order).
    """
    if devices is None or len(list(devices)) == 0:
        device = _resolve_device('auto')
    else:
        device = torch.device(list(devices)[0])
    num_views = array.shape[0]

    # Probe the kernel's output shape/dtype on the first batch so a SINGLE host output array can be
    # pre-allocated and every batch can write its view-slice in place.  This avoids the per-batch
    # result list + the final concatenate, which together hold extra full-size copies of the result
    # on the host.  Writing in place bounds the host footprint to input + output (~2x).
    probe_hi = min(batch_size, num_views)
    with torch.no_grad():
        probe = kernel(torch.as_tensor(np.ascontiguousarray(array[0:probe_hi]),
                                       device=device)).cpu().numpy()
    output = np.empty((num_views,) + probe.shape[1:], dtype=probe.dtype)
    output[0:probe_hi] = probe
    del probe

    _fill_view_batches(array, kernel, output, batch_size, device, probe_hi, num_views, desc=desc)
    return output
