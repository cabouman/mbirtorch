import torch


def _default_device(device=None):
    """The device name to compute on: the given one, or the repo's automatic choice with MPS replaced by the CPU,
    since the hsnt solvers accumulate in float64, which MPS does not support."""
    if device is not None and device != 'auto':
        return str(torch.device(device))
    from ..tomography_model import _resolve_device
    resolved = _resolve_device('auto')
    return 'cpu' if resolved.type == 'mps' else str(resolved)
