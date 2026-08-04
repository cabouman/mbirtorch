"""Phase 0 smoke test: the package imports and torch runs a tensor op on every
available backend (CPU always; MPS or CUDA when present)."""

import torch

import mbirtorch


def test_import():
    assert mbirtorch.__version__


def _devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def test_tensor_op_on_each_backend():
    for dev in _devices():
        x = torch.arange(8, dtype=torch.float32, device=dev)
        y = (x * 2).sum().item()
        assert y == 56.0, f"backend {dev}"
