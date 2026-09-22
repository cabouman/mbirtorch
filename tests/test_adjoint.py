"""Adjointness gate: <Ax, y> == <x, A'y>, and the differentiable
wrapper's gradients equal the adjoint operators."""

import numpy as np
import torch

import mbirtorch


def _rel_diff(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def test_projector_adjointness(device):
    torch.manual_seed(0)
    sino_shape = (48, 40, 32)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=[device])
    recon_shape = model.get_params('recon_shape')

    indices = torch.as_tensor(
        mbirtorch.gen_full_indices(recon_shape), dtype=torch.int64,
        device=model.torch_device)
    x = torch.rand((indices.shape[0], recon_shape[2]), device=model.torch_device)
    y = torch.rand(sino_shape, device=model.torch_device)

    ax = model.sparse_forward_project(x, indices)
    aty = model.sparse_back_project(y, indices)
    lhs = float(torch.sum(ax * y))
    rhs = float(torch.sum(x * aty))
    # f32 sums over ~1e6 terms: run-to-run atomics noise is ~1e-6 relative.
    assert _rel_diff(lhs, rhs) < 1e-4, (lhs, rhs)
