"""Phase 1 gate: adjointness <Ax, y> == <x, A'y>, and the differentiable
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
    model = mbirtorch.ParallelBeamModel(sino_shape, angles, device=device)
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


def test_differentiable_wrapper_gradients(device):
    torch.manual_seed(0)
    sino_shape = (24, 16, 16)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles, device=device)
    recon_shape = model.get_params('recon_shape')

    volume = torch.rand(tuple(recon_shape), device=model.torch_device,
                        requires_grad=True)
    y = torch.rand(sino_shape, device=model.torch_device)

    sino = mbirtorch.forward_project_differentiable(model, volume)
    loss = torch.sum(sino * y)
    loss.backward()

    # d/dv <A v, y> = A' y (zero outside the ROR mask).
    expected = model.back_project(y, output_sharded=True)
    grad = volume.grad
    rel_max = float((grad - expected).abs().max() / expected.abs().max())
    assert rel_max < 1e-5, rel_max


def test_torch_projector_module(device):
    sino_shape = (12, 8, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles, device=device)
    projector = mbirtorch.TorchProjector(model)
    recon_shape = model.get_params('recon_shape')
    volume = torch.rand(tuple(recon_shape), device=model.torch_device)
    sino = projector(volume)
    assert tuple(sino.shape) == sino_shape
    back = projector.adjoint(sino)
    assert tuple(back.shape) == tuple(recon_shape)
