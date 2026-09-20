"""Check the differentiable projectors in mbirtorch.autograd.

Each test uses a small parallel beam model so the whole file runs in seconds.
"""

import numpy as np
import torch

import mbirtorch


SINO_SHAPE = (16, 32, 32)


def _model(device):
    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(devices=[device])
    return model


def _rel_err(value, reference):
    return abs(value - reference) / max(abs(value), abs(reference), 1e-30)


def _max_rel_err(tensor, reference):
    return float((tensor - reference).abs().max() / reference.abs().max())


def test_adjointness(device):
    """<A x, y> equals <x, A' y> for the differentiable operators."""
    torch.manual_seed(0)
    model = _model(device)
    recon_shape = model.get_params('recon_shape')

    x = torch.rand(tuple(recon_shape), device=model.torch_device)
    y = torch.rand(SINO_SHAPE, device=model.torch_device)

    ax = mbirtorch.forward_project_differentiable(model, x)
    aty = mbirtorch.back_project_differentiable(model, y)
    lhs = float(torch.sum(ax * y))
    rhs = float(torch.sum(x * aty))

    print(f'adjointness relative error on {device}: {_rel_err(lhs, rhs):.3e}')
    assert _rel_err(lhs, rhs) < 1e-5, (lhs, rhs)


def test_forward_projector_gradient(device):
    """d/dx sum(A x * y) equals A' y."""
    torch.manual_seed(0)
    model = _model(device)
    recon_shape = model.get_params('recon_shape')

    x = torch.rand(tuple(recon_shape), device=model.torch_device,
                   requires_grad=True)
    y = torch.rand(SINO_SHAPE, device=model.torch_device)

    torch.sum(mbirtorch.forward_project_differentiable(model, x) * y).backward()
    expected = mbirtorch.back_project_differentiable(model, y)

    error = _max_rel_err(x.grad, expected)
    print(f'forward gradient max relative error on {device}: {error:.3e}')
    assert error < 1e-5, error


def test_back_projector_gradient(device):
    """d/dy sum(A' y * x) equals A x."""
    torch.manual_seed(0)
    model = _model(device)
    recon_shape = model.get_params('recon_shape')

    x = torch.rand(tuple(recon_shape), device=model.torch_device)
    y = torch.rand(SINO_SHAPE, device=model.torch_device, requires_grad=True)

    torch.sum(mbirtorch.back_project_differentiable(model, y) * x).backward()
    expected = mbirtorch.forward_project_differentiable(model, x)

    error = _max_rel_err(y.grad, expected)
    print(f'back gradient max relative error on {device}: {error:.3e}')
    assert error < 1e-5, error


def test_torch_projector_module(device):
    """A forward pass and a backward pass through TorchProjector leave a
    finite, nonzero gradient on the input volume."""
    torch.manual_seed(0)
    model = _model(device)
    recon_shape = model.get_params('recon_shape')
    projector = mbirtorch.TorchProjector(model)

    x = torch.rand(tuple(recon_shape), device=model.torch_device,
                   requires_grad=True)
    sinogram = projector(x)
    assert tuple(sinogram.shape) == SINO_SHAPE

    loss = torch.sum(sinogram ** 2)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().max()) > 0.0

    assert tuple(projector.adjoint(
        torch.rand(SINO_SHAPE, device=model.torch_device)).shape) == tuple(recon_shape)


def test_gradcheck_tiny_model():
    """torch.autograd.gradcheck on a tiny model, CPU only."""
    torch.manual_seed(0)
    sino_shape = (8, 8, 4)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=['cpu'])
    recon_shape = model.get_params('recon_shape')

    x = torch.rand(tuple(recon_shape), dtype=torch.float64, requires_grad=True)

    # The projector computes in float32, so gradcheck needs a step large
    # enough that the finite difference survives float32 rounding.
    assert torch.autograd.gradcheck(
        lambda v: mbirtorch.forward_project_differentiable(model, v),
        (x,), eps=1e-3, atol=1e-3, rtol=1e-2)
