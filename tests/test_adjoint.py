"""Adjointness gate: <Ax, y> == <x, A'y>, and the differentiable
wrapper's gradients equal the adjoint operators."""

import numpy as np
import pytest
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


def test_differentiable_wrapper_gradients(device):
    torch.manual_seed(0)
    sino_shape = (24, 16, 16)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=[device])
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
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=[device])
    projector = mbirtorch.TorchProjector(model)
    recon_shape = model.get_params('recon_shape')
    volume = torch.rand(tuple(recon_shape), device=model.torch_device)
    sino = projector(volume)
    assert tuple(sino.shape) == sino_shape
    back = projector.adjoint(sino)
    assert tuple(back.shape) == tuple(recon_shape)


def test_differentiable_wrappers_refuse_the_divided_form():
    """The wrappers already refuse a MULTI-DEVICE MODEL, but that check says
    nothing about the array: a volume or sinogram divided across devices by
    some other model reaches a single-device model unexamined and dies on a
    missing '.to'.  Both entries refuse it by name instead.  Two 'virtual' CPU
    devices build the divided form, so this runs everywhere."""
    from mbirtorch._sharding import Placement, Shards

    sino_shape = (12, 8, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=['cpu'])
    recon_shape = tuple(model.get_params('recon_shape'))

    def divided(array, axis):
        placement = Placement(['cpu', 'cpu'], axis=axis,
                              axis_len=array.shape[axis])
        pieces = [array.narrow(axis, start, end - start).clone()
                  for _, (start, end) in placement.shard_ranges()]
        return Shards(pieces, placement)

    volume = divided(torch.rand(recon_shape), -1)
    sinogram = divided(torch.rand(sino_shape), 0)

    for name, call in (
            ('forward_project_differentiable',
             lambda: mbirtorch.forward_project_differentiable(model, volume)),
            ('back_project_differentiable',
             lambda: mbirtorch.back_project_differentiable(model, sinogram)),
            ('forward_project_differentiable',
             lambda: mbirtorch.TorchProjector(model)(volume)),
            ('back_project_differentiable',
             lambda: mbirtorch.TorchProjector(model).adjoint(sinogram))):
        with pytest.raises(TypeError) as refusal:
            call()
        message = str(refusal.value)
        assert name in message, message
        assert 'divided device form' in message, message
        assert 'shards.gather()' in message, message
