"""Differentiable projectors, for coupling to deep-learning pipelines.

The forward and back projectors are exact adjoint pairs by construction
(tests/test_adjoint.py verifies <Ax, y> == <x, A'y>), so each is the correct
autograd backward of the other -- no autodiff through the kernel internals is
needed.  ``forward_project_differentiable`` exposes gradient flow through the
volume; ``TorchProjector`` wraps it as an ``nn.Module`` in the LEAP style so a
learned-prior pipeline can insert the physics operator like any layer.
"""

import torch



class _ForwardProjectFunction(torch.autograd.Function):
    """A: voxel cylinders at fixed indices -> sinogram; backward is A'."""

    @staticmethod
    def forward(ctx, voxel_values, model, pixel_indices):
        ctx.model = model
        ctx.pixel_indices = pixel_indices
        return model.sparse_forward_project(voxel_values, pixel_indices)

    @staticmethod
    def backward(ctx, grad_sinogram):
        grad_values = ctx.model.sparse_back_project(
            grad_sinogram.contiguous(), ctx.pixel_indices)
        return grad_values, None, None


class _BackProjectFunction(torch.autograd.Function):
    """A': sinogram -> voxel cylinders at fixed indices; backward is A."""

    @staticmethod
    def forward(ctx, sinogram, model, pixel_indices):
        ctx.model = model
        ctx.pixel_indices = pixel_indices
        return model.sparse_back_project(sinogram, pixel_indices)

    @staticmethod
    def backward(ctx, grad_values):
        grad_sinogram = ctx.model.sparse_forward_project(
            grad_values.contiguous(), ctx.pixel_indices)
        return grad_sinogram, None, None


def forward_project_differentiable(model, volume):
    """Differentiable full-volume forward projection.

    The input may live on any device and dtype: it is moved to the model's
    device as float32 through a DIFFERENTIABLE ``.to`` (autograd then returns
    the gradient on the input's own device and dtype -- without this, a CPU or
    float64 leaf against a CUDA/MPS model would fail at forward or, worse, only
    at backward).  The ROR indices come from the model's per-shape
    cache rather than being rebuilt and re-uploaded per call.

    Args:
        model: a TomographyModel (e.g. ParallelBeamModel).
        volume: (num_rows, num_cols, num_slices) tensor; may require grad.

    Returns:
        (num_views, num_det_rows, num_det_channels) tensor with gradient flow
        back to ``volume`` (the gather into cylinders is differentiable, and
        the projector pair supplies the operator's gradient).
    """
    volume = volume.to(device=model.torch_device, dtype=torch.float32)
    indices = model.full_indices_device()
    voxel_values = volume.reshape(-1, volume.shape[-1])[indices]
    return _ForwardProjectFunction.apply(voxel_values, model, indices)


def back_project_differentiable(model, sinogram):
    """Differentiable full-volume back projection (adjoint of the above; the
    same device/dtype normalization and index caching apply)."""
    recon_shape = model.get_params('recon_shape')
    sinogram = sinogram.to(device=model.torch_device, dtype=torch.float32)
    indices = model.full_indices_device()
    cylinders = _BackProjectFunction.apply(sinogram, model, indices)
    volume = torch.zeros(recon_shape[0] * recon_shape[1], cylinders.shape[-1],
                         dtype=cylinders.dtype, device=cylinders.device)
    volume = volume.index_put((indices,), cylinders)
    return volume.reshape(tuple(recon_shape[:2]) + (cylinders.shape[-1],))


class TorchProjector(torch.nn.Module):
    """The forward operator as an ``nn.Module`` (LEAP-style).

    forward(volume) -> sinogram, differentiable in ``volume``.  The adjoint is
    exposed as ``.adjoint(sinogram)``.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, volume):
        return forward_project_differentiable(self.model, volume)

    def adjoint(self, sinogram):
        return back_project_differentiable(self.model, sinogram)
