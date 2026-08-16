.. _AutogradDocs:

=========================
Differentiable Projectors
=========================

MBIRTorch exposes its forward and back projectors as differentiable PyTorch operations, so
the physics operator can be inserted into a deep-learning pipeline like any other layer.

The forward and back projectors are an exact adjoint pair by construction, so each is the
correct autograd backward of the other, and no autodiff through the kernel internals is
needed.

Note that these operators run on a single device: configure the model with
``model.configure_devices(1)`` before using them in training.

Functional Interface
--------------------

.. autofunction:: mbirtorch.autograd.forward_project_differentiable

.. autofunction:: mbirtorch.autograd.back_project_differentiable


Module Interface
----------------

``TorchProjector`` wraps the differentiable forward projector as an ``nn.Module``, so a
learned-prior pipeline can hold the physics operator as a submodule.

.. autoclass:: mbirtorch.autograd.TorchProjector
   :show-inheritance:
   :members:
