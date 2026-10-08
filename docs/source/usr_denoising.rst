.. _DenoisingDocs:

=========
Denoising
=========

The denoiser is the identity geometry: the forward model is :math:`y = x + w` with
:math:`w` white Gaussian noise, so the reconstruction is the maximum a posteriori (MAP) estimate
of the volume under the qGGMRF prior :math:`h`,

.. math::

    H(y) = \arg\min_x \left\{ \frac{1}{2 \sigma_{noise}^2}\|y - x\|^{2} + h(x) \right\}.

A MAP denoiser is handy in a Plug-and-Play loop, where it is the image model.  For denoising
on its own, other denoisers may do better.  The noise level is estimated from the volume unless
``sigma_noise`` is given; a larger value smooths more.  The ``sharpness`` parameter, default 0,
adjusts the result the same way as in a reconstruction.

.. code-block:: python

    denoiser = mbirtorch.QGGMRFDenoiser(noisy.shape)
    denoised, denoise_dict = denoiser.denoise(noisy)

Constructor
-----------

.. autoclass:: mbirtorch.QGGMRFDenoiser
   :show-inheritance:

Denoising
---------

.. automethod:: mbirtorch.QGGMRFDenoiser.denoise

.. automethod:: mbirtorch.QGGMRFDenoiser.denoise_stack
