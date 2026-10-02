======
Theory
======

This page gives an overview of the theory behind MBIRTorch.  For the detailed documentation of the
forward model and the algorithms, download this
`zip file <https://www.datadepot.rcac.purdue.edu/bouman/data/tomography_geometry.zip>`_.


Model-Based Iterative Reconstruction
------------------------------------

Model-based iterative reconstruction (MBIR) computes the image as the solution of an optimization
problem,

.. math::

    {\hat x} = \arg \min_x \left\{ f(x) + h(x) \right\} ,

where :math:`f(x)` is the forward model term and :math:`h(x)` is the prior model term.  The forward
model term measures how well the image explains the sinogram.  The prior model term measures how
well the image matches what images are expected to look like.  The vectorized coordinate descent
algorithm described below solves this problem.

Both terms have low-level parameters, named below.  In normal use, these parameters are set
automatically from two meta-parameters, ``sharpness`` and ``snr_db``, as described on the
:doc:`Advanced Features <advanced_features>` page.  Setting a low-level parameter directly turns the
automatic setting off.


Forward model
~~~~~~~~~~~~~

The forward model term has the form

.. math::

    f(x) = \frac{1}{2 \sigma_y^2} \Vert y - Ax \Vert_\Lambda^2 ,

where :math:`\Vert y \Vert_\Lambda^2 = y^T \Lambda y`.  The symbols are:

- :math:`y` is the sinogram, the ``sinogram`` argument of ``recon``.
- :math:`x` is the image to be reconstructed.
- :math:`A` is the linear projection operator of the imaging geometry.
- :math:`\Lambda` is a diagonal matrix of sinogram weights, the ``weights`` argument of ``recon``.
  Each weight is the assumed inverse noise variance of one sinogram entry, relative to the others.
- :math:`\sigma_y` is the assumed standard deviation of the measurement noise, the parameter
  ``sigma_y``.  It is set automatically from ``snr_db``, the assumed signal to noise ratio of the
  sinogram in dB.

The function ``gen_weights(sinogram, weight_type)`` computes the weights for four noise models:

- ``'unweighted'``: :math:`\Lambda = 1`, the same weight for every entry.
- ``'transmission'``: :math:`\Lambda = e^{-y}`, the noise model for transmission CT at a fixed dose.
- ``'transmission_root'``: :math:`\Lambda = e^{-y/2}`, a weaker weighting that is often used with
  transmission data to make the noise more uniform across the image.
- ``'emission'``: :math:`\Lambda = 1/(|y| + 0.1)`, the noise model for emission CT.

The two transmission models assume that the sinogram is in units of negative log attenuation.


Prior model
~~~~~~~~~~~

The default prior is the qGGMRF prior,

.. math::

    h(x) = \sum_{ \{s,r\} \in {\cal P}} b_{s,r} \, \rho ( x_s - x_r) ,

where :math:`{\cal P}` is the set of neighboring voxel pairs.  Each voxel has six neighbors, four in
its slice and one in each adjacent slice.  The potential function is

.. math::

    \rho ( \Delta ) = \frac{|\Delta |^p }{ p \sigma_x^p } \left( \frac{\left| \frac{\Delta }{ T \sigma_x } \right|^{q-p}}{1 + \left| \frac{\Delta }{ T \sigma_x } \right|^{q-p}} \right) .

The symbols are:

- :math:`\sigma_x` is the regularization parameter, ``sigma_x``.  Larger values give sharper
  images.  It is set automatically from ``sharpness``, as :math:`\sigma_x = 0.2 \cdot 2^{\text{sharpness}} \cdot \hat\sigma`,
  where :math:`\hat\sigma` is an estimate of the standard deviation of the image.
- :math:`b_{s,r}` weights each neighbor pair.  The parameter ``qggmrf_nbr_wts`` gives the relative
  weights along the row, column, and slice directions.
- :math:`p` and :math:`q` shape the potential function, ``p`` and ``q``.  The defaults are
  :math:`p = 2.0` and :math:`q = 1.2`.  For small differences the potential is quadratic, and
  for large differences it grows as :math:`|\Delta|^q`, which preserves edges.
- :math:`T` sets where the potential changes from one behavior to the other, ``T``.  The
  default is 1.0.


Proximal map prior
~~~~~~~~~~~~~~~~~~

The Plug-and-Play method :cite:`venkatakrishnan2013plug,sreehari2016plug` replaces the prior with a
denoiser, which can be any algorithm, including a neural network.  It alternates between the
denoiser and a proximal map of the forward model.  The method ``prox_map(prox_input, sinogram)``
computes that proximal map by solving

.. math::

    {\hat x} = \arg \min_x \left\{ f(x) + \frac{1}{2\sigma_p^2} \Vert x - v \Vert^2 \right\} ,

where :math:`v` is the ``prox_input`` argument and :math:`\sigma_p` is the parameter
``sigma_prox``.  When ``sigma_prox`` is not given, it is set automatically from ``sharpness`` in
the same way as :math:`\sigma_x`.


Vectorized Coordinate Descent
-----------------------------

MBIRTorch solves the optimization problem with multi-granular vectorized coordinate descent
(MG-VCD) :cite:`2024CV4SciencePoster`.  The algorithm converges quickly and reliably, and it is
implemented in PyTorch, so it runs on CPUs and GPUs.  It does not need a preconditioner designed
for a particular geometry, as gradient-based methods usually do.  That is what allows one
implementation to support all of the geometries in MBIRTorch.
