.. _MigrationDocs:

======================
Migrating from MBIRJAX
======================

MBIRTorch replaces MBIRJAX, which is now legacy software.  The two packages have the same
interface, so most scripts run after the import line is changed.  This page lists the
differences that a script is likely to hit.  See :ref:`InstallationDocs` to install MBIRTorch.


Imports
-------

Change the package name in the import line.  ``import mbirjax`` becomes ``import mbirtorch``.
The submodule paths are unchanged, so ``mbirjax.preprocess`` becomes ``mbirtorch.preprocess``.


Arrays
------

Both packages accept a numpy array wherever a sinogram, a weight array, or a volume is expected,
and both return numpy arrays by default.  A script that reads its data with numpy needs no
change.  Where MBIRJAX accepts a JAX array, MBIRTorch accepts a torch tensor.


Renamed methods
---------------

The direct reconstruction methods are renamed so that every reconstruction method starts with
``recon``.  The arguments are unchanged.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - MBIRJAX
     - MBIRTorch
   * - ``model.direct_recon(sinogram)``
     - ``model.recon_direct(sinogram)``
   * - ``model.fbp_recon(sinogram)``
     - ``model.recon_fbp(sinogram)``
   * - ``model.fdk_recon(sinogram)``
     - ``model.recon_fdk(sinogram)``
   * - ``model.split_sino_recon(sinogram)``
     - ``model.recon_split_sino(sinogram)``
   * - ``preprocess.mar.recon_plastic_metal(model, sinogram, weights)``
     - ``model.recon_plastic_metal(sinogram, weights)``


Devices
-------

MBIRTorch chooses its devices on its own.  It prefers CUDA, then Apple's Metal, then the CPU.
To choose the devices yourself, call ``model.configure_devices``, for example
``model.configure_devices(devices=['cpu'])``.  The MBIRJAX parameter ``use_gpu`` does not exist.
See :ref:`usr_multi_gpu` for reconstruction on several GPUs.


Removed names
-------------

MBIRJAX helper functions specific to JAX, and its debugging and plotting utilities, have no
counterpart in MBIRTorch.  A name that fails to import is one of these.  The features that
MBIRTorch adds are described in :doc:`overview`.
