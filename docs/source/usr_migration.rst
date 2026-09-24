.. _MigrationDocs:

======================
Migrating from MBIRJAX
======================

MBIRTorch replaces MBIRJAX, which is now legacy software.  The two packages were written to
have the same interface, so most scripts run after changing the import line.  This page lists
the places where the two packages differ, so that you can find and fix them in your own code.

Installation
------------

Install MBIRTorch from PyPI into a Python 3.11 or later environment::

    pip install mbirtorch

The ``torch`` dependency installs automatically, and on a Linux machine with an NVIDIA GPU the
default torch wheel includes CUDA support.  See :ref:`InstallationDocs` for installation from
source and for the conda environment used by the test suite and the documentation build.

Imports
-------

Replace the package name in the import line.  ``import mbirjax`` becomes ``import mbirtorch``,
and ``import mbirjax as mj`` becomes ``import mbirtorch as mt``.

The submodule paths are unchanged.  ``mbirjax.preprocess``, ``mbirjax.hsnt``, ``mbirjax.vcls``
and ``mbirjax.mace4d`` become ``mbirtorch.preprocess``, ``mbirtorch.hsnt``, ``mbirtorch.vcls``
and ``mbirtorch.mace4d``.  The loaders keep their module names as well, so
``mbirjax.preprocess.nsi`` becomes ``mbirtorch.preprocess.nsi``.

Arrays and devices
------------------

Both packages accept a numpy array wherever a sinogram, a weight array, or a volume is asked
for, and both return a numpy array by default.  A script that reads its data with numpy and
displays the result with the slice viewer needs no change here.

The other accepted input forms differ.  MBIRJAX accepts a JAX array.  MBIRTorch accepts a
torch tensor, on the host or on a device, and it also accepts the sharded form returned by
``prepare_sino_for_devices``.  Passing the sharded form to several reconstructions of one
large sinogram moves the data to the devices once.

The device form of the output differs in the same way.  With ``output_sharded=True``, MBIRJAX
returns a JAX array sharded across the model's devices.  MBIRTorch returns a torch tensor when
the model uses one device, and a ``Shards`` container holding one tensor per device when the
model uses several.

MBIRTorch chooses its devices without being asked.  It prefers CUDA, then MPS, then the CPU.
On a machine with several CUDA devices, the first reconstruction on a model chooses how many
of them to use, based on measured speed at that problem size and on whether the layout fits in
memory.  Call ``model.configure_devices(num_devices=n)`` to fix the count instead, or set the
environment variable ``MBIRTORCH_NUM_DEVICES`` to fix it for a whole process.  MBIRJAX has no
counterpart to that environment variable, and its own ``MBIRJAX_NUM_CPU_DEVICES`` has no
counterpart in MBIRTorch.

API differences
---------------

Each row below is a change to make in code that calls MBIRJAX.

.. list-table::
   :header-rows: 1
   :widths: 36 36 28

   * - MBIRJAX
     - MBIRTorch
     - Note
   * - ``model.direct_recon(sinogram)``
     - ``model.recon_direct(sinogram)``
     - Renamed.  The arguments are the same.
   * - ``model.fbp_recon(sinogram)``
     - ``model.recon_fbp(sinogram)``
     - Renamed.  Parallel beam and multi-axis parallel beam.
   * - ``model.fdk_recon(sinogram)``
     - ``model.recon_fdk(sinogram)``
     - Renamed.  Cone beam and translation.
   * - ``model.split_sino_recon(sino)``
     - ``model.recon_split_sino(sino)``
     - Renamed.  The arguments are the same.
   * - ``preprocess.mar.recon_plastic_metal(model, sino, weights)``
     - ``model.recon_plastic_metal(sino, weights)``
     - A function became a method.  The model is no longer the first
       argument, and there is no ``output_sharded`` argument.
   * - ``model.recon(..., compute_prior_loss=True)``
     - ``model.recon(...)``
     - The ``compute_prior_loss`` argument is gone from ``recon`` and from
       ``initialize_recon``.  Remove it from the call.
   * - ``model.set_params(use_gpu='none')``
     - ``model.configure_devices(devices=['cpu'])``
     - ``use_gpu`` is not a parameter in MBIRTorch.
   * - ``model.configure_devices(devices)``
     - ``model.configure_devices(num_devices=1, devices=None, like=None)``
     - The device count, an explicit device list, and another model to
       match are now separate arguments.
   * - ``model.set_view_parameters(view_params)``
     - ``model.set_params(angles=new_angles)``
     - No separate method.  Set the view parameter array through
       ``set_params``, using the name reported by
       ``get_params('view_params_name')``.
   * - ``model.vcd_recon(...)``
     - none
     - Not part of the public interface in MBIRTorch.  Call ``recon``.
   * - ``preprocess.nsi.convert_nsi_to_mbirjax_params``
     - ``preprocess.nsi.convert_nsi_to_mbirtorch_params``
     - The package name appears in the function name.
   * - ``preprocess.zeiss.convert_zeiss_to_mbirjax_params``
     - ``preprocess.zeiss.convert_zeiss_to_mbirtorch_params``
     - The same rename applies in ``preprocess.zeiss_tct``.
   * - ``logfile_path='~/.mbirjax/logs/recon.log'``
     - ``logfile_path='~/.mbirtorch/logs/recon.log'``
     - The default log path follows the package name.
   * - ``ParallelBeamModel(sinogram_shape, angles)``
     - ``ParallelBeamModel(sinogram_shape, angles, view_batch_size=None, compile_mode='auto')``
     - Two optional arguments are added.  Existing calls need no change.
       The same two arguments are added to ``ConeBeamModel``,
       ``MultiAxisParallelModel``, ``TranslationModel`` and
       ``TomographyModel``.
   * - ``hsnt.dehydrate(data, num_materials=3, safety_factor=2)``
     - ``hsnt.l2_dehydrate(data, num_materials=3, safety_factor=2)``
     - Renamed; same arguments and results.  ``hsnt.dehydrate`` now fits
       the Poisson likelihood of the counts and takes different arguments,
       described in :ref:`HSNTDocs`.
   * - ``hsnt.hyper_denoise(data, num_materials=3, safety_factor=2)``
     - ``hsnt.l2_hyper_denoise(data, num_materials=3, safety_factor=2)``
     - Renamed, as for ``dehydrate``.

These names exist in MBIRJAX and have no counterpart in MBIRTorch:
``get_platform``, ``get_device_platform``, ``memory_report``,
``display_translation_vectors``, ``debug_plot_partitions``, ``debug_plot_indices``,
``plot_granularity_and_loss``, ``make_figure_folder``, ``download_and_extract_tar``,
``gen_pixel_partition_grid`` and ``gen_pixel_partition_blue_noise``.

The names that MBIRTorch adds are listed under `Features new in MBIRTorch`_ below.

A side by side example
----------------------

The following script makes a phantom, projects it to get a sinogram, reconstructs the
sinogram, and displays the result.  The MBIRJAX version reads as follows.

.. code-block:: python

    import numpy as np
    import mbirjax

    phantom, sinogram, params = mbirjax.generate_demo_data(
        model_type='parallel', object_type='shepp-logan',
        num_views=128, num_det_rows=128, num_det_channels=128)
    angles = params['angles']

    ct_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)
    ct_model.set_params(sharpness=1.0)
    recon, recon_dict = ct_model.recon(sinogram)

    nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
    print(f'Normalized RMS error: {nrmse:.3f}')

    mbirjax.slice_viewer(phantom, recon, data_dicts=[None, recon_dict],
                         title='Phantom (left) and MBIR reconstruction (right)')

The MBIRTorch version differs only in the package name.

.. code-block:: python

    import numpy as np
    import mbirtorch

    phantom, sinogram, params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='shepp-logan',
        num_views=128, num_det_rows=128, num_det_channels=128)
    angles = params['angles']

    ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
    ct_model.set_params(sharpness=1.0)
    recon, recon_dict = ct_model.recon(sinogram)

    nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
    print(f'Normalized RMS error: {nrmse:.3f}')

    mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict],
                           title='Phantom (left) and MBIR reconstruction (right)')

Features new in MBIRTorch
-------------------------

MBIRTorch adds several things that MBIRJAX does not have.  The forward and back projectors are
available as differentiable PyTorch operations, so the physics operator can be used as a layer
in a training pipeline, described in :ref:`AutogradDocs`.  A geometry viewer draws the source,
the detector, and the reconstruction volume for a model, through ``geometry_viewer`` and the
``GeometryScene`` and ``GeometryFigure`` classes.  The functions ``recon_simple_parallel`` and
``recon_simple_cone`` reconstruct from a sinogram and the projection angles in one call.  The
``mbirtorch.mace`` module provides a general multi-agent consensus equilibrium framework, which
in MBIRJAX exists only as the 4D reconstruction model ``MACE4DModel``.  The preprocessing
subpackage adds ``preprocess.geometry_calibration`` for estimating detector offset and
detector rotation from the data.  ``QGGMRFDenoiser`` adds ``denoise_stack`` for denoising a
stack of volumes in batches, and ``TomographyModel`` adds ``project_points``, ``recon_slice_z``
and ``nearest_recon_slice``.  Finally, the compiled projector kernels are cached under
``~/.mbirtorch``, so compiled code is reused by later runs, and ``clear_cache`` empties that
cache.  Both packages can spread one reconstruction across several GPUs, and
:ref:`usr_multi_gpu` describes how MBIRTorch chooses the number of devices.
