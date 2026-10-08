.. _TomographyModelDocs:


================
Tomography Model
================

The ``TomographyModel`` provides the basic interface for all specific geometries for tomographic projection
and reconstruction.

Constructor
-----------

.. autoclass:: mbirtorch.TomographyModel


Reconstruction and Projection
-----------------------------

.. automethod:: mbirtorch.TomographyModel.recon

.. automethod:: mbirtorch.TomographyModel.recon_direct

.. automethod:: mbirtorch.TomographyModel.recon_split_sino

.. automethod:: mbirtorch.TomographyModel.recon_plastic_metal

.. automethod:: mbirtorch.TomographyModel.forward_project

.. automethod:: mbirtorch.TomographyModel.back_project

``prox_map`` is for Plug-and-Play loops that alternate a reconstruction step with a denoiser.

.. automethod:: mbirtorch.TomographyModel.prox_map


Parameter Handling
------------------

.. automethod:: mbirtorch.TomographyModel.set_params

.. automethod:: mbirtorch.TomographyModel.get_params

.. automethod:: mbirtorch.TomographyModel.print_params


Recon FOV and Voxel Spacing
---------------------------

The region of reconstruction and the voxel size are computed from the detector once, when the
model is built.  After changing a detector or geometry parameter with ``set_params``, call
``auto_set_recon_geometry`` to recompute them, or the reconstruction comes out at the wrong
scale.  Then, if needed, enlarge or shrink the region with ``resize_recon_fov``.  The FAQ on the
region of reconstruction in :ref:`DemosFAQs` walks through both.

.. automethod:: mbirtorch.TomographyModel.auto_set_recon_geometry

.. automethod:: mbirtorch.TomographyModel.resize_recon_fov

.. automethod:: mbirtorch.TomographyModel.get_magnification


Choosing the GPUs
-----------------

On a machine with several GPUs, a reconstruction uses them with no change to your script.
Call ``configure_devices`` to choose the devices yourself, or set the environment variable
``MBIRTORCH_NUM_DEVICES`` to set the number of GPUs for a whole process.  See
:doc:`usr_multi_gpu` for how several GPUs are used and how to get the most out of them.

.. automethod:: mbirtorch.TomographyModel.configure_devices


.. _SaveLoadDocs:

Saving and Loading
------------------

``export_recon_hdf5`` writes a reconstruction and its ``recon_dict`` to one HDF5 file, and
``import_recon_hdf5`` reads them back::

    recon, recon_dict = ct_model.recon(sinogram)
    mbirtorch.export_recon_hdf5('output/recon.h5', recon, recon_dict)
    recon, recon_dict = mbirtorch.import_recon_hdf5('output/recon.h5')

The file holds the volume as the dataset ``recon`` in right-hand axis order, (slice, col, row),
so that other programs read it the natural way, and it holds the recon parameters, model
parameters, log, and notes as attributes.  The slice viewers open it.

.. autofunction:: mbirtorch.export_recon_hdf5

.. autofunction:: mbirtorch.import_recon_hdf5


.. _detailed-parameter-docs:

Parameters
----------

The parameters a model holds, with their defaults, are listed on the :ref:`ParametersDocs` page.
