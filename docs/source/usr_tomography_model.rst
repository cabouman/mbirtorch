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

.. automethod:: mbirtorch.TomographyModel.prox_map

.. automethod:: mbirtorch.TomographyModel.initialize_prox

.. automethod:: mbirtorch.TomographyModel.forward_project

.. automethod:: mbirtorch.TomographyModel.back_project

.. automethod:: mbirtorch.TomographyModel.recon_split_sino

.. automethod:: mbirtorch.TomographyModel.recon_plastic_metal

.. automethod:: mbirtorch.TomographyModel.project_points


Parameter Handling
------------------

.. automethod:: mbirtorch.TomographyModel.set_params

.. automethod:: mbirtorch.TomographyModel.get_params

.. automethod:: mbirtorch.TomographyModel.print_params


Recon FOV and Voxel Spacing
---------------------------

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

.. automethod:: mbirtorch.TomographyModel.save_recon_hdf5

.. automethod:: mbirtorch.TomographyModel.load_recon_hdf5


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
