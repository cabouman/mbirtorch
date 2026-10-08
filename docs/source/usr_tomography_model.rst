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


Device Configuration
--------------------

On a machine with multiple GPUs, MBIRTorch automatically divides a reconstruction across
them to increase the available memory and reduce reconstruction time -- with no change to
your script, and for every geometry.  The device count is chosen once per model, when its
first reconstruction starts: measured speed thresholds decide how many devices are worth
using, and a memory check confirms the layout fits.  The methods below give explicit
control over which devices are used.  Per-device memory use is reported by ``mbirtorch.get_memory_stats()``.
See :doc:`usr_multi_gpu` for a full discussion.

.. automethod:: mbirtorch.TomographyModel.configure_devices

.. automethod:: mbirtorch.TomographyModel.prepare_sino_for_devices

.. REPLACED(device_summary): MBIRJAX documents a ``device_summary`` property here, which
   reports the devices its automatic selection chose.  MBIRTorch reports the layout a run
   settled on in the run log's device line, and ``get_memory_stats`` covers the per-device
   reporting, so the property will not be ported.


.. _SaveLoadDocs:

Saving and Loading
------------------

.. automethod:: mbirtorch.TomographyModel.save_recon_hdf5

.. automethod:: mbirtorch.TomographyModel.load_recon_hdf5


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
