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

.. automethod:: mbirtorch.TomographyModel.direct_recon

.. automethod:: mbirtorch.TomographyModel.prox_map

.. automethod:: mbirtorch.TomographyModel.forward_project

.. automethod:: mbirtorch.TomographyModel.back_project


Parameter Handling
------------------

``TomographyModel`` inherits its parameter accessors from ``ParameterHandler``.  MBIRJAX
exports that class at package level, so its ``automodule`` documented it implicitly;
mbirtorch does not export it, so it is documented explicitly here.  This also supplies the
cross-reference target that ``:show-inheritance:`` needs on every model class.

.. autoclass:: mbirtorch.parameter_handler.ParameterHandler

.. automethod:: mbirtorch.TomographyModel.set_params

.. automethod:: mbirtorch.parameter_handler.ParameterHandler.get_params

.. automethod:: mbirtorch.parameter_handler.ParameterHandler.print_params

.. PENDING(param_accessors): restore these two directives when get_all_params and
   get_recon_dict are ported to mbirtorch.

   .. automethod:: mbirtorch.TomographyModel.get_all_params

   .. automethod:: mbirtorch.TomographyModel.get_recon_dict


Recon Shape and Voxel Spacing
-----------------------------

.. automethod:: mbirtorch.TomographyModel.auto_set_recon_geometry

.. automethod:: mbirtorch.TomographyModel.scale_recon_shape

.. automethod:: mbirtorch.TomographyModel.get_magnification


Device Configuration
--------------------

On a machine with multiple GPUs, MBIRTorch can divide a reconstruction across them to
increase the available memory and reduce reconstruction time, for every geometry.  Unlike
MBIRJAX, this is opt-in rather than automatic: a model uses a single device until
``configure_devices`` widens it.  The methods below give explicit control over which devices
are used.  Per-device memory use is reported by ``mbirtorch.get_memory_stats()``.
See :doc:`usr_multi_gpu` for a full discussion.

.. automethod:: mbirtorch.TomographyModel.configure_devices

.. automethod:: mbirtorch.TomographyModel.prepare_sino_for_devices

.. REPLACED(device_summary): MBIRJAX documents a ``device_summary`` property here, which
   reports the devices its automatic selection chose.  MBIRTorch has no automatic
   selection to report, since ``configure_devices`` sets the layout explicitly, and
   ``get_memory_stats`` covers the per-device reporting.  The property will not be ported.


.. PENDING(save_load_hdf5): restore the section below when save_recon_hdf5 and
   load_recon_hdf5 are ported to mbirtorch.  NOTE: the _SaveLoadDocs label lives here, and
   usr_utilities.rst and usr_api_overview.rst both reference it -- their matching PENDING
   blocks must be restored at the same time.

   .. _SaveLoadDocs:

   Saving and Loading
   ------------------

   .. automethod:: mbirtorch.TomographyModel.save_recon_hdf5

   .. automethod:: mbirtorch.TomographyModel.load_recon_hdf5


.. _detailed-parameter-docs:

Parameter Documentation
-----------------------

See the :ref:`Primary Parameters <ParametersDocs>` page.
