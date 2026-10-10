.. _UserAPIDocs:

========
User API
========

A reconstruction is three calls: build a model of the scan geometry, reconstruct, and look
at the result.  Everything else is reached from the menu on the left.

.. automodule:: mbirtorch
   :no-index:

Build a model
-------------

* :class:`~mbirtorch.ParallelBeamModel`: parallel rays.
* :class:`~mbirtorch.ConeBeamModel`: rays from a point source, circular or helical.
* :class:`~mbirtorch.MultiAxisParallelModel`: parallel rays with a per-view elevation; laminography.
* :class:`~mbirtorch.TranslationModel`: cone beam views of a translated object; under development.

See :ref:`GeometryModelsDocs`.

Reconstruct
-----------

.. autosummary::

   TomographyModel.recon
   TomographyModel.set_params
   TomographyModel.forward_project
   TomographyModel.back_project

See :ref:`TomographyModelDocs` and :ref:`ParametersDocs`.

Check memory and time
---------------------

.. autosummary::

   estimate_resources

See :ref:`usr_estimate_resources`.

View and save
-------------

.. autosummary::

   view_utils.slice_viewer
   view_utils.slice_viewer4d
   export_recon_hdf5

See :ref:`Utilities`.

Prepare data
------------

A scanner loader, one per supported instrument, reads the scan and returns the sinogram and the
model in one call: ``nsi``, ``zeiss``, ``zeiss_tct``, and ``pymbir`` each have a
``get_sino_and_model``.  See :ref:`ScannerLoaders`.

More
----

* :ref:`MACE4DDocs`: 4D reconstruction of a moving object.
* :ref:`VCLSDocs`: pick the few view angles that reconstruct an object best.
* :ref:`DenoisingDocs`: the MAP denoiser, for Plug-and-Play loops.
* :ref:`AutogradDocs`: the projectors as differentiable PyTorch operations.
* :ref:`usr_multi_gpu`: reconstruction across several GPUs.
* :ref:`HSNTDocs`: hyperspectral neutron transmission data, with its HDF5 import and export.

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Classes

   usr_parameters
   usr_tomography_model
   usr_geometry_models
   usr_mace4d
   usr_autograd
   usr_preprocess
   usr_utilities
   usr_vcls
   usr_hsnt
