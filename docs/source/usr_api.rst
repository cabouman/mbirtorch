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

View and save
-------------

.. autosummary::

   view_utils.slice_viewer
   export_recon_hdf5

See :ref:`Utilities`.

Prepare data
------------

One call per scanner loads a scan and returns the sinogram and the model: ``nsi``, ``zeiss``,
``zeiss_tct``, and ``pymbir`` each have a ``get_sino_and_model``.  See :ref:`PreprocessDocs`.

More
----

* :ref:`MACE4DDocs`: 4D reconstruction of a moving object.
* :ref:`VCLSDocs`: pick the few view angles that reconstruct an object best.
* :ref:`DenoisingDocs`: the MAP denoiser, for Plug-and-Play loops.
* :ref:`AutogradDocs`: the projectors as differentiable PyTorch operations.
* :ref:`usr_multi_gpu`: reconstruction across several GPUs.

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
