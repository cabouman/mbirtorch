.. _UserAPIOverviewDocs:

=================
User API Overview
=================

Most functions can be accessed by importing mbirtorch and creating a model or through mbirtorch directly.  Most
commonly used functions are described below.  See :ref:`DemosFAQs` for examples.  Navigate individual pages under :ref:`UserAPIDocs` for more details.

.. DIVERGENCE(automodule members): see the note on the same directive in usr_api.rst --
   the ":members: :undoc-members: :show-inheritance:" options are dropped so this page
   renders what mbirjax's does.  Narrowing __all__ does not substitute for dropping them.

.. automodule:: mbirtorch
   :no-index:

Geometry Models
---------------

The first step is to create an instance with a specific geometry. This is done by initializing one of the following geometry classes:

.. autosummary::

   ParallelBeamModel
   ConeBeamModel
   TranslationModel
   MultiAxisParallelModel

Reconstruction and Projection
-----------------------------

Each geometry class is derived from :ref:`TomographyModelDocs`, which includes a number of powerful methods listed below for manipulating sinograms and reconstructions.
Detailed documentation for each geometry class is provided in :ref:`ParallelBeamModelDocs` and :ref:`ConeBeamModelDocs`.

Note that :ref:`ParallelBeamModelDocs` also includes ``recon_fbp`` and :ref:`ConeBeamModelDocs` includes ``recon_fdk``
for direct (non-iterative) reconstruction in the case of many views and low-noise data.

.. autosummary::

   TomographyModel.recon
   TomographyModel.resize_recon_fov
   TomographyModel.prox_map
   TomographyModel.forward_project
   TomographyModel.back_project

Denoising
---------

The identity geometry: its reconstruction is the MAP denoiser under the qGGMRF prior, handy in
a Plug-and-Play loop.  See :ref:`DenoisingDocs`.

.. autosummary::

   QGGMRFDenoiser.denoise

Differentiable Projectors
-------------------------

See :ref:`AutogradDocs` for the projectors exposed as differentiable PyTorch operations, for
use inside a deep-learning pipeline.

.. autosummary::

   autograd.forward_project_differentiable
   autograd.back_project_differentiable
   autograd.TorchProjector

Parameter Handling
------------------

See :ref:`ParametersDocs` for a description of the parameters.
Users can set, get, and printout parameters using the following primary methods.

.. autosummary::

   TomographyModel.set_params
   TomographyModel.get_params
   TomographyModel.print_params


Saving and Loading
------------------

* A reconstruction and the dict of parameters and logs returned from :meth:`~mbirtorch.TomographyModel.recon` are written to one HDF5 file and read back with :func:`~mbirtorch.export_recon_hdf5` and :func:`~mbirtorch.import_recon_hdf5`.

.. autosummary::

   export_recon_hdf5
   import_recon_hdf5


Utilities
---------

See :ref:`Utilities` for details on Utility Functions.
These include variety of functions for viewing, generating weights, exporting/importing data,
generating synthetic data, and clearing the on-disk compile cache.

.. autosummary::

   view_utils.slice_viewer
   view_utils.slice_viewer4d
   view_utils.geometry_viewer
   median_filter3d
   vcd_utils.gen_weights
   vcd_utils.gen_weights_mar
   utilities.download_and_extract
   utilities.generate_3d_shepp_logan_low_dynamic_range
   utilities.clear_cache


Preprocessing
-------------

See :ref:`PreprocessDocs` for details on Preprocessing Functions.
These functions various methods to compute and correct the sinogram data as needed.
The following are functions specific to NSI scanners.  See `demo_nsi.py <https://github.com/cabouman/mbirtorch_applications/tree/main/nsi>`__ in the
`mbirtorch_applications <https://github.com/cabouman/mbirtorch_applications>`__ repo.

It also includes functions for processing cone beam and parallel beam data to remove artifacts from metal, detector defects.

The ``preprocess`` subpackage loads lazily: ``import mbirtorch`` does not pull in its
dependency stack, and ``import mbirtorch.preprocess`` or the first attribute access loads
it.


Sparse View Selection
---------------------

Before a scan, pick the few view angles that best reconstruct an object like the reference.
See :ref:`VCLSDocs`.

.. autosummary::

   vcls.get_opt_views
