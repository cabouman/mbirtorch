.. _ParallelBeamModelDocs:

===================
Parallel Beam Model
===================

The ``ParallelBeamModel`` class implements a geometry and reconstruction model for parallel-beam computed tomography.
This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.
It also implements some parallel-beam specific functions such as FBP (Filtered Back Projection) reconstruction.

Note that for parallel-beam geometry the default value of ``delta_voxel`` = ``delta_det_channel`` = 1 ALU,
which results in pixel spacing that is the same as detector channel spacing.
However, these parameters can be changed by the user with the ``TomographyModel.set_params()`` method.
The spacing between slices of the reconstruction are fixed to be the same as the spacing between detector rows.

See the API docs for the :class:`~mbirtorch.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``ParallelBeamModel``.

Simple Reconstruction
---------------------

For a basic reconstruction, one function call takes the sinogram and the view angles and returns
the reconstruction.  The class below gives full control over the geometry, the parameters, and the
reconstruction itself.

.. autofunction:: mbirtorch.recon_simple_parallel

Constructor
-----------

.. autoclass:: mbirtorch.ParallelBeamModel
   :show-inheritance:

Alternative Reconstruction
--------------------------

.. automethod:: mbirtorch.ParallelBeamModel.recon_fbp

.. automethod:: mbirtorch.ParallelBeamModel.recon_split_sino
