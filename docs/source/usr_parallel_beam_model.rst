.. _ParallelBeamModelDocs:

===================
Parallel Beam Model
===================

.. plot:: figs/geom_parallel.py
   :align: center
   :width: 60%

In the parallel beam geometry the rays of every view are parallel, as from a source at
infinity, and the object rotates about an axis parallel to the detector columns.  The
model is built from the sinogram shape and the view angles.  The voxels are cubes with
spacing ``delta_voxel``, which defaults to the detector channel spacing
``delta_det_channel`` of 1 ALU, and the reconstruction covers the width and the height of
the detector.  Change these with :meth:`~mbirtorch.TomographyModel.set_params`, then call
``auto_set_recon_geometry`` so the reconstruction geometry follows; see :ref:`ParametersDocs`.

.. code-block:: python

    ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
    recon, recon_dict = ct_model.recon(sinogram)

Constructor
-----------

.. autoclass:: mbirtorch.ParallelBeamModel
   :show-inheritance:

Reconstruction
--------------

``recon`` is the iterative reconstruction.  ``recon_fbp`` is filtered back projection, fast and
non-iterative, which ``recon`` uses as its starting point.  ``recon_split_sino`` reconstructs a
sinogram too large for memory in overlapping sections of detector rows, on groups of GPUs side
by side.

.. automethod:: mbirtorch.ParallelBeamModel.recon

.. automethod:: mbirtorch.ParallelBeamModel.recon_fbp

.. automethod:: mbirtorch.ParallelBeamModel.recon_split_sino
