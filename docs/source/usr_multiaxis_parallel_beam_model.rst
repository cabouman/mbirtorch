.. _MultiAxisParallelBeamModelDocs:

=========================
Multi-Axis Parallel Model
=========================

.. plot:: figs/geom_multiaxis.py
   :align: center
   :width: 60%

The multi-axis parallel geometry is the parallel beam geometry with a second angle: each view
has an azimuth, the usual rotation about the object's z axis, and an elevation, the angle at
which the rays leave the xy plane.  With a constant elevation it is parallel beam
laminography, and with zero elevation it is the parallel beam model.  The model is built from
the sinogram shape and a two-column array of angles.  The voxels are cubes with spacing
``delta_voxel``, which defaults to the detector channel spacing of 1 ALU.

.. code-block:: python

    angles = np.stack([azimuth, elevation], axis=1)     # radians, one row per view
    ct_model = mbirtorch.MultiAxisParallelModel(sinogram.shape, angles)
    recon, recon_dict = ct_model.recon(sinogram)

Constructor
-----------

.. autoclass:: mbirtorch.MultiAxisParallelModel
   :show-inheritance:

Reconstruction
--------------

``recon`` is the iterative reconstruction.  ``recon_fbp`` is filtered back projection, fast and
non-iterative, which ``recon`` uses as its starting point.

.. automethod:: mbirtorch.MultiAxisParallelModel.recon

.. automethod:: mbirtorch.MultiAxisParallelModel.recon_fbp
