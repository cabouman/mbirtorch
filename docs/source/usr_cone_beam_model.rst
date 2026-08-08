.. _ConeBeamModelDocs:

===============
Cone Beam Model
===============

The ``ConeBeamModel`` class implements a geometry and reconstruction model for cone beam computed tomography.
This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.
It also implements some cone-beam specific functions such as FDK (Feldkamp-Davis-Kress) reconstruction.

For cone-beam geometry, the default detector channel spacing is ``delta_det_channel`` is 1 ALU,
and the voxels are 3D cubes with spacing ``delta_voxel``.

The default voxel spacing is set to ``delta_voxel = delta_det_channel / magnification`` where ``magnification = source_detector_dist / source_iso_dist``.
This implies that as the magnification increases, the default voxel spacing decreases.
However, these parameters can be changed by the user using the ``TomographyModel.set_params()`` method.

See the API docs for the :class:`~mbirtorch.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``ConeBeamModel``.

Constructor
-----------

.. autoclass:: mbirtorch.ConeBeamModel
   :show-inheritance:

Alternative Reconstruction
--------------------------

.. automethod:: mbirtorch.ConeBeamModel.fdk_recon

.. PENDING(split_sino_recon): restore the directive below when
   ConeBeamModel.split_sino_recon is ported.  A full-logic port is chartered: the nsi
   split-sinogram demo calls it directly, and it nearly doubles the feasible cone recon
   size at a fixed GPU count.

   .. automethod:: mbirtorch.ConeBeamModel.split_sino_recon
