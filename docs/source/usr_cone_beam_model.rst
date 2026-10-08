.. _ConeBeamModelDocs:

===============
Cone Beam Model
===============

.. plot:: figs/geom_cone.py
   :align: center
   :width: 60%

In the cone beam geometry the rays of a view diverge from a point source to a flat or a
curved detector, and the object rotates about an axis parallel to the detector columns.
The scan can be circular or helical.  Besides the sinogram shape and the view angles, the
model needs the distance from the source to the detector and from the source to the
rotation axis, in the same units as the detector pixel pitch.  The voxels are cubes with
spacing ``delta_voxel``, which defaults to ``delta_det_channel / magnification`` with
``magnification = source_detector_dist / source_iso_dist``, so a larger magnification gives
smaller voxels.  Change these with :meth:`~mbirtorch.TomographyModel.set_params`; see
:ref:`ParametersDocs`.

.. code-block:: python

    ct_model = mbirtorch.ConeBeamModel(sinogram.shape, angles,
                                       source_detector_dist=source_detector_dist,
                                       source_iso_dist=source_iso_dist)
    recon, recon_dict = ct_model.recon(sinogram)

Along the rotation axis, the reconstruction is centered on the band of the object the detector
sees.  Shift it with ``recon_slice_offset`` and extend it at either end with
``axial_pad_fraction``, both set with ``set_params`` after the model is built and described in
the cone beam table of :ref:`ParametersDocs`.

Constructor
-----------

.. autoclass:: mbirtorch.ConeBeamModel
   :show-inheritance:

Reconstruction
--------------

``recon`` is the iterative reconstruction.  ``recon_fdk`` is Feldkamp-Davis-Kress filtered back
projection, fast and non-iterative, which ``recon`` uses as its starting point.
``recon_split_sino`` reconstructs a sinogram too large for memory in two overlapping halves.

.. automethod:: mbirtorch.ConeBeamModel.recon

.. automethod:: mbirtorch.ConeBeamModel.recon_fdk

.. automethod:: mbirtorch.ConeBeamModel.recon_split_sino
