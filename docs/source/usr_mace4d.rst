.. _MACE4DDocs:

=================
4D Reconstruction
=================

One continuous scan of a moving object is reconstructed as one volume per time frame.  The
views are taken in time order and divided into overlapping angular windows, one per frame.
``frames_per_rotation`` sets how many frames make up a rotation and ``frame_overlap_factor``
how many frames share a view, so each frame spans ``frame_overlap_factor * 360 /
frames_per_rotation`` degrees.  Wider frames have more views and less noise, at the cost of
time resolution.  The frames are reconstructed together by the MACE algorithm of
:cite:`mace4d`, with a prior along time as well as space.  A built-in filter along the frame
axis removes the periodic modulation that the overlapping windows produce.

The model is built on the model of the whole scan.

.. code-block:: python

    ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
    mace4d = mbirtorch.MACE4DModel(ct_model, frames_per_rotation=6, frame_overlap_factor=2.0)
    recon_4d, recon_dict = mace4d.recon(sinogram, max_iterations=10)
    mbirtorch.slice_viewer4d(recon_4d, data_dicts=recon_dict)

Constructor
-----------

.. autoclass:: mbirtorch.MACE4DModel
   :show-inheritance:

Reconstruction
--------------

.. automethod:: mbirtorch.MACE4DModel.recon

.. automethod:: mbirtorch.MACE4DModel.set_params

.. automethod:: mbirtorch.MACE4DModel.set_device_pool
