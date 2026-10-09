.. _TranslationModelDocs:

=================
Translation Model
=================

.. plot:: figs/geom_translation.py
   :align: center
   :width: 60%

In translation computed tomography (TCT) the object does not rotate.  Each view is a cone beam
projection of the object after a translation, so the geometry suits thin, flat objects.
Besides the sinogram shape, the model needs the translation of the object at each view, in
ALU, and the distance from the source to the detector and from the source to the object.

This model is under development, and its interface may change.

.. code-block:: python

    ct_model = mbirtorch.TranslationModel(sinogram.shape, translation_vectors,
                                          source_detector_dist=source_detector_dist,
                                          source_iso_dist=source_iso_dist)
    recon, recon_dict = ct_model.recon(sinogram)

Constructor
-----------

.. autoclass:: mbirtorch.TranslationModel
   :show-inheritance:

Reconstruction
--------------

``recon`` is the iterative reconstruction.  ``recon_fdk`` is the non-iterative reconstruction
that ``recon`` uses as its starting point.

.. automethod:: mbirtorch.TranslationModel.recon

.. automethod:: mbirtorch.TranslationModel.recon_fdk
