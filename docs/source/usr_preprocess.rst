.. _PreprocessDocs:

=============
Preprocessing
=============

The ``preprocess`` package turns a scan into a sinogram and a model that is ready to reconstruct.
For a supported scanner one call does it.  The other functions correct a sinogram, build one from
raw scans, or work on a reconstruction.  Scripts that use them are in the
`nsi <https://github.com/cabouman/mbirtorch_applications/tree/main/nsi>`__,
`zeiss <https://github.com/cabouman/mbirtorch_applications/tree/main/zeiss>`__, and
`tct <https://github.com/cabouman/mbirtorch_applications/tree/main/tct>`__ folders of the
`mbirtorch_applications <https://github.com/cabouman/mbirtorch_applications>`__ repository.


.. _ScannerLoaders:

Scanner loaders
---------------

A scanner loader is a function for one specific instrument.  It reads that instrument's files,
computes the sinogram, and returns it with a model of the right geometry and its parameters set.
Each is a ``get_sino_and_model`` function in a module named for the instrument.  The instruments
with loaders are North Star Imaging (NSI) scanners, Zeiss Versa and Ultra scanners, Zeiss
translation CT, Zeiss VoluMax, and the ORNL HDF5 format.

.. code-block:: python

    import mbirtorch
    import mbirtorch.preprocess as mtp

    sino, model = mtp.nsi.get_sino_and_model(dataset_dir)
    weights = mbirtorch.gen_weights(sino, weight_type='transmission_root')
    recon, recon_dict = model.recon(sino, weights=weights)

The Zeiss translation loader also returns a weight mask as a third value.

North Star Imaging (NSI)
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mbirtorch.preprocess.nsi

.. autofunction:: get_sino_and_model

Zeiss Versa and Ultra
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mbirtorch.preprocess.zeiss

.. autofunction:: get_sino_and_model

Zeiss translation CT
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: mbirtorch.preprocess.zeiss_tct

.. autofunction:: get_sino_and_model

Zeiss VoluMax
^^^^^^^^^^^^^

.. currentmodule:: mbirtorch.preprocess.volumax

.. autofunction:: get_sino_and_model

ORNL pymbir
^^^^^^^^^^^

.. currentmodule:: mbirtorch.preprocess.pymbir

.. autofunction:: get_sino_and_model


Correcting a sinogram
---------------------

.. currentmodule:: mbirtorch.preprocess

These take the sinogram a scanner loader returns and give back a corrected one, before the
reconstruction.

.. code-block:: python

    sino = mtp.BH_correction(sino, alpha=0.1)
    sino = mtp.remove_all_stripe(sino)

.. autofunction:: BH_correction
.. autofunction:: remove_all_stripe
.. autofunction:: remove_stripe_fw
.. autofunction:: remove_sino_offset
.. autofunction:: correct_background_offset
.. autofunction:: correct_det_rotation


Building a sinogram
-------------------

For an instrument without a scanner loader, start from the object, blank, and dark scans and run
the same steps the loaders run.  The loaders' ``load_scans_and_params`` functions return the raw
scans and the scanner's parameters when you want to start from those.

.. code-block:: python

    obj_scan, blank_scan, dark_scan, defects = mtp.crop_view_data(obj_scan, blank_scan, dark_scan,
                                                                  crop_pixels_sides=20,
                                                                  defective_pixel_array=defects)
    sino = mtp.scan_to_sino(obj_scan, blank_scan, dark_scan, defects, downsample_factor=(2, 2))
    sino = mtp.correct_background_offset(sino, option='per_view')
    sino = mtp.correct_zinger_pixels(sino)
    model = mbirtorch.ConeBeamModel(sino.shape, angles, source_detector_dist=sdd, source_iso_dist=sid)

.. autofunction:: crop_view_data
.. autofunction:: scan_to_sino
.. autofunction:: correct_zinger_pixels


Working on a reconstruction
---------------------------

.. autofunction:: apply_cylindrical_mask
.. autofunction:: segment_plastic_metal
.. autofunction:: multi_threshold_otsu


Calibrating geometry
--------------------

``estimate_det_channel_offset`` finds the center of rotation, ``det_channel_offset``, from the
sinogram alone, for a scan whose metadata got it wrong.  It compares each view with the view
opposite to it, so it needs a scan over a full rotation.  On any other scan it warns and returns
the model's current value.

``fit_det_alignment`` needs a first reconstruction, which is the costly part.  It fits each view
to the forward projection of that reconstruction and returns two dicts: ``model_params``, the detector offsets to set on the model, and ``view_params``,
the per-view corrections to apply to the data, which are each view's deviation from the global
offsets and, with ``rotation=True``, its detector rotation.  ``correct_det_alignment`` resamples
the views by those corrections, once, with bicubic interpolation.  Set the channel offset first,
then estimate, update the model, and resample:

.. code-block:: python

    ct_model.set_params(det_channel_offset=mtp.estimate_det_channel_offset(ct_model, sino))
    model_params, view_params = mtp.fit_det_alignment(ct_model, sino, ct_model.recon_direct(sino),
                                                           rotation=True)
    ct_model.set_params(**model_params)
    sino = mtp.correct_det_alignment(ct_model, sino, view_params)
    recon, recon_dict = ct_model.recon(sino)

Each step can be skipped.  Leaving out the last line corrects the model and leaves the data as it
is.  Passing a dict with only the ``det_rotation`` key removes the rotation and nothing else.  A
second round, with a reconstruction from the corrected data, refines the per-view corrections.

.. autofunction:: fit_det_alignment
.. autofunction:: correct_det_alignment
.. autofunction:: estimate_det_channel_offset
