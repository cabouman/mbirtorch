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
translation CT, and the ORNL HDF5 format.

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

ORNL pymbir
^^^^^^^^^^^

.. currentmodule:: mbirtorch.preprocess.pymbir

.. autofunction:: get_sino_and_model


Correcting a sinogram
---------------------

.. currentmodule:: mbirtorch.preprocess

These take the sinogram a scanner loader returns and give back a corrected one.  Beam hardening and stripe
corrections are applied before the reconstruction.  View alignment needs a first reconstruction,
so it comes after one.

.. code-block:: python

    sino = mtp.BH_correction(sino, alpha=0.1)
    sino = mtp.remove_all_stripe(sino)

.. autofunction:: BH_correction
.. autofunction:: remove_all_stripe
.. autofunction:: remove_stripe_fw
.. autofunction:: remove_sino_offset
.. autofunction:: correct_background_offset
.. autofunction:: correct_det_rotation
.. autofunction:: align_sino_views


Building a sinogram from raw scans
----------------------------------

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
.. autofunction:: finalize_model


Working on a reconstruction
---------------------------

.. autofunction:: apply_cylindrical_mask
.. autofunction:: segment_plastic_metal
.. autofunction:: multi_threshold_otsu


Geometry calibration
--------------------

.. currentmodule:: mbirtorch.preprocess.geometry_calibration

The ``geometry_calibration`` module estimates scan geometry from the sinogram itself.  Vendor
metadata sometimes gets a geometry parameter wrong, and sometimes it leaves the parameter out.  The
functions here estimate two such parameters from the data: the center of rotation, which is
``det_channel_offset``, and the detector rotation in radians.  They also show the evidence behind
each estimate.

Run these functions after defective-pixel interpolation, background offset correction, and stripe
removal, and before :func:`~mbirtorch.preprocess.align_sino_views`.  Stripe removal comes first
because a gain stripe sits at a fixed channel, and a geometry estimate would take that stripe for a
feature of the object.  Alignment comes last because it shifts each view on its own.  A wrong
``det_channel_offset`` looks like a per-view shift, so aligning first would remove part of the error
that a calibration is meant to find.

The automatic workflow estimates the channel offset, then the detector rotation at that offset, then
the channel offset again at that rotation.  The two quantities are coupled, so the second estimate of
the offset is the better one.  There is no single driver function yet, so the three calls are made in
order:

.. code-block:: python

    from mbirtorch.preprocess import geometry_calibration as gc

    offset = gc.estimate_det_channel_offset(ct_model, sino)
    rotation = gc.estimate_det_rotation(ct_model, sino, det_channel_offset=offset.value)
    offset = gc.estimate_det_channel_offset(ct_model, sino, det_rotation=rotation.value)
    ct_model, sino = gc.apply_calibration(ct_model, sino, [rotation, offset])
    recon, recon_dict = ct_model.recon(sino)

The manual workflow reconstructs one slice per candidate value and lets you choose the value by eye.
Use it for a scan the estimators refuse, and to check an estimate the automatic workflow made:

.. code-block:: python

    import numpy as np
    import mbirtorch
    from mbirtorch.preprocess import geometry_calibration as gc

    values = np.linspace(-4.0, 4.0, 17)
    slices = gc.parameter_sweep(ct_model, sino, 'det_channel_offset', values)
    mbirtorch.slice_viewer(slices, title='det_channel_offset sweep')
    chosen = 8                                    # the index of the slice that looked best
    ct_model.set_params(det_channel_offset=values[chosen])

The estimators accept a parallel-beam or a cone-beam scan over a full rotation.  Four kinds of input
are refused with an error: a scan over less than a full rotation, a helical scan, a multiaxis scan,
and a sinogram that is already divided across devices.  A divided sinogram has to be gathered to the
host first.  :func:`parameter_sweep` accepts any parallel-beam or cone-beam scan, but not a
translation scan.

When the scanner loader supplies a detector tilt, prefer it over the estimate, and check the slices far from
the central plane before applying an estimate, because a detector rotation displaces those slices
most.  Treat a rotation-direction answer that comes with a warning as undecided.

.. autofunction:: estimate_det_channel_offset
.. autofunction:: estimate_det_rotation
.. autofunction:: check_rotation_direction
.. autofunction:: conjugate_difference
.. autofunction:: parameter_sweep
.. autofunction:: apply_calibration
.. autofunction:: build_reduced_problem
.. autofunction:: reduce_sinogram

.. The seven field names are excluded because the class docstring above documents each of them.
   Without the exclusion, autodoc documents every field a second time as "Alias for field number n".

.. autoclass:: CalibrationResult
   :members:
   :exclude-members: parameter, value, score, candidates, scores, method, reduction
