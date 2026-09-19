.. _PreprocessDocs:

=============
Preprocessing
=============

The ``preprocess`` module provides scanner-specific preprocessing and more general preprocessing to compute and correct the sinogram data.
See `demo_nsi.py <https://github.com/cabouman/mbirtorch_applications/tree/main/nsi>`__ in the
`mbirtorch_applications <https://github.com/cabouman/mbirtorch_applications>`__ repo for example uses.

One-Call Preprocessing
----------------------

Each supported scanner allows one-call preprocessing with the scanner's ``get_sino_and_model`` function, which loads a scan, computes its sinogram, and returns a ready-to-reconstruct model.

.. code-block:: python

    sino, model = mbirtorch.preprocess.nsi.get_sino_and_model(dataset_dir)
    weights = mbirtorch.gen_weights(sino, weight_type='transmission_root')
    recon, recon_dict = model.recon(sino, weights=weights)

The call selects the correct geometry class for the scanner (for example, the Zeiss reader picks
``ParallelBeamModel`` for an Ultra scan and ``ConeBeamModel`` for a Versa scan) and computes the
reconstruction geometry from the real detector parameters, so the returned model is ready to be used.
Reconstruction weights can be generated with :func:`mbirtorch.vcd_utils.gen_weights`.

.. DIVERGENCE(gen_weights ref): mbirjax writes this role as :func:`mbirjax.gen_weights`,
   which is the single warning its own docs build reports -- gen_weights is documented
   under its module path, not the package path.  Fixed here rather than inherited.


NorthStar Instrument (NSI) reader
---------------------------------

.. currentmodule:: mbirtorch.preprocess.nsi

.. autofunction:: get_sino_and_model
.. autofunction:: load_scans_and_params


Zeiss Versa and Ultra reader
----------------------------

.. currentmodule:: mbirtorch.preprocess.zeiss

.. autofunction:: get_sino_and_model
.. autofunction:: load_scans_and_params


Zeiss translation tomography functions
--------------------------------------

.. currentmodule:: mbirtorch.preprocess.zeiss_tct

.. autofunction:: get_sino_and_model
.. autofunction:: load_scans_and_params


PYMBIR functions
----------------

.. currentmodule:: mbirtorch.preprocess.pymbir

.. autofunction:: get_sino_and_model


General preprocess functions
----------------------------

.. currentmodule:: mbirtorch.preprocess

.. autofunction:: compute_sino_transmission
.. autofunction:: detect_blank_margins
.. autofunction:: apply_detector_crop
.. autofunction:: align_sino_views
.. autofunction:: interpolate_defective_pixels
.. autofunction:: correct_det_rotation
.. autofunction:: correct_background_offset
.. autofunction:: downsample_view_data
.. autofunction:: crop_view_data
.. autofunction:: apply_cylindrical_mask
.. autofunction:: save_cone_preprocessing
.. autofunction:: load_cone_preprocessing
.. autofunction:: read_tif_stack_dir
.. autofunction:: read_tif_img


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
host first.  :func:`parameter_sweep` accepts every scan the readers produce.

The estimators were checked on synthetic data and on real scans from an NSI scanner and a Zeiss
Versa scanner.  On the real scans the channel offset agreed with the vendor's value to better than
a tenth of a channel.  The detector rotation estimate followed known rotations added to the real
scans with a slope of one, but its zero point depended on the object.  On one scan it read 0.044
degrees.  A fine sweep of directly reconstructed slices far from the central plane put the detector
rotation of that scan near 0.15 degrees, and the vendor's recorded tilt was 0.167 degrees.  When
the reader supplies a tilt, prefer it, and check the slices far from the central plane before
applying an estimate, because a detector rotation displaces those slices most.  The
rotation-direction
check gave the right answer whenever its margin was above its warning threshold, and it warned on
the one scan where it did not.  Treat an answer that comes with the warning as undecided.

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


MAR utilities
-------------

.. currentmodule:: mbirtorch.preprocess

.. autofunction:: gen_huber_weights
.. autofunction:: BH_correction
.. autofunction:: fit_beam_hardening_curve
.. autofunction:: fit_inverse_beam_hardening_curve
.. autofunction:: apply_beam_hardening_curve
.. autofunction:: apply_inverse_beam_hardening_curve

Stripe/Ring/Offset Removal
--------------------------

.. currentmodule:: mbirtorch.preprocess

.. autofunction:: remove_all_stripe
.. autofunction:: remove_stripe_fw
.. autofunction:: remove_sino_offset


Segmentation functions
----------------------

.. currentmodule:: mbirtorch.preprocess

.. autofunction:: multi_threshold_otsu
.. autofunction:: segment_plastic_metal

