=================
Advanced Features
=================

After a first reconstruction, the parameters below are the ones most often adjusted.  Each section
names the parameters or methods for one task.  The full list of parameters is in :ref:`ParametersDocs`.


Setting parameters
------------------

Parameters are set on the model object::

    ct_model.set_params(sharpness=1.5, det_channel_offset=2.0)

The reconstruction shape and the voxel spacing are computed from the geometry parameters when the
model is created.  A later change to a detector spacing does not update them.  After changing
``delta_det_channel`` or ``delta_det_row``, call ``ct_model.auto_set_recon_geometry()`` to recompute
``recon_shape`` and ``delta_voxel``.

To see every parameter and its current value, call ``ct_model.print_params()``.


Image quality
-------------

Two parameters control the tradeoff between resolution and noise:

- ``sharpness`` (default 1.0): larger values give sharper edges and more noise, and smaller values
  give smoother images.  Try this parameter first.
- ``snr_db`` (default 30.0): the assumed signal to noise ratio of the sinogram in dB.  Larger
  values give sharper images.

Both parameters set the underlying regularization automatically.  Setting ``sigma_x`` or ``sigma_y``
directly turns the automatic setting off.


Reconstruction size and voxel spacing
-------------------------------------

The reconstruction array size and voxel pitch are set automatically to cover the field of view.
To reconstruct a different region, set these parameters:

- ``recon_shape``: a tuple ``(num_rows, num_cols, num_slices)``.
- ``delta_voxel``: the spacing between voxels in each direction.
- ``delta_det_channel`` and ``delta_det_row``: the spacing between detector channels and between
  detector rows.

All spacings are in arbitrary length units, which are explained in :ref:`Unit Conversion <ALU_conversion_label>`.


Objects larger than the field of view
-------------------------------------

When part of the object projects outside the detector at some views, the measurements that the
reconstruction cannot explain produce a bright ring at the edge of the volume and a bias in the
interior.  The two directions are handled separately.

- **Along the rotation axis (cone beam only):** set ``axial_pad_fraction`` to pad the slice axis
  at each end.  The value is a single fraction or a ``(top, bottom)`` pair.  The default 0 adds no
  slices, and 1 pads each end out to the farthest slice reached by any measured ray.
- **Across the rotation axis:** MBIRTorch prints a warning when the object appears to extend past
  the detector.  In that case, call ``ct_model.scale_recon_shape(s, s)`` with ``s`` of 1.1 or
  more before reconstructing.


Sinogram weights
----------------

The weights array has the same shape as the sinogram.  Each entry is the assumed inverse noise
variance of the corresponding sinogram entry.  Weights for the common noise models come from
``gen_weights``::

    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')
    recon, recon_dict = ct_model.recon(sinogram, weights=weights)

The weight types are ``'unweighted'``, ``'transmission'``, ``'transmission_root'``, and
``'emission'``.  The transmission types assume that the sinogram is in units of negative log
attenuation, as described in :ref:`PreprocessDocs`.

For objects that contain dense metal, ``gen_weights_mar`` reduces the weight of sinogram entries
that pass through the metal::

    weights = mbirtorch.gen_weights_mar(ct_model, sinogram, init_recon=None)

Passing a first reconstruction as ``init_recon`` gives a better estimate of where the metal is.


Geometry offsets
----------------

Two parameters correct for a detector that is not centered on the rotation axis:

- ``det_channel_offset``: the offset of the center of rotation from the center of the detector,
  along the channel direction.
- ``det_row_offset``: the offset along the row direction.

Both are in the same units as the detector spacing.  The channel offset can be estimated from the
sinogram, as shown in ``demo_10_geometry_calibration.py`` in :ref:`DemosFAQs`.


Large volumes and multiple GPUs
-------------------------------

On a machine with more than one GPU, a reconstruction is spread across the GPUs automatically.
The sinogram is divided by view and the reconstruction by slice, so the memory needed on each
GPU falls as more are used.  To choose the devices yourself, call
``ct_model.configure_devices(num_devices=n)`` or pass a ``devices`` list.  See :ref:`usr_multi_gpu`.

A volume that still does not fit can be reconstructed with
:meth:`~mbirtorch.TomographyModel.recon_split_sino`, which splits the detector rows into
overlapping bands, reconstructs one band at a time, and joins the results.

The amount of progress information printed during a reconstruction is set by ``verbose``.  The
default 1 prints basic information, 0 prints nothing, and 2 or 3 print more.
