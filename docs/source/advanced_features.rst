=================
Advanced Features
=================

After a first reconstruction, the parameters below are the ones most often adjusted.  Each section
names the parameters or methods for one task.  The full list of parameters is in :ref:`ParametersDocs`.


Setting parameters
------------------

Parameters are set on the model object::

    ct_model.set_params(sharpness=1.5, det_channel_offset=2.0)

After changing a detector parameter, call ``ct_model.auto_set_recon_geometry()`` so that the
reconstruction geometry follows; see `Changing the reconstruction field of view`_ below.

To see every parameter and its current value, call ``ct_model.print_params()``.


Tuning image quality
--------------------

Two parameters control the tradeoff between resolution and noise:

- ``sharpness`` (default 1.0): larger values give sharper edges and more noise, and smaller values
  give smoother images.  Try this parameter first.
- ``snr_db`` (default 30.0): the assumed signal to noise ratio of the sinogram in dB.  Larger
  values give sharper images.

Both parameters set the underlying regularization automatically.  Setting ``sigma_x`` or ``sigma_y``
directly turns the automatic setting off.


Setting detector parameters
---------------------------
Four parameters describe the detector, all in arbitrary length units:

- ``delta_det_channel`` and ``delta_det_row``: the spacing between detector channels and between
  detector rows.
- ``det_channel_offset``: the offset of the center of rotation from the center of the detector,
  along the channel direction.  A wrong value gives rings near the center of the reconstruction.
- ``det_row_offset``: the offset of the source-to-detector line from the center of the detector,
  along the row direction.

The scanner loaders, the functions for specific instruments described in :ref:`ScannerLoaders`, set
them from the scanner's files.  To set them yourself::

    ct_model.set_params(delta_det_channel=0.1, det_channel_offset=2.0)
    ct_model.auto_set_recon_geometry()

The second line is required after any change to these parameters; without it the reconstruction
comes out at the wrong scale.  The channel offset can be estimated from the sinogram; see the
geometry calibration section of :ref:`PreprocessDocs`.


Changing the reconstruction field of view
-----------------------------------------

The reconstruction fills a box of voxels, the field of view (FOV).  When the model is built, the
box is set from the detector: across the rotation axis it covers what the detector sees, and
along the rotation axis it is centered on the band of the object the detector sees.  Two reasons
to change it: the object is larger than the detector's view, which leaves a bright ring at the
edge of the reconstruction and a bias inside, and MBIRTorch warns about it; or you want only part
of the volume, to save time and memory.

Do it in this order:

1. Set the detector parameters, if any need changing, such as the detector spacings or offsets.
2. Call ``ct_model.auto_set_recon_geometry()``.  This recomputes the box and the voxel size from
   the detector.  Without it the reconstruction comes out at the wrong scale, and it also resets
   any change made in the next step, so it comes first.
3. Resize the box with ``resize_recon_fov``, one scale factor per direction.  The voxel size stays
   the same; a factor above 1 enlarges the box and a factor below 1 shrinks it.

.. code-block:: python

    ct_model.set_params(delta_det_channel=0.1)
    ct_model.auto_set_recon_geometry()
    ct_model.resize_recon_fov(1.2, 1.2)              # 20 percent wider across the rotation axis

For a cone beam scan, the box can also be moved along the rotation axis with
``recon_slice_offset``, in ALU, with a positive value moving it down relative to the detector,
and padded at both ends with ``axial_pad_fraction``, a fraction from 0, no padding, to 1, out to
the farthest slice any measured ray reaches:

.. code-block:: python

    ct_model.set_params(recon_slice_offset=offset, axial_pad_fraction=0.5)

The box and the voxel size can also be set directly, with ``recon_shape`` and ``delta_voxel``;
they and the detector parameters are described in :ref:`ParametersDocs`.  All spacings are in
arbitrary length units, explained in :ref:`Unit Conversion <ALU_conversion_label>`.


Sinogram weighting
------------------

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
