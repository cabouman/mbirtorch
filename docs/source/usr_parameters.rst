.. _ParametersDocs:

==========
Parameters
==========

A model holds the parameters listed here.  Read one with ``get_params('name')``, change it with
``set_params(name=value)``, and print them all with ``print_params()``.  Lengths are in ALU
(arbitrary length units); see :ref:`param-units`.  The parameters particular to one geometry,
such as the source and detector distances of cone beam, are described with that geometry's
constructor on its own page.


The three you set
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``sharpness``
     - 1.0
     - Larger values give a sharper reconstruction, smaller values a smoother one.
   * - ``snr_db``
     - 30.0
     - The assumed signal-to-noise ratio of the sinogram in dB.  Larger values trust the data
       more and give a sharper, noisier reconstruction.
   * - ``positivity_flag``
     - False
     - If True, the reconstruction is constrained to be non-negative.


Reconstruction geometry
-----------------------

These are set for you by :meth:`~mbirtorch.TomographyModel.auto_set_recon_geometry`, which runs
when a model is built.  If you change a detector parameter with ``set_params``, call
``auto_set_recon_geometry`` again so that these follow.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``recon_shape``
     - from the sinogram
     - The array size of the reconstruction, (num_rows, num_cols, num_slices).
   * - ``delta_voxel``
     - from the detector
     - The voxel spacing.  It is set to ``delta_det_channel / magnification``, where the
       magnification is that of a voxel at the center of rotation.
   * - ``voxel_row_aspect``
     - 1.0
     - The voxel spacing along rows divided by the spacing along columns.
   * - ``voxel_slice_aspect``
     - 1.0
     - The voxel spacing along slices divided by the spacing along columns.
   * - ``use_ror_mask``
     - True
     - Which voxel columns are reconstructed.  True reconstructs the ellipse inscribed in the
       (row, col) plane, False the whole plane, and a 2D array of 0s and 1s of that shape
       reconstructs the voxel columns marked 1.

If the object extends past the detector laterally, the reconstruction warns.  Enlarge the
reconstruction with :meth:`~mbirtorch.TomographyModel.resize_recon_fov`, ``resize_recon_fov(s, s)``
with ``s`` of 1.1 or more.  For cone beam, the slice range can be padded with the cone beam
parameter ``axial_pad_fraction``.


Detector geometry
-----------------

Every geometry has these four.  They are given to the constructor or set with ``set_params``.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``delta_det_channel``
     - 1.0
     - The spacing between detector channels.
   * - ``delta_det_row``
     - 1.0
     - The spacing between detector rows.
   * - ``det_channel_offset``
     - 0.0
     - The offset of the center of rotation from the center of the detector, along the
       channels.
   * - ``det_row_offset``
     - 0.0
     - The offset of the source-to-detector line from the center of the detector, along the
       rows.


.. _param-units:

Units
-----

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``alu_unit``
     - None
     - The physical unit of one ALU, such as ``"cm"``.
   * - ``alu_value``
     - 1.0
     - The number of those units in one ALU.  With ``alu_unit = "cm"`` and ``alu_value = 0.5``,
       one ALU is 0.5 cm, which converts the geometry and the reconstruction to physical units.
       The preprocessing functions set both from the scanner's files.


Advanced
--------

These are rarely changed.  Setting ``sigma_x``, ``sigma_y``, or ``sigma_prox`` yourself turns
automatic regularization off (``auto_regularize_flag`` becomes False) so that your value is used;
setting ``sharpness`` or ``snr_db`` turns it back on.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``sigma_y``
     - 1.0
     - The assumed standard deviation of the sinogram noise.  Set automatically from the
       sinogram and ``snr_db``.
   * - ``sigma_x``
     - 1.0
     - The scale of the prior, which sets how much neighboring voxels are expected to differ.
       Set automatically from the sinogram and ``sharpness``.
   * - ``sigma_prox``
     - 1.0
     - The standard deviation of the proximal map prior used by
       :meth:`~mbirtorch.TomographyModel.prox_map`.  Set automatically when not given.
   * - ``auto_regularize_flag``
     - True
     - Whether ``sigma_y``, ``sigma_x``, and ``sigma_prox`` are set automatically.
   * - ``qggmrf_nbr_wts``
     - [1.0, 1.0, 1.0]
     - The relative strength of the regularization along rows, columns, and slices.
   * - ``p``, ``q``, ``T``
     - 2.0, 1.2, 1.0
     - The shape parameters of the QGGMRF prior: the potential grows with the power ``q`` of a
       small difference and the power ``p`` of a large one, and ``T`` sets where the change
       happens.
   * - ``max_alpha``
     - 1.5
     - The largest step the update of a voxel may take.
   * - ``granularity``
     - [1, 2, 4, ..., 128]
     - The numbers of subsets into which the voxels are divided for the partitions used by the
       reconstruction, from one subset (every voxel at once) to 128.
   * - ``partition_sequence``
     - [2, 4, 6, 7, 8, 9, 10, ...]
     - For each iteration, the index into ``granularity`` of the partition to use.  The default
       starts coarse and settles on the finest partitions.


Set by the model
----------------

These are recorded by the model and appear in ``print_params()``.  You do not set them.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Parameter
     - What it is
   * - ``geometry_type``
     - The name of the model class.
   * - ``sinogram_shape``
     - (num_views, num_det_rows, num_det_channels), given to the constructor.
   * - ``angles``
     - The view angles in radians, given to the constructor.
   * - ``verbose``
     - How much the model prints: 0 nothing, 1 progress, 2 or 3 more detail.  This one you may set.
   * - ``view_params_name``, ``file_format``
     - Bookkeeping for the saved parameter files.
