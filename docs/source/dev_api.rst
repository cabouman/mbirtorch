Developer API reference
=======================

**MBIRTorch** can be extended with new scanner geometries by subclassing
:ref:`TomographyModelDocs`.  The best reference is the existing geometry classes --
:class:`~mbirtorch.ParallelBeamModel` and :class:`~mbirtorch.ConeBeamModel`.
They are the canonical, tested templates; this page summarizes what a new geometry
must provide.

Core projector interface
------------------------

A geometry supplies two projection *bodies* -- module-level functions that project a
batch of views at a time -- plus a small amount of geometry plumbing:

* A **forward body**: project a batch of voxel cylinders into a batch of sinogram views.
* A **back body**: back project a batch of views onto the voxel cylinders, summed over
  the batch.  It takes a ``coeff_power`` argument (1 for the gradient, 2 for the
  Hessian diagonal).
* ``_view_batch_bodies`` -- return the (forward, back) pair.  The bodies must be
  module-level pure functions, never bound methods: the driver compiles them with
  ``torch.compile``, one instance per device, and a bound method would pin the model in
  the compile cache.
* ``_view_batch_args`` -- the keyword arguments the bodies need, read from the
  parameters on every call so a ``set_params`` change is always picked up.
* ``verify_valid_params``, ``get_magnification``, ``get_psf_radius``, and
  ``auto_set_recon_geometry`` -- parameter validation, the iso-to-detector scale
  factor, the projection footprint radius, and the default reconstruction geometry.
* ``_transient_cols`` -- the column count of the geometry's dominant per-view
  transient, which is what the driver's view-batch budget divides.  The base class
  returns the runtime band length, and that is right only for a row-aligned
  geometry (parallel beam).  The other three shipped geometries -- cone beam,
  translation, and multi-axis parallel -- override it with a width derived from the
  parameters, because their bodies hold a transient of the full slice and detector
  height whatever band is asked for.  A geometry that leaves the base value in
  place when it should not lets the driver oversize the view batch, so the run
  peaks above what the memory check priced it at.

The driver (``projectors.py``) owns everything else: the view-batch loop, the budget
arithmetic, per-device compiled instances, and the compile lock.  The skeleton at the
bottom of this page shows the signatures.

Multi-device (sharded) support
------------------------------

Reconstructions are spread across the available devices automatically -- the recon by
slice and the sinogram by view (see :doc:`dev_sharding_overview` for the architecture).
Most of the machinery is inherited from :ref:`TomographyModelDocs`; a new geometry
usually needs only:

* **The banded seams.**  The bodies take ``slice_start`` and ``band_slices`` arguments
  through which the multi-device back projection asks for a band of slices at a time.
  A geometry whose detector row ``r`` maps one-to-one to recon slice ``r`` (parallel
  beam) sets the class attribute ``rows_track_slices = True``; its band then rides in
  the sinogram's row axis and the bodies can assert the seam defaults.  A geometry
  where one slice spreads over a range of detector rows (cone beam) must honor
  ``slice_start`` and ``band_slices`` explicitly.
* **Uneven shards** -- a device count need not divide the sharded axis, so any
  per-slice or per-view operation must be written against the length of the block it
  is handed, and must accept a block of length zero.  This is the one subtlety that
  must be correct for sharding.

:class:`~mbirtorch.ParallelBeamModel` (row-aligned) and :class:`~mbirtorch.ConeBeamModel`
(banded, with ``slice_start`` honored throughout) are the two worked examples to study.

Geometry skeleton
-----------------

The following is a starting skeleton for the interface above.  It is **not** the whole
picture -- study the existing geometry classes and :doc:`dev_sharding_overview` for the
multi-device pieces, and :doc:`dev_projector_kernels` for how a geometry can add
hand-written kernels later.

.. include:: _static/new_model_template.py
   :code: python
