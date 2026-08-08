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

The driver (``projectors.py``) owns everything else: the view-batch loop, the memory
budget, per-device compiled instances, and the compile lock.  The skeleton at the bottom
of this page shows the signatures.

Multi-device (sharded) support
------------------------------

Reconstructions are spread across the available devices automatically -- the recon by
slice and the sinogram by view (see :doc:`dev_sharding_overview` for the architecture).
Most of the machinery is inherited from :ref:`TomographyModelDocs`; a new geometry
usually needs only:

* **The banded seams.**  The bodies take ``slice_start`` and ``band_slices`` arguments
  through which the multi-device drivers ask for a band of slices at a time.  A
  geometry whose detector row ``r`` maps one-to-one to recon slice ``r`` (parallel
  beam) sets the class attribute ``rows_track_slices = True``; its band then rides in
  the sinogram's row axis and the bodies can assert the seam defaults.  A geometry
  where one slice spreads over a range of detector rows (cone beam) must honor
  ``slice_start`` and ``band_slices`` explicitly.
* **Inert padding** -- any per-slice or per-view operation must be written against the
  device-form (padded) length, not the real count.  This is the one subtlety that must
  be correct for sharding.

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
