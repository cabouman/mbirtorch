.. _ShardingOverview:

=================
Sharding Overview
=================

This page describes how MBIRTorch spreads a single reconstruction across several
devices.  It is a developer-oriented overview of the architecture -- enough to
follow how the pieces fit together and where they live in the code, not a
line-by-line specification.  For the user-facing view (how to turn it on and
control it), see :doc:`usr_multi_gpu`.

**Scope.**  Sharding runs within a single process across the chosen devices
(multiple GPUs, or CPU devices).  It works for all of the library's
geometries: parallel-beam, cone-beam, translation, and multi-axis parallel.  A
device count need not divide the axis it splits: the shards then differ in
length by at most one, and no data is padded.  Results can still differ slightly
with the device count, and the difference decays as iterations proceed.
Multi-node execution is out of scope.

The device layout is chosen automatically on CUDA with two or more visible
devices.  Each device count carries a measured speed floor
(``_widening_floors.py``), and a count below its floor is tried only after
every admitted count, so capacity still widens a reconstruction that needs the
memory.  :meth:`~mbirtorch.TomographyModel.configure_devices` is the explicit
door out of that choice, and ``configure_devices(num_devices=1)`` is the
reproducibility pin.  :doc:`usr_multi_gpu` states both rules in full.


The two shardings
-----------------

MBIRTorch shards the two large arrays along complementary axes:

* the **reconstruction is sharded by slice** -- each device (a *slice-owner*)
  holds a contiguous band of slices of the voxel cylinders,
  ``(num_pixels, slices_per_device)``;
* the **sinogram is sharded by view** -- each device (a *view-owner*) holds a
  block of views and is responsible for producing *all* detector rows for those
  views.

.. figure:: figs/sharding-structure.png
   :width: 90%
   :align: center

   The recon is sharded by slice (left); the sinogram by view (right).  Every
   slice-shard must reach every view-shard, so projection is an all-to-all
   between the two layouts.

Because the two arrays are sharded on different axes, projecting between them is
an all-to-all: every slice-shard contributes to every view-shard, and vice
versa.  Neither projector moves the whole volume at once, but the two bound
different things.  The forward bounds the **transfer**: a view-owner brings over
one batch of voxel cylinders at a time, and each cylinder spans the whole slice
axis, so the projector call itself covers every slice-shard together.  The back
projection bounds **both**, because it processes one slice-shard at a time and
one band of slices within that shard.


Placement and Shards: the device layout and the device form
------------------------------------------------------------

Device layout is described by a single object, ``Placement`` (in
``mbirtorch/_sharding.py``).  A placement records:

* the list of ``devices``;
* which array ``axis`` is sharded;
* the ``axis_len``, the length of that axis, which ``shard_ranges`` splits into
  one contiguous block per device.  The blocks differ in length by at most one, and
  the longer blocks come first.

A model carries two placements -- ``recon_placement`` (slice axis) and
``sino_placement`` (view axis) -- and these are the **single source of truth**
for where every array lives; there is no separate "main device" or "is-sharded"
state.  A single device is the trivial one-shard case, and the placement
functions return plain tensors there, so the ``n = 1`` path is unchanged.

The *device form* of a sharded array is a ``Shards`` container: the per-device
tensors plus their placement.  Torch's logically-global sharded array, DTensor, was deliberately rejected as immature
for the index-heavy kernels in MBIRTorch, so there is no global array here.  ``Shards`` is a
plain container instead: it holds the tensors, checks on construction that each
shard really lives on its placement's device, and exposes ``gather()`` as the host
exit.  The drivers and the VCD loop operate on the per-device tensors directly.

One layout is refused: a device holding no data on **either** axis, since it
would do no work.  That happens exactly when the device count exceeds both the
view count and the slice count.  A device idle on only one axis is legal and
useful.  With fewer slices
than devices the extra devices still project their views, which dominates there;
with fewer views than devices they still hold slice shards and run the prior and
updates, which dominate there (``_check_no_empty_shard`` in
``tomography_model.py``).


The forward's cylinder transfer
-------------------------------

**Forward projection** (recon → sinogram) is the **cylinder transfer**, on all four
geometries: parallel-beam, cone-beam, translation and multi-axis parallel.
Each view-owner partitions the pixel axis into batches, collects each batch's
full-height voxel cylinder from every slice-owner, and makes one projector call
per batch over its own views and the whole slice range.  When the loop
finishes, the sinogram is already in the view-sharded layout.

What one gather costs is set by the width of the pixel batch and not by the
device count, so the transient does not grow with the problem at a fixed batch.
``forward_project_pixel_batch`` on the model sets that batch
(``transfer_cylinder_batch`` in ``_sharding.py`` and
``_sparse_forward_project_cylinders`` in ``tomography_model.py``).


Back projection by bands
-------------------------

Within a slice-shard the back projection's work is subdivided once more, into
**bands** of slices.  It runs a double loop: over slice-shards (outer) and over
bands (inner).  Banding bounds the size of the transient buffers that arise
while moving data between the two shardings.

.. figure:: figs/sharding-back-bands.png
   :width: 90%
   :align: center

   Back projection: each view-owner back-projects to a single band; the
   contributions are reduced (summed) over views into the recon's slice
   sharding.

**Back projection** (sinogram → recon) is the forward's adjoint: each
view-owner back-projects into a band, and the per-view contributions are
reduced (summed) and scattered into the slice-owner that holds those slices --
a reduce-scatter, ``sum_band_to_owner``.  The result is already in the
slice-sharded recon layout.

**The owner adds the partials a slab of rows at a time.**  It allocates the
running total for the band once, then adds each arriving partial into that total
in slabs of at most 256 MiB.  So beyond the running total the owner holds one
slab per source, and a slab is the same number of bytes however many devices
there are.  The simpler alternative is to move all n partials across first and
add them together; that holds n whole bands at once, and since each band is 1/n
of the volume, it costs the same at every device count and never gets smaller.
Adding in slabs changes neither which numbers are added nor the order they are
added in, so the result matches the one-shot sum bit for bit.

**The default band is the whole shard.**  Each band costs a fixed amount of
orchestration, so dividing a shard into several bands costs time.  With the
compiled kernels in place, narrower bands measured 2 to 23 percent more busy
time at parallel 1024 on two devices, depending on the band length.  An earlier
reading of 47 to 66 percent predates those kernels and overstated the cost.  A
single device never reaches this driver at all, because the trivial placement
uses the plain projectors.  A smaller band is still a real **memory** lever,
since each per-band partial scales with the band, and the band also sets the
size of the running total the slabs are added into.  Bands already reduced in
this pass are held either way.  Set ``back_project_slice_band`` on the model to
opt in (``_slice_band_length`` in ``tomography_model.py``).


Why cone beam projects whole cylinders
--------------------------------------

In parallel-beam geometry detector row ``r`` maps only to slice ``r`` with no
cross-row mixing, so a device can produce just the slices a destination needs by
first slicing its sinogram views to those rows -- the transient cylinder buffer
can be held at one destination's slice range.  ``ParallelBeamModel`` declares this
with ``rows_track_slices = True``.

Cone beam cannot do this cheaply: magnification maps a single slice to a
*data-dependent band* of detector rows (dependent on the cone angle and the
particular slice), so the slice and detector-row axes are coupled.  This
projection could proceed one band of slices at a time, but then each detector row
requires multiple updates from a single voxel cylinder.  Alternatively, each view
owner can collect all the bands for a batch of pixels and then project the
resulting full cylinders.

A view-owner therefore collects the **full voxel cylinder** once and then projects
the full cylinder, rather than restricting the computation to a band up front.
This coupling is also why the sinogram is sharded by view (not by detector row)
uniformly across geometries.


Projector kernels
-----------------

The kernels that run within each band or shard -- the torch.compile view-batch
bodies, the shared horizontal-fan contract, and the optional hand-written Triton
kernels layered on top of them -- are described in :doc:`dev_projector_kernels`.


The QGGMRF prior and halos
--------------------------

The QGGMRF prior is local: each voxel interacts only with its neighbors.  Across
slices that means a slice-owner can evaluate the prior on its own band entirely
locally **except** at the boundary between two adjacent slice-shards, where it
needs the neighbor's boundary slice.  Those boundary slices are exchanged as
**halos** between neighboring owners: the last slice of the shard to the left and
the first slice of the shard to the right, each moved to the receiving shard's
device.  At the outer edge of the volume there is no neighbor, so the boundary
condition is reflection; a single shard is reflected at both edges and needs no
halos at all (``exchange_qggmrf_halos`` in ``_sharding.py``).

Halo staging happens host-side and so cannot live inside a compiled region, which
drives the single- vs multi-device split below.


Single- and multi-device paths
-------------------------------

The prior-bearing reconstruction loop takes one of two paths depending on the
device count:

* **One device** -- the trivial placement, the plain projectors, and the compiled
  per-device bodies.  A single shard is reflected at both edges, so no halos are
  needed.
* **Several devices** -- a Python loop runs the passes, staging the halos
  host-side once per pass (the part that cannot be compiled), while the
  on-device per-pass work stays compiled.

Both paths converge to the same result; only the boundary handling and the loop
structure differ.

The QGGMRF denoiser takes the same two paths.  ``QGGMRFDenoiser.denoise``
selects a device layout through the same policy a reconstruction uses, and on
several devices it slice-shards the image and runs the sweep shard by shard,
staging the qGGMRF halos once per pass (``_denoise_sharded`` in
``denoising.py``).  Sharding the denoiser was measured slower than one device
at every size probed, so the automatic path widens a denoise only when it does
not fit on one device.


Multi-device execution: thread pools
------------------------------------

The steps above all share a common substrate: run a kernel on each device, then
combine the per-device outputs.  Within a single step the per-device kernels are
independent (each device works on its own shard or band); the cross-device
communication -- the broadcast in forward projection, the reduce-scatter in back
projection -- happens *between* steps, not inside them.  The design problem is to
launch those per-device kernels and reassemble their outputs **without bouncing
data through host memory**.

MBIRTorch uses a **thread pool with one worker per device**: each thread operates
on its device's tensors in place, runs the per-device kernel, and keeps its result
on-device (``device_pool`` / ``run_per_device`` in ``_sharding.py``).  Torch needs no thread-local
default device, because tensors carry their own device, so a worker simply operates
on the tensors it is given.  And a single device short-circuits to a direct call on
the calling thread, with no pool and no thread hop, so code written uniformly over
shards costs a plain function call at ``n = 1``.

``run_per_device`` performs no synchronization.  Results come back in device order
rather than completion order, and a caller that needs values materialized
synchronizes explicitly.  That is what lets a caller overlap the next band's
transfer with the current band's compute.


Safe cross-device transfer
--------------------------

All cross-device movement goes through one primitive, ``move_shard``.  It exists
because JAX's ``device_put`` silently corrupted device-resident transfers on some
GPUs (L40S).  Torch's ``tensor.to()`` has no known equivalent failure, but the
empirical probe is kept as near-free paranoia: ``is_dev2dev_safe`` moves a small
known array between two devices once per configuration and checks that the value
survives.  When the probe fails, ``move_shard`` routes transfers through host
memory, which is always correct and slower, and warns once per process.

The principle is worth stating on its own: the hardware is tested rather than
assumed, once per device set, at a cost of microseconds.


Where this lives in the code
-----------------------------

* ``mbirtorch/_sharding.py`` -- ``Placement``, ``Shards``, the transfer
  primitives, the forward's cylinder transfer, the back projection's band reduce,
  the halo exchange, and the thread-pool execution helpers.  MBIRJAX splits
  these across a ``_sharding/`` package; here they are one module.
* ``mbirtorch/tomography_model.py`` -- the placement chokepoints, the sharded
  drivers, the band-length and pixel-batch policies, and the sharded VCD
  engine.
* ``mbirtorch/projectors.py`` -- the geometry-agnostic view-range drivers and the
  torch.compile plumbing.
* ``mbirtorch/horizontal_fan.py`` -- the shared horizontal-fan kernels and the
  fan data contract.
* the per-geometry view-batch bodies in ``parallel_beam.py``, ``cone_beam.py``,
  ``multiaxis_parallel.py``, and ``translation_model.py``.

The correctness gates for the sharding invariants are in ``tests/test_sharding.py``.
The strongest of them compare against a single-device reference: the sharded
projectors, the cone FDK, and the full VCD reconstruction must each match the
single-device result, and must still match when the axes do not divide evenly
and the shards differ in length.  The adjoint property is checked directly on
the projector pair, the halo exchange is checked against a full-volume prior,
and the awkward layouts (more devices than slices, more devices than views, a
fully idle device) each have their own test.
