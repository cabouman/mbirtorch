.. _usr_multi_gpu:

Multi-GPU Reconstruction
========================

MBIRTorch can spread a single reconstruction across multiple GPUs (or, on a machine with no
GPU, across CPU devices).  This **increases the available memory**, so you can reconstruct
larger volumes than fit on one GPU, and on large problems it **reduces reconstruction
time**.  It works for all of the library's geometries: parallel-beam,
cone-beam, translation, and multi-axis parallel.  :class:`~mbirtorch.QGGMRFDenoiser`
follows the same rules through :meth:`~mbirtorch.QGGMRFDenoiser.denoise`, with one
practical difference.  No measured problem size makes a multi-device denoise faster,
so an automatic denoise spreads only when it cannot fit on one device.

The default: spread across the visible GPUs
-------------------------------------------

Multi-device reconstruction is **automatic**.  On a machine with two or more
GPUs, a reconstruction can use them with no change to your script::

    recon, recon_dict = model.recon(sinogram)   # uses the GPUs that fit

However, depending on the problem, the code may not use all available GPUs.  The
automatic choice is made once per model, at its first reconstruction.  Later
reconstructions on the same model reuse the layout, and the choice runs again only when
the model's sinogram or reconstruction shape changes.

The automatic choice applies two rules, in this order.

**Speed first.**  More devices is not always faster: below a certain problem size the
per-device work no longer covers the cost of splitting it, and a small reconstruction
spread over four GPUs can run several times slower than the same reconstruction on one.
So each device count carries a measured **speed floor** -- a problem size, in sinogram
elements, below which the automatic path does not prefer that count.
``configure_devices`` can be used to set the number of GPUs explicitly.

**Capacity always wins.**  A count below its floor is set aside, not discarded.
Before the first large allocation, MBIRTorch estimates the memory each candidate layout
would need and takes the largest *preferred* count whose per-device share fits, with a
safety margin; if none of them fits, it falls back through the set-aside counts rather
than refusing to run.  So a reconstruction that genuinely needs four GPUs still gets
them.  If no layout fits at all, the reconstruction fails immediately with the shortfall
named, rather than failing mid-run with an out-of-memory error.

**Feedback.** The run log's device line names any count the automatic choice left on the table and
why, so idle GPUs are never silent -- either because a count is below its speed floor or
because it did not fit.  Both ways of naming the devices yourself --
:meth:`~mbirtorch.TomographyModel.configure_devices` and ``MBIRTORCH_NUM_DEVICES`` --
use what you specify.  They differ on the memory check: ``configure_devices`` skips it,
while a count pinned by the environment variable is still checked and can still be
refused.  To
keep the automatic search but disable the speed floors alone, set
``MBIRTORCH_WIDENING_GUARD=0``, which restores the pure largest-count-that-fits
behavior.

Results can differ slightly with the device count, and the difference decays as iterations
proceed (measured to fall from 6.1e-3 at 3 iterations to 8.8e-4 at 10).  To pin a run to one
device for exact reproducibility::

    model.configure_devices(num_devices=1)

The environment variable ``MBIRTORCH_NUM_DEVICES`` pins the device count process-wide,
which is convenient for a test suite or a batch queue.

Choosing the devices
--------------------

To take the layout out of the library's hands, call
:meth:`~mbirtorch.TomographyModel.configure_devices`.  A layout set this way is final: the
automatic choice never runs again on that model.  It takes a count, an explicit device list, or
another model to match.  Use
``torch.cuda.device_count()`` to see how many CUDA devices are visible::

    import torch
    torch.cuda.device_count()                             # e.g. 4

    model.configure_devices(2)                            # the first 2 CUDA devices
    model.configure_devices(devices=["cuda:0", "cuda:2"]) # exactly these devices
    model.configure_devices(1)                            # back to a single device
    other_model.configure_devices(like=model)             # match this model's devices

Passing a count requires that many CUDA devices to be visible, and raises otherwise.  Pass
an explicit ``devices`` list to select particular GPUs, for example to leave one free for
another job on the same node.  A list of CPU entries such as ``["cpu", "cpu"]`` runs the
multi-device paths on the CPU, which is useful for testing without a GPU.

A configuration set this way is kept.  A later geometry-changing ``set_params`` rebuilds the
placements from the new shapes while preserving the devices you chose.

One layout is refused: a device that would hold no real data on **either** axis, because it
would do no work at all.  Layouts where a device is idle on only one axis are allowed and
useful.  With fewer slices than devices, the extra devices still project their views; with
fewer views than devices, they still hold slice shards and run the prior.

To run on `num_cpus CPUs` when a GPU is present, call
``model.configure_devices(devices=['cpu']*num_cpus)``.

Matching one model to another
-----------------------------

``configure_devices(like=other_model)`` copies another model's devices.  This matters when
two models work on the same volume, as a Plug-and-Play or ADMM loop does when it alternates
:meth:`~mbirtorch.TomographyModel.prox_map` on a reconstruction model with
:meth:`~mbirtorch.QGGMRFDenoiser.denoise` on a denoiser.  Placed on the same devices, the
two models pass the volume between them in its device form, so it stays where it was
computed instead of being gathered to the host and scattered again on every half-iteration::

    denoiser = mbirtorch.QGGMRFDenoiser(ct_model.get_params('recon_shape'))
    denoiser.configure_devices(like=ct_model)

    volume, _ = ct_model.prox_map(volume, sinogram, output_sharded=True)
    volume, _ = denoiser.denoise(volume, output_sharded=True)

Build the denoiser at the other model's **recon** shape, as above, not its sinogram shape.
The two models must agree on that shape, and ``configure_devices`` raises naming both shapes
if they do not.  Configure the model being copied first: one still on the automatic layout
has not chosen its devices yet, and the pair would diverge when it settles.

This makes reconstruction volumes interchangeable, not sinograms.  A denoiser's "sinogram"
is its image, so its sinogram placement divides an image by slice, while a projection
model's divides a sinogram by view; those are different things.  To place two models on the
same devices without exchanging arrays, pass ``devices=`` instead.

A denoiser configured this way runs on the devices you gave it.  The automatic choice would
hold a denoise to one device at any size, because splitting one has never been measured
faster; that measurement is of a call that returns its result to the host, and a loop
keeping the volume on the devices does not pay what made splitting lose.

Tips for efficiency
-------------------

* **Reuse a prepared sinogram.**  If you reconstruct the same large sinogram several times,
  :meth:`~mbirtorch.TomographyModel.prepare_sino_for_devices` distributes it across the devices once, so
  the host-to-device transfer is not repeated on every reconstruction.
* **Keep results on-device.**  Passing ``output_sharded=True`` to ``recon``, ``recon_fbp``, or
  ``recon_fdk`` returns the reconstruction in its device form (the slice shards), with no
  gather back to the host, so it can feed another on-device step directly.
* **Trade memory for time with a smaller band.**  Setting ``back_project_slice_band`` on the
  model streams the back projection's slice axis in smaller pieces.  This is a memory lever:
  a measured 2-device run at the 1024 class saved about 0.5 GB of per-device peak for about
  2 percent more time at the 252-slice band, and narrower bands saved slightly more memory
  for more time.  Leave it unset unless a run is memory-constrained.  The forward projection
  has its own lever, ``forward_project_pixel_batch``, which sets how many voxel cylinders it
  moves between devices at once.
* **More devices is sometimes slower.** This mostly concerns small problems. See the next section.

What to expect from more GPUs
-----------------------------

**Memory first.**  The reconstruction is sharded by slice, so the per-device share of the
volume falls as the device count rises.  Reconstructing a volume that does not fit on one
GPU is the primary reason to use more of them.

**Speed only on large problems.**  Reconstruction time does *not* scale linearly with the
device count, and on small problems more devices is worse.  Measured on H100 for a
parallel-beam model, over a warm 3-iteration reconstruction:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Volume
     - 1 device
     - 2 devices
     - 4 devices
   * - 512 x 448 x 384
     - 1.31 s
     - 1.29 s
     - 2.10 s
   * - 1024 x 1008 x 992
     - 21.3 s
     - 14.2 s
     - 10.8 s

The large volume improves by 1.5x at two devices and 2.0x at four.  The small volume shows
no real gain at two devices and is *worse* at four devices than at one.  Small reconstructions
are orchestration-bound: the per-band, per-device coordination grows with the device count
and outweighs the reduced compute per device.  These are hardware-dependent measurements,
not guarantees.  Reach for more devices when you need the memory, or when the problem is
large.

Behind the scenes
-----------------

Internally MBIRTorch uses *sharding*: the sinogram is split across the devices by view
and the reconstruction volume by slice.  Projecting between the two layouts requires
communication between devices, and the projectors move the data in bounded pieces, so no
single transfer holds the whole volume.  A device count need not divide the sinogram or
the volume evenly; the shares then differ in size by at most one view or one slice.
A reconstruction runs within a single process; multi-node execution is
out of scope.  For the developer-facing architecture, see :doc:`dev_sharding_overview`.
