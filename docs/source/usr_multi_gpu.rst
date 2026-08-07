.. _usr_multi_gpu:

Multi-GPU Reconstruction
========================

MBIRTorch can spread a single reconstruction across multiple GPUs (or, on a machine with no
GPU, across CPU devices).  This **increases the available memory**, so you can reconstruct
larger volumes than fit on one GPU, and on large problems it **reduces reconstruction
time**.  It works for the parallel-beam and cone-beam geometries.

Turning it on
-------------

Multi-device reconstruction is **opt-in**.  A model uses a single device until you widen it
with :meth:`~mbirtorch.TomographyModel.configure_devices`::

    model.configure_devices(num_devices=2)   # use the first 2 CUDA devices
    recon, recon_params = model.recon(sinogram)

This differs from MBIRJAX, which divides a reconstruction across all visible GPUs
automatically.  MBIRTorch asks you to choose, for three reasons.  Device placement is
explicit state in the engine: ``configure_devices`` is the single entry point that validates
a layout and rebuilds the per-device projectors, and a silent default would hide that step.
Multi-device runs also produce compiled-variant float differences within a documented
envelope, so changing the device count changes values slightly; that trade should be chosen
rather than inherited.  Finally, automatic widening on a shared or heterogeneous machine
risks out-of-memory failures, and the admission check that would catch them does not exist
yet.

Choosing the devices
--------------------

``configure_devices`` takes either a count or an explicit device list.  Use
``torch.cuda.device_count()`` to see how many CUDA devices are visible::

    import torch
    torch.cuda.device_count()                             # e.g. 4

    model.configure_devices(2)                            # the first 2 CUDA devices
    model.configure_devices(devices=["cuda:0", "cuda:2"]) # exactly these devices
    model.configure_devices(1)                            # back to a single device

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

There is no ``use_gpu`` parameter.  To run on the CPU when a GPU is present, construct the
model on the CPU or pass CPU entries in ``devices``.

Tips for efficiency
-------------------

* **Reuse a prepared sinogram.**  If you reconstruct the same large sinogram several times,
  :meth:`~mbirtorch.TomographyModel.prepare_sino_for_devices` distributes it across the devices once, so
  the host-to-device transfer is not repeated on every reconstruction.
* **Keep results on-device.**  Passing ``output_sharded=True`` to ``recon``, ``fbp_recon``, or
  ``fdk_recon`` returns the reconstruction in its device form (the slice shards), with no
  gather back to the host, so it can feed another on-device step directly.
* **Trade memory for time with a smaller band.**  Setting ``forward_project_slice_band`` or
  ``back_project_slice_band`` on the model streams the slice axis in smaller pieces.  This is
  a memory lever: on a measured 4-device 512-cell run it took peak memory from 6.6 GiB to
  2.6 GiB for about 8 percent more time.  Leave it unset unless a run is memory-constrained.
* **More devices is often slower.**  See the next section; this matters more in MBIRTorch
  than the equivalent advice does in MBIRJAX.

What to expect from more GPUs
-----------------------------

**Memory first.**  The reconstruction is sharded by slice, so the per-device share of the
volume falls as the device count rises.  Reconstructing a volume that does not fit on one
GPU is the primary reason to use more of them.

**Speed only on large problems.**  Reconstruction time does *not* scale linearly with the
device count, and on small problems more devices is markedly worse.  Measured on H100, over
a warm 3-iteration reconstruction:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Volume
     - 1 device
     - 2 devices
     - 4 devices
   * - 512 x 448 x 384
     - 3.12 s
     - 2.96 s
     - 9.11 s
   * - 1024 x 1008 x 992
     - 94.0 s
     - 70.7 s
     - 60.2 s

The large volume improves by 1.33x at two devices and 1.56x at four.  The small volume gets
*worse* at four devices, by roughly a factor of three.  Small reconstructions are
orchestration-bound: the per-band, per-device coordination grows with the device count and
outweighs the reduced compute per device.  These are hardware-dependent measurements, not
guarantees.  Reach for more devices when you need the memory, or when the problem is large.

Behind the scenes
-----------------

Internally MBIRTorch uses *sharding*: the sinogram is split across the devices by view and the
reconstruction volume by slice, and the two are combined with a small amount of banded
communication between devices.  When a count does not divide evenly, the data is zero-padded
to equal shares and the padding is kept exactly inert, so the result is independent of the
number of devices.  A reconstruction runs within a single process; multi-node execution is
out of scope.  For the developer-facing architecture, see :doc:`dev_sharding_overview`.
