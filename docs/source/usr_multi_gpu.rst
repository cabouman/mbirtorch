.. _usr_multi_gpu:

Multi-GPU Reconstruction
========================

MBIRTorch can spread a single reconstruction across multiple GPUs (or, on a machine with no
GPU, across CPU devices).  This **increases the available memory**, so you can reconstruct
larger volumes than fit on one GPU, and on large problems it **reduces reconstruction
time**.  It works for the parallel-beam and cone-beam geometries.

The default: spread across the visible GPUs
-------------------------------------------

As in MBIRJAX, multi-device reconstruction is **automatic**.  On a machine with two or more
CUDA devices, a reconstruction spreads across them with no change to your script::

    recon, recon_dict = model.recon(sinogram)   # uses the GPUs that fit

Before the first large allocation, MBIRTorch estimates the memory each candidate layout
would need and picks the largest device count whose per-device share fits, with a safety
margin.  If no layout fits, the reconstruction fails immediately with the shortfall named,
rather than failing mid-run with an out-of-memory error.  Two model attributes tune this
check: ``model.skip_memory_preflight = True`` runs without it, and
``model.memory_preflight_margin`` (default 0.15) is the fraction by which the estimate is
padded before it is compared with the free memory.

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
automatic choice never runs again on that model, and the memory check no longer
second-guesses the count you gave.  It takes either a count or an explicit device list.  Use
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

There is no ``use_gpu`` parameter.  To run on the CPU when a GPU is present, call
``model.configure_devices(devices=['cpu'])``.

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
to equal shares and the padding is kept exactly inert, so the padding never changes the
result.  A reconstruction runs within a single process; multi-node execution is
out of scope.  For the developer-facing architecture, see :doc:`dev_sharding_overview`.
