.. _usr_multi_gpu:

Multi-GPU Reconstruction
========================

MBIRTorch can spread one reconstruction across several GPUs.  This gives two things.  A volume
that does not fit on one GPU can be reconstructed, and a large reconstruction runs faster.  On a
small problem, more GPUs do not help and can be slower, because the cost of coordinating the
devices outweighs the work saved on each one.


What happens by default
-----------------------

On a machine with two or more GPUs, a reconstruction can use them with no change to your
script::

    recon, recon_dict = ct_model.recon(sinogram)

The device count is chosen at the first reconstruction on a model.  MBIRTorch takes the largest
count that is expected to be faster at that problem size and whose share of the volume fits in
each GPU's memory.  A count that is not expected to be faster is still used when it is the only
way to fit.  If nothing fits, the reconstruction stops at once and reports how much memory is
short.  The run log's device line reports the
count chosen and why any available GPU was not used.

Later reconstructions on the same model reuse the choice.  It is made again only when the
sinogram shape or the reconstruction shape changes.


Choosing the devices yourself
-----------------------------

Call ``configure_devices`` to set the devices.  A layout set this way is kept, and the automatic
choice does not run again on that model.  The call takes a device count, a list of devices, or
another model to copy::

    import torch
    torch.cuda.device_count()                                # the GPUs visible

    ct_model.configure_devices(2)                            # the first two GPUs
    ct_model.configure_devices(devices=['cuda:0', 'cuda:2']) # exactly these two
    ct_model.configure_devices(1)                            # one GPU
    denoiser.configure_devices(like=ct_model)                # the same devices as ct_model

Pass a list to leave a GPU free for another job on the same machine.  A list of CPU entries,
such as ``['cpu', 'cpu']``, runs the multi-device code on the CPU, which is useful for testing.

To set the device count for a whole process, for example a batch job, set the environment
variable ``MBIRTORCH_NUM_DEVICES``.


Getting the most out of several GPUs
------------------------------------

- **Reconstruct large volumes.**  The volume is divided by slice, so each GPU holds one share of
  it.  A volume that does not fit on one GPU is the main reason to use more.
- **Send the sinogram once.**  If you reconstruct the same sinogram several times, for example
  while adjusting parameters, call ``prepare_sino_for_devices`` once and pass its result to each
  reconstruction.  The copy from the host to the GPUs then happens once.
- **Keep results on the GPUs.**  Pass ``output_sharded=True`` to a reconstruction method to get
  the result in its on-device form instead of a numpy array.  A loop that alternates a
  reconstruction step with another on-device step, such as a Plug-and-Play loop, avoids copying
  the volume to the host and back each time.  The two models must be on the same devices, which
  ``configure_devices(like=ct_model)`` arranges.
- **Lower peak memory if a run does not fit.**  Setting ``back_project_slice_band`` on the model
  to a number of slices makes the back projection work in shorter bands, which lowers the memory
  each GPU needs at some cost in time.  Leave it unset otherwise.


Reproducibility
---------------

The result of a reconstruction differs slightly with the device count, and the difference
shrinks as the iterations proceed.  To get the same result every time, pin the run to one
device with ``ct_model.configure_devices(1)`` and seed numpy's random number generator before
calling ``recon``.

For how the work is divided among the devices, see :doc:`dev_sharding_overview`.
