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


.. _usr_estimate_resources:

Checking memory and time before a job
-------------------------------------

``estimate_resources`` tells you how much GPU memory a reconstruction needs and about how long
it takes on the GPUs you name.  It needs no data and no GPU, so you can run it on a laptop
before you submit a job::

    estimate = mbirtorch.estimate_resources(ct_model, gpu_model='H100', num_gpus=8)
    print(estimate)

It reports the direct, full, and split reconstructions::

                                  GPU memory needed    fits    time, 15 iterations
    direct reconstruction         56.9 GiB             yes     about 1 minute
    full reconstruction           78.3 GiB             no      about 70 minutes
    split reconstruction          44.9 GiB             yes     about 71 minutes

Here the full reconstruction does not fit, so use ``recon_split_sino``.  Times are accurate to
about a factor of 1.5.

The full reference for ``estimate_resources`` is on the :ref:`TomographyModelDocs` page.


.. _usr_split_groups:

Split reconstruction on several GPUs
------------------------------------

``recon_split_sino`` reconstructs a volume too large for the GPUs in sections, and it uses the
GPUs by one rule: the GPUs are divided into equal groups, as many as possible, so that each group
can hold a section.  The groups reconstruct sections side by side, and each group takes its
sections one after another.

- With 8 GPUs, where a section needs 3, the GPUs can divide as 8 groups of 1, 4 of 2, 2 of 4, or
  1 of 8.  Groups of 1 or 2 are too small, so 2 groups of 4 work side by side.
- With 5 GPUs, where a section needs 3, the only equal groups are 5 of 1 or 1 of 5.  One GPU is
  too small, so all 5 reconstruct each section in turn.

For parallel beam a section keeps at least ``min_slices_per_section`` slices, 200 by default.
For cone beam the two halves are the sections, so there are 2 groups when a half fits on half the
GPUs, and otherwise 1.  The result's ``split_params`` reports the groups used.  A device choice
made with ``configure_devices`` limits the GPUs the split may use, and the grouping stays
automatic.


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
