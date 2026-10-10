.. _DemosFAQs:

==============
Demos and FAQs
==============

Demos
-----

The demo scripts are in the `demo folder <https://github.com/cabouman/mbirtorch/blob/main/demo/>`__.
Follow the installation instructions in :ref:`InstallationDocs`, then run any script directly.
Each is short and self-contained; adjust the parameters near the top and rerun to see their effect.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Script
     - What it demonstrates
   * - ``demo_1_parallel_basics.py``
     - The basic pipeline: make a phantom, project it to a sinogram, reconstruct, view.
   * - ``demo_2_cone_beam.py``
     - Cone beam geometry, simulated measurement noise, noise weighting, saving results.
   * - ``demo_3_helical.py``
     - Helical cone beam scanning and reconstruction.
   * - ``demo_4_multiaxis.py``
     - The multi-axis parallel geometry (laminography): tilted views and their reconstruction.
   * - ``demo_5_mace4d.py``
     - 4D reconstruction of a moving object, one volume per time frame, against FBP per window.
   * - ``demo_6_view_selection.py``
     - Sparse view selection: pick the few view angles that reconstruct a reference object best.
   * - ``demo_7_hyperspectral.py``
     - Hyperspectral reconstruction: dehydrate the scan to three components, reconstruct them, rehydrate, against FBP per wavelength.
   * - ``demo_8_autograd.py``
     - The differentiable projectors: reconstruction by gradient descent in PyTorch.


FAQs
----

Q: How do I load my scanner's data?
+++++++++++++++++++++++++++++++++++

A: Scanner data arrives as a folder of radiographs with a blank scan, a dark scan, and the
scanner's description of the geometry, and turning that into a sinogram and a correctly set up model
takes several steps that are easy to get wrong.  MBIRTorch has a scanner loader for each supported
instrument: a ``get_sino_and_model`` function that does all of the steps in one call and returns the
sinogram and a model ready to reconstruct, for example
``mbirtorch.preprocess.nsi.get_sino_and_model(dataset_dir)``.  The loaders and the instruments they
support are listed in :ref:`ScannerLoaders`.  For an instrument without a loader, the functions the
loaders use are available on their own; see :ref:`PreprocessDocs`.

Q: How can I check my scan geometry before reconstructing?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: Open the geometry viewer on the model: ``mbirtorch.geometry_viewer(ct_model)``.
It draws the source, the detector, the reconstruction volume, and the
rotation axis for one view at a time, marks detector pixel (0, 0) and
voxel (0, 0, 0), and reports whether the region of reconstruction
projects inside the detector.  Pass ``sinogram=`` to paint a view of the
data on the detector face, and ``compare=dict(det_channel_offset=...)``
to draw a second geometry over the first.  See :ref:`GeometryViewerDocs`.

Q: Why is there a bright ring around my reconstruction?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: If the object does not project completely inside the detector, then MBIR will produce a bright ring
around the edge of the reconstruction to account for the portion of the object that projects to the detector in only some views.
The reconstruction warns with "Lateral FOV truncation detected" when it sees this.  At every view angle, material
outside the field of view contributes to the measurements, but no reconstruction voxel is available to explain it.
The result is a bright ring at the reconstruction boundary, a bias across the whole interior, and slowed
convergence.

You can improve the reconstruction by enlarging the region of reconstruction:

.. code-block:: python

        ct_model.resize_recon_fov(1.2, 1.2)

Note that the scale factor need only be large enough to give some padding around the region of valid projection --
it does not need to match the size of the true object.  Larger scale factors will lead to increased time and memory.
The next question explains the region of reconstruction and the ways to change it.

Q: How do I make the region of reconstruction larger or smaller, or move it up or down?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: The region of reconstruction is the box of voxels the reconstruction fills.  When a model is
built, it is set from the detector: across the rotation axis it covers what the detector sees, and
along the rotation axis it is centered on the band of the object the detector sees.  Its size is
the parameter ``recon_shape``, in voxels, and its voxel spacing is ``delta_voxel``.

**To make it larger or smaller**, call ``resize_recon_fov`` with one scale factor per direction,
rows, columns, and slices.  The voxel size stays the same and the number of voxels is multiplied by
the factor, so a factor above 1 enlarges the region and a factor below 1 shrinks it, and the region
stays centered where it was:

.. code-block:: python

        ct_model.resize_recon_fov(1.2, 1.2)          # 20 percent wider across the rotation axis
        ct_model.resize_recon_fov(1.0, 1.0, 0.5)     # half as tall along the rotation axis

Enlarging it is the fix when the object extends past the detector, see the question above.
Shrinking it saves time and memory when only part of the volume is of interest.

**To move it up or down** along the rotation axis, in a cone beam scan, set ``recon_slice_offset``
in ALU.  A positive value moves the region down relative to the detector, toward the higher
detector row indices.  Shrinking and moving together reconstruct one part of a tall object:

.. code-block:: python

        ct_model.resize_recon_fov(1.0, 1.0, 0.5)             # the central half of the slices
        ct_model.set_params(recon_slice_offset=offset)       # moved to the part you want, in ALU

A cone beam region can also be padded at both ends along the rotation axis with
``axial_pad_fraction``; see :ref:`ParametersDocs`.

**Do these last.**  If you change a detector or geometry parameter after building the model, such
as ``delta_det_channel``, ``delta_det_row``, ``det_channel_offset``, ``det_row_offset``, or the source
distances, call ``ct_model.auto_set_recon_geometry()`` right after, and only then resize or move the
region.  That call recomputes ``recon_shape``, ``delta_voxel``, and ``recon_slice_offset`` from the
detector, so it undoes any resizing or moving done before it; and without it the region keeps the
old voxel size, and the reconstruction comes out at the wrong scale.  See the next question.

Q: I changed a detector parameter and now my reconstruction is wrong.  Why?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: The reconstruction geometry, the voxel size and the region of reconstruction, is computed from
the detector parameters once, when the model is built.  Changing a detector parameter afterwards
with ``set_params`` does not recompute it, so the voxels no longer match the detector and the
reconstruction is distorted or at the wrong scale.  After any change to ``delta_det_channel``,
``delta_det_row``, ``det_channel_offset``, ``det_row_offset``, ``source_detector_dist``, or
``source_iso_dist``, call:

.. code-block:: python

        ct_model.set_params(delta_det_channel=0.1)
        ct_model.auto_set_recon_geometry()

Then resize or move the region of reconstruction if you need to, as in the question above.

Q: Why is my reconstruction blurry?
+++++++++++++++++++++++++++++++++++

A:  If your reconstruction is blurry, the first thing to try is to increase the sharpness parameter.  Values of
``sharpness=1.0`` or ``sharpness=1.5`` are typical, but larger values can further improve sharpness.
You can also increase the assumed SNR by setting the parameter ``snr_db=35`` or ``snr_db=40``. This is similar to increasing sharpness but will also create higher contrast edges in the reconstruction.

If the reconstruction remains blurry, it is often the case that some geometry parameter is incorrectly set for your data.
A common problem is an incorrect center of rotation, which is the parameter ``det_channel_offset``.
You can estimate that parameter from the sinogram with
:func:`~mbirtorch.preprocess.estimate_det_channel_offset`.
The next thing to check is the rotation direction.
A blurry cone beam reconstruction can also come from an incorrect ``source_detector_dist`` or ``source_iso_dist``.

Q: Why does my reconstruction have artifacts?
+++++++++++++++++++++++++++++++++++++++++++++

There are many reasons that a reconstruction may have artifacts including noise, blurring, streaks, cupping, etc.

First, make sure you are using the geometry (parallel or cone) that matches your data.
Parallel beam geometry is faster and could be used for cone beam data, but it may not be accurate if the source is too
close to the object.

For transmission tomography, it is critically important to preprocess the raw photon measurements by normalizing by an air-scan and taking the negative log of the ratio.
The scanner loaders in ``mbirtorch.preprocess`` (see :ref:`ScannerLoaders`) do this, and the functions they use are available for other instruments.

In cone-beam scans, it is sometimes the case that the rotation direction is reversed.
The symptom is a reconstruction that is subtly warped, with shapes distorted and the top and
bottom of the object mirrored.  Correct the direction by taking the negative of your view angles, or
equivalently by reversing their order with ``angles[::-1]``.

A common artifact is rings near the center of the reconstruction that are generated when the center-of-rotation is
not in the center of the detector.  The parameter that repositions the center-of-rotation is ``det_channel_offset``.
Estimate it from the sinogram with :func:`~mbirtorch.preprocess.estimate_det_channel_offset`
and set it with ``ct_model.set_params(det_channel_offset=...)``.

If your reconstruction is blurry, see the FAQ above.

If the reconstruction is too noisy, you might try reducing the value of the ``sharpness`` or ``snr_db`` parameters (discussed
more in the FAQ above on blurry reconstructions).
You can also improve reconstruction quality by using the ``weights`` array that can be generated using the ``gen_weights()`` method.
The weights provide information on the reliability of the sinogram values, with larger weights indicating higher reliability.

Streaks are often caused by metal in the object being scanned.
One advantage of MBIR is that it generally has fewer metal artifacts, but some artifacts typically remain.
Using weights will reduce metal artifacts, and the function ``gen_weights_mar()`` can be used to generate weights that further reduce metal artifacts.

Cupping is typically caused by beam hardening with polychromatic X-ray sources.
This can be partially corrected with a low order polynomial correction.
The preprocessing utilities include ``BH_correction`` for this.

Ring artifacts away from the center of reconstruction are typically caused by detector nonuniformity.
Detector nonuniformity results from the variation in detector sensitivity from pixel to pixel.
This variation is taken out to some degree by air scan normalization, but some variation may remain.
These variations will lead to concentric rings in the reconstruction.
The preprocessing utilities include ``remove_all_stripe`` and ``remove_stripe_fw`` for the
sinogram stripes that produce these rings, and ``remove_sino_offset`` for a residual sinogram offset.

A bright ring at the outer *boundary* of the reconstruction -- typically accompanied by the
"Lateral FOV truncation detected" warning -- means the object extends past the field of view; see the question above.

Q: How can I do larger reconstructions?
+++++++++++++++++++++++++++++++++++++++

A: MBIRTorch runs on both CPU and GPU computers, but we strongly recommend the use of GPUs for large reconstructions since they are much faster.
On a GPU, the size of the reconstruction is typically limited by the amount of GPU memory.
So you should find a fast GPU with the largest possible memory. These days that is typically 40GB to 80GB of GPU memory.
The GPU will be hosted on a CPU, and it is best if that CPU also has even a larger amount of memory, ideally greater than 200GB.

Note that a 2K x 2K x 2K reconstruction occupies 32GB of memory, not counting the sinogram or memory needed for processing.
If your machine has multiple GPUs, MBIRTorch automatically divides the reconstruction across them: the memory
available for the problem grows roughly in proportion to the number of GPUs.  Large reconstructions typically get
faster as well, but small ones do not; see :doc:`usr_multi_gpu` for the measured behavior and the details.
To check before you submit a job whether a reconstruction fits and how long it takes, use
``mbirtorch.estimate_resources``, described in :ref:`usr_estimate_resources`.
If you have no GPU, all processing is done on the CPU.

If your reconstruction is still too large, use :meth:`~mbirtorch.TomographyModel.recon_split_sino`, which splits the
detector rows into overlapping sections, reconstructs each section, and joins the results.  A cone beam
reconstruction splits into two halves.  A parallel beam reconstruction splits into as many sections as the memory
requires, either the number chosen from the memory of your GPUs or the number you ask for with
``slices_per_section``.  On several GPUs the sections are reconstructed side by side; see :ref:`usr_split_groups`.  With a cone beam system you can also reconstruct a subset of the slices by shrinking and moving the region of
reconstruction, as in the question on the region of reconstruction above, and you can make sure axial padding is
disabled (``axial_pad_fraction=0``, the default) if that padding is what pushes you over the memory limit.

We continue to improve the time and memory efficiency of MBIRTorch.

Q: What are the differences between (iterative) recon and recon_fbp/recon_fdk?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: The primary reconstruction method in MBIRTorch is iterative reconstruction (``mbirtorch.TomographyModel.recon``)
using a Bayesian formulation that balances a data-fitting loss function with a prior function on the reconstruction that
reduces noise while maintaining sharp edges. This approach updates the reconstruction multiple times in order to
minimize the sum of these two loss functions.

In contrast, FBP (``mbirtorch.ParallelBeamModel.recon_fbp``) and FDK (``mbirtorch.ConeBeamModel.recon_fdk``) are direct
methods, in which the sinograms are filtered and then backprojected once to form the reconstruction. In this case,
there is no prior information and no attempt to denoise the sinogram or the reconstruction.

In general, FBP and FDK work well when the number of views is large (at least as large as the number of channels in the
detector) and the sinograms have little noise.  Iterative reconstruction typically works better when there are
relatively few views and/or the sinograms are noisy.  Iterative reconstruction takes more time and memory than
FBP/FDK but can produce significantly better reconstructions when the collected data is less than ideal.

Q: What does the warning about torch running host operations on one thread mean?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A: Torch runs its host (CPU) tensor operations on a pool of threads, and it takes the size of
that pool from the environment variables ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` when Python
starts, honoring the smaller of the two.  Some shells and cluster modules set both to one.  Every
host-side step in MBIRTorch then runs on one core, however many the machine or the job allocation
holds: the consensus update of a MACE reconstruction, CPU reconstructions, the denoiser statistics,
the preprocessing, and the host side of every copy from a device.  A 4D MACE reconstruction has
run 2.6 times slower for this reason alone.

MBIRTorch warns at import when torch has one thread and more cores are available.  To fix it,
unset the two variables before starting Python, or call ``torch.set_num_threads(n)`` with the
number of cores you want to use, before the reconstruction.  If you set the variables to one on
purpose, for instance to run several processes on one node, the warning can be ignored or
silenced with the ``warnings`` module.

