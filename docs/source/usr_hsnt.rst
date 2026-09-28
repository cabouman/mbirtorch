.. _HSNTDocs:

================
Hyperspectral CT
================

The ``hsnt`` module processes hyperspectral neutron transmission data, which record the transmission of every
detector pixel in each of many wavelength bins.  A sample made of a few materials has a low-rank attenuation
X = W H: a nonnegative spectrum per component in H and a map per component in W.  The components span the
materials' spectra but need not be the materials themselves.  The module estimates W and H from the counts
(dehydration), multiplies them back into denoised data (rehydration), estimates the number of components, and
reads and writes the hsnt HDF5 layout.  See ``demo_13_hsnt.py`` for a worked example, and `Command line`_ for
running the same steps on files.  The solvers accumulate in float64, so they run on CUDA or the CPU; on a Mac the
automatic device choice uses the CPU rather than MPS.  On a small GPU, a streamed solve that fails in cuBLAS or
cuSOLVER is most likely out of memory: the command line then prints the memory plan and suggests half the chunk
(``--chunk-pixels``) or ``--device cpu``, and ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` can reduce
fragmentation.

.. currentmodule:: mbirtorch.hsnt


Dehydration and rehydration
---------------------------

Dehydration is the maximum-likelihood factorization of the attenuation with W, H >= 0: it minimizes the non-negative
attenuation (NNAL) loss sum[exp(-X) + T X] of the transmission T, the Poisson negative log-likelihood of the counts up
to a constant when the open beam is the same in every pixel and bin.  The result is stored as the maps
(``subspace_data``), the spectra (``subspace_basis``) and the data type.  When the number of components is not given
it is estimated by likelihood-ratio tests, run on at most 16,384 random pixels at full resolution and on pooled
pixels; each compares a component's gain in likelihood with a noise floor taken from the smallest gains of the same
search.  When the true rank is near ``max_rank`` those gains include real components, the floor is set too high and
the rank too low (a rank-6 test problem at 3 counts per bin returned rank 1), and the answer can change with
``max_rank``: raise it when the gains do not level off.  Pooling needs the image axes: data given as (pixels, bins)
are tested at full resolution only, and below 64 counts per bin, where the pooled test would run, the estimate warns
that it can miss components (on a three-material sphere phantom of 65,536 pixels the full-resolution test gave rank 1
up to 3 counts per bin, where pooling gave 3).  Data that do not fit in the device memory are factorized by chunks of
pixels.

At a few counts per bin the likelihood can keep rising along a component that grows on the zero counts, so the fit
has no maximum: it can run to ``max_steps``, and the maps and spectra then depend on ``max_steps``.  On a sphere
phantom (48 x 48 pixels, 150 bins) the fit ran to ``max_steps`` at 1 to 1.5 counts per bin, with attenuations of
order 10^17 on some zero-count entries; at 2 counts the stop came first, at about 10^8.  The fit warns whenever it
puts an attenuation above 50 on a zero count, and the command line's report records the largest and how many entries
exceed 50.  A component can end with its attenuation almost entirely on zero counts: it then models the zero counts
rather than a material, and the fit has one component fewer for the materials.  The fit names such a component
(99% of its attenuation or more on zero counts) in a warning, and the report records each component's share
(``zero_count_share``) and the components named (``zero_count_components``).  On the sphere phantom at rank 3 it
happened in 7 of 18 fits of 65,536 pixels at 1 to 3 counts per bin, and in 2 of 10 fits of 10^6 pixels at 3 counts.
Where it happens is not a matter of the solver alone: on the smaller phantom the fit that gives a component to the zero
counts can have the higher likelihood.

The maximum-likelihood spectra are biased at low dose by the truncation of the pixel coefficients at zero.
``dehydrate`` offers two other estimators through ``spectra``.  ``'unconstrained'`` removes the bias by dropping the
bound while the spectra are estimated, and gains when the pixels are many.  On a sphere phantom at 3 to 10 counts per
bin (scored after the best linear mixing of the fitted spectra onto the true ones; medians over four seeds), it gained
-0.4 to +2.3 dB at 6.5 x 10^4 pixels, +0.8 to +4.9 dB at 1.3 x 10^5 and up to +7.6 dB at 2.6 x 10^5 to 5.2 x 10^5,
where the gain varies with the seed and the dose, and 7 to 12 dB at 10^6 pixels and 3 counts, at 0.07 to 0.16 nats per
pixel above the maximum-likelihood loss; below 2 counts per bin it gained nothing at these sizes.  ``'support'``
instead identifies the coefficients whose true value is zero, holds them at zero, and refits the rest; it needs the
dose.  It decides in the basis the maximum-likelihood fit ends in, which is some mixture of the materials: where the
components are far from the pure materials, a pixel of one material needs several of them, and the selection mostly
separates the sample from the background (on the 1M-pixel sphere phantom from 3 counts per bin up, its maps are
exactly zero on 98.6 to 100% of the background pixels).  Its gain is therefore in the spectra at low dose rather than
in the maps.

Given a ``subspace_basis``, ``dehydrate`` fits only the maps: each pixel's maximum-likelihood coefficients for those
spectra.  Data too large to hold at once, such as the many views of a scan, can then be dehydrated piece by piece
against spectra fitted on part of them.

MBIRJAX's ``dehydrate`` is a scikit-learn NMF of the attenuation, which MBIRTorch does not include; its keywords
raise a ``TypeError`` here.

.. autofunction:: dehydrate
.. autofunction:: rehydrate
.. autofunction:: hyper_denoise
.. autofunction:: estimate_rank


Import/Export
-------------

.. autofunction:: import_hsnt_data_hdf5
.. autofunction:: create_hsnt_metadata
.. autofunction:: export_hsnt_data_hdf5


Synthetic data
--------------

.. autofunction:: generate_hyper_data
.. autofunction:: load_material_basis


Command line
------------

``mbirtorch-hsnt`` (also ``python -m mbirtorch.hsnt``) runs the dehydration on an HDF5 file in the hsnt layout, on
a directory of TIFF images, one per wavelength bin, or on a directory of such directories, one per view.  A stack of
counts is normalized by an open-beam stack (``--open-beam``; a directory of observations is averaged), which all the
views share; transmissions and attenuations are used as they are.  Entries whose transmission is NaN or infinite
become zero counts and negative transmissions (from negative counts, for example) are clipped at zero, with a
warning, as ``dehydrate`` treats an array.  ``inspect``, ``convert``, ``dehydrate`` and ``denoise`` run the data
checks (zero counts, dead pixels and bins, dose) and log them; ``--strict`` stops on a failed one.

.. code-block:: bash

   mbirtorch-hsnt inspect data.h5 --estimate-rank                  # checks, statistics and the rank estimate
   mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5
   mbirtorch-hsnt dehydrate sample.h5 -o results/                  # rank estimated from the data
   mbirtorch-hsnt rehydrate results/sample_dehydrated.h5 --wave-range 100:200 -o results/
   mbirtorch-hsnt denoise sample.h5 -o results/                    # denoised data, plus the dehydrated file

``convert`` reads the input in blocks of bins, so its memory stays near ``--memory-budget`` whatever the size of the
stack (or one bin of every selected view, when that is larger), and writes the hsnt layout, by default to
``<stem>_converted.h5``; the converted file keeps the dose and the source bin indices, and stores a zero transmission
(a bin with no counts) as an infinite attenuation.  ``dehydrate`` writes ``<stem>_dehydrated.h5``, which
:func:`import_hsnt_data_hdf5` reads, a JSON report of the checks, parameters, timings and losses, and plots of the
maps and spectra.  ``rehydrate`` writes the product back as hyperspectral data, for all bins or a ``--wave-range``.
``denoise`` does both, and writes the report too.  The solve runs whole on the device when it fits and is streamed by
chunks of pixels otherwise.  ``--spectra unconstrained`` and ``--spectra support`` select the spectra estimators
described above.  Streamed, their accuracy depends on the polish passes, ``--max-passes`` (default 5), which trade time
for SNR.  On the 1M-pixel sphere phantom at dose 3 in 8 chunks, five passes left the streamed maximum-likelihood fit
0.022 nats per pixel above the fit solved whole, with its spectra within 2.3 dB of it, and the streamed
``unconstrained`` and ``support`` spectra 4.7 to 10.6 dB and 3.6 to 4.9 dB below the same estimators solved whole.  With
40 passes, at four to six times the time, all three came within 0.5 dB of their fits solved whole (the
maximum-likelihood fit stopped by itself after 32).  At dose 30, five passes left the maximum-likelihood fit within
0.001 nats per pixel, while ``support`` stayed up to 6 dB short even after 40.  The memory plan solves whole when the
device holds the data; ``--mode full`` insists on it.  An uncompiled solve needs about 1.4 times the memory of a
compiled one, so on an 80 GB GPU a 1M-pixel, 1200-bin fit is solved whole when compiled (the default there) and
streamed with ``--compile off``.  A streamed fit starts from spectra
fitted on a random subsample of the pixels (16,384, or fewer when memory is short), and its polish passes refine them:
a material with no pixels in the subsample is not found, and the gap to the fit solved whole grows with the number of
pixels over the subsample size (the passes stop at ``--max-passes``, with a warning).  An output is never one of the
inputs, an existing output is replaced only with ``--overwrite``, and a run that fails leaves no partial file.  The
outputs keep the input's angles, wavelengths and geometry entries, matched to the selected views and bins.
``--wave-range`` counts source bins in every subcommand, also on a converted or dehydrated file, and ``--dose`` is the
open-beam count per pixel and source bin, before any ``--wave-bin`` grouping.  Run any subcommand with ``-h`` for the
options most runs need, and with ``--help-all`` for every option, including the solver, memory, rank-test and
support-selection settings.

Two input options correct the data before the fit.  ``--background-boxes`` names boxes free of the sample, each as
``Y0:Y1,X0:X1`` in full-resolution pixels, separated by spaces.  In each bin, each detector tile's transmission is
divided by that of its boxes (for counts, their summed counts over their summed open beam).  This corrects a sample
run and an open beam of different exposure, and the dose becomes the sample's.  ``--background-tiles RxC`` splits the
detector into tiles calibrated separately, each by the boxes whose centers it holds; the default is a preset's tiles,
else one tile.  Without calibration, a data check warns when the most transparent regions read a transmission outside
what a matched exposure gives: the ratio bias the loader expects at the open beam's counts, plus a 3% margin and their
noise (at least 5% on the low side).  ``--open-beam-smoothing W`` smooths the averaged open beam in each bin, at full
resolution, before the division, with a W x W window: the normalized square root of the outer product of two Hamming
windows.  It leaves out the dead pixels, those with no count in any selected bin of any observation (read once more
for this), and keeps the zero counts of live pixels.  The noise model then counts the open beam as more observations,
by the reduction of its variance measured across the observations; the check reports it.

Multi-view data
^^^^^^^^^^^^^^^

A tomographic scan converts to one file of all its views.  The spectra can be fitted on a few views, and the maps of
the rest for those spectra with ``--basis``; for a scan too large to load at once, run that step on ranges of views
(``--views 0:10``, ``--views 10:20``, ...), each with its own output.

.. code-block:: bash

   mbirtorch-hsnt convert projections/ --open-beam open_beam/ -o scan.h5       # views x rows x cols x bins
   mbirtorch-hsnt dehydrate scan.h5 --views 0:4 -o fit/                         # spectra from four views
   mbirtorch-hsnt dehydrate scan.h5 --basis fit/scan_dehydrated.h5 -o all/      # the maps of every view

The views of a directory of view directories are read in the natural order of their names (``view_2`` before
``view_10``), and ``--views`` selects a range in that order.  The maps of a scan, ``subspace_data`` of shape
(views, rows, cols, rank), are one sinogram per component.  Each reconstructs as any sinogram does, and the spectra
turn the reconstructed components back into a hyperspectral volume.  A converted TIFF scan carries no angles, so give
them, one per view in that order, when the metadata lack them:

.. code-block:: python

   import numpy as np
   import mbirtorch as mt
   from mbirtorch import hsnt

   (maps, spectra, dataset_type), meta = hsnt.import_hsnt_data_hdf5("all/scan_dehydrated.h5")
   angles = meta["angles"]                                                        # degrees, or None
   if angles is None:
       angles = np.linspace(0, 180, maps.shape[0], endpoint=False)               # the scan's own angles here
   model = mt.ParallelBeamModel(maps.shape[:3], np.deg2rad(angles))
   recons = np.stack([model.recon(maps[..., r])[0] for r in range(maps.shape[-1])], axis=-1)
   volume = hsnt.rehydrate([recons, spectra, dataset_type], hyperspectral_idx=[300, 600, 900])

Instrument-specific settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These settings apply to ORNL SNAP data only.  The ``ornl-snap`` preset of ``--background-boxes`` is for its
512 x 512 detector of four 256 x 256 chips: it calibrates each chip by the 100 x 100 box in the chip's outer corner,
as the MBIRJAX hsnt preprocessing for SNAP data does, which assumes a sample clear of the corners.  That preprocessing
also smoothed the open beam with a 3 x 3 window and kept the source bins 100 to 2599 of a 2782-bin stack, since the
first bins lie on the rising edge of the flux and the last hold few counts.  The detector's noise is correlated between
neighboring pixels (correlation about 0.6 at one pixel, gone by five), so the 3 x 3 window reduces the open beam's
variance to about 0.65 rather than the 0.22 of independent pixels:

.. code-block:: bash

   mbirtorch-hsnt convert Ni_cylinder_projections/ --open-beam open_beam/ --wave-range 100:2600 \
       --background-boxes ornl-snap --open-beam-smoothing 3 -o Ni_cylinder.h5

The MBIRJAX preprocessing transposed each image as it read it, so the rows of its processed files are the columns of
the TIFF images: its outputs are the transpose of what ``convert`` writes (``np.swapaxes(data, 1, 2)`` compares
them), and boxes or crops taken from its scripts need their Y and X ranges swapped.  The ``ornl-snap`` boxes and chips
are symmetric under that transpose.  Its reconstruction therefore took the TIFF columns as the detector rows, the
direction of the rotation axis: to reconstruct a SNAP scan as it did, pass ``np.swapaxes(maps, 1, 2)`` as the
sinograms in the example above.  It also shifted the four chips 2 pixels apart before reconstructing, which the package
does not do.
