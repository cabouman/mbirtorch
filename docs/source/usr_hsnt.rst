.. _HSNTDocs:

================
Hyperspectral CT
================

These are MBIRTorch functions specific to hyperspectral CT processing, exporting/importing, and generating synthetic data.


Dehydration/Rehydration
-----------------------

Dehydration is the maximum-likelihood NNAL factorization of the attenuation, X = W H with W, H >= 0, stored as
the maps (``subspace_data``), the spectra (``subspace_basis``) and the data type; rehydration multiplies them
back. The rank is the number of materials, estimated by likelihood-ratio tests when not given.

.. autofunction:: mbirtorch.hsnt.dehydrate
.. autofunction:: mbirtorch.hsnt.rehydrate
.. autofunction:: mbirtorch.hsnt.hyper_denoise
.. autofunction:: mbirtorch.hsnt.estimate_rank

The scikit-learn L2 dehydration of Chowdhury et al. (2025) is kept as a baseline for comparison plots:

.. autofunction:: mbirtorch.hsnt.l2_dehydrate
.. autofunction:: mbirtorch.hsnt.l2_hyper_denoise


Import/Export
-------------

.. autofunction:: mbirtorch.hsnt.import_hsnt_data_hdf5
.. autofunction:: mbirtorch.hsnt.create_hsnt_metadata
.. autofunction:: mbirtorch.hsnt.export_hsnt_data_hdf5


Generate Synthetic Data
-----------------------

.. autofunction:: mbirtorch.hsnt.generate_hyper_data

Command line
------------

``mbirtorch-hsnt`` (also ``python -m mbirtorch.hsnt``) runs the NNAL factorization, and the denoising built on
it, from the shell on an HDF5 dataset in the package layout or on a directory of TIFF images, one per wavelength
bin. A stack of counts is
normalised by an open-beam stack (``--open-beam``, a directory of observations is averaged); transmissions or
attenuations are used as they are, with the type inferred from the values unless ``--input-type`` is given.
Every subcommand loads the data, runs the checks (non-finite values, negatives, zero counts, dead pixels and
bins, dose, memory) and logs what the solver will see; ``--strict`` stops on a failed check, ``-vv`` shows
per-file detail.

.. code-block:: bash

   mbirtorch-hsnt inspect data.h5 --estimate-rank                  # checks, statistics, an upper bound on the rank
   mbirtorch-hsnt inspect sample_tifs/ --open-beam open_beam/
   mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5     # read the TIFFs once, streamed
   mbirtorch-hsnt dehydrate sample.h5 -o results/                  # rank estimated from the data
   mbirtorch-hsnt dehydrate sample_tifs/ --open-beam open_beam/ --rank 2 --downsample 2
   mbirtorch-hsnt rehydrate results/sample_dehydrated.h5 --wave-range 100:200 -o results/
   mbirtorch-hsnt denoise sample.h5 -o results/                    # denoised data, plus the dehydrated file

The rank is estimated by default when ``--rank`` is not given: ranks 1 to ``--max-rank`` are fitted on a pixel
subsample and each added component is kept while the log-likelihood it gains exceeds twice the noise floor (a
noise-only component gains about half its parameter count), with the dose calibrated from the residual of the
most flexible fit so a nominal open-beam dose does not matter. The log prints the gain of every component and
the threshold, and the report stores them; ``--rank N`` overrides. Because every component gets a free
coefficient per pixel, the noise floor grows with the pixel count as fast as a faint material's evidence does, so
the test is also run on spatially pooled pixels (``--rank-pool``, blocks chosen so the pooled count is about half
the bin count) where the floor is far lower, and the larger rank is taken; this recovers a faint material at low
dose that the full-resolution test misses. After the solve the log reports the
reduced chi-square of the fit against Poisson noise when the dose is known: near 1 the residual is at the noise
level, well above 1 the rank is too small or the model misspecified, well below 1 the fit follows the noise.

``convert`` streams: it reads the input in blocks of bins (TIFF images decoded in parallel, the open-beam
observations averaged block by block), normalises, checks and writes each block into the output, so its memory is
the ``--memory-budget`` (256 MiB by default; ``--block-bins`` fixes the block) whatever the size of the stack, and
the next block is read while the current one is processed, and its output is chunked in 16-bin slabs along the spectral axis. The data checks run as accumulations over the blocks
and are stored in the file's ``checks`` attribute. ``convert`` also takes an HDF5 file, to downsample, bin or
re-type an existing dataset.

``dehydrate`` writes ``<stem>_dehydrated.h5`` in the dehydrated layout (``subspace_data`` holds the material maps,
``subspace_basis`` the spectra; :func:`~mbirtorch.hsnt.import_hsnt_data_hdf5` reads it), ``<stem>_report.json``
with the checks, parameters, timings, loss and memory, and PNG plots of the maps and spectra; ``--rehydrate``
also writes the denoised data. ``rehydrate`` takes such a file and writes ``<stem>_rehydrated.h5`` in the
hyperspectral layout (``data`` with the spectral axis last, ``dataset_type``), for all bins or a ``--wave-range``
of them and as attenuation or transmission (``--as-type``), block by block so the array is never held twice.
``denoise`` does both in one run and writes ``<stem>_denoised.h5`` plus the dehydrated file unless
``--no-dehydrated``. The solve runs whole on the device
when it fits and is streamed by chunks otherwise (``--mode``); ``--spectra unconstrained`` and ``--spectra support``
select the bias-corrected spectra estimators. ``--dry-run`` loads,
checks and plans without solving.
