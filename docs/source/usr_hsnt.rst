.. _HSNTDocs:

================
Hyperspectral CT
================

These are MBIRTorch functions specific to hyperspectral CT processing, exporting/importing, and generating synthetic data.


Dehydration/Rehydration
-----------------------

.. autofunction:: mbirtorch.hsnt.hyper_denoise
.. autofunction:: mbirtorch.hsnt.dehydrate
.. autofunction:: mbirtorch.hsnt.rehydrate


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
   mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5     # read the TIFFs once
   mbirtorch-hsnt factorize sample.h5 -o results/                  # rank estimated from the data
   mbirtorch-hsnt factorize sample_tifs/ --open-beam open_beam/ --rank 2 --downsample 2 --gauge
   mbirtorch-hsnt denoise sample.h5 -o results/                    # denoised data, plus the factors

The rank is estimated by default when ``--rank`` is not given: ranks 1 to ``--max-rank`` are fitted on a pixel
subsample and each added component is kept while the log-likelihood it gains exceeds twice the noise floor (a
noise-only component gains about half its parameter count), with the dose calibrated from the residual of the
most flexible fit so a nominal open-beam dose does not matter. The log prints the gain of every component and
the threshold, and the report stores them; ``--rank N`` overrides. At very low dose a weak material can fall
below the threshold, so check the gain table when a component is expected. After the solve the log reports the
reduced chi-square of the fit against Poisson noise when the dose is known: near 1 the residual is at the noise
level, well above 1 the rank is too small or the model misspecified, well below 1 the fit follows the noise.

``denoise`` factorizes and rehydrates: it writes ``<stem>_denoised.h5`` in the package's hyperspectral layout
(``data`` with the spectral axis last, ``dataset_type``), block by block so the array is never held twice, plus
the factors unless ``--no-factors``; ``--as-type`` chooses attenuation or transmission for the outputs.
``factorize`` writes ``<stem>_factors.h5`` in the dehydrated layout (``subspace_data`` holds the material maps,
``subspace_basis`` the spectra; :func:`~mbirtorch.hsnt.import_hsnt_data_hdf5` reads it and
:func:`~mbirtorch.hsnt.rehydrate` reconstructs the denoised data), ``<stem>_report.json`` with the checks,
parameters, timings, loss and memory, and PNG plots of the maps and spectra. The solve runs whole on the device
when it fits and is streamed by chunks otherwise (``--mode``); ``--spectra unconstrained`` and ``--spectra support``
select the bias-corrected spectra estimators, ``--gauge`` the pure-pixel fix of the maps. ``--dry-run`` loads,
checks and plans without solving.
