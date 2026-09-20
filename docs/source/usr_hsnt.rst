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

``mbirtorch-hsnt`` (also ``python -m mbirtorch.hsnt``) runs the NNAL factorization from the shell on an HDF5
dataset in the package layout or on a directory of TIFF images, one per wavelength bin. A stack of counts is
normalised by an open-beam stack (``--open-beam``, a directory of observations is averaged); transmissions or
attenuations are used as they are, with the type inferred from the values unless ``--input-type`` is given.
Every subcommand loads the data, runs the checks (non-finite values, negatives, zero counts, dead pixels and
bins, dose, memory) and logs what the solver will see; ``--strict`` stops on a failed check, ``-vv`` shows
per-file detail.

.. code-block:: bash

   mbirtorch-hsnt inspect data.h5 --estimate-rank                  # checks, statistics, an upper bound on the rank
   mbirtorch-hsnt inspect sample_tifs/ --open-beam open_beam/
   mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5     # read the TIFFs once
   mbirtorch-hsnt factorize sample.h5 --rank 2 -o results/
   mbirtorch-hsnt factorize sample_tifs/ --open-beam open_beam/ --rank 2 --downsample 2 --gauge

``factorize`` writes ``<stem>_factors.h5`` in the dehydrated layout (``subspace_data`` holds the material maps,
``subspace_basis`` the spectra; :func:`~mbirtorch.hsnt.import_hsnt_data_hdf5` reads it and
:func:`~mbirtorch.hsnt.rehydrate` reconstructs the denoised data), ``<stem>_report.json`` with the checks,
parameters, timings, loss and memory, and PNG plots of the maps and spectra. The solve runs whole on the device
when it fits and is streamed by chunks otherwise (``--mode``); ``--spectra unconstrained`` and ``--spectra support``
select the bias-corrected spectra estimators, ``--gauge`` the pure-pixel fix of the maps. ``--dry-run`` loads,
checks and plans without solving.
