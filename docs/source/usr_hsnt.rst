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
running the same steps on files.

.. currentmodule:: mbirtorch.hsnt


Dehydration and rehydration
---------------------------

Dehydration is the maximum-likelihood factorization of the attenuation with W, H >= 0: it minimizes the
non-negative attenuation (NNAL) loss sum[exp(-X) + T X] of the transmission T, the Poisson negative log-likelihood
of the counts up to a constant when the open beam is the same in every pixel and bin.  The result is stored as the
maps (``subspace_data``), the spectra (``subspace_basis``) and the data type.  When the number of components is
not given it is estimated by likelihood-ratio tests.  Data that do not fit in the device memory are factorized by
chunks of pixels.

The maximum-likelihood spectra are biased at low dose by the truncation of the pixel coefficients at zero.
``dehydrate`` offers two other estimators through ``spectra``.  ``'unconstrained'`` removes the bias by dropping
the bound while the spectra are estimated, and pays when the pixels are many.  ``'support'`` instead identifies
the coefficients whose true value is zero, holds them at zero, and refits the rest; it needs the dose.

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

``mbirtorch-hsnt`` (also ``python -m mbirtorch.hsnt``) runs the dehydration on an HDF5 file in the hsnt layout or
on a directory of TIFF images, one per wavelength bin.  A stack of counts is normalized by an open-beam stack
(``--open-beam``; a directory of observations is averaged); transmissions and attenuations are used as they are.
Every subcommand runs the data checks (non-finite values, negatives, zero counts, dead pixels and bins, dose) and
logs them; ``--strict`` stops on a failed one.

.. code-block:: bash

   mbirtorch-hsnt inspect data.h5 --estimate-rank                  # checks, statistics and the rank estimate
   mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5
   mbirtorch-hsnt dehydrate sample.h5 -o results/                  # rank estimated from the data
   mbirtorch-hsnt rehydrate results/sample_dehydrated.h5 --wave-range 100:200 -o results/
   mbirtorch-hsnt denoise sample.h5 -o results/                    # denoised data, plus the dehydrated file

``convert`` reads the input in blocks of bins, so its memory stays near ``--memory-budget`` whatever the size of the
stack, and writes the hsnt layout.  ``dehydrate`` writes ``<stem>_dehydrated.h5``, which
:func:`import_hsnt_data_hdf5` reads, a JSON report of the checks, parameters, timings and losses, and plots of the
maps and spectra.  ``rehydrate`` writes the product back as hyperspectral data, for all bins or a
``--wave-range``.  ``denoise`` does both.  The solve runs whole on the device when it fits and is streamed by chunks
of pixels otherwise.  ``--spectra unconstrained`` and ``--spectra support`` select the spectra estimators described
above.  Run any subcommand with ``-h`` for the options most runs need, and with ``--help-all`` for every option,
including the solver, memory, rank-test and support-selection settings.
