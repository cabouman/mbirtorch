.. _HSNTDocs:

================
Hyperspectral CT
================

The ``hsnt`` module processes hyperspectral neutron transmission data: the transmission of every detector pixel in
each of many wavelength bins.  A sample made of a few materials has a low-rank attenuation X = W H, with a
nonnegative spectrum per component in H and a map per component in W.  The module estimates W and H from the data
(dehydration), multiplies them back into denoised data (rehydration), estimates the number of components, and
reads and writes the hsnt HDF5 layout.  The scripts in
`experiments/hsnt <https://github.com/cabouman/mbirtorch/blob/main/experiments/hsnt/>`__ are worked examples, and
`Command line`_ runs the same steps on files.

.. currentmodule:: mbirtorch.hsnt


Dehydration and rehydration
---------------------------

Dehydration fits W and H by maximum likelihood: the Poisson likelihood of the counts, written in terms of the
transmission T = exp(-X).  The natural input is therefore the transmission, the counts divided by the open beam,
and a zero count is simply a zero transmission:

.. code-block:: python

   transmission = counts / open_beam                       # normalize by the detector calibration
   maps, spectra, dataset_type = hsnt.dehydrate(transmission, dataset_type='transmission')
   denoised = hsnt.rehydrate([maps, spectra, dataset_type])

The dehydrated form ``[maps, spectra, dataset_type]`` is small: the maps have one entry per pixel and component, and
the spectra one per component and bin.  ``rehydrate`` returns the same quantity as the input, transmission or
attenuation.  When the number of components is not given, it is estimated by likelihood-ratio tests.  The components
span the materials' spectra but need not be the materials themselves.  Data too large for the device are fitted by
chunks of pixels.  Given a ``subspace_basis``, ``dehydrate`` fits only the maps, so a large scan can be dehydrated
piece by piece against spectra fitted on part of it.

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

A simulated scan for trying the method: a phantom of three materials, three synthetic spectra,
and the hyperspectral sinogram of the phantom through a tomography model, with Poisson counts.
``demo_7_hyperspectral.py`` dehydrates, reconstructs, and rehydrates one.

.. autofunction:: gen_material_phantom
.. autofunction:: synthetic_material_basis
.. autofunction:: generate_hyper_sinogram
.. autofunction:: generate_hyper_data


Command line
------------

``mbirtorch-hsnt`` runs the same steps from the shell, on an HDF5 file in the hsnt layout or on a directory of TIFF
images, one per wavelength bin.  Its subcommands are ``inspect``, ``convert`` (TIFFs to HDF5, normalized by an open
beam), ``dehydrate``, ``rehydrate`` and ``denoise``:

.. code-block:: bash

   mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ -o sample.h5
   mbirtorch-hsnt denoise sample.h5 -o results/

``mbirtorch-hsnt <subcommand> -h`` lists the options.
