.. _Utilities:

=========
Utilities
=========

MBIRTorch contains utilities for viewing, downloading, exporting/importing, generating
synthetic data, and managing the on-disk compile cache.

Saving and loading models and reconstructions is handled through TomographyModel: :ref:`SaveLoadDocs`.


3D Data Viewer
--------------

.. autofunction:: mbirtorch.view_utils.slice_viewer

Here is an example showing views of a modified Shepp-Logan phantom, with changing intensity window and displayed slice:

.. image:: https://www.math.purdue.edu/~buzzard/images/slice_viewer_demo.gif
   :alt: An animated image of the slice viewer.


General Purpose
---------------

.. autofunction:: mbirtorch.utilities.stitch_arrays
.. autofunction:: mbirtorch.utilities.get_ct_model
.. autofunction:: mbirtorch.utilities.copy_ct_model
.. autofunction:: mbirtorch.utilities.build_model


Weight Generation
-----------------

.. autofunction:: mbirtorch.vcd_utils.gen_weights
.. autofunction:: mbirtorch.vcd_utils.gen_weights_mar


IO Functions
------------

As noted above, saving and loading models and reconstructions is handled through TomographyModel: :ref:`SaveLoadDocs`.

The functions here are for direct interactions with files.

.. autofunction:: mbirtorch.utilities.download_and_extract
.. autofunction:: mbirtorch.utilities.save_data_hdf5
.. autofunction:: mbirtorch.utilities.load_data_hdf5
.. autofunction:: mbirtorch.utilities.export_recon_hdf5
.. autofunction:: mbirtorch.utilities.import_recon_hdf5


.. _synthetic-data-generation:

Synthetic Data Generation
-------------------------

.. autofunction:: mbirtorch.utilities.generate_demo_data
.. autofunction:: mbirtorch.utilities.generate_3d_shepp_logan_reference
.. autofunction:: mbirtorch.utilities.generate_3d_shepp_logan_low_dynamic_range

.. PENDING(synthetic_data): gen_translation_phantom is blocked on the translation
   geometry, whose page is staged in source/_pending/.  Restore this directive last in
   this section, matching mbirjax's order.

   .. autofunction:: mbirtorch.utilities.gen_translation_phantom


Cache Management
----------------

MBIRTorch keeps one on-disk cache, of compiled ``torch.compile`` artifacts, so that a fresh
process reuses prior compilations instead of recompiling.  This section has no MBIRJAX
counterpart; the MBIRJAX analog is its JAX compilation cache.

.. autofunction:: mbirtorch.utilities.clear_cache
