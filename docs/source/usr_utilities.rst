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

The slice viewer shows the data.  The scan geometry itself is drawn by
``mbirtorch.geometry_viewer``, which shows where the source, the detector, and
the reconstruction volume sit for one view at a time; see
:ref:`GeometryViewerDocs`.


.. _Viewer4DDocs:

4D Data Viewer
--------------

.. autofunction:: mbirtorch.view_utils.slice_viewer4d

A 3D volume shown next to 4D volumes stays fixed in time, so it serves as a reference for the
moving ones.  In a space-time plane (t-x, t-y, or t-z) each row of the image is one frame, and
the two sliders pick the line of the volume that is shown.  An edge that moves from frame to
frame then appears as a slanted or zigzag line.

The figures show the viewer on a synthetic 4D volume: a Shepp-Logan phantom that shifts by up
to two pixels, in a direction that turns 60 degrees per frame, next to the unshifted phantom.
In the x-y plane, the mean inside an ROI on the phantom's edge rises and falls every 6 frames,
and the static phantom gives a flat line.  In the t-y plane the shift appears as a zigzag of
the edges, and the static phantom's edges are straight.

.. plot:: figs/slice_viewer4d.py
   :alt: The 4D viewer in the x-y plane with an ROI and its mean against frame, and in the
         t-y plane, where the edges of the shifting phantom zigzag.


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

Saving and loading a reconstruction is described under :ref:`SaveLoadDocs`.

.. autofunction:: mbirtorch.utilities.download_and_extract
.. autofunction:: mbirtorch.utilities.save_volume_as_gif


.. _synthetic-data-generation:

Synthetic Data Generation
-------------------------

.. autofunction:: mbirtorch.utilities.generate_demo_data
.. autofunction:: mbirtorch.utilities.generate_3d_shepp_logan_reference
.. autofunction:: mbirtorch.utilities.generate_3d_shepp_logan_low_dynamic_range

.. autofunction:: mbirtorch.utilities.gen_translation_phantom


Cache Management
----------------

MBIRTorch keeps one on-disk cache, of compiled ``torch.compile`` artifacts, so that a fresh
process reuses prior compilations instead of recompiling.  These are stored in
`~/.mbirtorch/torch_cache`.  The cache exists to make cold starts fast -- with it, a
fresh process reuses prior compilations instead of recompiling (roughly 14 s
down to 2 s for a first small reconstruction).  It grows with the number of
distinct compiled shapes and typically stays in the tens of megabytes; it is
never cleaned automatically.  To remove it:

.. code-block:: python

    import mbirtorch
    mbirtorch.clear_cache()   # deletes ~/.mbirtorch entirely (recreated empty)


The location can be redirected by setting the `TORCHINDUCTOR_CACHE_DIR`
environment variable before the first compile (e.g. to node-local or scratch
storage on a cluster, where home quotas are tight); `clear_cache()` does not
touch a redirected location.

Everything else the package caches is in-memory only and is freed with the
objects that hold it (e.g. the per-model pixel-index cache); nothing besides
`~/.mbirtorch` is written to disk.


.. autofunction:: mbirtorch.utilities.clear_cache
