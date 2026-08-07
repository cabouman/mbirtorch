Pending pages
=============

This directory holds documentation pages ported from ``mbirjax/docs`` whose
underlying modules have **not been ported to mbirtorch yet**.  Each page is a
near-verbatim copy of its mbirjax original, so that when the module lands the
prose does not have to be re-derived.

These pages are **not built**: ``conf.py`` lists ``_pending/**`` in
``exclude_patterns``.  Every page here is pure autodoc against a module that
does not exist, so building it would fail to import and render an empty page.

Landing a page
--------------

When the corresponding module is ported to mbirtorch, the page becomes live in
two mechanical steps:

1. ``git mv docs/source/_pending/<page>.rst docs/source/``
2. Add ``<page>`` to the parent toctree, and its ``:ref:`` bullet to the parent
   list, in the file named in the table below.

Then rebuild and fix whatever ``nitpicky`` reports.  The module paths in these
pages are already renamed to ``mbirtorch``, but the function and class names are
still mbirjax's; any that change during the port must be updated at that point.

Smaller pending fragments
-------------------------

Whole pages are only part of the picture.  Individual sections, directives, and
sentences that depend on unported code are commented out **in place** on the live
pages, each marked with a ``PENDING(<topic>)`` comment.  To find them all::

    grep -rn "PENDING(" docs/source

Restore those fragments at the same time as the corresponding code.  Two of them
are coupled: the ``_SaveLoadDocs`` label is defined in the ``PENDING(save_load_hdf5)``
block in ``usr_tomography_model.rst``, and ``usr_utilities.rst`` and
``usr_api_overview.rst`` each reference it from their own blocks, so all three must
be restored together or the build will report an undefined label.

A ``PENDING(...)`` marker means the content is expected back.  Where mbirjax
documents something that mbirtorch has replaced and will **not** port, the marker
is ``REPLACED(...)`` instead, and it names the replacement::

    grep -rn "REPLACED(" docs/source

There are four today: ``use_gpu`` and ``device_summary`` (replaced by
``configure_devices`` and ``get_memory_stats``), ``split_sino_recon`` (subsumed by
the multi-device engine), and the ``split_sino_recon`` bullet in
``advanced_features.rst``.  Those blocks are kept as a record of the divergence, not
as work to do.

Contents
--------

.. list-table::
   :header-rows: 1
   :widths: 32 26 42

   * - Page
     - Module it documents
     - Parent page to add it to
   * - ``usr_preprocess.rst``
     - ``mbirtorch.preprocess``
     - ``usr_api.rst``
   * - ``usr_vcls.rst``
     - ``mbirtorch.vcls``
     - ``usr_api.rst``
   * - ``usr_hsnt.rst``
     - ``mbirtorch.hsnt``
     - ``usr_api.rst``
   * - ``usr_translation_model.rst``
     - ``mbirtorch.translation_model``
     - ``usr_geometry_models.rst``
   * - ``usr_multiaxis_parallel_beam_model.rst``
     - ``mbirtorch.multiaxis_parallel``
     - ``usr_geometry_models.rst``

The parent pages carry a comment at the removed toctree entry marking where each
line goes back.
