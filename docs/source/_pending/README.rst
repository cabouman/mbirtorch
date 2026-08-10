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

Restore those fragments at the same time as the corresponding code.  Blocks can be
coupled: when a commented-out block defines a ``:ref:`` label, every block that
references that label has to come back with it, or the build reports an undefined
label.  Before restoring one block on its own, grep for the labels it defines.

A ``PENDING(...)`` marker means the content is expected back.  Where mbirjax
documents something that mbirtorch has replaced and will **not** port, the marker
is ``REPLACED(...)`` instead, and it names the replacement::

    grep -rn "REPLACED(" docs/source

There are two today: ``use_gpu``, replaced by ``configure_devices``, and
``device_summary``, replaced by ``get_memory_stats``.  Those blocks are kept as a
record of the divergence, not as work to do.

Declared and documented move together
-------------------------------------

``mbirtorch/__init__.py`` declares ``__all__`` as the public surface, and the
invariant is **documented if and only if declared**.  When a ``PENDING`` block
restores a module-level function, add that function's name to ``__all__`` in the
same change.  Methods and the ``preprocess`` subpackage are not ``__all__``
entries; only module-level names are.

Contents
--------

.. list-table::
   :header-rows: 1
   :widths: 32 26 42

   * - Page
     - Module it documents
     - Parent page to add it to
   * - ``usr_multiaxis_parallel_beam_model.rst``
     - ``mbirtorch.multiaxis_parallel``
     - ``usr_geometry_models.rst``

The parent pages carry a comment at the removed toctree entry marking where each
line goes back.
