.. _UserAPIDocs:

========
User API
========

See :ref:`UserAPIOverviewDocs` for a high-level introduction to the main features of MBIRTorch, then browse
individual pages for more detail. See :ref:`DemosFAQs` for examples.

* :ref:`UserAPIOverviewDocs`
* :ref:`ParametersDocs`
* :ref:`TomographyModelDocs`
* :ref:`GeometryModelsDocs`
* :ref:`DenoisingDocs`
* :ref:`AutogradDocs`
* :ref:`PreprocessDocs`
* :ref:`Utilities`

.. PENDING(user_pages): restore these bullets, and the matching toctree entries below,
   when vcls and hsnt are ported.  Their pages are staged in source/_pending/ --
   see source/_pending/README.rst.

   * :ref:`VCLSDocs`
   * :ref:`HSNTDocs`


.. DIVERGENCE(automodule members): mbirjax's copy of this directive carries
   ":members: :undoc-members: :show-inheritance:".  Those options are dropped here on
   purpose, to reproduce mbirjax's RENDERED page rather than its directive text.
   mbirjax/__init__.py defines no __all__, so autodoc treats every name as an imported
   member and documents NONE of them; mbirjax's rendered page is 16.7 KB.
   mbirtorch/__init__.py declares an explicit __all__, which autodoc honors.

   Narrowing __all__ to the public surface (2026-08-07) was tried and is NOT sufficient:
   with the options restored the page measured 140 KB against mbirjax's 16.7 KB.  The
   reason is that ":members:" documents each CLASS AND ALL ITS METHODS -- 250 entries
   across the seven exported classes -- which no narrowing of __all__ affects, since the
   classes must stay in it.  Restoring the options also re-raised 8 warnings from method
   docstrings that reference private helpers (_get_estimate_of_recon_std,
   _sharding.run_per_device, get_psf_radii).

   Restoring these options therefore requires a different mechanism, not a narrower
   __all__.  The measured numbers are in plans/torch_port/docs.md.

.. automodule:: mbirtorch
   :no-index:

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Classes

   usr_api_overview
   usr_parameters
   usr_tomography_model
   usr_geometry_models
   usr_denoising
   usr_autograd
   usr_preprocess
   usr_utilities
