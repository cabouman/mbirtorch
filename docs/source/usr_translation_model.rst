.. _TranslationModelDocs:

=================
Translation Model
=================

The ``TranslationModel`` class implements a geometry and reconstruction model for translation computed tomography.
This class inherits all behaviors and attributes of the :ref:`TomographyModelDocs`.

This is an experimental tomography model in alpha testing.
It includes ``fdk_recon`` for direct (non-iterative) reconstruction, used as the
initializer for the iterative ``recon()``.  (The mbirjax page's statement that no
direct reconstruction exists is outdated there as well.)

See the API docs for the :class:`~mbirtorch.TomographyModel` class for details on a wide range
of functions that can be implemented using the ``TranslationModel``.

Constructor
-----------

.. autoclass:: mbirtorch.TranslationModel
   :show-inheritance:


