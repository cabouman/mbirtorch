.. _GeometryModelsDocs:

================
Geometry Models
================

A reconstruction starts by building the model of the scanner geometry.  Every model has the
same reconstruction, projection, and parameter methods, described in :ref:`TomographyModelDocs`.

.. list-table::
   :widths: 40 60

   * - .. plot:: figs/geom_parallel.py
          :align: center
          :width: 100%
     - :ref:`ParallelBeamModelDocs`

       The rays of each view are parallel.  Synchrotron and neutron scans.
   * - .. plot:: figs/geom_cone.py
          :align: center
          :width: 100%
     - :ref:`ConeBeamModelDocs`

       The rays diverge from a point source, in a circular or helical scan, to a flat or
       curved detector.  Laboratory and industrial X-ray scanners.
   * - .. plot:: figs/geom_multiaxis.py
          :align: center
          :width: 100%
     - :ref:`MultiAxisParallelBeamModelDocs`

       Parallel rays with a per-view elevation angle.  Laminography.
   * - .. plot:: figs/geom_translation.py
          :align: center
          :width: 100%
     - :ref:`TranslationModelDocs`

       Cone beam views of a translated, non-rotating object.  Thin, flat objects.  Under
       development.

.. toctree::
   :hidden:
   :maxdepth: 4

   usr_parallel_beam_model
   usr_cone_beam_model
   usr_multiaxis_parallel_beam_model
   usr_translation_model
