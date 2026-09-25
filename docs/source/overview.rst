========
Overview
========

**MBIRTorch** is a Python package for Model Based Iterative Reconstruction (MBIR) of images from tomographic data.

- **Image quality:** MBIR uses a forward (sensor) model and a prior (image) model, which gives the best image quality.
- **Ease of use:** automatic parameter selection produces a good reconstruction the first time.
- **Speed:** the vectorized coordinate descent (VCD) algorithm converges fast, and PyTorch runs it on CPUs or GPUs,
  spreading one reconstruction across multiple GPUs (see :doc:`usr_multi_gpu`).
- **Flexibility:** an object-oriented Python interface, and proximal map interfaces for Plug-and-Play priors.

See :ref:`DemosFAQs` for demos as Jupyter notebooks and Python scripts, and :ref:`InstallationDocs` to install from source.

**Geometries**

MBIRTorch supports the imaging geometries below.  Each is a class with its own
forward and back projection, and a new geometry can be added by writing a new
class.

* **Parallel beam** (:class:`~mbirtorch.ParallelBeamModel`): parallel rays from a source at infinity.
* **Cone beam** (:class:`~mbirtorch.ConeBeamModel`): rays from a point source to a flat or curved detector, in a circular or helical scan.
* **Multi-axis parallel** (:class:`~mbirtorch.MultiAxisParallelModel`): parallel rays at a per-view elevation angle to the rotation axis, including laminography.
* **4D reconstruction** (:class:`~mbirtorch.MACE4DModel`): one continuous scan of a moving object, reconstructed as one volume per time frame.

.. list-table::

    * - .. plot:: figs/geom_parallel.py
           :align: center
           :width: 100%

           Parallel-beam geometry

      - .. plot:: figs/geom_cone.py
           :align: center
           :width: 100%

           Cone-beam geometry

    * - .. plot:: figs/geom_multiaxis.py
           :align: center
           :width: 100%

           Multi-axis parallel geometry

      - .. plot:: figs/recon_4d.py
           :align: center
           :width: 100%

           4D reconstruction: overlapping angular windows, one volume per frame
