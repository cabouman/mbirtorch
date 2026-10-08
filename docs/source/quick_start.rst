===========
Quick Start
===========

This page takes you from installation to a reconstruction of your own data in four steps.


1. Install
----------

Follow the :ref:`installation instructions <InstallationDocs>`.  A conda environment or a
Python virtual environment is recommended.


2. Run the first demo
---------------------

The demo scripts are in the `demo folder <https://github.com/cabouman/mbirtorch/blob/main/demo/>`__
of the repository.  The first one makes a phantom, projects it to a sinogram, reconstructs
it, and opens a viewer::

    python demo/demo_1_parallel_basics.py

The demo needs no data and runs on a laptop CPU in about a minute.  If it runs, the
installation is working.  The other demos are listed in :ref:`DemosFAQs`.


3. Reconstruct your own data
----------------------------

Put your data in two numpy arrays:

- ``sinogram``: a 3D array with shape ``(views, detector rows, detector channels)``.
  Each detector row is perpendicular to the rotation axis.  Each view is stored in raster
  order, left to right and top to bottom, as seen looking from the source to the detector.
- ``angles``: a 1D array with the rotation angle of each view, in radians.

The sinogram holds line integrals, not raw photon counts.  For transmission data, divide each
view by the air scan and take the negative log of the ratio.  The function
:func:`mbirtorch.preprocess.scan_to_sino` does this from the object, blank, and dark scans,
and it also fills the defective pixels.  See :ref:`PreprocessDocs` for more.

Build a model from the sinogram shape and the angles, reconstruct, and open a viewer::

    import mbirtorch
    ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
    recon, recon_dict = ct_model.recon(sinogram)
    mbirtorch.slice_viewer(recon, title='MBIRTorch reconstruction')

The result ``recon`` is a 3D numpy array with shape ``(rows, columns, slices)``.  The dictionary
``recon_dict`` holds the parameters used and the run log.

For cone-beam data, the model also needs the two source distances, in the same units as the
detector pixel pitch::

    ct_model = mbirtorch.ConeBeamModel(sinogram.shape, angles,
                                       source_detector_dist=source_detector_dist,
                                       source_iso_dist=source_iso_dist)

The default parameters usually produce a good reconstruction on the first try.


4. Adjust the reconstruction
----------------------------

Every setting is a parameter of the model, set with ``set_params`` before calling ``recon``.
The one worth trying first is ``sharpness``.  Its default is 1.0.  A higher value gives
crisper edges and more noise, and a lower value gives smoother images::

    ct_model.set_params(sharpness=1.5)
    recon, recon_dict = ct_model.recon(sinogram)

The geometry parameters, such as the offset of the center of rotation, are set the same way.
The parameters are described in :ref:`ParametersDocs`, the model classes are
:class:`~mbirtorch.ParallelBeamModel` and :class:`~mbirtorch.ConeBeamModel`, and the most used
functions are summarized in :ref:`UserAPIOverviewDocs`.
