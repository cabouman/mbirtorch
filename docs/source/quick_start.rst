===========
Quick Start
===========

MBIRTorch is designed to give reconstructions using just a few lines of code.  Assuming your sinogram is a numpy array
in the shape (views, rows, channels) from a parallel beam projection, you can create and visualize a reconstruction using:

.. code-block::

    import mbirtorch
    recon, recon_dict = mbirtorch.recon_simple_parallel(sinogram, angles)
    mbirtorch.slice_viewer(recon, title='MBIRTorch Recon')

(:func:`~mbirtorch.recon_simple_cone` is the cone-beam counterpart.)  For more control -- sharpness and
other regularization parameters, weights, geometry offsets, device choices, and the other reconstruction
methods -- create a model object and call its methods:

.. code-block::

    ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
    recon, recon_dict = ct_model.recon(sinogram)

Demos
~~~~~

The best way to start is to:

- **Install MBIRTorch** using the instructions provided on the :ref:`Installation Page <InstallationDocs>`.  We recommend installing in a conda environment or pip virtual environment.
- **View and run demos** in :ref:`DemosFAQs`

You can then adapt these demos to suit your needs.


Application
~~~~~~~~~~~

Below are simple instructions on how to do your first reconstruction.  For a first pass,
``recon_simple_parallel(sinogram, angles)`` performs the initialize and reconstruct steps below in one call.

- **Get your data:**

  - Import your ``sinogram`` data as a 3D numpy array organized by ``(views, detector rows, detector channels (columns))``.
  - Create a 1D numpy array called ``angles`` that contains the rotation angle **in radians** of each view.

  Note that each row of sinogram data is assumed to be perpendicular to the rotation axis and each view is assumed to be in conventional raster order (i.e., left-to-right, top-to-bottom) looking through the object from the source to the detector.


  For transmission tomography, it is critically important to preprocess the raw photon measurements by normalizing by an air-scan and taking the negative log of the ratio.  We provide simple preprocessing utilities in ``mbirtorch.preprocess`` for doing this, and we plan to provide more utilities for specific instruments in the future.

- **Initialize a model:**

  - Run ``model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)`` to initialize a parallel beam model.

  You will then use the ``model`` object to perform the various reconstruction functions.


- **Reconstruct and visualize:**

  - Run ``recon, recon_dict = model.recon(sinogram)`` to reconstruct a volume in  ``(rows, columns, slices)`` format using MBIR.
  - Call ``mbirtorch.slice_viewer(recon, title='MBIRTorch reconstruction')``

Even the default parameter settings will usually produce a good quality reconstruction.
