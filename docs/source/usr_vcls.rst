.. _VCLSDocs:

=====================
Sparse View Selection
=====================

Given a reference object like the one to be scanned, ``get_opt_views`` picks the view angles
that reconstruct it best, so that a scan with few views loses as little as possible.  It
minimizes the view covariance loss of J. Lin et al., "Tomographic Sparse View Selection using
the View Covariance Loss," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
The reference can be a prior reconstruction or a phantom of the object, and a region of interest
can restrict where the views are judged.

.. code-block:: python

    ct_model = mbirtorch.ParallelBeamModel(sinogram_shape, candidate_angles)
    view_inds, vcl = mbirtorch.get_opt_views(ct_model, reference_object, num_selected_views=20)
    selected_angles = candidate_angles[view_inds]
    mbirtorch.show_image_with_projection_rays(reference_object[:, :, 0],
                                              rotation_angles_rad=selected_angles)

Scripts are in the `vcls <https://github.com/cabouman/mbirtorch_applications/tree/main/vcls>`__
folder of `mbirtorch_applications <https://github.com/cabouman/mbirtorch_applications>`__.

.. autofunction:: mbirtorch.vcls.get_opt_views
.. autofunction:: mbirtorch.vcls.show_image_with_projection_rays
