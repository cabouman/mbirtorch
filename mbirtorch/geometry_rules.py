"""The geometry rules the four scan geometries share, in one place.

All four models use one right-handed object frame whose z axis is the axis of
rotation.  The center of reconstruction voxel (i, j, k) sits at

    x = delta_voxel * (j - (num_cols - 1) / 2)
    y = delta_voxel_row * (i - (num_rows - 1) / 2)
    z = delta_voxel_slice * (k - (num_slices - 1) / 2) + recon_slice_offset

so the column index runs along x, the row index along y, and the slice index
along z.  The two pitches are ``delta_voxel_row = voxel_row_aspect *
delta_voxel`` and ``delta_voxel_slice = voxel_slice_aspect * delta_voxel``, and
``recon_slice_offset`` is zero for a geometry that has no such parameter.

A point that lands on the detector is described by the pair (u, v): u is
measured along the channels and v along the rows, both from the point where the
central ray meets the detector.  The two index rules turn that pair into
fractional indices into a detector of ``num_det_rows`` by ``num_det_channels``:

    channel = (u + det_channel_offset) / delta_det_channel + (num_det_channels - 1) / 2
    row     = (v + det_row_offset) / delta_det_row + (num_det_rows - 1) / 2

:meth:`mbirtorch.TomographyModel.project_points` and the projector bodies both
go through the functions below, so each rule is written out exactly once and
the public map cannot drift away from the map the projectors implement.

These are plain functions that are called from inside the compiled projector
bodies, so they read no model parameters and use nothing but the tensors and
numbers handed to them.
"""

import torch

_F32 = torch.float32


def pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel, delta_voxel_row):
    """The unrotated in-plane position of each recon pixel, (x_tilde, y_tilde).

    ``pixel_indices`` is the flat index i * num_cols + j of each pixel, (P,).
    The row index i runs along y and the column index j runs along x, and the
    volume is centered on the origin in both.  Returns two (P,) float32
    tensors.
    """
    row_index = (pixel_indices // num_cols).to(_F32)
    col_index = (pixel_indices % num_cols).to(_F32)
    # Note the change in order from (i, j) to (y, x).
    y_tilde = delta_voxel_row * (row_index - (num_rows - 1) / 2.0)
    x_tilde = delta_voxel * (col_index - (num_cols - 1) / 2.0)
    return x_tilde, y_tilde


def rotate_about_z(x_tilde, y_tilde, cosine, sine):
    """Positions rotated about the z axis, one rotation per view.

    ``x_tilde`` and ``y_tilde`` are (P,); ``cosine`` and ``sine`` are the
    cosine and sine of each view's angle, (Vb, 1).  A positive angle carries
    the +x axis toward the +y axis.  Returns x and y, each (Vb, P).
    """
    x = cosine * x_tilde[None, :] - sine * y_tilde[None, :]
    y = sine * x_tilde[None, :] + cosine * y_tilde[None, :]
    return x, y


def channel_index(u, delta_det_channel, det_channel_offset, num_channels):
    """The fractional channel index of detector coordinate ``u``.

    ``u`` is measured along the channels from the point where the central ray
    meets the detector.  Index (num_channels - 1) / 2 is the middle of the
    grid, and a positive ``det_channel_offset`` moves the grid toward negative
    u, which moves a fixed object's image toward a higher channel index.
    """
    det_center_channel = (num_channels - 1) / 2.0
    return (u + det_channel_offset) / delta_det_channel + det_center_channel


def row_index(v, delta_det_row, det_row_offset, num_rows):
    """The fractional row index of detector coordinate ``v``; the row
    counterpart of :func:`channel_index`."""
    det_center_row = (num_rows - 1) / 2.0
    return (v + det_row_offset) / delta_det_row + det_center_row
