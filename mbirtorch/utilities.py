"""Phantom generation, ported from mbirjax.utilities (the low-dynamic-range
Shepp-Logan used by the demos).

The build is plain numpy: the phantom is a host-side reference object in
mbirjax too (its jax machinery there exists only to bound peak memory at very
large sizes, not needed here yet).  The ellipsoid definitions and the
attenuation-scale formula are copied verbatim.

Boundary note for cross-framework comparisons: each ellipsoid is a <= 1
threshold on a float quadratic, so voxels exactly at an ellipsoid boundary can
flip between frameworks (f32 vs f64 grid arithmetic).  The golden test
therefore allows a small fraction of differing boundary voxels.
"""

import numpy as np


def add_ellipsoid(current_volume, grids, z_locations, x0, y0, z0, a, b, c,
                  angle=0, intensity=1.0):
    """
    Add an ellipsoid to an existing volume.

    Args:
        current_volume (ndarray): 3D volume, (rows, cols, slices).
        grids (tuple): (x_grid, y_grid) in-plane coordinate grids, (rows, cols).
        z_locations (ndarray): 1D array of z coordinates of the slices.
        x0, y0, z0 (float): ellipsoid center.
        a, b, c (float): x, y, z radii.
        angle (float): rotation of the ellipsoid in the xy plane around
            (x0, y0), in degrees.
        intensity (float): the constant value of the ellipsoid to be added.

    Returns:
        ndarray: current_volume + ellipsoid.
    """
    x_grid, y_grid = grids
    cos_angle = np.cos(np.deg2rad(angle))
    sin_angle = np.sin(np.deg2rad(angle))
    Xr = cos_angle * (x_grid - x0) + sin_angle * (y_grid - y0)
    Yr = -sin_angle * (x_grid - x0) + cos_angle * (y_grid - y0)

    # Which xy locations can be inside this ellipsoid, then the z extent per slice.
    xy_norm = Xr ** 2 / a ** 2 + Yr ** 2 / b ** 2
    z_norm = (z_locations - z0) ** 2 / c ** 2
    inside = (xy_norm[:, :, None] + z_norm[None, None, :]) <= 1
    return current_volume + intensity * inside.astype(np.float32)


def _add_shepp_logan_ellipsoids(phantom, grids, z_locations):
    """Add the nine standard low-dynamic-range Shepp-Logan ellipsoids to
    ``phantom`` (definitions copied verbatim from mbirjax)."""
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0, 0, 0.69, 0.92, 0.9, intensity=1)
    # Smaller ellipsoids and other structures
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.0184, 0, 0.6624, 0.874, 0.88, intensity=-0.8)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0.22, 0, 0, 0.41, 0.16, 0.21, angle=108, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.22, 0, 0, 0.31, 0.11, 0.22, angle=72, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.35, 0, 0.21, 0.25, 0.5, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.08, -0.605, 0, 0.046, 0.023, 0.02, angle=0, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.605, 0, 0.023, 0.023, 0.02, angle=0, intensity=0.1)
    return phantom


# Semi-axes (rows, cols, slices) of the MAIN Shepp-Logan ellipsoid -- the
# largest structure, which dominates the longest line integral.  Must match the
# first ellipsoid above.
_MAIN_ELLIPSOID_SEMI_AXES = (0.69, 0.92, 0.9)


def _shepp_logan_attenuation_scale(phantom_shape, target_max_attenuation):
    """Intensity scale so the peak forward projection of the phantom is
    ~``target_max_attenuation`` (verbatim mbirjax formula; assumes
    ``delta_voxel ~= 1``, since the phantom cannot see the projector's voxel
    spacing)."""
    longest_path_voxels = max(s * n for s, n in
                              zip(_MAIN_ELLIPSOID_SEMI_AXES, phantom_shape))
    interior_intensity = 0.28  # approximate average intensity along the center
    return (target_max_attenuation / longest_path_voxels) / interior_intensity


def generate_3d_shepp_logan_low_dynamic_range(phantom_shape,
                                              target_max_attenuation=None):
    """
    Generates a 3D Shepp-Logan phantom with specified dimensions.

    Args:
        phantom_shape (tuple): Phantom shape in (rows, columns, slices).
        target_max_attenuation (float, optional): If given, scale the phantom so
            that the peak line integral through it (its forward projection) is
            roughly this value, independent of the array shape.  Without it, the
            sinogram grows linearly with the array size (a ray crosses more
            voxels); real -log-attenuation sinograms sit around 0 to 6-8.
            Default None leaves the phantom unscaled.

    Returns:
        numpy.ndarray: a float32 array of shape ``phantom_shape`` with the voxel
        intensities of the phantom.

    Note:
        The build holds a few phantom-sized transients, so very large shapes use
        proportionally more peak memory (mbirjax's blocked/sharded builds exist
        for that regime and are not ported).
    """
    n_rows, n_cols, n_slices = phantom_shape
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, n_rows),
                                 np.linspace(-1, 1, n_cols), indexing='ij')
    z_locations = np.linspace(-1, 1, n_slices)

    phantom = _add_shepp_logan_ellipsoids(
        np.zeros(phantom_shape, dtype=np.float32), (x_grid, y_grid), z_locations)
    if target_max_attenuation is not None:
        phantom = phantom * np.float32(
            _shepp_logan_attenuation_scale(phantom_shape, target_max_attenuation))
    return phantom.astype(np.float32)
