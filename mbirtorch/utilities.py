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

import os
import shutil

import numpy as np


def clear_cache(_root=None):
    """Delete mbirtorch's on-disk state: the ``~/.mbirtorch`` directory.

    Today that directory holds only the persistent torch.compile cache
    (``~/.mbirtorch/torch_cache``; see the setup block in
    ``mbirtorch/__init__.py``), and anything the package adds under
    ``~/.mbirtorch`` in the future is removed with it.  The directory is
    recreated empty, so a running process is unaffected beyond paying the
    compile cost again on its next cold compile.  A cache redirected
    elsewhere via the ``TORCHINDUCTOR_CACHE_DIR`` environment variable is
    NOT touched -- that location is user-managed.

    The per-model pixel-index cache is unrelated: it is in-memory only (a
    single entry on the model instance, freed with the model) and never
    reaches disk.

    Args:
        _root: internal/testing override of the directory to clear.

    Returns:
        str: the path that was cleared.
    """
    root = os.path.expanduser("~/.mbirtorch") if _root is None else str(_root)
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    return root


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


def makedirs(file_path):
    """Create the parent directories of ``file_path`` if they do not exist."""
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(parent_dir, exist_ok=True)


def _to_host(array):
    """Move a numpy array or torch tensor (any device) to a host numpy array."""
    if hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def load_data_hdf5(file_path):
    """
    Load a numpy array from an HDF5 file.

    This function loads an array stored in an HDF5 file using :func:`save_data_hdf5`.
    It also loads any associated attributes and returns them as a dict.

    Args:
        file_path (str): Path to the HDF5 file containing the reconstructed volume.

    Returns:
        tuple: (array, data_dict)
            - array (ndarray): The array saved by :func:`save_data_hdf5`
            - data_dict (dict): A dict with the attributes for the data array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If more than one dataset is not found in the file.

    Example:
        >>> import mbirtorch
        >>> recon, recon_dict = mbirtorch.load_data_hdf5("output/recon_volume.h5")
        >>> recon.shape
        (64, 256, 256)
    """
    import h5py
    with h5py.File(file_path, "r") as f:
        array_names = [key for key in f.keys()]  # If this h5 file was created with save_data_hdf5, then there will be only one key
        if len(array_names) > 1:
            raise ValueError('More than one array found in {}. Unable to load.'.format(file_path))
        data_name = array_names[0]
        array = f[data_name][()]
        data_dict = dict()
        for name in f[data_name].attrs.keys():
            data_dict[name] = f[data_name].attrs[name]

        return array, data_dict


def _write_hdf5_streaming(file_path, array_name, out_shape, dtype, produce_slab, attributes_dict=None):
    """Create an HDF5 dataset of out_shape/dtype and fill it slab-by-slab along axis 0.

    produce_slab(i0, i1) returns the contiguous slab written to dset[i0:i1].  Only one slab is
    held at a time, so a large or strided source is never fully copied.
    """
    import h5py
    from .view_utils import convert_subdicts_to_strings
    makedirs(file_path)
    with h5py.File(file_path, 'w') as f:
        dset = f.create_dataset(array_name, shape=out_shape, dtype=dtype)
        if len(out_shape) == 0:
            dset[...] = produce_slab(0, 0)
        else:
            row_bytes = np.dtype(dtype).itemsize * int(np.prod(out_shape[1:], dtype=np.int64))
            slab = max(1, (1 << 30) // max(row_bytes, 1))   # ~1 GiB per write
            for i in range(0, out_shape[0], slab):
                dset[i:i + slab] = produce_slab(i, min(i + slab, out_shape[0]))
        if isinstance(attributes_dict, dict):
            attributes_dict = convert_subdicts_to_strings(attributes_dict)
            for key, value in attributes_dict.items():
                dset.attrs[key] = value


def save_data_hdf5(file_path, array, array_name='array', attributes_dict=None):
    """
    Save an array to an HDF5 file, optionally including metadata as attributes.
    The resulting structure has a single dataset with one array and associated text attributes.
    These can be retrieved using :func:`load_data_hdf5`.

    Args:
        file_path (str): Full path to the output HDF5 file. Directories will be created if they do not exist.
        array (ndarray or tensor): The volume data to save.
        array_name (str): Name of the dataset within the HDF5 file. Defaults to 'array'.
        attributes_dict (dict, optional): Dictionary of attributes to store as metadata in the dataset.
            Keys must be strings, and values should be serializable as HDF5 attributes.

    Returns:
        None

    Example:
        >>> import numpy as np
        >>> volume = np.random.rand(64, 64, 64)
        >>> attrs = {'voxel_size': '1.0mm', 'modality': 'CT'}
        >>> save_data_hdf5('output/recon.h5', volume, array_name='recon', attributes_dict=attrs)

    Example:
        >>> recon, recon_dict = ct_model.recon(sinogram)
        >>> recon_info = {'ALU units': '0.3mm', 'sinogram name': 'test part 038'}
        >>> file_path = './output/test_part_038.h5'
        >>> mbirtorch.save_data_hdf5(file_path, recon, recon_info)
    """
    array = _to_host(array) if hasattr(array, 'detach') else array

    # Stream the array to disk slab-by-slab (no full contiguous copy, even for a strided view).
    def produce_slab(i0, i1):
        return np.asarray(array) if array.ndim == 0 else np.ascontiguousarray(array[i0:i1])

    _write_hdf5_streaming(file_path, array_name, array.shape, array.dtype, produce_slab, attributes_dict)


def export_recon_hdf5(file_path, recon, recon_dict=None, remove_flash=False, radial_margin=10, top_margin=10, bottom_margin=10):
    """
    Export a 3D reconstruction volume to an HDF5 file with optional post-processing.

    This function works with either numpy arrays or torch tensors.
    The function also transposes the reconstruction to right-hand coordinates (slice, col, row),
    and writes the reconstruction and optional metadata to an HDF5 file.

    Args:
        file_path (str): Full path to the output HDF5 file. Parent directories will be created if they do not exist.
        recon (ndarray or tensor): 3D volume in (row, col, slice) order. Will be converted to NumPy before writing.
        recon_dict (dict, optional): Dictionary of attributes to store as metadata in the dataset.
        remove_flash (bool, optional): Whether to apply a cylindrical mask to remove peripheral and top/bottom slices. Defaults to False.
        radial_margin (int, optional): Margin in pixels to subtract from the cylinder radius. Defaults to 10.
        top_margin (int, optional): Number of top slices to set to zero along the Z-axis. Defaults to 10.
        bottom_margin (int, optional): Number of bottom slices to set to zero along the Z-axis. Defaults to 10.

    Example:
        >>> import numpy as np
        >>> recon = np.ones((128, 128, 64))  # (row, col, slice) order
        >>> export_recon_hdf5("output/recon_volume.h5", recon, recon_dict={"scan_id": "sample1"})
    """
    # Move the input to the host (NumPy) first so numpy and device tensors collapse to one host case.
    recon = _to_host(recon)

    if not remove_flash:
        # Transposed view; save_data_hdf5 streams it slab-by-slab, so no full copy is made.
        save_data_hdf5(file_path, np.transpose(recon, (2, 1, 0)), 'recon', recon_dict)
        return

    # remove_flash: mask + transpose + write one slab at a time, so no full masked volume is built.
    # Slabbing along the slice axis keeps full (rows, cols), so apply_cylindrical_mask gives the
    # identical circular mask per slab; we just map the global top/bottom margins to each slab.
    from . import preprocess
    num_rows, num_cols, num_slices = recon.shape

    def produce_slab(s0, s1):
        ds = s1 - s0
        local_top = min(max(top_margin - s0, 0), ds)                       # global top slices in this slab
        local_bottom = min(max(s1 - (num_slices - bottom_margin), 0), ds)  # global bottom slices in this slab
        block = preprocess.apply_cylindrical_mask(recon[:, :, s0:s1], radial_margin, local_top, local_bottom)
        return np.ascontiguousarray(np.transpose(block, (2, 1, 0)))        # (ds, C, R)

    _write_hdf5_streaming(file_path, 'recon', (num_slices, num_cols, num_rows), recon.dtype,
                          produce_slab, recon_dict)


def import_recon_hdf5(file_path):
    """
    Import a 3D reconstruction volume from an HDF5 file.

    This function loads a reconstruction volume and associated metadata from an HDF5 file,
    and reorders the volume axes from the file's (slice, col, row) layout to (row, col, slice)
    to match MBIRTORCH conventions, so a volume written by export_recon_hdf5 is recovered unchanged.

    Args:
        file_path (str): Path to the HDF5 file containing the reconstruction volume.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing:
            - recon (np.ndarray): The reconstructed 3D volume in (row, col, slice) order.
            - recon_dict (dict): Dictionary containing metadata associated with the reconstruction.

    Example:
        >>> from mbirtorch import import_recon_hdf5
        >>> recon, recon_dict = import_recon_hdf5("output/recon_volume.h5")
        >>> print(recon.shape)
        (128, 128, 64)
    """
    recon, recon_dict = load_data_hdf5(file_path=file_path)

    recon = np.transpose(recon, axes=(2, 1, 0))

    return recon, recon_dict
