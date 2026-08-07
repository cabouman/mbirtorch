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


def _resolve_geometry_class(geometry_type):
    """Resolve a model class from a ``geometry_type`` string (the class-identity entry recorded by
    ``get_all_params`` and the scan readers)."""
    import mbirtorch
    geometry_type = str(geometry_type)
    for name in ('ConeBeamModel', 'ParallelBeamModel', 'TranslationModel'):
        if name in geometry_type:
            model_class = getattr(mbirtorch, name, None)
            if model_class is None:
                raise ValueError(f"The geometry class {name} is not available in mbirtorch yet.")
            return model_class
    raise ValueError(f"Cannot resolve a model class for geometry_type {geometry_type!r}.")


def build_model(required_params, optional_params=None, regularization=None):
    """
    Construct a model from parameter dicts and compute its reconstruction geometry.

    The single place the ``construct -> set_params -> auto_set_recon_geometry`` sequence lives, so a
    caller never forgets the final ``auto_set_recon_geometry`` (which would leave the reconstruction
    grid sized with default detector pitches).  Because ``required_params`` carries ``geometry_type``
    (see :meth:`~mbirtorch.TomographyModel.get_all_params`), the correct model class is resolved here
    and ``(required_params, optional_params)`` is a self-contained model description -- calling this
    reads like calling the constructor through the new interface.

    Args:
        required_params (dict): The model constructor's arguments, including ``geometry_type`` (as
            returned in the first element of ``get_all_params``).
        optional_params (dict, optional): Additional parameters applied with ``set_params`` (detector
            pitches, offsets, ``delta_voxel``, ``recon_shape``, ...).  Defaults to None.
        regularization (dict, optional): Recon-time regularization parameters applied with
            ``set_params``.  Defaults to None (the model's default regularization).

    Returns:
        TomographyModel: the constructed model, with ``auto_set_recon_geometry`` applied.
    """
    required_params = dict(required_params)
    model_class = _resolve_geometry_class(required_params.pop('geometry_type'))
    model = model_class(**required_params)

    optional_params = dict(optional_params) if optional_params else {}
    # A pinned recon_shape must be applied AFTER auto_set_recon_geometry, or the automatic pass would
    # overwrite it (the translation reader pins recon_shape; a faithful save/load round-trip relies
    # on this ordering).
    pinned_recon_shape = optional_params.pop('recon_shape', None)
    # Apply the structural/optional params WITH name validation, so a typo'd key still raises; then
    # apply the regularization knobs with no_warning to suppress the "directly setting regularization"
    # advisory (this is a faithful rebuild, not a user hand-setting sigma_x).
    if optional_params:
        model.set_params(**optional_params)
    if regularization:
        model.set_params(no_warning=True, **regularization)
    model.auto_set_recon_geometry()
    if pinned_recon_shape is not None:
        model.set_params(no_warning=True, recon_shape=pinned_recon_shape)
    return model


def download_and_extract(download_url, save_dir):
    """
    Download or copy a file from a URL or local file path. If the file is a tarball (.tar, .tar.gz, etc.), extract it
    into the specified directory. Supports Google Drive links, standard HTTP/HTTPS URLs, and local paths.

    If the file already exists in the save directory, it will not be re-downloaded or copied.

    Args:
        download_url (str): URL or local file path to the file. Supported formats include:
            - Google Drive shared links
            - HTTP/HTTPS URLs
            - Local file paths
        save_dir (str): Directory where the file will be saved and extracted (if applicable).

    Returns:
        str:
            - For tar files: Path to the extracted top-level directory.
            - For other files: Path to the downloaded or copied file.

    Raises:
        RuntimeError: If the file cannot be downloaded, copied, or extracted.
        ValueError: If the Google Drive URL is invalid or tar file has no top-level directory.

    Examples:
        >>> extracted_dir = download_and_extract("https://example.com/data.tar.gz", "./data")
        >>> file_path = download_and_extract("https://drive.google.com/file/d/1ABC123/view", "./data")
        >>> result = download_and_extract("/path/to/local/data.tar.gz", "./data")
    """
    import re
    import subprocess
    import tarfile
    import urllib.error
    import urllib.request
    from urllib.parse import urlparse

    def is_google_drive_url(url):
        """Check if URL is a Google Drive link"""
        return "drive.google.com" in url

    def is_tar_file(filename):
        """Check if file is a tar archive based on extension"""
        tar_extensions = ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']
        return any(filename.lower().endswith(ext) for ext in tar_extensions)

    def extract_google_drive_id(url):
        """Extract Google Drive file ID from URL"""
        pattern = r"(?:https?:\/\/)?(?:www\.)?drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Invalid Google Drive URL format")

    parsed = urlparse(download_url)
    is_url = parsed.scheme in ('http', 'https')
    is_google_drive = is_url and is_google_drive_url(download_url)

    if is_google_drive:
        file_id = extract_google_drive_id(download_url)
        marker_file = os.path.join(save_dir, f".gdrive_{file_id}")

        if os.path.exists(marker_file):
            with open(marker_file, 'r') as f:
                actual_filename = f.read().strip()
            file_path = os.path.join(save_dir, actual_filename)
            filename = actual_filename

            if os.path.exists(file_path):
                is_download = False
            else:
                is_download = True

        else:
            filename = f"gdrive_{file_id}"
            is_download = True
    else:
        filename = os.path.basename(parsed.path if is_url else download_url)
        file_path = os.path.join(save_dir, filename)
        if os.path.exists(file_path):
            is_download = False
        else:
            is_download = True

    if is_download:
        os.makedirs(save_dir, exist_ok=True)

        if is_url:
            if is_google_drive:
                print("Downloading file from Google Drive...")
                import gdown
                try:
                    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

                    downloaded_path = gdown.download(gdrive_url, output=None, quiet=False)
                    if downloaded_path and os.path.isfile(downloaded_path):
                        actual_filename = os.path.basename(downloaded_path)
                        target_path = os.path.join(save_dir, actual_filename)
                        shutil.move(downloaded_path, target_path)
                        file_path = target_path
                        filename = actual_filename

                        with open(marker_file, 'w') as f:
                            f.write(actual_filename)
                    else:
                        raise RuntimeError("Google Drive download failed or returned invalid path")

                    print(f"Download successful! File saved to {file_path}")
                except Exception as e:
                    raise RuntimeError(f"Google Drive download failed: {str(e)}")
            else:
                print("Downloading file...")
                try:
                    urllib.request.urlretrieve(download_url, file_path)
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        raise RuntimeError(f'HTTP {e.code}: authentication failed!')
                    elif e.code == 403:
                        raise RuntimeError(f'HTTP {e.code}: URL forbidden!')
                    elif e.code == 404:
                        raise RuntimeError(f'HTTP {e.code}: URL not found!')
                    else:
                        raise RuntimeError(f'HTTP {e.code}: {e.reason}')
                except urllib.error.URLError as e:
                    res = subprocess.run(
                        ["curl", "-L", "--fail", "-o", file_path, download_url],
                        capture_output=True, text=True
                    )
                    if res.returncode != 0:
                        raise RuntimeError(f"Download failed with curl: {res.stderr.strip() or res.stdout.strip()}")
                print(f"Download successful! File saved to {file_path}")
        else:
            print(f"Copying local file from {download_url} to {file_path}...")
            if not os.path.isfile(download_url):
                raise RuntimeError(f"Provided file path does not exist: {download_url}")
            shutil.copy2(download_url, file_path)
            print(f"Copy successful! File saved to {file_path}")

        if is_tar_file(filename):
            print(f"Extracting tarball file to {save_dir}...")
            try:
                with tarfile.open(file_path, 'r') as tar_file:
                    tar_file.extractall(save_dir)
                print(f"Extraction successful!")

                top_level_dir = get_top_level_tar_dir(file_path)
                extracted_path = os.path.join(save_dir, top_level_dir)
                return extracted_path
            except Exception as e:
                raise RuntimeError(f"Failed to extract tar file: {str(e)}")
        else:
            return file_path

    if is_google_drive and not is_download:
        try:
            with open(marker_file, 'r') as f:
                actual_filename = f.read().strip()
            file_path = os.path.join(save_dir, actual_filename)
            filename = actual_filename
        except:
            file_path = os.path.join(save_dir, filename)

    if is_tar_file(filename):
        top_level_dir = get_top_level_tar_dir(file_path)
        file_path = os.path.join(save_dir, top_level_dir)

    return file_path


def get_top_level_tar_dir(tar_path, max_entries=1):
    """
    Determine the top-level directory inside a tarball file by sampling up to max_entries members.

    Parameters
    ----------
    tar_path : str
        Path to the tarball file.
    max_entries : int
        Maximum number of entries to sample.

    Returns
    -------
    dir_name : str
        The name of the top-level directory.
    """
    import tarfile
    top_levels = set()

    with tarfile.open(tar_path, 'r') as tar:
        for i, member in enumerate(tar):
            if not member.name.strip():
                continue
            top_dir = member.name.split('/')[0]
            top_levels.add(top_dir)

            if len(top_levels) > 1 or i + 1 >= max_entries:
                break
    if len(top_levels) == 1:
        dir_name = top_levels.pop()
    else:
        raise ValueError("No top level directory found in {}".format(tar_path))
    return dir_name


def calc_tct_recon_params(source_det_dist, source_iso_dist, delta_det_row, delta_det_channel, sinogram_shape, translation_vectors, voxel_row_aspect=1.0, voxel_slice_aspect=1.0):
    """
    Calculate the translation geometry parameters: recon_shape, delta_voxel, voxel_row_aspect

    Args:
        source_det_dist (float): distance from the X-ray source to the detector (in ALU)
        source_iso_dist (float): distance from the X-ray source to the isocenter (in ALU)
        delta_det_row (float): the spacing between detector rows (in ALU)
        delta_det_channel (float): the spacing between detector channels (in ALU)
        sinogram_shape (tuple): Shape of the sinogram as (num_views, num_det_rows, num_det_channels)
        translation_vectors (numpy array): A (num_views, 3) array of translations (x, y, z) in ALU
        voxel_row_aspect (float): the aspect ratio between delta_voxel_row and delta_voxel. Defaults to 1.0
        voxel_slice_aspect (float): the aspect ratio between delta_voxel_slice and delta_voxel. Defaults to 1.0

    Returns:
        recon_shape (tuple): Shape of the reconstruction shape as (num_recon_rows, num_recon_cols, num_recon_slices)
        delta_voxel (float): the voxel pitch at isocenter (in ALU)
        voxel_row_aspect (float): the aspect ratio between delta_voxel_row and delta_voxel
    """
    # Get parameters
    num_views, num_det_rows, num_det_channels = sinogram_shape

    # Calculate magnification
    magnification = source_det_dist / source_iso_dist

    # Calculate the width and height of the detector in ALU
    detect_box = np.array([delta_det_channel * num_det_channels, delta_det_row * num_det_rows])

    # Compute avg_view_slope = tan(cone_angle/2) along the x and z directions
    # This is the average slope of a view that a pixel at iso sees.
    # detect_box/4 = distance from the (center of the detector) to (halfway to the edge of the detector).
    # Using the average seems to be better than using the maximum.
    avg_view_slope = (detect_box / 4) / source_det_dist

    # Compute detector pixel pitch at iso
    # Note that this may differ from delta_voxel
    # However, we will use det_pixel_pitch_iso to calculate both the number rows and their pitch
    det_pixel_pitch_iso_vec = np.array([delta_det_row, delta_det_channel]) / magnification
    det_pixel_pitch_iso = np.max(det_pixel_pitch_iso_vec)

    # Set delta_voxel
    delta_voxel = float(det_pixel_pitch_iso)

    # Compute delta_voxel in slice dimension
    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    ######### Compute the row pitch based on a heuristic #########
    # The following code will result in an isotropic voxel when the avg_view_slope > 76 deg.
    nominal_row_pitch = 4.0 * det_pixel_pitch_iso_vec / avg_view_slope
    nominal_row_pitch = np.max(nominal_row_pitch)  # Take the maximum of the nominal pitches along x and z
    delta_recon_row = np.maximum(nominal_row_pitch, det_pixel_pitch_iso)  # Ensure that the row resolution is not higher than the (x,z) detector resolution
    delta_recon_row = float(delta_recon_row)

    ##### Compute voxel row aspect
    # In translation geometry, anisotropic row spacing is usually needed for good reconstruction results.
    #
    # If voxel_row_aspect == 1.0 (default value), assume the user did not explicitly specify
    # a row aspect ratio, and automatically compute it using the current TCT row-pitch heuristic.
    #
    # Otherwise, use the user-defined voxel_row_aspect to determine delta_recon_row.
    if voxel_row_aspect == 1.0:
        voxel_row_aspect = delta_recon_row / delta_voxel
    else:
        delta_recon_row = voxel_row_aspect * delta_voxel

    # Compute cube = (width, depth, height) of the scanned region in ALU
    max_translation = np.amax(translation_vectors, axis=0)  # Translate object right/up when positive
    min_translation = np.amin(translation_vectors, axis=0)  # Translate object left/down when negative
    cube = max_translation - min_translation

    # Compute recon_box = (num_recon_cols, num_recon_slices) of the reconstruction volume.
    # The reconstruction box size is determined using:
    #   delta_voxel for the column direction
    #   delta_voxel_slice for the slice direction
    recon_box = np.ceil(np.array([cube[0], cube[2]]) / np.array([delta_voxel, delta_voxel_slice]))

    # ************ Use a heuristic to determine a reasonable number of rows *************
    # Compute the number of unknown pixels per view
    num_pixels_per_view = ((recon_box[0] + num_det_rows) * (recon_box[1] + num_det_channels)) / num_views
    num_measurements_per_view = num_det_channels * num_det_rows
    # Select the number of rows so that (number of unknowns) = 2*(the number of measurements)
    num_recon_rows = 2 * np.ceil(num_measurements_per_view / num_pixels_per_view)

    # Make sure the object extends no further than halfway to the source
    max_recon_rows = np.floor((source_iso_dist - cube[1]) / delta_recon_row)
    if max_recon_rows < 1:
        print(f"[Error] Computed max_recon_rows = {max_recon_rows} < 1. This suggests the object extends beyond the source.")
    num_recon_rows = np.minimum(num_recon_rows, max_recon_rows)

    # Set the parameters to their computed values
    num_recon_cols, num_recon_slices = recon_box
    num_recon_cols = int(num_recon_cols)
    num_recon_rows = int(num_recon_rows)
    num_recon_slices = int(num_recon_slices)
    recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)

    return recon_shape, delta_voxel, voxel_row_aspect


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
