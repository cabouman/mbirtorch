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
import warnings
from enum import Enum

import numpy as np
from . import _sharding


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
    """Move a numpy array, torch tensor (any device), or sharded volume to a
    host numpy array.  A sharded volume is concatenated on the host and any
    zero-padding of its sharded axis is cropped, so the result equals the
    single-device volume."""
    if isinstance(array, _sharding.Shards):
        # gather() already returns numpy; do not convert it again.
        out = array.gather()
        pl = array.placement
        if pl.real_size is not None and pl.padded_size > pl.real_size:
            sel = [slice(None)] * out.ndim
            sel[pl.axis % out.ndim] = slice(0, pl.real_size)
            out = out[tuple(sel)]
        return out
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
    array = (_to_host(array)
             if (hasattr(array, 'detach') or isinstance(array, _sharding.Shards))
             else array)

    # Stream the array to disk slab-by-slab (no full contiguous copy, even for a strided view).
    def produce_slab(i0, i1):
        return np.asarray(array) if array.ndim == 0 else np.ascontiguousarray(array[i0:i1])

    _write_hdf5_streaming(file_path, array_name, array.shape, array.dtype, produce_slab, attributes_dict)


def export_recon_hdf5(file_path, recon, recon_dict=None, remove_flash=False, radial_margin=10, top_margin=10, bottom_margin=10):
    """
    Export a 3D reconstruction volume to an HDF5 file with optional post-processing.

    This function works with numpy arrays, torch tensors, and sharded volumes (a ``Shards``
    container): a sharded volume is gathered to the host at this file boundary and any
    zero-padding of its slice axis is cropped, so the file equals the single-device export.
    The function also transposes the reconstruction to right-hand coordinates (slice, col, row),
    and writes the reconstruction and optional metadata to an HDF5 file.

    Args:
        file_path (str): Full path to the output HDF5 file. Parent directories will be created if they do not exist.
        recon (ndarray, tensor, or Shards): 3D volume in (row, col, slice) order. Will be converted to NumPy before writing.
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


def stitch_arrays(array_list, overlap, axis=2, ramp_overlap=None):
    """
    Concatenate arrays along one axis while linearly blending a fixed overlap
    between adjacent arrays.

    This behaves like a concatenate except that for each adjacent pair, the
    first `overlap_length` elements of the second array and the last
    `overlap_length` elements of the current result are combined by a piece-wise linear cross‑fade.

    All non‑`axis` dimensions must match across inputs.

    Args:
        array_list (list of ndarray or tensor): Sequence of 2+ arrays to stitch.  The result is
            built on the inputs' own array module, so host (NumPy) inputs stitch on the host (no
            gather to a single device) and device tensors stitch on-device.
        overlap (int): Number of elements overlapped between arrays.
            Must be `>= 1` and not exceed the length of any input along `axis`.
        axis (int, optional): Axis along which to stitch. Defaults to 2.
        ramp_overlap (int, optional): Target number of blended (0 < w < 1) elements. Defaults to None.

    Returns:
        ndarray or tensor: Stitched array, on the inputs' own array module (host NumPy in -> host
        out, tensor in -> on-device out). Its shape equals the input shape with the
        length along `axis` equal to:

            sum(len_k) - (len(array_list) - 1) * overlap_length

        where `len_k` are the lengths of each input along `axis`.

    Raises:
        ValueError: If fewer than two arrays are provided, if non‑`axis`
            dimensions differ, or if any array is shorter than
            `overlap_length` along `axis`.

    Example:
        >>> import numpy as np
        >>> a0 = np.arange(2*2*5.).reshape(2, 2, 5)
        >>> a1 = np.arange(2*2*6.).reshape(2, 2, 6)
        >>> out = stitch_arrays([a0, a1], overlap=3, axis=2)
        >>> out.shape
        (2, 2, 8)

        # 8 comes from 5 + 6 - 3 (one overlap between two arrays).
    """
    import torch

    # Check for valid input
    if not isinstance(array_list, list) or len(array_list) < 2:
        raise ValueError('array_list must be a list of 2 or more arrays.')
    for dim in range(array_list[0].ndim):
        lengths = [array.shape[dim] for array in array_list]
        if dim != axis:
            if np.amax(lengths) != np.amin(lengths):
                raise ValueError('The shapes of the arrays in array_list must be the same except in the dimension specified by axis.')
        if dim == axis:
            if np.amin(lengths) < overlap:
                raise ValueError('Each array must have length at least overlap in the dimension specified by axis.')

    # Create weights for blending two arrays
    # ramp_overlap is the target number of blended (0 < w < 1) pixels
    # However, if ramp_overlap and overlap have different parities, then ramp_overlap is decremented to match parity.
    if ramp_overlap is None:
        ramp_overlap = overlap // 2  # default: ramp over ~half the overlap
    ramp_overlap = min(ramp_overlap, overlap)
    ramp_overlap -= (overlap - ramp_overlap) % 2  # match overlap's parity -> symmetric plateaus
    ramp_overlap = max(ramp_overlap, overlap % 2)  # floor at 0 (even overlap) or 1 (odd overlap)
    flat_pad = (overlap - ramp_overlap) // 2  # equal plateau on each side

    # Build the blend weights and assemble on the inputs' OWN array module so the result stays where
    # the inputs live: host (NumPy) arrays stitch on the HOST (no gather to a single device), device
    # tensors stitch on-device.  split_sino_recon relies on this -- it passes host halves, so the full
    # volume is never reassembled on one GPU (which would defeat the half-at-a-time memory saving and
    # OOM for a recon too large to fit whole).  float32 weights avoid upcasting a float32 recon to f64.
    is_torch = any(isinstance(a, torch.Tensor) for a in array_list)
    if is_torch:
        device = next(a.device for a in array_list if isinstance(a, torch.Tensor))
        ramp = (torch.arange(ramp_overlap, dtype=torch.float32, device=device) + 1) / (ramp_overlap + 1)
        weights = torch.cat([torch.zeros(flat_pad, dtype=torch.float32, device=device), ramp,
                             torch.ones(flat_pad, dtype=torch.float32, device=device)])
        swap, cat = torch.swapaxes, torch.cat
        array_list = [torch.as_tensor(a, device=device) for a in array_list]
    else:
        ramp = (np.arange(ramp_overlap, dtype=np.float32) + 1) / (ramp_overlap + 1)  # strictly between 0 and 1
        weights = np.concatenate([np.zeros(flat_pad, dtype=np.float32), ramp,
                                  np.ones(flat_pad, dtype=np.float32)])
        swap, cat = np.swapaxes, np.concatenate

    # Broadcast weights to match array dimensions
    weights_shape = [1] * array_list[0].ndim
    weights_shape[0] = len(weights)
    weights = weights.reshape(weights_shape)

    # Start with the first array in the list
    stitched = swap(array_list[0], 0, axis)

    # Iterate through each subsequent array in the list
    for next_array in array_list[1:]:
        # Extract the overlap from the current end of the stitched array and the beginning of the next array
        overlap_current = stitched[-overlap:]
        next_array = swap(next_array, 0, axis)
        overlap_next = next_array[:overlap]

        # Weighted average for the overlapping part
        weighted_overlap = (1 - weights) * overlap_current + weights * overlap_next

        # Replace the overlap in the stitched array
        stitched = cat([stitched[:-overlap], weighted_overlap], 0)

        # Append the non-overlapping remainder of the next array
        stitched = cat([stitched, next_array[overlap:]], 0)

    return swap(stitched, 0, axis)


def copy_ct_model(ct_model, new_angles=None, new_helical_z_shifts=None, new_num_det_rows=None, new_num_det_cols=None):
    """
    Create a TomographyModel with the same type and parameters as the given ct_model except with the new input angles
    and a corresponding sinogram shape.  Restricted to ParallelBeam and ConeBeam models.

    If the user explicitly set the devices on ct_model with configure_devices, the copy
    gets the same devices.  Otherwise the copy chooses its own devices when it is used.

    Args:
        ct_model (TomographyModel): The model to copy.
        new_angles (ndarray of float, optional): 1D vector of projection angles in radians.
            If None, then use the angles in ct_model. Defaults to None.
        new_helical_z_shifts (ndarray of float, optional): 1D vector of per-view axial shifts in ALU for ConeBeamModel.
            Defaults to None.
        new_num_det_rows (int, optional): Number of detector rows in the new model.
            If None, then use the num_det_rows in ct_model. Defaults to None.
        new_num_det_cols (int, optional): Number of detector columns in the new model.
            If None, then use the num_det_cols in ct_model. Defaults to None.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    if str(type(ct_model)).find('ConeBeamModel') > 0:
        is_cone = True
    elif str(type(ct_model)).find('ParallelBeamModel') > 0:
        is_cone = False
    else:
        raise TypeError('copy_ct_model() is restricted to ConeBeam and ParallelBeam Models')

    # get_all_params is the single source of truth for reading the params back out: it gives the
    # constructor args with the view components already unpacked (angles + helical_z_shifts for cone)
    # and geometry_type in required, so build_model can reconstruct the class.
    required, optional, regularization = ct_model.get_all_params()

    old_angles = required['angles']
    new_shape = list(required['sinogram_shape'])

    if is_cone:
        old_helical_z_shifts = required['helical_z_shifts']
        if new_angles is None and new_helical_z_shifts is None:
            new_helical_z_shifts = old_helical_z_shifts
        elif new_angles is not None and new_helical_z_shifts is None:
            if np.any(np.asarray(old_helical_z_shifts) != 0):
                raise ValueError('copy_ct_model: new_helical_z_shifts must be specified when changing angles for a helical scan.')
            new_helical_z_shifts = np.zeros_like(new_angles)
        elif new_angles is not None and new_helical_z_shifts is not None:
            if len(new_angles) != len(new_helical_z_shifts):
                raise ValueError('copy_ct_model: new_angles and new_helical_z_shifts must have the same length.')
        elif new_angles is None and new_helical_z_shifts is not None:
            if len(old_helical_z_shifts) != len(new_helical_z_shifts):
                raise ValueError('copy_ct_model: new_helical_z_shifts must have the same length as the existing angles.')
        required['helical_z_shifts'] = new_helical_z_shifts

    if new_angles is None:
        new_angles = old_angles
    new_shape[0] = len(new_angles)
    if new_num_det_rows is not None:
        new_shape[1] = new_num_det_rows
    if new_num_det_cols is not None:
        new_shape[2] = new_num_det_cols
    required['angles'] = new_angles
    required['sinogram_shape'] = tuple(new_shape)

    # The sinogram shape changed, so drop recon_shape and let build_model's auto pass recompute it.
    optional.pop('recon_shape', None)
    new_model = build_model(required, optional, regularization)
    # If the user explicitly set the devices, the copy inherits them.
    if not ct_model.device_layout_is_automatic:
        new_model.configure_devices(devices=list(ct_model.sino_placement.devices))
    return new_model


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


def merge_log_files(merged_path, labeled_paths):
    """Merge temp log files into one file, each under a section header, and remove the temps.

    Missing temps are skipped; if none exist, no file is written.  A closing line names
    the merged file, since the per-part 'Logs written to' lines name temps that no
    longer exist.

    Args:
        merged_path (str): Path of the merged output file.
        labeled_paths (iterable): (label, path) pairs in the order they should appear.
    """
    labeled_paths = [(label, path) for label, path in labeled_paths
                     if path is not None and os.path.exists(path)]
    if not labeled_paths:
        return
    with open(merged_path, 'w') as merged:
        for label, path in labeled_paths:
            merged.write('======== {} ========\n'.format(label))
            with open(path, 'r') as f:
                merged.write(f.read())
            os.remove(path)
        merged.write('Merged logs written to {}\n'.format(
            os.path.abspath(merged_path)))


def get_ct_model(geometry_type, sinogram_shape, angles, source_detector_dist=None, source_iso_dist=None, helical_z_shifts=None):
    """
    Create an instance of TomographyModel with the given parameters

    Args:
        geometry_type (str): 'parallel' or 'cone'
        sinogram_shape (tuple list of int): (num_views, num_rows, num_channels)
        angles (ndarray of float): 1D vector of projection angles in radians
        source_detector_dist (float or None, optional): Distance in ALU from source to detector.  Defaults to None for geometries that don't need this.
        source_iso_dist (float or None, optional): Distance in ALU from source to iso.  Defaults to None for geometries that don't need this.
        helical_z_shifts (ndarray, optional):
            Per-view axial shifts (ALU), same length as angles.
            Required when use_helical=True.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    import mbirtorch

    if geometry_type == 'cone':
        model = mbirtorch.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                        source_iso_dist=source_iso_dist, helical_z_shifts=helical_z_shifts)
    elif geometry_type == 'parallel':
        if helical_z_shifts is not None:
            warnings.warn("Helical mode (helical_z_shifts) is only supported for geometry_type='cone'; ignoring z_shifts.", UserWarning)
        model = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    return model


def _gen_ellipsoid(x_grid, y_grid, z_grid, x0, y0, z0, a, b, c, gray_level, alpha=0, beta=0, gamma=0):
    """
    Return an image with a 3D ellipsoid in a 3D plane with a center of [x0,y0,z0] and ...

    Args:
        x_grid(ndarray): 3D grid of X coordinate values.
        y_grid(ndarray): 3D grid of Y coordinate values.
        z_grid(ndarray): 3D grid of Z coordinate values.
        x0(float): horizontal center of ellipsoid.
        y0(float): vertical center of ellipsoid.
        z0(float): normal center of ellipsoid.
        a(float): X-axis radius.
        b(float): Y-axis radius.
        c(float): Z-axis radius.
        gray_level(float): Gray level for the ellipse.
        alpha(float): [Default=0.0] counter-clockwise angle of rotation by X-axis in radians.
        beta(float): [Default=0.0] counter-clockwise angle of rotation by Y-axis in radians.
        gamma(float): [Default=0.0] counter-clockwise angle of rotation by Z-axis in radians.

    Return:
        ndarray: 3D array with the same shape as x_grid, y_grid, and z_grid

    """
    # Generate Rotation Matrix.
    rx = np.array([[1, 0, 0], [0, np.cos(-alpha), -np.sin(-alpha)], [0, np.sin(-alpha), np.cos(-alpha)]])
    ry = np.array([[np.cos(-beta), 0, np.sin(-beta)], [0, 1, 0], [-np.sin(-beta), 0, np.cos(-beta)]])
    rz = np.array([[np.cos(-gamma), -np.sin(-gamma), 0], [np.sin(-gamma), np.cos(-gamma), 0], [0, 0, 1]])
    r = np.dot(rx, np.dot(ry, rz))

    cor = np.array([x_grid.flatten() - x0, y_grid.flatten() - y0, z_grid.flatten() - z0])

    image = ((np.dot(r[0], cor)) ** 2 / a ** 2 + (np.dot(r[1], cor)) ** 2 / b ** 2 + (
        np.dot(r[2], cor)) ** 2 / c ** 2 <= 1.0) * gray_level

    return image.reshape(x_grid.shape)


def generate_3d_shepp_logan_reference(phantom_shape):
    """
    Generate a 3D Shepp Logan phantom based on below reference.

    Kak AC, Slaney M. Principles of computerized tomographic imaging. Page.102. IEEE Press, New York, 1988. https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf

    Args:
        phantom_shape (tuple or list of ints): num_rows, num_cols, num_slices

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols

    Note:
        This function produces 6 intermediate arrays that each have shape phantom_shape, so if phantom_shape is
        large, then this will use a lot of peak memory.
    """

    # The function describing the phantom is defined as the sum of 10 ellipsoids inside a 2×2×2 cube:
    sl3d_paras = [
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.98},
        {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.02},
        {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.02},
        {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.02}
    ]

    num_rows, num_cols, num_slices = phantom_shape
    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    axis_z = np.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = np.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    for el_paras in sl3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                               z0=el_paras['z0'],
                               a=el_paras['a'], b=el_paras['b'], c=el_paras['c'],
                               gamma=el_paras['gamma'] / 180.0 * np.pi,
                               gray_level=el_paras['gray_level'])

    return image.transpose((1, 0, 2))


def gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing):
    """
    Generate translation vectors for lateral (x) and axial (z) displacements.

    Args:
        num_x_translations (int): Number of x-direction translations
        num_z_translations (int): Number of z-direction translations
        x_spacing (float): Spacing between x translations in ALU
        z_spacing (float): Spacing between z translations in ALU

    Returns:
        np.ndarray: Array of shape (num_views, 3) with translation vectors [dx, dy, dz]
    """
    num_views = num_x_translations * num_z_translations
    translation_vectors = np.zeros((num_views, 3))

    x_center = (num_x_translations - 1) / 2
    z_center = (num_z_translations - 1) / 2

    idx = 0
    for row in range(num_z_translations):
        for col in range(num_x_translations):
            dx = (col - x_center) * x_spacing
            dz = (row - z_center) * z_spacing
            dy = 0
            translation_vectors[idx] = [dx, dy, dz]
            idx += 1

    return translation_vectors


def gen_cube_phantom(recon_shape, device=None):
    """Code to generate a simple phantom """
    import torch

    # Compute phantom height and width
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom_rows = num_recon_rows // 4  # Phantom height
    phantom_cols = num_recon_cols // 4  # Phantom width

    # Allocate phantom memory.  float32 explicitly: mbirjax's jnp.array does
    # this downcast for free (jax defaults to 32-bit), torch.as_tensor keeps
    # whatever numpy gave it, and float64 both doubles the memory and is
    # unsupported on mps.
    phantom = np.zeros((num_recon_rows, num_recon_cols, num_recon_slices),
                       dtype=np.float32)

    # Compute start and end locations
    start_rows = (num_recon_rows - phantom_rows) // 2
    stop_rows = (num_recon_rows + phantom_rows) // 2
    start_cols = (num_recon_cols - phantom_cols) // 2
    stop_cols = (num_recon_cols + phantom_cols) // 2
    for slice_index in np.arange(num_recon_slices):
        shift_cols = int(slice_index * phantom_cols / num_recon_slices)
        phantom[start_rows:stop_rows, (shift_cols + start_cols):(shift_cols + stop_cols), slice_index] = 1.0 / max(
            phantom_rows, phantom_cols)

    return torch.as_tensor(phantom, device=device)


def get_helical_half_rotation_slice_range(
    ct_model,
    helical_pitch,
    helical_z_shifts,
):
    """
    Return the contiguous slice range whose z positions are visible for at least
    half a rotation.

    Assumes helical_z_shifts are monotone and uniformly sampled.

    Returns:
        start_slice, stop_slice
        where stop_slice is exclusive.
    """
    recon_shape = ct_model.get_params('recon_shape')
    delta_voxel, voxel_slice_aspect, recon_slice_offset = ct_model.get_params(
        ['delta_voxel', 'voxel_slice_aspect', 'recon_slice_offset']
    )
    sinogram_shape = ct_model.get_params('sinogram_shape')
    delta_det_row = ct_model.get_params('delta_det_row')
    magnification = ct_model.get_magnification()

    num_slices = recon_shape[2]
    num_det_rows = sinogram_shape[1]

    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    # Slice-center z locations in reconstruction coordinates.
    k = np.arange(num_slices)
    z_k = delta_voxel_slice * (k - (num_slices - 1) / 2.0) + recon_slice_offset

    # Detector height mapped to isocenter.
    det_height_iso = num_det_rows * delta_det_row / magnification

    # Table travel range.
    z_shift_min = np.min(helical_z_shifts)
    z_shift_max = np.max(helical_z_shifts)

    # Extra interior trim needed when pitch > 1.
    # For pitch <= 1, the table-travel endpoints already have at least
    # half-rotation visibility, so no trim is needed.
    trim = 0.5 * det_height_iso * np.maximum(float(helical_pitch) - 1.0, 0.0)

    z_min = z_shift_min + trim
    z_max = z_shift_max - trim

    valid_slice_mask = (z_k >= z_min) & (z_k <= z_max)

    valid_indices = np.where(np.asarray(valid_slice_mask))[0]
    if len(valid_indices) == 0:
        raise ValueError(
            "No slices are visible for at least half a rotation. "
            "Check helical_pitch, helical_z_range, num_views, and detector height."
        )

    start_slice = int(valid_indices[0])
    stop_slice = int(valid_indices[-1] + 1)

    return start_slice, stop_slice


class ObjectType(str, Enum):
    SHEPP_LOGAN = 'shepp-logan'
    CUBE = 'cube'


class ModelType(str, Enum):
    PARALLEL = 'parallel'
    CONE = 'cone'
    TRANSLATION = 'translation'


def generate_demo_data(
    object_type='shepp-logan',
    model_type='cone',
    num_views=64,
    num_det_rows=96,
    delta_det_row=1,
    num_det_channels=128,
    delta_det_channel=1,
    num_x_translations=7,
    num_z_translations=7,
    x_spacing=22,
    z_spacing=22,
    use_helical=False,
    helical_pitch=None,
    helical_z_range=None,
    helical_z_center=0.0,
    use_curved_detector=False,
    voxel_row_aspect=1.0,
    voxel_slice_aspect=1.0,
    target_max_attenuation=None,
    devices=None,
):
    """
    Create a simple object and a sinogram for demonstration purposes.

    This function will create a 3D volume (aka object or phantom) of the specified type, then use the model type and
    parameters to create a simulated sinogram.  The object type 'shepp-logan' gives a simplified version of the
    classic Shepp-Logan test phantom, and type 'cube' gives a simple cube object.

    The output sinogram has shape (num_views, num_det_rows, num_det_channels); each 2D array
    sinogram[view_index] is a simulated image from the detector, with num_det_rows indicating the
    vertical size and num_det_channels the horizontal size.

    Args:
        object_type (str, optional): One of 'shepp-logan' or 'cube'.  Defaults to 'shepp-logan'.
        model_type (str, optional): One of 'parallel' or 'cone'.  Defaults to 'cone'.  The
            translation geometry is not available yet, and asking for it raises
            NotImplementedError.
        num_views (int, optional):  Number of views in the output sinogram.  Defaults to 64. Ignored when model_type is 'translation'
        num_det_rows (int, optional): Number of rows (vertical) in the output sinogram.  Defaults to 96.
        num_det_channels (int, optional): Number of channels (horizontal) in the output sinogram.  Defaults to 128.
        num_x_translations (int, optional): Number of horizontal translations for translation mode.  Defaults to 7.
        num_z_translations (int, optional): Number of vertical translations for translation mode.  Defaults to 7.
        x_spacing (float, optional): Horizontal spacing between translations in ALU.  Defaults to 22.
        z_spacing (float, optional): Vertical spacing between translations in ALU.  Defaults to 22.
        use_helical (bool, optional):
            If True and model_type == 'cone', generate a helical cone-beam trajectory by
            supplying per-view z_shifts to ConeBeamModel. Defaults to False.
        helical_pitch (float, optional):
            Helical pitch (dimensionless) for helical mode.
            pitch = (table travel per rotation) / (det height at iso).  This is the fraction of the detector height at iso traveled per rotation.
        helical_z_range (float, optional): Total axial travel over the scan in ALU for helical mode.
        helical_z_center (float, optional): Midpoint of axial travel over the scan in ALU for helical mode.
        use_curved_detector (bool, optional): (cone beam geometry parameter)
        voxel_row_aspect (float, optional): Aspect ratio for recon rows relative to columns.  Defaults to 1.0.
        voxel_slice_aspect (float, optional): Aspect ratio for recon slices relative to rows.  Defaults to 1.0.
        target_max_attenuation (float, optional): Target max sinogram attenuation for Shepp-Logan phantom.  Defaults to None, for which each voxel is in the range [0, 1].  May not be accurate if any detector or voxel dimensions are not 1.
        devices (sequence of devices, optional): Devices to run the generation on.  Defaults to None,
            which uses the model's automatic selection.  This only affects where the work runs, not
            the result.

    Returns:
        tuple: (object, sinogram, params)
            - object: the phantom volume, shape recon_shape = (num_rows, num_cols, num_slices).
              A host numpy float32 array, for either object type.
            - sinogram: shape (num_views, num_det_rows, num_det_channels).
            - params (dict): contains 'angles' and, for 'cone', also 'source_detector_dist' and 'source_iso_dist'.

        sinogram is always a host NumPy array (what ``recon`` prefers).
    """
    import mbirtorch

    # Coerce types to Enum
    object_type = ObjectType(object_type)
    model_type = ModelType(model_type)

    start_angle = -np.pi
    end_angle = np.pi

    # Initialize model

    if model_type == ModelType.PARALLEL:
        start_angle = 0
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)
        ct_model_for_generation = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
        ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
        ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
        ct_model_for_generation.auto_set_recon_geometry()
        params = {'angles': angles, 'voxel_row_aspect': voxel_row_aspect, 'voxel_slice_aspect': voxel_slice_aspect}
    elif model_type == ModelType.CONE:
        # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist/2
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        if not use_helical:
            angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)
            ct_model_for_generation = mbirtorch.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                                              source_iso_dist=source_iso_dist, use_curved_detector=use_curved_detector)
            ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
            ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
            ct_model_for_generation.auto_set_recon_geometry()
            params = {'angles': angles, 'source_detector_dist': source_detector_dist, 'source_iso_dist': source_iso_dist,
                      'use_curved_detector': use_curved_detector, 'voxel_row_aspect': voxel_row_aspect, 'voxel_slice_aspect': voxel_slice_aspect}
        else:
            # Require both helical_pitch and helical_z_range
            if helical_pitch is None or helical_z_range is None:
                raise ValueError("Helical trajectory requires both helical_pitch and helical_z_range.")

            # Compute magnification
            if np.isinf(source_detector_dist):
                magnification = 1
            else:
                magnification = source_detector_dist / source_iso_dist

            # detector height mapped to iso, in ALU
            det_height_iso = float(num_det_rows) * (delta_det_row / magnification)

            # Travel per rotation (ALU) and derived rotations/views-per-rotation
            z_per_rot = float(helical_pitch) * det_height_iso
            if z_per_rot <= 0:
                raise ValueError(f"helical_pitch must be > 0 (got {helical_pitch}).")
            if float(helical_z_range) < 0:
                raise ValueError(f"helical_z_range must be >= 0 (got {helical_z_range}).")

            # Derived number of rotations and views per rotation
            if float(helical_z_range) == 0.0: # circular reconstruction
                num_rotations = 1.0
                views_per_rotation = float(num_views)
            else:
                num_rotations = float(helical_z_range) / z_per_rot
                if num_rotations <= 0:
                    raise ValueError("Derived num_rotations <= 0; check pitch/z_range.")
                views_per_rotation = float(num_views) / num_rotations

            # Angles: advance by 2*pi/views_per_rotation each view
            angle_step = (2.0 * np.pi) / views_per_rotation
            angles = start_angle + angle_step * np.arange(num_views)

            # z_shifts: span z_range across scan, centered at z_center
            z0 = float(helical_z_center) - 0.5 * float(helical_z_range)
            z1 = float(helical_z_center) + 0.5 * float(helical_z_range)
            helical_z_shifts = np.linspace(z0, z1, num_views, endpoint=True)

            ct_model_for_generation = mbirtorch.ConeBeamModel(
                sinogram_shape,
                angles,
                source_detector_dist=source_detector_dist,
                source_iso_dist=source_iso_dist,
                helical_z_shifts=helical_z_shifts,
                use_curved_detector=use_curved_detector
            )
            ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
            ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
            ct_model_for_generation.auto_set_recon_geometry()

            params = {
                'angles': angles,
                'source_detector_dist': source_detector_dist,
                'source_iso_dist': source_iso_dist,
                'helical_z_shifts': helical_z_shifts,
                'use_curved_detector': use_curved_detector,
                'voxel_row_aspect': voxel_row_aspect,
                'voxel_slice_aspect': voxel_slice_aspect
            }
    elif model_type == ModelType.TRANSLATION:
        # The lines below are the translation path carried over from mbirjax.  They are
        # kept so that porting TranslationModel is a deletion of this raise, and they
        # cannot run until then: without the raise the branch dies partway through on a
        # missing attribute, which says nothing about what is actually unavailable.
        raise NotImplementedError(
            "generate_demo_data does not support model_type='translation': the "
            "translation geometry (TranslationModel) is not ported to mbirtorch yet.  "
            "Use 'parallel' or 'cone'.")
        source_iso_dist = min(num_det_rows, num_det_channels) / 2
        source_detector_dist = source_iso_dist
        translation_vectors = gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)
        num_views = translation_vectors.shape[0]
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        ct_model_for_generation = mbirtorch.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist,
                                                             source_iso_dist=source_iso_dist)
        params = {'translation_vectors': translation_vectors}
    else:
        raise ValueError(f'Invalid model type. Expected one of {[m.value for m in ModelType]}, got {model_type}')

    # Pin the generation model to the requested devices so the phantom projection and the returned
    # sinogram share one layout.  None leaves the automatic selection in place.
    if devices is not None:
        ct_model_for_generation.configure_devices(devices=list(devices))

    # Generate the phantom.  The phantom builders return host arrays, so no device layout is needed
    # here; the forward projection below handles device placement itself.
    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    phantom_shape = recon_shape
    embed_slice_start = 0
    embed_slice_stop = recon_shape[2]
    if model_type == ModelType.CONE and use_helical:
        embed_slice_start, embed_slice_stop = get_helical_half_rotation_slice_range(
            ct_model_for_generation,
            helical_pitch,
            helical_z_shifts
        )
        phantom_shape = (
            recon_shape[0],
            recon_shape[1],
            embed_slice_stop - embed_slice_start,
        )
    if object_type == ObjectType.SHEPP_LOGAN:
        phantom_core = generate_3d_shepp_logan_low_dynamic_range(
            phantom_shape, target_max_attenuation=target_max_attenuation)
    elif object_type == ObjectType.CUBE:
        # gen_cube_phantom returns a tensor, as its mbirjax counterpart returns
        # a jax array.  This function promises host numpy for both object
        # types, so convert here rather than hand back two different things.
        phantom_core = gen_cube_phantom(phantom_shape).cpu().numpy()
    else:
        raise ValueError(f'Invalid object type. Expected one of {[o.value for o in ObjectType]}, got {object_type}')
    if model_type == ModelType.CONE and use_helical:
        # Embed the partial-slice phantom into the full recon volume.  For a helical scan only the
        # slices seen for at least half a rotation get phantom content; the rest stay zero.
        phantom = np.zeros(recon_shape, dtype=np.float32)
        phantom[:, :, embed_slice_start:embed_slice_stop] = phantom_core
    else:
        phantom = phantom_core

    # Forward project, keeping the sinogram in its device form, then gather it to a host array on a
    # separate line so the whole sinogram is never routed through a single device at large sizes.
    print('Creating sinogram')
    sinogram_sharded = ct_model_for_generation.forward_project(phantom, output_sharded=True)
    sinogram = ct_model_for_generation._gather_sinogram(sinogram_sharded)

    del ct_model_for_generation
    return phantom, sinogram, params
