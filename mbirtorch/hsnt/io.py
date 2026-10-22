import contextlib
import os
import warnings

import h5py
import numpy as np


# This dict lists the allowed metadata keys and describes the value of each one.
KEY_DESCRIPTIONS = {
    "dataset_name": "Character string with the name of the dataset.",
    "dataset_type": "'attenuation' or 'transmission'.",
    "dataset_modality": "'hyperspectral neutron'.",
    "wavelengths": "Array of wavelength values in Angstroms.",
    "alu_unit": "Character string defining geometry unit (e.g., 'mm' or 'cm').",
    "alu_value": "Float that represents the value of 1 ALU in the defined unit.",
    "delta_det_channel": "Detector channel spacing in ALU.",
    "delta_det_row": "Detector row spacing in ALU.",
    "dataset_geometry": "'parallel' or 'cone'.",
    "angles": "Array of view angles in degrees.",
    "det_channel_offset": "Assumed offset between center of rotation and center of detector in ALU.",
    "source_detector_dist": "Distance from source to detector in ALU.",
    "source_iso_dist": "Distance from source to iso in ALU."
}

# This dict lists the values that each of these keys accepts.
VALIDATION_RULES = {
    "dataset_type": (None, "attenuation", "transmission"),
    "dataset_modality": (None, "hyperspectral neutron"),
    "dataset_geometry": (None, "parallel", "cone"),
}

ALLOWED_KEYS = list(KEY_DESCRIPTIONS.keys())


def _validate_key(key, value):
    """This function warns when a key in ``VALIDATION_RULES`` has a value that is not allowed."""
    if key in VALIDATION_RULES and value not in VALIDATION_RULES[key]:
        valid_options = [v for v in VALIDATION_RULES[key] if v is not None]
        warnings.warn(f"Invalid '{key}': should be one of {valid_options}.")


def _with_key_docstring(style):
    """This function returns a decorator that replaces ``{_KEY_DOCS}`` in a docstring with the
    descriptions of the metadata keys."""
    indent = "\t- " if style == "dict" else "\t"
    text = "\n".join(f"{indent}{k}: {v}" for k, v in KEY_DESCRIPTIONS.items())

    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.replace("{_KEY_DOCS}", text)
        return func

    return decorator


def import_hsnt_data_hdf5(filename):
    """This function imports a hyperspectral dataset and its metadata from an HDF5 file.
    The spectral axis of hyperspectral data is always the last axis.

    The file can hold the data in hyperspectral form or in dehydrated form.  If the function
    cannot read the data, it warns and returns None for the data.

    Args:
        filename (str): The path to the HDF5 file.

    Returns:
        [data, metadata]: For hyperspectral data, data is an ndarray with the spectral axis last.
            For dehydrated data, data is the list [subspace_data, subspace_basis].
            metadata is a dict with the keys of ``create_hsnt_metadata``.
    """
    data = None
    metadata = {key: None for key in ALLOWED_KEYS}

    try:
        with h5py.File(filename, "r") as f:
            group = f

            dehydrated = "subspace_data" in group and "subspace_basis" in group

            if dehydrated:
                data = [group["subspace_data"][()], group["subspace_basis"][()]]
            elif "data" in group:
                data = group["data"][()]
            else:
                warnings.warn(f"No HSNT data found in HDF5 file '{filename}'. Returning data=None.")

            for key in ALLOWED_KEYS:
                if key == "dataset_type" and dehydrated:    # dehydrated data is always attenuation
                    continue
                if key in group:
                    value = group[key][()]
                    if isinstance(value, (bytes, np.bytes_)):
                        value = value.decode()
                    elif isinstance(value, np.ndarray) and value.shape == ():
                        value = value.item()
                    metadata[key] = value
    except Exception as error:
        warnings.warn(f"Could not import HSNT data from HDF5 file '{filename}': {error}. Returning data=None.")
        data = None

    for key, value in metadata.items():
        _validate_key(key, value)

    return [data, metadata]


@_with_key_docstring("arg")
def create_hsnt_metadata(**kwargs):
    """This function creates the metadata dict of a dataset.

    Each key that is not given has the value None.

    Args:
    {_KEY_DOCS}

    Returns:
        dict: The metadata.

    Example:
        >>> metadata = create_hsnt_metadata(dataset_name="sample1", dataset_type="transmission",
        ...     wavelengths=np.linspace(1.0, 5.0, 50), angles=np.linspace(0, 180, 10))
    """
    for key in kwargs.keys():
        if key not in ALLOWED_KEYS:
            warnings.warn(f"Ignoring invalid key '{key}' in arguments.")

    metadata = {k: kwargs.get(k, None) for k in ALLOWED_KEYS}

    for key, value in metadata.items():
        _validate_key(key, value)

    return metadata


def export_hsnt_data_hdf5(filename, data, metadata=None):
    """This function exports a hyperspectral dataset and its metadata to an HDF5 file.
    The spectral axis of hyperspectral data is always the last axis.

    Args:
        filename (str): The path to the HDF5 file.
        data (ndarray or list): For hyperspectral data, an ndarray with the spectral axis last.
            For dehydrated data, the list [subspace_data, subspace_basis].
        metadata (dict, optional): The dict from ``create_hsnt_metadata``.  Defaults to None.
    """
    if metadata is None:
        metadata = {}

    dehydrated = isinstance(data, list) and len(data) == 2

    for key, value in metadata.items():
        _validate_key(key, value)

    with h5py.File(filename, "w") as f:
        group = f

        if dehydrated:
            group.create_dataset("subspace_data", data=data[0])
            group.create_dataset("subspace_basis", data=data[1])
        else:
            group.create_dataset("data", data=data)

        for key, value in metadata.items():
            if key not in ALLOWED_KEYS:
                warnings.warn(f"Ignoring invalid key '{key}' in metadata.")
                continue
            if value is None or (key == "dataset_type" and dehydrated):
                continue
            if isinstance(value, str):
                group.create_dataset(key, data=np.bytes_(value))
            else:
                group.create_dataset(key, data=value)


@contextlib.contextmanager
def _written_atomically(path):
    """This function yields a temporary path in the directory of ``path``.

    When the ``with`` block completes, the function renames the temporary file to ``path``.  If
    the block fails, the function deletes the temporary file, so no partial file remains.
    """
    root, ext = os.path.splitext(path)
    tmp = f"{root}.partial{ext}"
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_metadata(group, metadata):
    """This function writes each hsnt metadata entry that the group does not already hold."""
    for key, value in metadata.items():
        if value is not None and key not in group:
            group.create_dataset(key, data=np.bytes_(value) if isinstance(value, str) else value)


def _decode(value):
    """This function converts an HDF5 scalar stored as bytes to a string.

    The function returns any other value unchanged.
    """
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else value


def _find_data_group(f, dataset=None):
    """This function finds the group that holds the 'data' dataset and returns (group, name).

    The function raises KeyError when no group holds 'data'.  It raises ValueError when several
    groups hold 'data' or when the file holds dehydrated data.
    """
    if dataset:
        if dataset not in f:
            groups = [k for k, v in f.items() if isinstance(v, h5py.Group)]
            raise KeyError(f"group {dataset!r} not found; groups at the root: {groups}")
        return f[dataset], dataset
    if "data" in f:
        return f, "/"
    if all(k in f for k in ("subspace_data", "subspace_basis")):
        raise ValueError("this file holds dehydrated data (subspace_data, subspace_basis), not hyperspectral data")
    candidates = [k for k, v in f.items() if isinstance(v, h5py.Group) and "data" in v]
    if len(candidates) == 1:
        return f[candidates[0]], candidates[0]
    if not candidates:
        raise KeyError(f"no 'data' dataset at the root or in a root group; root members: {list(f.keys())}")
    raise ValueError(f"several groups hold a 'data' dataset: {candidates}; choose one")


def _data_selection(shape, views=None, downsample=1):
    """This function returns an index into a 2D, 3D, or 4D 'data' dataset and the shape it selects.

    The index selects the requested views and every ``downsample``-th row and column.  The shape
    is (views, rows, cols).
    """
    ndim = len(shape)
    step = slice(None, None, downsample)
    if ndim == 2:
        return (slice(None),), (1, shape[0], 1)
    if ndim == 3:
        return (step, step), (1,) + np.empty(shape[:2], dtype=bool)[step, step].shape
    if ndim == 4:
        sel = (slice(*views) if views else slice(None), step, step)
        return sel, np.empty(shape[:3], dtype=bool)[sel].shape
    raise ValueError(f"data has {ndim} dimensions; expected (views, rows, cols, bins), (rows, cols, bins) or "
                     "(pixels, bins)")


def _create_hyperspectral(f, shape, dataset_type, chunks):
    """This function creates and returns an empty float32 'data' dataset in the hsnt file format.

    The dataset has shape (views, rows, cols, bins).  The caller fills it in blocks.
    """
    d = f.create_dataset("data", shape=shape, dtype=np.float32, chunks=chunks)
    f.create_dataset("dataset_type", data=np.bytes_(dataset_type))
    f.create_dataset("dataset_modality", data=np.bytes_("hyperspectral neutron"))
    return d
