import warnings

import h5py
import numpy as np


# The allowed metadata keys and what each one holds.
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

# The values each of these keys accepts.
VALIDATION_RULES = {
    "dataset_type": (None, "attenuation", "transmission"),
    "dataset_modality": (None, "hyperspectral neutron"),
    "dataset_geometry": (None, "parallel", "cone"),
}

ALLOWED_KEYS = list(KEY_DESCRIPTIONS.keys())


def _validate_key(key, value):
    """Warn when a key listed in VALIDATION_RULES has a value that is not
    allowed."""
    if key in VALIDATION_RULES and value not in VALIDATION_RULES[key]:
        valid_options = [v for v in VALIDATION_RULES[key] if v is not None]
        warnings.warn(f"Invalid '{key}': should be one of {valid_options}.")


def _with_key_docstring(style):
    """Return a decorator that replaces ``{_KEY_DOCS}`` in a docstring with
    the key descriptions."""
    indent = "\t- " if style == "dict" else "\t"
    text = "\n".join(f"{indent}{k}: {v}" for k, v in KEY_DESCRIPTIONS.items())

    def decorator(func):
        if func.__doc__:
            func.__doc__ = func.__doc__.replace("{_KEY_DOCS}", text)
        return func

    return decorator


@_with_key_docstring("dict")
def import_hsnt_data_hdf5(filename):
    """
    Import a hyperspectral dataset and metadata from an HDF5 file.

    Args:
        filename: Path to the HDF5 file.

    Returns:
        A list containing hyperspectral data and parameters in the form [data, metadata].
            - data: ndarray with spectral last axis (hyperspectral form), a list (dehydrated form), or None.
            - metadata: A dictionary with the keys shown below.

    Keys:
    {_KEY_DOCS}
    """
    data = None
    metadata = {key: None for key in ALLOWED_KEYS}

    try:
        with h5py.File(filename, "r") as f:
            group = f

            dehydrated = all(k in group for k in ["subspace_data", "subspace_basis", "dataset_type"])

            if dehydrated:
                dataset_type = group["dataset_type"][()]
                if isinstance(dataset_type, (bytes, np.bytes_)):
                    dataset_type = dataset_type.decode()
                data = [group["subspace_data"][()],
                        group["subspace_basis"][()],
                        dataset_type]
            elif "data" in group:
                data = group["data"][()]
            else:
                warnings.warn(f"No HSNT data found in HDF5 file '{filename}'. Returning data=None.")

            for key in ALLOWED_KEYS:
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
    """
    Create a dictionary of parameters (metadata) associated with a hyperspectral neutron dataset.

    Args:
    {_KEY_DOCS}

    Returns:
        dict: Dictionary containing hyperspectral neutron dataset parameters (metadata).

    Example:
        >>> metadata = create_hsnt_metadata(
        ...     dataset_name="sample1",
        ...     dataset_type="attenuation",
        ...     dataset_modality="hyperspectral neutron",
        ...     wavelengths=np.linspace(1.0, 5.0, 50),
        ...     alu_unit="mm",
        ...     alu_value=1.0,
        ...     dataset_geometry="parallel",
        ...     angles=np.linspace(0, 180, 10)
        ... )
        >>> print(metadata["dataset_name"])
        sample1
    """
    for key in kwargs.keys():
        if key not in ALLOWED_KEYS:
            warnings.warn(f"Ignoring invalid key '{key}' in arguments.")

    metadata = {k: kwargs.get(k, None) for k in ALLOWED_KEYS}

    for key, value in metadata.items():
        _validate_key(key, value)

    return metadata


@_with_key_docstring("dict")
def export_hsnt_data_hdf5(filename, data, metadata=None):
    """
    Export a hyperspectral dataset and metadata to an HDF5 file.

    Args:
        filename: Path to the HDF5 file.
        data: ndarray with spectral last axis (hyperspectral form) or a list (dehydrated form).
        metadata: A dictionary with the keys shown below. Use create_hsnt_metadata to create a metadata dictionary.

    Keys:
    {_KEY_DOCS}

    Returns:
        None. Creates an HDF5 file with the corresponding structure.
    """
    if metadata is None:
        metadata = {}

    dehydrated = (isinstance(data, list)
                  and len(data) == 3
                  and isinstance(data[2], str)
                  and data[2] in VALIDATION_RULES["dataset_type"][1:])

    for key, value in metadata.items():
        _validate_key(key, value)

    with h5py.File(filename, "w") as f:
        group = f

        if dehydrated:
            group.create_dataset("subspace_data", data=data[0])
            group.create_dataset("subspace_basis", data=data[1])
            group.create_dataset("dataset_type", data=np.bytes_(data[2]))
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


def _decode(value):
    """A string from an HDF5 scalar that may be stored as bytes."""
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else value


def _find_data_group(f, dataset=None):
    """The group holding the hyperspectral 'data' dataset: the named group, else the root, else the only root group
    that has one. Returns (group, name); raises KeyError when there is none and ValueError when the file holds
    dehydrated data or the choice is ambiguous."""
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
    """The index over the spatial axes of a 2-, 3- or 4-D 'data' dataset (spectral axis last) that selects the views
    and every downsample-th row and column, and the (views, rows, cols) it gives. A 2-D dataset is (pixels, bins)."""
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
    """Create the 'data' dataset of the hsnt layout, float32 of shape (views, rows, cols, bins), with its
    dataset_type and dataset_modality entries. Returns the dataset, to be filled in blocks."""
    d = f.create_dataset("data", shape=shape, dtype=np.float32, chunks=chunks)
    f.create_dataset("dataset_type", data=np.bytes_(dataset_type))
    f.create_dataset("dataset_modality", data=np.bytes_("hyperspectral neutron"))
    return d
