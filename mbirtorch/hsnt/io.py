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
