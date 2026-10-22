"""This module reads, writes, simulates, and compresses hyperspectral neutron data.

The function ``dehydrate`` factors the data into a few nonnegative spectra and their maps by
maximum likelihood.  The function ``rehydrate`` multiplies the maps by the spectra to rebuild the
data.  The function ``hyper_denoise`` denoises data by calling both, and ``estimate_rank``
estimates the number of spectra.  The functions ``import_hsnt_data_hdf5``,
``create_hsnt_metadata``, and ``export_hsnt_data_hdf5`` read and write the hsnt HDF5 format.  The
functions ``generate_hyper_sinogram``, ``gen_material_phantom``, ``synthetic_material_basis``, and
``generate_hyper_data`` simulate data.
"""
from .denoise import hyper_denoise, dehydrate, rehydrate
from .rank import estimate_rank
from .io import (KEY_DESCRIPTIONS, VALIDATION_RULES, ALLOWED_KEYS, _validate_key, _with_key_docstring,
                 import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5)
from .simulate import (generate_hyper_data, synthetic_material_basis, gen_material_phantom,
                       generate_hyper_sinogram)

__all__ = [
    "dehydrate", "rehydrate", "hyper_denoise", "estimate_rank",
    "import_hsnt_data_hdf5", "create_hsnt_metadata", "export_hsnt_data_hdf5",
    "generate_hyper_data", "synthetic_material_basis", "gen_material_phantom", "generate_hyper_sinogram",
]
