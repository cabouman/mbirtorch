"""Hyperspectral neutron data: maximum-likelihood dehydration and the hsnt HDF5 format.

Modules:

    denoise         dehydrate, rehydrate, hyper_denoise: the maximum-likelihood factorization in the package's
                    dehydrated form, with the spectra estimators and streaming behind dehydrate
    rank            estimate_rank: the number of components by likelihood-ratio tests
    io              the hsnt HDF5 format: import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5
    simulate        generate_hyper_data, load_material_basis
    cli             the mbirtorch-hsnt command line (with loading, outputs and plots)
    _fit, factorization, spectra, _streaming, _newton, _linalg, _loss, _device   the solver internals

matplotlib is imported only by the functions that use it.
"""
from .denoise import hyper_denoise, dehydrate, rehydrate
from .rank import estimate_rank
from .io import (KEY_DESCRIPTIONS, VALIDATION_RULES, ALLOWED_KEYS, _validate_key, _with_key_docstring,
                 import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5)
from .simulate import generate_hyper_data, load_material_basis

__all__ = [
    "dehydrate", "rehydrate", "hyper_denoise", "estimate_rank",
    "import_hsnt_data_hdf5", "create_hsnt_metadata", "export_hsnt_data_hdf5",
    "generate_hyper_data", "load_material_basis",
]
