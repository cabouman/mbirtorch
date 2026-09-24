"""Hyperspectral neutron data: maximum-likelihood factorization, dehydration and the hsnt HDF5 format.

Modules:

    denoise         dehydrate, rehydrate, hyper_denoise: the factorization in the package's dehydrated form
    rank            estimate_rank, pool_pixels: the number of materials by likelihood-ratio tests
    factorization   nnal_factorization: the maximum-likelihood factorization of the transmission ratio
    _streaming      stream_factorization: the same fit over chunks of pixels, for data larger than device memory
    spectra         unconstrained_spectra, support_selected_spectra, select_supports, auto_penalty
    l2_baseline     l2_dehydrate, l2_hyper_denoise: the scikit-learn NMF dehydration, kept as a baseline
    io              the hsnt HDF5 format: import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5
    simulate        generate_hyper_data, generate_sphere_data, load_material_basis, material_basis_wavelengths
    plots           compare_spectra
    cli             the mbirtorch-hsnt command line
    _loss, _linalg, _newton, _multiplicative, _lbfgsb, _device   solver internals

matplotlib and scikit-learn are imported only by the functions that use them.
"""
from ._loss import stable_nnal
from ._streaming import stream_factorization
from .factorization import nnal_factorization
from .spectra import unconstrained_spectra, support_selected_spectra, select_supports, auto_penalty
from .denoise import hyper_denoise, dehydrate, rehydrate
from .rank import estimate_rank, pool_pixels
from .l2_baseline import l2_hyper_denoise, l2_dehydrate, _estimate_subspace_dimension
from .io import (KEY_DESCRIPTIONS, VALIDATION_RULES, ALLOWED_KEYS, _validate_key, _with_key_docstring,
                 import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5)
from .simulate import generate_hyper_data, generate_sphere_data, load_material_basis, material_basis_wavelengths
from .plots import compare_spectra

__all__ = [
    "dehydrate", "rehydrate", "hyper_denoise", "estimate_rank", "pool_pixels",
    "nnal_factorization", "stream_factorization", "stable_nnal",
    "unconstrained_spectra", "support_selected_spectra", "select_supports", "auto_penalty",
    "l2_dehydrate", "l2_hyper_denoise",
    "import_hsnt_data_hdf5", "create_hsnt_metadata", "export_hsnt_data_hdf5",
    "generate_hyper_data", "generate_sphere_data", "load_material_basis", "material_basis_wavelengths",
    "compare_spectra",
]
