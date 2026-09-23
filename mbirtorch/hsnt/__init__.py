"""Hyperspectral neutron tomography: NNAL factorization, spectral denoising and the hsnt HDF5 format.

Submodules, one line each:

    _loss               the non-negative attenuation loss (stable_nnal), its derivatives and its per-row sums
    _linalg             randomized SVD, the NNDSVDa initialization and the small batched SPD / Cholesky solves
    _multiplicative     the shifted multiplicative, quadratic and scalar-step Newton updates
    _newton             projected block-Newton and joint Newton-PCG solvers, the compiled kernels and their tuning constants
    _streaming          stream_factorization: the out-of-core factorization over chunks
    factorization       the `optimize` driver and the `nnal_factorization` front end
    spectra             unconstrained_spectra and support_selected_spectra, the re-estimates that remove the truncation bias
    denoise             dehydrate / rehydrate / hyper_denoise: the NNAL factorization as the package's dehydrated form
    rank                estimate_rank: the number of materials by likelihood-ratio tests (full resolution and pooled pixels)
    l2_baseline         mothballed scikit-learn L2 dehydration (l2_dehydrate, l2_hyper_denoise), a baseline for comparison plots
    io                  the hsnt HDF5 format: import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5 (h5py)
    simulate            generate_hyper_data, the Ni/Cu/Al phantom
    plots               compare_spectra

Every name of every submodule is re-exported here, underscore-prefixed ones
included, so ``from mbirtorch.hsnt import X`` works for any X (the package began
as one module whose private helpers callers import directly); ``__all__`` lists
the public ones.

The tuning constants ``_ARMIJO_FLOOR``, ``_TRUST_FLOOR`` and ``_ACTIVE_TOL``
are owned by ``_newton`` and every solver reads them through that module at
call time. The copies bound here are plain floats and are for reading only:
to monkeypatch one, target the owner, ``mbirtorch.hsnt._newton._ARMIJO_FLOOR``,
not ``mbirtorch.hsnt._ARMIJO_FLOOR``. (``_COMPILED_KERNELS`` is a dict, so the
name here is the same object as the owner's.)

matplotlib is imported lazily, inside the three functions that plot
(compare_spectra, and the verbose plotting blocks of generate_hyper_data and
l2_baseline._estimate_subspace_dimension), so importing this package does not import it.
"""
from ._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives, _nnal_rowwise
from ._linalg import _randomized_svd, nndsvda, _batched_spd_solve, _joint_blocks, _joint_dot
from ._multiplicative import _shifted, _rebalance, _reseed_dead, multiplicative_update
from ._newton import (_COMPILED_KERNELS, _ARMIJO_FLOOR, _TRUST_FLOOR, _ACTIVE_TOL, _kernels, _two_metric_direction, solve_W, block_newton_step,
                      block_newton_optimize, _joint_newton_pcg, joint_newton_optimize)
from ._streaming import _h_stats_accumulate, _h_direction, stream_factorization
from .factorization import optimize, nnal_factorization
from ._lbfgsb import lbfgsb_optimize
from .spectra import unconstrained_spectra, support_selected_spectra, select_supports, auto_penalty
from .denoise import hyper_denoise, dehydrate, rehydrate
from .rank import estimate_rank, pool_pixels
from .l2_baseline import l2_hyper_denoise, l2_dehydrate, _estimate_subspace_dimension
from .io import (KEY_DESCRIPTIONS, VALIDATION_RULES, ALLOWED_KEYS, _validate_key, _with_key_docstring,
                 import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5)
from .simulate import generate_hyper_data, generate_sphere_data
from .plots import compare_spectra

__all__ = [
    "hyper_denoise", "dehydrate", "rehydrate", "estimate_rank", "l2_dehydrate", "l2_hyper_denoise",
    "import_hsnt_data_hdf5", "create_hsnt_metadata", "export_hsnt_data_hdf5",
    "generate_hyper_data", "generate_sphere_data",
    "nnal_factorization", "stable_nnal", "stable_nnal_derivatives",
    "compare_spectra",
    "stream_factorization",
    "unconstrained_spectra", "support_selected_spectra", "select_supports", "auto_penalty",
    "nndsvda", "optimize", "block_newton_optimize", "joint_newton_optimize", "lbfgsb_optimize", "block_newton_step",
]
