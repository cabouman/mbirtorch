"""mbirtorch: model-based iterative reconstruction for computed tomography, in PyTorch.

The package reconstructs volumes from tomographic measurements for several
scanner geometries.  The public API takes numpy arrays and returns numpy
arrays by default; pass ``output_sharded=True`` to get the device tensor
instead.  All available GPUs are used automatically.
"""

__version__ = "0.1.1"

# The torch.compile caches are pinned under ~/.mbirtorch, so compiled code survives a process.
# These settings take effect only if mbirtorch is imported before torch compiles anything.
import os as _os

_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                       _os.path.expanduser("~/.mbirtorch/torch_cache"))
_os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
# Triton caches its kernels separately from inductor, so that cache is pinned too.
_os.environ.setdefault("TRITON_CACHE_DIR",
                       _os.path.expanduser("~/.mbirtorch/triton_cache"))

from typing import TYPE_CHECKING

from .parallel_beam import ParallelBeamModel, recon_simple_parallel
from .cone_beam import ConeBeamModel, recon_simple_cone
from .translation_model import TranslationModel
from .multiaxis_parallel import MultiAxisParallelModel, MultiAxisParallelBeamModel
from .denoising import QGGMRFDenoiser
from .tomography_model import TomographyModel
from .autograd import (TorchProjector, forward_project_differentiable,
                       back_project_differentiable)
from .vcd_utils import (gen_weights, gen_weights_mar, gen_full_indices,
                        gen_pixel_partition, gen_set_of_pixel_partitions,
                        gen_partition_sequence, get_2d_ror_mask)
from .denoising import median_filter3d
from .qggmrf import (qggmrf_gradient_and_hessian_at_indices, get_b_from_nbr_wts,
                     b_tilde_by_definition, qggmrf_loss)
from .utilities import (generate_3d_shepp_logan_low_dynamic_range, clear_cache,
                        makedirs, load_data_hdf5, save_data_hdf5,
                        export_recon_hdf5, import_recon_hdf5,
                        build_model, download_and_extract,
                        copy_ct_model, stitch_arrays, save_volume_as_gif,
                        get_ct_model, generate_demo_data,
                        generate_3d_shepp_logan_reference, gen_cube_phantom,
                        gen_translation_vectors, gen_translation_phantom,
                        get_helical_half_rotation_slice_range,
                        merge_log_files)
from .memory_stats import get_memory_stats

# __all__ is the declared public surface, and autodoc documents exactly these names.
# The other imported names remain importable, but they are not promised as public API.
__all__ = [
    "ParallelBeamModel", "ConeBeamModel", "TranslationModel",
    "MultiAxisParallelModel", "TomographyModel", "QGGMRFDenoiser",
    "recon_simple_parallel", "recon_simple_cone",
    "TorchProjector", "forward_project_differentiable",
    "back_project_differentiable", "gen_weights", "gen_weights_mar",
    "median_filter3d", "download_and_extract", "build_model",
    "save_data_hdf5", "load_data_hdf5", "export_recon_hdf5",
    "import_recon_hdf5",
    "generate_3d_shepp_logan_low_dynamic_range", "clear_cache",
    "get_memory_stats", "SliceViewer", "VolumeStack", "slice_viewer",
    "GeometryScene", "GeometryFigure", "geometry_viewer",
    "stitch_arrays", "get_ct_model", "copy_ct_model", "save_volume_as_gif",
    "MACE4DModel", "temporal_filter_matrix", "apply_temporal_filter",
    "generate_demo_data", "generate_3d_shepp_logan_reference",
    # These hsnt and vcls names resolve lazily through __getattr__.
    "hyper_denoise", "dehydrate", "rehydrate", "estimate_rank", "l2_dehydrate", "l2_hyper_denoise", "import_hsnt_data_hdf5",
    "create_hsnt_metadata", "export_hsnt_data_hdf5", "generate_hyper_data",
    "get_opt_views", "show_image_with_projection_rays",
]

# Lazy exports (PEP 562) resolve on first attribute access, so importing mbirtorch does
# not pull in matplotlib or the preprocess, hsnt and vcls dependencies.
_VIEWER_EXPORTS = ("SliceViewer", "VolumeStack", "slice_viewer",
                   "GeometryScene", "GeometryFigure", "geometry_viewer")

_LAZY_MODULES = ("preprocess", "hsnt", "vcls", "mace")

# Each package level name is mapped to the module that defines it.  A new
# public function in one of those modules needs a line here.
_LAZY_NAMES = {
    'hyper_denoise': 'hsnt', 'dehydrate': 'hsnt', 'rehydrate': 'hsnt', 'estimate_rank': 'hsnt',
    'l2_dehydrate': 'hsnt', 'l2_hyper_denoise': 'hsnt',
    'import_hsnt_data_hdf5': 'hsnt', 'create_hsnt_metadata': 'hsnt',
    'export_hsnt_data_hdf5': 'hsnt', 'generate_hyper_data': 'hsnt',
    'subsample_R_gamma': 'vcls', 'max_abs_neighbor_diff': 'vcls',
    'get_opt_views': 'vcls', 'compute_view_basis_functions': 'vcls',
    'compute_cov_matrix': 'vcls', 'compute_vcl': 'vcls',
    'compute_opt_angle_subset': 'vcls', 'get_2d_subsampling_indices': 'vcls',
    'show_image_with_projection_rays': 'vcls', 'reorder_by_priority': 'vcls',
    # The blue noise pattern is a 382 KB array literal, loaded on first use.
    'bn256': 'bn256',
    # The one call function mace() is reached as mbirtorch.mace.mace.  It is
    # not exported at package level.
    'MACE': 'mace', 'Task': 'mace', 'ForwardProxAgent': 'mace',
    'QGGMRFDenoiserAgent': 'mace', 'HyperplaneAgent': 'mace',
    'resolve_device_pool': 'mace',
    'MACE4DModel': 'mace4d',
    'temporal_filter_matrix': 'mace4d', 'apply_temporal_filter': 'mace4d',
}

# Editors and type checkers do not call __getattr__, so the lazy names are listed below as
# imports that never execute.  tests/test_lazy_exports.py checks them against the tables above.
if TYPE_CHECKING:
    from . import preprocess, hsnt, vcls, mace
    from .view_utils import (SliceViewer, VolumeStack, slice_viewer,
                             GeometryScene, GeometryFigure, geometry_viewer)
    from .mace import (MACE, Task, ForwardProxAgent, QGGMRFDenoiserAgent, HyperplaneAgent,
                       resolve_device_pool)
    from .mace4d import MACE4DModel, temporal_filter_matrix, apply_temporal_filter
    from .hsnt import (hyper_denoise, dehydrate, rehydrate, estimate_rank,
                       l2_dehydrate, l2_hyper_denoise,
                       import_hsnt_data_hdf5, create_hsnt_metadata,
                       export_hsnt_data_hdf5, generate_hyper_data)
    from .vcls import (subsample_R_gamma, max_abs_neighbor_diff, get_opt_views,
                       compute_view_basis_functions, compute_cov_matrix,
                       compute_vcl, compute_opt_angle_subset,
                       get_2d_subsampling_indices,
                       show_image_with_projection_rays, reorder_by_priority)
    from .bn256 import bn256


def __getattr__(name):
    import importlib
    if name in _VIEWER_EXPORTS:
        from . import view_utils
        value = getattr(view_utils, name)
        globals()[name] = value  # Later accesses skip this hook.
        return value
    if name in _LAZY_MODULES:
        value = importlib.import_module('.' + name, __name__)
        globals()[name] = value
        return value
    if name in _LAZY_NAMES:
        module = importlib.import_module('.' + _LAZY_NAMES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
