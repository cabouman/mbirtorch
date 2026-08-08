"""mbirtorch: a PyTorch port of mbirjax (parallel-beam geometry).

The public API mirrors mbirjax where implemented: numpy in, numpy out by
default, with device tensors available via ``output_sharded=True`` (the name
kept for API compatibility; here it means "return the device tensor").
"""

__version__ = "0.0.1"

# ── persistent torch.compile cache ────────────────────────────────────────────
# The inductor cache directory defaults to /tmp/torchinductor_<user>, which the
# OS may clean; pin it to a stable per-user location so compiled artifacts
# survive across processes and reboots (the mbirjax ~/.mbirjax/jax_cache
# analog).  The FX-graph cache is what makes a FRESH PROCESS reuse prior
# compilations; enable it explicitly for torch versions where it is not the
# default.  setdefault keeps both overridable per-run via the environment, and
# -- like mbirjax's TF_CPP_MIN_LOG_LEVEL -- this is effective only if mbirtorch
# is imported before torch triggers its first compile, which any
# import-mbirtorch-first program satisfies.  Dynamo TRACING still runs per
# process (the cache skips inductor codegen, not tracing), so a cold process
# keeps a small residual warmup.  ``mbirtorch.clear_cache()`` removes the
# whole ~/.mbirtorch directory (see utilities.py).
import os as _os

_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                       _os.path.expanduser("~/.mbirtorch/torch_cache"))
_os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

from .parallel_beam import ParallelBeamModel
from .cone_beam import ConeBeamModel
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
                        build_model, download_and_extract)
from .memory_stats import get_memory_stats

# __all__ is the DECLARED public surface, and autodoc honors it: every name here is
# documented by ``automodule:: mbirtorch :members:``.  It is deliberately narrower than
# the import list above -- the VCD and qGGMRF helpers stay importable as attributes
# (mbirtorch.gen_full_indices still works, and the tests use that spelling) but are not
# promised as public API, matching the surface mbirjax documents.
__all__ = [
    "ParallelBeamModel", "ConeBeamModel", "TomographyModel", "QGGMRFDenoiser",
    "TorchProjector", "forward_project_differentiable",
    "back_project_differentiable", "gen_weights", "gen_weights_mar",
    "median_filter3d", "download_and_extract", "build_model",
    "save_data_hdf5", "load_data_hdf5", "export_recon_hdf5",
    "import_recon_hdf5",
    "generate_3d_shepp_logan_low_dynamic_range", "clear_cache",
    "get_memory_stats", "SliceViewer", "VolumeStack", "slice_viewer",
]

# ── lazy exports (PEP 562) ───────────────────────────────────────────────────
# The viewer names resolve on first attribute access so that a headless
# `import mbirtorch` never imports matplotlib; most mbirtorch runs (batch
# recons, tests) never open a viewer.  The preprocess subpackage resolves the
# same way, so `import mbirtorch` never pays for its dependency stack (osqp
# pulls scipy.sparse, plus cv2, tifffile, and the loaders -- about a third of
# the package's cold import before this).  `mbirtorch.preprocess` and the
# direct form `import mbirtorch.preprocess` both keep working; only WHEN the
# subpackage loads changes.  mbirjax imports its preprocess eagerly, so this
# is a deliberate, behavior-preserving improvement over the mirror.
_VIEWER_EXPORTS = ("SliceViewer", "VolumeStack", "slice_viewer")


def __getattr__(name):
    if name in _VIEWER_EXPORTS:
        from . import view_utils
        value = getattr(view_utils, name)
        globals()[name] = value  # cache: later accesses skip this hook
        return value
    if name == 'preprocess':
        import importlib
        value = importlib.import_module('.preprocess', __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
