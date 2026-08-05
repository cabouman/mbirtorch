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
# keeps a small residual warmup.
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
from .vcd_utils import (gen_weights, gen_full_indices, gen_pixel_partition,
                        gen_set_of_pixel_partitions, gen_partition_sequence,
                        get_2d_ror_mask)
from .qggmrf import (qggmrf_gradient_and_hessian_at_indices, get_b_from_nbr_wts,
                     b_tilde_by_definition)
from .utilities import generate_3d_shepp_logan_low_dynamic_range

__all__ = [
    "ParallelBeamModel", "ConeBeamModel", "TomographyModel", "QGGMRFDenoiser", "TorchProjector",
    "forward_project_differentiable", "back_project_differentiable",
    "gen_weights", "gen_full_indices", "gen_pixel_partition",
    "gen_set_of_pixel_partitions", "gen_partition_sequence", "get_2d_ror_mask",
    "qggmrf_gradient_and_hessian_at_indices", "get_b_from_nbr_wts",
    "b_tilde_by_definition", "generate_3d_shepp_logan_low_dynamic_range",
]
