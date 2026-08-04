"""mbirtorch: a PyTorch port of mbirjax (Phase 1: the parallel-beam vertical slice).

The public API mirrors mbirjax where implemented: numpy in, numpy out by
default, with device tensors available via ``output_sharded=True`` (the name
kept for API compatibility; here it means "return the device tensor").  The
port plan and parity gates live in mbirjax_plans/plans/torch_port/.
"""

__version__ = "0.0.1"

from .parallel_beam import ParallelBeamModel
from .tomography_model import TomographyModel
from .autograd import (TorchProjector, forward_project_differentiable,
                       back_project_differentiable)
from .vcd_utils import (gen_weights, gen_full_indices, gen_pixel_partition,
                        gen_set_of_pixel_partitions, gen_partition_sequence,
                        get_2d_ror_mask)
from .qggmrf import (qggmrf_gradient_and_hessian_at_indices, get_b_from_nbr_wts,
                     b_tilde_by_definition)

__all__ = [
    "ParallelBeamModel", "TomographyModel", "TorchProjector",
    "forward_project_differentiable", "back_project_differentiable",
    "gen_weights", "gen_full_indices", "gen_pixel_partition",
    "gen_set_of_pixel_partitions", "gen_partition_sequence", "get_2d_ror_mask",
    "qggmrf_gradient_and_hessian_at_indices", "get_b_from_nbr_wts",
    "b_tilde_by_definition",
]
