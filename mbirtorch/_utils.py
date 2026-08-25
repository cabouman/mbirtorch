"""Parameter defaults and small shared helpers.

Defines the Param dataclass and the default parameter dictionaries.  The
names, values, and recompile flags of those defaults are fixed by an external
reference and must not be changed here.
"""

import copy
from dataclasses import dataclass
from typing import Any

FILE_FORMAT_NUMBER = 1.0

#: The multiple a hand-written kernel's width argument is rounded up to.
#: Triton compiles a separate, faster kernel for each integer argument it can
#: prove is a multiple of 16.  Two kernels have been measured against their
#: unspecialized compilation, and they cost different amounts and for
#: different reasons.  The cone back kernel used more registers and ran at
#: roughly half the rate.  The multiaxis forward kernel ran at about a third
#: of the rate with the SAME 32 registers and no spills, and its unspecialized
#: compilation was 4 percent more PTX and 7 percent more cubin; what the
#: specialization buys there was not isolated further.
KERNEL_WIDTH_MULTIPLE = 16


def padded_kernel_width(width):
    """``width`` rounded up to the next multiple of 16.

    A width that is already a multiple of 16 is returned unchanged, so a
    caller can compare the result against its input and take its original
    path when the two agree.

    This is the ONE definition of the rule.  The Triton kernel wrappers call
    it to size the launch and the arrays they allocate, and the memory ledger
    calls it to charge those same arrays, so the code and the charge cannot
    disagree.

    The rule covers every width-class argument a kernel receives, including
    the bound it masks its vector axis against, not only the arrays it
    allocates.  The multiaxis forward wrapper padded its allocation and its
    stride but passed the real detector row count as that bound, and it cost a
    factor of 3.1 at every row count that was not a multiple of 16 (measured
    2026-08-24; the record is multigpu_findings.md section 1.51 in the plans
    repository).

    Args:
        width (int): a non-negative length -- a slice band, a detector row
            count, or a value-column count.

    Returns:
        int: the padded length.
    """
    width = int(width)
    remainder = width % KERNEL_WIDTH_MULTIPLE
    if remainder == 0:
        return width
    return width + KERNEL_WIDTH_MULTIPLE - remainder


@dataclass
class Param:
    val: Any
    recompile_flag: bool = True

    def __repr__(self):
        return f"Param(val={self.val}, recompile_flag={self.recompile_flag})"


# The names, values, and recompile flags below are fixed; do not change them here.
_forward_model_defaults_dict = {
    'geometry_type': Param(None, False),
    'file_format': Param(FILE_FORMAT_NUMBER, False),
    'sinogram_shape': Param(None, True),
    'delta_det_channel': Param(1.0, True),
    'delta_det_row': Param(1.0, True),
    'det_row_offset': Param(0.0, True),
    'det_channel_offset': Param(0.0, True),
    'sigma_y': Param(1.0, False),
    'alu_unit': Param(None, False),
    'alu_value': Param(1.0, False),
}

_recon_model_defaults_dict = {
    'recon_shape': Param(None, True),
    'delta_voxel': Param(None, True),
    'voxel_row_aspect': Param(1.0, True),
    'voxel_slice_aspect': Param(1.0, True),
    'sigma_x': Param(1.0, False),
    'sigma_prox': Param(1.0, False),
    'p': Param(2.0, False),
    'q': Param(1.2, False),
    'T': Param(1.0, False),
    'qggmrf_nbr_wts': Param([1.0, 1.0, 1.0], False),  # row_nbr_wt, col_nbr_wt, slice_nbr_wt
}

_reconstruction_defaults_dict = {
    'auto_regularize_flag': Param(True, False),
    'positivity_flag': Param(False, False),
    'snr_db': Param(30.0, False),
    'sharpness': Param(1.0, False),
    # 4 independent 128-subset partitions, cycled after warmup (covers 103
    # iterations; last entry repeats after that).
    'granularity': Param([1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128], False),
    'partition_sequence': Param([2, 4, 6] + [7, 8, 9, 10] * 25, False),
    'verbose': Param(1, False),
    'max_alpha': Param(1.5, False),
    'use_ror_mask': Param(True, False),
}

dicts = [_forward_model_defaults_dict,
         _recon_model_defaults_dict,
         _reconstruction_defaults_dict]

recon_defaults_dict = {}
for _d in dicts:
    recon_defaults_dict = {**recon_defaults_dict, **_d}

all_param_keys = list(recon_defaults_dict.keys())

recon_param_names = ['num_iterations', 'granularity', 'partition_sequence', 'fm_rmse',
                     'prior_loss', 'regularization_params', 'stop_threshold_change_pct',
                     'alpha_values', 'delta_norm_per_slice']

_AUTO_REGULARIZATION_PARAM_NAMES = ('sigma_y', 'sigma_x', 'sigma_prox')


def get_default_params():
    return copy.deepcopy(recon_defaults_dict)
