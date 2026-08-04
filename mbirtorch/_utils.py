"""Parameter defaults and small shared helpers.

Ported from mbirjax._utils: the Param dataclass and the default parameter
dictionaries are copied verbatim (same names, values, and recompile flags) so
the two packages stay parameter-compatible.  The jax-specific OOM-marker and
ParamNames-literal tooling is not ported.
"""

import copy
from dataclasses import dataclass
from typing import Any

FILE_FORMAT_NUMBER = 1.0


@dataclass
class Param:
    val: Any
    recompile_flag: bool = True

    def __repr__(self):
        return f"Param(val={self.val}, recompile_flag={self.recompile_flag})"


# The values below are copied from mbirjax._utils (2026-08-04) and must track it.
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
