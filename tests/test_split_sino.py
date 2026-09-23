"""Gates for recon_split_sino, in both geometries that implement it.

The split reconstruction must approximately equal recon() on the same inputs.
"""

import numpy as np
import pytest

import mbirtorch


def _set_hand_pitch(model, delta_voxel_scale):
    """Scale the model's voxel pitch as a user setting it by hand would: the recon shape stays the
    automatic one, so the volume covers a different physical extent than the detector's field of
    view, and a copy of the model must keep that pitch to describe the same volume."""
    if delta_voxel_scale != 1.0:
        model.set_params(no_warning=True,
                         delta_voxel=delta_voxel_scale * float(model.get_params('delta_voxel')))


def _small_cone_case(delta_voxel_scale=1.0):
    cell = (32, 32, 32)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                    source_iso_dist=2 * cell[2])
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    _set_hand_pitch(model, delta_voxel_scale)
    rshape = tuple(model.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rshape)
    sino = model.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / sino.max(), weight_type='transmission_root')
    return model, sino, weights


def test_split_approximates_full_recon():
    """The split reconstruction approximates recon() on the same inputs, and a parent whose voxel
    pitch was set by hand hands that pitch to both halves, so the split reconstructs the same
    physical volume."""
    model, sino, weights = _small_cone_case(delta_voxel_scale=0.8)
    pitch = float(model.get_params('delta_voxel'))
    np.random.seed(0)
    full, _ = model.recon(sino, weights=weights, max_iterations=8)
    np.random.seed(0)
    split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=4,
                                               max_iterations=8)
    assert split.shape == full.shape
    nrmse = float(np.linalg.norm(split - full) / np.linalg.norm(full))
    print(f"split vs full NRMSE = {nrmse:.4f}")
    assert nrmse < 0.1
    for key in ('model_params_top', 'model_params_bottom'):
        assert float(split_dict[key]['delta_voxel']) == pytest.approx(pitch)
    sp = split_dict['split_params']
    assert sp['half_overlap_sino'] >= 4 and sp['half_overlap_recon'] > sp['half_overlap_sino'] // 2
    assert 'recon_params_top' in split_dict and 'recon_params_bottom' in split_dict
