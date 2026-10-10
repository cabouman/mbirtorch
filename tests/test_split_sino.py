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
    phantom = mbirtorch.gen_shepp_logan_3d(rshape)
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
    full, full_dict = model.recon(sino, weights=weights, max_iterations=8)
    np.random.seed(0)
    split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=4,
                                               max_iterations=8)
    assert split.shape == full.shape
    nrmse = float(np.linalg.norm(split - full) / np.linalg.norm(full))
    print(f"split vs full NRMSE = {nrmse:.4f}")
    assert nrmse < 0.1
    # The halves take their regularization from the whole sinogram and its weights, so it
    # equals what recon set from the same inputs.
    expected = full_dict['recon_params']['regularization_params']
    for key in ('recon_params_top', 'recon_params_bottom'):
        assert split_dict[key]['regularization_params'] == pytest.approx(expected)
    for key in ('model_params_top', 'model_params_bottom'):
        assert float(split_dict[key]['delta_voxel']) == pytest.approx(pitch)
    sp = split_dict['split_params']
    assert sp['half_overlap_sino'] >= 4 and sp['half_overlap_recon'] > sp['half_overlap_sino'] // 2
    assert 'recon_params_top' in split_dict and 'recon_params_bottom' in split_dict


def _small_parallel_case():
    cell = (48, 24, 32)  # views, detector rows (which are slices), channels
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(cell, angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    rshape = tuple(model.get_params('recon_shape'))
    phantom = mbirtorch.gen_shepp_logan_3d(rshape)
    sino = model.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / sino.max(), weight_type='transmission_root')
    return model, sino, weights


def test_parallel_split_matches_recon_regularization():
    """A parallel-beam split with weights sets the regularization recon sets on the same
    inputs, and its reconstruction approximates recon's."""
    model, sino, weights = _small_parallel_case()
    np.random.seed(0)
    full, full_dict = model.recon(sino, weights=weights, max_iterations=8)
    np.random.seed(0)
    split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=4,
                                               slices_per_section=10, max_iterations=8)
    assert split.shape == full.shape
    assert split_dict['split_params']['num_sections'] == 3
    expected = full_dict['recon_params']['regularization_params']
    for part in split_dict['recon_params_sections']:
        assert part['regularization_params'] == pytest.approx(expected)
    nrmse = float(np.linalg.norm(split - full) / np.linalg.norm(full))
    print(f"parallel split vs full NRMSE = {nrmse:.4f}")
    assert nrmse < 0.1


# ── the groups of GPUs that reconstruct sections side by side ─────────────────

def test_group_rule_divides_devices_into_equal_groups():
    """The devices are divided into equal groups, as many as possible, so that each group can hold a section."""
    choose = mbirtorch.TomographyModel._choose_gpus_per_group
    needs_three = lambda group: len(group) >= 3
    assert choose(list(range(5)), needs_three) == 5          # 5 divides only as 5 x 1 or 1 x 5
    assert choose(list(range(8)), needs_three) == 4          # 8 gives 2 groups of 4
    assert choose(list(range(8)), lambda g: len(g) >= 1) == 1
    assert choose(list(range(8)), lambda g: False) == 8      # nothing fits: all devices, one group
    # Cone beam: at most 2 groups, so 8 devices give 2 groups of 4 when a half fits on 4.
    assert choose(list(range(8)), lambda g: len(g) >= 2, max_groups=2) == 4
    assert choose(list(range(8)), lambda g: len(g) >= 5, max_groups=2) == 8


def test_sections_side_by_side_match_sections_run_alone():
    """Two sections reconstructed in threads give exactly the result of each reconstructed alone."""
    model, sino, weights = _small_parallel_case()
    num_rows = sino.shape[1]
    ranges = [(0, num_rows // 2 + 4), (num_rows // 2 - 4, num_rows)]

    def job(lo, hi):
        def run(group, rng):
            section = mbirtorch.utilities.copy_ct_model(model, new_num_det_rows=hi - lo, no_warning=True)
            section.set_params(no_warning=True, auto_regularize_flag=False,
                               recon_shape=(model.get_params('recon_shape')[0],
                                            model.get_params('recon_shape')[1], hi - lo))
            if group is not None:
                section.configure_devices(devices=group)
            recon, _ = section.recon(sino[:, lo:hi, :], weights=weights[:, lo:hi, :], max_iterations=4,
                                     print_logs=False, logfile_path=None, rng=rng)
            return np.asarray(recon)
        return run

    jobs = [job(lo, hi) for lo, hi in ranges]
    # The runner seeds each section's generator from the global state, so the same seeds come
    # from the same global seed.
    np.random.seed(0)
    seeds = np.random.randint(0, 2 ** 31 - 1, size=len(jobs))
    alone = [j(None, np.random.default_rng(int(seed))) for j, seed in zip(jobs, seeds)]
    # Two groups of one CPU device each run the two sections side by side.
    np.random.seed(0)
    together = mbirtorch.TomographyModel._run_split_sections(jobs, ['cpu', 'cpu'], 1)
    for a, t in zip(alone, together):
        assert np.array_equal(a, t)


def test_split_result_reports_the_layout_and_uses_section_names():
    model, sino, weights = _small_parallel_case()
    np.random.seed(0)
    with pytest.warns(DeprecationWarning, match='slices_per_part'):
        split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=4,
                                                   slices_per_part=10, max_iterations=2,
                                                   print_logs=False, logfile_path=None)
    params = split_dict['split_params']
    assert params['num_sections'] == 3
    assert params['slices_per_section'] == -(-sino.shape[1] // 3)   # 3 sections of 8 kept rows
    assert len(params['section_slice_ranges']) == 3
    assert params['num_groups'] == 1 and params['gpus_per_group'] == 1    # one CPU device
    assert set(split_dict) == {'recon_params_sections', 'recon_log_sections', 'notes_sections',
                               'model_params_sections', 'split_params'}


def test_cone_split_reports_the_layout():
    model, sino, weights = _small_cone_case()
    np.random.seed(0)
    split, split_dict = model.recon_split_sino(sino, weights=weights, max_iterations=2,
                                               print_logs=False, logfile_path=None)
    params = split_dict['split_params']
    assert params['num_groups'] == 1 and params['gpus_per_group'] == 1    # one CPU device
    assert split.shape == tuple(model.get_params('recon_shape'))
