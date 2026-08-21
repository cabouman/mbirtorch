"""Gates for recon_split_sino, in both geometries that implement it.

Contract tests: the split reconstruction approximately equals recon() on the
same inputs; a 2-device layout on the parent carries to the halves and
produces the same result as the single-device split; a split too close to the
volume boundary falls back to a standard recon with a warning.  Cross-framework
parity against the mbirjax golden is measured loosely (iterative recons in the
loop, as for the MAR end-to-end gate); the geometry-derived overlaps must
match mbirjax exactly.

The parallel beam tests below cover the same contract for a split into any
number of parts, plus what only that geometry has: an explicit part size, the
memory estimate that picks the part count when none is given, and the parts'
slice ranges tiling the volume.
"""

import os
import warnings

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _memory_ledger, _sharding

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")


def _small_cone_case():
    cell = (32, 32, 32)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                    source_iso_dist=2 * cell[2])
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    rshape = tuple(model.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rshape)
    sino = model.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / sino.max(), weight_type='transmission_root')
    return model, sino, weights


def test_split_approximates_full_recon():
    model, sino, weights = _small_cone_case()
    np.random.seed(0)
    full, _ = model.recon(sino, weights=weights, max_iterations=8)
    np.random.seed(0)
    split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=4,
                                               max_iterations=8)
    assert split.shape == full.shape
    nrmse = float(np.linalg.norm(split - full) / np.linalg.norm(full))
    print(f"split vs full NRMSE = {nrmse:.4f}")
    assert nrmse < 0.1
    sp = split_dict['split_params']
    assert sp['half_overlap_sino'] >= 4 and sp['half_overlap_recon'] > sp['half_overlap_sino'] // 2
    assert 'recon_params_top' in split_dict and 'recon_params_bottom' in split_dict


def test_split_preserves_device_layout():
    # A 2-device parent layout must carry to the halves and reproduce the single-device split.
    model, sino, weights = _small_cone_case()
    np.random.seed(0)
    split1, _ = model.recon_split_sino(sino, weights=weights, half_overlap=4, max_iterations=5)

    model2, _, _ = _small_cone_case()
    model2.configure_devices(devices=['cpu', 'cpu'])
    np.random.seed(0)
    split2, _ = model2.recon_split_sino(sino, weights=weights, half_overlap=4, max_iterations=5)
    err = float(np.max(np.abs(split1 - split2)) / max(np.max(np.abs(split1)), 1e-30))
    print(f"sharded vs single split rel_max = {err:.2e}")
    # Multi-device runs carry the documented compiled-variant float envelope (see the multi-GPU
    # docs): changing the device count changes values slightly.  Measured ~6e-4 here.
    assert err < 5e-3
    # The parent layout itself is untouched by the split.
    assert len(model2.sino_placement.devices) == 2


def test_split_fallback_warns_and_recons():
    tiny = mbirtorch.ConeBeamModel((8, 6, 12), np.linspace(0, 2 * np.pi, 8, endpoint=False),
                                   source_detector_dist=48.0, source_iso_dist=24.0)
    tiny.configure_devices(devices=['cpu'])
    tiny.set_params(no_warning=True, verbose=0)
    tsino = np.ones((8, 6, 12), dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        recon, _ = tiny.recon_split_sino(tsino, max_iterations=2)
    assert any('falling back' in str(x.message) for x in w)
    assert recon.shape == tuple(tiny.get_params('recon_shape'))


def test_split_rejects_bad_inputs():
    model, sino, weights = _small_cone_case()
    with pytest.raises(ValueError):
        model.recon_split_sino(sino, half_overlap=1)
    with pytest.raises(AssertionError):
        model.recon_split_sino(sino[0])
    with pytest.raises(AssertionError):
        model.recon_split_sino(sino, weights=weights[:, :4, :])


def _small_parallel_case():
    # (views, detector rows, detector channels).  Rows are recon slices in this geometry, so the
    # 20 rows give 20 slices: enough for a 2-part and a 3-part split at half_overlap=3, which
    # needs 2 * half_overlap slices in every part.
    cell = (28, 20, 28)
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(cell, angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    rshape = tuple(model.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rshape)
    sino = model.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / sino.max(), weight_type='transmission_root')
    return model, sino, weights


def _fits_after(num_refusals):
    """A stand-in for the memory predicate that refuses the first num_refusals part counts and
    accepts the next one, so the estimate's loop can be checked without real memory pressure."""
    state = {'calls': 0}

    def predicate(self, *args, **kwargs):
        state['calls'] += 1
        return state['calls'] > num_refusals

    return predicate


def test_parallel_split_approximates_full_recon():
    model, sino, weights = _small_parallel_case()
    np.random.seed(0)
    full, _ = model.recon(sino, weights=weights, max_iterations=8)
    np.random.seed(0)
    split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                               max_iterations=8, slices_per_part=10)
    assert split.shape == full.shape
    nrmse = float(np.linalg.norm(split - full) / np.linalg.norm(full))
    print(f"parallel 2-part split vs full NRMSE = {nrmse:.4f}")
    # Measured 0.032 over three seeds; gated a factor of ~3 above that.
    assert nrmse < 0.1
    sp = split_dict['split_params']
    assert sp['num_parts'] == 2 and sp['estimated'] is False
    assert sp['half_overlap_sino'] == 3 and sp['half_overlap_recon'] == 3
    assert sp['part_slice_ranges'] == [(0, 10), (10, 20)]
    assert sp['slices_per_part'] == 10


def test_parallel_split_in_three_parts():
    model, sino, weights = _small_parallel_case()
    num_slices = model.get_params('recon_shape')[2]
    np.random.seed(0)
    full, _ = model.recon(sino, weights=weights, max_iterations=8)
    np.random.seed(0)
    # 7 slices per part over 20 slices gives three parts of uneven size.
    split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                               max_iterations=8, slices_per_part=7)
    assert split.shape == full.shape
    nrmse = float(np.linalg.norm(split - full) / np.linalg.norm(full))
    print(f"parallel 3-part split vs full NRMSE = {nrmse:.4f}")
    # Measured 0.038 over three seeds; the same gate as the two-part split.
    assert nrmse < 0.1
    for key in ('recon_params_parts', 'recon_log_parts', 'notes_parts', 'model_params_parts'):
        assert len(split_dict[key]) == 3
    sp = split_dict['split_params']
    assert sp['num_parts'] == 3 and sp['slices_per_part'] == 7
    # The kept ranges tile the volume: consecutive, no gaps, no repeats, covering every slice.
    ranges = sp['part_slice_ranges']
    assert ranges[0][0] == 0 and ranges[-1][1] == num_slices
    assert all(hi == next_lo for (_lo, hi), (next_lo, _next_hi) in zip(ranges, ranges[1:]))


def test_parallel_split_ignores_align_split_grid():
    # Rows and slices share one grid here, so the flag has nothing to align and must not change a
    # single value.  float32 reductions on the CPU depend on the thread count, so the comparison
    # runs on one torch thread, which makes a repeated recon reproducible bit for bit.
    model, sino, weights = _small_parallel_case()
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        np.random.seed(0)
        plain, _ = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                          max_iterations=4, slices_per_part=10,
                                          align_split_grid=False)
        np.random.seed(0)
        aligned, aligned_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                                       max_iterations=4, slices_per_part=10,
                                                       align_split_grid=True)
    finally:
        torch.set_num_threads(num_threads)
    assert np.array_equal(plain, aligned)
    assert aligned_dict['split_params']['align_split_grid'] is True
    assert aligned_dict['split_params']['grid_shift_alu'] == 0.0


def test_parallel_split_rejects_bad_inputs():
    model, sino, weights = _small_parallel_case()
    with pytest.raises(ValueError):
        model.recon_split_sino(sino, half_overlap=1)
    # A part has to keep 2 * half_overlap slices, so a smaller part size is refused.
    with pytest.raises(ValueError, match='half_overlap'):
        model.recon_split_sino(sino, half_overlap=3, slices_per_part=5)
    with pytest.raises(AssertionError):
        model.recon_split_sino(sino[0])
    with pytest.raises(AssertionError):
        model.recon_split_sino(sino, weights=weights[:, :4, :])

    # The divided device form is refused for all three arrays: the split works from the host.
    sino_placement = _sharding.Placement(['cpu', 'cpu'], axis=0, axis_len=sino.shape[0])
    placed_sino = _sharding.Shards(
        [torch.as_tensor(sino[start:end]) for _d, (start, end)
         in sino_placement.shard_ranges()], sino_placement)
    with pytest.raises(ValueError, match='sharded form'):
        model.recon_split_sino(placed_sino, half_overlap=3)
    with pytest.raises(ValueError, match='sharded form'):
        model.recon_split_sino(sino, weights=placed_sino, half_overlap=3)
    recon_shape = tuple(model.get_params('recon_shape'))
    recon_placement = _sharding.Placement(['cpu', 'cpu'], axis=-1, axis_len=recon_shape[-1])
    placed_init = _sharding.Shards(
        [torch.zeros(recon_shape[:2] + (end - start,))
         for _d, (start, end) in recon_placement.shard_ranges()], recon_placement)
    with pytest.raises(ValueError, match='sharded form'):
        model.recon_split_sino(sino, init_recon=placed_init, half_overlap=3)


@pytest.mark.parametrize('num_refusals, expected_parts', [(1, 2), (2, 3)])
def test_parallel_split_estimates_the_part_count(monkeypatch, num_refusals, expected_parts):
    # With no explicit part size the method walks the part counts upward and takes the first one
    # the memory model accepts.
    model, sino, weights = _small_parallel_case()
    monkeypatch.setattr(mbirtorch.TomographyModel, '_fits_available_devices',
                        _fits_after(num_refusals))
    np.random.seed(0)
    split, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                               max_iterations=4)
    assert split.shape == tuple(model.get_params('recon_shape'))
    sp = split_dict['split_params']
    assert sp['num_parts'] == expected_parts and sp['estimated'] is True
    assert len(sp['part_slice_ranges']) == expected_parts


def test_parallel_split_estimate_of_one_part_is_a_plain_recon(monkeypatch):
    # The whole volume fits, so there is nothing to split and no warning to give.
    model, sino, weights = _small_parallel_case()
    monkeypatch.setattr(mbirtorch.TomographyModel, '_fits_available_devices', _fits_after(0))
    np.random.seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        recon, recon_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                                   max_iterations=4)
    assert not any('falling back' in str(x.message) for x in caught)
    assert recon.shape == tuple(model.get_params('recon_shape'))
    assert 'recon_params' in recon_dict and 'split_params' not in recon_dict


def test_parallel_split_estimate_reads_the_device_budget(monkeypatch):
    """The estimate prices its candidate parts through the memory model the reconstruction itself
    uses, so the device budget is what moves it.  Only the budget reading is stood in for here."""
    model, sino, weights = _small_parallel_case()
    # A budget that holds anything: one part, which is a plain recon.
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', lambda device: 1 << 40)
    np.random.seed(0)
    _recon, whole_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                                max_iterations=1)
    assert 'split_params' not in whole_dict
    # A budget that holds nothing: the estimate runs out of part counts and takes the largest one
    # the overlaps allow, leaving the parts' own preflight to speak if they still do not fit.
    monkeypatch.setattr(_memory_ledger, 'device_budget_bytes', lambda device: 1)
    np.random.seed(0)
    _recon, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                                max_iterations=1)
    sp = split_dict['split_params']
    assert sp['num_parts'] == 3 and sp['estimated'] is True


def test_parallel_split_of_one_part_and_the_thin_volume_fallback():
    # An explicit part size at least as large as the volume asks for one part, which is a plain
    # recon, and that is not a fallback: no warning.
    model, sino, weights = _small_parallel_case()
    num_slices = model.get_params('recon_shape')[2]
    np.random.seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        recon, recon_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                                   max_iterations=4,
                                                   slices_per_part=num_slices)
    assert not any('falling back' in str(x.message) for x in caught)
    assert recon.shape == tuple(model.get_params('recon_shape'))
    assert 'split_params' not in recon_dict

    # Fewer than 4 * half_overlap slices leaves no valid split at all, which does warn.
    tiny = mbirtorch.ParallelBeamModel((8, 6, 12), np.linspace(0, np.pi, 8, endpoint=False))
    tiny.configure_devices(devices=['cpu'])
    tiny.set_params(no_warning=True, verbose=0)
    tsino = np.ones((8, 6, 12), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        recon, _ = tiny.recon_split_sino(tsino, max_iterations=2)
    assert any('falling back' in str(x.message) for x in caught)
    assert recon.shape == tuple(tiny.get_params('recon_shape'))


def test_parallel_split_preserves_device_layout(monkeypatch):
    # A parent whose devices the caller fixed hands the same devices to every part.  The parts'
    # reconstructions are stood in for, so this costs no iterations.
    model, sino, weights = _small_parallel_case()
    seen = []

    def record_devices(self, sinogram, **kwargs):
        seen.append((list(self.sino_placement.devices), self.device_layout_is_automatic))
        return np.zeros(tuple(self.get_params('recon_shape')), dtype=np.float32), {}

    monkeypatch.setattr(mbirtorch.TomographyModel, 'recon', record_devices)
    recon, split_dict = model.recon_split_sino(sino, weights=weights, half_overlap=3,
                                               slices_per_part=10)
    assert len(seen) == 2
    parent_devices = list(model.sino_placement.devices)
    for devices, is_automatic in seen:
        assert devices == parent_devices
        assert is_automatic is False
    assert recon.shape == tuple(model.get_params('recon_shape'))
    assert split_dict['split_params']['num_parts'] == 2


@pytest.mark.goldens
@pytest.mark.skipif(not os.path.exists(_npz_path), reason="no preprocess goldens")
def test_split_golden_parity():
    golden = np.load(_npz_path)
    cell = tuple(int(v) for v in golden["mar_cell"])
    model = mbirtorch.ConeBeamModel(cell, golden["mar_angles"],
                                    source_detector_dist=float(golden["mar_sdd"]),
                                    source_iso_dist=float(golden["mar_sid"]))
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    np.random.seed(19)
    recon, split_dict = model.recon_split_sino(golden["mar_sino"].copy(),
                                               weights=golden["mar_weights"].copy(),
                                               half_overlap=4, max_iterations=5)
    sp = split_dict['split_params']
    # The geometry-derived overlaps are deterministic and must match mbirjax exactly.
    assert sp['half_overlap_sino'] == int(golden["split_overlap_sino"])
    assert sp['half_overlap_recon'] == int(golden["split_overlap_recon"])
    assert recon.shape == golden["split_recon"].shape
    nrmse = float(np.linalg.norm(recon - golden["split_recon"]) / np.linalg.norm(golden["split_recon"]))
    print(f"split cross-framework NRMSE = {nrmse:.4f} (measured; recons in the loop)")
    assert nrmse < 0.1
