"""Gates for ConeBeamModel.split_sino_recon.

Contract tests: the split reconstruction approximately equals recon() on the
same inputs; a 2-device layout on the parent carries to the halves and
produces the same result as the single-device split; a split too close to the
volume boundary falls back to a standard recon with a warning.  Cross-framework
parity against the mbirjax golden is measured loosely (iterative recons in the
loop, as for the MAR end-to-end gate); the geometry-derived overlaps must
match mbirjax exactly.
"""

import os
import warnings

import numpy as np
import pytest

import mbirtorch

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
    split, split_dict = model.split_sino_recon(sino, weights=weights, half_overlap=4,
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
    split1, _ = model.split_sino_recon(sino, weights=weights, half_overlap=4, max_iterations=5)

    model2, _, _ = _small_cone_case()
    model2.configure_devices(devices=['cpu', 'cpu'])
    np.random.seed(0)
    split2, _ = model2.split_sino_recon(sino, weights=weights, half_overlap=4, max_iterations=5)
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
        recon, _ = tiny.split_sino_recon(tsino, max_iterations=2)
    assert any('falling back' in str(x.message) for x in w)
    assert recon.shape == tuple(tiny.get_params('recon_shape'))


def test_split_rejects_bad_inputs():
    model, sino, weights = _small_cone_case()
    with pytest.raises(ValueError):
        model.split_sino_recon(sino, half_overlap=1)
    with pytest.raises(AssertionError):
        model.split_sino_recon(sino[0])
    with pytest.raises(AssertionError):
        model.split_sino_recon(sino, weights=weights[:, :4, :])


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
    recon, split_dict = model.split_sino_recon(golden["mar_sino"].copy(),
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
