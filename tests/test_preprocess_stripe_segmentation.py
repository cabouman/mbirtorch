"""Golden parity gates for mbirtorch.preprocess stripe removal and segmentation.

Golden data comes from tests/generate_preprocess_goldens.py (run in the
mbirjax env).  The Otsu thresholds gate on EXACT match (integer bin
arithmetic on shared inputs); the stripe chains are host-side scipy/numpy and
gate at 1e-5 rel-max with measured floors printed.
"""

import os

import numpy as np
import pytest
import torch

import mbirtorch.preprocess as mtp

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_npz_path),
    reason="no preprocess goldens: run tests/generate_preprocess_goldens.py in the mbirjax env")


@pytest.fixture(scope="module")
def golden():
    return np.load(_npz_path)


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def test_remove_all_stripe(golden):
    out = mtp.remove_all_stripe(golden["stripe_sino"].copy(), snr=3,
                                large_filter_size=11, small_filter_size=5)
    err = _rel_max(out, golden["stripe_all"])
    print(f"remove_all_stripe rel_max = {err:.2e}")
    assert err < 1e-5


def test_remove_stripe_fw(golden):
    out = mtp.remove_stripe_fw(golden["stripe_sino"].copy())
    err = _rel_max(out, golden["stripe_fw"])
    print(f"remove_stripe_fw rel_max = {err:.2e}")
    assert err < 1e-5


def test_remove_sino_offset(golden):
    out = mtp.remove_sino_offset(golden["offset_in"].copy())
    err = _rel_max(out, golden["offset_out"])
    print(f"remove_sino_offset rel_max = {err:.2e}")
    assert err < 1e-5


def test_multi_threshold_otsu_exact(golden):
    # Integer bin arithmetic on shared inputs: thresholds must match mbirjax exactly.
    th3 = np.array(mtp.multi_threshold_otsu(golden["otsu_img"].copy(), classes=3), dtype=np.float64)
    th4 = np.array(mtp.multi_threshold_otsu(golden["otsu_img"].copy(), classes=4), dtype=np.float64)
    thm = np.array(mtp.multi_threshold_otsu(golden["otsu_img"].copy(), classes=3,
                                            valid_mask=golden["otsu_mask"].copy()), dtype=np.float64)
    assert np.array_equal(th3, golden["otsu_3"])
    assert np.array_equal(th4, golden["otsu_4"])
    assert np.array_equal(thm, golden["otsu_masked"])


def test_multi_threshold_otsu_torch_input(golden):
    th_np = mtp.multi_threshold_otsu(golden["otsu_img"].copy(), classes=3)
    th_t = mtp.multi_threshold_otsu(torch.as_tensor(golden["otsu_img"].copy()), classes=3)
    assert np.array_equal(np.array(th_np), np.array(th_t))


def test_segment_plastic_metal(golden):
    pm, mm, ps, ms = mtp.segment_plastic_metal(golden["seg_vol"].copy(), num_metal=2)
    assert np.array_equal(pm, golden["seg_pm"])
    assert np.array_equal(np.stack([np.asarray(m) for m in mm]), golden["seg_mm"])
    err_ps = abs(ps - float(golden["seg_ps"])) / max(abs(float(golden["seg_ps"])), 1e-30)
    err_ms = _rel_max(np.array(ms), golden["seg_ms"])
    print(f"segment scales rel err = {err_ps:.2e} (plastic), {err_ms:.2e} (metal)")
    assert err_ps < 1e-5 and err_ms < 1e-5


def test_segment_plastic_metal_torch_input(golden):
    pm_n, mm_n, ps_n, ms_n = mtp.segment_plastic_metal(golden["seg_vol"].copy(), num_metal=2)
    pm_t, mm_t, ps_t, ms_t = mtp.segment_plastic_metal(
        torch.as_tensor(golden["seg_vol"].copy()), num_metal=2)
    assert isinstance(pm_t, torch.Tensor)
    assert np.array_equal(pm_n, pm_t.numpy())
    assert all(np.array_equal(np.asarray(a), b.numpy()) for a, b in zip(mm_n, mm_t))
    assert np.allclose([ps_n] + list(ms_n), [ps_t] + list(ms_t), rtol=1e-6)
