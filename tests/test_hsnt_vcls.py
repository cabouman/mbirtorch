"""Golden parity gates for the hsnt and vcls modules.

Both modules use seeded randomness, so the goldens fix the seeds and the
parity gates compare seeded runs on shared inputs (the plan's increment-6
gates): dehydrate-then-rehydrate round-trip and the hsnt HDF5 format, and a
small seeded get_opt_views case against the mbirjax golden.  hsnt is
host-side numpy/sklearn shared code, so it gates tight; vcls has each
framework's own projections inside, so the view SELECTION must match and the
VCL value gates at a measured tolerance.
"""

import os

import numpy as np
import pytest

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_npz_path),
    reason="no preprocess goldens: run tests/generate_preprocess_goldens.py in the mbirjax env")


@pytest.fixture(scope="module")
def golden():
    return np.load(_npz_path)


def test_dehydrate_rehydrate_parity(golden):
    # Shared input data (the golden carries the seeded simulated scan); the NMF path is the same
    # numpy/sklearn code in both packages.  random_state MUST match the seed the golden was written
    # with (tests/generate_preprocess_goldens.py): it pins sklearn's nndsvd initialization, which
    # otherwise draws from the global RNG.  Unseeded, the factors are a different NMF local optimum
    # every run and this test failed ~35% of the time (over 500 unseeded runs: err_d p50 1.4e-3 /
    # p99 9.7e-3, err_r p50 9.1e-5 / p99 7.8e-4 -- even the product gate flaked, ~2% of runs).
    dehydrated = mbirtorch.dehydrate(golden["hsnt_data"].copy(), num_materials=3, random_state=52,
                                     verbose=0)
    sub_data, sub_basis, dataset_type = dehydrated
    assert dataset_type == 'attenuation'
    assert sub_data.shape == golden["hsnt_sub_data"].shape
    assert sub_basis.shape == golden["hsnt_sub_basis"].shape
    err_d = float(np.max(np.abs(sub_data - golden["hsnt_sub_data"])) /
                  max(np.max(np.abs(golden["hsnt_sub_data"])), 1e-30))
    err_b = float(np.max(np.abs(sub_basis - golden["hsnt_sub_basis"])) /
                  max(np.max(np.abs(golden["hsnt_sub_basis"])), 1e-30))
    rehydrated = mbirtorch.rehydrate(dehydrated)
    err_r = float(np.max(np.abs(rehydrated - golden["hsnt_rehydrated"])) /
                  max(np.max(np.abs(golden["hsnt_rehydrated"])), 1e-30))
    print(f"hsnt rel_max = {err_d:.2e} (sub_data), {err_b:.2e} (basis), {err_r:.2e} (rehydrated)")
    # With both sides seeded alike this is a bit-exactness check, not a tolerance check: hsnt is
    # shared numpy/sklearn code, so the same seed reaches the same NMF optimum in both packages.
    # Measured 0.0 / 0.0 / 0.0 here, and bit-identical across the two conda envs (numpy 2.5.1 +
    # scipy 1.18 vs 2.4.6 + 1.17).  The gate stays at 1e-6 rather than asserting equality because
    # both of those are arm64 miniforge builds on one machine -- a different platform or BLAS could
    # still perturb the last bits.  A blown gate here means the shared NMF path diverged, NOT that
    # the tolerance needs raising; loosening it back to the old 2e-3 would only re-hide that.
    assert err_d < 1e-6 and err_b < 1e-6 and err_r < 1e-6


def test_rehydrate_of_golden_dehydration_is_exact(golden):
    # Rehydration is a matmul; on the golden dehydrated arrays it must match to float precision.
    out = mbirtorch.rehydrate([golden["hsnt_sub_data"].copy(), golden["hsnt_sub_basis"].copy(),
                               'attenuation'])
    err = float(np.max(np.abs(out - golden["hsnt_rehydrated"])) /
                max(np.max(np.abs(golden["hsnt_rehydrated"])), 1e-30))
    print(f"rehydrate-of-golden rel_max = {err:.2e}")
    assert err < 1e-6


def test_read_mbirjax_hsnt_file(golden):
    path = os.path.join(GOLDEN_DIR, 'preprocess_goldens_hsnt.h5')
    if not os.path.exists(path):
        pytest.skip('mbirjax hsnt golden not generated')
    data, metadata = mbirtorch.import_hsnt_data_hdf5(path)
    sub_data, sub_basis, dataset_type = data
    assert np.allclose(sub_data, golden["hsnt_sub_data"])
    assert np.allclose(sub_basis, golden["hsnt_sub_basis"])
    assert dataset_type == 'attenuation'
    assert metadata['dataset_name'] == 'golden'


def test_hsnt_hdf5_round_trip(tmp_path, golden):
    dehydrated = [golden["hsnt_sub_data"].copy(), golden["hsnt_sub_basis"].copy(), 'attenuation']
    metadata = mbirtorch.hsnt.create_hsnt_metadata(dataset_name='rt', dataset_type='attenuation',
                                                   angles=np.array([0.0, 90.0]))
    path = os.path.join(str(tmp_path), 'hsnt.h5')
    mbirtorch.export_hsnt_data_hdf5(path, dehydrated, metadata)
    data2, md2 = mbirtorch.import_hsnt_data_hdf5(path)
    assert np.array_equal(data2[0], dehydrated[0])
    assert np.array_equal(data2[1], dehydrated[1])
    assert data2[2] == 'attenuation'
    assert md2['dataset_name'] == 'rt' and np.allclose(md2['angles'], [0.0, 90.0])


def test_get_opt_views_seeded_golden(golden):
    model = mbirtorch.ParallelBeamModel((24, 8, 32), golden["vcls_angles"], device='cpu')
    model.set_params(no_warning=True, verbose=0)
    inds, vcl = mbirtorch.get_opt_views(model, golden["vcls_ref"].copy(), num_selected_views=5,
                                        r_1=0.05, seed=3)
    err = abs(vcl - float(golden["vcls_value"])) / max(abs(float(golden["vcls_value"])), 1e-30)
    print(f"vcls: selected {list(inds)} vs golden {list(golden['vcls_inds'])}; "
          f"vcl rel err = {err:.2e}")
    assert np.array_equal(np.asarray(inds), golden["vcls_inds"])
    assert err < 1e-2
