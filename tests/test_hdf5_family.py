"""Round-trip and cross-package format gates for the HDF5 save/load family.

The mbirjax-written files in tests/goldens (from generate_preprocess_goldens.py)
pin the on-disk format as shared between the two packages; the round-trip tests
cover the mbirtorch write path.  get_all_params gates on a constructor
round-trip: rebuilding a model from required_params reproduces the parameters.
"""

import os

import numpy as np
import pytest
import torch

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")

pytestmark = [pytest.mark.goldens, pytest.mark.skipif(
    not os.path.exists(_npz_path),
    reason="no preprocess goldens: run tests/generate_preprocess_goldens.py in the mbirjax env")]


@pytest.fixture(scope="module")
def golden():
    return np.load(_npz_path)


def test_save_load_data_round_trip(tmp_path):
    vol = np.random.RandomState(5).rand(6, 7, 8).astype(np.float32)
    path = os.path.join(str(tmp_path), 'sub', 'vol.h5')   # exercises makedirs
    attrs = {'scan_id': 'abc', 'nested': {'a': 1}}
    mbirtorch.save_data_hdf5(path, vol, array_name='volume', attributes_dict=attrs)
    out, out_attrs = mbirtorch.load_data_hdf5(path)
    assert np.array_equal(out, vol)
    assert out_attrs['scan_id'] == 'abc'
    assert 'a' in out_attrs['nested']


def test_save_data_accepts_tensor(tmp_path):
    vol = torch.rand(4, 5, 6)
    path = os.path.join(str(tmp_path), 'vol.h5')
    mbirtorch.save_data_hdf5(path, vol, array_name='volume')
    out, _ = mbirtorch.load_data_hdf5(path)
    assert np.array_equal(out, vol.numpy())


def test_export_import_round_trip(tmp_path):
    vol = np.random.RandomState(6).rand(9, 11, 13).astype(np.float32)
    path = os.path.join(str(tmp_path), 'recon.h5')
    mbirtorch.export_recon_hdf5(path, vol, recon_dict={'scan_id': 's'})
    out, out_dict = mbirtorch.import_recon_hdf5(path)
    assert np.array_equal(out, vol)          # transpose is undone on import
    assert out_dict['scan_id'] == 's'


def test_export_remove_flash_matches_mask(tmp_path):
    import mbirtorch.preprocess as mtp
    vol = np.random.RandomState(7).rand(12, 12, 10).astype(np.float32)
    path = os.path.join(str(tmp_path), 'recon.h5')
    mbirtorch.export_recon_hdf5(path, vol.copy(), remove_flash=True,
                                radial_margin=2, top_margin=2, bottom_margin=3)
    out, _ = mbirtorch.import_recon_hdf5(path)
    ref = mtp.apply_cylindrical_mask(vol.copy(), radial_margin=2, top_margin=2, bottom_margin=3)
    assert np.array_equal(out, ref)


def test_read_mbirjax_data_file(golden):
    path = os.path.join(GOLDEN_DIR, 'preprocess_goldens_data.h5')
    if not os.path.exists(path):
        pytest.skip('mbirjax data golden not generated')
    out, attrs = mbirtorch.load_data_hdf5(path)
    assert np.array_equal(out, golden['h5_vol'])
    assert attrs['scan_id'] == 'sample1'


def test_read_mbirjax_export_files(golden):
    path = os.path.join(GOLDEN_DIR, 'preprocess_goldens_export.h5')
    if not os.path.exists(path):
        pytest.skip('mbirjax export golden not generated')
    out, out_dict = mbirtorch.import_recon_hdf5(path)
    assert np.array_equal(out, golden['h5_vol'])
    assert out_dict['scan_id'] == 'sample1'

    flash_path = os.path.join(GOLDEN_DIR, 'preprocess_goldens_export_flash.h5')
    if os.path.exists(flash_path):
        import mbirtorch.preprocess as mtp
        out_f, _ = mbirtorch.import_recon_hdf5(flash_path)
        ref = mtp.apply_cylindrical_mask(golden['h5_vol'].copy(), radial_margin=2,
                                         top_margin=2, bottom_margin=3)
        err = float(np.max(np.abs(out_f - ref)))
        print(f"flash export vs mask max abs diff = {err:.2e}")
        assert err < 1e-6


@pytest.fixture(scope="module")
def cone_model():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    model = mbirtorch.ConeBeamModel((8, 10, 12), angles, source_detector_dist=100.0,
                                    source_iso_dist=50.0)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0, sharpness=1.5, det_channel_offset=0.25)
    return model


def test_get_all_params_partition(cone_model):
    required, optional, regularization = cone_model.get_all_params()
    assert required['sinogram_shape'] == (8, 10, 12)
    assert required['source_detector_dist'] == 100.0
    assert len(required['angles']) == 8
    assert len(required['helical_z_shifts']) == 8
    assert 'geometry_type' in required
    assert 'det_channel_offset' in optional and optional['det_channel_offset'] == 0.25
    assert regularization['sharpness'] == 1.5
    assert 'sharpness' not in optional and 'sigma_y' not in optional
    # No overlap between the three dicts.
    keys = list(required) + list(optional) + list(regularization)
    assert len(keys) == len(set(keys))


def test_get_all_params_constructor_round_trip(cone_model):
    required, optional, regularization = cone_model.get_all_params()
    required = dict(required)
    required.pop('geometry_type')
    rebuilt = mbirtorch.ConeBeamModel(**required)
    rebuilt.configure_devices(devices=['cpu'])
    rebuilt.set_params(no_warning=True, **optional)
    rebuilt.set_params(no_warning=True, **regularization)
    req2, opt2, reg2 = rebuilt.get_all_params()
    for key, val in required.items():
        assert np.allclose(np.asarray(req2[key], dtype=np.float64),
                           np.asarray(val, dtype=np.float64)), key
    for key, val in {**optional, **regularization}.items():
        v2 = {**opt2, **reg2}[key]
        if isinstance(val, (int, float, np.floating)):
            assert np.isclose(float(v2), float(val)), key
        else:
            assert np.array_equal(np.asarray(v2), np.asarray(val)), key


def test_save_load_recon_hdf5(tmp_path, cone_model):
    recon = np.random.RandomState(8).rand(*cone_model.get_params('recon_shape')).astype(np.float32)
    recon_dict = cone_model.get_recon_dict(recon_params={'num_iterations': 3}, notes='test scan')
    path = os.path.join(str(tmp_path), 'recon.h5')
    cone_model.save_recon_hdf5(path, recon, recon_dict=recon_dict)
    out, out_dict = mbirtorch.TomographyModel.load_recon_hdf5(path)
    assert np.array_equal(out, recon)
    assert out_dict['notes'] == 'test scan'
    assert 'num_iterations' in out_dict['recon_params']
    assert 'geometry_type' in out_dict['model_params']


def test_get_recon_dict_str_format(cone_model):
    d = cone_model.get_recon_dict(recon_params={'a': 1}, str_format=True)
    assert all(isinstance(v, str) for v in d.values())
