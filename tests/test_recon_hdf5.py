"""export_recon_hdf5 and import_recon_hdf5: the file layout, the round trip of the volume and
the recon_dict, old files, the viewer's reader and writer, and the deprecated names."""
import warnings

import h5py
import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _sharding, utilities
from mbirtorch.viewers import slice_figure


def _volume(shape=(6, 5, 4), seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _recon_dict():
    return {'recon_params': {'iterations': 10, 'angles': np.linspace(0, np.pi, 2000),
                             'sharpness': np.float32(0.5), 'nested': {'a': [1, 2], 'b': 'text'}},
            'model_params': {'recon_shape': (6, 5, 4), 'delta_voxel': 1.0},
            'recon_log': 'line one\nline two',
            'notes': 'a note'}


def _assert_dict_round_trip(out, ref):
    assert set(out) == set(ref)
    assert out['recon_log'] == ref['recon_log'] and out['notes'] == ref['notes']
    assert out['recon_params']['iterations'] == 10
    assert np.array_equal(out['recon_params']['angles'], ref['recon_params']['angles'])
    assert out['recon_params']['angles'].dtype == np.float64
    assert out['recon_params']['sharpness'] == pytest.approx(0.5)
    assert out['recon_params']['nested'] == {'a': [1, 2], 'b': 'text'}
    assert out['model_params'] == {'recon_shape': [6, 5, 4], 'delta_voxel': 1.0}


def test_file_layout_is_right_handed_and_self_describing(tmp_path):
    vol = _volume()
    path = str(tmp_path / 'recon.h5')
    mbirtorch.export_recon_hdf5(path, vol, _recon_dict())
    with h5py.File(path, 'r') as f:
        assert list(f.keys()) == ['recon']
        dset = f['recon']
        assert dset.shape == (4, 5, 6)                      # (slice, col, row)
        assert dset.chunks == (1, 5, 6)
        assert np.array_equal(dset[()], np.transpose(vol, (2, 1, 0)))
        assert dset.attrs['format'] == 'mbirtorch_recon_v2'
        assert dset.attrs['axes'] == 'slice,col,row'
        assert dset.attrs['mbirtorch_version'] == mbirtorch.__version__
        assert dset.attrs['recon_log'] == 'line one\nline two'
        assert dset.attrs['recon_params'].startswith('{')


@pytest.mark.parametrize('kind', ['numpy', 'tensor', 'sharded'])
def test_round_trip_3d(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(utilities, '_HDF5_SLAB_BYTES', 64)     # several slabs for a small volume
    vol = _volume()
    if kind == 'numpy':
        recon = vol
    elif kind == 'tensor':
        recon = torch.as_tensor(vol)
    else:
        pl = _sharding.Placement(['cpu', 'cpu'], axis=2, axis_len=vol.shape[2])
        recon = _sharding.Shards([torch.as_tensor(vol[:, :, s0:s1].copy()) for _d, (s0, s1) in pl.shard_ranges()], pl)
    path = str(tmp_path / f'{kind}.h5')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mbirtorch.export_recon_hdf5(path, recon, _recon_dict())
        out, out_dict = mbirtorch.import_recon_hdf5(path)
    assert out.shape == vol.shape and out.dtype == vol.dtype
    assert np.array_equal(out, vol)
    _assert_dict_round_trip(out_dict, _recon_dict())


def test_round_trip_4d(tmp_path):
    vol = _volume((3, 6, 5, 4))
    path = str(tmp_path / 'recon4d.h5')
    mbirtorch.export_recon_hdf5(path, vol, {'notes': 'four'})
    with h5py.File(path, 'r') as f:
        assert f['recon'].shape == (3, 4, 5, 6)             # (time, slice, col, row)
        assert f['recon'].attrs['axes'] == 'time,slice,col,row'
    out, out_dict = mbirtorch.import_recon_hdf5(path)
    assert np.array_equal(out, vol)
    assert out_dict == {'notes': 'four'}


def test_remove_flash_matches_the_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(utilities, '_HDF5_SLAB_BYTES', 64)
    vol = _volume((12, 12, 9)) + 5
    path = str(tmp_path / 'flash.h5')
    mbirtorch.export_recon_hdf5(path, vol, remove_flash=True, radial_margin=1, top_margin=2, bottom_margin=3)
    out, _ = mbirtorch.import_recon_hdf5(path)
    expected = mbirtorch.preprocess.apply_cylindrical_mask(vol, 1, 2, 3)
    assert np.array_equal(out, expected)
    # The 4D path masks every frame the same way.
    vol4 = np.stack([vol, 2 * vol])
    mbirtorch.export_recon_hdf5(path, vol4, remove_flash=True, radial_margin=1, top_margin=2, bottom_margin=3)
    out4, _ = mbirtorch.import_recon_hdf5(path)
    assert np.array_equal(out4[0], expected) and np.array_equal(out4[1], 2 * expected)


def test_old_files_are_read_by_dataset_name(tmp_path):
    vol = _volume()
    # An old export file: dataset 'recon' in (slice, col, row) with text attributes and no format tag.
    old_export = str(tmp_path / 'old_export.h5')
    with h5py.File(old_export, 'w') as f:
        d = f.create_dataset('recon', data=np.transpose(vol, (2, 1, 0)))
        d.attrs['notes'] = 'old'
    with pytest.warns(UserWarning, match='old layout'):
        out, out_dict = mbirtorch.import_recon_hdf5(old_export)
    assert np.array_equal(out, vol) and out_dict == {'notes': 'old'}
    # An old viewer file: dataset 'volume' in (row, col, slice).
    old_viewer = str(tmp_path / 'old_viewer.h5')
    with h5py.File(old_viewer, 'w') as f:
        f.create_dataset('volume', data=vol)
    with pytest.warns(UserWarning, match='read as stored'):
        out, out_dict = mbirtorch.import_recon_hdf5(old_viewer)
    assert np.array_equal(out, vol) and out_dict == {}


def test_viewer_reads_and_writes_the_same_layout(tmp_path):
    vol = _volume()
    path = str(tmp_path / 'from_export.h5')
    mbirtorch.export_recon_hdf5(path, vol, _recon_dict())
    # The viewer's reader permutes by the axes attribute, so it shows (row, col, slice).
    array, data_dict = slice_figure.VolumeStack.read_file_array(path)
    assert np.array_equal(array, vol)
    assert data_dict['axes'] == 'slice,col,row' and data_dict['notes'] == 'a note'
    # The viewer's writer produces a file import_recon_hdf5 reads back unchanged.
    saved = str(tmp_path / 'from_viewer.h5')
    slice_figure._save_data_hdf5(saved, vol, 'recon', {'notes': 'from the viewer', 'axes': 'stale'})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out, out_dict = mbirtorch.import_recon_hdf5(saved)
    assert np.array_equal(out, vol)
    assert out_dict['notes'] == 'from the viewer'
    # A 4D volume takes the same path.
    vol4 = _volume((2, 6, 5, 4))
    slice_figure._save_data_hdf5(saved, vol4, 'recon', None)
    array, _ = slice_figure.VolumeStack.read_file_array(saved)
    assert np.array_equal(array, vol4)


def test_deprecated_names_warn_and_work(tmp_path):
    vol = _volume()
    model = mbirtorch.ParallelBeamModel((4, 6, 5), np.linspace(0, np.pi, 4, endpoint=False))
    path = str(tmp_path / 'deprecated.h5')
    with pytest.warns(FutureWarning, match='export_recon_hdf5'):
        model.save_recon_hdf5(path, vol, {'notes': 'x'})
    with pytest.warns(FutureWarning, match='import_recon_hdf5'):
        out, out_dict = mbirtorch.TomographyModel.load_recon_hdf5(path)
    assert np.array_equal(out, vol) and out_dict == {'notes': 'x'}
    with pytest.warns(FutureWarning):
        mbirtorch.save_data_hdf5(path, vol, 'array')
    with pytest.warns(FutureWarning):
        out, _ = mbirtorch.load_data_hdf5(path)
    assert np.array_equal(out, vol)
