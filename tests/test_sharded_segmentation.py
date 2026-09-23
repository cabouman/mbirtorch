"""Gate for the sharded path of segment_plastic_metal: run the same volume
through the function whole and split into shards, and require the same
answer.  Identical is the right bar for THIS volume: the counts are exact
integers and the masks are pure thresholding, so a threshold moves only if
the sharded binning displaces a count across the boundary the DP picks.  That
is a measured expectation, not a guarantee -- the device-side binning is
float32 and truncating where numpy's is float64 with an edge-correction pass
(see _sharded_masked_histogram) -- so a future volume that trips it is a
tolerance question, not a bug in this gate.

The second test covers the HDF5 writers.  save_data_hdf5 and export_recon_hdf5
take shards directly and must write exactly what writing the gathered volume
writes, on both sharding axes.
"""

import numpy as np
import pytest
import torch

import mbirtorch
import mbirtorch.preprocess as mtp
from mbirtorch import _sharding


def _test_volume():
    """A small volume with three intensity classes and an odd slice count,
    so a 2-shard split gives shards of 6 and 5 slices."""
    rng = np.random.default_rng(3)
    vol = rng.uniform(0.0, 0.02, size=(40, 40, 11)).astype(np.float32)
    vol[10:30, 10:30, 2:9] += 0.05      # plastic
    vol[18:22, 18:22, 3:8] += 0.2       # metal
    return vol


def _as_shards(vol, n_shards):
    """Split the volume's slice axis over n CPU shards, as the engine does."""
    placement = _sharding.Placement(['cpu'] * n_shards, axis=-1,
                                    axis_len=vol.shape[2])
    tensors = [torch.as_tensor(vol[:, :, s0:s1].copy())
               for _dev, (s0, s1) in placement.shard_ranges()]
    return _sharding.Shards(tensors, placement)


def test_sharded_segmentation_matches_unsharded():
    vol = _test_volume()
    ref_p, ref_m, ref_ps, ref_ms = mtp.segment_plastic_metal(
        torch.as_tensor(vol), num_metal=1)

    shards = _as_shards(vol, 2)
    p, m, ps, ms = mtp.segment_plastic_metal(shards, num_metal=1)

    # Masks come back sharded on the same layout as the input.
    assert isinstance(p, _sharding.Shards) and p.placement is shards.placement
    assert np.array_equal(p.gather(), ref_p.numpy())
    assert np.array_equal(m[0].gather(), ref_m[0].numpy())
    # Scale factors agree to float rounding (sums accumulate per shard).
    assert ps == pytest.approx(ref_ps, rel=1e-5)
    assert ms[0] == pytest.approx(ref_ms[0], rel=1e-5)


def test_sharded_hdf5_writes_match_the_whole_volume_write(tmp_path, monkeypatch):
    """save_data_hdf5 and export_recon_hdf5 take a Shards directly and write
    exactly what writing the gathered volume writes -- same shape, dtype and
    content.  The slab size is shrunk so the sharded writes stream several
    slabs instead of gathering the whole volume, and both sharding axes are
    covered (views: axis 0; recon slices: last axis)."""
    import os
    from mbirtorch import utilities
    monkeypatch.setattr(utilities, '_HDF5_SLAB_BYTES', 256)

    def shard_on(vol, axis, n):
        pl = _sharding.Placement(['cpu'] * n, axis=axis, axis_len=vol.shape[axis])
        tensors = []
        for _d, (s0, s1) in pl.shard_ranges():
            cut = [slice(None)] * vol.ndim
            cut[axis] = slice(s0, s1)
            tensors.append(torch.as_tensor(vol[tuple(cut)].copy()))
        return _sharding.Shards(tensors, pl)

    vol = _test_volume()

    ref_path = os.path.join(str(tmp_path), 'ref_save.h5')
    out_path = os.path.join(str(tmp_path), 'sharded_save.h5')
    mbirtorch.save_data_hdf5(ref_path, vol, 'recon')
    mbirtorch.save_data_hdf5(out_path, shard_on(vol, 0, 2), 'recon')
    ref, _ = mbirtorch.load_data_hdf5(ref_path)
    out, _ = mbirtorch.load_data_hdf5(out_path)
    assert out.shape == ref.shape == vol.shape
    assert out.dtype == ref.dtype
    assert np.array_equal(out, ref)

    ref_path = os.path.join(str(tmp_path), 'ref_export.h5')
    out_path = os.path.join(str(tmp_path), 'sharded_export.h5')
    mbirtorch.export_recon_hdf5(ref_path, vol)
    mbirtorch.export_recon_hdf5(out_path, shard_on(vol, 2, 2))
    ref, _ = mbirtorch.import_recon_hdf5(ref_path)
    out, _ = mbirtorch.import_recon_hdf5(out_path)
    assert out.shape == ref.shape
    assert np.array_equal(out, ref)
