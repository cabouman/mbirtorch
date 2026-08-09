"""Gate for the sharded path of segment_plastic_metal: run the same volume
through the function whole and split into shards, and require the same
answer.  Identical is the right bar: the histogram counts are integers, so
the thresholds match exactly, and the masks are pure thresholding.
"""

import numpy as np
import pytest
import torch

import mbirtorch
import mbirtorch.preprocess as mtp
from mbirtorch import _sharding


def _test_volume():
    """A small volume with three intensity classes and an odd slice count,
    so a 2-shard split pads the slice axis (11 -> 12)."""
    rng = np.random.default_rng(3)
    vol = rng.uniform(0.0, 0.02, size=(40, 40, 11)).astype(np.float32)
    vol[10:30, 10:30, 2:9] += 0.05      # plastic
    vol[18:22, 18:22, 3:8] += 0.2       # metal
    return vol


def _as_shards(vol, n_shards):
    """Split the volume's slice axis over n CPU shards, zero-padding as the
    engine does, and return (shards, valid_mask, num_real_slices)."""
    placement = _sharding.Placement(['cpu'] * n_shards, axis=-1,
                                    real_size=vol.shape[2])
    padded = np.zeros(vol.shape[:2] + (placement.padded_size,), dtype=vol.dtype)
    padded[:, :, :vol.shape[2]] = vol
    tensors = [torch.as_tensor(padded[:, :, s0:s1])
               for _dev, (s0, s1) in placement.shard_ranges(placement.padded_size)]
    return (_sharding.Shards(tensors, placement),
            placement.real_mask(3), placement.real_size)


def test_sharded_segmentation_matches_unsharded():
    vol = _test_volume()
    ref_p, ref_m, ref_ps, ref_ms = mtp.segment_plastic_metal(
        torch.as_tensor(vol), num_metal=1)

    shards, valid_mask, num_real = _as_shards(vol, 2)
    p, m, ps, ms = mtp.segment_plastic_metal(
        shards, num_metal=1, valid_mask=valid_mask, num_real_slices=num_real)

    # Masks come back sharded on the same layout as the input.
    assert isinstance(p, _sharding.Shards) and p.placement is shards.placement
    p_full = p.gather()[:, :, :num_real]
    m_full = m[0].gather()[:, :, :num_real]
    assert np.array_equal(p_full, ref_p.numpy())
    assert np.array_equal(m_full, ref_m[0].numpy())
    # Padded slices of the masks are exactly zero.
    assert not p.gather()[:, :, num_real:].any()
    # Scale factors agree to float rounding (sums accumulate per shard).
    assert ps == pytest.approx(ref_ps, rel=1e-5)
    assert ms[0] == pytest.approx(ref_ms[0], rel=1e-5)


def test_one_shard_input_returns_one_shard_output():
    """The array-forms rule: you get back the form you put in."""
    vol = _test_volume()
    shards, valid_mask, num_real = _as_shards(vol, 1)
    p, m, _, _ = mtp.segment_plastic_metal(
        shards, num_metal=1, valid_mask=valid_mask, num_real_slices=num_real)
    assert isinstance(p, _sharding.Shards) and p.placement.n_devices == 1
    assert isinstance(m[0], _sharding.Shards)


def test_export_recon_hdf5_accepts_shards(tmp_path):
    """Exporting a sharded volume writes the same file as exporting it whole:
    gathered at the file boundary, padding cropped."""
    import os
    vol = _test_volume()
    shards, _mask, _nreal = _as_shards(vol, 2)

    ref_path = os.path.join(str(tmp_path), 'ref.h5')
    out_path = os.path.join(str(tmp_path), 'sharded.h5')
    mbirtorch.export_recon_hdf5(ref_path, vol)
    mbirtorch.export_recon_hdf5(out_path, shards)

    ref, _ = mbirtorch.load_data_hdf5(ref_path)
    out, _ = mbirtorch.load_data_hdf5(out_path)
    assert out.shape == ref.shape          # padding cropped: 12 -> 11 slices
    assert np.array_equal(out, ref)
