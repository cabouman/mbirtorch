"""Gate for the sharded path of segment_plastic_metal: run the same volume
through the function whole and split into shards, and require the same
answer.  Note the sharded binning is NOT exact in general (float32 bucketing
vs np.histogram's float64 -- see _sharded_masked_histogram), but on this
volume the two paths happen to bin identically, and the test pins that the
thresholds and masks agree here.
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


def _small_mar_case(devices):
    """A small cone model with a plastic cube and one metal insert."""
    cell = (16, 16, 16)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=64,
                                    source_iso_dist=32)
    model.configure_devices(devices=devices)
    model.set_params(no_warning=True, verbose=0)
    shape = tuple(model.get_params('recon_shape'))
    vol = np.zeros(shape, dtype=np.float32)
    vol[4:12, 4:12, 4:12] = 0.02
    vol[6:9, 6:9, 6:9] = 0.2
    sino = np.asarray(model.forward_project(vol))
    return model, sino, vol


def test_sharded_bh_correction_matches_single_device():
    """correct_sino_plastic_metal on 2 CPU shards vs 1 device.  Greg's A7
    gates: the maxima are order-invariant (exact by construction); the fit
    sums combine as per-shard doubles on the host, so the corrected sinogram
    gates at the full-pipeline tolerance (discrete constraint selection can
    amplify float differences)."""
    ref_model, sino, vol = _small_mar_case(['cpu'])
    np.random.seed(0)          # the VCD pixel orderings come from the global RNG
    ref = mtp.recon_plastic_metal(ref_model, sino, None, num_metal=1,
                                  num_BH_iterations=2, max_iterations=2,
                                  verbose=0, logfile_path=None)

    sh_model, _, _ = _small_mar_case(['cpu', 'cpu'])
    np.random.seed(0)
    out = mtp.recon_plastic_metal(sh_model, sino, None, num_metal=1,
                                  num_BH_iterations=2, max_iterations=2,
                                  verbose=0, logfile_path=None)

    assert out.shape == ref.shape
    rel = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    print(f"sharded vs single MAR recon rel_max = {rel:.2e}")
    assert rel < 1e-3


def test_sharded_save_and_export_stream_by_slab(tmp_path, monkeypatch):
    """Sharded saves gather one slab at a time (never the whole volume) and
    still write byte-identical files.  The slab size is shrunk so several
    slabs are written; both sharding axes are covered (views: axis 0;
    recon slices: last axis, padded 11 -> 12)."""
    import os
    from mbirtorch import _sharding, utilities
    monkeypatch.setattr(utilities, '_HDF5_SLAB_BYTES', 256)

    def as_shards(vol, axis, n):
        pl = _sharding.Placement(['cpu'] * n, axis=axis, real_size=vol.shape[axis])
        pad = list(vol.shape)
        pad[axis] = pl.padded_size
        padded = np.zeros(pad, dtype=vol.dtype)
        sel = [slice(None)] * vol.ndim
        sel[axis] = slice(0, vol.shape[axis])
        padded[tuple(sel)] = vol
        tensors = []
        for _d, (s0, s1) in pl.shard_ranges(pl.padded_size):
            cut = [slice(None)] * vol.ndim
            cut[axis] = slice(s0, s1)
            tensors.append(torch.as_tensor(padded[tuple(cut)]))
        return _sharding.Shards(tensors, pl)

    vol = np.random.RandomState(8).rand(9, 7, 11).astype(np.float32)

    p1 = os.path.join(str(tmp_path), 'axis0.h5')
    mbirtorch.save_data_hdf5(p1, as_shards(vol, 0, 2), array_name='volume')
    out, _ = mbirtorch.load_data_hdf5(p1)
    assert np.array_equal(out, vol)

    p2 = os.path.join(str(tmp_path), 'slices.h5')
    mbirtorch.export_recon_hdf5(p2, as_shards(vol, 2, 2))
    out, _ = mbirtorch.import_recon_hdf5(p2)
    assert np.array_equal(out, vol)
