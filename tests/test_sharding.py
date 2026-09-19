"""Multi-device units: Placement, Shards, safe transfer, the forward's
cylinder transfer, and the back projection's band reduce.  Runs everywhere:
placement/banding logic uses two 'virtual' CPU devices (['cpu', 'cpu'] --
transfers are no-ops but every range/pad/mask/assembly path executes), and
real cross-device movement uses the cpu<->mps pair when MPS is available (the
local analog of the 2-GPU platform)."""

import math

import numpy as np
import pytest
import torch

from mbirtorch import _sharding
from mbirtorch._sharding import (Placement, Shards, is_dev2dev_safe,
                                 move_shard, sum_band_to_owner)


def test_the_split_is_balanced_and_the_gather_keeps_every_element():
    """The four properties of the split, over sizes and device counts that
    test_placement_ranges_split_evenly_within_one does not reach.

    Three properties describe the blocks themselves.  They tile the axis with
    no gap and no overlap, their lengths differ by at most one, and the longer
    blocks come first.  The fourth property is about the gather: a Shards built
    on an uneven split concatenates back to the original array with nothing
    dropped.  The sizes below include device counts above the axis length,
    because that is the only way a block of length zero arises.
    """
    for size, count in [(7, 2), (6, 2), (5, 4), (9, 4), (3, 4), (1, 3),
                        (0, 2), (12, 3), (17, 5)]:
        p = Placement(["cpu"] * count, axis=0, axis_len=size)
        ranges = [r for _, r in p.shard_ranges()]
        lengths = [e - s for s, e in ranges]
        assert len(ranges) == count, (size, count)
        assert ranges[0][0] == 0 and ranges[-1][1] == size, (size, count)
        assert all(ranges[k][1] == ranges[k + 1][0]
                   for k in range(count - 1)), (size, count)
        assert sum(lengths) == size, (size, count)
        assert max(lengths) - min(lengths) <= 1, (size, count, lengths)
        assert lengths == sorted(lengths, reverse=True), (size, count, lengths)

        full = np.arange(3 * size, dtype=np.float32).reshape(size, 3)
        sh = Shards([torch.as_tensor(full[s:e]) for s, e in ranges], p)
        assert np.array_equal(sh.gather(), full), (size, count)

    # An explicit axis length overrides the placement's own, and a single
    # device is the trivial placement that owns the whole axis.
    p = Placement(["cpu", "cpu"], axis=0, axis_len=7)
    assert p.n_devices == 2 and not p.is_trivial
    assert [r for _, r in p.shard_ranges(6)] == [(0, 3), (3, 6)]
    q = Placement(["cpu"], axis=-1, axis_len=5)
    assert q.is_trivial and [r for _, r in q.shard_ranges()] == [(0, 5)]

    # No size to split, and none on the placement, is an error that says so.
    with pytest.raises(ValueError, match="needs an axis length"):
        Placement(["cpu"], axis=0).shard_ranges()

    # The band tiling the banded back driver walks splits an extent the same
    # balanced way.  A slice-owner with no slices arrives with an extent of 0
    # and a band length of 0, which the ceil division cannot take, so the
    # answer there is no bands and the driver's loop runs zero times.
    import mbirtorch
    bounds = mbirtorch.TomographyModel._balanced_slice_bounds
    assert bounds(0, 0) == [] and bounds(0, 4) == []
    assert bounds(-2, 0) == [] and bounds(-2, 4) == []
    for extent, band_len in [(1, 1), (1, 4), (6, 2), (7, 3), (5, 5), (9, 4)]:
        b = bounds(extent, band_len)
        lengths = [e - s for s, e in b]
        assert b[0][0] == 0 and b[-1][1] == extent       # covers [0, extent)
        assert all(b[k][1] == b[k + 1][0] for k in range(len(b) - 1))
        assert max(lengths) <= band_len
        assert max(lengths) - min(lengths) <= 1


def test_the_gather_rebuilds_the_array_exactly_on_either_axis():
    """A gather only moves bytes, so what comes back has to be the original
    array element for element rather than merely close to it.

    Nothing here computes a value, so allowing a tolerance would hide the one
    way this can really go wrong -- a shard copied into the wrong place -- so
    the comparison is exact.  The cases cover both axes a sharded array is cut
    on (a recon's last axis, a sinogram's view axis), splits that divide
    evenly and splits that do not, one, two and three devices, and several
    dtypes, since the copy works in bytes and dtypes of different widths move
    different amounts of them.
    """
    # (6, 4, 6) divides evenly by 2 and by 3 on both of the axes below, so
    # every shard is the same length; (7, 4, 5) divides evenly by neither, so
    # the shards differ in length and each one lands at a different offset.
    for shape in [(6, 4, 6), (7, 4, 5)]:
        for axis in [0, -1]:
            for count in [1, 2, 3]:
                for dtype in [torch.float32, torch.float64, torch.int64,
                              torch.uint8]:
                    case = (shape, axis, count, dtype)
                    full = torch.arange(shape[0] * shape[1] * shape[2]).reshape(
                        shape).to(dtype)
                    ref = full.numpy()
                    p = Placement(["cpu"] * count, axis=axis,
                                  axis_len=shape[axis])
                    index = [slice(None)] * 3
                    parts = []
                    for _dev, (start, end) in p.shard_ranges():
                        index[axis] = slice(start, end)
                        parts.append(full[tuple(index)].contiguous())
                    untouched = [t.clone() for t in parts]

                    gathered = Shards(parts, p).gather()
                    assert gathered.shape == shape, case
                    assert gathered.dtype == ref.dtype, case
                    assert np.array_equal(gathered, ref), case
                    # Callers may index the result any way they like, and the
                    # concatenate this replaced always handed back a
                    # C-contiguous array, so this one must as well.
                    assert gathered.flags["C_CONTIGUOUS"], case

                    # It must also be an array of its own: writing to it may
                    # not reach back into the shards it was built from.
                    gathered[:] = 0
                    assert all(torch.equal(t, u)
                               for t, u in zip(parts, untouched)), case

    # A container with the wrong number of shards is refused when it is built,
    # before anything is copied.
    p = Placement(["cpu", "cpu"], axis=0, axis_len=6)
    with pytest.raises(ValueError, match="shard tensors"):
        Shards([torch.zeros(3, 4)], p)


def test_the_slab_gather_rebuilds_the_array_exactly_at_every_slot_size(
        monkeypatch):
    """CUDA shards are gathered in slabs through staging slots, several host
    threads per shard (_fill_from_shards_in_slabs).  That is arithmetic on
    offsets, the same on any device, so it is exercised here on CPU tensors,
    with the slots unpinned and every copy blocking, and the result must be
    the original array element for element, as for the plain gather.  The
    cases cover the two axes a sharded array is cut on and a middle axis,
    splits that divide evenly and splits that do not, more devices than
    entries (a shard of zero length), several element sizes, one to five
    threads per shard, and slots from one element, which splits every row of
    a block into pieces, to one that holds a whole shard."""
    for shape, axis in [((6, 5, 7), -1), ((7, 5, 6), 0), ((5, 7, 6), 1),
                        ((6, 5, 3), -1), ((6, 5, 0), -1)]:
        for count in [1, 2, 3, 4]:
            for dtype in [torch.float32, torch.float64, torch.uint8]:
                full = torch.arange(math.prod(shape)).reshape(shape).to(dtype)
                ref = full.numpy()
                p = Placement(["cpu"] * count, axis=axis, axis_len=shape[axis])
                index = [slice(None)] * 3
                parts = []
                for _dev, (start, end) in p.shard_ranges():
                    index[axis] = slice(start, end)
                    parts.append(full[tuple(index)].contiguous())
                untouched = [t.clone() for t in parts]
                size = full.element_size()
                for threads in [1, 2, 5]:
                    monkeypatch.setattr(_sharding, "_gather_threads_per_shard",
                                        lambda n, threads=threads: threads)
                    for slot_bytes in [1, 3 * size, 7 * size, 2 ** 20]:
                        case = (shape, axis, count, dtype, threads, slot_bytes)
                        monkeypatch.setattr(_sharding, "GATHER_SLOT_BYTES",
                                            slot_bytes)
                        out = torch.empty(shape, dtype=dtype)
                        _sharding._fill_from_shards_in_slabs(parts, out,
                                                             axis % 3)
                        assert np.array_equal(out.numpy(), ref), case
                        assert all(torch.equal(t, u)
                                   for t, u in zip(parts, untouched)), case


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="needs a CUDA device")
def test_the_cuda_gather_rebuilds_the_array_exactly_in_slabs(monkeypatch):
    """The gather of CUDA shards takes the slab path -- pinned slots, copies
    that return at once, several threads per shard -- which the CPU cases
    above cannot reach.  The shards are spread over the visible devices, one
    device serving several shards when there are few, the threads per shard
    are forced to three, and the slot is made small so that every shard moves
    in many slabs and a row is split; the result must be the original array
    element for element at every slot size."""
    default_slot = _sharding.GATHER_SLOT_BYTES
    monkeypatch.setattr(_sharding, "_gather_threads_per_shard", lambda n: 3)
    n_cuda = torch.cuda.device_count()
    for shape, axis in [((6, 5, 1001), -1), ((1001, 6, 5), 0),
                        ((6, 5, 3), -1), ((6, 5, 0), -1)]:
        for count in [1, 2, 4]:
            devices = [torch.device("cuda", i % n_cuda) for i in range(count)]
            for dtype in [torch.float32, torch.uint8]:
                full = torch.arange(math.prod(shape)).reshape(shape).to(dtype)
                ref = full.numpy()
                p = Placement(devices, axis=axis, axis_len=shape[axis])
                index = [slice(None)] * 3
                parts = []
                for dev, (start, end) in p.shard_ranges():
                    index[axis] = slice(start, end)
                    parts.append(full[tuple(index)].to(dev))
                for slot_bytes in [64, 2 ** 20, default_slot]:
                    case = (shape, axis, count, dtype, slot_bytes)
                    monkeypatch.setattr(_sharding, "GATHER_SLOT_BYTES",
                                        slot_bytes)
                    gathered = Shards(parts, p).gather()
                    assert gathered.flags["C_CONTIGUOUS"], case
                    assert np.array_equal(gathered, ref), case


def test_the_streamed_reduce_matches_the_one_shot_sum_exactly(monkeypatch):
    """The reduce moves each arriving partial in bounded row slabs, so the
    owner never holds more than one slab per source above its running total.

    Streaming partitions the ELEMENTS: each element is still accumulated in
    partial order, so the streamed result is bit for bit the one-shot sum and
    not merely close to it.  And the partials are read, never written, so a
    caller may still use them afterwards.
    """
    from mbirtorch import _sharding
    owner = torch.device("cpu")
    partials = [torch.rand(37, 5) for _ in range(4)]
    untouched = [p.clone() for p in partials]
    one_shot = ((partials[0] + partials[1]) + partials[2]) + partials[3]
    # 40 bytes is two rows of a 5-column float32 band, so this runs 19 slabs
    # rather than the single slab the default budget would give at this size.
    monkeypatch.setattr(_sharding, "REDUCE_SLAB_BYTES", 40)
    assert _sharding.reduce_slab_rows(37, 5 * 4) == 2
    total = sum_band_to_owner(partials, owner)
    assert torch.equal(total, one_shot)
    assert all(torch.equal(p, u) for p, u in zip(partials, untouched))
    # A single partial is still handed straight back, with no copy made.
    assert sum_band_to_owner(partials[:1], owner) is partials[0]


def test_the_streamed_reduce_leaves_the_sharded_back_projection_unchanged(
        monkeypatch):
    """The test above pins the reduce; this one pins the driver that calls it.

    At test sizes a whole band fits inside one slab and moves in a single
    piece, so the streaming path runs end to end only when the budget is
    forced down.  Without this the suite would never execute a multi-slab
    reduce through the real back projection.
    """
    import mbirtorch
    from mbirtorch import _sharding
    sino_shape = (9, 7, 8)                       # 7 slices over 2 devices
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build(devices):
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        if devices != ["cpu"]:
            m.configure_devices(devices=devices)
        return m

    rng = np.random.default_rng(11)
    sino = rng.standard_normal(sino_shape).astype(np.float32)
    reference = build(["cpu"]).back_project(sino)
    # Two rows of a four-slice band at a time: many slabs, not one.
    monkeypatch.setattr(_sharding, "REDUCE_SLAB_BYTES", 2 * 4 * 4)
    m2 = build(["cpu", "cpu"])
    streamed = m2._gather_recon(m2.back_project(sino, output_sharded=True))
    rel = np.max(np.abs(streamed - reference)) / np.max(np.abs(reference))
    assert rel < 1e-5, rel


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason="needs a second local device (mps)")
def test_real_cross_device_transfer_cpu_mps():
    devs = ["cpu", "mps"]
    assert is_dev2dev_safe(devs)
    band = torch.rand(64, 16)
    copies = {torch.device(d): move_shard(band, torch.device(d)) for d in devs}
    assert copies[torch.device("mps")].device.type == "mps"
    # A transfer copies bytes, so the round trip is exact, not merely close.
    assert torch.equal(copies[torch.device("mps")].cpu(), band)
    partials = [copies[torch.device("cpu")] * 2.0,
                copies[torch.device("mps")] * 3.0]
    total = sum_band_to_owner(partials, torch.device("cpu"))
    assert total.device.type == "cpu"
    ref_total = band * 5.0
    rel = float((total - ref_total).abs().max()
                / max(float(ref_total.abs().max()), 1e-30))
    assert rel < 1e-6, rel
    # The host-bounce fallback path copies bytes as well, so it is exact too.
    bounced = move_shard(band.to("mps"), torch.device("cpu"), dev2dev_safe=False)
    assert torch.equal(bounced, band)


def test_model_shard_and_gather_roundtrip():
    # configure_devices widens the placements; _shard_sinogram/_shard_recon then split a
    # real-shape array into per-device shards and the gathers concatenate them
    # back -- a lossless round trip.  Two 'virtual' CPU devices keep
    # this runnable everywhere.
    import mbirtorch
    from mbirtorch._sharding import Shards
    sino_shape = (10, 6, 8)                      # 10 views over 2 devices
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    m = mbirtorch.ParallelBeamModel(sino_shape, angles)
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    m.configure_devices(devices=["cpu", "cpu"])

    sino = np.random.RandomState(0).rand(*sino_shape).astype(np.float32)
    sh = m._shard_sinogram(sino)
    assert isinstance(sh, Shards) and len(sh.tensors) == 2
    assert sh.tensors[0].shape[0] == 5           # 10 views split evenly
    assert np.array_equal(m._gather_sinogram(sh), sino)
    assert m._shard_sinogram(sh) is sh           # pass-through when placed

    rs = tuple(m.get_params('recon_shape'))
    recon = np.random.RandomState(1).rand(*rs).astype(np.float32)
    rh = m._shard_recon(recon)
    assert isinstance(rh, Shards)
    # slice axis 6 -> 3+3, and a NON-dividing case (7 detector rows -> 7
    # recon slices over 2 devices) splits 4+3:
    m2 = mbirtorch.ParallelBeamModel((10, 7, 8), np.linspace(0, np.pi, 10,
                                     endpoint=False))
    m2.configure_devices(devices=["cpu"])
    m2.set_params(no_warning=True, verbose=0)
    m2.configure_devices(devices=["cpu", "cpu"])
    r9 = np.random.RandomState(2).rand(*tuple(m2.get_params('recon_shape'))
                                       ).astype(np.float32)
    rh9 = m2._shard_recon(r9)
    assert [int(t.shape[-1]) for t in rh9.tensors] == [4, 3]
    assert np.array_equal(m2._gather_recon(rh9), r9)            # lossless

    # The banded drivers consume shards directly.
    assert isinstance(m.sparse_back_project(sino, np.arange(4)), Shards)
    m.configure_devices(1)                       # back to trivial
    assert m.sino_placement.is_trivial
    assert torch.is_tensor(m._shard_sinogram(sino))


def _banded_case(devices, sino_shape=(8, 6, 8)):
    import mbirtorch
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    m = mbirtorch.ParallelBeamModel(sino_shape, angles)
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    rs = tuple(m.get_params('recon_shape'))
    rng = np.random.RandomState(3)
    idx = np.sort(rng.choice(rs[0] * rs[1], size=40, replace=False))
    vals = rng.rand(len(idx), rs[2]).astype(np.float32)
    sino = rng.rand(*sino_shape).astype(np.float32)
    ref_fwd = m.sparse_forward_project(vals, idx).cpu().numpy()
    ref_back = m.sparse_back_project(sino, idx).cpu().numpy()
    ref_back2 = m.sparse_back_project(sino, idx, coeff_power=2).cpu().numpy()
    m.configure_devices(devices=devices)
    return m, idx, vals, sino, ref_fwd, ref_back, ref_back2


def test_banded_projectors_match_single_device():
    # The banded sharded forward/back on two virtual CPU devices reproduce the
    # single-device values (forward exactly one producer per row; back a
    # float-noise sum reorder), and the shard round trip preserves layout.
    from mbirtorch._sharding import Shards
    m, idx, vals, sino, ref_fwd, ref_back, ref_back2 = _banded_case(["cpu", "cpu"])
    fwd = m.sparse_forward_project(vals, idx)
    assert isinstance(fwd, Shards)
    rel = (np.max(np.abs(m._gather_sinogram(fwd) - ref_fwd))
           / max(np.max(np.abs(ref_fwd)), 1e-30))
    assert rel < 1e-5, rel
    back = m.sparse_back_project(sino, idx)
    assert isinstance(back, Shards)
    rel = (np.max(np.abs(back.gather() - ref_back))
           / max(np.max(np.abs(ref_back)), 1e-30))
    assert rel < 1e-5, rel
    back2 = m.sparse_back_project(sino, idx, coeff_power=2)
    rel = (np.max(np.abs(back2.gather() - ref_back2))
           / max(np.max(np.abs(ref_back2)), 1e-30))
    assert rel < 1e-5, rel


def test_banded_projectors_adjoint():
    m, idx, vals, sino, *_ = _banded_case(["cpu", "cpu"])
    fwd = m.sparse_forward_project(vals, idx)
    back = m.sparse_back_project(sino, idx)
    lhs = float(np.sum(m._gather_sinogram(fwd) * sino))
    rhs = float(np.sum(vals * back.gather()))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (lhs, rhs)


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason="needs a second local device (mps)")
def test_banded_projectors_real_two_devices():
    m, idx, vals, sino, ref_fwd, ref_back, _ = _banded_case(["cpu", "mps"])
    fwd = m.sparse_forward_project(vals, idx)
    assert fwd.tensors[1].device.type == "mps"
    rel = (np.max(np.abs(m._gather_sinogram(fwd) - ref_fwd))
           / max(np.max(np.abs(ref_fwd)), 1e-30))
    assert rel < 1e-5, rel
    back = m.sparse_back_project(sino, idx)
    rel = (np.max(np.abs(back.gather() - ref_back))
           / max(np.max(np.abs(ref_back)), 1e-30))
    assert rel < 1e-5, rel


def _cone_banded_case(devices, cell=(8, 8, 8)):
    import mbirtorch
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    m = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                source_iso_dist=2 * cell[2])
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    rs = tuple(m.get_params('recon_shape'))
    rng = np.random.RandomState(5)
    idx = np.sort(rng.choice(rs[0] * rs[1], size=30, replace=False))
    vals = rng.rand(len(idx), rs[2]).astype(np.float32)
    sino = rng.rand(*cell).astype(np.float32)
    ref_fwd = m.sparse_forward_project(vals, idx).cpu().numpy()
    ref_back = m.sparse_back_project(sino, idx).cpu().numpy()
    m.configure_devices(devices=devices)
    return m, idx, vals, sino, ref_fwd, ref_back


def _torch_body_case(m, devices, sino_shape, seed):
    """The shared tail of the two cases below: take the single-device
    references for a seeded sparse problem, then place the model on
    ``devices``.  Same return shape as ``_cone_banded_case``."""
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    rs = tuple(m.get_params('recon_shape'))
    rng = np.random.RandomState(seed)
    num_pixels = min(20, rs[0] * rs[1])
    idx = np.sort(rng.choice(rs[0] * rs[1], size=num_pixels, replace=False))
    vals = rng.rand(len(idx), rs[2]).astype(np.float32)
    sino = rng.rand(*sino_shape).astype(np.float32)
    ref_fwd = m.sparse_forward_project(vals, idx).cpu().numpy()
    ref_back = m.sparse_back_project(sino, idx).cpu().numpy()
    m.configure_devices(devices=devices)
    return m, idx, vals, sino, ref_fwd, ref_back


def _multiaxis_banded_case(devices, sino_shape=(8, 7, 8)):
    """A multiaxis model on virtual CPU devices, plus its single-device
    reference.  This cell's recon is (8, 8, 7), so at three devices neither
    the view axis nor the slice axis divides and both split unevenly."""
    import mbirtorch
    azimuth = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    elevation = np.linspace(-0.4, 0.4, sino_shape[0])
    m = mbirtorch.MultiAxisParallelModel(
        sino_shape, np.stack([azimuth, elevation], axis=1))
    return _torch_body_case(m, devices, sino_shape, seed=7)


def _translation_banded_case(devices, sino_shape=(4, 20, 16)):
    """A translation model on virtual CPU devices, plus its single-device
    reference.  Its recon is (1, 12, 8): four views over three devices and
    eight slices over three both split unevenly."""
    import mbirtorch
    tvecs = mbirtorch.gen_translation_vectors(2, 2, x_spacing=3.0,
                                              z_spacing=2.0)
    m = mbirtorch.TranslationModel(sino_shape, tvecs,
                                   source_detector_dist=128.0,
                                   source_iso_dist=32.0)
    return _torch_body_case(m, devices, sino_shape, seed=9)


def test_cone_banded_projectors_match_single_device():
    # Cone bands spread over many rows: the banded forward ACCUMULATES
    # full-row partials and the banded back consumes the full local sinogram
    # per band -- both must reproduce the single-device values (band tiling is
    # a sum reorder, so float noise only).
    m, idx, vals, sino, ref_fwd, ref_back = _cone_banded_case(["cpu", "cpu"])
    fwd = m.sparse_forward_project(vals, idx)
    rel = (np.max(np.abs(m._gather_sinogram(fwd) - ref_fwd))
           / max(np.max(np.abs(ref_fwd)), 1e-30))
    assert rel < 1e-5, rel
    back = m.sparse_back_project(sino, idx)
    rel = (np.max(np.abs(back.gather() - ref_back))
           / max(np.max(np.abs(ref_back)), 1e-30))
    assert rel < 1e-5, rel
    lhs = float(np.sum(m._gather_sinogram(fwd) * sino))
    rhs = float(np.sum(vals * back.gather()))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4


def test_qggmrf_halos_match_full_volume():
    # Per-shard qGGMRF gradient/Hessian with exchanged halos, concatenated
    # across shards, must equal the full single-device computation exactly:
    # the halo supplies the true cross-boundary delta that the reflected
    # boundary would otherwise zero.
    from mbirtorch import qggmrf
    from mbirtorch._sharding import (Placement, Shards, exchange_qggmrf_halos)
    rng = np.random.RandomState(9)
    rows, cols, S = 6, 5, 8
    flat = torch.as_tensor(rng.rand(rows * cols, S).astype(np.float32))
    idx = torch.as_tensor(np.sort(rng.choice(rows * cols, 12, replace=False)))
    params = ((1.0, 1.0, 1.0, 1.0, 0.8, 0.8), 0.3, 2.0, 1.2, 1.0)
    ref_g, ref_h = qggmrf.qggmrf_gradient_and_hessian_at_indices(
        flat, (rows, cols, S), idx, params)

    p = Placement(["cpu", "cpu"], axis=-1, axis_len=S)
    shards = Shards([flat[:, s:e] for _, (s, e) in p.shard_ranges(S)], p)
    lh, rh = exchange_qggmrf_halos(shards)
    assert lh[0] is None and rh[-1] is None
    assert torch.equal(lh[1], flat[:, 3]) and torch.equal(rh[0], flat[:, 4])
    parts = [qggmrf.qggmrf_gradient_and_hessian_at_indices(
        shards.tensors[i], (rows, cols, S), idx, params,
        left_halo=lh[i], right_halo=rh[i]) for i in range(2)]
    g = torch.cat([pg for pg, _ in parts], dim=1)
    h = torch.cat([ph for _, ph in parts], dim=1)
    rel_g = float((g - ref_g).abs().max()
                  / max(float(ref_g.abs().max()), 1e-30))
    rel_h = float((h - ref_h).abs().max()
                  / max(float(ref_h.abs().max()), 1e-30))
    assert rel_g < 1e-6 and rel_h < 1e-6, (rel_g, rel_h)

    # A shard that holds no slices sends no halo and receives none.  The last
    # shard that owns slices therefore gets None on its right, which the prior
    # maps to the reflected boundary condition at the last real slice.  The
    # shards here are built by hand: widths 2, 3, and 0 over three devices.
    rng3 = np.random.RandomState(23)
    num_pixels, S3 = 7, 5
    flat3 = torch.as_tensor(rng3.rand(num_pixels, S3).astype(np.float32))
    p3 = Placement(["cpu"] * 3, axis=-1, axis_len=S3)
    shards3 = Shards([flat3[:, 0:2], flat3[:, 2:5], flat3[:, 5:5]], p3)
    lh3, rh3 = exchange_qggmrf_halos(shards3)
    # The boundary between the two shards that own slices carries the same
    # values it carries when no shard is empty.
    assert torch.equal(lh3[1], flat3[:, 1]) and torch.equal(rh3[0], flat3[:, 2])
    assert lh3[0] is None and rh3[1] is None  # volume start, last real slice
    assert lh3[2] is None and rh3[2] is None  # the shard with no slices


def test_sharded_vcd_recon_matches_single_device():
    # The whole multi-device VCD loop, end to end: a seeded 4-iteration recon on two
    # virtual CPU devices must reproduce the single-device run -- same
    # partitions (shared global np.random stream), banded projections, halo'd
    # prior, host-combined line search.  Tolerances sit at the loop's
    # rerun-jitter scale plus the sharded sum reorders.
    import mbirtorch
    from mbirtorch._sharding import Shards
    sino_shape = (8, 6, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build():
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        return m

    m1 = build()
    rs = tuple(m1.get_params('recon_shape'))
    phantom = np.zeros(rs, dtype=np.float32)
    phantom[1:-1, 1:-1, 1:-1] = 1.0
    sino = m1.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / np.max(sino),
                                    weight_type='transmission_root')
    np.random.seed(31)
    ref, ref_dict = m1.recon(sino, weights=weights, max_iterations=4,
                             stop_threshold_change_pct=0.0)

    m2 = build()
    m2.configure_devices(devices=["cpu", "cpu"])
    np.random.seed(31)
    out, out_dict = m2.recon(sino, weights=weights, max_iterations=4,
                             stop_threshold_change_pct=0.0)
    assert isinstance(out, np.ndarray) and out.shape == ref.shape
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    fm1 = np.array(ref_dict['recon_params']['fm_rmse'])
    fm2 = np.array(out_dict['recon_params']['fm_rmse'])
    print(f"sharded vcd: recon rel_max {rel:.2e}, fm diff "
          f"{np.max(np.abs(fm1 - fm2)):.2e}")
    assert rel < 5e-4, rel
    rel_fm = np.max(np.abs(fm2 - fm1)) / max(np.max(np.abs(fm1)), 1e-30)
    assert rel_fm < 1e-4, rel_fm

    # Unweighted path (constant weights -> sharded ones Hessian seam).
    m3 = build()
    m3.configure_devices(devices=["cpu", "cpu"])
    np.random.seed(31)
    out_u, _ = m3.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)
    m4 = build()
    np.random.seed(31)
    ref_u, _ = m4.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)
    rel_u = np.max(np.abs(out_u - ref_u)) / max(np.max(np.abs(ref_u)), 1e-30)
    assert rel_u < 5e-4, rel_u


def test_placements_refresh_on_geometry_change():
    # Panel finding: a geometry-changing set_params after configure_devices
    # left the placements' axis lengths stale, and the placement functions silently
    # TRUNCATED sharded arrays.  The recompile hook now rebuilds placements
    # from the current shapes.
    import mbirtorch
    m = mbirtorch.ParallelBeamModel((8, 6, 8), np.linspace(0, np.pi, 8,
                                    endpoint=False))
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    m.configure_devices(devices=["cpu", "cpu"])
    m.set_params(sinogram_shape=(12, 10, 8),
                 angles=np.linspace(0, np.pi, 12, endpoint=False))
    m.auto_set_recon_geometry()
    assert m.sino_placement.axis_len == 12
    assert m.recon_placement.axis_len == m.get_params('recon_shape')[2]
    sino = np.random.RandomState(0).rand(12, 10, 8).astype(np.float32)
    sh = m._shard_sinogram(sino)
    # Splitting and gathering only copies, so the round trip is exact.
    assert np.array_equal(m._gather_sinogram(sh), sino)


def test_cone_sharded_fdk_matches_single_device():
    # Panel finding: sharded cone FDK crashed (Shards.shape in the filter
    # preamble; helical z-weight on the container).  Circular and helical
    # sharded FDK now match single-device values.
    import mbirtorch
    cell = (8, 8, 8)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    for shifts in (None, np.linspace(-2.0, 2.0, cell[0]).astype(np.float32)):
        m = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=32,
                                    source_iso_dist=16,
                                    helical_z_shifts=shifts)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        sino = np.random.RandomState(1).rand(*cell).astype(np.float32)
        ref = m.recon_fdk(sino)
        m.configure_devices(devices=["cpu", "cpu"])
        out = m.recon_fdk(sino)
        assert isinstance(out, np.ndarray)
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
        assert rel < 1e-5, (shifts is None, rel)


def test_uneven_sharded_vcd_recon_matches_single_device():
    # The decisive uneven gate: a seeded recon with non-dividing views AND
    # slices on two virtual CPUs reproduces the single-device run (weighted
    # and constant-weight paths).
    import mbirtorch
    sino_shape = (9, 7, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build():
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        return m

    m1 = build()
    rs = tuple(m1.get_params('recon_shape'))
    phantom = np.zeros(rs, dtype=np.float32)
    phantom[1:-1, 1:-1, 1:-1] = 1.0
    sino = m1.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / np.max(sino),
                                    weight_type='transmission_root')
    np.random.seed(47)
    ref, ref_dict = m1.recon(sino, weights=weights, max_iterations=4,
                             stop_threshold_change_pct=0.0)

    m2 = build()
    m2.configure_devices(devices=["cpu", "cpu"])
    np.random.seed(47)
    out, out_dict = m2.recon(sino, weights=weights, max_iterations=4,
                             stop_threshold_change_pct=0.0)
    assert out.shape == ref.shape
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    fm1 = np.array(ref_dict['recon_params']['fm_rmse'])
    fm2 = np.array(out_dict['recon_params']['fm_rmse'])
    print(f"uneven sharded vcd: recon rel_max {rel:.2e}, fm diff "
          f"{np.max(np.abs(fm1 - fm2)):.2e}")
    assert rel < 5e-4, rel
    rel_fm = np.max(np.abs(fm2 - fm1)) / max(np.max(np.abs(fm1)), 1e-30)
    assert rel_fm < 1e-4, rel_fm

    # Constant weights: the ones-Hessian seam.
    m3 = build()
    np.random.seed(48)
    ref_u, _ = m3.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)
    m4 = build()
    m4.configure_devices(devices=["cpu", "cpu"])
    np.random.seed(48)
    out_u, _ = m4.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)
    rel_u = np.max(np.abs(out_u - ref_u)) / max(np.max(np.abs(ref_u)), 1e-30)
    assert rel_u < 5e-4, rel_u

    # The layout underneath that run: the 9 views split 5 + 4 and (parallel
    # row<->slice tie) the 7 slices split 4 + 3, and the gathers put both axes
    # back together unchanged, copies only.
    rng = np.random.default_rng(3)
    raw_sino = rng.standard_normal(sino_shape).astype(np.float32)
    prepared = m2.prepare_sino_for_devices(raw_sino)
    shapes = [tuple(t.shape) for t in prepared.tensors]
    assert shapes == [(5, 7, 8), (4, 7, 8)]
    assert np.array_equal(m2._gather_sinogram(prepared), raw_sino)
    # A prepared (device-form) array re-enters _shard_sinogram unchanged.
    assert m2._shard_sinogram(prepared) is prepared
    # Weights ride the same seam.
    _, w = m2.prepare_sino_for_devices(raw_sino,
                                       weights=np.abs(raw_sino) + 0.5)
    assert [tuple(t.shape) for t in w.tensors] == shapes
    vol = rng.standard_normal(rs).astype(np.float32)
    placed = m2._shard_recon(vol)
    assert [int(t.shape[-1]) for t in placed.tensors] == [4, 3]
    assert np.array_equal(m2._gather_recon(placed), vol)

    # Forward and back through the banded drivers on the same non-dividing
    # axes equal the single-device values.
    fwd_ref = m1.forward_project(vol)
    fwd_2 = m2._gather_sinogram(m2.forward_project(vol, output_sharded=True))
    rel_f = np.max(np.abs(fwd_2 - fwd_ref)) / np.max(np.abs(fwd_ref))
    assert rel_f < 1e-5, rel_f
    bp_ref = m1.back_project(raw_sino)
    bp_dev = m2.back_project(raw_sino, output_sharded=True)
    # Each device holds its own share of the slice axis: 4 + 3.
    assert [int(t.shape[-1]) for t in bp_dev.tensors] == [4, 3]
    rel_b = (np.max(np.abs(m2._gather_recon(bp_dev) - bp_ref))
             / np.max(np.abs(bp_ref)))
    assert rel_b < 1e-5, rel_b


def test_five_views_and_five_slices_configure_and_recon_on_four_devices():
    """A layout the padded split refused and the balanced split admits.

    Under the pad, five views and five slices over four devices left the last
    device with two padded views and two padded slices, so it owned no real
    data on either axis and the empty-shard rule refused the layout.  The
    balanced split gives that device one view and one slice instead, so the
    layout is legal.  This is the one user-visible change in behavior from
    removing the pad.  No other test reaches it, because a layout the old rule
    refused could not appear in the suite at all.

    The configure_devices call below is the acceptance assertion, since it
    raises on a refused layout.  The recon that follows shows that the layout
    also reproduces the single-device values.
    """
    import mbirtorch
    sino_shape = (5, 5, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build():
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        return m

    m1 = build()
    rs = tuple(m1.get_params('recon_shape'))
    assert rs[2] == 5                    # parallel beam ties slices to rows
    phantom = np.zeros(rs, dtype=np.float32)
    phantom[1:-1, 1:-1, 1:-1] = 1.0
    sino = m1.forward_project(phantom)
    np.random.seed(83)
    ref, _ = m1.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)

    m4 = build()
    m4.configure_devices(devices=["cpu"] * 4)
    # Both axes split 2, 1, 1, 1, so no device is empty on either axis.
    assert [e - s for _d, (s, e)
            in m4.sino_placement.shard_ranges()] == [2, 1, 1, 1]
    assert [e - s for _d, (s, e)
            in m4.recon_placement.shard_ranges()] == [2, 1, 1, 1]
    np.random.seed(83)
    out, _ = m4.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    print(f"five views and five slices on four devices: rel {rel:.2e}")
    assert rel < 5e-4, rel


def test_cone_sharded_vcd_recon_matches_single_device():
    # The cone VCD loop on two devices: the DC-damping profile now splits per
    # shard (dev_index seam), so the multi-device guard is gone.  A seeded
    # recon at a dividing cell and at a non-dividing cell reproduces the
    # single-device run.
    import mbirtorch
    for cell in ((8, 8, 8), (9, 7, 8)):
        angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)

        def build():
            m = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=32,
                                        source_iso_dist=16)
            m.configure_devices(devices=["cpu"])
            m.set_params(no_warning=True, verbose=0)
            return m

        m1 = build()
        rs = tuple(m1.get_params('recon_shape'))
        phantom = np.zeros(rs, dtype=np.float32)
        phantom[1:-1, 1:-1, 1:-1] = 1.0
        sino = m1.forward_project(phantom)
        np.random.seed(53)
        ref, _ = m1.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0)

        m2 = build()
        m2.configure_devices(devices=["cpu", "cpu"])
        np.random.seed(53)
        out, _ = m2.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0)
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
        print(f"cone sharded vcd {cell}: rel_max {rel:.2e}")
        assert rel < 5e-4, (cell, rel)


def test_sub_band_streaming_matches_unstreamed():
    # Force 2-slice sub-bands through the banded back driver (the default
    # bounds give one band at test sizes) on a non-dividing parallel cell AND
    # a cone cell: values must match the single-device references at the
    # sharded drivers' established tolerances -- streaming is a pure partition
    # of the work.  The forward, which transfers whole cylinders and walks no
    # bands, is checked on the same models and must not move either.
    import mbirtorch
    sino_shape = (9, 7, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    m1 = mbirtorch.ParallelBeamModel(sino_shape, angles)
    m1.configure_devices(devices=["cpu"])
    m1.set_params(no_warning=True, verbose=0)
    rs = tuple(m1.get_params('recon_shape'))
    rng = np.random.default_rng(9)
    vol = rng.standard_normal(rs).astype(np.float32)
    sino = rng.standard_normal(sino_shape).astype(np.float32)
    fwd_ref = m1.forward_project(vol)
    bp_ref = m1.back_project(sino)

    m2 = mbirtorch.ParallelBeamModel(sino_shape, angles)
    m2.configure_devices(devices=["cpu"])
    m2.set_params(no_warning=True, verbose=0)
    m2.configure_devices(devices=["cpu", "cpu"])
    m2.back_project_slice_band = 2
    fwd = m2._gather_sinogram(m2.forward_project(vol, output_sharded=True))
    bp = m2._gather_recon(m2.back_project(sino, output_sharded=True))
    assert np.max(np.abs(fwd - fwd_ref)) / np.max(np.abs(fwd_ref)) < 1e-5
    assert np.max(np.abs(bp - bp_ref)) / np.max(np.abs(bp_ref)) < 1e-5

    cell = (8, 8, 8)
    cangles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    c1 = mbirtorch.ConeBeamModel(cell, cangles, source_detector_dist=32,
                                 source_iso_dist=16)
    c1.configure_devices(devices=["cpu"])
    c1.set_params(no_warning=True, verbose=0)
    crs = tuple(c1.get_params('recon_shape'))
    cvol = rng.standard_normal(crs).astype(np.float32)
    csino = rng.standard_normal(cell).astype(np.float32)
    cfwd_ref = c1.forward_project(cvol)
    cbp_ref = c1.back_project(csino)
    c2 = mbirtorch.ConeBeamModel(cell, cangles, source_detector_dist=32,
                                 source_iso_dist=16)
    c2.configure_devices(devices=["cpu"])
    c2.set_params(no_warning=True, verbose=0)
    c2.configure_devices(devices=["cpu", "cpu"])
    c2.back_project_slice_band = 2
    cfwd = c2._gather_sinogram(c2.forward_project(cvol, output_sharded=True))
    cbp = c2._gather_recon(c2.back_project(csino, output_sharded=True))
    assert np.max(np.abs(cfwd - cfwd_ref)) / np.max(np.abs(cfwd_ref)) < 1e-5
    assert np.max(np.abs(cbp - cbp_ref)) / np.max(np.abs(cbp_ref)) < 1e-5


def test_thin_volume_more_devices_than_slices():
    # The thin-volume extension (beyond mbirjax): more devices than slices is
    # a legal layout -- the extra devices carry views (the dominant compute
    # and memory) while their slice shards hold no slices at all.
    # Seeded n=4 vcd vs n=1 on a 3-slice parallel cell and a 3-row cone cell.
    import mbirtorch
    sino_shape = (16, 3, 16)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build():
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        return m

    m1 = build()
    rs = tuple(m1.get_params('recon_shape'))
    assert rs[2] == 3
    phantom = np.zeros(rs, dtype=np.float32)
    phantom[1:-1, 1:-1, :] = 1.0
    sino = m1.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / np.max(sino),
                                    weight_type='transmission_root')
    np.random.seed(61)
    ref, _ = m1.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)

    m2 = build()
    m2.configure_devices(devices=["cpu"] * 4)   # 4 devices, 3 slices
    assert [e - s for _d, (s, e) in m2.recon_placement.shard_ranges()][-1] == 0
    np.random.seed(61)
    out, _ = m2.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    print(f"thin parallel n4 vs n1: rel {rel:.2e}")
    assert rel < 5e-4, rel
    # The device form of the same run, one iteration long.  The fourth device
    # owns no slices, so its block is zero-length on the slice axis and full
    # size on the other two.  The three devices that do own slices carry the
    # single-device values.
    np.random.seed(63)
    ref1, _ = m1.recon(sino, weights=weights, max_iterations=1,
                       stop_threshold_change_pct=0.0)
    np.random.seed(63)
    dev_recon, _ = m2.recon(sino, weights=weights, max_iterations=1,
                            stop_threshold_change_pct=0.0, output_sharded=True)
    assert tuple(dev_recon.tensors[-1].shape) == (rs[0], rs[1], 0)
    assert dev_recon.tensors[-1].dtype == dev_recon.tensors[0].dtype
    rel1 = (np.max(np.abs(m2._gather_recon(dev_recon) - ref1))
            / max(np.max(np.abs(ref1)), 1e-30))
    assert rel1 < 5e-4, rel1

    cell = (8, 3, 8)
    cangles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)

    def cbuild():
        m = mbirtorch.ConeBeamModel(cell, cangles, source_detector_dist=32,
                                    source_iso_dist=16)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        return m

    c1 = cbuild()
    crs = tuple(c1.get_params('recon_shape'))
    cph = np.zeros(crs, dtype=np.float32)
    cph[1:-1, 1:-1, :] = 1.0
    csino = c1.forward_project(cph)
    np.random.seed(62)
    cref, _ = c1.recon(csino, max_iterations=2, stop_threshold_change_pct=0.0)
    c2 = cbuild()
    c2.configure_devices(devices=["cpu"] * 4)
    assert [e - s for _d, (s, e) in c2.recon_placement.shard_ranges()][-1] == 0
    np.random.seed(62)
    cout, _ = c2.recon(csino, max_iterations=2, stop_threshold_change_pct=0.0)
    crel = np.max(np.abs(cout - cref)) / max(np.max(np.abs(cref)), 1e-30)
    print(f"thin cone n4 vs n1: rel {crel:.2e}")
    assert crel < 5e-4, crel


def test_sparse_view_more_devices_than_views():
    # The sparse-view extension: more devices than views is a legal layout --
    # the extra devices hold slice shards and run the prior and updates (the
    # dominant work with few views) while their view shards hold no views at
    # all.  Seeded n=4 vcd vs n=1 on a 3-view parallel cell and a 3-view cone
    # cell.
    import mbirtorch
    sino_shape = (3, 16, 16)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build():
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        return m

    m1 = build()
    rs = tuple(m1.get_params('recon_shape'))
    phantom = np.zeros(rs, dtype=np.float32)
    phantom[1:-1, 1:-1, 1:-1] = 1.0
    sino = m1.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / np.max(sino),
                                    weight_type='transmission_root')
    np.random.seed(71)
    ref, _ = m1.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)

    m2 = build()
    m2.configure_devices(devices=["cpu"] * 4)   # 4 devices, 3 views
    assert [e - s for _d, (s, e) in m2.sino_placement.shard_ranges()][-1] == 0
    np.random.seed(71)
    out, _ = m2.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    print(f"sparse-view parallel n4 vs n1: rel {rel:.2e}")
    # Tolerance calibrated to THIS cell's inherent multi-device floor: with
    # DIVIDING views (16), the same cell reads
    # 1.8e-3 at n=2 and 1.9e-3 at n=4 (staged-halo staleness within a
    # partition pass plus cross-device float reorders -- the mbirjax
    # structure).  The sparse-view layout adds nothing beyond that floor.
    assert rel < 5e-3, rel
    # The fourth device owns no views, so its sinogram block is zero-length on
    # the view axis and full size on the detector axes.  The three devices that
    # do own views carry the single-device values.  A forward projection is a
    # single pass through the projectors, so it gates at 1e-5 rather than at
    # the iterated level above.
    fwd_dev = m2.forward_project(phantom, output_sharded=True)
    assert tuple(fwd_dev.tensors[-1].shape) == (0,) + sino_shape[1:]
    assert fwd_dev.tensors[-1].dtype == fwd_dev.tensors[0].dtype
    rel_fwd = (np.max(np.abs(m2._gather_sinogram(fwd_dev) - sino))
               / max(np.max(np.abs(sino)), 1e-30))
    assert rel_fwd < 1e-5, rel_fwd

    cell = (3, 8, 8)
    cangles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)

    def cbuild():
        m = mbirtorch.ConeBeamModel(cell, cangles, source_detector_dist=32,
                                    source_iso_dist=16)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        return m

    c1 = cbuild()
    crs = tuple(c1.get_params('recon_shape'))
    cph = np.zeros(crs, dtype=np.float32)
    cph[1:-1, 1:-1, 1:-1] = 1.0
    csino = c1.forward_project(cph)
    np.random.seed(72)
    cref, _ = c1.recon(csino, max_iterations=2, stop_threshold_change_pct=0.0)
    c2 = cbuild()
    c2.configure_devices(devices=["cpu"] * 4)
    assert [e - s for _d, (s, e) in c2.sino_placement.shard_ranges()][-1] == 0
    np.random.seed(72)
    cout, _ = c2.recon(csino, max_iterations=2, stop_threshold_change_pct=0.0)
    crel = np.max(np.abs(cout - cref)) / max(np.max(np.abs(cref)), 1e-30)
    print(f"sparse-view cone n4 vs n1: rel {crel:.2e}")
    assert crel < 5e-3, crel   # same calibration as the parallel case above


# ── the forward's cylinder transfer ──────────────────────────────────────────
# What may FAIL here, and what may only be recorded.  The value bar these
# tests hold is the one the library already ships: the kernel-parity floor the
# suites above enforce, at the 1e-5 relative these cone cases use on CPU.  The
# multi-GPU measurement also registered an EXPECTATION beside that floor.  The
# cylinder transfer sat about 1.5e-06 relative from the one-device anchor at
# the 1024-class cell, measured 2026-08-10 on four H100s in job mg10.  That
# expectation is recorded so a later reading well outside it is visible to a
# human; it is deliberately NOT a threshold, and nothing here asserts it.  The
# distances below are printed for that comparison.  On CPU the runs are
# deterministic and the transfer's calls are the single-device call shape, so
# what these tests do assert is exact-path mechanics rather than that bar.
def _cone_cylinder_case(devices, cell=(8, 8, 8), pixel_batch=None):
    """A cone model on virtual CPU devices, plus its single-device
    reference.  The multi-device forward is the cylinder transfer."""
    m, idx, vals, sino, ref_fwd, ref_back = _cone_banded_case(devices, cell)
    if pixel_batch is not None:
        m.forward_project_pixel_batch = pixel_batch
    return m, idx, vals, sino, ref_fwd, ref_back


def test_the_cylinder_transfer_assembles_the_whole_slice_axis(monkeypatch):
    # The primitive: every slice-owner's rows [p0:p1] moved to one target and
    # concatenated along the SLICE axis, in shard (global slice) order.  A
    # single shard short-circuits the concatenation.
    from mbirtorch import _sharding as sharding
    from mbirtorch._sharding import transfer_cylinder_batch
    rng = np.random.default_rng(11)
    full = torch.as_tensor(rng.standard_normal((9, 6)).astype(np.float32))
    shards = [full[:, 0:2].contiguous(), full[:, 2:4].contiguous(),
              full[:, 4:6].contiguous()]
    cyl = transfer_cylinder_batch(shards, 3, 7, torch.device("cpu"))
    assert cyl.shape == (4, 6)
    assert torch.equal(cyl, full[3:7])
    # A degenerate range is legal and empty; one shard is returned as itself.
    empty = transfer_cylinder_batch(shards, 5, 5, torch.device("cpu"))
    assert empty.shape == (0, 6)
    one = transfer_cylinder_batch(shards[:1], 0, 9, torch.device("cpu"))
    assert torch.equal(one, shards[0])
    # The host-bounce path is value-correct too (dev2dev_safe False).
    bounced = transfer_cylinder_batch(shards, 0, 9, torch.device("cpu"),
                                 dev2dev_safe=False)
    assert torch.equal(bounced, full)

    # The same assembly across two real devices when a second one is present:
    # a transfer copies bytes, so the result is exact.
    if torch.backends.mps.is_available():
        real_full = torch.rand(32, 8)
        real_shards = [real_full[:, :4].contiguous().to("cpu"),
                       real_full[:, 4:].contiguous().to("mps")]
        real_cyl = transfer_cylinder_batch(real_shards, 8, 16,
                                           torch.device("mps"))
        assert real_cyl.device.type == "mps"
        assert torch.equal(real_cyl.cpu(), real_full[8:16])

    # The mechanics witness.  The cone forward must call
    # transfer_cylinder_batch; each transfer takes one piece per
    # slice-owner and yields cylinders that are the batch wide and the
    # WHOLE slice axis tall; and each projector call runs at
    # slice_start=0 over that whole axis for the owner's own views.
    batch, n = 4, 2
    m, idx, vals, _sino, _ref_fwd, _ref_back = _cone_cylinder_case(
        ["cpu"] * n, pixel_batch=batch)
    slices = m.recon_placement.axis_len
    transfers, calls = [], []
    real_gather = sharding.transfer_cylinder_batch

    def spy_gather(shard_tensors, p0, p1, target, dev2dev_safe=True):
        out = real_gather(shard_tensors, p0, p1, target, dev2dev_safe)
        transfers.append((len(shard_tensors), p0, p1, tuple(out.shape)))
        return out

    real_call = m.projector_functions.sparse_forward_project_view_range

    def spy_call(band_values, pixel_indices, view_range, slice_start=0,
                 dev_index=0, plan=None, accumulate_into=None):
        calls.append((tuple(band_values.shape), int(pixel_indices.shape[0]),
                      tuple(view_range), slice_start))
        return real_call(band_values, pixel_indices, view_range,
                         slice_start=slice_start, dev_index=dev_index,
                         plan=plan, accumulate_into=accumulate_into)

    monkeypatch.setattr(sharding, "transfer_cylinder_batch", spy_gather)
    monkeypatch.setattr(m.projector_functions,
                        "sparse_forward_project_view_range", spy_call)
    m.sparse_forward_project(vals, idx)

    expected_batches = -(-len(idx) // batch)
    assert len(transfers) == n * expected_batches
    for pieces, p0, p1, shape in transfers:
        assert pieces == n                      # one piece per slice-owner
        assert shape == (p1 - p0, slices)       # the batch, at every slice
        assert p1 - p0 <= batch
    # One projector call per (pixel batch, view-owner), each over the whole
    # slice axis anchored at 0 and over that owner's own views.
    assert len(calls) == n * expected_batches
    spans = [(v0, v1) for _, _, (v0, v1), _ in calls]
    for cyl_shape, n_pixels, (v0, v1), slice_start in calls:
        assert slice_start == 0 and cyl_shape[1] == slices
        assert n_pixels == cyl_shape[0] and v1 > v0
    assert set(spans) == {span for _d, span in m.sino_placement.shard_ranges()
                          if span[1] > span[0]}


def test_cylinder_transfer_matches_single_device_at_every_batch():
    # The values gate on virtual CPU devices: a full-height call at
    # slice_start=0 is the single-device call shape, so the sharded forward
    # must reproduce the single-device values -- at one batch covering the
    # pass, and at batches that force several.
    for n in (2, 3):
        for batch in (None, 1, 5, 10 ** 6):
            m, idx, vals, _sino, ref_fwd, _ref_back = _cone_cylinder_case(
                ["cpu"] * n, pixel_batch=batch)
            fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))
            rel = np.max(np.abs(fwd - ref_fwd)) / np.max(np.abs(ref_fwd))
            print(f"cone cylinder transfer n={n} batch={batch}: rel {rel:.2e}")
            assert rel < 1e-5, (n, batch, rel)


def test_cylinder_transfer_holds_the_adjoint_on_uneven_axes():
    # The back driver walks slice bands where the forward moves whole
    # cylinders, so
    # the pair must stay adjoint -- on a cell whose axes do not divide (9
    # views and 7 slices over 2 devices), where the shards differ in length.
    m, idx, vals, sino, ref_fwd, ref_back = _cone_cylinder_case(
        ["cpu", "cpu"], cell=(9, 7, 8), pixel_batch=4)
    fwd = m.sparse_forward_project(vals, idx)
    back = m.sparse_back_project(sino, idx)
    rel = (np.max(np.abs(m._gather_sinogram(fwd) - ref_fwd))
           / max(np.max(np.abs(ref_fwd)), 1e-30))
    assert rel < 1e-5, rel
    rel_b = (np.max(np.abs(back.gather() - ref_back))
             / max(np.max(np.abs(ref_back)), 1e-30))
    assert rel_b < 1e-5, rel_b
    lhs = float(np.sum(m._gather_sinogram(fwd) * sino))
    rhs = float(np.sum(vals * back.gather()))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (lhs, rhs)


def test_cylinder_transfer_recon_matches_single_device():
    # The end-to-end gate: a seeded cone reconstruction on two virtual CPU
    # devices must reproduce the single-device run, which is where the two
    # summation orders the transfer changed (the vertical sum into the
    # body, the
    # pixel sum out of it) would show up if they were not inside the value
    # class the forward already has.
    import mbirtorch
    cell = (8, 8, 8)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)

    def build(devices):
        m = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=32,
                                    source_iso_dist=16)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        if len(devices) > 1:
            m.configure_devices(devices=devices)
        return m

    m1 = build(["cpu"])
    rs = tuple(m1.get_params('recon_shape'))
    phantom = np.zeros(rs, dtype=np.float32)
    phantom[1:-1, 1:-1, 1:-1] = 1.0
    sino = m1.forward_project(phantom)
    np.random.seed(31)
    ref, _ = m1.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)

    sharded = build(["cpu", "cpu"])
    sharded.forward_project_pixel_batch = 8
    np.random.seed(31)
    out, _ = sharded.recon(sino, max_iterations=2,
                            stop_threshold_change_pct=0.0)
    scale = max(np.max(np.abs(ref)), 1e-30)
    rel = np.max(np.abs(out - ref)) / scale
    # Printed as well as asserted: a later reading well above the registered
    # expectation is worth a human's attention before it reaches the floor.
    print(f"cone recon vs n1: cylinder transfer {rel:.2e}")
    assert rel < 5e-3, rel     # the shipped parity floor, as above


# ── the same transfer, on the row-aligned geometry ───────────────────────────
# Parallel takes the cylinder transfer for a different measured reason
# than cone.  It CAN produce its detector rows from a slice band, but
# its forward kernel runs about twice as efficiently per slice on the
# full-width block of values the transfer hands it as on a shard-width
# block (measured 2026-08-10 on one
# H100, at 0.0411 ms per slice on a 1008-wide block against 0.0823 on a
# 504-wide one, with the device count held at one).
#
# The value bar was expected to be EQUALITY here, on the argument that each
# detector row keeps a single producing call and CPU sums are deterministic.
# The row half of that is true.  Equality is not, and the measurement that
# settled it is recorded
# because it is worth knowing before anyone tries again (2026-08-10, this
# suite, virtual CPU devices).  Run first in a fresh interpreter, the cylinder
# transfer reproduces the single-device sinogram bit for bit.  Run once other
# shapes have gone through the same per-device bodies -- which is what a full
# suite run does -- it lands in the float32 epsilon class instead, at 1.1e-07
# to 4.0e-07 over these cells.  The cause: the per-device bodies are
# separately torch.compiled, and what a compiled body emits depends on the
# shapes its instance has already seen, so two devices can differ in the last
# bit on identical inputs.  Bit-equality is therefore a property of the
# process, not of the driver shape, and these tests hold the transfer to a
# relative bar instead.
def _parallel_cylinder_case(devices, sino_shape=(8, 6, 8), pixel_batch=None):
    """A parallel model on virtual CPU devices, plus its single-device
    reference.  The multi-device forward is the cylinder transfer."""
    m, idx, vals, sino, ref_fwd, ref_back, _b2 = _banded_case(devices,
                                                              sino_shape)
    if pixel_batch is not None:
        m.forward_project_pixel_batch = pixel_batch
    return m, idx, vals, sino, ref_fwd, ref_back


def test_parallel_cylinder_transfer_matches_the_single_device_values():
    # The values gate.  One call per view-owner over every pixel is the
    # single-device call in every respect that sets a value: the same voxel
    # columns in one array, the whole slice range anchored at 0, and each
    # detector row produced by that one call and no other.  A row taking
    # contributions from more than one call would show up here as an
    # order-one error, not as a last bit.
    for n in (2, 3):
        # None takes the shipped batch, which covers a pass this size in one
        # call; the large value asks for that explicitly.
        for batch in (None, 10 ** 6):
            m, idx, vals, _sino, ref_fwd, _rb = _parallel_cylinder_case(
                ["cpu"] * n, pixel_batch=batch)
            fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))
            scale = np.max(np.abs(ref_fwd))
            rel = np.max(np.abs(fwd - ref_fwd)) / scale
            print(f"parallel cylinder transfer n={n} batch={batch}: "
                  f"rel {rel:.2e}")
            assert rel < 1e-5, (n, batch, rel)

    # The one summation order this shape does change for a row-aligned
    # geometry: several pixel batches turn a single accumulation over every
    # pixel into a host-side sum of per-batch partials.  Nothing about the
    # rows moves, so what is left is float noise in the same class as above
    # (measured 1.0e-07 to 1.6e-07 here).  This is the case that runs at
    # production sizes, where the pass is far wider than one batch.
    for batch in (1, 5, 7):
        m, idx, vals, _sino, ref_fwd, _rb = _parallel_cylinder_case(
            ["cpu", "cpu"], pixel_batch=batch)
        fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))
        rel = np.max(np.abs(fwd - ref_fwd)) / np.max(np.abs(ref_fwd))
        print(f"parallel cylinder transfer, {batch}-pixel batches: "
              f"rel {rel:.2e}")
        assert rel < 1e-5, (batch, rel)

    # Two layouts a row-aligned geometry has to assemble correctly: one whose
    # axes do not divide the device count, and one with more devices than
    # views.  Every block this driver assembles -- including the empty one it
    # builds for a view-owner with no views -- has to carry the detector row
    # count, or the shards do not concatenate at all.
    for shape, devs in (((9, 7, 8), 2),        # neither axis divides
                        ((3, 7, 8), 4)):       # an owner with no views
        m, idx, vals, sino, ref_fwd, ref_back = _parallel_cylinder_case(
            ["cpu"] * devs, sino_shape=shape, pixel_batch=10 ** 6)
        real_rows = shape[1]
        fwd = m.sparse_forward_project(vals, idx)
        assert all(t.shape[1] == real_rows for t in fwd.tensors), shape
        rel = (np.max(np.abs(m._gather_sinogram(fwd) - ref_fwd))
               / max(np.max(np.abs(ref_fwd)), 1e-30))
        assert rel < 1e-5, (shape, rel)
        # The back driver is untouched, so the pair stays adjoint.
        back = m.sparse_back_project(sino, idx)
        rel_b = (np.max(np.abs(back.gather() - ref_back))
                 / max(np.max(np.abs(ref_back)), 1e-30))
        assert rel_b < 1e-5, (shape, rel_b)
        lhs = float(np.sum(m._gather_sinogram(fwd) * sino))
        rhs = float(np.sum(vals * back.gather()))
        assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (shape, lhs, rhs)

    # The accumulation the multi-batch passes above rely on, taken at the
    # projector itself: handed a block it adds into that block and hands back
    # the same object, so accumulating one call's values onto another's
    # doubles them exactly.  torch's CPU scatter reduces in parallel and is
    # therefore not reproducible run to run, so the threads are pinned to one
    # while two evaluations of the same body are compared.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        m, idx, vals = _banded_case(["cpu"])[:3]
        pf = m.projector_functions
        num_views = int(m.get_params('sinogram_shape')[0])
        t_vals = torch.as_tensor(vals)
        t_idx = torch.as_tensor(idx, dtype=torch.int64)
        once = pf.sparse_forward_project_view_range(t_vals, t_idx,
                                                    (0, num_views))
        running = pf.sparse_forward_project_view_range(t_vals, t_idx,
                                                       (0, num_views))
        assert torch.equal(once, running)         # the control
        same = pf.sparse_forward_project_view_range(t_vals, t_idx,
                                                    (0, num_views),
                                                    accumulate_into=running)
        assert same is running
        assert torch.equal(running, once + once)
    finally:
        torch.set_num_threads(threads)


def test_parallel_cylinder_transfer_recon_matches_single_device():
    # The end-to-end gate, where the subset passes call the forward on small
    # pixel sets and the pixel batch above therefore bites: a seeded parallel
    # reconstruction on two virtual CPU devices must reproduce the
    # single-device run within the loop's own multi-device floor.
    import mbirtorch
    sino_shape = (8, 6, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build(devices):
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        if len(devices) > 1:
            m.configure_devices(devices=devices)
        return m

    m1 = build(["cpu"])
    rs = tuple(m1.get_params('recon_shape'))
    phantom = np.zeros(rs, dtype=np.float32)
    phantom[1:-1, 1:-1, 1:-1] = 1.0
    sino = m1.forward_project(phantom)
    np.random.seed(31)
    ref, _ = m1.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0)

    sharded = build(["cpu", "cpu"])
    sharded.forward_project_pixel_batch = 8
    np.random.seed(31)
    out, _ = sharded.recon(sino, max_iterations=3,
                            stop_threshold_change_pct=0.0)
    scale = max(np.max(np.abs(ref)), 1e-30)
    rel = np.max(np.abs(out - ref)) / scale
    print(f"parallel recon vs n1: cylinder transfer {rel:.2e}")
    assert rel < 5e-4, rel     # the sharded VCD loop's own floor at this cell


# ── the same transfer, on the two geometries with no hand-written kernels ────
# Translation and multiaxis have the same band-independent per-call cost cone
# has, so the shape was expected to help them, and on 2026-08-17 it was
# measured on four H100s at each geometry's production cell rather than
# argued.  The transfer was faster at every device count than the slice-banded
# walk it replaced: the multiaxis forward 1.27x at two devices and 1.86x at
# four, its composed reconstruction 1.13x and 1.20x; the translation forward
# 1.86x and 25.4x, its composed reconstruction 1.37x and 1.94x.  The
# translation four-device figure is that large because the banded walk there
# ran slower than one device does.  Per-device peak memory was lower at the
# shipped pixel batch on every arm, and every value sat between 9e-7 and
# 2.5e-5 from the one-device reference against a 1e-3 gate.  Both defaults
# moved with that reading, and the banded walk was removed on 2026-08-17.
#
# The bar here is the one the parallel and cone cases use: the 1e-5 relative
# these CPU cases already enforce.  The recorded caveat above the parallel
# section applies unchanged -- bit-equality is a property of the process's
# compile state, not of the driver shape.
@pytest.mark.parametrize('geometry', ('multiaxis', 'translation'))
def test_torch_body_geometries_take_the_gather(geometry):
    case = {'multiaxis': _multiaxis_banded_case,
            'translation': _translation_banded_case}[geometry]
    # Two and three devices: at three, both sharded axes split unevenly (see
    # the case helpers), which is where a driver that assumes equal blocks
    # would show up.
    for n in (2, 3):
        m, idx, vals, _sino, ref_fwd, _rb = case(["cpu"] * n)
        out = m._gather_sinogram(m.sparse_forward_project(vals, idx))
        scale = max(np.max(np.abs(ref_fwd)), 1e-30)
        rel = np.max(np.abs(out - ref_fwd)) / scale
        print(f"{geometry} cylinder transfer n={n}: rel {rel:.2e}")
        assert rel < 1e-5, (geometry, n, rel)

    # Several pixel batches, which is what a production pass runs: the single
    # accumulation over every pixel becomes a sum of per-batch partials.  The
    # back driver walks slice bands, so the pair still has to be adjoint.
    for batch in (1, 5):
        m, idx, vals, sino, ref_fwd, ref_back = case(["cpu", "cpu"])
        m.forward_project_pixel_batch = batch
        fwd = m.sparse_forward_project(vals, idx)
        rel = (np.max(np.abs(m._gather_sinogram(fwd) - ref_fwd))
               / max(np.max(np.abs(ref_fwd)), 1e-30))
        print(f"{geometry} cylinder transfer, {batch}-pixel batches: "
              f"rel {rel:.2e}")
        assert rel < 1e-5, (geometry, batch, rel)
        back = m.sparse_back_project(sino, idx)
        rel_b = (np.max(np.abs(back.gather() - ref_back))
                 / max(np.max(np.abs(ref_back)), 1e-30))
        assert rel_b < 1e-5, (geometry, batch, rel_b)
        lhs = float(np.sum(m._gather_sinogram(fwd) * sino))
        rhs = float(np.sum(vals * back.gather()))
        assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (geometry, batch,
                                                              lhs, rhs)


# ── one pixel at a time ──────────────────────────────────────────────────────
# The cylinder transfer's pixel batching hands the projectors a one-pixel call
# whenever a batch, or the remainder of a batch, is a single pixel, and a user
# can ask for one directly.  On linux with torch 2.13.0, CPU inductor
# miscompiles exactly that case in both parallel bodies and lands the pixel's
# footprint one detector channel off (measured 2026-08-11: 6.56e-02 relative on
# the forward, 5.04e-02 on the back, on the 8x6x8 cell below; eager is right,
# and so is every width of two or more).  The driver pads a one-pixel call to
# two and takes the padding back out.  These two tests hold that: the first is
# the property a user cares about, the second is the padding itself.  Both pass
# on any machine whose compiler is sound -- macOS is one -- so their value is
# the linux nightly.
def test_parallel_solo_pixel_projections_match_the_full_pass():
    # A pixel projects the same whether it is asked for alone or with the
    # others.  Forward: the projections of the single pixels sum to the whole
    # pass, because the forward is linear in the voxels and each pixel writes
    # its own footprint into the same sinogram.  Back: one pixel's cylinder is
    # that pixel's row of the whole pass, computed from the same sinogram.  A
    # body that reads a one-pixel call differently shows up here as an
    # order-one error, not as a last bit.
    m, idx, vals, sino, ref_fwd, ref_back, _b2 = _banded_case(["cpu"])
    solo_fwd = np.zeros_like(ref_fwd)
    for i in range(len(idx)):
        solo_fwd += m.sparse_forward_project(vals[i:i + 1],
                                             idx[i:i + 1]).cpu().numpy()
    rel = np.max(np.abs(solo_fwd - ref_fwd)) / np.max(np.abs(ref_fwd))
    print(f"parallel solo-pixel forward sum: rel {rel:.2e}")
    assert rel < 1e-5, rel

    for i in (0, 1, len(idx) // 2, len(idx) - 1):
        row = m.sparse_back_project(sino, idx[i:i + 1]).cpu().numpy()
        assert row.shape == (1, ref_back.shape[1])
        rel_back = (np.max(np.abs(row[0] - ref_back[i]))
                    / np.max(np.abs(ref_back[i])))
        print(f"parallel solo-pixel back, pixel {i}: rel {rel_back:.2e}")
        assert rel_back < 1e-5, (i, rel_back)


def test_the_minimum_pixel_width_padding_keeps_the_values():
    # The padding itself, held against the eager bodies it must agree with.
    # The forward's padded column carries zero values at a repeated pixel
    # index, and the forward output has no pixel axis, so the padded call is
    # bit-identical and nothing is sliced off.  The back's output does carry
    # the pixel axis, so the padded call's extra row is sliced away and the row
    # that stays must be exact, not close -- at both coefficient powers.  A
    # call that is already wide enough goes through untouched.
    from mbirtorch import ConeBeamModel, projectors
    from mbirtorch.parallel_beam import (ParallelBeamModel,
                                         _parallel_back_view_batch,
                                         _parallel_forward_view_batch)
    # Declared by the geometry whose bodies need it, and by no other.
    assert ParallelBeamModel.min_compiled_pixel_width == 2
    assert ConeBeamModel.min_compiled_pixel_width == 1

    m, idx, vals, sino, _rf, _rb, _b2 = _banded_case(["cpu"])
    args = m._view_batch_args()
    view_params = torch.as_tensor(np.asarray(m.get_params('angles')),
                                  dtype=torch.float32)
    one_idx = torch.as_tensor(idx[7:8], dtype=torch.int64)
    one_vals = torch.as_tensor(vals[7:8])
    sino_t = torch.as_tensor(sino)
    widths = []

    def spy_forward(values, pixel_indices, *a, **kw):
        widths.append(int(pixel_indices.shape[0]))
        return _parallel_forward_view_batch(values, pixel_indices, *a, **kw)

    def spy_back(sino_batch, pixel_indices, *a, **kw):
        widths.append(int(pixel_indices.shape[0]))
        return _parallel_back_view_batch(sino_batch, pixel_indices, *a, **kw)

    padded_fwd = projectors.forward_at_min_pixel_width(spy_forward, 2)
    padded_back = projectors.back_at_min_pixel_width(spy_back, 2)

    assert torch.equal(
        padded_fwd(one_vals, one_idx, view_params, **args),
        _parallel_forward_view_batch(one_vals, one_idx, view_params, **args))
    assert widths == [2]                     # the body never saw one pixel
    for power in (1, 2):
        wrapped = padded_back(sino_t, one_idx, view_params,
                              coeff_power=power, **args)
        plain = _parallel_back_view_batch(sino_t, one_idx, view_params,
                                          coeff_power=power, **args)
        assert wrapped.shape == plain.shape
        assert torch.equal(wrapped, plain), power
    assert widths == [2, 2, 2]

    all_idx = torch.as_tensor(idx, dtype=torch.int64)
    padded_fwd(torch.as_tensor(vals), all_idx, view_params, **args)
    padded_back(sino_t, all_idx, view_params, **args)
    assert widths[-2:] == [len(idx), len(idx)]

    # And the driver wraps what it compiles, only that: a hand-written kernel
    # body comes back from maybe_compile as itself, cannot be miscompiled, and
    # must keep its identity and its cost attribute.
    pf = m.projector_functions
    raw_fwd, raw_back = m._view_batch_bodies()
    for bound, raw in ((pf._fwd_body_per_dev[0], raw_fwd),
                       (pf._back_body_per_dev[0], raw_back)):
        assert bound.__name__.startswith('padded_') == (bound is not raw)



# ── the divided form at the public entries ───────────────────────────────────
# Some entries take a divided array and keep it divided; others work on one
# whole array and refuse it.

def _two_device_case(sino_shape=(10, 7, 8)):
    """A small parallel-beam model on two 'virtual' CPU devices, with a
    sinogram and a reconstruction in both the whole and the divided form.  The
    7 detector rows give 7 recon slices, which split 4 + 3, so the two shards
    have different local slice counts."""
    import mbirtorch
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=["cpu"])
    model.set_params(no_warning=True, verbose=0)
    model.configure_devices(devices=["cpu", "cpu"])
    rng = np.random.RandomState(11)
    sino = rng.rand(*sino_shape).astype(np.float32)
    recon = rng.rand(*tuple(model.get_params('recon_shape'))).astype(np.float32)
    return model, sino, recon, model._shard_sinogram(sino), model._shard_recon(recon)


def test_get_voxels_at_indices_keeps_a_divided_recon_divided():
    """Selecting cylinders flattens the row and column axes and picks rows, so
    it never touches the slice axis the shards are cut on.  Each shard is
    indexed where it lives, and the assembled result equals the whole-array
    one exactly -- values move, nothing is computed."""
    from mbirtorch._sharding import Shards
    model, _sino, recon, _sino_sh, recon_sh = _two_device_case()
    recon_shape = tuple(model.get_params('recon_shape'))
    assert [int(t.shape[-1]) for t in recon_sh.tensors] == [4, 3]
    indices = torch.as_tensor(
        np.sort(np.random.RandomState(12).choice(
            recon_shape[0] * recon_shape[1], size=11, replace=False)),
        dtype=torch.int64)

    whole = model.get_voxels_at_indices(torch.as_tensor(recon), indices)
    assert np.array_equal(
        whole.numpy(), recon.reshape(-1, recon.shape[-1])[indices.numpy()])

    divided = model.get_voxels_at_indices(recon_sh, indices)
    assert isinstance(divided, Shards)
    assert divided.placement == recon_sh.placement
    assert [int(t.shape[-1]) for t in divided.tensors] == [4, 3]
    assert np.array_equal(divided.gather(), whole.numpy())

    # An index set covering the whole grid is the plain flattening, and it
    # still comes back divided the same way.
    all_indices = torch.arange(recon_shape[0] * recon_shape[1], dtype=torch.int64)
    flat = model.get_voxels_at_indices(recon_sh, all_indices)
    assert np.array_equal(flat.gather(), recon.reshape(-1, recon.shape[-1]))
