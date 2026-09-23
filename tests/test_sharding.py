"""Multi-device units: Placement, Shards, the forward's cylinder transfer, the
back projection's band reduce, and the sharded reconstructions built on them.
Runs everywhere: the placement and banding logic uses two 'virtual' CPU
devices (['cpu', 'cpu'] -- transfers are no-ops but every range, pad, mask
and assembly path executes).  The one test that needs real hardware is the
slab gather, which takes a path CPU shards cannot reach and skips without
CUDA."""

import math

import numpy as np
import pytest
import torch

from mbirtorch import _sharding
from mbirtorch._sharding import Placement, Shards


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
