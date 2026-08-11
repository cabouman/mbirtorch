"""Multi-device units: Placement, Shards, safe transfer, and the
banded adjoint pair.  Runs everywhere: placement/banding logic uses two
'virtual' CPU devices (['cpu', 'cpu'] -- transfers are no-ops but every
range/pad/mask/assembly path executes), and real cross-device movement uses
the cpu<->mps pair when MPS is available (the local analog of the 2-GPU
platform)."""

import numpy as np
import pytest
import torch

from mbirtorch._sharding import (Placement, Shards, broadcast_band_to_views,
                                 device_pool, is_dev2dev_safe, move_shard,
                                 run_per_device, sum_band_to_owner)


def test_placement_ranges_and_padding():
    p = Placement(["cpu", "cpu"], axis=0, real_size=7)
    assert p.n_devices == 2 and not p.is_trivial
    assert p.padded_size == 8 and p.is_padded
    ranges = p.padded_shard_ranges()
    assert [(s, e) for _, (s, e), _ in ranges] == [(0, 4), (4, 8)]
    assert [v for _, _, v in ranges] == [4, 3]      # last shard: 3 real + 1 pad
    mask = p.real_mask(3)
    assert mask.shape == (8, 1, 1)
    assert mask.sum() == 7 and not mask[7, 0, 0]

    q = Placement(["cpu"], axis=-1, real_size=5)
    assert q.is_trivial and not q.is_padded and q.real_mask(2) is None
    with pytest.raises(ValueError, match="evenly shard"):
        p.shard_ranges(7)


def test_shards_gather_roundtrip():
    p = Placement(["cpu", "cpu"], axis=0, real_size=6)
    full = np.arange(24, dtype=np.float32).reshape(6, 4)
    parts = [torch.as_tensor(full[s:e]) for _, (s, e) in p.shard_ranges(6)]
    sh = Shards(parts, p)
    assert np.array_equal(sh.gather(), full)
    with pytest.raises(ValueError, match="shard tensors"):
        Shards(parts[:1], p)


def test_banded_adjoint_pair_values():
    # broadcast then reduce reproduces a single-device sum exactly on the
    # virtual 2-CPU placement (transfers are no-ops; the pattern is real).
    owners = [torch.device("cpu"), torch.device("cpu")]
    band = torch.rand(5, 3)
    copies = broadcast_band_to_views(band, owners)
    assert all(torch.equal(c, band) for c in copies.values())
    partials = [torch.rand(5, 3) for _ in owners]
    total = sum_band_to_owner(partials, owners[0])
    assert torch.allclose(total, partials[0] + partials[1])


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
    sino_shape = (9, 7, 8)                       # padded slices, 2 devices
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


def test_run_per_device_order_and_pool():
    devs = ["cpu", "cpu", "cpu"]
    out = run_per_device(devs, lambda i, d: (i, str(d)))
    assert [i for i, _ in out] == [0, 1, 2]
    with device_pool(3) as pool:
        out2 = run_per_device(devs, lambda i, d: i * 10, executor=pool)
        out3 = run_per_device(devs, lambda i, d: i * 100, executor=pool)
    assert out2 == [0, 10, 20] and out3 == [0, 100, 200]


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason="needs a second local device (mps)")
def test_real_cross_device_transfer_cpu_mps():
    devs = ["cpu", "mps"]
    assert is_dev2dev_safe(devs)
    band = torch.rand(64, 16)
    copies = broadcast_band_to_views(band, [torch.device(d) for d in devs])
    assert copies[torch.device("mps")].device.type == "mps"
    assert torch.allclose(copies[torch.device("mps")].cpu(), band)
    partials = [copies[torch.device("cpu")] * 2.0,
                copies[torch.device("mps")] * 3.0]
    total = sum_band_to_owner(partials, torch.device("cpu"))
    assert total.device.type == "cpu"
    assert torch.allclose(total, band * 5.0, atol=1e-6)
    # The host-bounce fallback path is value-correct too.
    bounced = move_shard(band.to("mps"), torch.device("cpu"), dev2dev_safe=False)
    assert torch.allclose(bounced, band)


def test_model_shard_and_gather_roundtrip():
    # configure_devices widens the placements; _shard_sinogram/_shard_recon then split a
    # real-shape array into padded per-device shards and the gathers crop the
    # padding back -- a lossless round trip.  Two 'virtual' CPU devices keep
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
    assert np.allclose(m._gather_sinogram(sh), sino)
    assert m._shard_sinogram(sh) is sh           # pass-through when placed

    rs = tuple(m.get_params('recon_shape'))
    recon = np.random.RandomState(1).rand(*rs).astype(np.float32)
    rh = m._shard_recon(recon)
    assert isinstance(rh, Shards)
    # slice axis 6 -> 3+3, and a NON-dividing case (7 detector rows -> 7
    # recon slices over 2 devices) pads with inert zeros:
    m2 = mbirtorch.ParallelBeamModel((10, 7, 8), np.linspace(0, np.pi, 10,
                                     endpoint=False))
    m2.configure_devices(devices=["cpu"])
    m2.set_params(no_warning=True, verbose=0)
    m2.configure_devices(devices=["cpu", "cpu"])
    r9 = np.random.RandomState(2).rand(*tuple(m2.get_params('recon_shape'))
                                       ).astype(np.float32)
    rh9 = m2._shard_recon(r9)
    assert rh9.placement.is_padded
    assert float(rh9.tensors[-1][..., -1].abs().max()) == 0.0   # zero tail
    assert np.allclose(m2._gather_recon(rh9), r9)               # crop restores

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
    assert np.allclose(m._gather_sinogram(fwd), ref_fwd, atol=1e-6)
    back = m.sparse_back_project(sino, idx)
    assert isinstance(back, Shards)
    assert np.allclose(back.gather(), ref_back, atol=1e-5)
    back2 = m.sparse_back_project(sino, idx, coeff_power=2)
    assert np.allclose(back2.gather(), ref_back2, atol=1e-5)


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
    assert np.allclose(m._gather_sinogram(fwd), ref_fwd, atol=1e-4)
    back = m.sparse_back_project(sino, idx)
    assert np.allclose(back.gather(), ref_back, atol=1e-4)


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


def test_cone_banded_projectors_match_single_device():
    # Cone bands spread over many rows: the banded forward ACCUMULATES
    # full-row partials and the banded back consumes the full local sinogram
    # per band -- both must reproduce the single-device values (band tiling is
    # a sum reorder, so float noise only).
    m, idx, vals, sino, ref_fwd, ref_back = _cone_banded_case(["cpu", "cpu"])
    fwd = m.sparse_forward_project(vals, idx)
    assert np.allclose(m._gather_sinogram(fwd), ref_fwd, atol=1e-5)
    back = m.sparse_back_project(sino, idx)
    assert np.allclose(back.gather(), ref_back, atol=1e-5)
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

    p = Placement(["cpu", "cpu"], axis=-1, real_size=S)
    shards = Shards([flat[:, s:e] for _, (s, e) in p.shard_ranges(S)], p)
    lh, rh = exchange_qggmrf_halos(shards)
    assert lh[0] is None and rh[-1] is None
    assert torch.equal(lh[1], flat[:, 3]) and torch.equal(rh[0], flat[:, 4])
    parts = [qggmrf.qggmrf_gradient_and_hessian_at_indices(
        shards.tensors[i], (rows, cols, S), idx, params,
        left_halo=lh[i], right_halo=rh[i]) for i in range(2)]
    g = torch.cat([pg for pg, _ in parts], dim=1)
    h = torch.cat([ph for _, ph in parts], dim=1)
    assert torch.allclose(g, ref_g, atol=1e-6)
    assert torch.allclose(h, ref_h, atol=1e-6)

    # An all-ones interface mask is a no-op; zeroing interface j decouples
    # slices j-1 and j exactly like a reflected edge there.
    mask = torch.ones(5)
    g1, h1 = qggmrf.qggmrf_gradient_and_hessian_at_indices(
        shards.tensors[0], (rows, cols, S), idx, params, right_halo=rh[0],
        interface_mask=mask)
    assert torch.allclose(g1, parts[0][0], atol=1e-7)
    mask4 = mask.clone(); mask4[-1] = 0.0
    g2, _ = qggmrf.qggmrf_gradient_and_hessian_at_indices(
        shards.tensors[0], (rows, cols, S), idx, params, right_halo=rh[0],
        interface_mask=mask4)
    g_ref, _ = qggmrf.qggmrf_gradient_and_hessian_at_indices(
        shards.tensors[0], (rows, cols, S), idx, params)
    assert torch.allclose(g2, g_ref, atol=1e-7)


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
    assert np.allclose(fm1, fm2, rtol=1e-3, atol=1e-4)

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
    # left the placements' real sizes stale, and the placement functions silently
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
    assert m.sino_placement.real_size == 12
    assert m.recon_placement.real_size == m.get_params('recon_shape')[2]
    sino = np.random.RandomState(0).rand(12, 10, 8).astype(np.float32)
    sh = m._shard_sinogram(sino)
    assert np.allclose(m._gather_sinogram(sh), sino)   # lossless again


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
        ref = m.fdk_recon(sino)
        m.configure_devices(devices=["cpu", "cpu"])
        out = m.fdk_recon(sino)
        assert isinstance(out, np.ndarray)
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
        assert rel < 1e-5, (shifts is None, rel)


def test_padded_placement_roundtrip_with_row_pad():
    # Non-dividing view AND slice axes on two virtual CPU devices: the device
    # form pads views 9->10 and (parallel row<->slice tie) rows/slices 7->8,
    # zero-filled; the gathers crop both axes back to the real counts.
    import mbirtorch
    sino_shape = (9, 7, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    m = mbirtorch.ParallelBeamModel(sino_shape, angles)
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    m.configure_devices(devices=["cpu", "cpu"])
    assert m.sino_placement.is_padded and m.recon_placement.is_padded
    rng = np.random.default_rng(3)
    sino = rng.standard_normal(sino_shape).astype(np.float32)

    prepared = m.prepare_sino_for_devices(sino)
    shapes = [tuple(t.shape) for t in prepared.tensors]
    assert shapes == [(5, 8, 8), (5, 8, 8)]
    # The zero tails: last device's padded view, and every device's row tail.
    assert float(torch.abs(prepared.tensors[1][-1]).sum()) == 0.0
    assert all(float(torch.abs(t[:, 7:]).sum()) == 0.0 for t in prepared.tensors)
    back = m._gather_sinogram(prepared)
    assert back.shape == sino_shape
    assert np.allclose(back, sino, atol=1e-7)
    # A prepared (device-form) array re-enters _shard_sinogram unchanged.
    again = m._shard_sinogram(prepared)
    assert again is prepared

    # Weights ride the same seam: padded entries are weightless.
    _, w = m.prepare_sino_for_devices(sino, weights=np.abs(sino) + 0.5)
    assert float(torch.abs(w.tensors[1][-1]).sum()) == 0.0

    # Recon-side padding: slice axis 7 -> 8, last shard's tail zero.
    recon_shape = tuple(m.get_params('recon_shape'))
    vol = rng.standard_normal(recon_shape).astype(np.float32)
    placed = m._shard_recon(vol)
    assert float(torch.abs(placed.tensors[1][..., -1]).sum()) == 0.0
    assert np.allclose(m._gather_recon(placed), vol, atol=1e-7)


def test_padded_banded_projectors_match_single_device():
    # Forward and back through the banded drivers on padded axes must equal
    # the single-device values on the REAL entries (padding inert).
    import mbirtorch
    sino_shape = (9, 7, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build(n):
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.configure_devices(devices=["cpu"])
        m.set_params(no_warning=True, verbose=0)
        if n > 1:
            m.configure_devices(devices=["cpu"] * n)
        return m

    m1, m2 = build(1), build(2)
    rs = tuple(m1.get_params('recon_shape'))
    rng = np.random.default_rng(5)
    vol = rng.standard_normal(rs).astype(np.float32)
    sino_ref = m1.forward_project(vol)
    sino_2 = m2._gather_sinogram(m2.forward_project(vol, output_sharded=True))
    rel = np.max(np.abs(sino_2 - sino_ref)) / np.max(np.abs(sino_ref))
    assert rel < 1e-5, rel

    sino = rng.standard_normal(sino_shape).astype(np.float32)
    bp_ref = m1.back_project(sino)
    bp_2 = m2._gather_recon(m2.back_project(sino, output_sharded=True))
    rel = np.max(np.abs(bp_2 - bp_ref)) / np.max(np.abs(bp_ref))
    assert rel < 1e-5, rel
    # The padded slice tail of the device form itself is zero (the forced-
    # zero invariant the prior relies on).
    bp_dev = m2.back_project(sino, output_sharded=True)
    assert float(torch.abs(bp_dev.tensors[-1][..., -1]).sum()) == 0.0


def test_padded_sharded_vcd_recon_matches_single_device():
    # The decisive padded gate: a seeded recon with non-dividing views AND
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
    print(f"padded sharded vcd: recon rel_max {rel:.2e}, fm diff "
          f"{np.max(np.abs(fm1 - fm2)):.2e}")
    assert rel < 5e-4, rel
    assert np.allclose(fm1, fm2, rtol=1e-3, atol=1e-4)

    # Constant weights: the padded ones Hessian seam.
    m3 = build()
    np.random.seed(48)
    ref_u, _ = m3.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)
    m4 = build()
    m4.configure_devices(devices=["cpu", "cpu"])
    np.random.seed(48)
    out_u, _ = m4.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0)
    rel_u = np.max(np.abs(out_u - ref_u)) / max(np.max(np.abs(ref_u)), 1e-30)
    assert rel_u < 5e-4, rel_u


def test_fully_idle_device_refused():
    # A device idle on ONE axis is legal (the thin-volume and sparse-view
    # extensions); a device with no real views AND no real slices would do
    # nothing at all and is refused.  5 views on 4 devices (slices real
    # everywhere) now configures; 3 views x 3 slices on 8 devices does not.
    import mbirtorch
    sino_shape = (5, 8, 8)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    m = mbirtorch.ParallelBeamModel(sino_shape, angles)
    m.configure_devices(devices=["cpu"])
    m.set_params(no_warning=True, verbose=0)
    m.configure_devices(devices=["cpu"] * 4)   # empty VIEW shard: allowed
    assert m.sino_placement.padded_shard_ranges()[-1][2] == 0

    m2 = mbirtorch.ParallelBeamModel((3, 3, 8),
                                     np.linspace(0, np.pi, 3, endpoint=False))
    m2.configure_devices(devices=["cpu"])
    m2.set_params(no_warning=True, verbose=0)
    try:
        m2.configure_devices(devices=["cpu"] * 8)
        raise AssertionError("expected ValueError for a fully idle device")
    except ValueError as e:
        assert 'no real views AND no real slices' in str(e)


def test_cone_sharded_vcd_recon_matches_single_device():
    # The cone VCD loop on two devices: the DC-damping profile now splits per
    # shard (dev_index seam), so the multi-device guard is gone.  A seeded
    # recon at a dividing cell and at a PADDED (non-dividing slices) cell
    # reproduces the single-device run.
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
        assert m2.recon_placement.is_padded == (rs[2] % 2 != 0)
        np.random.seed(53)
        out, _ = m2.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0)
        rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
        print(f"cone sharded vcd {cell}: rel_max {rel:.2e}")
        assert rel < 5e-4, (cell, rel)


def test_sub_band_streaming_matches_unstreamed():
    # Force 2-slice sub-bands through both banded drivers (the default
    # bounds give one band at test sizes) on a padded parallel cell AND a
    # cone cell: values must match the single-device references exactly at
    # the banded drivers' established tolerances -- streaming is a pure
    # partition of the work.
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
    m2.forward_project_slice_band = 2
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
    c2.forward_project_slice_band = 2
    c2.back_project_slice_band = 2
    cfwd = c2._gather_sinogram(c2.forward_project(cvol, output_sharded=True))
    cbp = c2._gather_recon(c2.back_project(csino, output_sharded=True))
    assert np.max(np.abs(cfwd - cfwd_ref)) / np.max(np.abs(cfwd_ref)) < 1e-5
    assert np.max(np.abs(cbp - cbp_ref)) / np.max(np.abs(cbp_ref)) < 1e-5


def test_move_shard_small_and_scalar_tensors():
    # The transfer primitive on the shapes the VCD loop actually moves besides
    # bands: 0-d line-search partials (the on-device combine) and per-slice
    # stat vectors -- on both the direct and the host-bounce paths.
    from mbirtorch._sharding import move_shard
    for safe in (True, False):
        s = torch.tensor(3.25)
        out = move_shard(s, torch.device("cpu"), dev2dev_safe=safe)
        assert out.ndim == 0 and float(out) == 3.25
        v = torch.arange(5, dtype=torch.float32)
        out_v = move_shard(v, torch.device("cpu"), dev2dev_safe=safe)
        assert torch.equal(out_v, v)
    if torch.backends.mps.is_available():
        s = torch.tensor(1.5)
        out = move_shard(s, torch.device("mps"), dev2dev_safe=True)
        assert out.device.type == "mps" and float(out.cpu()) == 1.5


def test_thin_volume_more_devices_than_slices():
    # The thin-volume extension (beyond mbirjax): more devices than slices is
    # a legal layout -- the extra devices carry views (the dominant compute
    # and memory) while their all-padded slice shards stay exactly inert.
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
    assert m2.recon_placement.padded_shard_ranges()[-1][2] == 0
    np.random.seed(61)
    out, _ = m2.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    print(f"thin parallel n4 vs n1: rel {rel:.2e}")
    assert rel < 5e-4, rel
    # The empty shard's device form stayed identically zero.
    dev_recon, _ = m2.recon(sino, weights=weights, max_iterations=1,
                            stop_threshold_change_pct=0.0, output_sharded=True)
    assert float(torch.abs(dev_recon.tensors[-1]).sum()) == 0.0

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
    assert c2.recon_placement.padded_shard_ranges()[-1][2] == 0
    np.random.seed(62)
    cout, _ = c2.recon(csino, max_iterations=2, stop_threshold_change_pct=0.0)
    crel = np.max(np.abs(cout - cref)) / max(np.max(np.abs(cref)), 1e-30)
    print(f"thin cone n4 vs n1: rel {crel:.2e}")
    assert crel < 5e-4, crel


def test_sparse_view_more_devices_than_views():
    # The sparse-view extension: more devices than views is a legal layout --
    # the extra devices hold slice shards and run the prior and updates (the
    # dominant work with few views) while their all-padded view shards stay
    # exactly inert.  Seeded n=4 vcd vs n=1 on a 3-view parallel cell and a
    # 3-view cone cell.
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
    assert m2.sino_placement.padded_shard_ranges()[-1][2] == 0
    np.random.seed(71)
    out, _ = m2.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    print(f"sparse-view parallel n4 vs n1: rel {rel:.2e}")
    # Tolerance calibrated to THIS cell's inherent multi-device floor: with
    # DIVIDING views (16) and no padding at all, the same cell reads
    # 1.8e-3 at n=2 and 1.9e-3 at n=4 (staged-halo staleness within a
    # partition pass plus cross-device float reorders -- the mbirjax
    # structure).  The sparse-view layout adds nothing beyond that floor.
    assert rel < 5e-3, rel
    # The empty view-owner's sinogram block stays identically zero.
    fwd_dev = m2.forward_project(phantom, output_sharded=True)
    assert float(torch.abs(fwd_dev.tensors[-1]).sum()) == 0.0

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
    assert c2.sino_placement.padded_shard_ranges()[-1][2] == 0
    np.random.seed(72)
    cout, _ = c2.recon(csino, max_iterations=2, stop_threshold_change_pct=0.0)
    crel = np.max(np.abs(cout - cref)) / max(np.max(np.abs(cref)), 1e-30)
    print(f"sparse-view cone n4 vs n1: rel {crel:.2e}")
    assert crel < 5e-3, crel   # same calibration as the parallel case above


# ── the forward's column gather (default off) ────────────────────────────────
# What may FAIL here, and what may only be recorded.  The value bar these
# tests hold is the one the library already ships: the kernel-parity floor the
# suites above enforce, at the 1e-5 relative these cone cases use on CPU.  The
# multi-GPU measurement also registered an EXPECTATION beside that floor -- the
# column gather sat about 1.5e-06 relative from the one-device anchor at the
# 1024-class cell (measured 2026-08-10 on four H100s, job mg10), against a
# banded walk that sat at its own repeat floor.  That expectation is recorded
# so a later reading well outside it is visible to a human weighing the
# tradeoff; it is deliberately NOT a threshold, and nothing here asserts it.
# The distances below are printed for that comparison.  On CPU the runs are
# deterministic and the gather's calls are the single-device call shape, so
# what these tests do assert is exact-path mechanics rather than that bar.
def _cone_column_case(devices, cell=(8, 8, 8), pixel_batch=None):
    """A cone model on virtual CPU devices with the column gather switched
    on, plus its single-device reference."""
    m, idx, vals, sino, ref_fwd, ref_back = _cone_banded_case(devices, cell)
    m.forward_column_gather = True
    if pixel_batch is not None:
        m.forward_project_pixel_batch = pixel_batch
    assert m._column_gather_forward()
    return m, idx, vals, sino, ref_fwd, ref_back


def test_gather_column_band_assembles_the_full_height_cylinder():
    # The primitive: every slice-owner's rows [p0:p1] moved to one target and
    # concatenated along the SLICE axis, in shard (global slice) order.  A
    # single shard short-circuits the concatenation.
    from mbirtorch._sharding import gather_column_band
    rng = np.random.default_rng(11)
    full = torch.as_tensor(rng.standard_normal((9, 6)).astype(np.float32))
    shards = [full[:, 0:2].contiguous(), full[:, 2:4].contiguous(),
              full[:, 4:6].contiguous()]
    cyl = gather_column_band(shards, 3, 7, torch.device("cpu"))
    assert cyl.shape == (4, 6)
    assert torch.equal(cyl, full[3:7])
    # A degenerate range is legal and empty; one shard is returned as itself.
    empty = gather_column_band(shards, 5, 5, torch.device("cpu"))
    assert empty.shape == (0, 6)
    one = gather_column_band(shards[:1], 0, 9, torch.device("cpu"))
    assert torch.equal(one, shards[0])
    # The host-bounce path is value-correct too (dev2dev_safe False).
    bounced = gather_column_band(shards, 0, 9, torch.device("cpu"),
                                 dev2dev_safe=False)
    assert torch.equal(bounced, full)


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason="needs a second local device (mps)")
def test_gather_column_band_moves_across_real_devices():
    from mbirtorch._sharding import gather_column_band
    full = torch.rand(32, 8)
    shards = [full[:, :4].contiguous().to("cpu"),
              full[:, 4:].contiguous().to("mps")]
    cyl = gather_column_band(shards, 8, 16, torch.device("mps"))
    assert cyl.device.type == "mps"
    assert torch.allclose(cyl.cpu(), full[8:16], atol=1e-6)


def test_column_gather_matches_single_device_at_every_batch(monkeypatch):
    # The values gate on virtual CPU devices: a full-height call at
    # slice_start=0 is the single-device call shape, so the gathered forward
    # must reproduce the single-device values -- at one batch covering the
    # pass, and at batches that force several.  The environment is cleared
    # first so a suite run forcing the banded walk cannot unseat the gather
    # this test is about -- the same pinning the banded tests do in reverse.
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    for n in (2, 3):
        for batch in (None, 1, 5, 10 ** 6):
            m, idx, vals, _sino, ref_fwd, _ref_back = _cone_column_case(
                ["cpu"] * n, pixel_batch=batch)
            fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))
            rel = np.max(np.abs(fwd - ref_fwd)) / np.max(np.abs(ref_fwd))
            print(f"cone column gather n={n} batch={batch}: rel {rel:.2e}")
            assert rel < 1e-5, (n, batch, rel)


def test_column_gather_holds_the_adjoint_and_the_padded_forms(monkeypatch):
    # The back driver is untouched, so the pair must stay adjoint with the
    # gather on -- on a padded cell (9 views, 7 rows over 2 devices pads both
    # axes), where the gathered cylinder carries the inert padded slice tail.
    # The environment is cleared first, for the reason above.
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    m, idx, vals, sino, ref_fwd, ref_back = _cone_column_case(
        ["cpu", "cpu"], cell=(9, 7, 8), pixel_batch=4)
    assert m.recon_placement.is_padded and m.sino_placement.is_padded
    fwd = m.sparse_forward_project(vals, idx)
    back = m.sparse_back_project(sino, idx)
    real = vals.shape[1]
    assert np.allclose(m._gather_sinogram(fwd), ref_fwd, atol=1e-5)
    assert np.allclose(back.gather()[:, :real], ref_back, atol=1e-5)
    lhs = float(np.sum(m._gather_sinogram(fwd) * sino))
    rhs = float(np.sum(vals * back.gather()[:, :real]))
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (lhs, rhs)


def test_column_gather_replaces_the_band_broadcast(monkeypatch):
    # The mechanics witness.  With the gather on, the cone forward must call
    # gather_column_band and must NOT broadcast a band; each gather takes one
    # piece per slice-owner and yields a cylinder that is the batch wide and
    # the WHOLE device-form slice axis tall; and each projector call runs at
    # slice_start=0 over that whole axis for the owner's own views.  The
    # environment is cleared first, for the reason above.
    from mbirtorch import _sharding as sharding
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    batch, n = 4, 2
    m, idx, vals, _sino, _ref_fwd, _ref_back = _cone_column_case(
        ["cpu"] * n, pixel_batch=batch)
    slices = m.recon_placement.padded_size
    gathers, broadcasts, calls = [], [], []
    real_gather = sharding.gather_column_band

    def spy_gather(shard_tensors, p0, p1, target, dev2dev_safe=True):
        out = real_gather(shard_tensors, p0, p1, target, dev2dev_safe)
        gathers.append((len(shard_tensors), p0, p1, tuple(out.shape)))
        return out

    def spy_broadcast(*args, **kwargs):
        broadcasts.append(args)
        raise AssertionError("the column gather must not broadcast a band")

    real_call = m.projector_functions.sparse_forward_project_view_range

    def spy_call(band_values, pixel_indices, view_range, slice_start=0,
                 dev_index=0, plan=None):
        calls.append((tuple(band_values.shape), int(pixel_indices.shape[0]),
                      tuple(view_range), slice_start))
        return real_call(band_values, pixel_indices, view_range,
                         slice_start=slice_start, dev_index=dev_index,
                         plan=plan)

    monkeypatch.setattr(sharding, "gather_column_band", spy_gather)
    monkeypatch.setattr(sharding, "broadcast_band_to_views", spy_broadcast)
    monkeypatch.setattr(m.projector_functions,
                        "sparse_forward_project_view_range", spy_call)
    m.sparse_forward_project(vals, idx)

    expected_batches = -(-len(idx) // batch)
    assert not broadcasts
    assert len(gathers) == n * expected_batches
    for pieces, p0, p1, shape in gathers:
        assert pieces == n                      # one piece per slice-owner
        assert shape == (p1 - p0, slices)       # the batch, at every slice
        assert p1 - p0 <= batch
    # One projector call per (pixel batch, view-owner), each over the whole
    # slice axis anchored at 0 and over that owner's own real views.
    assert len(calls) == n * expected_batches
    spans = [(v0, v1) for _, _, (v0, v1), _ in calls]
    for cyl_shape, n_pixels, (v0, v1), slice_start in calls:
        assert slice_start == 0 and cyl_shape[1] == slices
        assert n_pixels == cyl_shape[0] and v1 > v0
    assert set(spans) == {
        (v0, v0 + valid) for _d, (v0, _v1), valid
        in m.sino_placement.padded_shard_ranges() if valid > 0}


def test_the_column_gather_runs_one_batch_ahead_of_the_projection(monkeypatch):
    # The prefetch witness.  Each view-owner issues the NEXT pixel batch's
    # gather before it projects the current batch, so that on real devices the
    # copies feeding one projection can be moving while another projection
    # runs.  On virtual CPU devices nothing moves and nothing can be timed, so
    # what is asserted here is the ORDER the driver issues its work in, which
    # is the part of the change that has to hold on every device.
    #
    # The order one worker records is g0, g1, p0, g2, p1, ... , g(K-1), p(K-2),
    # p(K-1): batch k+1's gather is issued before batch k is projected, the
    # first gather runs before the loop, and the last batch has nothing to
    # gather ahead of it.  The entry and exit of each projection are both
    # recorded, so the witness is not merely that the gather precedes the
    # accumulation -- it precedes the projector call entirely.
    #
    # Each worker runs in its own thread, so events are kept per thread.  A
    # pool thread is allowed to run more than one worker when one finishes
    # before the next is submitted, and it would then record two workers'
    # sequences end to end; the check reads blocks rather than the whole list
    # so that it witnesses the order either way.
    #
    # The environment knob is cleared first, the way the tests around this one
    # already do: it forces the gather off whatever the model says, so a suite
    # run that sets it would otherwise decide what this test measures.
    import threading
    from mbirtorch import _sharding as sharding
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    batch, n = 4, 2
    m, idx, vals, _sino, ref_fwd, _rb = _cone_column_case(
        ["cpu"] * n, pixel_batch=batch)
    n_batches = -(-len(idx) // batch)
    assert n_batches > 1                       # or there is no prefetch to see
    events = {}
    real_gather = sharding.gather_column_band
    real_call = m.projector_functions.sparse_forward_project_view_range

    def spy_gather(shard_tensors, p0, p1, target, dev2dev_safe=True):
        # Recorded at ENTRY: what is being witnessed is when the gather is
        # issued, not when it returns.
        events.setdefault(threading.get_ident(), []).append(f'g{p0 // batch}')
        return real_gather(shard_tensors, p0, p1, target, dev2dev_safe)

    def spy_call(band_values, pixel_indices, view_range, slice_start=0,
                 dev_index=0, plan=None):
        seq = events.setdefault(threading.get_ident(), [])
        # Number the projections within THIS worker, which begins at its own
        # first gather, so that a thread running a second worker starts over
        # at zero rather than counting on from the first.
        first = len(seq) - 1 - seq[::-1].index('g0')
        k = sum(1 for e in seq[first:] if e.endswith('-in'))
        seq.append(f'p{k}-in')
        block = real_call(band_values, pixel_indices, view_range,
                          slice_start=slice_start, dev_index=dev_index,
                          plan=plan)
        seq.append(f'p{k}-out')
        return block

    monkeypatch.setattr(sharding, "gather_column_band", spy_gather)
    monkeypatch.setattr(m.projector_functions,
                        "sparse_forward_project_view_range", spy_call)
    fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))

    expected = ['g0']
    for k in range(n_batches):
        if k + 1 < n_batches:
            expected.append(f'g{k + 1}')
        expected += [f'p{k}-in', f'p{k}-out']
    assert events, "the column gather did not run"
    recorded = 0
    for seq in events.values():
        # Every worker of this cell owns real views, so each ran the whole
        # sequence; a thread holds a whole number of them.
        assert len(seq) % len(expected) == 0, seq
        for start in range(0, len(seq), len(expected)):
            assert seq[start:start + len(expected)] == expected, seq
            recorded += 1
    assert recorded == n                       # one sequence per view-owner
    # The prefetch moves WHEN a gather is issued and nothing else, so the
    # values are the ones the path already produced.
    assert np.allclose(fwd, ref_fwd, atol=1e-5)

    # And the values hold across batch widths that force several batches,
    # including one that leaves a short final batch (30 pixels over 7).
    for width in (1, 3, 7):
        mb, idxb, valsb, _s, ref_b, _rb2 = _cone_column_case(
            ["cpu"] * n, pixel_batch=width)
        out = mb._gather_sinogram(mb.sparse_forward_project(valsb, idxb))
        rel = np.max(np.abs(out - ref_b)) / np.max(np.abs(ref_b))
        print(f"cone gather one batch ahead, {width}-pixel batches: "
              f"rel {rel:.2e}")
        assert rel < 1e-5, (width, rel)


def test_the_column_gather_is_on_by_default_and_scoped_to_its_geometry(
        monkeypatch):
    # The switch: on unless refused, refused on a geometry the shape has never
    # been measured on however it is asked, and overridable from the
    # environment either way so one session can run both shapes over the same
    # inputs.  The environment is cleared first, because this test reads the
    # DEFAULT and a suite run may be forcing the path around it.
    import mbirtorch
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    cone, _idx, _vals, _sino, _f, _b = _cone_banded_case(["cpu", "cpu"])
    assert cone.column_gather_geometry
    assert cone._column_gather_forward()              # default on
    cone.forward_column_gather = False                # the rollback
    assert not cone._column_gather_forward()
    cone.forward_column_gather = True
    assert cone._column_gather_forward()

    # The row-aligned geometry declares the same capability, on its own
    # measurement, and is on by default in the same way.
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    par = mbirtorch.ParallelBeamModel((8, 6, 8), angles)
    assert par.column_gather_geometry
    assert par._column_gather_forward()
    par.forward_column_gather = False
    assert not par._column_gather_forward()
    par.forward_column_gather = True
    assert par._column_gather_forward()

    # A geometry that has never been timed on the shape refuses it however it
    # is asked: translation shares cone's banded branch, and an argument that
    # it should gain too is not a measurement.
    trans = mbirtorch.TranslationModel(
        (4, 6, 8), np.zeros((4, 3), dtype=np.float32),
        source_detector_dist=32.0, source_iso_dist=16.0)
    trans.forward_column_gather = True
    assert not trans.column_gather_geometry
    assert not trans._column_gather_forward()

    # The environment wins over the attribute in BOTH directions: `cone` holds
    # an explicit True and `refused` an explicit False, and each env value
    # drives the two models to the same answer.
    import os
    refused = _cone_banded_case(["cpu", "cpu"])[0]
    refused.forward_column_gather = False
    for value, expected in (("1", True), ("on", True),
                            ("0", False), ("off", False)):
        os.environ[COLUMN_GATHER_ENV_VAR] = value
        try:
            assert refused._column_gather_forward() is expected
            assert cone._column_gather_forward() is expected
        finally:
            del os.environ[COLUMN_GATHER_ENV_VAR]
    assert not refused._column_gather_forward()
    unset = _cone_banded_case(["cpu", "cpu"])[0]
    assert unset._column_gather_forward()             # the shipped default


@pytest.mark.parametrize('geometry', ('cone', 'parallel'))
def test_the_banded_walk_is_what_runs_with_the_switch_off(geometry,
                                                          monkeypatch):
    # The rollback, exercised on both geometries that can take the gather:
    # switching the gather off selects the banded walk, which broadcasts
    # bands and gathers no columns.  The switch has to be refused explicitly
    # now that unset means on, and the environment knob is cleared first for
    # the reason above.
    from mbirtorch import _sharding as sharding
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    if geometry == 'cone':
        m, idx, vals, _sino, ref_fwd, _rb = _cone_banded_case(["cpu", "cpu"])
    else:
        m, idx, vals, _sino, ref_fwd, _rb, _b2 = _banded_case(["cpu", "cpu"])
    m.forward_column_gather = False
    assert m.column_gather_geometry and not m._column_gather_forward()
    broadcasts = []
    real_broadcast = sharding.broadcast_band_to_views

    def spy_broadcast(band, view_owners, dev2dev_safe=True):
        broadcasts.append(tuple(band.shape))
        return real_broadcast(band, view_owners, dev2dev_safe)

    def refuse(*args, **kwargs):
        raise AssertionError("the banded walk must not gather columns")

    monkeypatch.setattr(sharding, "broadcast_band_to_views", spy_broadcast)
    monkeypatch.setattr(sharding, "gather_column_band", refuse)
    fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))
    assert broadcasts
    assert np.allclose(fwd, ref_fwd, atol=1e-5)


def test_column_gather_recon_matches_single_device(monkeypatch):
    # The end-to-end gate: a seeded cone reconstruction on two virtual CPU
    # devices with the gather on must reproduce the single-device run, which
    # is where the two changed summation orders (the vertical sum into the
    # body, the pixel sum out of it) would show up if they were not inside
    # the value class the forward already has.  The environment is cleared
    # first so each of the three runs below is the shape it names.
    import mbirtorch
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
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

    banded = build(["cpu", "cpu"])
    banded.forward_column_gather = False    # unset now means the gather
    np.random.seed(31)
    banded_out, _ = banded.recon(sino, max_iterations=2,
                                 stop_threshold_change_pct=0.0)
    gathered = build(["cpu", "cpu"])
    gathered.forward_column_gather = True
    gathered.forward_project_pixel_batch = 8
    np.random.seed(31)
    out, _ = gathered.recon(sino, max_iterations=2,
                            stop_threshold_change_pct=0.0)
    scale = max(np.max(np.abs(ref)), 1e-30)
    rel = np.max(np.abs(out - ref)) / scale
    rel_banded = np.max(np.abs(banded_out - ref)) / scale
    # Printed rather than asserted against each other: which of the two sits
    # closer to the anchor is the reading the registered expectation is for.
    print(f"cone recon vs n1: column gather {rel:.2e}, "
          f"banded {rel_banded:.2e}")
    assert rel < 5e-3, rel     # the shipped parity floor, as above


# ── the same gather, on the row-aligned geometry ─────────────────────────────
# Parallel takes the column gather for a different measured reason than cone.
# It CAN produce its detector rows from a slice band -- the banded walk does
# exactly that -- but its forward kernel runs about twice as efficiently per
# slice on the full-width block of values the gather hands it as on the
# shard-width blocks the band hands it (measured 2026-08-10 on one H100, at
# 0.0411 ms per slice on a 1008-wide block against 0.0823 on a 504-wide one,
# with the device count held at one).
#
# The value bar was expected to be EQUALITY here, on the argument that each
# detector row keeps a single producing call and CPU sums are deterministic.
# The row half of that is true, and the mechanics test below asserts it
# directly.  Equality is not, and the measurement that settled it is recorded
# because it is worth knowing before anyone tries again (2026-08-10, this
# suite, virtual CPU devices).  Run first in a fresh interpreter, BOTH the
# banded walk and the column gather reproduce the single-device sinogram bit
# for bit.  Run once other shapes have gone through the same per-device
# bodies -- which is what a full suite run does -- both land in the float32
# epsilon class instead, the banded walk at 1.6e-07 to 4.0e-07 and the gather
# at 1.1e-07 to 4.0e-07 over the same cells.  The cause is the same for both:
# the per-device bodies are separately torch.compiled, and what a compiled
# body emits depends on the shapes its instance has already seen, so two
# devices can differ in the last bit on identical inputs.  Bit-equality is
# therefore a property of the process, not of the driver shape, and these
# tests hold the gather against the shape it replaces instead.
def _parallel_column_case(devices, sino_shape=(8, 6, 8), pixel_batch=None):
    """A parallel model on virtual CPU devices with the column gather
    switched on, plus its single-device reference."""
    m, idx, vals, sino, ref_fwd, ref_back, _b2 = _banded_case(devices,
                                                              sino_shape)
    m.forward_column_gather = True
    if pixel_batch is not None:
        m.forward_project_pixel_batch = pixel_batch
    assert m._column_gather_forward()
    return m, idx, vals, sino, ref_fwd, ref_back


def test_parallel_column_gather_matches_the_shape_it_replaces(monkeypatch):
    # The environment is cleared first so each leg below runs the shape it
    # names, whatever a suite run is forcing around this test.
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    # The values gate.  Both shapes are run over the same inputs in the same
    # process and both are held to the same bar, which is the reading that
    # does not move with the compile state above; the two distances are
    # printed side by side for the same reason.  One call per view-owner over
    # every pixel is the single-device call in every respect that sets a
    # value: the same voxel columns in one array, the whole slice range
    # anchored at 0, and each detector row produced by that one call and no
    # other.  A row taking contributions from more than one call would show up
    # here as an order-one error, not as a last bit.
    for n in (2, 3):
        # None takes the shipped batch, which covers a pass this size in one
        # call; the large value asks for that explicitly.
        for batch in (None, 10 ** 6):
            m, idx, vals, _sino, ref_fwd, _rb = _banded_case(["cpu"] * n)[:6]
            m.forward_column_gather = False   # unset now means the gather
            banded = m._gather_sinogram(m.sparse_forward_project(vals, idx))
            m.forward_column_gather = True
            if batch is not None:
                m.forward_project_pixel_batch = batch
            fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))
            scale = np.max(np.abs(ref_fwd))
            rel = np.max(np.abs(fwd - ref_fwd)) / scale
            rel_banded = np.max(np.abs(banded - ref_fwd)) / scale
            print(f"parallel column gather n={n} batch={batch}: rel {rel:.2e},"
                  f" banded {rel_banded:.2e}")
            assert rel < 1e-5 and rel_banded < 1e-5, (n, batch, rel,
                                                      rel_banded)

    # The one summation order this shape does change for a row-aligned
    # geometry: several pixel batches turn a single accumulation over every
    # pixel into a host-side sum of per-batch partials.  Nothing about the
    # rows moves, so what is left is float noise in the same class as above
    # (measured 1.0e-07 to 1.6e-07 here).  This is the case that runs at
    # production sizes, where the pass is far wider than one batch.
    for batch in (1, 5, 7):
        m, idx, vals, _sino, ref_fwd, _rb = _parallel_column_case(
            ["cpu", "cpu"], pixel_batch=batch)
        fwd = m._gather_sinogram(m.sparse_forward_project(vals, idx))
        rel = np.max(np.abs(fwd - ref_fwd)) / np.max(np.abs(ref_fwd))
        print(f"parallel column gather, {batch}-pixel batches: rel {rel:.2e}")
        assert rel < 1e-5, (batch, rel)


def test_parallel_column_gather_gathers_columns_and_sizes_its_rows_by_them(
        monkeypatch):
    # The environment is cleared first, for the reason above.
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    # The mechanics witness, plus the row-aligned fact the banded walk used to
    # supply by construction.  With the gather on, the parallel forward calls
    # gather_column_band and broadcasts no band; each cylinder is one pixel
    # batch by the WHOLE device-form slice axis; each projector call runs at
    # slice_start=0 over that whole axis for the owner's own views; and the
    # block that comes back is as TALL as the cylinder, because a row-aligned
    # body sizes its output by the values it was handed.  That last one is why
    # the assembled shard carries the device form's padded row count and not
    # the real detector rows.
    from mbirtorch import _sharding as sharding
    batch, n = 4, 2
    m, idx, vals, _sino, _ref_fwd, _rb = _parallel_column_case(
        ["cpu"] * n, pixel_batch=batch)
    slices = m.recon_placement.padded_size
    channels = int(m.get_params('sinogram_shape')[2])
    gathers, calls = [], []
    real_gather = sharding.gather_column_band

    def spy_gather(shard_tensors, p0, p1, target, dev2dev_safe=True):
        out = real_gather(shard_tensors, p0, p1, target, dev2dev_safe)
        gathers.append((len(shard_tensors), p0, p1, tuple(out.shape)))
        return out

    def spy_broadcast(*args, **kwargs):
        raise AssertionError("the column gather must not broadcast a band")

    real_call = m.projector_functions.sparse_forward_project_view_range

    def spy_call(band_values, pixel_indices, view_range, slice_start=0,
                 dev_index=0, plan=None):
        block = real_call(band_values, pixel_indices, view_range,
                          slice_start=slice_start, dev_index=dev_index,
                          plan=plan)
        calls.append((tuple(band_values.shape), tuple(view_range), slice_start,
                      tuple(block.shape)))
        return block

    monkeypatch.setattr(sharding, "gather_column_band", spy_gather)
    monkeypatch.setattr(sharding, "broadcast_band_to_views", spy_broadcast)
    monkeypatch.setattr(m.projector_functions,
                        "sparse_forward_project_view_range", spy_call)
    fwd = m.sparse_forward_project(vals, idx)

    expected_batches = -(-len(idx) // batch)
    assert len(gathers) == n * expected_batches
    for pieces, p0, p1, shape in gathers:
        assert pieces == n                      # one piece per slice-owner
        assert shape == (p1 - p0, slices)       # the batch, at every slice
        assert p1 - p0 <= batch
    assert len(calls) == n * expected_batches
    for cyl_shape, (v0, v1), slice_start, block_shape in calls:
        assert slice_start == 0 and cyl_shape[1] == slices
        assert block_shape == (v1 - v0, slices, channels)
    assert set((v0, v1) for _c, (v0, v1), _s, _b in calls) == {
        (v0, v0 + valid) for _d, (v0, _v1), valid
        in m.sino_placement.padded_shard_ranges() if valid > 0}
    # And the shard the driver assembles carries those same rows.
    assert all(tuple(t.shape[1:]) == (slices, channels) for t in fwd.tensors)


def test_parallel_column_gather_holds_the_padded_and_sparse_view_forms(
        monkeypatch):
    # The environment is cleared first, for the reason above.
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR
    monkeypatch.delenv(COLUMN_GATHER_ENV_VAR, raising=False)
    # The two forms where a row-aligned geometry's DEVICE shape differs from
    # its problem shape.  A padded slice axis pads the sinogram's detector
    # rows with it, so every block this driver assembles -- including the
    # empty one it builds for a view-owner with no real views -- has to be the
    # padded row count.  Sized at the real detector rows instead, which is the
    # count a row-RANGE geometry's blocks carry, the shards do not concatenate
    # at all.
    for shape, devs in (((9, 7, 8), 2),        # both axes padded
                        ((3, 7, 8), 4)):       # padded rows, an empty owner
        m, idx, vals, sino, ref_fwd, ref_back = _parallel_column_case(
            ["cpu"] * devs, sino_shape=shape, pixel_batch=10 ** 6)
        assert m.recon_placement.is_padded and m.sino_placement.is_padded
        real_rows = shape[1]
        fwd = m.sparse_forward_project(vals, idx)
        assert all(t.shape[1] == m.recon_placement.padded_size
                   for t in fwd.tensors), shape
        # The padded row tail stays identically zero, as the entry fill left
        # it: the gathered cylinder's padded slice tail is zero, and a
        # row-aligned body maps those columns straight to those rows.
        assert max(float(t[:, real_rows:].abs().max()) for t in fwd.tensors) \
            == 0.0, shape
        assert np.allclose(m._gather_sinogram(fwd), ref_fwd, atol=1e-5), shape
        # The back driver is untouched, so the pair stays adjoint.
        back = m.sparse_back_project(sino, idx)
        assert np.allclose(back.gather()[:, :vals.shape[1]], ref_back,
                           atol=1e-5)
        lhs = float(np.sum(m._gather_sinogram(fwd) * sino))
        rhs = float(np.sum(vals * back.gather()[:, :vals.shape[1]]))
        assert abs(lhs - rhs) / max(abs(rhs), 1e-30) < 1e-4, (shape, lhs, rhs)


def test_parallel_column_gather_recon_matches_single_device():
    # The end-to-end gate, where the subset passes call the forward on small
    # pixel sets and the pixel batch above therefore bites: a seeded parallel
    # reconstruction on two virtual CPU devices with the gather on must
    # reproduce the single-device run within the loop's own multi-device
    # floor, which the banded walk beside it is read against.
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

    banded = build(["cpu", "cpu"])
    np.random.seed(31)
    banded_out, _ = banded.recon(sino, max_iterations=3,
                                 stop_threshold_change_pct=0.0)
    gathered = build(["cpu", "cpu"])
    gathered.forward_column_gather = True
    gathered.forward_project_pixel_batch = 8
    np.random.seed(31)
    out, _ = gathered.recon(sino, max_iterations=3,
                            stop_threshold_change_pct=0.0)
    scale = max(np.max(np.abs(ref)), 1e-30)
    rel = np.max(np.abs(out - ref)) / scale
    rel_banded = np.max(np.abs(banded_out - ref)) / scale
    print(f"parallel recon vs n1: column gather {rel:.2e}, "
          f"banded {rel_banded:.2e}")
    assert rel < 5e-4, rel     # the sharded VCD loop's own floor at this cell


# ── one pixel at a time ──────────────────────────────────────────────────────
# The column gather's pixel batching hands the projectors a one-pixel call
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

