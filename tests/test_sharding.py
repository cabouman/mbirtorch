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
