"""Regression tests: set_params semantics, engine paths, and the
compile/eager and weights equivalences the suite previously never asserted."""

import warnings

import numpy as np
import pytest
import torch

import mbirtorch


def _small_model(device="cpu", **kwargs):
    sino_shape = (24, 16, 16)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    m = mbirtorch.ParallelBeamModel(sino_shape, angles, device=device, **kwargs)
    m.set_params(no_warning=True, verbose=0)
    return m


def _box_problem(model):
    recon_shape = model.get_params('recon_shape')
    phantom = np.zeros(tuple(recon_shape), dtype=np.float32)
    r0, c0, s0 = [max(1, n // 4) for n in recon_shape]
    phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    return phantom, model.forward_project(phantom)


# ── set_params semantics (mbirjax parity) ────────────────────────────────────
def test_manual_sigma_disables_auto_regularization():
    model = _small_model()
    with pytest.warns(UserWarning, match="disable auto-regularization"):
        model.set_params(sigma_x=0.123)
    assert model.get_params('auto_regularize_flag') is False
    # recon must USE the manual value, not overwrite it.
    _, sinogram = _box_problem(model)
    np.random.seed(0)
    _, recon_dict = model.recon(sinogram, max_iterations=1,
                                stop_threshold_change_pct=0.0)
    reg = recon_dict['recon_params']['regularization_params']
    assert abs(reg['sigma_x'] - 0.123) < 1e-12


def test_sharpness_reenables_auto_regularization():
    model = _small_model()
    model.set_params(no_warning=True, auto_regularize_flag=False)
    with pytest.warns(UserWarning, match="re-enabled auto-regularization"):
        model.set_params(sharpness=0.5)
    assert model.get_params('auto_regularize_flag') is True


def test_unknown_parameter_raises():
    model = _small_model()
    with pytest.raises(ValueError, match="not a recognized parameter"):
        model.set_params(not_a_param=1)


def test_multi_step_geometry_change_allowed():
    # mbirjax defers validation to recon entry, so a transiently-inconsistent
    # state between set_params and auto_set_recon_geometry must not raise.
    model = _small_model()
    new_shape = (30, 20, 20)
    new_angles = np.linspace(0, np.pi, new_shape[0], endpoint=False)
    model.set_params(sinogram_shape=new_shape, angles=new_angles)
    model.auto_set_recon_geometry()
    assert model.get_params('recon_shape')[2] == new_shape[1]
    sino = model.forward_project(np.ones(tuple(model.get_params('recon_shape')),
                                         dtype=np.float32))
    assert sino.shape == new_shape


def test_sinogram_shape_validated():
    model = _small_model()
    bad = np.zeros((10, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="sinogram does not have the shape"):
        np.random.seed(0)
        model.recon(bad, max_iterations=1)


def test_prox_input_shape_validated():
    model = _small_model()
    _, sinogram = _box_problem(model)
    recon_shape = tuple(model.get_params('recon_shape'))
    bad = np.zeros(recon_shape[::-1], dtype=np.float32).transpose(0, 1, 2)
    if bad.shape == recon_shape:
        pytest.skip("cell is a cube; transposed shape identical")
    with pytest.raises(ValueError, match="prox_input does not have the correct size"):
        np.random.seed(0)
        model.prox_map(bad, sinogram, max_iterations=1)


# ── engine paths ─────────────────────────────────────────────────────────────
def test_positivity_path(device):
    model = _small_model(device)
    model.set_params(no_warning=True, positivity_flag=True)
    phantom, sinogram = _box_problem(model)
    # Negative-going noise would drive an unconstrained recon negative.
    rng = np.random.RandomState(3)
    noisy = sinogram + 0.1 * np.max(sinogram) * rng.randn(*sinogram.shape).astype(np.float32)
    np.random.seed(0)
    recon, recon_dict = model.recon(noisy, max_iterations=3,
                                    stop_threshold_change_pct=0.0)
    assert float(recon.min()) >= -1e-5
    fm = recon_dict['recon_params']['fm_rmse']
    assert fm[-1] < fm[0]


def test_restart_contract():
    model = _small_model()
    _, sinogram = _box_problem(model)
    np.random.seed(11)
    r3, d3 = model.recon(sinogram, max_iterations=3, stop_threshold_change_pct=0.0)
    # Restart: two more iterations continuing the partition sequence.
    np.random.seed(12)
    rr, dr = model.recon(sinogram, init_recon=r3, max_iterations=5,
                         first_iteration=3, stop_threshold_change_pct=0.0)
    assert dr['recon_params']['num_iterations'] == 2
    # The restart continues improving on the run it resumed.
    assert dr['recon_params']['fm_rmse'][-1] <= d3['recon_params']['fm_rmse'][-1] + 1e-6


def test_weights_none_equals_explicit_ones():
    model = _small_model()
    _, sinogram = _box_problem(model)
    np.random.seed(5)
    r_none, _ = model.recon(sinogram, max_iterations=2, stop_threshold_change_pct=0.0)
    np.random.seed(5)
    r_ones, _ = model.recon(sinogram, weights=np.ones_like(sinogram),
                            max_iterations=2, stop_threshold_change_pct=0.0)
    rel = np.max(np.abs(r_none - r_ones)) / max(np.max(np.abs(r_ones)), 1e-30)
    assert rel < 1e-5, rel


def test_compile_on_off_value_equality():
    results = {}
    for mode in ("auto", "off"):
        model = _small_model(compile_mode=mode)
        _, sinogram = _box_problem(model)
        np.random.seed(7)
        recon, _ = model.recon(sinogram, max_iterations=2,
                               stop_threshold_change_pct=0.0)
        results[mode] = recon
    rel = (np.max(np.abs(results["auto"] - results["off"]))
           / max(np.max(np.abs(results["off"])), 1e-30))
    assert rel < 1e-4, rel


def test_zero_recon_nmae_does_not_raise():
    # An identically-zero recon gives recon_l1 == 0 and the
    # nmae division raised ZeroDivisionError where mbirjax's jnp division
    # yields nan and continues.  Construct that state directly: manual sigmas
    # (auto-regularization off, so the zero sinogram cannot zero sigma_y --
    # sigma_y = 0 raises in mbirjax too and is out of scope), a zero sinogram,
    # and a zero init leave the recon identically zero after the update.
    model = _small_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.set_params(sigma_y=1.0, sigma_x=1.0, sigma_prox=1.0)
    sino_shape = tuple(model.get_params('sinogram_shape'))
    np.random.seed(0)
    recon, _ = model.recon(np.zeros(sino_shape, dtype=np.float32),
                           init_recon=0, max_iterations=1,
                           stop_threshold_change_pct=0.0)
    assert np.all(np.isfinite(recon))
    assert float(np.abs(recon).max()) == 0.0


# ── differentiable-wrapper normalization ─────────────────────────────────────
def test_autograd_cpu_and_f64_leaves():
    model = _small_model("cpu")
    recon_shape = tuple(model.get_params('recon_shape'))
    y = torch.rand(tuple(model.get_params('sinogram_shape')))
    volume = torch.rand(recon_shape, dtype=torch.float64, requires_grad=True)
    sino = mbirtorch.forward_project_differentiable(model, volume)
    loss = torch.sum(sino * y.to(sino.device))
    loss.backward()
    assert volume.grad is not None
    assert volume.grad.dtype == torch.float64
    assert volume.grad.device == volume.device


# ── padded device-form invariance (guards for the future sharding port) ──────
# A sharding port zero-pads the view axis (and, for some geometries, the row
# axis) of device-form sinograms; weights pad with zeros too.  These tests lock
# the guards that keep padded entries from silently biasing statistics.
def _pad_axes(arr, view_extra=5, row_extra=3):
    pad = [(0, view_extra), (0, row_extra)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(np.asarray(arr), pad)


def test_auto_regularization_ignores_padded_views_and_rows():
    model = _small_model()
    _, sinogram = _box_problem(model)
    weights = mbirtorch.gen_weights(sinogram / np.max(sinogram),
                                    weight_type='transmission_root')
    reference = model.auto_set_regularization_params(sinogram, weights=weights)

    padded = _small_model().auto_set_regularization_params(
        _pad_axes(sinogram), weights=_pad_axes(weights))
    for name, ref_val in reference.items():
        assert padded[name] == pytest.approx(ref_val, rel=1e-6), name


def test_auto_regularization_ignores_padding_with_default_weights():
    model = _small_model()
    _, sinogram = _box_problem(model)
    reference = model.auto_set_regularization_params(sinogram)
    padded = _small_model().auto_set_regularization_params(_pad_axes(sinogram))
    for name, ref_val in reference.items():
        assert padded[name] == pytest.approx(ref_val, rel=1e-6), name


def test_forward_model_loss_real_count_normalization():
    rng = np.random.default_rng(0)
    err = torch.tensor(rng.normal(size=(6, 5, 4)).astype(np.float32))
    weights = torch.tensor(rng.uniform(0.5, 1.5, size=(6, 5, 4)).astype(np.float32))
    sigma_y = 0.7
    loss = mbirtorch.TomographyModel.get_forward_model_loss

    padded_err = torch.zeros((8, 7, 4))
    padded_err[:6, :5] = err
    padded_w = torch.zeros((8, 7, 4))
    padded_w[:6, :5] = weights

    ref = float(loss(err, sigma_y, weights))
    padded = float(loss(padded_err, sigma_y, padded_w,
                        num_real_elements=err.numel()))
    assert padded == pytest.approx(ref, rel=1e-6)

    # The scalar-weights branch is element-count-independent by construction.
    ref_s = float(loss(err, sigma_y, weights=2.0))
    padded_s = float(loss(padded_err, sigma_y, weights=2.0,
                          num_real_elements=err.numel()))
    assert padded_s == pytest.approx(ref_s, rel=1e-6)


def test_vcd_iteration_stats_real_size_normalization():
    rng = np.random.default_rng(1)
    err = torch.tensor(rng.normal(size=(6, 5, 4)).astype(np.float32))
    flat = torch.tensor(rng.normal(size=(10, 3)).astype(np.float32))
    padded_err = torch.zeros((8, 7, 4))
    padded_err[:6, :5] = err

    ref = mbirtorch.TomographyModel._vcd_iteration_stats(err, flat, 0.7)
    padded = mbirtorch.TomographyModel._vcd_iteration_stats(
        padded_err, flat, 0.7, num_real_elements=err.numel(),
        real_sino_size=float(err.numel()))
    for r, p in zip(ref, padded):
        assert float(p) == pytest.approx(float(r), rel=1e-6)


def test_placement_chokepoints_validate_and_place():
    # The _shard_*/_gather_* seams (single-device forms of the mbirjax
    # chokepoints): numpy in -> float32 tensor on the model device, with the
    # future sharded axis checked; gathers return host numpy.
    model = _small_model()
    sino_shape = tuple(model.get_params('sinogram_shape'))
    recon_shape = tuple(model.get_params('recon_shape'))

    sino = model._shard_sinogram(np.zeros(sino_shape, dtype=np.float64))
    assert torch.is_tensor(sino) and sino.dtype == torch.float32
    rec = model._shard_recon(np.zeros(recon_shape, dtype=np.float64))
    assert torch.is_tensor(rec) and rec.dtype == torch.float32
    flat = model._shard_recon(np.zeros((7, recon_shape[2]), dtype=np.float32))
    assert tuple(flat.shape) == (7, recon_shape[2])

    with pytest.raises(ValueError, match='view axis'):
        model._shard_sinogram(np.zeros((sino_shape[0] + 1,) + sino_shape[1:],
                                       dtype=np.float32))
    with pytest.raises(ValueError, match='slice axis'):
        model._shard_recon(np.zeros(recon_shape[:2] + (recon_shape[2] + 1,),
                                    dtype=np.float32))

    assert isinstance(model._gather_sinogram(sino), np.ndarray)
    assert isinstance(model._gather_recon(rec), np.ndarray)


def test_gen_weights_ct_model_places_on_device():
    model = _small_model()
    sino = np.ones(tuple(model.get_params('sinogram_shape')), dtype=np.float32)
    w = mbirtorch.gen_weights(sino, 'transmission_root', ct_model=model)
    assert torch.is_tensor(w)
    w_host = mbirtorch.gen_weights(sino, 'transmission_root')
    assert isinstance(w_host, np.ndarray)
    assert np.allclose(w.cpu().numpy(), w_host)


def test_clear_cache_empties_and_recreates(tmp_path):
    root = tmp_path / "fake_mbirtorch"
    (root / "torch_cache" / "sub").mkdir(parents=True)
    (root / "torch_cache" / "sub" / "artifact.bin").write_bytes(b"x" * 128)
    cleared = mbirtorch.clear_cache(_root=root)
    assert cleared == str(root)
    assert root.is_dir() and list(root.iterdir()) == []
    # Clearing an absent directory just (re)creates it.
    mbirtorch.clear_cache(_root=root / "never_existed")
    assert (root / "never_existed").is_dir()


def test_vcd_checkpoint_resume_matches_continuous():
    # A chunked run (3 + 3 + 3 iterations through checkpoints) must reproduce
    # a continuous 9-iteration run: the partitions are shared, the
    # per-iteration subset shuffles consume the same global np.random stream
    # across the call boundaries, and the resume path picks up the exact
    # (recon, error_sinogram, fm_hessian) state.  The comparison is
    # tolerance-based, not bitwise: even two identical seeded continuous runs
    # differ by ~1e-6 on CPU (parallel index_add_ reduction ordering), and the
    # chunked run need only sit at that same rerun-jitter floor.
    #
    # The resume makes NO defensive copies (the advertised contract, Greg
    # 2026-08-05): the caller's arrays become the loop's working buffers and
    # are updated in place where memory-compatible, so the checkpoint dict and
    # the passed init_recon reflect the RESUMED state afterward -- which is
    # exactly what makes chaining through them consistent.
    model = _small_model()
    _, sinogram = _box_problem(model)
    weights = mbirtorch.gen_weights(sinogram / np.max(sinogram),
                                    weight_type='transmission_root')
    np.random.seed(3)
    (_, _, _, partitions, seq, _, _) = model.initialize_recon(
        sinogram, weights=weights, max_iterations=9)

    np.random.seed(7)
    ref, ref_stats = model.vcd_recon(sinogram, partitions, seq, 0.0,
                                     weights=weights, init_recon=0)

    np.random.seed(7)
    r3, s3, ck = model.vcd_recon(sinogram, partitions, seq[:3], 0.0,
                                 weights=weights, init_recon=0,
                                 return_checkpoint=True)
    ck_err_before = ck['error_sinogram'].clone()
    r6, s6, ck6 = model.vcd_recon(sinogram, partitions, seq[3:6], 0.0,
                                  weights=weights, init_recon=r3,
                                  init_error_sinogram=ck['error_sinogram'],
                                  fm_hessian=ck['fm_hessian'],
                                  first_iteration=3, return_checkpoint=True)
    # In-place contract (CPU, memory-compatible inputs): the returned recon
    # shares init_recon's storage, and the checkpoint now holds the resumed
    # error state.
    assert r6.data_ptr() == r3.data_ptr() and torch.equal(r6, r3)
    assert not torch.equal(ck['error_sinogram'], ck_err_before)
    assert torch.equal(ck['error_sinogram'], ck6['error_sinogram'])

    r9, s9 = model.vcd_recon(sinogram, partitions, seq[6:], 0.0,
                             weights=weights, init_recon=r6,
                             init_error_sinogram=ck6['error_sinogram'],
                             fm_hessian=ck6['fm_hessian'], first_iteration=6)

    assert float(torch.max(torch.abs(r9 - ref))) < 1e-4
    fm_chunked = np.concatenate([s3[0], s6[0], s9[0]])
    assert np.allclose(fm_chunked, ref_stats[0], rtol=1e-4, atol=1e-5)


def test_vcd_resume_requires_init_recon():
    model = _small_model()
    _, sinogram = _box_problem(model)
    np.random.seed(0)
    (_, _, _, partitions, seq, _, _) = model.initialize_recon(
        sinogram, max_iterations=2)
    with pytest.raises(ValueError, match='init_error_sinogram requires init_recon'):
        model.vcd_recon(sinogram, partitions, seq, 0.0,
                        init_error_sinogram=np.zeros_like(sinogram))


def test_sino_ones_device_form_seam():
    # The constant-weights Hessian path must build its ones sinogram through
    # the device-form seam (real entries 1, padded entries 0 under a future
    # sharding port) -- never a bare torch.ones at whatever shape the device
    # arrays have.  Lock the seam contract and the weights=None wiring.
    model = _small_model()
    sino_shape = tuple(model.get_params('sinogram_shape'))
    ones = model._sino_ones_device_form()
    assert tuple(ones.shape) == sino_shape
    assert ones.dtype == torch.float32
    assert float(ones.min()) == 1.0 and float(ones.max()) == 1.0
    ones64 = model._sino_ones_device_form(torch.zeros(2, dtype=torch.float64))
    assert ones64.dtype == torch.float64

    h_none = model.compute_hessian_diagonal()
    h_ones = model.compute_hessian_diagonal(
        weights=np.ones(sino_shape, dtype=np.float32))
    assert np.allclose(h_none, h_ones, rtol=1e-6, atol=0)


def test_compute_prior_loss_records_pm_loss():
    # The compute_prior_loss path (ported with the vcd_recon sweep): pm_loss
    # recorded per iteration, positive and finite; qggmrf_loss cross-checked
    # against mbirjax at 2.3e-7 rel (2026-08-05, seeded 12x11x10 volume).
    # mbirjax parity quirk kept: recorded only at verbose >= 1.
    model = _small_model()
    _, sinogram = _box_problem(model)
    np.random.seed(0)
    (_, _, _, partitions, seq, _, _) = model.initialize_recon(
        sinogram, max_iterations=2)

    model.set_params(no_warning=True, verbose=1)
    np.random.seed(1)
    _, losses = model.vcd_recon(sinogram, partitions, seq, 0.0,
                                compute_prior_loss=True)
    pm = losses[1]
    assert pm.shape == (2,) and np.all(np.isfinite(pm)) and np.all(pm > 0)

    model.set_params(no_warning=True, verbose=0)
    np.random.seed(1)
    _, losses0 = model.vcd_recon(sinogram, partitions, seq, 0.0,
                                 compute_prior_loss=True)
    assert np.all(losses0[1] == 0.0)

    # The loss itself penalizes roughness.
    params = ((1.0,) * 4 + (0.8,) * 2, 0.3, 2.0, 1.2, 1.0)
    smooth = np.ones((8, 8, 8), dtype=np.float32)
    rough = np.random.RandomState(2).rand(8, 8, 8).astype(np.float32)
    assert mbirtorch.qggmrf_loss(rough, params) > mbirtorch.qggmrf_loss(smooth, params)


def test_get_memory_stats_structure():
    # One dict per processor, devices first then 'CPU', each with byte counts;
    # printing to a file-like works (the vcd_recon verbose>=2 route).
    import io
    stats = mbirtorch.get_memory_stats(print_results=False)
    assert stats[-1]['id'] == 'CPU'
    for entry in stats:
        for key, value in entry.items():
            if key != 'id':
                assert isinstance(value, int) and value >= 0, (entry['id'], key)
    buf = io.StringIO()
    mbirtorch.get_memory_stats(file=buf)
    text = buf.getvalue()
    assert 'CPU' in text and 'bytes_in_use' in text and 'GB' in text


def test_vcd_verbose2_memory_dump_runs():
    model = _small_model()
    model.set_params(no_warning=True, verbose=2)
    _, sinogram = _box_problem(model)
    np.random.seed(0)
    (_, _, _, partitions, seq, _, _) = model.initialize_recon(
        sinogram, max_iterations=1)
    np.random.seed(1)
    recon, _ = model.vcd_recon(sinogram, partitions, seq, 0.0, init_recon=0)
    assert np.all(np.isfinite(recon.cpu().numpy()))


def test_apply_update_functional_no_copy():
    # _apply_update returns the state tensors for a functional interface; the
    # returns are the SAME storage (no transient copy) and the in-place
    # updates are applied.  (The chained-resume test guards the same storage
    # stability end-to-end through the compiled loop.)
    from mbirtorch.tomography_model import _apply_update
    flat_recon = torch.zeros(5, 3)
    error_sinogram = torch.ones(2, 2, 2)
    idx = torch.tensor([1, 3])
    delta = torch.ones(2, 3)
    alpha = torch.tensor(0.5)
    delta_sinogram = torch.full((2, 2, 2), 0.2)
    fr, es, delta_sumsq, ell1 = _apply_update(
        flat_recon, error_sinogram, idx, delta, alpha, delta_sinogram)
    assert fr.data_ptr() == flat_recon.data_ptr()
    assert es.data_ptr() == error_sinogram.data_ptr()
    assert float(flat_recon[1, 0]) == 1.0 and float(flat_recon[0, 0]) == 0.0
    assert abs(float(error_sinogram[0, 0, 0]) - 0.9) < 1e-6
    assert float(ell1) == 6.0
    assert delta_sumsq.shape == (3,) and float(delta_sumsq[0]) == 2.0
