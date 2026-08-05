"""Panel-review regression tests: set_params semantics, engine paths, and the
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
    # The panel finding: an identically-zero recon gives recon_l1 == 0 and the
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
