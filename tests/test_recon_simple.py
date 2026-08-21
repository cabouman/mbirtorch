"""Gates for the one-call reconstruction functions.

``recon_simple_parallel`` and ``recon_simple_cone`` build a model, set
sharpness, and call recon.  These tests hold that contract: each returns what
recon returns, matches the equivalent model-based calls with the same seed,
accepts weights and tensor inputs, applies sharpness, and reproduces the
scaled direct reconstruction at max_iterations=0.

The problem is deliberately tiny so both geometries run in a few seconds.
"""

import types

import numpy as np
import pytest
import torch

import mbirtorch

# One small synthetic cell, shared by both geometries.
_CELL = dict(num_views=16, num_det_rows=8, num_det_channels=16)

# Both paths run the same arithmetic on the same device with the same seed, so
# the only expected difference is float summation order.  Tolerances are
# absolute, scaled to the volume's largest value.
_MATCH_TOL = 1e-5


def _rel_max(out, ref):
    """Largest absolute difference, relative to the reference's largest value."""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def _close(out, ref, tol=_MATCH_TOL):
    """allclose at a tolerance scaled to the reference's largest value."""
    scale = max(float(np.max(np.abs(np.asarray(ref)))), 1e-30)
    return np.allclose(np.asarray(out), np.asarray(ref), rtol=0, atol=tol * scale)


@pytest.fixture(scope="module", params=["parallel", "cone"])
def case(request):
    """A small sinogram plus the two ways to reconstruct it: the one-call
    function and the equivalent model-based calls."""
    geometry = request.param
    _, sinogram, params = mbirtorch.generate_demo_data(
        model_type=geometry, object_type="cube", **_CELL)
    angles = params["angles"]

    if geometry == "parallel":
        def simple(**kwargs):
            return mbirtorch.recon_simple_parallel(sinogram, angles, **kwargs)

        def make_model():
            return mbirtorch.ParallelBeamModel(sinogram.shape, angles)

        def direct(model):
            return model.recon_fbp(sinogram)
    else:
        source_detector_dist = params["source_detector_dist"]
        source_iso_dist = params["source_iso_dist"]

        def simple(**kwargs):
            return mbirtorch.recon_simple_cone(
                sinogram, angles, source_detector_dist, source_iso_dist,
                **kwargs)

        def make_model():
            return mbirtorch.ConeBeamModel(
                sinogram.shape, angles,
                source_detector_dist=source_detector_dist,
                source_iso_dist=source_iso_dist)

        def direct(model):
            return model.recon_fdk(sinogram)

    recon_shape = tuple(make_model().get_params("recon_shape"))
    return types.SimpleNamespace(geometry=geometry, sinogram=sinogram,
                                 angles=angles, simple=simple,
                                 make_model=make_model, direct=direct,
                                 recon_shape=recon_shape)


def test_returns_recon_and_dict(case):
    np.random.seed(0)
    recon, recon_dict = case.simple(max_iterations=3)
    assert isinstance(recon, np.ndarray)
    assert recon.shape == case.recon_shape
    assert np.isfinite(recon).all() and np.max(np.abs(recon)) > 0
    # The four entries the docstring promises, as recon documents them.
    assert set(recon_dict) == {"recon_params", "recon_log", "notes",
                               "model_params"}
    assert recon_dict["recon_params"]["num_iterations"] == 3


def test_matches_model_path(case):
    """The one-call function equals building the model and calling recon."""
    np.random.seed(0)
    simple_recon, _ = case.simple(sharpness=0.5, max_iterations=3)

    np.random.seed(0)
    model = case.make_model()
    model.set_params(sharpness=0.5)
    model_recon, _ = model.recon(case.sinogram, max_iterations=3)

    print(f"{case.geometry}: one-call vs model rel_max = "
          f"{_rel_max(simple_recon, model_recon):.2e}")
    assert _close(simple_recon, model_recon)


def test_weights_of_ones_match_no_weights(case):
    """The weights path runs, and all-ones weights reproduce the default."""
    ones = np.ones_like(case.sinogram)
    np.random.seed(0)
    weighted, _ = case.simple(weights=ones, max_iterations=3)
    np.random.seed(0)
    unweighted, _ = case.simple(max_iterations=3)
    print(f"{case.geometry}: ones-weights vs no-weights rel_max = "
          f"{_rel_max(weighted, unweighted):.2e}")
    assert _close(weighted, unweighted)


def test_zero_iterations_is_scaled_direct(case):
    """max_iterations=0 returns the direct reconstruction scaled to fit the
    data: the iterative loop's initializer, handed back before any iteration.
    The scale is the one that minimizes the error sinogram, so it also has to
    fit the data at least as well as the unscaled direct reconstruction."""
    np.random.seed(0)
    recon, recon_dict = case.simple(max_iterations=0)
    assert recon.shape == case.recon_shape
    assert np.isfinite(recon).all() and np.max(np.abs(recon)) > 0
    assert recon_dict["recon_params"]["num_iterations"] == 0

    model = case.make_model()
    direct_recon = case.direct(model)
    scale = float(np.sum(recon * direct_recon) / np.sum(direct_recon ** 2))
    print(f"{case.geometry}: zero-iteration scale = {scale:.4f}, rel_max after "
          f"scaling = {_rel_max(recon, scale * direct_recon):.2e}")
    assert scale > 0
    assert _close(recon, scale * direct_recon)

    def sino_error(volume):
        return float(np.linalg.norm(model.forward_project(volume)
                                    - case.sinogram))

    assert sino_error(recon) <= sino_error(direct_recon)


def test_sharpness_is_applied(case):
    """The sharpness passed in reaches the model: it is recorded in the
    parameter snapshot, and the prior width sigma_x scales as 2 ** sharpness,
    so raising sharpness by 2 multiplies sigma_x by 4."""
    np.random.seed(0)
    _, soft_dict = case.simple(sharpness=0.0, max_iterations=0)
    np.random.seed(0)
    _, sharp_dict = case.simple(sharpness=2.0, max_iterations=0)

    assert soft_dict["model_params"]["sharpness"] == 0.0
    assert sharp_dict["model_params"]["sharpness"] == 2.0
    ratio = (sharp_dict["model_params"]["sigma_x"]
             / soft_dict["model_params"]["sigma_x"])
    print(f"{case.geometry}: sigma_x ratio = {ratio:.4f}")
    assert np.isclose(ratio, 4.0, rtol=1e-5)


def test_accepts_tensor_inputs(case):
    """A torch sinogram and torch angles give the same result as numpy ones."""
    np.random.seed(0)
    from_numpy, _ = case.simple(max_iterations=0)

    sinogram_t = torch.as_tensor(case.sinogram)
    angles_t = torch.as_tensor(np.asarray(case.angles, dtype=np.float32))
    np.random.seed(0)
    if case.geometry == "parallel":
        from_tensor, tensor_dict = mbirtorch.recon_simple_parallel(
            sinogram_t, angles_t, max_iterations=0)
    else:
        model = case.make_model()
        from_tensor, tensor_dict = mbirtorch.recon_simple_cone(
            sinogram_t, angles_t,
            model.get_params("source_detector_dist"),
            model.get_params("source_iso_dist"), max_iterations=0)
    print(f"{case.geometry}: tensor vs numpy inputs rel_max = "
          f"{_rel_max(from_tensor, from_numpy):.2e}")
    assert _close(from_tensor, from_numpy)
    # The recorded shape is a plain tuple whether the sinogram came in as a
    # numpy array or as a tensor.
    recorded_shape = tensor_dict["model_params"]["sinogram_shape"]
    assert type(recorded_shape) is tuple
    assert recorded_shape == tuple(case.sinogram.shape)


def test_refuses_a_sinogram_in_the_divided_form():
    """The one-call functions read the geometry off the sinogram's shape, which
    an array divided across devices does not have.  Both refuse it by name
    rather than failing on a missing attribute in the line that builds the
    model.  Two 'virtual' CPU devices build the divided form, so no real
    second device is needed."""
    from mbirtorch._sharding import Placement, Shards

    sinogram = np.random.RandomState(0).rand(4, 6, 8).astype(np.float32)
    angles = np.linspace(0, np.pi, sinogram.shape[0], endpoint=False)
    placement = Placement(['cpu', 'cpu'], axis=0, axis_len=sinogram.shape[0])
    divided = Shards([torch.as_tensor(sinogram[start:end])
                      for _, (start, end) in placement.shard_ranges()], placement)

    calls = [
        ('recon_simple_parallel', 'sinogram',
         lambda: mbirtorch.recon_simple_parallel(divided, angles)),
        ('recon_simple_parallel', 'weights',
         lambda: mbirtorch.recon_simple_parallel(sinogram, angles,
                                                 weights=divided)),
        ('recon_simple_cone', 'sinogram',
         lambda: mbirtorch.recon_simple_cone(divided, angles, 600.0, 400.0)),
        ('recon_simple_cone', 'weights',
         lambda: mbirtorch.recon_simple_cone(sinogram, angles, 600.0, 400.0,
                                             weights=divided)),
    ]
    for function_name, argument_name, call in calls:
        with pytest.raises(TypeError) as refusal:
            call()
        message = str(refusal.value)
        assert function_name in message, message
        assert argument_name in message, message
        assert 'divided device form' in message, message
        assert 'shards.gather()' in message, message
