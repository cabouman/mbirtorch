"""Gates for the one-call reconstruction functions.

``recon_simple_parallel`` and ``recon_simple_cone`` build a model, set
sharpness, and call recon.  These tests hold that contract: each matches the
equivalent model-based calls with the same seed, and reproduces the scaled
direct reconstruction at max_iterations=0.

The problem is deliberately tiny so both geometries run in a few seconds.
"""

import types

import numpy as np
import pytest

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
