"""The standing kernel-times-sharding gate: two or more CUDA devices.

The composed single-device gates exercise the Triton kernels thoroughly and
cannot see this combination, because a single device never runs the banded
multi-device drivers at all -- the trivial placement short-circuits to the
plain projectors.  The multi-device value gates, in turn, predate the
kernels.  Nothing measured the two together until the isolation matrix that
prompted the interim selection rule, and this file is that matrix promoted to
a standing gate.

Two things are asserted.  A two-device reconstruction with BOTH kernels bound
must match the single-device torch-body reference within the multi-device
float floor; that is the composition a multi-GPU user actually runs.  And a
model pinned to a device other than cuda:0 must give the same values, which is
the case the launch-context bug below reached.

The forward's history is the reason this file exists.  Its kernels once read
order one against the torch bodies here, because a Triton launch targets the
launching thread's current device and the banded drivers launch from worker
threads.  The wrappers now bracket their launches on the tensors' device, and
the diagnosis, probe matrix, and repair live in the kernel-sharding findings
in the plans repo.

A protocol note this campaign paid for: `compile_mode='off'` does NOT disable
the kernels.  Selection is availability-driven, not compile-driven, so an arm
that intends the plain torch engine forces the torch bodies explicitly rather
than assuming eager means unkernelled.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch.cone_beam import _cone_back_view_batch, _cone_forward_view_batch
from mbirtorch.multiaxis_parallel import (_multiaxis_back_view_batch,
                                          _multiaxis_forward_view_batch)
from mbirtorch.parallel_beam import (_parallel_back_view_batch,
                                     _parallel_forward_view_batch)

requires_two_cuda = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="the kernel-times-sharding gate needs at least two CUDA devices")

# A DIVIDING cell, so the measured n>1 spread IS this cell's own float floor
# rather than a padding effect (the per-cell calibration rule).
CELL = (256, 64, 64)
VCD_ITERATIONS = 3
VCD_SEED = 4321
# The established multi-device float-divergence scale for cells of this class.
FLOOR = 5e-3


# The (forward, back) bodies of each geometry, keyed by name so an arm that
# names a geometry cannot silently get another one's bodies.
TORCH_BODIES = {
    "parallel": (_parallel_forward_view_batch, _parallel_back_view_batch),
    "cone": (_cone_forward_view_batch, _cone_back_view_batch),
    "multiaxis": (_multiaxis_forward_view_batch, _multiaxis_back_view_batch),
}


def _build(geometry):
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, CELL[0], endpoint=False)
        model = mbirtorch.ParallelBeamModel(CELL, angles, compile_mode="off")
    elif geometry == "multiaxis":
        # Two angles per view: the azimuth over a half turn, and an elevation
        # swept across +/- 0.5 radians, which is how this geometry is built
        # everywhere else in the suite.  The cell's own slice count follows
        # from that range (the recon height divides by the smallest
        # |cos(elevation)|), and it still divides by two.
        azimuth = np.linspace(0, np.pi, CELL[0], endpoint=False)
        elevation = np.linspace(-0.5, 0.5, CELL[0])
        model = mbirtorch.MultiAxisParallelModel(
            CELL, np.stack([azimuth, elevation], axis=1), compile_mode="off")
    else:
        angles = np.linspace(0, 2 * np.pi, CELL[0], endpoint=False)
        sdd = 4 * CELL[2]
        model = mbirtorch.ConeBeamModel(CELL, angles, source_detector_dist=sdd,
                                        source_iso_dist=sdd, compile_mode="off")
    model.set_params(no_warning=True, verbose=0)
    return model


def _force_torch_bodies(model, geometry):
    """Bind the plain torch bodies in BOTH directions.

    Setting compile_mode='off' does not do this: the kernels are chosen by
    their availability gates, not by the compile setting.
    """
    bodies = TORCH_BODIES[geometry]
    model._view_batch_bodies = lambda: bodies
    model.create_projectors()


def _reconstruct(model, sinogram, weights):
    np.random.seed(VCD_SEED)
    recon, _info = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0)
    return np.asarray(recon, dtype=np.float32)


def _rel(reference, other):
    return float(np.max(np.abs(reference - other))
                 / max(float(np.max(np.abs(reference))), 1e-30))


@pytest.fixture(scope="module")
def problem():
    """One phantom and sinogram per geometry, built on a single device."""
    data = {}
    for geometry in ("parallel", "cone", "multiaxis"):
        model = _build(geometry)
        model.configure_devices(1)
        recon_shape = tuple(model.get_params("recon_shape"))
        phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = np.asarray(model.forward_project(phantom), dtype=np.float32)
        weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)
        data[geometry] = (sinogram, weights)
    return data


@requires_two_cuda
@pytest.mark.parametrize("geometry", ["parallel", "cone"])
def test_the_default_selection_matches_the_torch_bodies_under_sharding(
        geometry, problem):
    """The composition a multi-GPU user actually gets.

    The default selection binds BOTH kernels wherever the availability gates
    pass, so this arm covers the forward and back kernels together and must
    sit at the same floor as the torch-body arms.  Before the launch-context
    repair the forward kernel read order one here, non-reproducibly; the
    bracket brought it to the kernel-parity class, measured at 3.4e-07 and
    1.1e-06 on two H100s.
    """
    sinogram, weights = problem[geometry]

    plain = _build(geometry)
    plain.configure_devices(2)
    _force_torch_bodies(plain, geometry)
    torch_arm = _reconstruct(plain, sinogram, weights)

    shipped = _build(geometry)
    shipped.configure_devices(2)
    # The arm check: this arm exists to measure the KERNELS, so a silent
    # availability decline must fail loudly rather than compare torch with
    # torch and pass vacuously.
    from mbirtorch.triton_cone import triton
    if triton is not None:
        fwd, back = shipped._view_batch_bodies()
        assert "triton" in fwd.__name__ and "triton" in back.__name__, (
            f"kernels not bound with triton importable: {fwd.__name__}, "
            f"{back.__name__}")
    default_arm = _reconstruct(shipped, sinogram, weights)

    rel = _rel(torch_arm, default_arm)
    assert rel < FLOOR, f"{geometry} default selection: {rel:.3e}"


@requires_two_cuda
@pytest.mark.parametrize("geometry", ["parallel", "cone", "multiaxis"])
def test_kernels_hold_on_a_single_nonzero_device(geometry, problem):
    """The trivial-placement twin of the launch-context defect.

    A model pinned to cuda:1 launches its kernels from the main thread,
    whose current device is 0, so an unbracketed launch would race exactly
    as the banded workers' did -- with no banded driver involved and no
    reduce to hide it.  The n=1 composed gates all run on device 0, where
    the thread-current and tensor devices agree, so only a nonzero pin can
    see this.  Projections suffice: the race showed at order one in a single
    forward call.
    """
    sinogram, _weights = problem[geometry]

    torch_model = _build(geometry)
    torch_model.configure_devices(devices=["cuda:1"])
    _force_torch_bodies(torch_model, geometry)
    recon_shape = tuple(torch_model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    torch_fwd = np.asarray(torch_model.forward_project(phantom), np.float32)
    torch_back = np.asarray(torch_model.back_project(sinogram), np.float32)

    kernel_model = _build(geometry)
    kernel_model.configure_devices(devices=["cuda:1"])
    # The arm check (see the default-selection test): the kernel arm must
    # not silently degrade to a torch-vs-torch comparison.
    from mbirtorch.triton_cone import triton
    if triton is not None:
        fwd, back = kernel_model._view_batch_bodies()
        assert "triton" in fwd.__name__ and "triton" in back.__name__, (
            f"kernels not bound on cuda:1 with triton importable: "
            f"{fwd.__name__}, {back.__name__}")
    kernel_fwd = np.asarray(kernel_model.forward_project(phantom), np.float32)
    kernel_back = np.asarray(kernel_model.back_project(sinogram), np.float32)

    fwd_rel = _rel(torch_fwd, kernel_fwd)
    back_rel = _rel(torch_back, kernel_back)
    assert fwd_rel < FLOOR, f"{geometry} forward on cuda:1: {fwd_rel:.3e}"
    assert back_rel < FLOOR, f"{geometry} back on cuda:1: {back_rel:.3e}"
