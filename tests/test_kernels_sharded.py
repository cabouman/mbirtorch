"""The standing kernel-times-sharding gate: two or more CUDA devices.

The composed single-device gates exercise the Triton kernels thoroughly and
cannot see this combination, because a single device never runs the banded
multi-device drivers at all -- the trivial placement short-circuits to the
plain projectors.  The multi-device value gates, in turn, predate the
kernels.  Nothing measured the two together until the isolation matrix that
prompted the interim selection rule, and this file is that matrix promoted to
a standing gate.

Four things are asserted, in the order they matter.

The torch-body arms must sit at the multi-device float floor, which is what
says the ENGINE is sound and gives every other arm its reference.  The
back-kernel arms must match the torch arms within that floor, which is the
evidence that earned the back kernels their default-on status at every device
count.  The forward-kernel arms must match the torch arms the same way, which
is the bar the repaired launch path passed when the interim selection rule
retired.  And the selection contract must hold on real hardware: the same
bodies bind at one device and at two.

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
from mbirtorch.parallel_beam import (_parallel_back_view_batch,
                                     _parallel_forward_view_batch)
# The kernel modules import without triton; only calling a wrapper needs it,
# and every test below is CUDA-gated.
from mbirtorch.triton_cone import (_cone_back_view_batch_triton,
                                   _cone_forward_view_batch_triton)
from mbirtorch.triton_parallel import (_parallel_back_view_batch_triton,
                                       _parallel_forward_view_batch_triton)

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
# How closely a back-kernel arm must track its torch arm.  The two differ only
# by the back body, and the isolation matrix measured them equal to four
# significant figures, so this is loose by two orders of magnitude and still
# catches any real divergence.
BACK_KERNEL_TOLERANCE = 1e-2


def _build(geometry):
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, CELL[0], endpoint=False)
        model = mbirtorch.ParallelBeamModel(CELL, angles, compile_mode="off")
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
    bodies = ((_parallel_forward_view_batch, _parallel_back_view_batch)
              if geometry == "parallel"
              else (_cone_forward_view_batch, _cone_back_view_batch))
    model._view_batch_bodies = lambda: bodies
    model.create_projectors()


def _force_one_kernel(model, geometry, direction):
    """Bind ONE kernel body against the torch body in the other direction.

    A mixed arm blames one direction on its own, which is how the isolation
    matrix separated the forward from the back.  The default selection binds
    both kernels, so a single-direction arm must be forced.
    """
    if geometry == "parallel":
        pair = ((_parallel_forward_view_batch_triton, _parallel_back_view_batch)
                if direction == "forward"
                else (_parallel_forward_view_batch,
                      _parallel_back_view_batch_triton))
    else:
        pair = ((_cone_forward_view_batch_triton, _cone_back_view_batch)
                if direction == "forward"
                else (_cone_forward_view_batch, _cone_back_view_batch_triton))
    model._view_batch_bodies = lambda: pair
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
    for geometry in ("parallel", "cone"):
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
def test_torch_bodies_hold_the_multi_device_float_floor(geometry, problem):
    """The engine's own reference arm.

    With no kernel bound in either direction, a sharded reconstruction must
    match the single-device one at the established float floor.  Every other
    arm in this file is read against this one.
    """
    sinogram, weights = problem[geometry]
    reference = None
    for count in (1, 2):
        model = _build(geometry)
        model.configure_devices(count)
        _force_torch_bodies(model, geometry)
        result = _reconstruct(model, sinogram, weights)
        if count == 1:
            reference = result
            continue
        rel = _rel(reference, result)
        assert rel < FLOOR, f"{geometry} torch bodies at n={count}: {rel:.3e}"


@requires_two_cuda
@pytest.mark.parametrize("geometry", ["parallel", "cone"])
def test_the_back_kernel_matches_the_torch_bodies_under_sharding(geometry,
                                                                 problem):
    """The evidence the back kernel's default-on status rests on.

    The two arms differ only in the back body.  The isolation matrix measured
    them equal to four significant figures at two and four devices in both
    geometries.
    """
    sinogram, weights = problem[geometry]

    plain = _build(geometry)
    plain.configure_devices(2)
    _force_torch_bodies(plain, geometry)
    torch_arm = _reconstruct(plain, sinogram, weights)

    mixed = _build(geometry)
    mixed.configure_devices(2)
    _force_one_kernel(mixed, geometry, "back")
    kernel_arm = _reconstruct(mixed, sinogram, weights)

    rel = _rel(torch_arm, kernel_arm)
    assert rel < BACK_KERNEL_TOLERANCE, f"{geometry} back kernel: {rel:.3e}"


@requires_two_cuda
@pytest.mark.parametrize("geometry", ["parallel", "cone"])
def test_the_forward_kernel_matches_the_torch_bodies_under_sharding(geometry,
                                                                    problem):
    """The repaired launch path, held to the bar that retired the interim.

    The two arms differ only in the forward body.  Before the repair this
    comparison read order one, non-reproducibly; the launch-context bracket
    brought it to the kernel-parity class, measured at 3.4e-07 and 1.1e-06
    on two H100s.  The floor here is the standing multi-device envelope,
    loose by three orders against that measurement.
    """
    sinogram, weights = problem[geometry]

    plain = _build(geometry)
    plain.configure_devices(2)
    _force_torch_bodies(plain, geometry)
    torch_arm = _reconstruct(plain, sinogram, weights)

    mixed = _build(geometry)
    mixed.configure_devices(2)
    _force_one_kernel(mixed, geometry, "forward")
    kernel_arm = _reconstruct(mixed, sinogram, weights)

    rel = _rel(torch_arm, kernel_arm)
    assert rel < FLOOR, f"{geometry} forward kernel: {rel:.3e}"


@requires_two_cuda
@pytest.mark.parametrize("geometry", ["parallel", "cone"])
def test_the_default_selection_matches_the_torch_bodies_under_sharding(
        geometry, problem):
    """The composition a multi-GPU user actually gets.

    The default selection binds BOTH kernels wherever the availability gates
    pass.  This arm is the mixed arms' composition, and it must sit at the
    same floor.
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
@pytest.mark.parametrize("geometry", ["parallel", "cone"])
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


@requires_two_cuda
@pytest.mark.parametrize("geometry", ["parallel", "cone"])
def test_kernel_selection_is_the_same_at_one_and_two_devices(geometry):
    """The selection contract, asserted on real hardware.

    The CPU tests pin the layout-independence RULE by forcing the
    availability gates.  This pins what a real multi-GPU machine actually
    binds, which is the thing a user gets: the same bodies at one device and
    at two, and the kernels wherever the gates pass.
    """
    model = _build(geometry)

    model.configure_devices(1)
    single_forward, single_back = model._view_batch_bodies()
    model.configure_devices(2)
    sharded_forward, sharded_back = model._view_batch_bodies()

    assert sharded_forward is single_forward
    assert sharded_back is single_back
    if "triton" in single_back.__name__:
        # Where the back kernel is usable, the forward's own gate passes on
        # the same toolchain, so both kernels are the expected binding.
        assert "triton" in single_forward.__name__
