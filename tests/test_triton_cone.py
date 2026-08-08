"""Value gates for the Triton cone kernels (the back and forward bodies).

Each kernel is an alternative view-batch BODY, so every gate here compares it
against the torch body it replaces at the same inputs: parity across the
geometry variants (flat, curved, helical) at every coefficient power the body
takes, parity across the banded seams, the explicit adjointness pairing
against the OTHER direction's torch body, and the poison-the-padding class (a
pixel count that is not a multiple of the kernel's pixel tile, where the
padded lanes must contribute exactly nothing).  Tolerances follow the design's
value gate -- rel 1e-5 on the gradient path, 1e-4 at coeff_power 2 -- which is
the mbirjax rounding carve-out for the kernels' sqrt-vs-atan2 cone divisor.

The forward kernel scatters with float atomics, so its sums are reordered from
launch to launch and it is not bit-reproducible; 1e-5 covers that too, and
test_cone_forward_kernel_repeat_consistency measures the run-to-run spread
rather than assuming it.

Everything that launches a kernel needs CUDA and skips without it; the
availability gates themselves are exercised on any machine.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import kernel_availability
from mbirtorch.cone_beam import _cone_back_view_batch, _cone_forward_view_batch
from mbirtorch.triton_cone import (CONE_BACK_BLOCK_P, CONE_FWD_BLOCK_P,
                                   _cone_back_view_batch_triton,
                                   _cone_forward_view_batch_triton)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the hand-written Triton kernels need a CUDA device")


def _cone_model(curved=False, helical=False, cell=(6, 12, 12), device="cuda",
                compile_mode="off"):
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    z_shifts = (np.linspace(-1.5, 1.5, cell[0]).astype(np.float32)
                if helical else None)
    model = mbirtorch.ConeBeamModel(cell, angles,
                                    source_detector_dist=4 * cell[2],
                                    source_iso_dist=2 * cell[2],
                                    helical_z_shifts=z_shifts,
                                    use_curved_detector=curved, 
                                    compile_mode=compile_mode)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


def _body_inputs(model, num_pixels=None, seed=0):
    """(sinogram, pixel_indices, view_params, body kwargs) for ONE view batch
    covering every view -- the shape the driver hands a body."""
    device = model.torch_device
    recon_shape = model.get_params('recon_shape')
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    if num_pixels is not None:
        pixel_indices = pixel_indices[:num_pixels]
    # A private generator: the seeded recon gates read the global streams.
    generator = torch.Generator().manual_seed(seed)
    sinogram = torch.rand(tuple(model.get_params('sinogram_shape')),
                          generator=generator).to(device)
    view_params = torch.as_tensor(model.get_params('view_params_array'),
                                  dtype=torch.float32, device=device)
    return sinogram, pixel_indices, view_params, model._view_batch_args()


def _voxel_values(model, pixel_indices, seed=3, num_cols=None):
    """(P, cols) voxel cylinders -- the shape the driver hands a forward
    body."""
    if num_cols is None:
        num_cols = int(model.get_params('recon_shape')[2])
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((int(pixel_indices.shape[0]), num_cols),
                      generator=generator).to(model.torch_device)


def _rel_max(out, ref):
    return float((out - ref).abs().max() / max(float(ref.abs().max()), 1e-30))


@requires_cuda
@pytest.mark.parametrize("geometry", ["flat", "curved", "helical"])
@pytest.mark.parametrize("coeff_power,tol", [(1, 1e-5), (2, 1e-4)])
def test_cone_back_kernel_parity(geometry, coeff_power, tol):
    # Curved detectors and helical z shifts reach the kernel only through the
    # two eager builders it shares with the torch body, so all three variants
    # exercise the same kernel with different contract values.
    model = _cone_model(curved=(geometry == "curved"),
                        helical=(geometry == "helical"))
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    reference = _cone_back_view_batch(sinogram, pixel_indices, view_params,
                                      coeff_power=coeff_power, **args)
    kernel_out = _cone_back_view_batch_triton(sinogram, pixel_indices,
                                              view_params,
                                              coeff_power=coeff_power, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"cone back triton parity ({geometry}, coeff_power={coeff_power}): "
          f"rel_max = {rel:.2e}")
    assert rel <= tol


@requires_cuda
def test_cone_back_kernel_banded_parity():
    # The banded seam: the z geometry is anchored on the FULL slice count, so
    # a tiling of the slice axis must reassemble the unbanded partial exactly
    # (each band owns its own output columns).
    model = _cone_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    num_slices = int(args['num_slices'])
    unbanded = _cone_back_view_batch_triton(sinogram, pixel_indices,
                                            view_params, **args)
    reference = _cone_back_view_batch(sinogram, pixel_indices, view_params,
                                      **args)
    bands = []
    for slice_start in range(0, num_slices, 5):
        band_slices = min(5, num_slices - slice_start)
        bands.append(_cone_back_view_batch_triton(
            sinogram, pixel_indices, view_params, slice_start=slice_start,
            band_slices=band_slices, **args))
        assert bands[-1].shape == (pixel_indices.shape[0], band_slices)
    tiled = torch.cat(bands, dim=1)
    assert _rel_max(tiled, unbanded) <= 1e-6
    rel = _rel_max(tiled, reference)
    print(f"cone back triton banded parity: rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_cone_back_kernel_adjointness():
    # <F x, a> == <x, B a> with F the TORCH forward body and B the kernel: the
    # pairing the whole projector contract rests on, and the check that would
    # catch a weight or index convention that drifted only in the kernel.
    model = _cone_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    num_slices = int(args['num_slices'])
    generator = torch.Generator().manual_seed(3)
    values = torch.rand((pixel_indices.shape[0], num_slices),
                        generator=generator).to(model.torch_device)
    forward = _cone_forward_view_batch(values, pixel_indices, view_params,
                                       **args)
    back = _cone_back_view_batch_triton(sinogram, pixel_indices, view_params,
                                        **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"cone back triton adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, CONE_BACK_BLOCK_P - 1,
                                        CONE_BACK_BLOCK_P + 1,
                                        3 * CONE_BACK_BLOCK_P + 7])
def test_cone_back_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes that must contribute exactly
    # nothing.  Two independent statements of that -- parity against the torch
    # body, and the invariant that a pixel's output does not depend on which
    # lane of which block it landed in (the same pixels inside a LARGER subset
    # must give the same values).
    model = _cone_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    subset = pixel_indices[:num_pixels]
    reference = _cone_back_view_batch(sinogram, subset, view_params, **args)
    kernel_out = _cone_back_view_batch_triton(sinogram, subset, view_params,
                                              **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _cone_back_view_batch_triton(sinogram, pixel_indices, view_params,
                                        **args)
    assert _rel_max(kernel_out, full[:num_pixels]) <= 1e-6


@requires_cuda
def test_cone_back_kernel_selection_and_end_to_end(monkeypatch):
    # The selection contract after the composed gate: the back kernel is ON with no
    # env var at all, wherever the probe and the self-check pass, and the kill
    # switch still forces the torch body.  A model built that way reproduces
    # the torch projector end to end THROUGH the driver (view batching, lazy
    # assembly, and the maybe_compile wrapper the body must survive without
    # being traced).
    from mbirtorch import projectors

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(kernel_availability.ENABLE_FWD_ENV_VAR, raising=False)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(compile_mode='auto')
        usable, reason = kernel_availability.cone_back_kernel_usable(model)
        assert isinstance(reason, str) and reason
        assert usable, reason
        assert model._view_batch_bodies()[1] is _cone_back_view_batch_triton
        # The forward body follows the same default-on protocol, so on a
        # node where its gate passes it is the kernel too.
        assert model._view_batch_bodies()[0] is _cone_forward_view_batch_triton

        model.create_projectors()
        # The driver holds the kernel body ITSELF, uncompiled, even with
        # compile_mode='auto' (the _mbirtorch_no_compile seam).
        assert (model.projector_functions._back_body_per_dev[0]
                is _cone_back_view_batch_triton)
        sinogram, pixel_indices, _, _ = _body_inputs(model)
        kernel_out = model.sparse_back_project(sinogram, pixel_indices)

        # The kill switch is read INSIDE the probe, so it takes effect across
        # a cache reset -- and it must reach the default-on kernel.
        monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        assert model._view_batch_bodies()[1] is _cone_back_view_batch
        model.create_projectors()
        reference = model.sparse_back_project(sinogram, pixel_indices)

        rel = _rel_max(kernel_out, reference)
        print(f"cone back triton end-to-end: rel_max = {rel:.2e}")
        assert rel <= 1e-5
        # ... and it ran eagerly, rather than reaching eager by way of a
        # compile failure that maybe_compile swallowed.
        assert not [k for k in projectors._COMPILE_ERRORS
                    if 'triton_cone' in k]
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@requires_cuda
@pytest.mark.parametrize("geometry", ["flat", "curved", "helical"])
def test_cone_forward_kernel_parity(geometry):
    # As for the back kernel: curved detectors and helical z shifts reach the
    # forward kernel only through the two eager builders it shares with the
    # torch body, so all three variants exercise the same kernel with
    # different contract values.  The forward body takes no coeff_power.
    model = _cone_model(curved=(geometry == "curved"),
                        helical=(geometry == "helical"))
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    reference = _cone_forward_view_batch(values, pixel_indices, view_params,
                                         **args)
    kernel_out = _cone_forward_view_batch_triton(values, pixel_indices,
                                                 view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"cone forward triton parity ({geometry}): rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_cone_forward_kernel_banded_parity():
    # The banded seam, forward form: each band carries its own slice of the
    # VALUES and every band writes the whole sinogram, so a tiling of the
    # slice axis SUMS to the unbanded projection (the back bands concatenate).
    # The z geometry is anchored on the full slice count, which is what makes
    # that true.
    model = _cone_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    num_slices = int(args['num_slices'])
    values = _voxel_values(model, pixel_indices)
    unbanded = _cone_forward_view_batch_triton(values, pixel_indices,
                                               view_params, **args)
    reference = _cone_forward_view_batch(values, pixel_indices, view_params,
                                         **args)
    tiled = None
    for slice_start in range(0, num_slices, 5):
        band = values[:, slice_start:slice_start + 5]
        block = _cone_forward_view_batch_triton(band, pixel_indices,
                                                view_params,
                                                slice_start=slice_start, **args)
        assert block.shape == unbanded.shape
        tiled = block if tiled is None else tiled + block
    assert _rel_max(tiled, unbanded) <= 1e-5
    rel = _rel_max(tiled, reference)
    print(f"cone forward triton banded parity: rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_cone_forward_kernel_adjointness():
    # <F x, a> == <x, B a> with F the kernel forward and B the TORCH back
    # body: the pairing the whole projector contract rests on, and the check
    # that would catch a weight or index convention that drifted only in the
    # kernel.
    model = _cone_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _cone_forward_view_batch_triton(values, pixel_indices,
                                              view_params, **args)
    back = _cone_back_view_batch(sinogram, pixel_indices, view_params, **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"cone forward triton adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, CONE_FWD_BLOCK_P - 1,
                                        CONE_FWD_BLOCK_P + 1,
                                        3 * CONE_FWD_BLOCK_P + 7])
def test_cone_forward_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes whose atomics must be masked
    # off entirely.  Two independent statements of that -- parity against the
    # torch body, and additivity over a pixel SPLIT (the forward sums all
    # pixels into one sinogram, so a subset's output is a partial sum, and the
    # two parts must reassemble the whole however the blocks were padded).
    model = _cone_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    subset, rest = pixel_indices[:num_pixels], pixel_indices[num_pixels:]
    reference = _cone_forward_view_batch(values[:num_pixels], subset,
                                         view_params, **args)
    kernel_out = _cone_forward_view_batch_triton(values[:num_pixels], subset,
                                                 view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _cone_forward_view_batch_triton(values, pixel_indices, view_params,
                                           **args)
    rest_out = _cone_forward_view_batch_triton(values[num_pixels:], rest,
                                               view_params, **args)
    assert _rel_max(kernel_out + rest_out, full) <= 1e-5


@requires_cuda
def test_cone_forward_kernel_repeat_consistency():
    # The forward scatters with tl.atomic_add, so the summation order over
    # pixels and taps is whatever the hardware schedules that launch: identical
    # inputs give results that agree to float rounding, not bit for bit.  This
    # measures that spread instead of assuming it -- if it ever prints above
    # ~1e-6 the parity tolerances above are the thing carrying it, and this is
    # where the evidence lives.
    model = _cone_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    first = _cone_forward_view_batch_triton(values, pixel_indices, view_params,
                                            **args)
    second = _cone_forward_view_batch_triton(values, pixel_indices,
                                             view_params, **args)
    rel = _rel_max(second, first)
    print(f"cone forward triton repeat consistency: rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_cone_kernel_batching_binds_the_cost_model():
    # The driver must batch a SELECTED kernel body by the kernel's own cost
    # model (the _view_batch_cost attribute riding on the wrapper), never by
    # the geometry's max(num_slices, num_rows) charge.  Readings through a
    # real default-selection driver: the bound bodies carry the cost
    # functions; their realized batch at this cell is the kernel chunk while
    # a torch body's is the 64 default; and at a fabricated large-cell charge
    # (pure arithmetic -- nothing is allocated) each body's batch matches its
    # own formula.  The full-cell arithmetic where the torch charge collapses
    # to view batch 1 is pinned in test_view_batching.py.
    from mbirtorch.triton_cone import (CONE_BACK_VIEW_CHUNK,
                                       _cone_back_view_batch_cost,
                                       _cone_forward_view_batch_cost)

    model = _cone_model(compile_mode='auto')
    usable, reason = kernel_availability.cone_back_kernel_usable(model)
    assert usable, reason
    usable, reason = kernel_availability.cone_forward_kernel_usable(model)
    assert usable, reason
    model.create_projectors()
    pf = model.projector_functions
    fwd, back = pf._fwd_body_per_dev[0], pf._back_body_per_dev[0]
    assert fwd._view_batch_cost is _cone_forward_view_batch_cost
    assert back._view_batch_cost is _cone_back_view_batch_cost

    args = model._view_batch_args()
    rows = int(model.get_params('sinogram_shape')[1])
    assert (pf._effective_view_batch(back, 100, rows, args)
            == CONE_BACK_VIEW_CHUNK)
    assert pf._effective_view_batch(_cone_back_view_batch, 100, rows,
                                    args) == 64

    # At this tiny cell the two charges COINCIDE numerically -- transient_cols
    # is 12, so the torch charge is P * 12 * 4 = 48 * P, the kernel's contract
    # coefficient -- so the dispatch proof is the 128-vs-64 pair above; here
    # each body's batch is checked against its own formula alone.
    num_pixels, big_rows = 772_882, 1008
    budget = pf._transient_budget_bytes()
    kernel_vb = pf._effective_view_batch(back, num_pixels, big_rows, args)
    torch_vb = pf._effective_view_batch(_cone_back_view_batch, num_pixels,
                                        big_rows, args)
    bytes_pv, chunk = _cone_back_view_batch_cost(num_pixels, big_rows, args)
    assert kernel_vb == max(1, min(chunk, budget // bytes_pv))
    torch_bytes = num_pixels * model._transient_cols(big_rows) * 4
    assert torch_vb == max(1, min(64, budget // torch_bytes))


@requires_cuda
def test_cone_kernel_view_range_loop_chunked_parity():
    # The view-range loop's chunk seams with the kernel bodies bound: an
    # explicit view_batch_size (which caps kernel batches exactly as it caps
    # torch ones) forces several batches, and the assembled/accumulated
    # results must match a single all-views kernel call.  The back path adds
    # partials across batches and the forward reorders its atomics, so both
    # comparisons read at the float-summation tolerance.
    model = _cone_model()
    usable, reason = kernel_availability.cone_back_kernel_usable(model)
    assert usable, reason
    usable, reason = kernel_availability.cone_forward_kernel_usable(model)
    assert usable, reason
    model.create_projectors()
    pf = model.projector_functions
    assert pf._fwd_body_per_dev[0] is _cone_forward_view_batch_triton
    assert pf._back_body_per_dev[0] is _cone_back_view_batch_triton

    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    num_views = int(view_params.shape[0])
    model.view_batch_size = 2
    assert pf._effective_view_batch(pf._fwd_body_per_dev[0],
                                    int(pixel_indices.shape[0]),
                                    int(values.shape[1]), args) == 2

    chunked_fwd = pf.sparse_forward_project_view_range(values, pixel_indices,
                                                       (0, num_views))
    one_call_fwd = _cone_forward_view_batch_triton(values, pixel_indices,
                                                   view_params, **args)
    assert _rel_max(chunked_fwd, one_call_fwd) <= 1e-5

    chunked_back = pf.sparse_back_project_view_range(sinogram, pixel_indices,
                                                     (0, num_views))
    one_call_back = _cone_back_view_batch_triton(sinogram, pixel_indices,
                                                 view_params, **args)
    assert _rel_max(chunked_back, one_call_back) <= 1e-5


@requires_cuda
def test_cone_forward_kernel_selection_and_end_to_end(monkeypatch):
    # The forward selection contract: OPT-IN, so the torch body stays selected
    # without the env var and the kernel is selected with it when the
    # self-check passes -- and a model built that way reproduces the torch
    # projector end to end THROUGH the driver.
    from mbirtorch import projectors

    # Build the torch-body reference under the kill switch, then lift it:
    # the default-on contract selects the kernel with NO opt-in.
    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(compile_mode='auto')
        assert model._view_batch_bodies()[0] is _cone_forward_view_batch
        model.create_projectors()
        _, pixel_indices, _, _ = _body_inputs(model)
        values = _voxel_values(model, pixel_indices)
        reference = model.sparse_forward_project(values, pixel_indices)

        monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        usable, reason = kernel_availability.cone_forward_kernel_usable(model)
        assert isinstance(reason, str) and reason
        assert usable, reason
        assert model._view_batch_bodies()[0] is _cone_forward_view_batch_triton

        model.create_projectors()
        # The driver holds the kernel body ITSELF, uncompiled, even with
        # compile_mode='auto' (the _mbirtorch_no_compile seam).
        assert (model.projector_functions._fwd_body_per_dev[0]
                is _cone_forward_view_batch_triton)
        kernel_out = model.sparse_forward_project(values, pixel_indices)

        rel = _rel_max(kernel_out, reference)
        print(f"cone forward triton end-to-end: rel_max = {rel:.2e}")
        assert rel <= 1e-5
        # ... and it ran eagerly, rather than reaching eager by way of a
        # compile failure that maybe_compile swallowed.
        assert not [k for k in projectors._COMPILE_ERRORS
                    if 'triton_cone' in k]
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_kernel_body_is_never_torch_compiled():
    # Runs everywhere: the driver compiles every body it is handed, and a
    # hand-written kernel body must be the exception.  torch.compile UNWRAPS
    # torch.compiler.disable, so the opt-out has to be honored here, in
    # maybe_compile, or the launch would be traced.
    from mbirtorch.projectors import maybe_compile

    assert maybe_compile(_cone_back_view_batch_triton, True,
                         instance_key=0) is _cone_back_view_batch_triton
    assert maybe_compile(_cone_forward_view_batch_triton, True,
                         instance_key=0) is _cone_forward_view_batch_triton
    # The torch bodies are still compiled, so the marker is not a blanket
    # opt-out.
    assert maybe_compile(_cone_back_view_batch, True,
                         instance_key=0) is not _cone_back_view_batch
    assert maybe_compile(_cone_forward_view_batch, True,
                         instance_key=0) is not _cone_forward_view_batch


def test_cone_back_self_check_is_a_cached_pair(monkeypatch):
    # Runs everywhere: the gate must answer (bool, str) and cache per device,
    # whatever the machine underneath.
    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(device='cpu')
        first = kernel_availability.cone_back_kernel_usable(model)
        assert isinstance(first, tuple) and len(first) == 2
        usable, reason = first
        assert isinstance(usable, bool)
        assert isinstance(reason, str) and reason
        assert kernel_availability.cone_back_kernel_usable(model) == first
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_cone_back_self_check_catches_a_broken_kernel(monkeypatch):
    # Runs everywhere: the self-check exists to catch a toolchain that
    # compiles the probe and then miscompiles (or fails to compile) the real
    # kernel.  With the probe forced to pass and the kernel body raising, the
    # gate must report a REASON and the model must keep the torch body --
    # never propagate the failure to a caller who only asked what was
    # available.
    from mbirtorch import triton_cone

    def _exploding_body(*args, **kwargs):
        raise RuntimeError('simulated broken kernel')

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(kernel_availability, '_probe_triton',
                        lambda: (True, 'forced-available probe'))
    monkeypatch.setattr(triton_cone, '_cone_back_view_batch_triton',
                        _exploding_body)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(device='cpu')
        usable, reason = kernel_availability.cone_back_kernel_usable(model)
        assert usable is False
        assert 'simulated broken kernel' in reason
        assert model._view_batch_bodies()[1] is _cone_back_view_batch
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_cone_back_gate_is_false_without_a_kernel_path(monkeypatch):
    # Two ways the fast path can be absent, both of which must produce a
    # REASON rather than an exception: the kill switch (any machine) and a
    # host with no CUDA at all (this machine, when it has none).
    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(device='cpu')
        usable, reason = kernel_availability.cone_back_kernel_usable(model)
        assert usable is False
        assert kernel_availability.DISABLE_ENV_VAR in reason
        assert model._view_batch_bodies()[1] is _cone_back_view_batch

        monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR)
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        if not torch.cuda.is_available():
            usable, reason = kernel_availability.cone_back_kernel_usable(model)
            assert usable is False
            assert 'CUDA' in reason or 'cuda' in reason
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_cone_forward_self_check_is_a_cached_pair(monkeypatch):
    # Runs everywhere: the gate must answer (bool, str) and cache per device,
    # whatever the machine underneath.
    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(device='cpu')
        first = kernel_availability.cone_forward_kernel_usable(model)
        assert isinstance(first, tuple) and len(first) == 2
        usable, reason = first
        assert isinstance(usable, bool)
        assert isinstance(reason, str) and reason
        assert kernel_availability.cone_forward_kernel_usable(model) == first
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_cone_forward_self_check_catches_a_broken_kernel(monkeypatch):
    # Runs everywhere: the forward's own copy of the back gate's argument --
    # a toolchain that compiles the probe and then miscompiles (or fails to
    # compile) the real kernel must produce a REASON, and the opted-in model
    # must fall back to the torch forward body rather than raise.
    from mbirtorch import triton_cone

    def _exploding_body(*args, **kwargs):
        raise RuntimeError('simulated broken kernel')

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setenv(kernel_availability.ENABLE_FWD_ENV_VAR, '1')
    monkeypatch.setattr(kernel_availability, '_probe_triton',
                        lambda: (True, 'forced-available probe'))
    monkeypatch.setattr(triton_cone, '_cone_forward_view_batch_triton',
                        _exploding_body)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(device='cpu')
        usable, reason = kernel_availability.cone_forward_kernel_usable(model)
        assert usable is False
        assert 'simulated broken kernel' in reason
        assert model._view_batch_bodies()[0] is _cone_forward_view_batch
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_cone_forward_gate_is_false_without_a_kernel_path(monkeypatch):
    # The forward's copy of the back's gate-reason test: the kill switch (any
    # machine) and a host with no CUDA at all (this machine, when it has none)
    # must each produce a REASON rather than an exception.  The opt-in is set
    # throughout, so what is being read here is the gate, not the policy.
    monkeypatch.setenv(kernel_availability.ENABLE_FWD_ENV_VAR, '1')
    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _cone_model(device='cpu')
        usable, reason = kernel_availability.cone_forward_kernel_usable(model)
        assert usable is False
        assert kernel_availability.DISABLE_ENV_VAR in reason
        assert model._view_batch_bodies()[0] is _cone_forward_view_batch

        monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR)
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        if not torch.cuda.is_available():
            usable, reason = kernel_availability.cone_forward_kernel_usable(model)
            assert usable is False
            assert 'CUDA' in reason or 'cuda' in reason
            assert model._view_batch_bodies()[0] is _cone_forward_view_batch
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_cone_forward_kernel_selects_by_default(monkeypatch):
    # The policy, stated separately from the gate: WITHOUT the opt-in the
    # torch forward body is selected whatever the gate would say, and the gate
    # is not even consulted (an un-opted-in process must not pay for the
    # self-check).  Runs everywhere -- the forced-available probe and the
    # sentinel gate make it machine-independent.
    calls = []

    def _spy(model):
        calls.append(model)
        return (True, 'forced-usable gate')

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(kernel_availability, 'cone_forward_kernel_usable', _spy)
    try:
        model = _cone_model(device='cpu')
        # Construction consults NOTHING.  The projectors, and with them the
        # availability gate, are built on first use, so a caller who only
        # inspects a model pays for no probe.
        assert not calls
        # Default-on contract (post composed-gate): the gate is consulted
        # with no opt-in, and its verdict alone selects the body.
        assert (model._view_batch_bodies()[0]
                is _cone_forward_view_batch_triton)
        assert len(calls) == 1
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_cone_kernel_selection_is_layout_independent():
    """The restored selection contract: the layout plays no part.

    An interim rule once withheld the forward kernel from sharded layouts.
    The defect behind it was the launch context, not the kernel, and the
    wrappers now bracket their launches on the tensors' device (the
    kernel-sharding findings in the plans repo).  Selection therefore
    consults the availability gates alone, and a layout change must neither
    drop a kernel nor latch a stale choice.

    This test runs on CPU by forcing the gates, so the RULE is pinned on any
    machine.  Whether a real kernel is usable is a separate question that
    the availability gates own.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(kernel_availability, 'cone_forward_kernel_usable',
                            lambda model: (True, 'forced'))
        monkeypatch.setattr(kernel_availability, 'cone_back_kernel_usable',
                            lambda model: (True, 'forced'))
        model = _cone_model(device='cpu')
        # Trivial placement: both kernels bind.
        fwd, back = model._view_batch_bodies()
        assert fwd is _cone_forward_view_batch_triton
        assert back is _cone_back_view_batch_triton
        # Non-trivial placement: the same selection.
        model.configure_devices(devices=['cpu', 'cpu'])
        fwd, back = model._view_batch_bodies()
        assert fwd is _cone_forward_view_batch_triton
        assert back is _cone_back_view_batch_triton
        # And back again, so a rebuilt layout re-selects rather than latching.
        model.configure_devices(devices=['cpu'])
        fwd, back = model._view_batch_bodies()
        assert fwd is _cone_forward_view_batch_triton
        assert back is _cone_back_view_batch_triton
    finally:
        monkeypatch.undo()
