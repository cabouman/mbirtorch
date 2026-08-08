"""Value gates for the Triton parallel-beam kernels (the parallel back and
forward bodies).

The cone battery's shape, applied to the degenerate geometry.  Each kernel is
an alternative view-batch BODY, so every gate here compares it against the
torch body it replaces at the same inputs: parity across geometry variants
that move the hfan contract (the projected footprint, the tap radius, the
detector offset) at every coefficient power the body takes, parity across the
banded seam, the explicit adjointness pairings -- kernel against the OTHER
direction's torch body, and the two kernels against each other -- and the
poison-the-padding class (a pixel count that is not a multiple of the kernel's
pixel tile, where the padded lanes must contribute exactly nothing).

Two things differ from the cone battery, both because the vertical fan is
gone.  The banded seam is a ROW band rather than a slice band with a z anchor:
rows track slices, so the back body bands by being handed fewer sinogram rows
and the forward body bands by being handed fewer value columns, and both
tilings CONCATENATE (the cone forward's bands sum instead).  And there is no
rounding carve-out to absorb -- no atan2-vs-sqrt divisor, no round-vs-floor
tie -- so these kernels differ from their bodies by float summation order
alone.  The tolerances stay at the design's figures (rel 1e-5 on the gradient
path, 1e-4 at coeff_power 2) because what a value gate must catch is a
miscompile, not a ULP.

The forward kernel scatters with float atomics, so its sums are reordered from
launch to launch and it is not bit-reproducible; 1e-5 covers that too, and
test_parallel_forward_kernel_repeat_consistency measures the run-to-run spread
rather than assuming it.

Everything that launches a kernel needs CUDA and skips without it; the
availability gates and the selection policy are exercised on any machine.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import kernel_availability
from mbirtorch.parallel_beam import (_parallel_back_view_batch,
                                     _parallel_forward_view_batch)
from mbirtorch.triton_parallel import (PARALLEL_BACK_BLOCK_P,
                                       PARALLEL_BACK_BLOCK_R,
                                       PARALLEL_FWD_BLOCK_P,
                                       PARALLEL_FWD_BLOCK_R,
                                       _parallel_back_view_batch_triton,
                                       _parallel_forward_view_batch_triton)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the hand-written Triton kernels need a CUDA device")

# The three contract variants the parity tests sweep.  They are not cosmetic:
# voxel_row_aspect widens the projected footprint enough to raise psf_radius
# from 1 to 2 (a longer tap loop, and W_p_c > 1 so the min(1, W_p_c) clip
# binds), and det_channel_offset shifts n_p off the channel grid so the tap
# centers and their trapezoid weights all move.
VARIANTS = {"square": {},
            "wide_voxel": {"row_aspect": 3.0},
            "det_offset": {"det_offset": 0.7}}


def _parallel_model(cell=(6, 12, 12), row_aspect=1.0, det_offset=0.0,
                    device="cuda", compile_mode="off"):
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(cell, angles, 
                                        compile_mode=compile_mode)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    if row_aspect != 1.0 or det_offset != 0.0:
        model.set_params(no_warning=True, voxel_row_aspect=row_aspect,
                         det_channel_offset=det_offset)
        model.auto_set_recon_geometry(no_warning=True)
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
    view_params = torch.as_tensor(model.get_params('angles'),
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
    # An identically zero reference would make every relative reading a free
    # pass, so the ruler is checked before it is used.
    scale = float(ref.abs().max())
    assert scale > 0.0, "the reference output is identically zero"
    return float((out - ref).abs().max()) / scale


@requires_cuda
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("coeff_power,tol", [(1, 1e-5), (2, 1e-4)])
def test_parallel_back_kernel_parity(variant, coeff_power, tol):
    model = _parallel_model(**VARIANTS[variant])
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    reference = _parallel_back_view_batch(sinogram, pixel_indices, view_params,
                                          coeff_power=coeff_power, **args)
    kernel_out = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                  view_params,
                                                  coeff_power=coeff_power,
                                                  **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"parallel back triton parity ({variant}, "
          f"coeff_power={coeff_power}, psf_radius={args['psf_radius']}): "
          f"rel_max = {rel:.2e}")
    assert rel <= tol


@requires_cuda
def test_parallel_back_kernel_row_band_parity():
    # The banded seam, row-aligned form: the driver hands a row-aligned
    # geometry a ROW BAND of the sinogram (no slice_start, no band_slices --
    # the body asserts both), and each band owns the matching output columns,
    # so a tiling of the row axis CONCATENATES into the unbanded partial.
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    num_rows = int(sinogram.shape[1])
    unbanded = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                view_params, **args)
    reference = _parallel_back_view_batch(sinogram, pixel_indices, view_params,
                                          **args)
    bands = []
    for row_start in range(0, num_rows, 5):
        band = sinogram[:, row_start:row_start + 5]
        bands.append(_parallel_back_view_batch_triton(band, pixel_indices,
                                                      view_params, **args))
        assert bands[-1].shape == (pixel_indices.shape[0], band.shape[1])
    tiled = torch.cat(bands, dim=1)
    assert _rel_max(tiled, unbanded) <= 1e-6
    rel = _rel_max(tiled, reference)
    print(f"parallel back triton row-band parity: rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_parallel_back_kernel_adjointness():
    # <F x, a> == <x, B a> with F the TORCH forward body and B the kernel: the
    # pairing the whole projector contract rests on, and the check that would
    # catch a weight or index convention that drifted only in the kernel.
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _parallel_forward_view_batch(values, pixel_indices, view_params,
                                           **args)
    back = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                            view_params, **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"parallel back triton adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, PARALLEL_BACK_BLOCK_P - 1,
                                        PARALLEL_BACK_BLOCK_P + 1,
                                        3 * PARALLEL_BACK_BLOCK_P + 7])
def test_parallel_back_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes that must contribute exactly
    # nothing.  Two independent statements of that -- parity against the torch
    # body, and the invariant that a pixel's output does not depend on which
    # lane of which block it landed in (the same pixels inside a LARGER subset
    # must give the same values).
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    subset = pixel_indices[:num_pixels]
    reference = _parallel_back_view_batch(sinogram, subset, view_params, **args)
    kernel_out = _parallel_back_view_batch_triton(sinogram, subset,
                                                  view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                            view_params, **args)
    assert _rel_max(kernel_out, full[:num_pixels]) <= 1e-6


@requires_cuda
def test_parallel_back_kernel_delegates_exotic_coeff_power():
    # The kernel's coefficient power is a constexpr branch over 1 and 2 -- the
    # only powers any caller in the package uses.  Anything else DELEGATES to
    # the torch body rather than diverging from it, so the wrapper stays a
    # total drop-in replacement.
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    reference = _parallel_back_view_batch(sinogram, pixel_indices, view_params,
                                          coeff_power=3, **args)
    delegated = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                 view_params, coeff_power=3,
                                                 **args)
    assert bool(torch.equal(delegated, reference))


@requires_cuda
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parallel_forward_kernel_parity(variant):
    # As for the back kernel: the geometry reaches the forward kernel only
    # through the one eager builder it shares with the torch body, so all
    # three variants exercise the same kernel with different contract values.
    # The forward body takes no coeff_power.
    model = _parallel_model(**VARIANTS[variant])
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    reference = _parallel_forward_view_batch(values, pixel_indices,
                                             view_params, **args)
    kernel_out = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                     view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"parallel forward triton parity ({variant}, "
          f"psf_radius={args['psf_radius']}): rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_parallel_forward_kernel_row_band_parity():
    # The banded seam, forward form: the forward carries its band in the
    # COLUMN count of the values and each band produces the matching detector
    # ROWS, so a tiling of the column axis CONCATENATES on the row axis --
    # where the cone forward's bands each write the whole sinogram and sum.
    model = _parallel_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    num_cols = int(values.shape[1])
    unbanded = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                   view_params, **args)
    reference = _parallel_forward_view_batch(values, pixel_indices,
                                             view_params, **args)
    blocks = []
    for col_start in range(0, num_cols, 5):
        band = values[:, col_start:col_start + 5]
        blocks.append(_parallel_forward_view_batch_triton(band, pixel_indices,
                                                          view_params, **args))
        assert blocks[-1].shape[1] == band.shape[1]
    tiled = torch.cat(blocks, dim=1)
    assert _rel_max(tiled, unbanded) <= 1e-5
    rel = _rel_max(tiled, reference)
    print(f"parallel forward triton row-band parity: rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_parallel_forward_kernel_adjointness():
    # <F x, a> == <x, B a> with F the kernel forward and B the TORCH back
    # body -- the mirror of the back kernel's pairing.
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                  view_params, **args)
    back = _parallel_back_view_batch(sinogram, pixel_indices, view_params,
                                     **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"parallel forward triton adjointness: lhs {lhs:.6f}, "
          f"rhs {rhs:.6f}, rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
def test_parallel_kernel_pair_adjointness():
    # The pairing that actually ships once both kernels are on: KERNEL forward
    # against KERNEL back.  The two tests above each hold one side fixed to
    # the torch body, so a convention that drifted in BOTH kernels together
    # would pass them and fail here.
    model = _parallel_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                  view_params, **args)
    back = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                            view_params, **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"parallel kernel-pair adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, PARALLEL_FWD_BLOCK_P - 1,
                                        PARALLEL_FWD_BLOCK_P + 1,
                                        3 * PARALLEL_FWD_BLOCK_P + 7])
def test_parallel_forward_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes whose atomics must be masked
    # off entirely.  Two independent statements of that -- parity against the
    # torch body, and additivity over a pixel SPLIT (the forward sums all
    # pixels into one sinogram, so a subset's output is a partial sum, and the
    # two parts must reassemble the whole however the blocks were padded).
    model = _parallel_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    subset, rest = pixel_indices[:num_pixels], pixel_indices[num_pixels:]
    reference = _parallel_forward_view_batch(values[:num_pixels], subset,
                                             view_params, **args)
    kernel_out = _parallel_forward_view_batch_triton(values[:num_pixels],
                                                     subset, view_params,
                                                     **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _parallel_forward_view_batch_triton(values, pixel_indices,
                                               view_params, **args)
    rest_out = _parallel_forward_view_batch_triton(values[num_pixels:], rest,
                                                   view_params, **args)
    assert _rel_max(kernel_out + rest_out, full) <= 1e-5


@requires_cuda
def test_parallel_kernels_span_several_row_chunks():
    # Every cell above is narrower than the kernels' row tile, so the row grid
    # axis holds exactly one program and a bug in its offset would not show.
    # _tile_size caps the tile at next_pow2(extent), so forcing a second chunk
    # takes more rows than the pinned tile: this cell gives the back kernel
    # several row chunks and the forward kernel at least two.
    cell = (4, 4 * PARALLEL_BACK_BLOCK_R, 16)
    model = _parallel_model(cell=cell)
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    assert int(sinogram.shape[1]) > PARALLEL_BACK_BLOCK_R
    assert int(values.shape[1]) > PARALLEL_FWD_BLOCK_R

    back_ref = _parallel_back_view_batch(sinogram, pixel_indices, view_params,
                                         **args)
    back_out = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                view_params, **args)
    back_rel = _rel_max(back_out, back_ref)
    fwd_ref = _parallel_forward_view_batch(values, pixel_indices, view_params,
                                           **args)
    fwd_out = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                  view_params, **args)
    fwd_rel = _rel_max(fwd_out, fwd_ref)
    print(f"parallel triton multi-row-chunk ({sinogram.shape[1]} rows): "
          f"back rel_max = {back_rel:.2e}, forward rel_max = {fwd_rel:.2e}")
    assert back_rel <= 1e-5
    assert fwd_rel <= 1e-5


@requires_cuda
def test_parallel_forward_kernel_repeat_consistency():
    # The forward scatters with tl.atomic_add, so the summation order over
    # pixels and taps is whatever the hardware schedules that launch: identical
    # inputs give results that agree to float rounding, not bit for bit.  This
    # measures that spread instead of assuming it -- if it ever prints above
    # ~1e-6 the parity tolerances above are the thing carrying it, and this is
    # where the evidence lives.
    model = _parallel_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    first = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                view_params, **args)
    second = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                 view_params, **args)
    rel = _rel_max(second, first)
    print(f"parallel forward triton repeat consistency: rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_parallel_kernel_batching_binds_the_cost_model():
    # The driver must batch a SELECTED kernel body by the kernel's own cost
    # model (the _view_batch_cost attribute riding on the wrapper), never by
    # the torch bodies' gather charge.  Three readings through a real
    # default-selection driver: the bound bodies carry the cost functions;
    # their realized batch at this cell is the kernel chunk while a torch
    # body's is the 64 default; and at a large-cell charge (fabricated
    # arithmetic -- nothing is allocated) the torch charge collapses to view
    # batch 1 where the kernel charge does not, which is the defect this
    # mechanism exists to fix.
    from mbirtorch.triton_parallel import (PARALLEL_BACK_VIEW_CHUNK,
                                           _parallel_back_view_batch_cost,
                                           _parallel_forward_view_batch_cost)

    model = _parallel_model(compile_mode='auto')
    usable, reason = kernel_availability.parallel_back_kernel_usable(model)
    assert usable, reason
    usable, reason = kernel_availability.parallel_forward_kernel_usable(model)
    assert usable, reason
    model.create_projectors()
    pf = model.projector_functions
    fwd, back = pf._fwd_body_per_dev[0], pf._back_body_per_dev[0]
    assert fwd._view_batch_cost is _parallel_forward_view_batch_cost
    assert back._view_batch_cost is _parallel_back_view_batch_cost

    args = model._view_batch_args()
    rows = int(model.get_params('sinogram_shape')[1])
    assert (pf._effective_view_batch(back, 100, rows, args)
            == PARALLEL_BACK_VIEW_CHUNK)
    assert pf._effective_view_batch(_parallel_back_view_batch, 100, rows,
                                    args) == 64

    big_args = dict(args, num_channels=992)
    num_pixels, big_rows = 772_882, 1008
    budget = pf._transient_budget_bytes()
    assert num_pixels * big_rows * 4 > budget
    kernel_vb = pf._effective_view_batch(back, num_pixels, big_rows, big_args)
    torch_vb = pf._effective_view_batch(_parallel_back_view_batch, num_pixels,
                                        big_rows, big_args)
    bytes_pv, chunk = _parallel_back_view_batch_cost(num_pixels, big_rows,
                                                     big_args)
    assert torch_vb == 1
    assert kernel_vb == max(1, min(chunk, budget // bytes_pv)) > 1


@requires_cuda
def test_parallel_kernel_view_range_loop_chunked_parity():
    # The view-range loop's chunk seams with the kernel bodies bound: an
    # explicit view_batch_size (which caps kernel batches exactly as it caps
    # torch ones) forces several batches, and the assembled/accumulated
    # results must match a single all-views kernel call.  The back path adds
    # partials across batches and the forward reorders its atomics, so both
    # comparisons read at the float-summation tolerance.
    model = _parallel_model()
    usable, reason = kernel_availability.parallel_back_kernel_usable(model)
    assert usable, reason
    usable, reason = kernel_availability.parallel_forward_kernel_usable(model)
    assert usable, reason
    model.create_projectors()
    pf = model.projector_functions
    assert pf._fwd_body_per_dev[0] is _parallel_forward_view_batch_triton
    assert pf._back_body_per_dev[0] is _parallel_back_view_batch_triton

    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    num_views = int(view_params.shape[0])
    model.view_batch_size = 2
    assert pf._effective_view_batch(pf._fwd_body_per_dev[0],
                                    int(pixel_indices.shape[0]),
                                    int(values.shape[1]), args) == 2

    chunked_fwd = pf.sparse_forward_project_view_range(values, pixel_indices,
                                                       (0, num_views))
    one_call_fwd = _parallel_forward_view_batch_triton(values, pixel_indices,
                                                       view_params, **args)
    assert _rel_max(chunked_fwd, one_call_fwd) <= 1e-5

    chunked_back = pf.sparse_back_project_view_range(sinogram, pixel_indices,
                                                     (0, num_views))
    one_call_back = _parallel_back_view_batch_triton(sinogram, pixel_indices,
                                                     view_params, **args)
    assert _rel_max(chunked_back, one_call_back) <= 1e-5


@requires_cuda
def test_parallel_back_kernel_selection_and_end_to_end(monkeypatch):
    # The selection contract after the composed gate: the back kernel is ON
    # with no environment variable at all, wherever the probe and the
    # self-check pass, and the kill switch still forces the torch body.  A
    # model built that way reproduces the torch projector end to end THROUGH
    # the driver (view batching, lazy accumulation, and the maybe_compile
    # wrapper the body must survive without being traced).  The torch
    # reference is built under the kill switch, because the default now
    # selects the kernel.
    from mbirtorch import projectors

    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _parallel_model(compile_mode='auto')
        assert model._view_batch_bodies()[1] is _parallel_back_view_batch
        model.create_projectors()
        sinogram, pixel_indices, _, _ = _body_inputs(model)
        reference = model.sparse_back_project(sinogram, pixel_indices)

        monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR)
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        usable, reason = kernel_availability.parallel_back_kernel_usable(model)
        assert isinstance(reason, str) and reason
        assert usable, reason
        assert (model._view_batch_bodies()[1]
                is _parallel_back_view_batch_triton)

        model.create_projectors()
        # The driver holds the kernel body ITSELF, uncompiled, even with
        # compile_mode='auto' (the _mbirtorch_no_compile seam).
        assert (model.projector_functions._back_body_per_dev[0]
                is _parallel_back_view_batch_triton)
        kernel_out = model.sparse_back_project(sinogram, pixel_indices)

        rel = _rel_max(kernel_out, reference)
        print(f"parallel back triton end-to-end: rel_max = {rel:.2e}")
        assert rel <= 1e-5

        # The kill switch reaches the opted-in kernel too; it is read INSIDE
        # the probe, so it takes effect across a cache reset.
        monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        assert model._view_batch_bodies()[1] is _parallel_back_view_batch
        # ... and the kernel ran eagerly, rather than reaching eager by way of
        # a compile failure that maybe_compile swallowed.
        assert not [k for k in projectors._COMPILE_ERRORS
                    if 'triton_parallel' in k]
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@requires_cuda
def test_parallel_forward_kernel_selection_and_end_to_end(monkeypatch):
    # The forward's copy of the back's default-on selection contract, with the
    # same kill-switch-built reference.
    from mbirtorch import projectors

    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _parallel_model(compile_mode='auto')
        assert model._view_batch_bodies()[0] is _parallel_forward_view_batch
        model.create_projectors()
        _, pixel_indices, _, _ = _body_inputs(model)
        values = _voxel_values(model, pixel_indices)
        reference = model.sparse_forward_project(values, pixel_indices)

        monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR)
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        usable, reason = kernel_availability.parallel_forward_kernel_usable(
            model)
        assert isinstance(reason, str) and reason
        assert usable, reason
        assert (model._view_batch_bodies()[0]
                is _parallel_forward_view_batch_triton)

        model.create_projectors()
        assert (model.projector_functions._fwd_body_per_dev[0]
                is _parallel_forward_view_batch_triton)
        kernel_out = model.sparse_forward_project(values, pixel_indices)

        rel = _rel_max(kernel_out, reference)
        print(f"parallel forward triton end-to-end: rel_max = {rel:.2e}")
        assert rel <= 1e-5
        assert not [k for k in projectors._COMPILE_ERRORS
                    if 'triton_parallel' in k]
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_parallel_kernel_bodies_are_never_torch_compiled():
    # Runs everywhere: the driver compiles every body it is handed, and a
    # hand-written kernel body must be the exception.  torch.compile UNWRAPS
    # torch.compiler.disable, so the opt-out has to be honored here, in
    # maybe_compile, or the launch would be traced.
    from mbirtorch.projectors import maybe_compile

    assert maybe_compile(_parallel_back_view_batch_triton, True,
                         instance_key=0) is _parallel_back_view_batch_triton
    assert maybe_compile(_parallel_forward_view_batch_triton, True,
                         instance_key=0) is _parallel_forward_view_batch_triton
    # The torch bodies are still compiled, so the marker is not a blanket
    # opt-out.
    assert maybe_compile(_parallel_back_view_batch, True,
                         instance_key=0) is not _parallel_back_view_batch
    assert maybe_compile(_parallel_forward_view_batch, True,
                         instance_key=0) is not _parallel_forward_view_batch


@pytest.mark.parametrize("gate_name", ["parallel_back_kernel_usable",
                                       "parallel_forward_kernel_usable"])
def test_parallel_self_check_is_a_cached_pair(gate_name, monkeypatch):
    # Runs everywhere: each gate must answer (bool, str) and cache per device,
    # whatever the machine underneath.
    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    gate = getattr(kernel_availability, gate_name)
    try:
        model = _parallel_model(device='cpu')
        first = gate(model)
        assert isinstance(first, tuple) and len(first) == 2
        usable, reason = first
        assert isinstance(usable, bool)
        assert isinstance(reason, str) and reason
        assert gate(model) == first
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@pytest.mark.parametrize(
    "gate_name,body_name,index,torch_body",
    [("parallel_back_kernel_usable", "_parallel_back_view_batch_triton",
      1, _parallel_back_view_batch),
     ("parallel_forward_kernel_usable", "_parallel_forward_view_batch_triton",
      0, _parallel_forward_view_batch)])
def test_parallel_self_check_catches_a_broken_kernel(
        gate_name, body_name, index, torch_body, monkeypatch):
    # Runs everywhere: the self-check exists to catch a toolchain that
    # compiles the probe and then miscompiles (or fails to compile) the real
    # kernel.  With the probe forced to pass and the kernel body raising, the
    # gate must report a REASON and the model must keep the
    # torch body -- never propagate the failure to a caller who only asked
    # what was available.
    from mbirtorch import triton_parallel

    def _exploding_body(*args, **kwargs):
        raise RuntimeError('simulated broken kernel')

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(kernel_availability, '_probe_triton',
                        lambda: (True, 'forced-available probe'))
    monkeypatch.setattr(triton_parallel, body_name, _exploding_body)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _parallel_model(device='cpu')
        usable, reason = getattr(kernel_availability, gate_name)(model)
        assert usable is False
        assert 'simulated broken kernel' in reason
        assert model._view_batch_bodies()[index] is torch_body
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@pytest.mark.parametrize(
    "gate_name,index,torch_body",
    [("parallel_back_kernel_usable", 1, _parallel_back_view_batch),
     ("parallel_forward_kernel_usable", 0, _parallel_forward_view_batch)])
def test_parallel_gate_is_false_without_a_kernel_path(
        gate_name, index, torch_body, monkeypatch):
    # Two ways the fast path can be absent, both of which must produce a
    # REASON rather than an exception: the kill switch (any machine) and a host
    # with no CUDA at all (this machine, when it has none).
    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    gate = getattr(kernel_availability, gate_name)
    try:
        model = _parallel_model(device='cpu')
        usable, reason = gate(model)
        assert usable is False
        assert kernel_availability.DISABLE_ENV_VAR in reason
        assert model._view_batch_bodies()[index] is torch_body

        monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR)
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        if not torch.cuda.is_available():
            usable, reason = gate(model)
            assert usable is False
            assert 'CUDA' in reason or 'cuda' in reason
            assert model._view_batch_bodies()[index] is torch_body
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@pytest.mark.parametrize(
    "gate_name,index,torch_body,kernel_body",
    [("parallel_back_kernel_usable", 1,
      _parallel_back_view_batch, _parallel_back_view_batch_triton),
     ("parallel_forward_kernel_usable", 0,
      _parallel_forward_view_batch, _parallel_forward_view_batch_triton)])
def test_parallel_kernels_select_by_default(gate_name, index, torch_body,
                                            kernel_body, monkeypatch):
    # The policy, stated separately from the gate: no environment variable is
    # consulted -- the gate's verdict alone decides, and a passing gate
    # selects the kernel by default.  Runs everywhere -- the sentinel gate
    # makes it machine-independent.
    calls = []
    verdict = {'usable': True}

    def _spy(model):
        calls.append(model)
        return (verdict['usable'], 'sentinel gate')

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(kernel_availability, gate_name, _spy)
    try:
        model = _parallel_model(device='cpu')
        calls.clear()
        assert model._view_batch_bodies()[index] is kernel_body
        assert len(calls) == 1

        verdict['usable'] = False
        assert model._view_batch_bodies()[index] is torch_body
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_parallel_kernel_selection_is_layout_independent():
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
        monkeypatch.setattr(kernel_availability, 'parallel_forward_kernel_usable',
                            lambda model: (True, 'forced'))
        monkeypatch.setattr(kernel_availability, 'parallel_back_kernel_usable',
                            lambda model: (True, 'forced'))
        model = _parallel_model(device='cpu')
        # Trivial placement: both kernels bind.
        fwd, back = model._view_batch_bodies()
        assert fwd is _parallel_forward_view_batch_triton
        assert back is _parallel_back_view_batch_triton
        # Non-trivial placement: the same selection.
        model.configure_devices(devices=['cpu', 'cpu'])
        fwd, back = model._view_batch_bodies()
        assert fwd is _parallel_forward_view_batch_triton
        assert back is _parallel_back_view_batch_triton
        # And back again, so a rebuilt layout re-selects rather than latching.
        model.configure_devices(devices=['cpu'])
        fwd, back = model._view_batch_bodies()
        assert fwd is _parallel_forward_view_batch_triton
        assert back is _parallel_back_view_batch_triton
    finally:
        monkeypatch.undo()
