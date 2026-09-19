"""Value gates for the Triton multiaxis-parallel back and forward kernels.

The cone and parallel batteries' shape, applied to the multiaxis geometry.  Each
kernel is an alternative view-batch BODY, so every gate here compares it against
the torch body it replaces
(:func:`mbirtorch.multiaxis_parallel._multiaxis_back_view_batch`,
:func:`mbirtorch.multiaxis_parallel._multiaxis_forward_view_batch`) at the same
inputs: parity across variants that move the geometry, parity across the banded
seams, the explicit adjointness pairings (each kernel against the opposite torch
body, and the kernel PAIR against itself), and the poison-the-padding class (a
pixel count that is not a multiple of the kernel's pixel tile, where the padded
lanes must contribute exactly nothing).

The variant axis here is ELEVATION, because that is the axis this geometry adds
to parallel beam and the axis that separates the slope of the slice-to-row map
from the vertical footprint that sets the trapezoid weight.  Zero elevation is
the degenerate parallel-beam case; the sweep and the strong constant tilt each
move the footprint, the clip bound and the mass-conserving amplitude.  Two more
variants move the contract without touching elevation: a nonzero det_row_offset
shifts the row anchor off the grid, and a wide voxel_row_aspect raises
psf_radius from 1 to 2, lengthening both tap loops.

The FORWARD kernel adds a class of its own: COVERAGE.  It inverts the
slice-to-row map instead of mirroring the torch body's scatter, so it has to
enumerate the recon slices that reach each detector row, and at a tilted view
more of them reach one row than the 2 * psf_radius + 1 taps the back kernel
uses.  The bound that sets the enumeration is checked directly against a brute
force count of the contributing slices (a CPU test, so it runs everywhere), and
a thin-slice cell where that bound provably exceeds psf_radius carries the
parity statement on a GPU.  The five variants below do NOT reach that case --
each of them needs one slice tap on each side, which psf_radius already covers
-- so the coverage tests use their own cells and say so.

One more class sits beside those.  Each wrapper rounds a width up to a multiple
of 16 before the launch -- the back kernel its slice band, the forward kernel
its detector row count -- so a width that is not one is computed with extra
columns or rows the wrapper then slices off.  Those tests read the values and
the returned view's stride, which is the width the wrapper really allocated.

Tolerances follow the design's value gate -- rel 1e-5 on the gradient path,
1e-4 at coeff_power 2.  The back kernel has one rounding carve-out to absorb,
the floor(m + 0.5) row center against the torch body's round-half-to-even (see
the module docstring of mbirtorch/triton_multiaxis.py for why the tie is
inert); otherwise both kernels differ from their bodies by float summation
order alone.

Where the BACK kernel is compared against ITSELF -- banded concatenation, a
pixel subset inside a larger one, a repeated launch -- the comparison is
torch.equal rather than a tolerance.  That kernel gathers into a register
accumulator and stores each output element once, with no atomic adds, and each
element's sum order is fixed by the loop nesting alone, independent of the tile
shape and of which band or pixel block the element landed in.  Bit equality is
therefore the statement to make there, and a tolerance would hide a real
change.  The FORWARD kernel scatters with atomic adds, whose order varies from
launch to launch, so its self-comparisons are tolerances and bit equality would
be the wrong statement; each such test says which one it is making and why.

The last group is SELECTION rather than value: the model binds these kernels
wherever their availability gates pass, so the tests at the end of the file
state what the model actually returns from ``_view_batch_bodies``, that a
reconstruction taking that route reproduces one on the torch bodies, and that
every way the fast path can be absent -- the kill switch, a host with no CUDA,
a kernel that raises -- ends in the torch bodies with a recorded reason.

Everything that launches a kernel needs CUDA and skips without it; the import
safety, the cost declarations, the coverage bound and the selection policy are
exercised on any machine.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import kernel_availability
from mbirtorch._utils import padded_kernel_width
from mbirtorch.multiaxis_parallel import (_multiaxis_back_view_batch,
                                          _multiaxis_forward_view_batch,
                                          _multiaxis_horizontal_data,
                                          _multiaxis_vertical_terms)
from mbirtorch.triton_multiaxis import (MULTIAXIS_BACK_BLOCK_P,
                                        MULTIAXIS_BACK_VIEW_CHUNK,
                                        MULTIAXIS_FWD_BLOCK_P,
                                        MULTIAXIS_FWD_BLOCK_R,
                                        MULTIAXIS_FWD_MAX_SLICE_RADIUS,
                                        MULTIAXIS_FWD_VIEW_CHUNK,
                                        _multiaxis_back_view_batch_cost,
                                        _multiaxis_back_view_batch_triton,
                                        _multiaxis_forward_view_batch_cost,
                                        _multiaxis_forward_view_batch_triton,
                                        _multiaxis_slice_tap_radius)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the hand-written Triton kernels need a CUDA device")

# The five contract variants the parity tests sweep.  The first three move the
# ELEVATION alone: zero (the parallel-beam case, where the vertical footprint
# is the voxel slice pitch and the amplitude is 1), a moderate sweep, and a
# strong constant tilt that stays under the model's 45-degree warning.  The
# last two move the contract without touching elevation.
VARIANTS = {"zero_elevation": {"elevation": "zero"},
            "elevation_sweep": {},
            "strong_tilt": {"elevation": "tilt"},
            "det_row_offset": {"det_row_offset": 0.7},
            "wide_voxel": {"row_aspect": 3.0}}


def _ma_model(cell=(6, 12, 12), elevation="sweep", row_aspect=1.0,
              det_row_offset=0.0, slice_aspect=1.0, device="cuda",
              compile_mode="off"):
    """A small multiaxis model, built as tests/test_multiaxis.py's _small_ma
    builds one: azimuths spanning pi and an elevation set chosen per variant.

    ``slice_aspect`` thins the voxel slice pitch against the detector row
    pitch, which is what lowers the slice-to-row slope below the point where
    psf_radius still covers the forward kernel's slice enumeration.  It is not
    part of the variant sweep; only the forward coverage tests set it."""
    num_views = cell[0]
    azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
    if elevation == "zero":
        tilt = np.zeros(num_views)
    elif elevation == "tilt":
        tilt = np.full(num_views, 0.7)
    else:
        tilt = np.linspace(-0.5, 0.5, num_views)
    model = mbirtorch.MultiAxisParallelModel(
        cell, np.stack([azimuth, tilt], axis=1), compile_mode=compile_mode)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    if row_aspect != 1.0 or det_row_offset != 0.0 or slice_aspect != 1.0:
        model.set_params(no_warning=True, voxel_row_aspect=row_aspect,
                         voxel_slice_aspect=slice_aspect,
                         det_row_offset=det_row_offset)
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
    # Multiaxis carries its per-view parameters as the (num_views, 2) angles
    # array: column 0 the azimuth, column 1 the elevation.
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
def test_multiaxis_back_kernel_parity(variant, coeff_power, tol):
    # The geometry reaches the kernel only through the two eager builders it
    # shares with the torch body, so all five variants exercise the same kernel
    # with different contract values.
    model = _ma_model(**VARIANTS[variant])
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    reference = _multiaxis_back_view_batch(sinogram, pixel_indices, view_params,
                                           coeff_power=coeff_power, **args)
    kernel_out = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                   view_params,
                                                   coeff_power=coeff_power,
                                                   **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"multiaxis back triton parity ({variant}, "
          f"coeff_power={coeff_power}, psf_radius={args['psf_radius']}): "
          f"rel_max = {rel:.2e}")
    assert rel <= tol


@requires_cuda
@pytest.mark.parametrize("band_slices", [5, 16])
def test_multiaxis_back_kernel_banded_parity(band_slices):
    # The banded seam: the slice-to-row map is anchored on the FULL slice
    # count, so a tiling of the slice axis must reassemble the unbanded partial
    # (each band owns its own output columns).  Band 5 is not a multiple of 16
    # and band 16 is, so both padding paths are tiled here.
    model = _ma_model(cell=(6, 32, 20))
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    num_slices = int(args['num_slices'])
    assert num_slices >= 32
    unbanded = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                 view_params, **args)
    reference = _multiaxis_back_view_batch(sinogram, pixel_indices, view_params,
                                           **args)
    bands = []
    for slice_start in range(0, num_slices, band_slices):
        length = min(band_slices, num_slices - slice_start)
        bands.append(_multiaxis_back_view_batch_triton(
            sinogram, pixel_indices, view_params, slice_start=slice_start,
            band_slices=length, **args))
        assert bands[-1].shape == (pixel_indices.shape[0], length)
    tiled = torch.cat(bands, dim=1)
    # Kernel against itself: bit equality, for the reason the module docstring
    # gives (no atomics, and a per-element sum order that the band cannot
    # change).
    assert bool(torch.equal(tiled, unbanded))
    rel = _rel_max(tiled, reference)
    print(f"multiaxis back triton banded parity (band {band_slices}): "
          f"rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
@pytest.mark.parametrize("band_slices", [5, 12, 16, 32])
def test_multiaxis_back_kernel_pads_the_band_to_a_multiple_of_16(band_slices):
    """The wrapper launches the band rounded up to a multiple of 16.

    Three statements, at every band this sweeps.  The returned values are the
    torch body's at the design's 1e-5 gate, whether or not the band was rounded
    up.  They are also the values the unbanded call produces over the same
    slices, so the columns the rounding added changed nothing.  And the
    returned view's row stride is the width the wrapper allocated, which is the
    real band when the band is already a multiple of 16 -- the path that has to
    stay exactly what it was.

    This cell's volume is at least 32 slices, so bands of 16 and 32 need no
    padding and bands of 5 and 12 do.  The tail band of each tiling is shorter
    than the requested band, and its padded columns address slices past the end
    of the volume, which is the case the kernel's address clamps exist for.
    """
    model = _ma_model(cell=(6, 32, 20))
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    num_slices = int(args['num_slices'])
    assert num_slices >= 32
    unbanded = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                 view_params, **args)
    for slice_start in range(0, num_slices, band_slices):
        length = min(band_slices, num_slices - slice_start)
        banded = _multiaxis_back_view_batch_triton(
            sinogram, pixel_indices, view_params, slice_start=slice_start,
            band_slices=length, **args)
        reference = _multiaxis_back_view_batch(
            sinogram, pixel_indices, view_params, slice_start=slice_start,
            band_slices=length, **args)
        assert banded.shape == reference.shape
        assert bool(banded.isfinite().all())
        assert _rel_max(banded, reference) <= 1e-5
        window = unbanded[:, slice_start:slice_start + length]
        assert bool(torch.equal(banded, window))
        padded = padded_kernel_width(length)
        assert banded.stride(0) == padded, (length, padded)
        assert banded.is_contiguous() == (padded == length)


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, 7, 9, 31])
def test_multiaxis_back_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes that must contribute exactly
    # nothing.  The counts straddle that tile (see MULTIAXIS_BACK_BLOCK_P and
    # the minimum tile the wrapper shrinks to).  Two independent statements --
    # parity against the torch body, and the invariant that a pixel's output
    # does not depend on which lane of which block it landed in (the same
    # pixels inside a LARGER subset must give the same values).
    assert MULTIAXIS_BACK_BLOCK_P > 1
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    subset = pixel_indices[:num_pixels]
    reference = _multiaxis_back_view_batch(sinogram, subset, view_params,
                                           **args)
    kernel_out = _multiaxis_back_view_batch_triton(sinogram, subset,
                                                   view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                             view_params, **args)
    assert bool(torch.equal(kernel_out, full[:num_pixels]))


@requires_cuda
def test_multiaxis_back_kernel_delegates_exotic_coeff_power():
    # The kernel's coefficient power is a constexpr branch over 1 and 2 -- the
    # only powers any caller in the package uses.  Anything else DELEGATES to
    # the torch body rather than diverging from it, so the wrapper stays a
    # total drop-in replacement.
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    reference = _multiaxis_back_view_batch(sinogram, pixel_indices, view_params,
                                           coeff_power=3, **args)
    delegated = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                  view_params, coeff_power=3,
                                                  **args)
    assert bool(torch.equal(delegated, reference))


@requires_cuda
def test_multiaxis_back_kernel_repeat_consistency():
    # The kernel has no atomic adds: it gathers on both detector axes into a
    # register accumulator and stores each output element once, so two
    # identical calls must agree BIT for bit.  A tolerance here would hide a
    # scatter creeping back in.
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    first = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                              view_params, **args)
    second = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                               view_params, **args)
    assert bool(torch.equal(second, first))


@requires_cuda
def test_multiaxis_back_kernel_adjointness():
    # <F x, a> == <x, B a> with F the TORCH forward body and B the kernel: the
    # pairing the whole projector contract rests on, and the check that would
    # catch a weight or index convention that drifted only in the kernel.  The
    # elevation sweep is the point -- at zero elevation the vertical fan is
    # degenerate and would not exercise the tilt terms.
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _multiaxis_forward_view_batch(values, pixel_indices, view_params,
                                            **args)
    back = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                             view_params, **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"multiaxis back triton adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
def test_multiaxis_back_kernel_band_overhanging_the_volume():
    """A band that runs past the last slice, which is where the validity mask
    is the only thing keeping the two bodies together.

    The torch body multiplies its output by (global slice index < num_slices),
    so the columns of an overhanging band are zero rather than whatever the
    slice-to-row map produced there.  The kernel applies the same test on the
    same global index.  This cell's slice count is not a multiple of the
    kernel's slice tile, so the overhanging columns are also padded lanes of
    the launch.
    """
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    num_slices = int(args['num_slices'])
    slice_start = num_slices - 4
    band_slices = 8
    kernel_out = _multiaxis_back_view_batch_triton(
        sinogram, pixel_indices, view_params, slice_start=slice_start,
        band_slices=band_slices, **args)
    reference = _multiaxis_back_view_batch(
        sinogram, pixel_indices, view_params, slice_start=slice_start,
        band_slices=band_slices, **args)
    assert kernel_out.shape == (pixel_indices.shape[0], band_slices)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    # The overhanging columns are identically zero in both, so the shared
    # relative ruler is read on the real columns and the zeros are stated
    # directly.
    assert bool((reference[:, 4:] == 0).all())
    assert bool((kernel_out[:, 4:] == 0).all())
    rel = _rel_max(kernel_out[:, :4], reference[:, :4])
    print(f"multiaxis back triton overhanging band: rel_max = {rel:.2e}")
    assert rel <= 1e-5


# The cell the forward COVERAGE tests use.  A thin voxel slice pitch against
# the detector row pitch lowers the slice-to-row slope without touching the
# vertical footprint, which is exactly what makes more slices reach one row
# than psf_radius counts.  It is deliberately not one of the VARIANTS: those
# sweep the contract for the parity gates, and every one of them happens to
# need a single slice tap on each side.
THIN_SLICE_CELL = dict(elevation="tilt", slice_aspect=0.5)


def _vertical_terms(model):
    """The vertical fan's per-view terms for a model's full pixel set, built by
    the same eager builders the bodies and the kernels use: (m0, slope, W_p_r,
    args)."""
    device = model.torch_device
    recon_shape = model.get_params('recon_shape')
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    view_params = torch.as_tensor(model.get_params('angles'),
                                  dtype=torch.float32, device=device)
    args = model._view_batch_args()
    azimuth, elevation = view_params[:, 0], view_params[:, 1]
    _, _, _, _, y = _multiaxis_horizontal_data(
        pixel_indices, azimuth, args['num_recon_rows'], args['num_recon_cols'],
        args['num_channels'], args['delta_voxel'], args['delta_voxel_row'],
        args['delta_det_channel'], args['det_channel_offset'])
    m0, slope, w_p_r, _, _ = _multiaxis_vertical_terms(
        y, azimuth, elevation, args['num_slices'], args['delta_voxel'],
        args['delta_voxel_row'], args['delta_voxel_slice'],
        args['delta_det_row'], args['det_row_offset'],
        args['recon_slice_offset'], args['num_rows_r'])
    return m0, slope, w_p_r, args


def _slice_reach(model):
    """How far the forward kernel's slice enumeration REALLY has to reach on
    this model, counted rather than derived.

    Returns the largest |k - round(k_center)| over every (pixel, detector row,
    slice) triple whose vertical trapezoid weight is nonzero -- the exact
    quantity :func:`mbirtorch.triton_multiaxis._multiaxis_slice_tap_radius` has
    to bound.  Brute force over the whole geometry, so it is an independent
    ruler for that bound and not a second copy of it.
    """
    m0, slope, w_p_r, args = _vertical_terms(model)
    device = m0.device
    k = torch.arange(int(args['num_slices']), dtype=torch.float32,
                     device=device)
    m = torch.arange(int(args['num_rows_r']), dtype=torch.float32,
                     device=device)
    reach = 0
    for v in range(m0.shape[0]):
        rise, width = float(slope[v, 0]), float(w_p_r[v, 0])
        m_p = m0[v][:, None] + rise * k[None, :]                      # (P, S)
        live = (m_p[:, :, None] - m[None, None, :]).abs() < (width + 1.0) / 2.0
        k_center = torch.floor((m[None, :] - m0[v][:, None]) / rise + 0.5)
        dist = (k[None, :, None] - k_center[:, None, :]).abs()      # (P, S, R)
        if bool(live.any()):
            reach = max(reach, int(dist[live].max()))
    # Same order as _vertical_terms above, minus m0, so the two cannot be
    # unpacked into each other's names by accident.
    return reach, slope, w_p_r, args


@pytest.mark.parametrize("variant", list(VARIANTS) + ["thin_slice"])
def test_multiaxis_forward_slice_tap_radius_covers_every_contributing_slice(
        variant):
    """The forward kernel's one hard requirement, checked on any machine.

    The kernel gathers where the torch body scatters, so it enumerates slices
    k = round(k_center) + t for |t| <= radius and anything outside that window
    is silently dropped.  This counts the contributing triples directly and
    asserts the wrapper's bound covers them, which is the statement a parity
    test can only make indirectly (a dropped slice at the edge of a trapezoid
    carries a small weight and can hide under a relative tolerance).
    """
    kwargs = (THIN_SLICE_CELL if variant == "thin_slice"
              else VARIANTS[variant])
    model = _ma_model(device='cpu', **kwargs)
    reach, slope, w_p_r, args = _slice_reach(model)
    radius = _multiaxis_slice_tap_radius(w_p_r, slope)
    print(f"multiaxis forward slice-tap coverage ({variant}): "
          f"reach = {reach}, bound = {radius}, psf_radius = "
          f"{args['psf_radius']}")
    assert radius >= reach
    # The bound is not free to grow without limit either.  Two statements of
    # that: it stays under the wrapper's cap, so these cells launch the kernel
    # rather than delegating, and the window it enumerates stays smaller than
    # the volume, so the gather never walks more slices than a scatter would
    # have touched.  A bound that inflated would show up here rather than only
    # in a sweep.  Neither is a tightness gate -- the bound carries a
    # deliberate margin, and zero_elevation is the case where that margin is
    # widest, because its boundary taps carry weight exactly zero.
    assert radius <= MULTIAXIS_FWD_MAX_SLICE_RADIUS
    assert 2 * radius + 1 <= int(args['num_slices'])


def test_multiaxis_forward_psf_radius_would_not_cover_a_thin_slice_pitch():
    """Why the forward carries its own radius instead of reusing psf_radius.

    psf_radius counts DETECTOR ROW taps around a slice; the forward's gather
    needs SLICE taps around a row, and the two differ by the slope of the
    slice-to-row map.  Where the slope falls well below 1 the second count
    exceeds the first, and a slice-tap loop bounded by psf_radius would drop
    real contributions.  This states that gap on the cell the coverage tests
    use, and states the honest converse for the parity variants: none of them
    reaches it, so they alone would not have caught the mistake.
    """
    model = _ma_model(device='cpu', **THIN_SLICE_CELL)
    reach, slope, w_p_r, args = _slice_reach(model)
    assert reach > int(args['psf_radius'])
    for variant, kwargs in VARIANTS.items():
        variant_reach, _, _, variant_args = _slice_reach(
            _ma_model(device='cpu', **kwargs))
        assert variant_reach <= int(variant_args['psf_radius']), variant


@requires_cuda
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_multiaxis_forward_kernel_parity(variant):
    # As for the back kernel: the geometry reaches the forward kernel only
    # through the two eager builders it shares with the torch body, so all five
    # variants exercise the same kernel with different contract values.  The
    # forward body takes no coeff_power.
    model = _ma_model(**VARIANTS[variant])
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    reference = _multiaxis_forward_view_batch(values, pixel_indices,
                                              view_params, **args)
    kernel_out = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                      view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"multiaxis forward triton parity ({variant}, "
          f"psf_radius={args['psf_radius']}): rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_multiaxis_forward_kernel_parity_where_coverage_bites():
    # The parity statement on the cell where the slice enumeration must reach
    # past psf_radius (see the coverage tests above, which establish that it
    # does).  A kernel that bounded its slice taps by psf_radius would pass
    # every parity test above and fail this one.
    model = _ma_model(**THIN_SLICE_CELL)
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    reference = _multiaxis_forward_view_batch(values, pixel_indices,
                                              view_params, **args)
    kernel_out = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                      view_params, **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    rel = _rel_max(kernel_out, reference)
    print(f"multiaxis forward triton parity (thin slice pitch): "
          f"rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_multiaxis_forward_kernel_banded_parity():
    # The banded seam, forward form: each band carries its own slice of the
    # VALUES and every band writes the whole sinogram, so a tiling of the slice
    # axis SUMS to the unbanded projection (the back kernel's bands
    # concatenate).  The slice-to-row map is anchored on the FULL slice count,
    # which is what makes that true.
    model = _ma_model(cell=(6, 32, 20))
    _, pixel_indices, view_params, args = _body_inputs(model)
    num_slices = int(args['num_slices'])
    values = _voxel_values(model, pixel_indices)
    unbanded = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                    view_params, **args)
    reference = _multiaxis_forward_view_batch(values, pixel_indices,
                                              view_params, **args)
    tiled = None
    for slice_start in range(0, num_slices, 5):
        band = values[:, slice_start:slice_start + 5]
        block = _multiaxis_forward_view_batch_triton(
            band, pixel_indices, view_params, slice_start=slice_start, **args)
        assert block.shape == unbanded.shape
        tiled = block if tiled is None else tiled + block
    # Kernel against itself, at a TOLERANCE rather than bit for bit: the
    # forward scatters with atomic adds, so the sum order differs between the
    # banded and unbanded launches and bit equality would be the wrong claim.
    band_rel = _rel_max(tiled, unbanded)
    rel = _rel_max(tiled, reference)
    print(f"multiaxis forward triton banded parity: "
          f"sum-vs-unbanded = {band_rel:.2e}, vs body = {rel:.2e}")
    assert band_rel <= 1e-6
    assert rel <= 1e-5


@requires_cuda
@pytest.mark.parametrize("cell,detector_rows", [((6, 12, 12), 12),
                                                ((6, 32, 20), 32)])
def test_multiaxis_forward_kernel_pads_the_detector_rows_to_a_multiple_of_16(
        cell, detector_rows):
    """The forward's width-class argument is its DETECTOR ROW count, which is
    the output's row stride, so that is what the wrapper rounds up.

    The 12-row detector is rounded up to 16 and the 32-row one is not, so both
    paths are exercised.  Three statements at each.  The returned shape is
    exactly (Vb, R, C) -- the padding never reaches a caller.  The values are
    the torch body's at the design's 1e-5 gate either way.  And the returned
    view's last stride is the row count the wrapper really ALLOCATED, which is
    the padded one; the torch body returns the same permuted layout at the real
    row count, so the two differ by exactly that padding and nothing else.
    """
    model = _ma_model(cell=cell)
    _, pixel_indices, view_params, args = _body_inputs(model)
    assert int(args['num_rows_r']) == detector_rows
    values = _voxel_values(model, pixel_indices)
    kernel_out = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                      view_params, **args)
    reference = _multiaxis_forward_view_batch(values, pixel_indices,
                                              view_params, **args)
    num_views = int(view_params.shape[0])
    assert kernel_out.shape == (num_views, detector_rows,
                                int(args['num_channels']))
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5
    # The permute leaves the channel-major row axis on the unit stride and the
    # channel axis on the allocated row count; the torch body's own permute of
    # a contiguous (Vb, C, R) does the same at the unpadded count.
    padded = padded_kernel_width(detector_rows)
    assert kernel_out.stride(1) == 1
    assert kernel_out.stride(2) == padded, (detector_rows, padded)
    assert reference.stride(1) == 1
    assert reference.stride(2) == detector_rows


@requires_cuda
@pytest.mark.parametrize("num_pixels", [1, 7, 9, 31])
def test_multiaxis_forward_kernel_pixel_padding(num_pixels):
    # Poison the padding: a pixel count that is not a multiple of the kernel's
    # pixel tile pads the last block with lanes whose atomics must be masked off
    # entirely.  The counts straddle that tile (see MULTIAXIS_FWD_BLOCK_P and
    # the minimum tile the wrapper shrinks to).  Two independent statements --
    # parity against the torch body, and additivity over a pixel SPLIT (the
    # forward sums all pixels into one sinogram, so a subset's output is a
    # partial sum and the two parts must reassemble the whole however the blocks
    # were padded).  The split is compared at a tolerance, not bit for bit:
    # the atomics reorder the sum over pixels, so the three launches add the
    # same terms in different orders by construction.
    assert MULTIAXIS_FWD_BLOCK_P > 1
    model = _ma_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    subset, rest = pixel_indices[:num_pixels], pixel_indices[num_pixels:]
    reference = _multiaxis_forward_view_batch(values[:num_pixels], subset,
                                              view_params, **args)
    kernel_out = _multiaxis_forward_view_batch_triton(values[:num_pixels],
                                                      subset, view_params,
                                                      **args)
    assert kernel_out.shape == reference.shape
    assert bool(kernel_out.isfinite().all())
    assert _rel_max(kernel_out, reference) <= 1e-5

    full = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                view_params, **args)
    rest_out = _multiaxis_forward_view_batch_triton(values[num_pixels:], rest,
                                                    view_params, **args)
    assert _rel_max(kernel_out + rest_out, full) <= 1e-6


@requires_cuda
def test_multiaxis_forward_kernel_repeat_consistency():
    # The forward scatters with tl.atomic_add, so the summation order over
    # pixels and taps is whatever the hardware schedules that launch: identical
    # inputs give results that agree to float rounding, not bit for bit.  This
    # measures that spread instead of assuming it -- if it ever prints above
    # ~1e-6 the parity tolerances above are the thing carrying it, and this is
    # where the evidence lives.
    model = _ma_model()
    _, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    first = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                 view_params, **args)
    second = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                  view_params, **args)
    rel = _rel_max(second, first)
    print(f"multiaxis forward triton repeat consistency: rel_max = {rel:.2e}")
    assert rel <= 1e-5


@requires_cuda
def test_multiaxis_forward_kernel_adjointness():
    # <F x, a> == <x, B a> with F the kernel forward and B the TORCH back body:
    # the pairing the whole projector contract rests on, and the check that
    # would catch a weight or index convention that drifted only in the kernel.
    # The elevation sweep is the point -- at zero elevation the vertical fan is
    # degenerate and would not exercise the tilt terms.
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                   view_params, **args)
    back = _multiaxis_back_view_batch(sinogram, pixel_indices, view_params,
                                      **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"multiaxis forward triton adjointness: lhs {lhs:.6f}, "
          f"rhs {rhs:.6f}, rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
def test_multiaxis_kernel_pair_adjointness():
    # The pairing that actually ships once both kernels are on: KERNEL forward
    # against KERNEL back.  The two adjointness tests above each hold one side
    # fixed to the torch body, so a convention that drifted in BOTH kernels
    # together would pass them and fail here.  It matters more for this
    # geometry than for the others, because the two kernels do not mirror each
    # other on the vertical axis -- the back gathers slices from rows and the
    # forward gathers rows from slices, through the two algebraic forms of the
    # one affine map.
    model = _ma_model()
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    forward = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                   view_params, **args)
    back = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                             view_params, **args)
    lhs = float((forward * sinogram).sum())
    rhs = float((values * back).sum())
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    print(f"multiaxis kernel-pair adjointness: lhs {lhs:.6f}, rhs {rhs:.6f}, "
          f"rel {rel:.2e}")
    assert rel <= 1e-4


@requires_cuda
def test_multiaxis_kernels_span_several_row_chunks():
    # Every cell above is narrower than the forward kernel's row tile, so its
    # row grid axis holds exactly one program and a bug in that offset would
    # not show.  _tile_size caps the tile at next_pow2(extent), so forcing more
    # chunks takes more rows than the pinned tile: this cell gives the forward
    # kernel several detector row chunks and the back kernel several slice
    # chunks.
    cell = (4, 4 * MULTIAXIS_FWD_BLOCK_R, 16)
    model = _ma_model(cell=cell)
    sinogram, pixel_indices, view_params, args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    assert int(args['num_rows_r']) > MULTIAXIS_FWD_BLOCK_R

    fwd_ref = _multiaxis_forward_view_batch(values, pixel_indices, view_params,
                                            **args)
    fwd_out = _multiaxis_forward_view_batch_triton(values, pixel_indices,
                                                   view_params, **args)
    fwd_rel = _rel_max(fwd_out, fwd_ref)
    back_ref = _multiaxis_back_view_batch(sinogram, pixel_indices, view_params,
                                          **args)
    back_out = _multiaxis_back_view_batch_triton(sinogram, pixel_indices,
                                                 view_params, **args)
    back_rel = _rel_max(back_out, back_ref)
    print(f"multiaxis triton multi-row-chunk ({args['num_rows_r']} rows): "
          f"forward rel_max = {fwd_rel:.2e}, back rel_max = {back_rel:.2e}")
    # 1e-4 here rather than the 1e-5 the small cells gate at, and the reason
    # is the row count, not the chunking.  The trapezoid weight subtracts two
    # row coordinates of size about num_rows_r, and the kernel and the torch
    # body round the m0 + slope * k forming them differently (the compiler may
    # fuse the multiply-add), so the weight carries an absolute perturbation of
    # about num_rows_r times float32 eps -- about 6e-5 at these 512 rows, read
    # about 2e-5 on both directions when this gate first ran.  The small-cell
    # parity tests above hold the 1e-5 statement where the coordinates cannot
    # inflate it; what this test states is that the chunk offsets are right,
    # and an offset bug would miss by orders of magnitude, not by rounding.
    assert fwd_rel <= 1e-4
    assert back_rel <= 1e-4


@requires_cuda
def test_multiaxis_forward_kernel_delegates_a_vanishing_slope():
    # The gather inverts the slice-to-row map, so it needs a slope bounded away
    # from zero -- the reason multiaxis_parallel.py's forward body scatters
    # instead.  A slice pitch thin enough to push the slice-tap radius past the
    # wrapper's cap DELEGATES to the torch body rather than launching an
    # enumeration it cannot cover, so the wrapper stays a total drop-in
    # replacement.
    model = _ma_model(cell=(4, 8, 8), elevation="tilt", slice_aspect=0.02)
    _, pixel_indices, view_params, args = _body_inputs(model)
    _, slope, w_p_r, _ = _vertical_terms(model)
    assert _multiaxis_slice_tap_radius(w_p_r, slope) \
        > MULTIAXIS_FWD_MAX_SLICE_RADIUS
    values = _voxel_values(model, pixel_indices)
    # The delegation is asserted by IDENTITY, not by comparing values: the
    # forward body scatters with scatter_add_ and index_add_, which are
    # nondeterministic on CUDA, so two calls of the body itself differ at
    # rounding and a value comparison would test that noise instead.  The
    # wrapper calls the body through this module-level name, so recording the
    # body's own return and asserting the wrapper handed it back unchanged is
    # the whole statement.
    import mbirtorch.triton_multiaxis as tm
    returned = []
    real_body = tm._multiaxis_forward_view_batch

    def recording_body(*body_args, **body_kwargs):
        out = real_body(*body_args, **body_kwargs)
        returned.append(out)
        return out

    try:
        tm._multiaxis_forward_view_batch = recording_body
        delegated = _multiaxis_forward_view_batch_triton(
            values, pixel_indices, view_params, **args)
    finally:
        tm._multiaxis_forward_view_batch = real_body
    assert len(returned) == 1
    assert delegated is returned[0]


def test_multiaxis_kernel_module_imports_and_declares_its_contract(monkeypatch):
    # Runs everywhere, including a CPU-only host with no triton at all.  Four
    # statements, each an existing rule for a kernel module, and each made for
    # BOTH wrappers: the module imports without triton (this test file's own
    # top-level import would have failed otherwise, and the reimport states it
    # directly); each wrapper raises a clear RuntimeError rather than a compile
    # error when triton is absent; each cost declaration answers the driver's
    # (bytes_per_view, view_chunk) pair with two positive integers; and each
    # wrapper carries the no-compile marker, which maybe_compile is the thing
    # that reads.
    import importlib

    from mbirtorch.projectors import maybe_compile

    module = importlib.import_module('mbirtorch.triton_multiaxis')
    assert module._multiaxis_back_view_batch_triton \
        is _multiaxis_back_view_batch_triton
    assert module._multiaxis_forward_view_batch_triton \
        is _multiaxis_forward_view_batch_triton

    args = {'num_channels': 20, 'num_rows_r': 24}
    bytes_per_view, view_chunk = _multiaxis_back_view_batch_cost(1000, 24, args)
    assert isinstance(bytes_per_view, int) and bytes_per_view > 0
    assert isinstance(view_chunk, int) and view_chunk > 0
    assert view_chunk == MULTIAXIS_BACK_VIEW_CHUNK
    assert (_multiaxis_back_view_batch_triton._view_batch_cost
            is _multiaxis_back_view_batch_cost)

    fwd_bytes, fwd_chunk = _multiaxis_forward_view_batch_cost(1000, 24, args)
    assert isinstance(fwd_bytes, int) and fwd_bytes > 0
    assert isinstance(fwd_chunk, int) and fwd_chunk > 0
    assert fwd_chunk == MULTIAXIS_FWD_VIEW_CHUNK
    # The forward's output plane is allocated at the PADDED row count, so the
    # charge reads the padded value and the code and the charge cannot
    # disagree (24 rows round up to 32).
    assert fwd_bytes == 12 * 1000 + 4 * 20 * padded_kernel_width(24)
    assert (_multiaxis_forward_view_batch_triton._view_batch_cost
            is _multiaxis_forward_view_batch_cost)

    for wrapper in (_multiaxis_back_view_batch_triton,
                    _multiaxis_forward_view_batch_triton):
        assert wrapper._mbirtorch_no_compile is True
        assert maybe_compile(wrapper, True, instance_key=0) is wrapper
    # The torch bodies are still compiled, so the marker is not a blanket
    # opt-out.
    assert maybe_compile(_multiaxis_back_view_batch, True,
                         instance_key=0) is not _multiaxis_back_view_batch
    assert maybe_compile(_multiaxis_forward_view_batch, True,
                         instance_key=0) is not _multiaxis_forward_view_batch

    # The no-triton guard, forced so it is read on any machine (on a CUDA host
    # triton is importable and the wrappers would launch instead).
    monkeypatch.setattr(module, 'triton', None)
    model = _ma_model(device='cpu')
    sinogram, pixel_indices, view_params, body_args = _body_inputs(model)
    values = _voxel_values(model, pixel_indices)
    with pytest.raises(RuntimeError, match='triton'):
        _multiaxis_back_view_batch_triton(sinogram, pixel_indices, view_params,
                                          **body_args)
    with pytest.raises(RuntimeError, match='triton'):
        _multiaxis_forward_view_batch_triton(values, pixel_indices, view_params,
                                             **body_args)


# ── selection ────────────────────────────────────────────────────────────────
# The model binds a kernel wherever its availability gate passes.  These tests
# state that contract: what a CUDA machine really binds and what it computes
# through the driver, and -- on any machine -- the policy itself, the caching,
# and every way the fast path can be absent.

# The tolerance a whole RECONSTRUCTION is compared at, where the projection
# tests above compare single calls at 1e-5.  A reconstruction is an iterative
# solver, so a per-call float difference is carried forward and reshaped by
# every later update; 5e-3 is the figure the other model-level gates on this
# geometry use (tests/test_multiaxis.py's sharded-vs-single recon, which
# records a measured spread of 9.4e-4 against that gate, and the
# kernel-times-sharding gate in tests/test_kernels_sharded.py).
RECON_TOLERANCE = 5e-3


@requires_cuda
def test_multiaxis_kernel_selection_and_end_to_end(monkeypatch):
    # The selection contract: both kernels are ON with no environment variable
    # at all, wherever the probe and the per-device self-checks pass, and the
    # kill switch still forces the torch bodies.  A model built that way
    # reproduces the torch projectors end to end THROUGH the driver (view
    # batching, lazy accumulation, and the maybe_compile wrapper the bodies
    # must survive without being traced).  The torch reference is built under
    # the kill switch, because the default now selects the kernels.
    from mbirtorch import projectors

    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _ma_model(compile_mode='auto')
        assert model._view_batch_bodies() == (_multiaxis_forward_view_batch,
                                              _multiaxis_back_view_batch)
        model.create_projectors()
        sinogram, pixel_indices, _, _ = _body_inputs(model)
        values = _voxel_values(model, pixel_indices)
        back_reference = model.sparse_back_project(sinogram, pixel_indices)
        fwd_reference = model.sparse_forward_project(values, pixel_indices)

        monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR)
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        for gate in (kernel_availability.multiaxis_back_kernel_usable,
                     kernel_availability.multiaxis_forward_kernel_usable):
            usable, reason = gate(model)
            assert isinstance(reason, str) and reason
            assert usable, reason
        assert model._view_batch_bodies() == (
            _multiaxis_forward_view_batch_triton,
            _multiaxis_back_view_batch_triton)

        model.create_projectors()
        # The driver holds the kernel bodies THEMSELVES, uncompiled, even with
        # compile_mode='auto' (the _mbirtorch_no_compile seam).
        assert (model.projector_functions._back_body_per_dev[0]
                is _multiaxis_back_view_batch_triton)
        assert (model.projector_functions._fwd_body_per_dev[0]
                is _multiaxis_forward_view_batch_triton)
        back_out = model.sparse_back_project(sinogram, pixel_indices)
        fwd_out = model.sparse_forward_project(values, pixel_indices)

        back_rel = _rel_max(back_out, back_reference)
        fwd_rel = _rel_max(fwd_out, fwd_reference)
        print(f"multiaxis triton end-to-end: back rel_max = {back_rel:.2e}, "
              f"forward rel_max = {fwd_rel:.2e}")
        assert back_rel <= 1e-5
        assert fwd_rel <= 1e-5

        # The kill switch reaches the selected kernels; it is read INSIDE the
        # probe, so it takes effect across a cache reset.
        monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()
        assert model._view_batch_bodies() == (_multiaxis_forward_view_batch,
                                              _multiaxis_back_view_batch)
        # ... and the kernels ran eagerly, rather than reaching eager by way of
        # a compile failure that maybe_compile swallowed.
        assert not [k for k in projectors._COMPILE_ERRORS
                    if 'triton_multiaxis' in k]
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@requires_cuda
def test_multiaxis_kernel_recon_matches_a_torch_bodies_recon(monkeypatch):
    # The composition a user on a CUDA machine actually gets: a whole seeded
    # reconstruction through the kernel route, against the same reconstruction
    # on the torch bodies.  The reference FORCES the torch bodies rather than
    # setting compile_mode, because selection is availability-driven and eager
    # does not mean unkernelled.
    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        cell = (12, 24, 20)
        model = _ma_model(cell=cell, compile_mode='off')
        # The arm check: this test exists to measure the KERNELS, so a silent
        # availability decline must fail loudly rather than compare torch with
        # torch and pass vacuously.
        fwd, back = model._view_batch_bodies()
        assert fwd is _multiaxis_forward_view_batch_triton
        assert back is _multiaxis_back_view_batch_triton

        recon_shape = model.get_params('recon_shape')
        phantom = mbirtorch.gen_translation_phantom(recon_shape, 'dots', None,
                                                    fill_rate=0.05)
        sinogram = np.asarray(model.forward_project(phantom), dtype=np.float32)
        np.random.seed(0)
        kernel_recon, _info = model.recon(sinogram, max_iterations=3,
                                          stop_threshold_change_pct=0.0,
                                          logfile_path=None)

        reference_model = _ma_model(cell=cell, compile_mode='off')
        reference_model._view_batch_bodies = lambda: (
            _multiaxis_forward_view_batch, _multiaxis_back_view_batch)
        reference_model.create_projectors()
        np.random.seed(0)
        torch_recon, _info = reference_model.recon(
            sinogram, max_iterations=3, stop_threshold_change_pct=0.0,
            logfile_path=None)

        kernel_recon = np.asarray(kernel_recon, dtype=np.float64)
        torch_recon = np.asarray(torch_recon, dtype=np.float64)
        rel = float(np.max(np.abs(kernel_recon - torch_recon))
                    / np.max(np.abs(torch_recon)))
        print(f"multiaxis triton recon vs torch-bodies recon: "
              f"rel_max = {rel:.2e}")
        assert rel < RECON_TOLERANCE
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@pytest.mark.parametrize("gate_name", ["multiaxis_back_kernel_usable",
                                       "multiaxis_forward_kernel_usable"])
def test_multiaxis_self_check_is_a_cached_pair(gate_name, monkeypatch):
    # Runs everywhere: each gate must answer (bool, str) and cache per device,
    # whatever the machine underneath.
    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    gate = getattr(kernel_availability, gate_name)
    try:
        model = _ma_model(device='cpu')
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
    [("multiaxis_back_kernel_usable", "_multiaxis_back_view_batch_triton",
      1, _multiaxis_back_view_batch),
     ("multiaxis_forward_kernel_usable",
      "_multiaxis_forward_view_batch_triton", 0,
      _multiaxis_forward_view_batch)])
def test_multiaxis_self_check_catches_a_broken_kernel(
        gate_name, body_name, index, torch_body, monkeypatch):
    # Runs everywhere: the self-check exists to catch a toolchain that compiles
    # the probe and then miscompiles (or fails to compile) the real kernel.
    # With the probe forced to pass and the kernel body raising, the gate must
    # report a REASON and the model must keep the torch body -- never propagate
    # the failure to a caller who only asked what was available.
    from mbirtorch import triton_multiaxis

    def _exploding_body(*args, **kwargs):
        raise RuntimeError('simulated broken kernel')

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(kernel_availability, '_probe_triton',
                        lambda: (True, 'forced-available probe'))
    monkeypatch.setattr(triton_multiaxis, body_name, _exploding_body)
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    try:
        model = _ma_model(device='cpu')
        usable, reason = getattr(kernel_availability, gate_name)(model)
        assert usable is False
        assert 'simulated broken kernel' in reason
        assert model._view_batch_bodies()[index] is torch_body
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


@pytest.mark.parametrize(
    "gate_name,index,torch_body",
    [("multiaxis_back_kernel_usable", 1, _multiaxis_back_view_batch),
     ("multiaxis_forward_kernel_usable", 0, _multiaxis_forward_view_batch)])
def test_multiaxis_gate_is_false_without_a_kernel_path(
        gate_name, index, torch_body, monkeypatch):
    # Two ways the fast path can be absent, both of which must produce a REASON
    # rather than an exception: the kill switch (any machine) and a host with
    # no CUDA at all (this machine, when it has none).  The switch is set and
    # unset inside the test, because it is read in the probe and the probe's
    # answer is cached until the cache is dropped.
    monkeypatch.setenv(kernel_availability.DISABLE_ENV_VAR, '1')
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    gate = getattr(kernel_availability, gate_name)
    try:
        model = _ma_model(device='cpu')
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
    [("multiaxis_back_kernel_usable", 1,
      _multiaxis_back_view_batch, _multiaxis_back_view_batch_triton),
     ("multiaxis_forward_kernel_usable", 0,
      _multiaxis_forward_view_batch, _multiaxis_forward_view_batch_triton)])
def test_multiaxis_kernels_select_by_default(gate_name, index, torch_body,
                                             kernel_body, monkeypatch):
    # The policy, stated separately from the gate: no environment variable is
    # consulted -- the gate's verdict alone decides, and a passing gate selects
    # the kernel by default.  Runs everywhere; the sentinel gate makes it
    # machine-independent.
    calls = []
    verdict = {'usable': True}

    def _spy(model):
        calls.append(model)
        return (verdict['usable'], 'sentinel gate')

    monkeypatch.delenv(kernel_availability.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(kernel_availability, gate_name, _spy)
    try:
        model = _ma_model(device='cpu')
        calls.clear()
        assert model._view_batch_bodies()[index] is kernel_body
        assert len(calls) == 1

        verdict['usable'] = False
        assert model._view_batch_bodies()[index] is torch_body
    finally:
        kernel_availability._reset_probe_cache()
        kernel_availability._reset_self_check_cache()


def test_multiaxis_selection_is_the_torch_bodies_on_a_cpu_host():
    # Where the not-routed test stood.  Routing must be a no-op on a machine
    # with no kernel path at all: the gates answer no without CUDA, so a CPU
    # host binds exactly the bodies it bound before the kernels existed.  On a
    # CUDA host the gates decide instead, and the selection tests above state
    # what they bind there.
    if torch.cuda.is_available():
        pytest.skip('a CUDA host selects through the gates; see '
                    'test_multiaxis_kernel_selection_and_end_to_end')
    model = _ma_model(device='cpu')
    forward_body, back_body = model._view_batch_bodies()
    assert forward_body is _multiaxis_forward_view_batch
    assert back_body is _multiaxis_back_view_batch
