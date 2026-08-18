"""The driver's view-batch rule: the batching model follows the BODY bound.

``Projectors._effective_view_batch`` chooses the view batch from the body it
is about to call.  A torch body (no ``_view_batch_cost`` attribute) batches by
the geometry's calibrated ``_transient_cols`` charge, exactly as it always
has; a hand-written kernel body carries a ``_view_batch_cost`` attribute
stating its own resident bytes per view and its nominal view chunk.  These
tests pin both paths and their interaction with ``view_batch_size``:

  - the torch path reproduces its long-standing numbers, including the
    parallel band-length rule, the cone ``max(num_slices, num_rows)``
    override, and the 64 default that ``view_batch_size=None`` resolves to;
  - a body with the cost attribute batches by (bytes_per_view, view_chunk),
    with the same budget cap and floor;
  - the two directions are consulted separately, so a mixed selection
    (kernel one way, torch body the other) batches each by its own model;
  - an explicit ``view_batch_size`` is the nominal for EVERY body;
  - the four kernel cost functions state the residency the design charges
    (contract bytes per (view, pixel) plus one sinogram-shaped plane per
    view) and ride on the wrapper functions the selection hook returns;
  - the shared rule that rounds a kernel's width argument up to a multiple
    of 16, which those charges read because it is what the wrappers
    allocate.

Everything here is arithmetic on function objects and tiny CPU models: no
kernel is ever launched, so the file runs anywhere.  The driver-level CUDA
checks (a chunked kernel loop against a single-batch reference, and the
realized batch through a real selection) live with the kernel batteries in
test_triton_cone.py / test_triton_parallel.py.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch._utils import KERNEL_WIDTH_MULTIPLE, padded_kernel_width
from mbirtorch.projectors import Projectors
from mbirtorch import triton_cone, triton_parallel


def _parallel_model(cell=(6, 12, 12), **kwargs):
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(cell, angles, 
                                        compile_mode='off', **kwargs)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


def _cone_model(cell=(6, 12, 12), **kwargs):
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles,
                                    source_detector_dist=4 * cell[2],
                                    source_iso_dist=2 * cell[2], 
                                    compile_mode='off', **kwargs)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


def _stub_body(bytes_per_view, view_chunk):
    """A body function carrying a kernel-style cost attribute."""
    def stub(*args, **kwargs):
        raise AssertionError('the batching tests never call a body')
    stub._view_batch_cost = lambda num_pixels, cols, args: (bytes_per_view,
                                                            view_chunk)
    return stub


def _torch_stub():
    def stub(*args, **kwargs):
        raise AssertionError('the batching tests never call a body')
    return stub


# ── the torch-body path: unchanged, and pinned ────────────────────────────────

def test_torch_body_default_resolves_to_64():
    # view_batch_size=None (the constructor default) resolves to the
    # long-standing 64 for a torch body with room under the budget.
    model = _parallel_model()
    assert model.view_batch_size is None
    pf = model.projector_functions
    vb = pf._effective_view_batch(_torch_stub(), 100, 12,
                                  model._view_batch_args())
    assert vb == Projectors.VIEW_BATCH_BODY_DEFAULT == 64


def test_torch_body_budget_cap_parallel_band_rule():
    # The parallel charge tracks the RUNTIME band length: at a pixel count
    # and band chosen so the CPU budget (a flat 2 GiB) binds, the batch is
    # budget // (P * band * 4), unchanged from the pre-kernel rule.
    model = _parallel_model()
    pf = model.projector_functions
    num_pixels, band = 2 ** 20, 100
    expected = (Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES
                // (num_pixels * band * 4))
    assert expected == 5
    vb = pf._effective_view_batch(_torch_stub(), num_pixels, band,
                                  model._view_batch_args())
    assert vb == expected


def test_torch_body_cone_override_ignores_band():
    # Cone's charge is the params-derived max(num_slices, num_rows), whatever
    # band the call requests: a 2-column band must still charge the full
    # width, and the spread between the two charges is what proves the
    # override is the one consulted.
    model = _cone_model()
    pf = model.projector_functions
    cols = model._transient_cols(2)
    assert cols == max(int(model.get_params('recon_shape')[2]),
                       int(model.get_params('sinogram_shape')[1])) > 2
    num_pixels = 10 ** 7
    budget = Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES
    expected = budget // (num_pixels * cols * 4)
    naive_band_charge = budget // (num_pixels * 2 * 4)
    assert expected != naive_band_charge
    vb = pf._effective_view_batch(_torch_stub(), num_pixels, 2,
                                  model._view_batch_args())
    assert vb == expected


def test_torch_body_explicit_view_batch_size_is_the_nominal():
    model = _parallel_model(view_batch_size=3)
    vb = model.projector_functions._effective_view_batch(
        _torch_stub(), 100, 12, model._view_batch_args())
    assert vb == 3


def test_torch_body_floor_is_one():
    # A charge past the whole budget still yields one view per batch.
    model = _parallel_model()
    vb = model.projector_functions._effective_view_batch(
        _torch_stub(), 10 ** 9, 10 ** 3, model._view_batch_args())
    assert vb == 1


def test_bound_bodies_carry_no_cost_attribute_off_cuda():
    # On a CPU build the selection hook binds the torch bodies, so the
    # driver's bound functions must not carry the kernel attribute -- the
    # discriminator the whole rule stands on.
    model = _parallel_model()
    pf = model.projector_functions
    for body in (pf._fwd_body_per_dev[0], pf._back_body_per_dev[0]):
        assert getattr(body, '_view_batch_cost', None) is None


# ── the kernel-body path ──────────────────────────────────────────────────────

def test_kernel_body_chunk_cap_binds_when_cheap():
    # A cheap kernel batch runs at its own chunk, not at the torch default.
    model = _parallel_model()
    vb = model.projector_functions._effective_view_batch(
        _stub_body(10 * 2 ** 20, 128), 100, 12, model._view_batch_args())
    assert vb == 128


def test_kernel_body_budget_cap_binds_when_expensive():
    # 300 MiB per view against the flat 2 GiB CPU budget: 6 views.
    model = _parallel_model()
    vb = model.projector_functions._effective_view_batch(
        _stub_body(300 * 2 ** 20, 128), 100, 12, model._view_batch_args())
    assert vb == 6


def test_kernel_body_floor_is_one():
    model = _parallel_model()
    vb = model.projector_functions._effective_view_batch(
        _stub_body(3 * 2 ** 30, 128), 100, 12, model._view_batch_args())
    assert vb == 1


def test_kernel_body_explicit_view_batch_size_caps_it_too():
    # The user's knob keeps its meaning for every body: an explicit nominal
    # replaces the kernel's chunk, in both directions of the comparison.
    model = _parallel_model(view_batch_size=3)
    vb = model.projector_functions._effective_view_batch(
        _stub_body(10 * 2 ** 20, 128), 100, 12, model._view_batch_args())
    assert vb == 3
    model = _parallel_model(view_batch_size=256)
    vb = model.projector_functions._effective_view_batch(
        _stub_body(10 * 2 ** 20, 128), 100, 12, model._view_batch_args())
    assert vb == 204  # min(256, 2 GiB // 10 MiB)


def test_mixed_selection_batches_each_direction_by_its_own_model():
    # Kernel one way, torch body the other (the self-check fallback shape):
    # the same driver, the same inputs, two different batches.
    model = _parallel_model()
    pf = model.projector_functions
    args = model._view_batch_args()
    num_pixels, band = 2 ** 20, 100
    kernel_vb = pf._effective_view_batch(_stub_body(10 * 2 ** 20, 128),
                                         num_pixels, band, args)
    torch_vb = pf._effective_view_batch(_torch_stub(), num_pixels, band, args)
    assert kernel_vb == 128
    assert torch_vb == 5


# ── the shared kernel-width padding rule ──────────────────────────────────────

def test_the_kernel_width_padding_rule_rounds_up_to_a_multiple_of_16():
    """One definition of the rule, read by the wrappers and by the ledger.

    Triton compiles a faster kernel for an integer argument it can prove is a
    multiple of 16, so a wrapper rounds its width argument up to one before
    the launch.  A width that is ALREADY a multiple must come back unchanged:
    that is what makes the padding cost nothing at the production widths, and
    what lets a wrapper compare the result against its input to take its
    original path.
    """
    assert KERNEL_WIDTH_MULTIPLE == 16
    for width in (16, 32, 48, 256, 512, 1008, 2016):
        assert padded_kernel_width(width) == width, width
    for width, padded in ((1, 16), (8, 16), (15, 16), (17, 32), (31, 32),
                          (252, 256), (504, 512)):
        assert padded_kernel_width(width) == padded, width
    # A width below one multiple rounds up to the first one rather than
    # staying where it is; zero has nothing to round.
    assert padded_kernel_width(1) == KERNEL_WIDTH_MULTIPLE
    assert padded_kernel_width(0) == 0
    # It never shrinks a width, and never adds a whole multiple.
    for width in range(1, 200):
        padded = padded_kernel_width(width)
        assert width <= padded < width + KERNEL_WIDTH_MULTIPLE, width
        assert padded % KERNEL_WIDTH_MULTIPLE == 0, width


# ── the four kernel cost functions ────────────────────────────────────────────

def test_parallel_cost_functions_state_the_designed_residency():
    # Both parallel wrappers allocate their plane at the width rounded up to
    # a multiple of 16, so both charges read the padded width: 8 columns are
    # charged as 16.
    args = {'num_channels': 12}
    bytes_pv, chunk = triton_parallel._parallel_back_view_batch_cost(
        1000, 8, args)
    assert bytes_pv == 16 * 1000 + 4 * 12 * 16
    assert chunk == triton_parallel.PARALLEL_BACK_VIEW_CHUNK
    bytes_pv, chunk = triton_parallel._parallel_forward_view_batch_cost(
        1000, 8, args)
    assert bytes_pv == 16 * 1000 + 4 * 12 * 16
    assert chunk == triton_parallel.PARALLEL_FWD_VIEW_CHUNK
    # A width that is already a multiple of 16 is charged unchanged.
    assert triton_parallel._parallel_back_view_batch_cost(
        1000, 16, args)[0] == 16 * 1000 + 4 * 12 * 16
    assert triton_parallel._parallel_forward_view_batch_cost(
        1000, 16, args)[0] == 16 * 1000 + 4 * 12 * 16


def test_cone_cost_functions_state_the_designed_residency():
    args = {'num_channels': 12, 'num_rows_r': 10}
    # The cone back kernel reads a sinogram copy the wrapper does NOT pad --
    # every address it forms is clamped into that copy -- so the back plane
    # term reads the real row count.
    bytes_pv, chunk = triton_cone._cone_back_view_batch_cost(1000, 10, args)
    assert bytes_pv == 48 * 1000 + 4 * 12 * 10
    assert chunk == triton_cone.CONE_BACK_VIEW_CHUNK
    # The forward's output plane spans the full detector rows whatever slice
    # band the values carry: the band length must not enter the charge.  The
    # wrapper allocates that plane at the row count rounded up to a multiple
    # of 16, so a 10-row detector is charged 16 rows.
    banded = triton_cone._cone_forward_view_batch_cost(1000, 3, args)
    unbanded = triton_cone._cone_forward_view_batch_cost(1000, 10, args)
    assert banded == unbanded
    assert banded[0] == 48 * 1000 + 4 * 12 * 16
    assert banded[1] == triton_cone.CONE_FWD_VIEW_CHUNK
    # A detector whose row count is already a multiple of 16 is charged
    # unchanged.
    divisible = triton_cone._cone_forward_view_batch_cost(
        1000, 10, {'num_channels': 12, 'num_rows_r': 16})
    assert divisible[0] == 48 * 1000 + 4 * 12 * 16


def test_cost_attributes_ride_on_the_wrappers():
    # The attribute must sit on the exact function objects the selection hook
    # returns; maybe_compile passes them through unchanged (the no-compile
    # marker), so what the driver binds is what carries the cost.
    pairs = [
        (triton_parallel._parallel_back_view_batch_triton,
         triton_parallel._parallel_back_view_batch_cost),
        (triton_parallel._parallel_forward_view_batch_triton,
         triton_parallel._parallel_forward_view_batch_cost),
        (triton_cone._cone_back_view_batch_triton,
         triton_cone._cone_back_view_batch_cost),
        (triton_cone._cone_forward_view_batch_triton,
         triton_cone._cone_forward_view_batch_cost),
    ]
    for wrapper, cost in pairs:
        assert wrapper._view_batch_cost is cost
        assert wrapper._mbirtorch_no_compile


# ── the gate-cell arithmetic the change exists for ────────────────────────────

@pytest.mark.parametrize("geometry,contract_bytes", [("parallel", 16),
                                                     ("cone", 48)])
def test_gate_cell_batches_match_the_design_table(geometry, contract_bytes):
    # The defect and its repair, as pure arithmetic at the large gate cell
    # (views, rows, channels) = (1024, 1008, 992), full ROR pixel set: the
    # torch charge (~3.1 GB per view) forces view batch 1, while the kernel
    # charge admits its chunk (parallel) or a ~50-view budget cap (cone).
    # The budget at this cell is the 2 GiB ceiling on any backend, so a CPU
    # model reproduces the CUDA arithmetic exactly.
    num_pixels = int(np.pi / 4 * 992 ** 2)
    rows, channels = 1008, 992
    budget = Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES
    torch_charge = num_pixels * rows * 4
    assert torch_charge > budget           # the defect: view batch 1
    model = _parallel_model() if geometry == "parallel" else _cone_model()
    pf = model.projector_functions
    args = dict(model._view_batch_args())
    args['num_channels'] = channels
    if geometry == "parallel":
        cost_fn = triton_parallel._parallel_back_view_batch_cost
    else:
        cost_fn = triton_cone._cone_back_view_batch_cost
    body = _torch_stub()
    body._view_batch_cost = cost_fn
    vb = pf._effective_view_batch(body, num_pixels, rows, args)
    expected = min(128, budget // (contract_bytes * num_pixels
                                   + 4 * channels * rows))
    assert vb == expected
    assert vb == (128 if geometry == "parallel" else 52)
