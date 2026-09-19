"""The driver's view-batch rule: the batching model follows the BODY bound.

``Projectors._effective_view_batch`` chooses the view batch from the body it
is about to call.  A torch body (no ``_view_batch_cost`` attribute) batches by
the geometry's calibrated ``_transient_cols`` charge, exactly as it always
has; a hand-written kernel body carries a ``_view_batch_cost`` attribute
stating its own resident bytes per view and its nominal view chunk.  These
tests pin:

  - the arithmetic of both paths: the torch path's parallel band-length rule,
    its cone ``max(num_slices, num_rows)`` override and its floor of one, and
    the kernel path's own chunk when it is cheap and budget cap when it is
    expensive;
  - the shared rule that rounds a kernel's width argument up to a multiple
    of 16, which the kernel charges read because it is what the wrappers
    allocate;
  - the batches at the large gate cell this change exists for.

Everything here is arithmetic on function objects and tiny CPU models: no
kernel is ever launched, so the file runs anywhere.  The driver-level CUDA
checks (a chunked kernel loop against a single-batch reference, and the
realized batch through a real selection) live with the kernel batteries in
test_triton_cone.py / test_triton_parallel.py.
"""

import numpy as np
import pytest

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


# ── the view-batch arithmetic, torch body and kernel body ─────────────────────

def test_view_batch_arithmetic_for_torch_and_kernel_bodies():
    """Every number the batching rule produces, in one place.

    A torch body is charged the geometry's calibrated column count; a kernel
    body is charged the bytes per view it states and is capped at its own
    view chunk.  Both are capped by the transient budget and floored at one
    view.
    """
    budget = Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES

    # The parallel torch charge tracks the RUNTIME band length: at a pixel
    # count and band chosen so the CPU budget (a flat 2 GiB) binds, the batch
    # is budget // (P * band * 4), unchanged from the pre-kernel rule.
    model = _parallel_model()
    pf = model.projector_functions
    args = model._view_batch_args()
    num_pixels, band = 2 ** 20, 100
    expected = budget // (num_pixels * band * 4)
    assert expected == 5
    assert pf._effective_view_batch(_torch_stub(), num_pixels, band,
                                    args) == expected

    # A charge past the whole budget still yields one view per batch.
    assert pf._effective_view_batch(_torch_stub(), 10 ** 9, 10 ** 3,
                                    args) == 1

    # A cheap kernel batch runs at its own chunk, not at the torch default.
    assert pf._effective_view_batch(_stub_body(10 * 2 ** 20, 128), 100, 12,
                                    args) == 128
    # 300 MiB per view against the flat 2 GiB CPU budget: 6 views.
    assert pf._effective_view_batch(_stub_body(300 * 2 ** 20, 128), 100, 12,
                                    args) == 6

    # Cone's torch charge is the params-derived max(num_slices, num_rows),
    # whatever band the call requests: a 2-column band must still charge the
    # full width, and the spread between the two charges is what proves the
    # override is the one consulted.
    cone = _cone_model()
    cone_pf = cone.projector_functions
    cols = cone._transient_cols(2)
    assert cols == max(int(cone.get_params('recon_shape')[2]),
                       int(cone.get_params('sinogram_shape')[1])) > 2
    num_pixels = 10 ** 7
    expected = budget // (num_pixels * cols * 4)
    naive_band_charge = budget // (num_pixels * 2 * 4)
    assert expected != naive_band_charge
    assert cone_pf._effective_view_batch(
        _torch_stub(), num_pixels, 2, cone._view_batch_args()) == expected


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
