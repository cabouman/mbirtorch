"""Gates for stitch_arrays.

stitch_arrays blends a fixed overlap between adjacent arrays and returns the
result where the inputs already live: NumPy in gives NumPy out, tensors give a
tensor on their own device.  Two inputs it cannot serve are refused with a
message that names the problem, rather than being quietly relocated or failing
on a missing attribute deep inside the function: tensors spread over more than
one device, and an array in the divided device form (a Shards container).

The refusal for the divided form and the normal-path values are checked on the
CPU, so they run everywhere.  The mixed-device refusal needs two real GPUs and
is skipped otherwise.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _sharding

requires_two_cuda = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="the mixed-device refusal needs at least two CUDA devices")


def _halves():
    """Two small arrays to stitch along the last axis, with distinct values."""
    first = np.arange(2 * 2 * 5.0).reshape(2, 2, 5).astype(np.float32)
    second = (np.arange(2 * 2 * 6.0).reshape(2, 2, 6) + 100.0).astype(np.float32)
    return first, second


# The stitched values for _halves() with overlap=3, recorded from the function
# itself.  With overlap 3 the ramp covers one element, so each output row keeps
# the first two elements of the first array, blends the third pair at weight
# 1/2, and then continues with the tail of the second array.  Pinning the
# numbers (they are all exact in float32) catches a change in the blend, not
# just a change in the shape.
EXPECTED_OVERLAP_3 = np.array(
    [[[0.0, 1.0, 2.0, 52.0, 102.0, 103.0, 104.0, 105.0],
      [5.0, 6.0, 7.0, 57.5, 108.0, 109.0, 110.0, 111.0]],
     [[10.0, 11.0, 12.0, 63.0, 114.0, 115.0, 116.0, 117.0],
      [15.0, 16.0, 17.0, 68.5, 120.0, 121.0, 122.0, 123.0]]],
    dtype=np.float32)


def test_stitch_arrays_numpy_values():
    first, second = _halves()
    out = mbirtorch.stitch_arrays([first, second], overlap=3, axis=2)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert np.array_equal(out, EXPECTED_OVERLAP_3)


def test_stitch_arrays_one_device_matches_numpy():
    """A tensor list, and a list that mixes a tensor with a NumPy array, both
    stay valid and give the same values as the all-NumPy call.  The mixed case
    is the one the device check must not break: only one device is named, so
    the NumPy array simply joins it."""
    first, second = _halves()
    tensors = mbirtorch.stitch_arrays([torch.as_tensor(first), torch.as_tensor(second)],
                                      overlap=3, axis=2)
    mixed = mbirtorch.stitch_arrays([torch.as_tensor(first), second], overlap=3, axis=2)
    for out in (tensors, mixed):
        assert isinstance(out, torch.Tensor) and out.device.type == 'cpu'
        assert np.array_equal(out.numpy(), EXPECTED_OVERLAP_3)


def test_stitch_arrays_refuses_divided_form():
    """A Shards holds one tensor per device, so it has no shape of its own.
    Two CPU shards are enough to build one; no GPU is involved."""
    first, second = _halves()
    placement = _sharding.Placement(['cpu', 'cpu'], axis=-1, axis_len=4)
    shards = _sharding.Shards([torch.as_tensor(first[..., :2]),
                               torch.as_tensor(first[..., 2:4])], placement)
    with pytest.raises(TypeError, match="divided device form"):
        mbirtorch.stitch_arrays([shards, second], overlap=3, axis=2)


@requires_two_cuda
def test_stitch_arrays_refuses_mixed_devices():
    first, second = _halves()
    with pytest.raises(ValueError, match="one device"):
        mbirtorch.stitch_arrays([torch.as_tensor(first).to('cuda:0'),
                                 torch.as_tensor(second).to('cuda:1')],
                                overlap=3, axis=2)
