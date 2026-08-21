"""Input and output forms for the streaming preprocessing entries.

These entries take a whole array on the host and batch it onto a device
themselves, so the divided device form (a ``Shards`` container, one tensor per
device) is not something they can work with.  Each one refuses it at entry with
a message that names the function, instead of failing partway through the
batching loop on a missing attribute.

The rest of the contract is the ordinary preprocessing one: NumPy in, NumPy
out, with a torch tensor also accepted for the array that gets batched.  Any
use of a GPU is internal.

Everything here runs on the CPU.  A Shards needs no real GPU -- two 'virtual'
CPU devices build one -- and the calls pass ``devices=['cpu']`` so the values
do not depend on what accelerator the host happens to have.  Which devices an
entry picks by default is a separate question, covered in
tests/test_sharded_pipeline.py.
"""

import numpy as np
import pytest
import torch

import mbirtorch.preprocess as mtp
from mbirtorch import _sharding

# Small enough that every call below is a fraction of a second, with more than
# one batch per call so the batching loop actually runs.
NUM_VIEWS, NUM_ROWS, NUM_COLS = 4, 6, 8
BATCH_ARGS = dict(batch_size=2, devices=['cpu'])


def _obj_scan():
    """An object scan whose values are all in (0, 1), so that dividing by a
    blank scan of ones gives a positive ratio and -log gives a finite result
    with no defective pixels to fill."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.2, 0.9, (NUM_VIEWS, NUM_ROWS, NUM_COLS)).astype(np.float32)


def _blank_and_dark():
    """A blank scan of ones and a dark scan of zeros, so the transmission step
    reduces to -log(obj_scan) and the expected values are easy to state."""
    return (np.ones((1, NUM_ROWS, NUM_COLS), dtype=np.float32),
            np.zeros((1, NUM_ROWS, NUM_COLS), dtype=np.float32))


def _sinogram():
    """A finite, positive sinogram.  The zinger threshold is estimated from the
    sinogram's own support, so the values have to be usable, not just finite."""
    rng = np.random.default_rng(1)
    return rng.uniform(0.1, 1.0, (NUM_VIEWS, NUM_ROWS, NUM_COLS)).astype(np.float32)


def _divided(array):
    """The same data in the divided device form: two contiguous blocks of views,
    one per device, held as a Shards.  Two 'virtual' CPU devices stand in for
    two GPUs -- the container is the same either way, and that is all the entry
    guards look at."""
    placement = _sharding.Placement(['cpu', 'cpu'], axis=0, axis_len=array.shape[0])
    tensors = [torch.as_tensor(array[start:end])
               for _, (start, end) in placement.shard_ranges()]
    return _sharding.Shards(tensors, placement)


def _refusal_calls():
    """One call per guarded entry, each with its batched array in the divided
    form.  Returned as (function name, no-argument callable) pairs so the test
    can check that the message names the function the caller actually called."""
    blank, dark = _blank_and_dark()
    divided_obj = _divided(_obj_scan())
    divided_sino = _divided(_sinogram())
    return [
        ('scan_to_sino',
         lambda: mtp.scan_to_sino(divided_obj, blank, dark, **BATCH_ARGS)),
        ('compute_sino_transmission',
         lambda: mtp.compute_sino_transmission(divided_obj, blank, dark, **BATCH_ARGS)),
        ('correct_det_rotation',
         lambda: mtp.correct_det_rotation(divided_sino, det_rotation=0.02,
                                          batch_size=2, devices=['cpu'])),
        ('downsample_view_data',
         lambda: mtp.downsample_view_data(divided_obj, blank, dark, (2, 2), **BATCH_ARGS)),
        ('correct_zinger_pixels',
         lambda: mtp.correct_zinger_pixels(divided_sino, **BATCH_ARGS)),
        ('BH_correction',
         lambda: mtp.BH_correction(divided_sino, [1.0, 0.2], **BATCH_ARGS)),
    ]


@pytest.mark.parametrize('function_name, call', _refusal_calls(),
                         ids=[name for name, _ in _refusal_calls()])
def test_entry_refuses_divided_form(function_name, call):
    """Every streaming entry refuses a divided array, and says which function
    was called, which argument was wrong, and how to fix it.  Without the check
    the call dies inside the batching loop with an AttributeError about a
    missing 'shape', which says none of those things."""
    with pytest.raises(TypeError) as refusal:
        call()
    message = str(refusal.value)
    assert function_name in message
    assert 'divided device form' in message
    assert 'shards.gather()' in message


def test_refusal_names_the_offending_argument():
    """The guard looks at every array argument, not just the first one, and the
    message names the ones that were passed in the divided form."""
    obj = _obj_scan()
    blank, dark = _blank_and_dark()
    with pytest.raises(TypeError, match='blank_scan'):
        mtp.scan_to_sino(obj, _divided(blank), dark, **BATCH_ARGS)


def test_scan_to_sino_numpy_in_numpy_out():
    obj = _obj_scan()
    blank, dark = _blank_and_dark()
    sino = mtp.scan_to_sino(obj, blank, dark, **BATCH_ARGS)
    assert isinstance(sino, np.ndarray)
    assert sino.shape == (NUM_VIEWS, NUM_ROWS, NUM_COLS)
    # No downsampling and no rotation, and a blank of ones over a dark of zeros,
    # so the whole thing reduces to -log(obj_scan).
    assert np.allclose(sino, -np.log(obj), atol=1e-5)


def test_scan_to_sino_tensor_in_numpy_out():
    """A torch tensor is accepted for the scan that gets batched, and the result
    still comes back as NumPy on the host -- the preprocessing functions return
    host arrays whatever they were handed."""
    obj = _obj_scan()
    blank, dark = _blank_and_dark()
    sino = mtp.scan_to_sino(torch.as_tensor(obj), blank, dark, **BATCH_ARGS)
    assert isinstance(sino, np.ndarray)
    assert np.allclose(sino, -np.log(obj), atol=1e-5)


def test_compute_sino_transmission_numpy_in_numpy_out():
    obj = _obj_scan()
    blank, dark = _blank_and_dark()
    sino = mtp.compute_sino_transmission(obj, blank, dark, **BATCH_ARGS)
    assert isinstance(sino, np.ndarray)
    assert np.allclose(sino, -np.log(obj), atol=1e-5)


def test_tensor_blank_and_dark_scans_match_numpy():
    """Torch tensors are accepted for the blank and dark scans too.  They are
    brought to the host and reduced there, so the result equals the all-numpy
    call exactly."""
    obj = _obj_scan()
    blank, dark = _blank_and_dark()
    blank_t, dark_t = torch.as_tensor(blank), torch.as_tensor(dark)
    expected = mtp.scan_to_sino(obj, blank, dark, **BATCH_ARGS)
    sino = mtp.scan_to_sino(obj, blank_t, dark_t, **BATCH_ARGS)
    assert isinstance(sino, np.ndarray)
    assert np.array_equal(sino, expected)
    expected = mtp.compute_sino_transmission(obj, blank, dark, **BATCH_ARGS)
    sino = mtp.compute_sino_transmission(obj, blank_t, dark_t, **BATCH_ARGS)
    assert isinstance(sino, np.ndarray)
    assert np.array_equal(sino, expected)


def test_correct_det_rotation_numpy_in_numpy_out():
    sino = _sinogram()
    # A zero rotation samples each pixel at its own location, so the values are
    # unchanged and the shape is easy to state.
    out = mtp.correct_det_rotation(sino, det_rotation=0.0, batch_size=2, devices=['cpu'])
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, sino, atol=1e-6)


def test_downsample_view_data_numpy_in_numpy_out():
    obj = _obj_scan()
    blank, dark = _blank_and_dark()
    out_obj, out_blank, out_dark, defective = mtp.downsample_view_data(
        obj, blank, dark, (2, 2), **BATCH_ARGS)
    for out in (out_obj, out_blank, out_dark):
        assert isinstance(out, np.ndarray)
    assert out_obj.shape == (NUM_VIEWS, NUM_ROWS // 2, NUM_COLS // 2)
    assert out_blank.shape == (1, NUM_ROWS // 2, NUM_COLS // 2)
    # Nothing was marked defective, so the updated list comes back empty.
    assert len(defective) == 0


def test_correct_zinger_pixels_numpy_in_numpy_out():
    sino = _sinogram()
    out = mtp.correct_zinger_pixels(sino, **BATCH_ARGS)
    assert isinstance(out, np.ndarray)
    assert out.shape == sino.shape
    # A zinger is a value below a negative threshold, and this sinogram is
    # entirely positive, so nothing is replaced.
    assert np.allclose(out, sino, atol=1e-6)


def test_bh_correction_numpy_in_numpy_out():
    sino = _sinogram()
    out = mtp.BH_correction(sino, [1.0, 0.2], **BATCH_ARGS)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, sino + 0.2 * sino ** 2, atol=1e-5)
