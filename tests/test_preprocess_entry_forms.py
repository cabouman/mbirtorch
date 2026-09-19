"""Input and output forms for the streaming preprocessing entries.

These entries take a whole array on the host and batch it onto a device
themselves.  The contract is the ordinary preprocessing one: NumPy in, NumPy
out, with a torch tensor also accepted for the arrays that get batched.  Any
use of a GPU is internal.

Everything here runs on the CPU: the calls pass ``devices=['cpu']`` so the
values do not depend on what accelerator the host happens to have.  Which
devices an entry picks by default is a separate question, covered in
tests/test_sharded_pipeline.py.
"""

import numpy as np
import mbirtorch.preprocess as mtp

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


def test_scan_to_sino_numpy_in_numpy_out():
    obj = _obj_scan()
    blank, dark = _blank_and_dark()
    sino = mtp.scan_to_sino(obj, blank, dark, **BATCH_ARGS)
    assert isinstance(sino, np.ndarray)
    assert sino.shape == (NUM_VIEWS, NUM_ROWS, NUM_COLS)
    # No downsampling and no rotation, and a blank of ones over a dark of zeros,
    # so the whole thing reduces to -log(obj_scan).
    assert np.allclose(sino, -np.log(obj), atol=1e-5)
