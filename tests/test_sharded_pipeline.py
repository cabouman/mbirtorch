"""Gate for the multi-device scan preprocessing driver: splitting the views
across devices must produce exactly the single-device answer.  Identical is
the right bar: the kernel is per-view, so no sum crosses a device boundary.
"""

import numpy as np
import pytest
import mbirtorch.preprocess as mtp
# ── the five view-batched preprocessing functions ────────────────────────────
# Each of these kernels is per-view.  No sum crosses a view boundary, so a
# view's result cannot depend on which device or which batch it landed in.
# The bar for agreement across device counts is therefore exact equality, not
# a tolerance.  The view count below is prime, so no device count divides it.

NUM_VIEWS = 23
DETECTOR_SHAPE = (12, 16)


def _scans():
    """An object scan with its blank and dark scans and its defective pixels."""
    rng = np.random.default_rng(11)
    obj = rng.uniform(0.5, 2.0, size=(NUM_VIEWS,) + DETECTOR_SHAPE).astype(np.float32)
    blank = rng.uniform(2.0, 3.0, size=(2,) + DETECTOR_SHAPE).astype(np.float32)
    dark = rng.uniform(0.0, 0.05, size=(2,) + DETECTOR_SHAPE).astype(np.float32)
    defective = np.array([[3, 4], [7, 9]], dtype=np.int64)
    return obj, blank, dark, defective


def _sinogram():
    """An object region on a near-zero background, growing along the view axis.

    Three kinds of negative pixel are planted for the zinger correction.  The
    large ones at -8 are zingers under any threshold.  The small ones at
    -0.15 and -0.25 sit on either side of the whole-sinogram threshold of
    about -0.23.  Growth along the view axis puts the per-shard thresholds at
    about -0.13, -0.23, and -0.32.  A threshold estimated after the views
    were split would therefore classify the small pixels differently on
    different shards, and the equality test below would report that.
    """
    rng = np.random.default_rng(5)
    sino = rng.uniform(0.0, 0.02, size=(NUM_VIEWS,) + DETECTOR_SHAPE).astype(np.float32)
    sino[:, 2:10, 3:13] = rng.uniform(0.4, 1.0, size=(NUM_VIEWS, 8, 10)).astype(np.float32)
    ramp = np.linspace(1.0, 5.0, NUM_VIEWS, dtype=np.float32)
    sino = (sino * ramp[:, None, None]).astype(np.float32)
    for view, row, col in ((2, 5, 6), (11, 3, 12), (20, 8, 2)):
        sino[view, row, col] = -8.0
    for view, row, col in ((4, 6, 7), (13, 4, 5), (18, 7, 9)):
        sino[view, row, col] = -0.15
    for view, row, col in ((6, 2, 8), (15, 9, 3), (21, 5, 11)):
        sino[view, row, col] = -0.25
    return sino


def _run_transmission(devices):
    obj, blank, dark, defective = _scans()
    return mtp.compute_sino_transmission(obj, blank, dark, defective_pixel_array=defective,
                                         batch_size=4, devices=devices)


def _run_rotation(devices):
    return mtp.correct_det_rotation(_sinogram(), det_rotation=0.05, batch_size=4,
                                    devices=devices)


def _run_downsample(devices):
    obj, blank, dark, defective = _scans()
    return mtp.downsample_view_data(obj, blank, dark, (2, 2), defective_pixel_array=defective,
                                    batch_size=4, devices=devices)[0]


def _run_zinger(devices):
    return mtp.correct_zinger_pixels(_sinogram(), zinger_pixel_ratio=0.1, num_passes=3,
                                     batch_size=4, devices=devices)


def _run_bh(devices):
    return mtp.BH_correction(_sinogram(), [1.0, 0.2, 0.1], batch_size=4, devices=devices)


RUNNERS = {'compute_sino_transmission': _run_transmission,
           'correct_det_rotation': _run_rotation,
           'downsample_view_data': _run_downsample,
           'correct_zinger_pixels': _run_zinger,
           'BH_correction': _run_bh}


@pytest.mark.parametrize('num_devices', [2, 3])
@pytest.mark.parametrize('function_name', list(RUNNERS))
def test_results_do_not_depend_on_the_device_count(function_name, num_devices):
    """Splitting the views over more devices returns the same array exactly."""
    run = RUNNERS[function_name]
    one_device = run(['cpu'])
    several_devices = run(['cpu'] * num_devices)
    assert np.array_equal(several_devices, one_device)
