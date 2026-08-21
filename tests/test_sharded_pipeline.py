"""Gate for the multi-device scan preprocessing driver: splitting the views
across devices must produce exactly the single-device answer.  Identical is
the right bar: the kernel is per-view, so no sum crosses a device boundary.

Also gates the device-count default that the preprocessing functions share.
``pipeline.permitted_devices`` resolves ``devices=None`` for all of them, so
its rule is tested directly and then again through each function.
"""

import numpy as np
import pytest
import torch

import mbirtorch.preprocess as mtp
from mbirtorch.preprocess import pipeline
from mbirtorch.preprocess import utilities as preprocess_utilities


@pytest.fixture
def unpinned(monkeypatch):
    """Clear the suite's device-count pin.

    The conftest fixture pins every test to one device, which is what keeps
    the suite deterministic on a multi-GPU host.  A test of the multi-device
    default has to opt out of that pin, and doing so explicitly keeps the
    pin's reach visible.
    """
    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES', raising=False)


def test_multi_device_view_split_matches_single():
    rng = np.random.default_rng(7)
    scans = rng.uniform(0.5, 2.0, size=(23, 12, 16)).astype(np.float32)
    gain = rng.uniform(1.0, 3.0, size=(1, 12, 16)).astype(np.float32)

    def kernel(batch):
        g = torch.as_tensor(gain, device=batch.device)
        return -torch.log(batch / g)

    ref = pipeline.map_view_batches(scans, kernel, batch_size=4,
                                    devices=['cpu'])
    out = pipeline.map_view_batches(scans, kernel, batch_size=4,
                                    devices=['cpu', 'cpu', 'cpu'])
    assert out.shape == ref.shape
    assert np.array_equal(out, ref)


# ── the shared devices=None default ──────────────────────────────────────────
def test_default_devices_are_one_device_without_cuda(monkeypatch, unpinned):
    """A machine with no visible CUDA device gets its default device.

    That default differs by machine: mps on a Mac, cpu on a host with no
    accelerator.  The resolver stands in for it here, so the test asserts
    the count and the source of the choice rather than a device name.
    """
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(pipeline, '_resolve_device',
                        lambda spec: torch.device('cpu'))
    assert pipeline.permitted_devices() == [torch.device('cpu')]


def test_default_devices_are_every_visible_cuda_device(monkeypatch, unpinned):
    """The devices are faked, so this runs on any machine."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    assert pipeline.permitted_devices() == ['cuda:0', 'cuda:1', 'cuda:2']


def test_the_pin_caps_the_default(monkeypatch):
    """MBIRTORCH_NUM_DEVICES caps the default count."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)

    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '2')
    assert pipeline.permitted_devices() == ['cuda:0', 'cuda:1']

    # A pin above the visible count cannot conjure devices.
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '8')
    assert pipeline.permitted_devices() == ['cuda:0', 'cuda:1', 'cuda:2']


def test_an_explicit_list_is_never_capped(monkeypatch):
    """An explicit list is the caller's, so the pin does not apply to it."""
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '1')
    assert pipeline.permitted_devices(['cpu', 'cpu', 'cpu']) == ['cpu'] * 3


def test_scan_to_sino_default_devices_honor_the_pin(monkeypatch):
    """MBIRTORCH_NUM_DEVICES caps scan_to_sino's 'all visible CUDA devices'
    default.  The pin is how the suite and the nightlies keep results off the
    host's device count; an entry that ignored it would quietly opt out.

    The devices are faked, so this runs on any machine: the kernel never runs
    (map_view_batches is replaced by a recorder), only the device choice does.
    """
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)

    seen = {}

    def record_devices(array, kernel, batch_size, devices=None):
        seen['devices'] = devices
        return np.asarray(array)

    monkeypatch.setattr(preprocess_utilities.pipeline, 'map_view_batches',
                        record_devices)

    obj = np.ones((3, 2, 2), dtype=np.float32)
    blank = 2 * np.ones((1, 2, 2), dtype=np.float32)
    dark = np.zeros((1, 2, 2), dtype=np.float32)

    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES', raising=False)
    preprocess_utilities.scan_to_sino(obj, blank, dark)
    assert seen['devices'] == ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '1')
    preprocess_utilities.scan_to_sino(obj, blank, dark)
    assert seen['devices'] == ['cuda:0']

    # A pin above the visible count cannot conjure devices.
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '8')
    preprocess_utilities.scan_to_sino(obj, blank, dark)
    assert seen['devices'] == ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    # An explicit list is the caller's and is never capped.
    preprocess_utilities.scan_to_sino(obj, blank, dark, devices=['cpu', 'cpu'])
    assert seen['devices'] == ['cpu', 'cpu']


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


@pytest.mark.parametrize('function_name', list(RUNNERS))
def test_the_default_reaches_the_driver(function_name, monkeypatch, unpinned):
    """Each function resolves devices=None to all the permitted devices.

    The devices are faked and the driver is replaced by a recorder, so no
    kernel runs here.  The device choice is the whole test.
    """
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)

    seen = {}

    def record_devices(array, kernel, batch_size, desc=None, devices=None):
        seen['devices'] = devices
        return np.asarray(array)

    monkeypatch.setattr(pipeline, 'map_view_batches', record_devices)

    RUNNERS[function_name](None)
    assert seen['devices'] == ['cuda:0', 'cuda:1', 'cuda:2']
