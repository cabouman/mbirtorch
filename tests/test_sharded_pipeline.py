"""Gate for the multi-device scan preprocessing driver: splitting the views
across devices must produce exactly the single-device answer.  Identical is
the right bar: the kernel is per-view, so no sum crosses a device boundary.

Also gates the one device-count decision preprocessing makes for itself --
scan_to_sino's ``devices=None`` default -- against the process-wide pin.
"""

import numpy as np
import torch

from mbirtorch.preprocess import pipeline
from mbirtorch.preprocess import utilities as preprocess_utilities


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
