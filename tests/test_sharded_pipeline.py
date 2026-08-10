"""Gate for the multi-device scan preprocessing driver: splitting the views
across devices must produce exactly the single-device answer.  Identical is
the right bar: the kernel is per-view, so no sum crosses a device boundary.
"""

import numpy as np
import torch

from mbirtorch.preprocess import pipeline


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
