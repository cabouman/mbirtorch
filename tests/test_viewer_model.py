"""Headless tests for the slice viewer's pure-numpy model (VolumeStack).

Two things are checked: that the array shapes a user hands the viewer are
normalized into 3D volumes with the right slice axis, and that the
region-of-interest statistics match a hand-written circular mask.  No
matplotlib is touched.
"""

import numpy as np
import pytest

from mbirtorch.viewers.slice_figure import VolumeStack, PLACEHOLDER_SHAPE


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_inputs_become_3d_arrays_with_slice_axis_perms(self):
        # 2D input gains a trailing slice axis.
        stack = VolumeStack([np.ones((4, 5))])
        assert stack.original_data[0].shape == (4, 5, 1)
        assert stack.data[0].shape == (4, 5, 1)

        # None becomes a zero placeholder volume.
        stack = VolumeStack([None])
        assert stack.original_data[0].shape == PLACEHOLDER_SHAPE
        assert np.all(stack.original_data[0] == 0)

        # A scalar slice_axis applies to every volume.
        stack = VolumeStack([make_volume((3, 4, 5))] * 2, slice_axis=0)
        assert stack.axes_perms == [[1, 2, 0], [1, 2, 0]]
        assert stack.data[0].shape == (4, 5, 3)

        # A per-volume slice_axis list gives each volume its own perm.
        stack = VolumeStack([make_volume((3, 4, 5))] * 2, slice_axis=[0, 2])
        assert stack.axes_perms == [[1, 2, 0], [0, 1, 2]]
# ---------------------------------------------------------------------------
# ROI statistics
# ---------------------------------------------------------------------------

class TestRoiStats:
    def test_mask_matches_manual_circle(self):
        volume = np.arange(100, dtype=float).reshape(10, 10)[..., None]
        stack = VolumeStack([volume])
        x, y, r = 4.0, 3.0, 2.5
        stats = stack.roi_stats(0, x, y, r)
        yv, xv = np.mgrid[:10, :10]
        values = volume[:, :, 0][(xv - x) ** 2 + (yv - y) ** 2 <= r ** 2]
        assert stats["mean"] == pytest.approx(values.mean())
        assert stats["std"] == pytest.approx(values.std())

        # A uniform region gives that value with zero spread.
        uniform = VolumeStack([np.full((10, 10, 3), 2.0)])
        stats = uniform.roi_stats(0, x=5.0, y=5.0, radius=2.0)
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["min"] == stats["max"] == pytest.approx(2.0)

        # Statistics come from the slice currently displayed.
        varying = np.zeros((8, 8, 4))
        varying[:, :, 3] = 9.0
        slice_stack = VolumeStack([varying])
        slice_stack.set_master_index(0)
        assert slice_stack.roi_stats(0, 4, 4, 2)["mean"] == pytest.approx(0.0)
        slice_stack.set_master_index(3)
        assert slice_stack.roi_stats(0, 4, 4, 2)["mean"] == pytest.approx(9.0)
