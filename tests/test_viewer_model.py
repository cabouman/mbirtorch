"""Headless unit tests for the slice viewer's pure-numpy model (VolumeStack).

These are the first viewer tests in either repo.  They cover the stage-1
checklist from the build spec: input normalization, perm round-trips,
proportional slice mapping across unequal depths, difference shape/perm
validation and restore, ROI statistics, range resolution, and the
npy/npz/h5 load branches including the 4D case.  No matplotlib is touched.
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
# Permutations
# ---------------------------------------------------------------------------

class TestPerms:
    def test_set_perm_round_trip(self):
        volume = make_volume((3, 4, 5))
        stack = VolumeStack([volume])
        assert stack.set_perm(0, 0) is True
        assert stack.data[0].shape == (4, 5, 3)
        np.testing.assert_array_equal(stack.data[0],
                                      np.transpose(volume, (1, 2, 0)))
        assert stack.set_perm(0, 2) is True
        np.testing.assert_array_equal(stack.data[0], volume)
        np.testing.assert_array_equal(stack.original_data[0], volume)

        # transpose swaps the two in-plane axes and undoes itself.
        stack.transpose(0)
        assert stack.axes_perms[0] == [1, 0, 2]
        np.testing.assert_array_equal(stack.data[0],
                                      np.transpose(volume, (1, 0, 2)))
        stack.transpose(0)
        assert stack.axes_perms[0] == [0, 1, 2]
        np.testing.assert_array_equal(stack.data[0], volume)

        # A difference volume stays a difference after a perm change.
        a, b = make_volume((4, 4, 6), 1), make_volume((4, 4, 6), 2)
        diff_stack = VolumeStack([a, b])
        diff_stack.apply_difference(0, 1)
        diff_stack.set_perm(0, [1, 0, 2])
        expected = np.transpose(b - a, (1, 0, 2))
        np.testing.assert_allclose(diff_stack.data[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Proportional master slice mapping
# ---------------------------------------------------------------------------

class TestSliceMapping:
    def test_shallow_volume_maps_proportionally_not_clipped(self):
        # The mbirjax _update_slice identity arithmetic would pin the shallow
        # volume at its last slice for any master above its depth; the
        # proportional map keeps it mid-range.
        stack = VolumeStack([make_volume((4, 4, 100)), make_volume((4, 4, 10))])
        stack.set_master_index(50)
        assert stack.cur_slices[1] == round(50 / 99 * 9)  # 5, not 9
        stack.set_master_index(99)
        assert stack.cur_slices[1] == 9
        stack.set_master_index(0)
        assert stack.cur_slices[1] == 0

        # The deepest volume follows the master index exactly.
        for master in [0, 33, 50, 99]:
            stack.set_master_index(master)
            assert stack.cur_slices[0] == master

        # The master index is clipped to the available range.
        clip_stack = VolumeStack([make_volume((4, 4, 10))])
        clip_stack.set_master_index(-5)
        assert clip_stack.master_index == 0
        clip_stack.set_master_index(1000)
        assert clip_stack.master_index == 9

        # A single-slice volume stays at slice 0 for every master index.
        thin_stack = VolumeStack([make_volume((4, 4, 10)), np.ones((4, 4))])
        for master in [0, 5, 9]:
            thin_stack.set_master_index(master)
            assert thin_stack.cur_slices[1] == 0

        # Changing a volume's slice axis preserves the master fraction.
        axis_stack = VolumeStack([make_volume((4, 4, 100)),
                                  make_volume((100, 80, 60))])
        axis_stack.set_master_index(30)
        before = axis_stack.master_fraction
        axis_stack.set_perm(1, 0)  # volume 1 depth becomes 100
        assert axis_stack.master_fraction == pytest.approx(before, abs=0.5 / 99)
        assert axis_stack.cur_slices[0] == axis_stack.master_index
        assert axis_stack.cur_slices[1] == round(axis_stack.master_fraction * 99)

        # slice_image returns the plane at the current index.
        volume = make_volume((4, 4, 10))
        image_stack = VolumeStack([volume])
        image_stack.set_master_index(7)
        np.testing.assert_array_equal(image_stack.slice_image(0),
                                      volume[:, :, 7])

    def test_all_single_slice_no_division_error(self):
        stack = VolumeStack([np.ones((4, 4)), np.ones((5, 5))])
        assert stack.max_slices == 1
        assert stack.set_master_index(0) == []
        assert stack.master_fraction == 0.0


# ---------------------------------------------------------------------------
# Intensity range
# ---------------------------------------------------------------------------

class TestRange:
    def test_range_defaults_explicit_values_and_none_bounds(self):
        # The default range spans every volume.
        a = np.zeros((3, 3, 3)); a[0, 0, 0] = -2.0
        b = np.zeros((3, 3, 3)); b[0, 0, 0] = 5.0
        stack = VolumeStack([a, b])
        assert (stack.vmin, stack.vmax) == (-2.0, 5.0)

        # Explicit bounds are used as given.
        stack = VolumeStack([make_volume((3, 3, 3))], vmin=-1.0, vmax=1.0)
        assert (stack.vmin, stack.vmax) == (-1.0, 1.0)

        # A None bound is filled in from the data.
        c = np.zeros((3, 3, 3)); c[0, 0, 0] = 4.0
        stack = VolumeStack([c], vmin=-1.0, vmax=1.0)
        vmin, vmax = stack.set_range(None, None)
        assert (vmin, vmax) == (0.0, 4.0)
        vmin, vmax = stack.set_range(None, 10.0)
        assert (vmin, vmax) == (0.0, 10.0)

    def test_equal_bounds_split_by_epsilon(self):
        stack = VolumeStack([np.full((3, 3, 3), 7.0)])
        assert stack.vmin < 7.0 < stack.vmax

        # The split scales with the data magnitude.
        big = VolumeStack([np.full((3, 3, 3), 1e12)])
        assert big.vmax > 1e12
        assert big.vmax - big.vmin >= 2e-6 * 1e12 * 0.9


# ---------------------------------------------------------------------------
# Difference images
# ---------------------------------------------------------------------------

class TestDifference:
    def test_apply_and_restore(self):
        a, b = make_volume((4, 5, 6), 1), make_volume((4, 5, 6), 2)
        stack = VolumeStack([a, b], slice_label=["A", "B"])
        stack.apply_difference(0, 1)
        np.testing.assert_allclose(stack.data[0], b - a, rtol=1e-6)
        assert stack.labels[0] == "Image 1 minus current: A"
        assert stack.is_difference(0)
        np.testing.assert_array_equal(stack.original_data[0], a)

        stack.restore(0)
        np.testing.assert_array_equal(stack.data[0], a)
        assert stack.labels[0] == "A"
        assert not stack.is_difference(0)

        # The absolute-difference option takes the magnitude.
        abs_stack = VolumeStack([a, b])
        abs_stack.apply_difference(0, 1, use_abs=True)
        np.testing.assert_allclose(abs_stack.data[0], np.abs(b - a), rtol=1e-6)
        assert abs_stack.labels[0].startswith("abs(Image 1 minus current): ")

    def test_transposed_baseline_reorients_comparison(self):
        # A transposed panel can still take a difference: the comparison is
        # re-oriented into the baseline's frame instead of being refused.
        a, b = make_volume((4, 5, 6), 1), make_volume((4, 5, 6), 2)
        stack = VolumeStack([a, b])
        stack.transpose(0)
        assert stack.can_difference(0, 1)
        stack.apply_difference(0, 1)
        expected = np.transpose(b, (1, 0, 2)) - np.transpose(a, (1, 0, 2))
        np.testing.assert_allclose(stack.data[0], expected, rtol=1e-6)

        # Volumes with different slice axes are re-oriented the same way.
        axes_stack = VolumeStack([a, b], slice_axis=[0, 2])
        assert axes_stack.can_difference(0, 1)
        axes_stack.apply_difference(0, 1)
        expected = np.transpose(b, (1, 2, 0)) - np.transpose(a, (1, 2, 0))
        np.testing.assert_allclose(axes_stack.data[0], expected, rtol=1e-6)

        # A pair transposed the same way differences in the shared frame.
        pair_stack = VolumeStack([a, b])
        pair_stack.set_perm(0, 0)
        pair_stack.set_perm(1, 0)
        assert pair_stack.can_difference(0, 1)
        pair_stack.apply_difference(0, 1)
        assert pair_stack.data[0].shape == (5, 6, 4)


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

    def test_x_is_column_y_is_row(self):
        volume = np.zeros((10, 10, 1))
        volume[2, 7, 0] = 5.0  # row 2, column 7
        stack = VolumeStack([volume])
        stats = stack.roi_stats(0, x=7.0, y=2.0, radius=0.5)
        assert stats["mean"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# File load
# ---------------------------------------------------------------------------

class TestFileLoad:
    def test_npy_and_npz_read(self, tmp_path):
        # .npy holds a single unnamed array.
        volume = make_volume((4, 5, 6))
        path = str(tmp_path / "vol.npy")
        np.save(path, volume)
        assert VolumeStack.list_file_arrays(path) is None
        array, data_dict = VolumeStack.read_file_array(path)
        np.testing.assert_array_equal(array, volume)
        assert data_dict is None

        # .npz lists its arrays by name and reads the one asked for.
        a, b = make_volume((3, 3, 3), 1), make_volume((4, 4, 4), 2)
        path = str(tmp_path / "arrays.npz")
        np.savez(path, first=a, second=b)
        names, shapes = VolumeStack.list_file_arrays(path)
        assert set(names) == {"first", "second"}
        assert shapes[names.index("second")] == (4, 4, 4)
        array, data_dict = VolumeStack.read_file_array(path, "second")
        np.testing.assert_array_equal(array, b)
        assert data_dict is None

    def test_h5_list_read_and_attrs(self, tmp_path):
        import h5py
        volume = make_volume((4, 5, 6))
        path = str(tmp_path / "vol.h5")
        with h5py.File(path, "w") as f:
            dataset = f.create_dataset("volume", data=volume)
            dataset.attrs["notes"] = "test notes"
            dataset.attrs["recon_params"] = "sharpness: 1.0"
        names, shapes = VolumeStack.list_file_arrays(path)
        assert names == ["volume"]
        assert shapes == [(4, 5, 6)]
        array, data_dict = VolumeStack.read_file_array(path, "volume")
        np.testing.assert_allclose(array, volume)
        assert data_dict == {"notes": "test notes",
                             "recon_params": "sharpness: 1.0"}

    def test_load_3d_replaces_volume(self):
        stack = VolumeStack([make_volume((4, 4, 4)), make_volume((4, 4, 4))])
        new = make_volume((6, 7, 8), 9)
        replaced = stack.load_array(1, new, data_dict={"src": "file"})
        assert replaced == [1]
        np.testing.assert_array_equal(stack.original_data[1], new)
        assert stack.data_dicts[1] == {"src": "file"}
        assert stack.cur_slices[1] == 4  # middle of depth 8
        assert stack.axes_perms[1] == [0, 1, 2]

        # Loading over a difference volume restores its label and state.
        a, b = make_volume((4, 4, 4), 1), make_volume((4, 4, 4), 2)
        diff_stack = VolumeStack([a, b], slice_label=["A", "B"])
        diff_stack.apply_difference(0, 1)
        diff_stack.load_array(0, make_volume((5, 5, 5)))
        assert not diff_stack.is_difference(0)
        assert diff_stack.labels[0] == "A"

        # Loading a deeper volume extends the master slice range.
        range_stack = VolumeStack([make_volume((4, 4, 4))])
        range_stack.load_array(0, make_volume((4, 4, 100)))
        assert range_stack.max_slices == 100
        assert range_stack.cur_slices[0] == 50
        changed = range_stack.set_master_index(99)
        assert range_stack.cur_slices[0] == 99 and changed == [0]

    def test_load_4d_fills_volumes(self):
        stack = VolumeStack([make_volume((4, 4, 4))] * 2)
        four_d = make_volume((5, 6, 7, 3), 4)
        replaced = stack.load_array(0, four_d, data_dict={"k": "v"})
        assert replaced == [0, 1]  # capped at n_volumes
        np.testing.assert_array_equal(stack.original_data[0], four_d[..., 0])
        np.testing.assert_array_equal(stack.original_data[1], four_d[..., 1])
        assert stack.data_dicts[0] == {"k": "v"}

        # With fewer slabs than volumes, the primary index falls back to the
        # last loaded volume and volumes beyond the slab count keep their data.
        capped = VolumeStack([make_volume((4, 4, 4))] * 3)
        untouched = capped.original_data[2].copy()
        four_d = make_volume((5, 6, 7, 2), 4)
        replaced = capped.load_array(2, four_d, data_dict={"k": "v"})
        assert replaced == [0, 1]
        assert capped.data_dicts[1] == {"k": "v"}
        np.testing.assert_array_equal(capped.original_data[2], untouched)
