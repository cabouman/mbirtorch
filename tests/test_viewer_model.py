"""Headless unit tests for the slice viewer's pure-numpy model (VolumeStack).

These are the first viewer tests in either repo.  They cover the stage-1
checklist from the build spec: input normalization, perm round-trips,
proportional slice mapping across unequal depths, difference shape/perm
validation and restore, ROI statistics, range resolution, and the
npy/npz/h5 load branches including the 4D case.  No matplotlib is touched.
"""

import numpy as np
import pytest

from mbirtorch.viewer import VolumeStack, PLACEHOLDER_SHAPE


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_2d_promoted_to_3d(self):
        stack = VolumeStack([np.ones((4, 5))])
        assert stack.original_data[0].shape == (4, 5, 1)
        assert stack.data[0].shape == (4, 5, 1)

    def test_none_becomes_placeholder(self):
        stack = VolumeStack([None])
        assert stack.original_data[0].shape == PLACEHOLDER_SHAPE
        assert np.all(stack.original_data[0] == 0)

    @pytest.mark.parametrize("bad", [np.ones(3), np.ones((2, 3, 4, 5))])
    def test_wrong_ndim_raises(self, bad):
        with pytest.raises(ValueError, match="2D or 3D"):
            VolumeStack([bad])

    def test_empty_datasets_raises(self):
        with pytest.raises(ValueError, match="At least one dataset"):
            VolumeStack([])

    def test_scalar_slice_axis_broadcast(self):
        stack = VolumeStack([make_volume((3, 4, 5))] * 2, slice_axis=0)
        assert stack.axes_perms == [[1, 2, 0], [1, 2, 0]]
        assert stack.data[0].shape == (4, 5, 3)

    def test_default_slice_axis_is_2(self):
        stack = VolumeStack([make_volume((3, 4, 5))])
        assert stack.axes_perms == [[0, 1, 2]]

    def test_per_volume_slice_axis(self):
        stack = VolumeStack([make_volume((3, 4, 5))] * 2, slice_axis=[0, 2])
        assert stack.axes_perms == [[1, 2, 0], [0, 1, 2]]

    def test_slice_axis_wrong_length_raises(self):
        with pytest.raises(ValueError, match="slice_axis"):
            VolumeStack([make_volume((3, 4, 5))] * 2, slice_axis=[0, 1, 2])

    def test_bad_slice_axis_raises(self):
        with pytest.raises(ValueError, match="slice_axis must be 0, 1, or 2"):
            VolumeStack([make_volume((3, 4, 5))], slice_axis=3)

    def test_label_broadcast_and_default(self):
        assert VolumeStack([None, None]).labels == ["Slice", "Slice"]
        assert VolumeStack([None, None], slice_label="View").labels == ["View", "View"]
        assert VolumeStack([None, None], slice_label=["a", "b"]).labels == ["a", "b"]

    def test_label_wrong_length_raises(self):
        with pytest.raises(ValueError, match="slice_label"):
            VolumeStack([None, None], slice_label=["a"])

    def test_data_dicts_single_dict_single_volume(self):
        stack = VolumeStack([None], data_dicts={"notes": "x"})
        assert stack.data_dicts == [{"notes": "x"}]

    def test_data_dicts_single_dict_two_volumes_raises(self):
        with pytest.raises(ValueError, match="data_dicts"):
            VolumeStack([None, None], data_dicts={"notes": "x"})

    def test_data_dicts_list_with_none_entries(self):
        stack = VolumeStack([None, None], data_dicts=[None, {"a": "1"}])
        assert stack.data_dicts == [None, {"a": "1"}]

    def test_data_dicts_bad_entry_raises(self):
        with pytest.raises(ValueError, match="data_dicts"):
            VolumeStack([None, None], data_dicts=[None, "not a dict"])

    def test_initial_slices_are_midpoints(self):
        stack = VolumeStack([make_volume((4, 4, 10)), make_volume((4, 4, 7))])
        assert stack.cur_slices == [5, 3]
        assert stack.master_index == 5


# ---------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------

class TestPerms:
    def test_perm_from_slice_axis(self):
        assert VolumeStack.perm_from_slice_axis(0) == [1, 2, 0]
        assert VolumeStack.perm_from_slice_axis(1) == [0, 2, 1]
        assert VolumeStack.perm_from_slice_axis(2) == [0, 1, 2]

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

    def test_set_perm_unchanged_returns_false(self):
        stack = VolumeStack([make_volume((3, 4, 5))])
        assert stack.set_perm(0, 2) is False
        assert stack.set_perm(0, [0, 1, 2]) is False

    def test_set_perm_invalid_raises(self):
        stack = VolumeStack([make_volume((3, 4, 5))])
        with pytest.raises(ValueError, match="[Pp]ermutation"):
            stack.set_perm(0, [0, 0, 2])

    def test_transpose_swaps_in_plane_axes(self):
        volume = make_volume((3, 4, 5))
        stack = VolumeStack([volume])
        stack.transpose(0)
        assert stack.axes_perms[0] == [1, 0, 2]
        np.testing.assert_array_equal(stack.data[0],
                                      np.transpose(volume, (1, 0, 2)))
        stack.transpose(0)
        assert stack.axes_perms[0] == [0, 1, 2]
        np.testing.assert_array_equal(stack.data[0], volume)

    def test_set_perm_on_difference_volume_keeps_difference(self):
        a, b = make_volume((4, 4, 6), 1), make_volume((4, 4, 6), 2)
        stack = VolumeStack([a, b])
        stack.apply_difference(0, 1)
        stack.set_perm(0, [1, 0, 2])
        expected = np.transpose(b - a, (1, 0, 2))
        np.testing.assert_allclose(stack.data[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Proportional master slice mapping
# ---------------------------------------------------------------------------

class TestSliceMapping:
    def test_deepest_volume_follows_master_exactly(self):
        stack = VolumeStack([make_volume((4, 4, 100)), make_volume((4, 4, 10))])
        for master in [0, 33, 50, 99]:
            stack.set_master_index(master)
            assert stack.cur_slices[0] == master

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

    def test_master_clipped_to_range(self):
        stack = VolumeStack([make_volume((4, 4, 10))])
        stack.set_master_index(-5)
        assert stack.master_index == 0
        stack.set_master_index(1000)
        assert stack.master_index == 9

    def test_changed_list_skips_unmoved_volumes(self):
        stack = VolumeStack([make_volume((4, 4, 100)), make_volume((4, 4, 10))])
        stack.set_master_index(50)
        # 50 -> 51 moves the deep volume by one but leaves the shallow one
        # at round(51/99*9) == round(50/99*9) == 5.
        changed = stack.set_master_index(51)
        assert changed == [0]
        assert stack.set_master_index(51) == []

    def test_single_slice_volume_stays_at_zero(self):
        stack = VolumeStack([make_volume((4, 4, 10)), np.ones((4, 4))])
        for master in [0, 5, 9]:
            stack.set_master_index(master)
            assert stack.cur_slices[1] == 0

    def test_all_single_slice_no_division_error(self):
        stack = VolumeStack([np.ones((4, 4)), np.ones((5, 5))])
        assert stack.max_slices == 1
        assert stack.set_master_index(0) == []
        assert stack.master_fraction == 0.0

    def test_axis_change_preserves_master_fraction(self):
        stack = VolumeStack([make_volume((4, 4, 100)), make_volume((100, 80, 60))])
        stack.set_master_index(30)
        before = stack.master_fraction
        stack.set_perm(1, 0)  # volume 1 depth becomes 100
        assert stack.master_fraction == pytest.approx(before, abs=0.5 / 99)
        assert stack.cur_slices[0] == stack.master_index
        assert stack.cur_slices[1] == round(stack.master_fraction * 99)

    def test_slice_image_tracks_master(self):
        volume = make_volume((4, 4, 10))
        stack = VolumeStack([volume])
        stack.set_master_index(7)
        np.testing.assert_array_equal(stack.slice_image(0), volume[:, :, 7])


# ---------------------------------------------------------------------------
# Intensity range
# ---------------------------------------------------------------------------

class TestRange:
    def test_defaults_span_all_volumes(self):
        a = np.zeros((3, 3, 3)); a[0, 0, 0] = -2.0
        b = np.zeros((3, 3, 3)); b[0, 0, 0] = 5.0
        stack = VolumeStack([a, b])
        assert (stack.vmin, stack.vmax) == (-2.0, 5.0)

    def test_explicit_values_respected(self):
        stack = VolumeStack([make_volume((3, 3, 3))], vmin=-1.0, vmax=1.0)
        assert (stack.vmin, stack.vmax) == (-1.0, 1.0)

    def test_equal_bounds_split_by_epsilon(self):
        stack = VolumeStack([np.full((3, 3, 3), 7.0)])
        assert stack.vmin < 7.0 < stack.vmax

    def test_equal_bounds_split_scales_with_magnitude(self):
        stack = VolumeStack([np.full((3, 3, 3), 1e12)])
        assert stack.vmax > 1e12
        assert stack.vmax - stack.vmin >= 2e-6 * 1e12 * 0.9

    def test_min_above_max_raises(self):
        stack = VolumeStack([make_volume((3, 3, 3))])
        with pytest.raises(ValueError, match="Minimum must be less than maximum"):
            stack.set_range(2.0, 1.0)

    def test_none_bounds_fill_from_data(self):
        a = np.zeros((3, 3, 3)); a[0, 0, 0] = 4.0
        stack = VolumeStack([a], vmin=-1.0, vmax=1.0)
        vmin, vmax = stack.set_range(None, None)
        assert (vmin, vmax) == (0.0, 4.0)
        vmin, vmax = stack.set_range(None, 10.0)
        assert (vmin, vmax) == (0.0, 10.0)


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

    def test_abs_difference(self):
        a, b = make_volume((4, 5, 6), 1), make_volume((4, 5, 6), 2)
        stack = VolumeStack([a, b])
        stack.apply_difference(0, 1, use_abs=True)
        np.testing.assert_allclose(stack.data[0], np.abs(b - a), rtol=1e-6)
        assert stack.labels[0].startswith("abs(Image 1 minus current): ")

    def test_shape_mismatch_rejected(self):
        stack = VolumeStack([make_volume((4, 5, 6)), make_volume((4, 5, 7))])
        assert not stack.can_difference(0, 1)
        with pytest.raises(ValueError, match="same shape"):
            stack.apply_difference(0, 1)

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

    def test_different_slice_axes_reorient_comparison(self):
        a, b = make_volume((4, 5, 6), 1), make_volume((4, 5, 6), 2)
        stack = VolumeStack([a, b], slice_axis=[0, 2])
        assert stack.can_difference(0, 1)
        stack.apply_difference(0, 1)
        expected = np.transpose(b, (1, 2, 0)) - np.transpose(a, (1, 2, 0))
        np.testing.assert_allclose(stack.data[0], expected, rtol=1e-6)

    def test_same_index_rejected(self):
        stack = VolumeStack([make_volume((4, 5, 6))] * 2)
        assert not stack.can_difference(1, 1)

    def test_matching_transposed_pair_allowed(self):
        stack = VolumeStack([make_volume((4, 5, 6), 1), make_volume((4, 5, 6), 2)])
        stack.set_perm(0, 0)
        stack.set_perm(1, 0)
        assert stack.can_difference(0, 1)
        stack.apply_difference(0, 1)
        assert stack.data[0].shape == (5, 6, 4)


# ---------------------------------------------------------------------------
# ROI statistics
# ---------------------------------------------------------------------------

class TestRoiStats:
    def test_uniform_region(self):
        volume = np.full((10, 10, 3), 2.0)
        stack = VolumeStack([volume])
        stats = stack.roi_stats(0, x=5.0, y=5.0, radius=2.0)
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["min"] == stats["max"] == pytest.approx(2.0)

    def test_mask_matches_manual_circle(self):
        volume = np.arange(100, dtype=float).reshape(10, 10)[..., None]
        stack = VolumeStack([volume])
        x, y, r = 4.0, 3.0, 2.5
        stats = stack.roi_stats(0, x, y, r)
        yv, xv = np.mgrid[:10, :10]
        values = volume[:, :, 0][(xv - x) ** 2 + (yv - y) ** 2 <= r ** 2]
        assert stats["mean"] == pytest.approx(values.mean())
        assert stats["std"] == pytest.approx(values.std())

    def test_off_image_circle_returns_none(self):
        stack = VolumeStack([np.ones((10, 10, 3))])
        assert stack.roi_stats(0, x=100.0, y=100.0, radius=2.0) is None

    def test_stats_follow_current_slice(self):
        volume = np.zeros((8, 8, 4))
        volume[:, :, 3] = 9.0
        stack = VolumeStack([volume])
        stack.set_master_index(0)
        assert stack.roi_stats(0, 4, 4, 2)["mean"] == pytest.approx(0.0)
        stack.set_master_index(3)
        assert stack.roi_stats(0, 4, 4, 2)["mean"] == pytest.approx(9.0)

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
    def test_npy_round_trip(self, tmp_path):
        volume = make_volume((4, 5, 6))
        path = str(tmp_path / "vol.npy")
        np.save(path, volume)
        assert VolumeStack.list_file_arrays(path) is None
        array, data_dict = VolumeStack.read_file_array(path)
        np.testing.assert_array_equal(array, volume)
        assert data_dict is None

    def test_npz_list_and_read_by_name(self, tmp_path):
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

    def test_unsupported_extension_raises(self, tmp_path):
        path = str(tmp_path / "vol.txt")
        with pytest.raises(ValueError, match="Unsupported file type"):
            VolumeStack.list_file_arrays(path)
        with pytest.raises(ValueError, match="Unsupported file type"):
            VolumeStack.read_file_array(path)

    def test_load_3d_replaces_volume(self):
        stack = VolumeStack([make_volume((4, 4, 4)), make_volume((4, 4, 4))])
        new = make_volume((6, 7, 8), 9)
        replaced = stack.load_array(1, new, data_dict={"src": "file"})
        assert replaced == [1]
        np.testing.assert_array_equal(stack.original_data[1], new)
        assert stack.data_dicts[1] == {"src": "file"}
        assert stack.cur_slices[1] == 4  # middle of depth 8
        assert stack.axes_perms[1] == [0, 1, 2]

    def test_load_2d_promotes(self):
        stack = VolumeStack([make_volume((4, 4, 4))])
        stack.load_array(0, np.ones((5, 6)))
        assert stack.original_data[0].shape == (5, 6, 1)

    def test_load_resets_transposed_perm_to_canonical(self):
        stack = VolumeStack([make_volume((4, 4, 4))], slice_axis=0)
        stack.transpose(0)
        assert stack.axes_perms[0] == [2, 1, 0]
        stack.load_array(0, make_volume((5, 6, 7)))
        assert stack.axes_perms[0] == [1, 2, 0]  # canonical for slice axis 0

    def test_load_4d_fills_volumes(self):
        stack = VolumeStack([make_volume((4, 4, 4))] * 2)
        four_d = make_volume((5, 6, 7, 3), 4)
        replaced = stack.load_array(0, four_d, data_dict={"k": "v"})
        assert replaced == [0, 1]  # capped at n_volumes
        np.testing.assert_array_equal(stack.original_data[0], four_d[..., 0])
        np.testing.assert_array_equal(stack.original_data[1], four_d[..., 1])
        assert stack.data_dicts[0] == {"k": "v"}

    def test_load_4d_primary_index_capped(self):
        stack = VolumeStack([make_volume((4, 4, 4))] * 3)
        untouched = stack.original_data[2].copy()
        four_d = make_volume((5, 6, 7, 2), 4)
        replaced = stack.load_array(2, four_d, data_dict={"k": "v"})
        assert replaced == [0, 1]
        # The primary index falls back to the last loaded volume, and the
        # volume beyond the 4D slab count keeps its data.
        assert stack.data_dicts[1] == {"k": "v"}
        np.testing.assert_array_equal(stack.original_data[2], untouched)

    def test_load_5d_raises(self):
        stack = VolumeStack([make_volume((4, 4, 4))])
        with pytest.raises(ValueError, match="2D, 3D, or 4D"):
            stack.load_array(0, np.ones((2, 2, 2, 2, 2)))

    def test_load_clears_difference_state(self):
        a, b = make_volume((4, 4, 4), 1), make_volume((4, 4, 4), 2)
        stack = VolumeStack([a, b], slice_label=["A", "B"])
        stack.apply_difference(0, 1)
        stack.load_array(0, make_volume((5, 5, 5)))
        assert not stack.is_difference(0)
        assert stack.labels[0] == "A"

    def test_load_updates_master_range(self):
        stack = VolumeStack([make_volume((4, 4, 4))])
        stack.load_array(0, make_volume((4, 4, 100)))
        assert stack.max_slices == 100
        assert stack.cur_slices[0] == 50
        changed = stack.set_master_index(99)
        assert stack.cur_slices[0] == 99 and changed == [0]
