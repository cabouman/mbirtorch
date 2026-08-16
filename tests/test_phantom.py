"""Phantom gates: the golden match against mbirjax, and the banded and blocked
builds against a build that takes every slice and every row at once.

The golden test compares two frameworks.  Both compute in float32, and a voxel's
value is the sum of the coefficients of the ellipsoids that contain it, so two
voxels with the same memberships get identical float32 values.  float32 rounding
is hardware dependent, and each ellipsoid is a <= 1 threshold on a float
quadratic, so a voxel within rounding distance of a boundary can land inside on
one framework and outside on the other.  That gate therefore allows a small
fraction of flipped voxels and requires exact agreement on every other voxel.

The band and block tests compare mbirtorch against itself, where the bar is
exact equality.  Every voxel of the phantom depends only on its own
coordinates, and each axis has one coordinate vector that the bands and the
blocks slice, so splitting the slices across devices or the rows into blocks
changes which array holds a value, never the value itself.  Virtual cpu devices
stand in for real ones, as elsewhere in the suite.
"""

import glob
import os

import numpy as np
import pytest

import mbirtorch
from mbirtorch import _sharding
from mbirtorch import utilities
from mbirtorch.preprocess import pipeline

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "golden_*.npz")))

# The nine Shepp-Logan ellipsoid coefficients are 1, -0.8, -0.2, -0.2 and five
# of 0.1 (see utilities._add_shepp_logan_ellipsoids).  A voxel that changes its
# membership of one ellipsoid moves by one of those coefficients, so 0.1 is the
# smallest step a flip can make and 1.0 is the largest.
SMALLEST_COEFFICIENT = 0.1
LARGEST_COEFFICIENT = 1.0


@pytest.mark.goldens
@pytest.mark.skipif(
    not _paths, reason="no goldens: run tests/generate_goldens.py in the mbirjax env")
def test_phantom_matches_golden():
    """The phantom against mbirjax's: exact except for a few membership flips.

    Both builds accumulate the same float32 coefficients in the same order, so
    a voxel inside the same set of ellipsoids in both builds has the same
    float32 value in both.  A difference therefore means the voxel is inside a
    different set, and each such flip moves the voxel by one coefficient.  The
    three assertions below check how many voxels flipped, how far a flip moved
    a voxel, and that no other voxel differs at all.
    """
    golden = np.load(_paths[0])
    recon_shape = tuple(int(x) for x in golden["recon_shape"])
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    ref = golden["phantom"]
    abs_diff = np.abs(phantom.astype(np.float64) - ref.astype(np.float64))
    # A flip moves a voxel by at least the smallest coefficient, so half of that
    # separates a flip from any other difference.
    flip = abs_diff >= 0.5 * SMALLEST_COEFFICIENT
    frac_flip = float(np.mean(flip))
    print(f"phantom differing voxels: {int(flip.sum())} / {flip.size} "
          f"({frac_flip:.2e})")

    # The budget is 1e-4.  Flips have been measured at about 1e-6 of the voxels
    # at 2048 cubed, and at 1.5e-7 or less between two backends and between the
    # two frameworks at every shape up to 320 cubed.  The budget is a hundred
    # times the largest of those, which leaves room for a machine that rounds
    # differently.  It is still far below what a real error produces.  A
    # coordinate axis off by one voxel would move a whole shell of voxels around
    # each ellipsoid, which is a percent-scale fraction of the volume.
    assert frac_flip <= 1e-4
    # A flip changes one ellipsoid's contribution, so it cannot move a voxel by
    # more than the largest coefficient.  The slack covers the rounding of the
    # sum, which differs by one term rather than by exactly that term.
    assert float(abs_diff.max()) <= LARGEST_COEFFICIENT + 1e-5
    # Every voxel that did not flip is equal bit for bit.  Equal memberships
    # give equal float32 values, so this needs no tolerance.
    assert np.array_equal(phantom[~flip], ref[~flip])


# ── the slice-banded build ───────────────────────────────────────────────────
@pytest.mark.parametrize("target_max_attenuation", [None, 3.0])
def test_slice_bands_equal_the_single_device_build(target_max_attenuation):
    """Three devices over ten slices, a count that does not divide the axis.

    The bands are then uneven, which is the only form the build has.  They cover
    the slice axis exactly, so there is no padded tail to crop.  The test
    asserts that split first, so that the equality below is known to have been
    measured on uneven bands.
    """
    phantom_shape = (12, 14, 10)
    devices = ['cpu'] * 3
    ranges = [r for _, r in _sharding.Placement(
        devices, axis=-1, axis_len=phantom_shape[2]).shard_ranges()]
    assert ranges == [(0, 4), (4, 7), (7, 10)]
    assert sum(stop - start for start, stop in ranges) == phantom_shape[2]

    single = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        phantom_shape, devices=['cpu'],
        target_max_attenuation=target_max_attenuation)
    banded = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        phantom_shape, devices=devices,
        target_max_attenuation=target_max_attenuation)

    # A host numpy float32 array either way: the phantom is a reference object.
    assert isinstance(banded, np.ndarray) and banded.dtype == np.float32
    assert banded.shape == phantom_shape
    assert np.array_equal(banded, single)


def test_a_device_count_above_the_slice_axis_leaves_an_empty_band():
    """Five devices over three slices, so two devices get no slices at all.

    An empty band is a legal band, and the build must produce the same phantom
    with two of them present as without.
    """
    phantom_shape = (8, 9, 3)
    devices = ['cpu'] * 5
    ranges = [r for _, r in _sharding.Placement(
        devices, axis=-1, axis_len=phantom_shape[2]).shard_ranges()]
    assert ranges == [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)]

    single = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        phantom_shape, devices=['cpu'])
    banded = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        phantom_shape, devices=devices)
    assert banded.shape == phantom_shape
    assert np.array_equal(banded, single)


# ── the row-blocked build ────────────────────────────────────────────────────
def test_row_blocking_does_not_change_the_phantom():
    """One row per block against every row in one block.

    ``max_block_gb`` sets the block size, so the two calls differ only in how
    many blocks the rows are built in.  The block counts are asserted first, so
    that the equality below is known to compare a blocked build against an
    unblocked one.
    """
    phantom_shape = (17, 13, 11)
    # A single-device band covers every slice, so the band shape is the phantom
    # shape here.
    assert utilities._phantom_block_rows(phantom_shape, 4.0) == phantom_shape[0]
    assert utilities._phantom_block_rows(phantom_shape, 1e-9) == 1

    unblocked = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        phantom_shape, devices=['cpu'], max_block_gb=4.0)
    blocked = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        phantom_shape, devices=['cpu'], max_block_gb=1e-9)
    assert np.array_equal(blocked, unblocked)


# ── the devices= default ─────────────────────────────────────────────────────
def test_the_default_device_list_comes_from_permitted_devices(monkeypatch):
    """devices=None is resolved by the shared preprocessing helper.

    The helper's own rule is gated in tests/test_sharded_pipeline.py, so this
    checks only the wiring: that None reaches it rather than being resolved
    here.
    """
    seen = []

    def record(devices=None):
        seen.append(devices)
        return ['cpu']

    monkeypatch.setattr(pipeline, 'permitted_devices', record)
    mbirtorch.generate_3d_shepp_logan_low_dynamic_range((6, 6, 4))
    assert seen == [None]
