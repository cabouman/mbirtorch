"""Phantom gate: the golden match against mbirjax.

The golden test compares two frameworks.  Both compute in float32, and a voxel's
value is the sum of the coefficients of the ellipsoids that contain it, so two
voxels with the same memberships get identical float32 values.  float32 rounding
is hardware dependent, and each ellipsoid is a <= 1 threshold on a float
quadratic, so a voxel within rounding distance of a boundary can land inside on
one framework and outside on the other.  That gate therefore allows a small
fraction of flipped voxels and requires exact agreement on every other voxel.
"""

import glob
import os

import numpy as np
import pytest

import mbirtorch
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
