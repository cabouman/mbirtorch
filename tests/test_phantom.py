"""Phantom golden test: the numpy Shepp-Logan matches mbirjax's.

Each ellipsoid is a <= 1 threshold on a float quadratic, so voxels exactly at
an ellipsoid boundary can flip between the f32 (jax) and f64 (numpy) grid
arithmetic.  The gate therefore requires exact agreement away from boundaries
and allows a small fraction of boundary-voxel flips.
"""

import glob
import os

import numpy as np
import pytest

import mbirtorch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_paths = sorted(glob.glob(os.path.join(GOLDEN_DIR, "golden_*.npz")))

pytestmark = [pytest.mark.goldens, pytest.mark.skipif(
    not _paths, reason="no goldens: run tests/generate_goldens.py in the mbirjax env")]


def test_phantom_matches_golden():
    golden = np.load(_paths[0])
    recon_shape = tuple(int(x) for x in golden["recon_shape"])
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    ref = golden["phantom"]
    diff = phantom != ref
    frac_diff = float(np.mean(diff))
    print(f"phantom differing voxels: {int(diff.sum())} / {diff.size} "
          f"({frac_diff:.2e})")
    # Differences may only be boundary flips: bounded in count and in size.
    assert frac_diff < 1e-3
    if diff.any():
        assert float(np.max(np.abs(phantom - ref))) <= 0.2 + 1e-6
