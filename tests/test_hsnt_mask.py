"""Tests of the ``sample_mask`` and ``check_mask`` options of ``dehydrate``."""
import warnings

import numpy as np
import pytest

import mbirtorch.hsnt as hsnt
from test_hsnt_solvers import _sphere_problem

N = 48
CENTERS = ((0.38, 0.38), (0.38, 0.62), (0.6, 0.5))      # the sphere centers of _sphere_problem
RADIUS2 = (0.22 * N) ** 2 * 1.1                         # a little larger than the spheres


@pytest.fixture(scope="module")
def spheres():
    """The three-sphere transmission image and a mask that covers the spheres."""
    T = _sphere_problem("cpu", n=N, dose=100.0).numpy().reshape(N, N, -1)
    yy, xx = np.mgrid[:N, :N] + 0.5
    inside = np.zeros((N, N), dtype=bool)
    for cy, cx in CENTERS:
        inside |= (yy - cy * N) ** 2 + (xx - cx * N) ** 2 < RADIUS2
    return T, inside, yy


def _dehydrate(T, mask, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        W, H = hsnt.dehydrate(T, "transmission", 3, sample_mask=mask, verbose=0, **kwargs)
    return W, H, [str(w.message) for w in caught]


def test_the_maps_are_zero_outside_the_mask_and_fit_inside_it(spheres):
    T, inside, _ = spheres
    W, H, messages = _dehydrate(T, inside, check_mask=True)
    assert W.shape == (N, N, 3) and np.isfinite(W).all()
    assert not W[~inside].any()
    assert not any("outside the sample_mask" in m for m in messages)
    W0, H0, _ = _dehydrate(T, None)
    err_masked = np.abs(np.exp(-(W @ H)) - T)[inside].mean()
    err_full = np.abs(np.exp(-(W0 @ H0)) - T)[inside].mean()
    assert err_masked < 1.1 * err_full


def test_check_mask_warns_when_the_mask_cuts_the_sample(spheres):
    T, inside, yy = spheres
    cut = inside & (yy < 0.55 * N)                      # removes most of the third sphere
    _, _, messages = _dehydrate(T, cut, check_mask=True)
    assert any("outside the sample_mask" in m for m in messages)


def test_a_mask_per_view_broadcasts_to_the_views(spheres):
    T, inside, _ = spheres
    data = np.stack([T, np.roll(T, 3, axis=1)])
    mask = np.stack([inside, np.roll(inside, 3, axis=1)])
    W, _, _ = _dehydrate(data, mask)
    assert W.shape == (2, N, N, 3)
    assert not W[~mask].any()
    assert W[mask].any()


def test_bad_masks_are_rejected(spheres):
    T, inside, _ = spheres
    with pytest.raises(ValueError, match="broadcast"):
        hsnt.dehydrate(T, "transmission", 3, sample_mask=inside[:-1], verbose=0)
    with pytest.raises(ValueError, match="False everywhere"):
        hsnt.dehydrate(T, "transmission", 3, sample_mask=np.zeros((N, N), dtype=bool), verbose=0)
    with pytest.raises(ValueError, match="boolean"):
        hsnt.dehydrate(T, "transmission", 3, sample_mask=inside.astype(np.float32) * 2, verbose=0)
