"""The 4D demo data: a moving phantom and the sinogram of one scan of it."""
import numpy as np
import pytest

import mbirtorch
from mbirtorch._phantoms_4d import _gen_moving_phantom, _rack_and_pinion


def test_rack_and_pinion_moves_and_keeps_its_mass():
    """The phantom changes over time, its two parts keep their values, and its mass stays
    within a few percent, since the parts only move.  With the defaults the wheel turns one
    tooth over the scan, so the end state equals the start state and the middle differs."""
    shape = (32, 32, 16)
    start, middle, end = (_rack_and_pinion(shape, t) for t in (0.0, 0.5, 1.0))
    assert start.shape == shape and start.dtype == np.float32
    assert set(np.unique(start)) == {0.0, np.float32(0.7), 1.0}
    assert np.any(start != middle)
    assert np.array_equal(start, end)
    assert abs(start.sum() - middle.sum()) < 0.05 * start.sum()


def test_gen_moving_phantom_rejects_unknown_names():
    with pytest.raises(ValueError, match='object_type'):
        _gen_moving_phantom('no-such-object', (8, 8, 4), 2)


def test_demo_data_4d_projects_each_step_from_its_own_object():
    """The sinogram has one view per angle, and the views of a step equal the projection of
    that step's object, so the scan is of a moving object."""
    num_views, num_steps = 24, 4
    phantom_4d, sinogram, params = mbirtorch.gen_demo_data_4d(
        num_views=num_views, num_rotations=1, num_det_rows=8, num_det_channels=32,
        num_steps=num_steps)
    assert phantom_4d.shape == (num_steps, 32, 32, 8)
    assert sinogram.shape == (num_views, 8, 32) and sinogram.dtype == np.float32
    assert params['angles'].shape == (num_views,)
    assert np.array_equal(params['step_of_view'], np.repeat(np.arange(num_steps), num_views // num_steps))

    ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, params['angles'])
    ct_model.set_params(verbose=0)
    for step in (0, num_steps - 1):
        views = params['step_of_view'] == step
        expected = ct_model.forward_project(phantom_4d[step])[views]
        assert np.allclose(sinogram[views], expected, atol=1e-5)
    # The scan is of a moving object: the middle views are not the first object's projection.
    middle = params['step_of_view'] == num_steps // 2
    assert not np.allclose(sinogram[middle], ct_model.forward_project(phantom_4d[0])[middle], atol=1e-3)
