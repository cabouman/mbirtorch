"""Gates for the demo-data generators."""

import numpy as np

import mbirtorch


def test_generate_demo_data_is_finite_float32_for_every_model_type():
    """Each generator must return a host numpy float32 phantom and a finite,
    non-empty sinogram of the shape its geometry implies.  The cube phantom
    itself must be torch float32: mbirjax's jnp.array downcasts to float32,
    while torch.as_tensor keeps numpy's float64, which doubles the memory and
    mps cannot hold at all."""
    import torch
    assert mbirtorch.gen_cube_phantom((8, 8, 4)).dtype == torch.float32

    phantom, sino, params = mbirtorch.generate_demo_data(
        model_type='cone', object_type='cube', num_views=12,
        num_det_rows=24, num_det_channels=32)
    # Host numpy float32 for either object type, as the docstring promises.
    assert isinstance(phantom, np.ndarray) and phantom.dtype == np.float32
    assert phantom.max() > 0 and np.isfinite(np.asarray(sino)).all()

    phantom, sino, params = mbirtorch.generate_demo_data(
        model_type='translation', object_type='cube',
        num_det_rows=24, num_det_channels=32)
    assert isinstance(phantom, np.ndarray) and phantom.dtype == np.float32
    assert sino.shape[0] == params['translation_vectors'].shape[0]
    assert phantom.max() > 0 and np.isfinite(np.asarray(sino)).all()
    assert np.asarray(sino).max() > 0

    phantom, sino, params = mbirtorch.generate_demo_data(
        model_type='multiaxis', elevation_degrees=25.0, object_type='cube',
        num_views=8, num_det_rows=16, num_det_channels=24)
    assert sino.shape == (8, 16, 24)
    assert params['angles'].shape == (8, 2)
    assert np.allclose(params['angles'][:, 1], np.deg2rad(25.0))
    assert np.isfinite(np.asarray(sino)).all() and np.asarray(sino).max() > 0
