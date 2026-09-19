"""The mbirtorch-side slice_viewer wrapper.

Handing the viewer a torch tensor on any device is what happens right after a
reconstruction, so the wrapper must convert it and open.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

import mbirtorch


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


@pytest.fixture
def open_viewer():
    created = []

    def _open(*datasets, **kwargs):
        kwargs.setdefault('block', False)
        with pytest.warns(UserWarning, match='non-interactive'):
            viewer = mbirtorch.slice_viewer(*datasets, **kwargs)
        created.append(viewer)
        return viewer

    yield _open
    import mbirtorch.viewers.slice_figure as viewer_module
    for viewer in created:
        plt.close(viewer.fig)
        if viewer in viewer_module._NONBLOCKING_VIEWERS:
            viewer_module._NONBLOCKING_VIEWERS.remove(viewer)
class TestTensorShim:
    def test_device_tensor_converted(self, open_viewer):
        import torch

        # A CPU tensor is converted to the equivalent numpy array.
        cpu_volume = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        viewer = open_viewer(cpu_volume)
        np.testing.assert_array_equal(viewer.stack.original_data[0],
                                      cpu_volume.numpy())

        # A numpy array is kept as is and None becomes the placeholder volume.
        array_volume = make_volume((4, 4, 4))
        viewer = open_viewer(array_volume, None)
        np.testing.assert_array_equal(viewer.stack.original_data[0],
                                      array_volume)
        assert viewer.stack.original_data[1].shape == (20, 20, 20)

        # A tensor on an accelerator is moved to the host first.
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            pytest.skip('no accelerator available')
        volume = torch.arange(60, dtype=torch.float32,
                              device=device).reshape(3, 4, 5)
        viewer = open_viewer(volume)
        np.testing.assert_array_equal(viewer.stack.original_data[0],
                                      volume.cpu().numpy())
