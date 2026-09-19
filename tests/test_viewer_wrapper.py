"""Stage-3 tests: the mbirtorch-side slice_viewer wrapper.

Covers the lazy __init__ export (headless import must not load matplotlib),
the torch-tensor shim, the data-dict-to-strings conversion, and passthrough
of the viewer-object return, keep-alive registry, and save_fn injection.
"""

import subprocess
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

import mbirtorch
from mbirtorch.view_utils import convert_subdicts_to_strings


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


class TestLazyExport:
    def test_headless_import_is_silent_and_lazy(self):
        # -W error turns any warning into a failure, and the module listing
        # proves matplotlib and the viewer modules were not imported.
        # geometry_rules is not a viewer module: it holds the projection rules
        # the four model classes share, so it loads with them.
        code = (
            "import sys, os; import mbirtorch; "
            "loaded = [m for m in sys.modules "
            "if m.startswith('matplotlib') or 'viewer' in m "
            "or 'view_utils' in m "
            "or ('geometry_' in m and 'geometry_rules' not in m)]; "
            "print(','.join(loaded) or 'CLEAN')"
        )
        result = subprocess.run(
            [sys.executable, '-W', 'error', '-c', code],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'CLEAN'


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


class TestDictConversion:
    def test_data_dicts_converted_to_strings(self, open_viewer):
        recon_dict = {
            'recon_params': {'num_iterations': 5, 'fm_rmse': [1.0, 0.5]},
            'model_params': {'sharpness': 1.0, 'recon_shape': (4, 4, 4)},
            'notes': 'plain string',
            'count': 7,
        }
        converted = convert_subdicts_to_strings(recon_dict)
        assert all(isinstance(v, str) for v in converted.values())
        assert 'num_iterations' in converted['recon_params']
        assert converted['notes'] == 'plain string'
        assert converted['count'] == '7'

        # Non-dict inputs come back unchanged.
        assert convert_subdicts_to_strings(None) is None
        assert convert_subdicts_to_strings('text') == 'text'

        # The wrapper applies the conversion to a list of dicts.
        viewer = open_viewer(make_volume((4, 4, 4)), make_volume((4, 4, 4)),
                             data_dicts=[{'recon_params':
                                          {'num_iterations': 3}}, None])
        stored = viewer.stack.data_dicts[0]
        assert isinstance(stored['recon_params'], str)
        assert 'num_iterations' in stored['recon_params']
        assert viewer.stack.data_dicts[1] is None

        # And to a single dict given for a single volume.
        viewer = open_viewer(make_volume((4, 4, 4)),
                             data_dicts={'model_params': {'sharpness': 0.0}})
        assert isinstance(viewer.stack.data_dicts[0]['model_params'], str)
