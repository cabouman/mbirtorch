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
    import mbirtorch.viewer as viewer_module
    for viewer in created:
        plt.close(viewer.fig)
        if viewer in viewer_module._NONBLOCKING_VIEWERS:
            viewer_module._NONBLOCKING_VIEWERS.remove(viewer)


class TestLazyExport:
    def test_viewer_names_in_public_api(self):
        for name in ('slice_viewer', 'SliceViewer', 'VolumeStack'):
            assert name in mbirtorch.__all__

    def test_headless_import_is_silent_and_lazy(self):
        # -W error turns any warning into a failure, and the module listing
        # proves matplotlib and the viewer modules were not imported.
        code = (
            "import sys, os; import mbirtorch; "
            "loaded = [m for m in sys.modules "
            "if m.startswith('matplotlib') or 'viewer' in m "
            "or 'view_utils' in m]; "
            "print(','.join(loaded) or 'CLEAN')"
        )
        result = subprocess.run(
            [sys.executable, '-W', 'error', '-c', code],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'CLEAN'

    def test_attribute_access_resolves_and_caches(self):
        assert callable(mbirtorch.slice_viewer)
        assert mbirtorch.SliceViewer is not None
        assert mbirtorch.VolumeStack is not None
        # After first access the names are plain module attributes.
        assert 'slice_viewer' in vars(mbirtorch)

    def test_unknown_attribute_still_raises(self):
        with pytest.raises(AttributeError, match='no attribute'):
            mbirtorch.not_a_real_name


class TestTensorShim:
    def test_cpu_tensor_converted(self, open_viewer):
        import torch
        volume = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        viewer = open_viewer(volume)
        np.testing.assert_array_equal(viewer.stack.original_data[0],
                                      volume.numpy())

    def test_device_tensor_converted(self, open_viewer):
        import torch
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

    def test_numpy_and_none_pass_through(self, open_viewer):
        volume = make_volume((4, 4, 4))
        viewer = open_viewer(volume, None)
        np.testing.assert_array_equal(viewer.stack.original_data[0], volume)
        assert viewer.stack.original_data[1].shape == (20, 20, 20)


class TestDictConversion:
    def test_nested_recon_dict_becomes_strings(self):
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

    def test_non_dict_inputs_unchanged(self):
        assert convert_subdicts_to_strings(None) is None
        assert convert_subdicts_to_strings('text') == 'text'

    def test_wrapper_converts_list_of_dicts(self, open_viewer):
        recon_dict = {'recon_params': {'num_iterations': 3}}
        viewer = open_viewer(make_volume((4, 4, 4)), make_volume((4, 4, 4)),
                             data_dicts=[recon_dict, None])
        stored = viewer.stack.data_dicts[0]
        assert isinstance(stored['recon_params'], str)
        assert 'num_iterations' in stored['recon_params']
        assert viewer.stack.data_dicts[1] is None

    def test_wrapper_converts_single_dict(self, open_viewer):
        viewer = open_viewer(make_volume((4, 4, 4)),
                             data_dicts={'model_params': {'sharpness': 0.0}})
        assert isinstance(viewer.stack.data_dicts[0]['model_params'], str)


class TestWrapperBehavior:
    def test_returns_viewer_with_keepalive(self, open_viewer):
        import mbirtorch.viewer as viewer_module
        viewer = open_viewer(make_volume((4, 4, 4)))
        assert isinstance(viewer, mbirtorch.SliceViewer)
        assert viewer in viewer_module._NONBLOCKING_VIEWERS

    def test_save_fn_passthrough(self, open_viewer):
        def recorder(*args):
            pass
        viewer = open_viewer(make_volume((4, 4, 4)), save_fn=recorder)
        assert viewer.save_fn is recorder

    def test_signature_kwargs_forwarded(self, open_viewer):
        viewer = open_viewer(make_volume((4, 4, 6)), vmin=-1.0, vmax=2.0,
                             slice_label='View', slice_axis=0,
                             cmap='viridis', title='fwd')
        assert (viewer.stack.vmin, viewer.stack.vmax) == (-1.0, 2.0)
        assert viewer.stack.labels == ['View']
        assert viewer.stack.slice_axes == [0]
        assert viewer.images[0].get_cmap().name == 'viridis'
