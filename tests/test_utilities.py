"""Gates for stitch_arrays and copy_ct_model.

stitch_arrays blends a fixed overlap between adjacent arrays and returns the
result where the inputs already live: NumPy in gives NumPy out, tensors give a
tensor on their own device.  The values are checked on the CPU, so they run
everywhere.

copy_ct_model keeps the parent's reconstruction geometry and re-derives only
the axes whose inputs changed.  The test here covers the case where only the
views change, on all four geometries: the pitch, the aspect ratio, the recon
shape and the slice offset must all survive the copy, with no warning.
"""

import warnings

import numpy as np
import pytest
import torch

import mbirtorch


def _halves():
    """Two small arrays to stitch along the last axis, with distinct values."""
    first = np.arange(2 * 2 * 5.0).reshape(2, 2, 5).astype(np.float32)
    second = (np.arange(2 * 2 * 6.0).reshape(2, 2, 6) + 100.0).astype(np.float32)
    return first, second


# The stitched values for _halves() with overlap=3, recorded from the function
# itself.  With overlap 3 the ramp covers one element, so each output row keeps
# the first two elements of the first array, blends the third pair at weight
# 1/2, and then continues with the tail of the second array.  Pinning the
# numbers (they are all exact in float32) catches a change in the blend, not
# just a change in the shape.
EXPECTED_OVERLAP_3 = np.array(
    [[[0.0, 1.0, 2.0, 52.0, 102.0, 103.0, 104.0, 105.0],
      [5.0, 6.0, 7.0, 57.5, 108.0, 109.0, 110.0, 111.0]],
     [[10.0, 11.0, 12.0, 63.0, 114.0, 115.0, 116.0, 117.0],
      [15.0, 16.0, 17.0, 68.5, 120.0, 121.0, 122.0, 123.0]]],
    dtype=np.float32)


def test_stitch_arrays_numpy_values():
    """The blended values, through each input form.  A NumPy list gives a
    NumPy array; a tensor list, and a list that mixes a tensor with a NumPy
    array, give a CPU tensor with the same values.  The mixed case is the one
    the device check must not break: only one device is named, so the NumPy
    array simply joins it."""
    first, second = _halves()
    out = mbirtorch.stitch_arrays([first, second], overlap=3, axis=2)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert np.array_equal(out, EXPECTED_OVERLAP_3)

    tensors = mbirtorch.stitch_arrays([torch.as_tensor(first), torch.as_tensor(second)],
                                      overlap=3, axis=2)
    mixed = mbirtorch.stitch_arrays([torch.as_tensor(first), second], overlap=3, axis=2)
    for out in (tensors, mixed):
        assert isinstance(out, torch.Tensor) and out.device.type == 'cpu'
        assert np.array_equal(out.numpy(), EXPECTED_OVERLAP_3)


# ── copy_ct_model keeps the parent's reconstruction geometry ────────────────
def _cone_model(num_det_rows=24, num_det_cols=48, helical_z_shifts=None):
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ConeBeamModel((8, num_det_rows, num_det_cols), angles, source_detector_dist=200.0,
                                    source_iso_dist=100.0, helical_z_shifts=helical_z_shifts,
                                    compile_mode='off')
    model.set_params(no_warning=True, verbose=0)
    return model


def _parallel_model(num_det_rows=24, num_det_cols=48):
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, num_det_rows, num_det_cols), angles, compile_mode='off')
    model.set_params(no_warning=True, verbose=0)
    return model


def _multiaxis_model(num_det_rows=24, num_det_cols=48):
    angles = np.stack([np.linspace(0, np.pi, 8, endpoint=False), np.full(8, 0.3)], axis=1)
    model = mbirtorch.MultiAxisParallelModel((8, num_det_rows, num_det_cols), angles, compile_mode='off')
    model.set_params(no_warning=True, verbose=0)
    return model


def _translation_model(num_det_rows=24, num_det_cols=48):
    vectors = np.stack([np.linspace(-6, 6, 8), np.zeros(8), np.linspace(-4, 4, 8)], axis=1)
    model = mbirtorch.TranslationModel((8, num_det_rows, num_det_cols), vectors, source_detector_dist=200.0,
                                       source_iso_dist=100.0, compile_mode='off')
    model.set_params(no_warning=True, verbose=0)
    return model


def _shape(model):
    return tuple(int(n) for n in model.get_params('recon_shape'))


def _copy_warnings(caught):
    return [w for w in caught if 'copy_ct_model' in str(w.message)]


@pytest.mark.parametrize('make_model', [_cone_model, _parallel_model, _multiaxis_model, _translation_model],
                         ids=['cone', 'parallel', 'multiaxis', 'translation'])
def test_copy_keeps_the_parents_geometry_when_only_the_views_change(make_model):
    """A plain copy, and a copy over a subset of the views, keep every hand-set value: the pitch, the
    aspect ratio, the recon shape and (cone, multiaxis) the slice offset, and warn about nothing."""
    parent = make_model()
    required, optional, _ = parent.get_all_params()
    rows, cols, slices = _shape(parent)
    hand_set = dict(delta_voxel=0.74 * float(parent.get_params('delta_voxel')), voxel_row_aspect=1.25,
                    recon_shape=(rows + 1, cols - 3, max(1, slices - 1)))
    if 'recon_slice_offset' in optional:
        hand_set['recon_slice_offset'] = 1.4
    parent.set_params(no_warning=True, **hand_set)

    view_key = 'translation_vectors' if 'translation_vectors' in required else 'angles'
    subset = np.asarray(required[view_key])[::2]
    subset_kwargs = {'new_' + view_key: subset}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        copies = [mbirtorch.copy_ct_model(parent), mbirtorch.copy_ct_model(parent, **subset_kwargs)]
    assert not _copy_warnings(caught)
    for copy, num_views in zip(copies, (8, 4)):
        assert tuple(copy.get_params('sinogram_shape')) == (num_views, 24, 48)
        assert _shape(copy) == hand_set['recon_shape']
        for name in ('delta_voxel', 'voxel_row_aspect', 'recon_slice_offset'):
            if name in hand_set:
                assert float(copy.get_params(name)) == pytest.approx(hand_set[name])
