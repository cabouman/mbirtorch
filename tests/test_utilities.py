"""Gates for stitch_arrays and copy_ct_model.

stitch_arrays blends a fixed overlap between adjacent arrays and returns the
result where the inputs already live: NumPy in gives NumPy out, tensors give a
tensor on their own device.  Two inputs it cannot serve are refused with a
message that names the problem, rather than being quietly relocated or failing
on a missing attribute deep inside the function: tensors spread over more than
one device, and an array in the divided device form (a Shards container).

The refusal for the divided form and the normal-path values are checked on the
CPU, so they run everywhere.  The mixed-device refusal needs two real GPUs and
is skipped otherwise.

copy_ct_model keeps the parent's reconstruction geometry and re-derives only
the axes whose inputs changed: rows or helical travel for the slices and the
slice offset, channels for the in-plane shape, both for a translation model.
A re-derived count is sized at the parent's voxel pitch, and a warning names
any value the parent had set by hand that the copy re-derived.
"""

import warnings

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _sharding

requires_two_cuda = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="the mixed-device refusal needs at least two CUDA devices")


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
    first, second = _halves()
    out = mbirtorch.stitch_arrays([first, second], overlap=3, axis=2)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert np.array_equal(out, EXPECTED_OVERLAP_3)


def test_stitch_arrays_one_device_matches_numpy():
    """A tensor list, and a list that mixes a tensor with a NumPy array, both
    stay valid and give the same values as the all-NumPy call.  The mixed case
    is the one the device check must not break: only one device is named, so
    the NumPy array simply joins it."""
    first, second = _halves()
    tensors = mbirtorch.stitch_arrays([torch.as_tensor(first), torch.as_tensor(second)],
                                      overlap=3, axis=2)
    mixed = mbirtorch.stitch_arrays([torch.as_tensor(first), second], overlap=3, axis=2)
    for out in (tensors, mixed):
        assert isinstance(out, torch.Tensor) and out.device.type == 'cpu'
        assert np.array_equal(out.numpy(), EXPECTED_OVERLAP_3)


def test_stitch_arrays_refuses_divided_form():
    """A Shards holds one tensor per device, so it has no shape of its own.
    Two CPU shards are enough to build one; no GPU is involved."""
    first, second = _halves()
    placement = _sharding.Placement(['cpu', 'cpu'], axis=-1, axis_len=4)
    shards = _sharding.Shards([torch.as_tensor(first[..., :2]),
                               torch.as_tensor(first[..., 2:4])], placement)
    with pytest.raises(TypeError, match="divided device form"):
        mbirtorch.stitch_arrays([shards, second], overlap=3, axis=2)


@requires_two_cuda
def test_stitch_arrays_refuses_mixed_devices():
    first, second = _halves()
    with pytest.raises(ValueError, match="one device"):
        mbirtorch.stitch_arrays([torch.as_tensor(first).to('cuda:0'),
                                 torch.as_tensor(second).to('cuda:1')],
                                overlap=3, axis=2)


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


@pytest.mark.parametrize('make_model', [_cone_model, _parallel_model, _multiaxis_model],
                         ids=['cone', 'parallel', 'multiaxis'])
def test_copy_re_derives_the_slices_when_the_rows_change(make_model):
    """Fewer detector rows re-derive the slice count (and a cone model's slice offset) for the new
    detector and keep a hand-set in-plane shape.  A parent at the automatic slice count gets no
    warning; a hand-set slice count is replaced with a warning that names it, which no_warning
    silences."""
    parent = make_model()
    rows, cols, slices = _shape(parent)
    parent.set_params(no_warning=True, recon_shape=(rows + 1, cols - 3, slices))
    automatic = make_model(num_det_rows=12)
    expected_slices = _shape(automatic)[2]
    if make_model is _parallel_model:
        assert expected_slices == 12         # one slice per detector row
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        copy = mbirtorch.copy_ct_model(parent, new_num_det_rows=12)
    assert not _copy_warnings(caught)
    assert _shape(copy) == (rows + 1, cols - 3, expected_slices)
    if 'recon_slice_offset' in parent.get_all_params()[1]:
        assert float(copy.get_params('recon_slice_offset')) == pytest.approx(
            float(automatic.get_params('recon_slice_offset')))

    if make_model is _parallel_model:
        return                               # a parallel slice count is never set by hand
    parent.set_params(no_warning=True, recon_shape=(rows + 1, cols - 3, slices - 1))
    with pytest.warns(UserWarning, match='slice count'):
        copy = mbirtorch.copy_ct_model(parent, new_num_det_rows=12)
    assert _shape(copy) == (rows + 1, cols - 3, expected_slices)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mbirtorch.copy_ct_model(parent, new_num_det_rows=12, no_warning=True)
    assert not _copy_warnings(caught)


def test_copy_re_derives_the_in_plane_shape_when_the_channels_change():
    """Fewer detector channels re-derive the in-plane shape for the new detector and keep the
    hand-set slice count and slice offset; the replaced in-plane shape is named in a warning."""
    parent = _cone_model()
    rows, cols, slices = _shape(parent)
    parent.set_params(no_warning=True, recon_shape=(rows + 1, cols - 3, slices - 1), recon_slice_offset=1.4)
    expected_in_plane = _shape(_cone_model(num_det_cols=32))[:2]
    with pytest.warns(UserWarning, match='in-plane'):
        copy = mbirtorch.copy_ct_model(parent, new_num_det_cols=32)
    assert _shape(copy) == expected_in_plane + (slices - 1,)
    assert float(copy.get_params('recon_slice_offset')) == pytest.approx(1.4)


def test_re_derived_counts_are_sized_at_the_parents_pitch():
    """A re-derived slice count covers the same extent at the parent's hand-set pitch as the
    automatic pass covers at its own pitch.  Parallel beam keeps one slice per row whatever the
    pitch."""
    parent = _cone_model()
    automatic_pitch = float(parent.get_params('delta_voxel'))
    pitch = 0.74 * automatic_pitch
    parent.set_params(no_warning=True, delta_voxel=pitch)
    automatic_slices = _shape(_cone_model(num_det_rows=12))[2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        copy = mbirtorch.copy_ct_model(parent, new_num_det_rows=12)
    assert not _copy_warnings(caught)
    assert float(copy.get_params('delta_voxel')) == pytest.approx(pitch)
    assert _shape(copy) == _shape(parent)[:2] + (int(np.ceil(automatic_slices * automatic_pitch / pitch)),)

    parent = _parallel_model()
    parent.set_params(no_warning=True, delta_voxel=0.74 * float(parent.get_params('delta_voxel')))
    assert _shape(mbirtorch.copy_ct_model(parent, new_num_det_rows=12))[2] == 12


def test_copy_re_derives_the_slices_when_the_helical_travel_changes():
    """A view subset with a shorter helical travel re-derives the slice count and slice offset for
    that travel and keeps the hand-set in-plane shape; the same travel under other angles keeps
    everything."""
    z_shifts = np.linspace(0.0, 6.0, 8)
    parent = _cone_model(helical_z_shifts=z_shifts)
    rows, cols, slices = _shape(parent)
    parent.set_params(no_warning=True, recon_shape=(rows + 1, cols - 3, slices), recon_slice_offset=1.4)
    angles = np.asarray(parent.get_all_params()[0]['angles'])

    automatic = mbirtorch.ConeBeamModel((4, 24, 48), angles[:4], source_detector_dist=200.0,
                                        source_iso_dist=100.0, helical_z_shifts=z_shifts[:4],
                                        compile_mode='off')
    with pytest.warns(UserWarning, match='recon_slice_offset'):
        copy = mbirtorch.copy_ct_model(parent, new_angles=angles[:4], new_helical_z_shifts=z_shifts[:4])
    assert _shape(copy) == (rows + 1, cols - 3, _shape(automatic)[2])
    assert float(copy.get_params('recon_slice_offset')) == pytest.approx(
        float(automatic.get_params('recon_slice_offset')))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        copy = mbirtorch.copy_ct_model(parent, new_angles=angles[::-1], new_helical_z_shifts=z_shifts)
    assert not _copy_warnings(caught)
    assert _shape(copy) == (rows + 1, cols - 3, slices)
    assert float(copy.get_params('recon_slice_offset')) == pytest.approx(1.4)


def test_translation_copy_re_derives_the_whole_shape_when_the_detector_changes():
    """The translation heuristic sizes every recon axis from both detector axes, so a detector
    change re-derives the whole shape, named in the warning when it was set by hand."""
    parent = _translation_model()
    rows, cols, slices = _shape(parent)
    parent.set_params(no_warning=True, recon_shape=(rows + 1, cols - 2, slices - 1))
    required, optional, regularization = parent.get_all_params()
    required = dict(required)
    required['sinogram_shape'] = (8, 12, 48)
    optional = {k: v for k, v in optional.items() if k != 'recon_shape'}
    automatic = mbirtorch.build_model(required, optional, regularization)
    with pytest.warns(UserWarning, match='in-plane recon shape and the slice count'):
        copy = mbirtorch.copy_ct_model(parent, new_num_det_rows=12)
    assert _shape(copy) == _shape(automatic)
