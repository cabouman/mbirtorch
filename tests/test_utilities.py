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

construct_time_frame_models splits a scan into overlapping windows of
consecutive views and builds one model per window.  The view slices are
integers, so they are checked exactly against pinned values, including a
scan over more than one rotation with its angles stored modulo one rotation.
The device helpers are checked against what torch reports.
"""

import warnings

import os

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _sharding
from mbirtorch.tomography_model import cpu_devices, default_devices, gpu_devices
from mbirtorch.utilities import construct_time_frame_models

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


# ── construct_time_frame_models splits a scan into overlapping frames ────────
def _scan_model(num_views, degrees_per_view, wrap=False):
    """A small cone-beam scan with one angle per view.  ``wrap`` stores the
    angles modulo one rotation, as scanner metadata often does."""
    degrees = degrees_per_view * np.arange(num_views)
    if wrap:
        degrees = degrees % 360.0
    model = mbirtorch.ConeBeamModel((num_views, 8, 10), np.deg2rad(degrees), source_detector_dist=100.0,
                                    source_iso_dist=50.0, compile_mode='off')
    model.set_params(no_warning=True, verbose=0)
    return model


def _pairs(view_slices):
    return [(view_slice.start, view_slice.stop) for view_slice in view_slices]


def test_time_frames_of_a_one_rotation_scan():
    """24 views over one rotation with the default parameters give five frames
    of eight views starting every four views.  Each frame model is a copy of
    the parent over the frame's angles: its sinogram shape has the frame's
    view count, its angles are exactly the parent's over the frame's slice,
    and its reconstruction geometry is the parent's."""
    parent = _scan_model(24, 15.0)
    model_list, view_slices = construct_time_frame_models(parent)
    assert len(model_list) == 5
    assert view_slices[1] == slice(4, 12)
    assert _pairs(view_slices) == [(0, 8), (4, 12), (8, 16), (12, 20), (16, 24)]
    parent_angles = np.asarray(parent.get_all_params()[0]['angles'])
    for frame, view_slice in zip(model_list, view_slices):
        assert tuple(frame.get_params('sinogram_shape')) == (8, 8, 10)
        assert _shape(frame) == _shape(parent)
        assert np.array_equal(np.asarray(frame.get_all_params()[0]['angles']), parent_angles[view_slice])


@pytest.mark.parametrize(
    'num_views, degrees_per_view, wrap, frames_per_rotation, frame_overlap_factor, expected',
    [
        # 600 degrees of views with the angles stored modulo one rotation: nine
        # frames of 48 views starting every 24 views.  The wrap adds one large
        # difference per rotation, which the median angular step ignores.
        (240, 2.5, True, 6, 2.0,
         [(0, 48), (24, 72), (48, 96), (72, 120), (96, 144), (120, 168), (144, 192), (168, 216), (192, 240)]),
        # The same scan with monotonic angles gives the same frames.
        (240, 2.5, False, 6, 2.0,
         [(0, 48), (24, 72), (48, 96), (72, 120), (96, 144), (120, 168), (144, 192), (168, 216), (192, 240)]),
        # A span of 135 degrees at 10 degrees per view is 13.5 views, rounded
        # to 14 (round half to even); the stride is 9 views; the last four
        # views fill no frame and are discarded.
        (36, 10.0, False, 4, 1.5, [(0, 14), (9, 23), (18, 32)]),
        # Three frames share each view: a span of 60 views at a stride of 20.
        (100, 3.6, False, 5, 3.0, [(0, 60), (20, 80), (40, 100)]),
    ],
    ids=['multi_rotation_wrapped', 'multi_rotation_monotonic', 'half_view_rounding', 'three_frame_overlap'])
def test_time_frame_view_slices(num_views, degrees_per_view, wrap, frames_per_rotation, frame_overlap_factor,
                                expected):
    """The view slices for a parameter set, checked exactly: they are integers
    fixed by the frame arithmetic, and each frame model's view count is the
    slice's length."""
    model_list, view_slices = construct_time_frame_models(
        _scan_model(num_views, degrees_per_view, wrap=wrap), frames_per_rotation=frames_per_rotation,
        frame_overlap_factor=frame_overlap_factor)
    assert _pairs(view_slices) == expected
    assert len(model_list) == len(expected)
    for frame, (start, stop) in zip(model_list, expected):
        assert tuple(frame.get_params('sinogram_shape')) == (stop - start, 8, 10)


def test_time_frames_reject_parameters_that_give_no_frames():
    """A stride or a span below one view, a span longer than the scan, angles
    with no spacing, and a model without one angle per view are each refused
    with a message that names the cause."""
    parent = _scan_model(24, 15.0)
    # At the default overlap factor the span is twice the stride, so a stride
    # below one view is caught by the span check first.
    with pytest.raises(ValueError, match='smaller than one view'):
        construct_time_frame_models(parent, frames_per_rotation=100)     # a 3.6 degree stride
    with pytest.raises(ValueError, match='stride smaller than one view'):
        construct_time_frame_models(parent, frames_per_rotation=100, frame_overlap_factor=5.0)   # an 18 degree span
    with pytest.raises(ValueError, match='frame span smaller than one view'):
        construct_time_frame_models(parent, frame_overlap_factor=0.05)   # a 3 degree span
    with pytest.raises(ValueError, match='cannot exceed the full scan'):
        construct_time_frame_models(parent, frame_overlap_factor=10.0)   # a 600 degree span
    with pytest.raises(ValueError, match='nonzero spacing'):
        construct_time_frame_models(_scan_model(24, 0.0))
    for model in (_multiaxis_model(), _translation_model()):
        with pytest.raises(ValueError, match='one angle per view'):
            construct_time_frame_models(model)


# ── the device helpers report the hardware ───────────────────────────────────
def test_device_helpers_report_the_hardware():
    """cpu_devices has one entry, gpu_devices lists every CUDA device or the
    MPS device or nothing, and default_devices is the GPU list when it is
    nonempty and the CPU device otherwise."""
    assert cpu_devices() == (torch.device('cpu'),)
    gpus = gpu_devices()
    if torch.cuda.is_available():
        assert gpus == tuple(torch.device('cuda', i) for i in range(torch.cuda.device_count()))
    elif torch.backends.mps.is_available():
        assert gpus == (torch.device('mps'),)
    else:
        assert gpus == ()
    defaults = default_devices()
    assert isinstance(defaults, list) and len(defaults) >= 1
    assert defaults == (list(gpus) or [torch.device('cpu')])


# ── save_volume_as_gif writes a real animation ───────────────────────────────
def test_save_volume_as_gif_writes_the_frames_each_form_selects(tmp_path):
    """Each form of the call selects its own set of frames, and every one of
    them reaches the file: a 4D volume plays over time by default, over time in
    a chosen plane when slice_axis names one, and through the slices of a
    single frame when slice_axis is the frame axis; a 3D volume plays over its
    first axis.  The frame count of the written file is the length of the axis
    the movie loops over, so the file is read back rather than trusted."""
    from PIL import Image

    volume_4d = np.random.default_rng(0).random((6, 12, 14, 10)).astype(np.float32)
    volume_3d = volume_4d[0]
    cases = [
        ('over time at the middle x', volume_4d, {}, 6),
        ('over time in an XY plane', volume_4d, dict(slice_axis=3), 6),
        ('through z of one frame', volume_4d, dict(frame_axis=3, slice_axis=0, slice_index=0), 10),
        ('a 3D volume over x', volume_3d, {}, 12),
    ]
    for name, volume, kwargs, expected in cases:
        path = str(tmp_path / f"{name.replace(' ', '_')}.gif")
        mbirtorch.save_volume_as_gif(volume, path, vmin=0, vmax=1, **kwargs)
        with Image.open(path) as written:
            print(f"{name}: {written.n_frames} frames, {os.path.getsize(path)} bytes")
            assert written.n_frames == expected

    # The frame duration is what fps asks for, in the hundredths of a second a
    # GIF stores.  Five frames per second is 200 ms.
    path = str(tmp_path / 'timed.gif')
    mbirtorch.save_volume_as_gif(volume_4d, path, fps=5, vmin=0, vmax=1)
    with Image.open(path) as written:
        assert written.info['duration'] == 200

    # A constant volume gives the display a zero-width window, which is widened
    # rather than left to divide by zero.
    flat = np.full((4, 6, 6, 6), 0.25, dtype=np.float32)
    mbirtorch.save_volume_as_gif(flat, str(tmp_path / 'flat.gif'))

    with pytest.raises(ValueError, match='3D volume'):
        mbirtorch.save_volume_as_gif(volume_3d, str(tmp_path / 'x.gif'), slice_axis=1)
    with pytest.raises(ValueError, match='must differ'):
        mbirtorch.save_volume_as_gif(volume_4d, str(tmp_path / 'x.gif'), frame_axis=1, slice_axis=1)
    with pytest.raises(ValueError, match='must be positive'):
        mbirtorch.save_volume_as_gif(volume_4d, str(tmp_path / 'x.gif'), fps=0)
    with pytest.raises(ValueError, match='3D .* or 4D'):
        mbirtorch.save_volume_as_gif(volume_3d[0], str(tmp_path / 'x.gif'))
