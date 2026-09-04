"""Tests for mbirtorch.preprocess.geometry_calibration.

The module builds a reduced problem, reduces a sinogram to match it, sweeps a geometry parameter,
and checks the rotation direction.  These tests cover the contracts in its docstrings: the sweep
stack has the shape the slice viewer pages through, a candidate equal to the model's current value
reproduces the full direct reconstruction, the reduced model keeps the full model's field of view in
ALU, the reduced sinogram equals a plain numpy reduction, a binned reconstruction puts the object in
the same place in ALU, nothing except apply_calibration changes the caller's state, and the refused
inputs raise.

Everything runs on CPU.  Almost every model is built with compile_mode='off', which keeps the suite
fast and is inherited by the reduced models the module builds.  One test uses the default compile
mode on a very small model, so the compiled path is covered too.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch.preprocess.geometry_calibration import (CalibrationResult, apply_calibration,
                                                       build_reduced_problem,
                                                       check_rotation_direction, parameter_sweep,
                                                       reduce_sinogram)
from mbirtorch.preprocess.utilities import correct_det_rotation
from mbirtorch.viewer import VolumeStack

NUM_VIEWS = 32
NUM_ROWS = 16
NUM_CHANNELS = 32


def _rel_max(out, ref):
    """The largest absolute difference divided by the largest absolute reference value."""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def _make_model(geometry, compile_mode='off'):
    """Build a small CPU model of the requested geometry."""
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    sinogram_shape = (NUM_VIEWS, NUM_ROWS, NUM_CHANNELS)
    if geometry == 'parallel':
        model = mbirtorch.ParallelBeamModel(sinogram_shape, angles, compile_mode=compile_mode)
    else:
        model = mbirtorch.ConeBeamModel(sinogram_shape, angles,
                                        source_detector_dist=4 * NUM_CHANNELS,
                                        source_iso_dist=2 * NUM_CHANNELS,
                                        compile_mode=compile_mode)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


def _phantom_sinogram(model):
    """Forward project the Shepp-Logan phantom that fits the model's recon shape."""
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(model.get_params('recon_shape'))
    return np.asarray(model.forward_project(phantom), dtype=np.float32)


def _result(parameter, value):
    """A CalibrationResult carrying one value, as an estimator would return."""
    return CalibrationResult(parameter=parameter, value=value, score=0.0,
                             candidates=np.array([value]), scores=np.array([0.0]),
                             method='test', reduction={})


def _copy_params(all_params):
    """A deep enough copy of a get_all_params result to compare against later."""
    return [{name: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
             for name, value in group.items()} for group in all_params]


def _params_equal(before, after):
    """True when two get_all_params results hold the same names and equal values."""
    for group_before, group_after in zip(before, after):
        if set(group_before) != set(group_after):
            return False
        for name, value in group_before.items():
            other = group_after[name]
            if value is None or other is None:
                if value is not other:
                    return False
            elif not np.array_equal(value, other):
                return False
    return True


def _centroid(image):
    """The intensity-weighted centroid, in index units, over values above half the maximum."""
    weights = np.where(image > 0.5 * np.max(image), image, 0.0).astype(np.float64)
    rows, cols = np.indices(image.shape)
    total = np.sum(weights)
    return float(np.sum(weights * rows) / total), float(np.sum(weights * cols) / total)


def _in_plane_alu(model, row, col):
    """Convert an in-plane recon position from index units to ALU, measured from the grid center."""
    recon_shape = model.get_params('recon_shape')
    delta_voxel, voxel_row_aspect = model.get_params(['delta_voxel', 'voxel_row_aspect'])
    row_alu = (row - (recon_shape[0] - 1) / 2.0) * voxel_row_aspect * delta_voxel
    col_alu = (col - (recon_shape[1] - 1) / 2.0) * delta_voxel
    return row_alu, col_alu


@pytest.fixture(scope='module')
def parallel_model():
    return _make_model('parallel')


@pytest.fixture(scope='module')
def cone_model():
    return _make_model('cone')


@pytest.fixture(scope='module')
def parallel_sino(parallel_model):
    return _phantom_sinogram(parallel_model)


@pytest.fixture(scope='module')
def cone_sino(cone_model):
    return _phantom_sinogram(cone_model)


@pytest.fixture(params=['parallel', 'cone'])
def model_and_sino(request):
    """The base model of one geometry and its sinogram, built once per module."""
    model = request.getfixturevalue(f'{request.param}_model')
    sino = request.getfixturevalue(f'{request.param}_sino')
    return request.param, model, sino


# ── the sweep stack and the viewer ────────────────────────────────────────────────────────────────

def test_sweep_shape_and_viewer_paging(model_and_sino):
    """The sweep returns a float32 stack with the candidates on the last axis.  The slice viewer's
    default slice axis pages through that axis, so the user sees one candidate per frame."""
    _, model, sino = model_and_sino
    recon_shape = model.get_params('recon_shape')
    stack = parameter_sweep(model, sino, 'det_channel_offset', [-1.0, 0.0, 1.0])
    assert stack.dtype == np.float32
    assert stack.shape == (recon_shape[0], recon_shape[1], 3)

    volume_stack = VolumeStack([stack])
    assert volume_stack.axes_perms == [[0, 1, 2]]
    assert volume_stack.data[0].shape[2] == 3


# ── sweep parity with the full direct reconstruction ──────────────────────────────────────────────

def test_sweep_matches_full_reconstruction(model_and_sino):
    """The candidate that equals the model's current parameter value reproduces the full model's
    direct reconstruction of the requested slice."""
    geometry, model, sino = model_and_sino
    num_slices = model.get_params('recon_shape')[2]
    middle = (num_slices - 1) // 2
    full_recon = np.asarray(model.recon_direct(sino))

    channel_stack = parameter_sweep(model, sino, 'det_channel_offset', [-1.0, 0.0, 1.0])
    channel_error = _rel_max(channel_stack[:, :, 1], full_recon[:, :, middle])
    print(f'{geometry} det_channel_offset parity rel_max = {channel_error:.2e}')
    assert channel_error <= 1e-5

    rotation_stack = parameter_sweep(model, sino, 'det_rotation', [0.0, 0.02])
    rotation_error = _rel_max(rotation_stack[:, :, 0], full_recon[:, :, middle])
    print(f'{geometry} det_rotation parity rel_max = {rotation_error:.2e}')
    assert rotation_error <= 1e-5

    if geometry == 'cone':
        row_stack = parameter_sweep(model, sino, 'det_row_offset', [-1.0, 0.0, 1.0])
        row_error = _rel_max(row_stack[:, :, 1], full_recon[:, :, middle])
        print(f'{geometry} det_row_offset parity rel_max = {row_error:.2e}')
        assert row_error <= 1e-5

    chosen_stack = parameter_sweep(model, sino, 'det_channel_offset', [0.0], slice_index=3)
    chosen_error = _rel_max(chosen_stack[:, :, 0], full_recon[:, :, 3])
    print(f'{geometry} slice_index=3 parity rel_max = {chosen_error:.2e}')
    assert chosen_error <= 1e-5


def test_sweep_parity_with_default_compile_mode():
    """The same parity holds when the model uses its default compile mode, which the reduced model
    inherits.  The model is kept tiny because compiling is the slow part."""
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    model = mbirtorch.ParallelBeamModel((16, 8, 16), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    sino = _phantom_sinogram(model)
    middle = (model.get_params('recon_shape')[2] - 1) // 2

    stack = parameter_sweep(model, sino, 'det_channel_offset', [-1.0, 0.0, 1.0])
    error = _rel_max(stack[:, :, 1], np.asarray(model.recon_direct(sino))[:, :, middle])
    print(f'compiled parallel parity rel_max = {error:.2e}')
    assert error <= 1e-5


# ── the reduced model's geometry ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('bin_factor', [1, 2, 4])
def test_reduced_field_of_view(model_and_sino, bin_factor):
    """Binning the detector coarsens the voxels and leaves the field of view in ALU unchanged, and it
    leaves det_channel_offset, which is in ALU, unchanged as well."""
    geometry, model, _ = model_and_sino
    reduced, _ = build_reduced_problem(model, view_stride=1, bin_factor=bin_factor,
                                       num_slab_slices=3)

    full_shape = model.get_params('recon_shape')
    full_delta = model.get_params('delta_voxel')
    full_row_aspect = model.get_params('voxel_row_aspect')
    reduced_shape = reduced.get_params('recon_shape')
    reduced_delta = reduced.get_params('delta_voxel')
    reduced_row_aspect = reduced.get_params('voxel_row_aspect')

    full_fov = (full_shape[0] * full_row_aspect * full_delta, full_shape[1] * full_delta)
    reduced_fov = (reduced_shape[0] * reduced_row_aspect * reduced_delta,
                   reduced_shape[1] * reduced_delta)
    print(f'{geometry} bin {bin_factor}: fov {reduced_fov} against {full_fov}')
    assert abs(reduced_fov[0] - full_fov[0]) <= reduced_delta
    assert abs(reduced_fov[1] - full_fov[1]) <= reduced_delta

    assert abs(reduced_delta - bin_factor * full_delta) <= 1e-6 * bin_factor * full_delta
    assert reduced.get_params('det_channel_offset') == model.get_params('det_channel_offset')


# ── the refused inputs ────────────────────────────────────────────────────────────────────────────

def test_refusals(cone_model, cone_sino, parallel_model, parallel_sino):
    """Every input the module documents as refused raises, and the message names the parameter at
    fault where the docstring says it does."""
    odd_model = mbirtorch.ParallelBeamModel((32, 16, 33),
                                            np.linspace(0, 2 * np.pi, 32, endpoint=False),
                                            compile_mode='off')
    odd_model.configure_devices(devices=['cpu'])
    odd_model.set_params(no_warning=True, verbose=0)
    with pytest.raises(ValueError, match='bin_factor'):
        build_reduced_problem(odd_model, view_stride=1, bin_factor=2, num_slab_slices=3)

    with pytest.raises(ValueError, match='view_stride'):
        build_reduced_problem(cone_model, view_stride=3, bin_factor=1, num_slab_slices=3)

    with pytest.raises(ValueError, match='slice_index'):
        build_reduced_problem(cone_model, view_stride=1, bin_factor=1, slice_index=99)

    translation_vectors = mbirtorch.gen_translation_vectors(2, 2, 4.0, 4.0)
    translation_model = mbirtorch.TranslationModel((4, 16, 16), translation_vectors,
                                                   source_detector_dist=8.0, source_iso_dist=8.0,
                                                   compile_mode='off')
    translation_model.configure_devices(devices=['cpu'])
    translation_model.set_params(no_warning=True, verbose=0)
    with pytest.raises(TypeError):
        build_reduced_problem(translation_model)

    with pytest.raises(ValueError):
        parameter_sweep(cone_model, cone_sino, 'sigma_y', [0.0])
    with pytest.raises(ValueError):
        parameter_sweep(parallel_model, parallel_sino, 'det_row_offset', [0.0])

    with pytest.raises(ValueError):
        check_rotation_direction(parallel_model, parallel_sino)

    _, reduction = build_reduced_problem(cone_model, view_stride=1, bin_factor=1,
                                         num_slab_slices=3)
    with pytest.raises(ValueError):
        reduce_sinogram(np.zeros((4, 4, 4), dtype=np.float32), reduction)


# ── the reduced sinogram's values ─────────────────────────────────────────────────────────────────

def _numpy_reduction(sino, reduction):
    """The reduced sinogram computed directly in numpy: take views, crop rows, average blocks."""
    stride, bin_factor = reduction['view_stride'], reduction['bin_factor']
    row_lo, row_hi = reduction['row_window']
    block = np.asarray(sino, dtype=np.float64)[::stride, row_lo:row_hi, :]
    num_views, num_rows, num_channels = block.shape
    return block.reshape(num_views, num_rows // bin_factor, bin_factor,
                         num_channels // bin_factor, bin_factor).mean(axis=(2, 4))


def test_reduce_sinogram_values(cone_model, cone_sino):
    """The reduced sinogram equals the same view selection, row crop, and block average done in
    numpy, from a numpy input and from a torch input.  With a detector rotation it equals the same
    reduction of the fully rotated sinogram, so rotating the kept rows about the full detector's
    center matches rotating the whole detector."""
    _, reduction = build_reduced_problem(cone_model, view_stride=2, bin_factor=2, num_slab_slices=3)
    reference = _numpy_reduction(cone_sino, reduction)

    out = reduce_sinogram(cone_sino, reduction)
    assert out.shape == reduction['sinogram_shape']
    assert np.allclose(out, reference, rtol=1e-5, atol=1e-6)

    out_tensor = reduce_sinogram(torch.as_tensor(cone_sino), reduction)
    assert np.allclose(out_tensor, reference, rtol=1e-5, atol=1e-6)

    rotated_sino = correct_det_rotation(np.array(cone_sino, copy=True), 0.03)
    rotated_reference = _numpy_reduction(rotated_sino, reduction)
    out_rotated = reduce_sinogram(cone_sino, reduction, det_rotation=0.03)
    print(f'rotated reduction max abs difference = '
          f'{np.max(np.abs(out_rotated - rotated_reference)):.2e}')
    assert np.allclose(out_rotated, rotated_reference, rtol=1e-4, atol=1e-5)

    # The reduction above keeps every detector row at this size, so the row crop is not exercised by
    # it.  A one-slice slab gives a narrow row window, which is the case the rotation has to get
    # right: the kept rows must turn about the full detector's center, not their own.
    _, narrow = build_reduced_problem(cone_model, view_stride=1, bin_factor=1, num_slab_slices=1)
    row_lo, row_hi = narrow['row_window']
    assert (row_lo, row_hi) != (0, NUM_ROWS)
    narrow_rotated = reduce_sinogram(cone_sino, narrow, det_rotation=0.03)
    assert np.allclose(narrow_rotated, rotated_sino[:, row_lo:row_hi, :], rtol=1e-4, atol=1e-5)


# ── binning and position ──────────────────────────────────────────────────────────────────────────

def test_binned_reduction_preserves_position(model_and_sino):
    """A small object away from the axis lands at the same place in ALU in the binned
    reconstruction as in the full one.  This checks that the detector offsets, which are in ALU,
    carry over to the binned model unchanged."""
    geometry, model, _ = model_and_sino
    recon_shape = model.get_params('recon_shape')
    rows, cols, _ = np.indices(recon_shape)
    phantom = (((rows - 20) ** 2 + (cols - 12) ** 2) <= 9).astype(np.float32)
    sino = np.asarray(model.forward_project(phantom), dtype=np.float32)

    full_recon = np.asarray(model.recon_direct(sino))
    full_centroid = _centroid(full_recon[:, :, (recon_shape[2] - 1) // 2])

    reduced, reduction = build_reduced_problem(model, view_stride=1, bin_factor=2,
                                               num_slab_slices=3)
    reduced_recon = np.asarray(reduced.recon_direct(reduce_sinogram(sino, reduction)))
    reduced_centroid = _centroid(reduced_recon[:, :, reduced_recon.shape[2] // 2])

    full_alu = _in_plane_alu(model, *full_centroid)
    reduced_alu = _in_plane_alu(reduced, *reduced_centroid)
    reduced_delta = reduced.get_params('delta_voxel')
    offsets = [(a - b) / reduced_delta for a, b in zip(reduced_alu, full_alu)]
    print(f'{geometry} binned centroid offset in reduced voxels = '
          f'{offsets[0]:.3f}, {offsets[1]:.3f}')
    assert abs(offsets[0]) < 0.5
    assert abs(offsets[1]) < 0.5


# ── no state change ───────────────────────────────────────────────────────────────────────────────

def test_functions_do_not_change_the_caller_state(cone_model, cone_sino):
    """Only apply_calibration changes state.  The reduction, the sweep, and the rotation check leave
    the model's parameters and the sinogram as they were."""
    params_before = _copy_params(cone_model.get_all_params())
    sino_before = cone_sino.copy()

    _, reduction = build_reduced_problem(cone_model, view_stride=2, bin_factor=2,
                                         num_slab_slices=3)
    reduce_sinogram(cone_sino, reduction)
    parameter_sweep(cone_model, cone_sino, 'det_channel_offset', [-1.0, 0.0, 1.0])
    check_rotation_direction(cone_model, cone_sino, view_stride=2, bin_factor=2)

    assert _params_equal(params_before, cone_model.get_all_params())
    assert np.array_equal(cone_sino, sino_before)
    assert cone_model.get_params('det_channel_offset') == 0.0


# ── applying a result ─────────────────────────────────────────────────────────────────────────────

def test_apply_calibration(cone_sino):
    """apply_calibration sets a model parameter, rotates the sinogram in place, accepts a dict of
    results, negates the view angles for a rotation direction of -1, and refuses a read-only
    sinogram and an unknown parameter name."""
    model = _make_model('cone')
    returned_model, _ = apply_calibration(model, cone_sino.copy(), _result('det_channel_offset', 0.7))
    assert returned_model is model
    assert model.get_params('det_channel_offset') == 0.7

    sino = np.array(cone_sino, dtype=np.float32, copy=True)
    rotated_reference = correct_det_rotation(np.array(cone_sino, copy=True), 0.02)
    _, returned_sino = apply_calibration(model, sino, _result('det_rotation', 0.02))
    assert returned_sino is sino
    assert np.allclose(returned_sino, rotated_reference, rtol=1e-4, atol=1e-5)

    read_only = np.array(cone_sino, dtype=np.float32, copy=True)
    read_only.flags.writeable = False
    with pytest.raises(ValueError):
        apply_calibration(model, read_only, _result('det_rotation', 0.02))

    apply_calibration(model, cone_sino.copy(),
                      {'det_channel_offset': _result('det_channel_offset', 0.3)})
    assert model.get_params('det_channel_offset') == 0.3

    fresh_model = _make_model('cone')
    angles_before = np.array(fresh_model.get_all_params()[0]['angles'], copy=True)
    apply_calibration(fresh_model, cone_sino.copy(), _result('rotation_direction', -1.0))
    assert np.array_equal(fresh_model.get_all_params()[0]['angles'], -angles_before)

    with pytest.raises(ValueError):
        apply_calibration(fresh_model, cone_sino.copy(), _result('not_a_parameter', 1.0))


# ── the rotation-direction check ──────────────────────────────────────────────────────────────────

def test_check_rotation_direction_on_cone_data(cone_model, cone_sino):
    """The check picks the direction the data were simulated with, for the angles as given and for
    the angles negated, and the two scores are separated by at least a factor of 1.5."""
    result = check_rotation_direction(cone_model, cone_sino, view_stride=2, bin_factor=1)
    print(f'forward scores = {result.scores}, ratio = {result.scores[1] / result.scores[0]:.2f}')
    assert result.value == 1.0
    assert np.array_equal(result.candidates, np.array([1.0, -1.0]))
    assert result.method == 'direct_residual'
    assert result.scores[1] / result.scores[0] >= 1.5

    angles = np.asarray(cone_model.get_all_params()[0]['angles'])
    reversed_model = mbirtorch.copy_ct_model(cone_model, new_angles=-angles,
                                             new_helical_z_shifts=np.zeros_like(angles))
    reversed_model.compile_mode = 'off'
    reversed_model.configure_devices(devices=['cpu'])
    reversed_model.set_params(no_warning=True, verbose=0)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        reversed_model.get_params('recon_shape'))
    reversed_sino = np.asarray(reversed_model.forward_project(phantom), dtype=np.float32)

    reversed_result = check_rotation_direction(cone_model, reversed_sino, view_stride=2,
                                               bin_factor=1)
    print(f'reversed scores = {reversed_result.scores}, '
          f'ratio = {reversed_result.scores[0] / reversed_result.scores[1]:.2f}')
    assert reversed_result.value == -1.0


# ── whole-extent reductions ───────────────────────────────────────────────────────────────────────

def test_helical_and_whole_extent_reductions(cone_model):
    """A helical scan and a request for the whole axial extent both keep every detector row and
    every recon slice.  The rotation-direction check refuses a helical scan."""
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    helical_model = mbirtorch.ConeBeamModel((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS), angles,
                                            source_detector_dist=4 * NUM_CHANNELS,
                                            source_iso_dist=2 * NUM_CHANNELS,
                                            helical_z_shifts=np.linspace(-2, 2, NUM_VIEWS),
                                            compile_mode='off')
    helical_model.configure_devices(devices=['cpu'])
    helical_model.set_params(no_warning=True, verbose=0)

    reduced, reduction = build_reduced_problem(helical_model, view_stride=2, bin_factor=2,
                                               num_slab_slices=3)
    assert reduction['axial_thinning'] is False
    assert reduction['row_window'] == (0, NUM_ROWS)
    assert reduction['recon_shape'][2] == reduced.get_params('recon_shape')[2]

    with pytest.raises(ValueError):
        check_rotation_direction(helical_model, np.zeros((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS),
                                                         dtype=np.float32))

    _, whole_extent = build_reduced_problem(cone_model, view_stride=2, bin_factor=2,
                                            num_slab_slices=None)
    assert whole_extent['axial_thinning'] is False
    assert whole_extent['row_window'] == (0, NUM_ROWS)


# ── an off-center band under rotation, and a helical sweep ────────────────────────────────────────

def test_rotation_of_a_band_away_from_the_detector_center(cone_model, cone_sino):
    """A band of rows away from the detector center, rotated by a large angle, still matches the
    crop of the fully rotated sinogram.  The rows the rotation reads beyond the band grow with the
    band's distance from the center, and the read margin has to account for that distance."""
    angle = 0.1
    rotated_sino = correct_det_rotation(np.array(cone_sino, copy=True), angle)
    for slice_index in (1, NUM_ROWS - 2):
        _, reduction = build_reduced_problem(cone_model, view_stride=1, bin_factor=1,
                                             num_slab_slices=1, slice_index=slice_index)
        row_lo, row_hi = reduction['row_window']
        assert row_lo > 0 or row_hi < NUM_ROWS
        out = reduce_sinogram(cone_sino, reduction, det_rotation=angle)
        difference = np.max(np.abs(out - rotated_sino[:, row_lo:row_hi, :]))
        print(f'slice {slice_index}: rows {row_lo}:{row_hi}, max abs difference {difference:.2e}')
        assert np.allclose(out, rotated_sino[:, row_lo:row_hi, :], rtol=1e-4, atol=1e-5)


def test_helical_sweep_reads_the_requested_slice():
    """On a helical scan the sweep keeps the whole volume, and the requested slice is read back
    from the reduced model's own slice grid.  The 0.0 candidate matches the full model's direct
    reconstruction of that slice."""
    angles = np.linspace(0, 4 * np.pi, NUM_VIEWS, endpoint=False)
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS), angles,
                                    source_detector_dist=4 * NUM_CHANNELS,
                                    source_iso_dist=2 * NUM_CHANNELS,
                                    helical_z_shifts=np.linspace(-4, 4, NUM_VIEWS),
                                    compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    sino = _phantom_sinogram(model)
    full_recon = np.asarray(model.recon_direct(sino))
    for slice_index in (3, (full_recon.shape[2] - 1) // 2):
        stack = parameter_sweep(model, sino, 'det_channel_offset', [0.0], slice_index=slice_index)
        error = _rel_max(stack[:, :, 0], full_recon[:, :, slice_index])
        print(f'helical slice {slice_index} parity rel_max = {error:.2e}')
        assert error <= 1e-5


# ── multiaxis, curved detector, and a parallel source ─────────────────────────────────────────────

def test_multiaxis_sweep_matches_full_reconstruction():
    """On a multiaxis parallel model with a 10 degree elevation, the row window computed from the
    elevation-dependent row map contains every row the slice needs, so the sweep candidate equal
    to the current offset reproduces the full direct reconstruction of that slice."""
    azimuths = np.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    elevations = np.deg2rad(10.0) * np.ones(NUM_VIEWS)
    angles = np.column_stack([azimuths, elevations]).astype(np.float32)
    model = mbirtorch.MultiAxisParallelModel((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS), angles,
                                             compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    sino = _phantom_sinogram(model)
    full_recon = np.asarray(model.recon_direct(sino))
    num_slices = full_recon.shape[2]
    for slice_index in (num_slices // 4, (num_slices - 1) // 2):
        stack = parameter_sweep(model, sino, 'det_channel_offset', [0.0], slice_index=slice_index)
        _, reduction = build_reduced_problem(model, view_stride=1, bin_factor=1, num_slab_slices=1,
                                             slice_index=slice_index)
        error = _rel_max(stack[:, :, 0], full_recon[:, :, slice_index])
        print(f'multiaxis slice {slice_index}: rows {reduction["row_window"]}, parity rel_max = {error:.2e}')
        assert error <= 1e-5


def test_curved_detector_refuses_rotation_and_sweeps_offset():
    """A curved detector refuses a det_rotation sweep, and its det_channel_offset sweep reproduces the
    full direct reconstruction, because the offset enters the same expression on both detectors."""
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS), angles,
                                    source_detector_dist=4 * NUM_CHANNELS,
                                    source_iso_dist=2 * NUM_CHANNELS, use_curved_detector=True,
                                    compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    sino = _phantom_sinogram(model)
    with pytest.raises(ValueError, match='curved'):
        parameter_sweep(model, sino, 'det_rotation', [0.0, 0.01])
    middle = (model.get_params('recon_shape')[2] - 1) // 2
    stack = parameter_sweep(model, sino, 'det_channel_offset', [0.0])
    error = _rel_max(stack[:, :, 0], np.asarray(model.recon_direct(sino))[:, :, middle])
    print(f'curved detector parity rel_max = {error:.2e}')
    assert error <= 1e-5
