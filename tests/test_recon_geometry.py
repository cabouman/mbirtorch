"""Gates for the automatic reconstruction geometry and for build_model's handling of supplied values.

The cone and multiaxis automatic passes center the volume on the z band the detector rows
illuminate, so a detector row offset moves the volume with the detector, and the cone padding is
measured from that band.  build_model keeps a recon_shape, delta_voxel, or recon_slice_offset the
parameter dicts carry, sizing the automatic shape at a supplied pitch, so a model round-trips through
get_all_params exactly.  apply_calibration moves the axial center with a calibrated row offset and
leaves the shape and pitch alone.
"""

import numpy as np
import pytest

import mbirtorch
from mbirtorch.preprocess.geometry_calibration import CalibrationResult, apply_calibration


def _cone(num_det_rows=24, helical_z_shifts=None, **params):
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ConeBeamModel((8, num_det_rows, 48), angles, source_detector_dist=200.0,
                                    source_iso_dist=100.0, helical_z_shifts=helical_z_shifts,
                                    compile_mode='off')
    model.set_params(no_warning=True, verbose=0, **params)
    return model


def _parallel(**params):
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 24, 48), angles, compile_mode='off')
    model.set_params(no_warning=True, verbose=0, **params)
    return model


def _multiaxis(elevations, **params):
    elevations = np.asarray(elevations, dtype=float)
    angles = np.stack([np.linspace(0, np.pi, len(elevations), endpoint=False), elevations], axis=1)
    model = mbirtorch.MultiAxisParallelModel((len(elevations), 24, 48), angles, compile_mode='off')
    model.set_params(no_warning=True, verbose=0, **params)
    return model


def _shape(model):
    return tuple(int(n) for n in model.get_params('recon_shape'))


def _row_offset_result(value):
    return CalibrationResult(parameter='det_row_offset', value=value, score=0.0,
                             candidates=np.array([value]), scores=np.array([0.0]), method='test',
                             reduction={})


# ── the automatic pass follows the detector ──────────────────────────────────
def test_cone_volume_is_centered_on_the_illuminated_band():
    """With a row offset the automatic slice offset is -det_row_offset / magnification, plus the
    center of any helical travel, and the slice count is unchanged."""
    model = _cone()
    slices = _shape(model)[2]
    magnification = model.get_magnification()
    model.set_params(no_warning=True, det_row_offset=3.0)
    model.auto_set_recon_geometry()
    assert float(model.get_params('recon_slice_offset')) == pytest.approx(-3.0 / magnification)
    assert _shape(model)[2] == slices

    helical = _cone(helical_z_shifts=np.linspace(0.0, 6.0, 8), det_row_offset=-2.0)
    helical.auto_set_recon_geometry()
    assert float(helical.get_params('recon_slice_offset')) == pytest.approx(3.0 + 2.0 / magnification)


def test_cone_padding_is_measured_from_the_band():
    """Padding extends each end by the far-side excess of its edge ray over the band's own edge, so
    the padded volume always contains the band, and a row offset small enough that each end's excess
    rounds to the same slice count shifts the padded volume by exactly the band shift."""
    model = _cone(axial_pad_fraction=1.0)
    model.auto_set_recon_geometry()
    magnification = model.get_magnification()
    reference_shape, reference_offset = _shape(model), float(model.get_params('recon_slice_offset'))
    assert reference_shape[2] > _shape(_cone())[2]                          # the padding is real
    half_band = 0.5 * 24 * float(model.get_params('delta_det_row')) / magnification
    for det_row_offset in (1.0, -1.5):
        model.set_params(no_warning=True, det_row_offset=det_row_offset)
        model.auto_set_recon_geometry()
        band_center = -det_row_offset / magnification
        offset = float(model.get_params('recon_slice_offset'))
        height = _shape(model)[2] * float(model.get_params('delta_voxel')) \
            * float(model.get_params('voxel_slice_aspect'))
        assert offset - 0.5 * height <= band_center - half_band + 1e-9
        assert offset + 0.5 * height >= band_center + half_band - 1e-9
        assert _shape(model) == reference_shape
        assert offset == pytest.approx(reference_offset + band_center)


def test_multiaxis_volume_covers_the_union_of_the_illuminated_bands():
    """Each view illuminates the band of z whose v = z cos(el) lies on its rows, shifted by the row
    offset; the volume is the union over the views, centered on it.  With no row offset this is the
    height 2 max_v / min|cos(el)| about zero."""
    elevations = [0.0, 0.3, 0.6]
    model = _multiaxis(elevations)
    max_v = 0.5 * 24 * float(model.get_params('delta_det_row'))
    pitch = float(model.get_params('delta_voxel')) * float(model.get_params('voxel_slice_aspect'))
    assert float(model.get_params('recon_slice_offset')) == pytest.approx(0.0)
    assert _shape(model)[2] == int(np.floor(2 * max_v / np.cos(0.6) / pitch))

    model.set_params(no_warning=True, det_row_offset=2.0)
    model.auto_set_recon_geometry()
    cos_el = np.cos(elevations)
    z_low, z_high = np.min((-max_v - 2.0) / cos_el), np.max((max_v - 2.0) / cos_el)
    assert float(model.get_params('recon_slice_offset')) == pytest.approx(0.5 * (z_low + z_high))
    assert _shape(model)[2] == int(np.floor((z_high - z_low) / pitch))


# ── build_model keeps what the dicts carry ───────────────────────────────────
def test_build_model_round_trips_hand_set_geometry():
    """Pitch, aspect ratio, recon shape, and slice offset set by hand come back from build_model."""
    cone = _cone(delta_voxel=0.37, voxel_row_aspect=1.25, recon_shape=(10, 12, 8), recon_slice_offset=1.4)
    rebuilt = mbirtorch.build_model(*cone.get_all_params())
    for name in ('delta_voxel', 'voxel_row_aspect', 'recon_slice_offset'):
        assert float(rebuilt.get_params(name)) == pytest.approx(float(cone.get_params(name)))
    assert _shape(rebuilt) == (10, 12, 8)

    parallel = _parallel(delta_voxel=0.8, recon_shape=(40, 44, 24))
    rebuilt = mbirtorch.build_model(*parallel.get_all_params())
    assert float(rebuilt.get_params('delta_voxel')) == pytest.approx(0.8)
    assert _shape(rebuilt) == (40, 44, 24)


def test_build_model_sizes_the_automatic_shape_at_a_supplied_pitch():
    """A supplied pitch without a shape gets the automatic shape rescaled to that pitch, so the
    volume covers the same extent; parallel beam keeps one slice per detector row."""
    for make_model, slices_are_rows in ((_cone, False), (_parallel, True)):
        model = make_model()
        required, optional, regularization = model.get_all_params()
        automatic_shape = _shape(model)
        automatic_pitch = float(optional.pop('delta_voxel'))
        optional.pop('recon_shape')
        optional['delta_voxel'] = 0.74 * automatic_pitch
        built = mbirtorch.build_model(required, optional, regularization)
        expected = [int(np.ceil(n * automatic_pitch / (0.74 * automatic_pitch))) for n in automatic_shape]
        if slices_are_rows:
            expected[2] = automatic_shape[2]
        assert float(built.get_params('delta_voxel')) == pytest.approx(0.74 * automatic_pitch)
        assert _shape(built) == tuple(expected)


# ── apply_calibration follows the detector ───────────────────────────────────
def test_apply_calibration_moves_the_axial_center_with_the_row_offset():
    """A calibrated row offset moves the volume by the change in the automatic center, so a center
    the caller chose keeps its place relative to the detector; shape and pitch are untouched."""
    model = _cone(delta_voxel=0.37, recon_shape=(10, 12, 8), recon_slice_offset=1.4)
    sino = np.zeros((8, 24, 48), dtype=np.float32)
    apply_calibration(model, sino, _row_offset_result(2.0))
    assert float(model.get_params('det_row_offset')) == pytest.approx(2.0)
    assert float(model.get_params('recon_slice_offset')) == pytest.approx(1.4 - 2.0 / model.get_magnification())
    assert float(model.get_params('delta_voxel')) == pytest.approx(0.37)
    assert _shape(model) == (10, 12, 8)

    parallel = _parallel(recon_shape=(40, 44, 24))
    apply_calibration(parallel, sino, _row_offset_result(2.0))
    assert float(parallel.get_params('det_row_offset')) == pytest.approx(2.0)
    assert _shape(parallel) == (40, 44, 24)
