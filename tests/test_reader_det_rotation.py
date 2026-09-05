"""The ``det_rotation`` argument of the Zeiss readers reaches the sinogram.

Each reader passes ``det_rotation`` to ``scan_to_sino``, which rotates every view as the sinogram
is computed.  The tests here replace the reader's ``load_scans_and_params`` with synthetic arrays,
so they read no scan file.  Each test then checks that the sinogram the reader computes at a
nonzero rotation equals the rotation that ``correct_det_rotation`` applies to the sinogram computed
at zero rotation, and that the default argument computes the sinogram at zero rotation.
"""

import numpy as np
import pytest

from mbirtorch.preprocess import zeiss, zeiss_tct
from mbirtorch.preprocess.utilities import correct_det_rotation

DET_ROTATION = 0.02                                 # radians, about 1.1 degrees
NUM_VIEWS, NUM_ROWS, NUM_CHANNELS = 12, 24, 32
BLANK_LEVEL = 1000.0

# The attenuating part of the object scan, in detector rows and channels.  It is kept away from the
# detector edges for two reasons.  The rotation then samples only rows and channels the detector
# has, and the background offset correction of the translation reader, which has no option to skip
# it, sees an edge region of zeros and estimates an offset of exactly zero.
OBJECT_ROWS = (10, 20)
OBJECT_CHANNELS = (10, 22)


def _synthetic_scans():
    """An object scan with structure, a blank scan at a constant level, and a dark scan of zeros."""
    rows = np.arange(NUM_ROWS)[None, :, None]
    channels = np.arange(NUM_CHANNELS)[None, None, :]
    views = np.arange(NUM_VIEWS)[:, None, None]
    inside = ((rows >= OBJECT_ROWS[0]) & (rows < OBJECT_ROWS[1])
              & (channels >= OBJECT_CHANNELS[0]) & (channels < OBJECT_CHANNELS[1]))
    # A transmission that varies along both detector axes and with the view, so that a rotation
    # changes the sinogram everywhere in the object region rather than only at its boundary.
    transmission = 0.3 + 0.4 * (channels - OBJECT_CHANNELS[0]) / 12.0 + 0.2 * np.sin(0.7 * rows + 0.3 * views)
    obj_scan = np.where(inside, BLANK_LEVEL * np.clip(transmission, 0.05, 0.95), BLANK_LEVEL)
    blank_scan = np.full((2, NUM_ROWS, NUM_CHANNELS), BLANK_LEVEL, dtype=np.float32)
    dark_scan = np.zeros((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS), dtype=np.float32)
    return obj_scan.astype(np.float32), blank_scan, dark_scan


def _zeiss_scan_params():
    """The geometry dict the Zeiss Versa reader reads, modeled on ``_zeiss_params`` in
    ``test_preprocess_loaders.py``, with the per-view shift arrays that ``correct_sino_shifts``
    reads.  The shifts are zero, so that step leaves the sinogram unchanged."""
    return dict(source_iso_dist=50.0, iso_det_dist=150.0, source_iso_dist_unit='mm', iso_det_dist_unit='mm',
                delta_det_channel=0.15, delta_det_row=0.15, delta_det_channel_unit='mm', delta_det_row_unit='mm',
                iso_pixel_pitch=0.05, iso_pixel_pitch_unit='mm', opt_mag=1.0,
                num_views=NUM_VIEWS, num_det_rows=NUM_ROWS, num_det_channels=NUM_CHANNELS,
                angles=np.linspace(0, 360, NUM_VIEWS, endpoint=False), angle_unit='deg',
                det_row_offset=0.0, det_channel_offset=0.0, scanner_type='versa',
                x_shifts=np.zeros(NUM_VIEWS), y_shifts=np.zeros(NUM_VIEWS))


def _tct_scan_params():
    """The geometry dict the Zeiss translation reader reads, modeled on ``_tct_params`` in
    ``test_preprocess_loaders.py``, with one object position per view."""
    return dict(source_iso_dist=50.0, iso_det_dist=150.0, source_iso_dist_unit='mm', iso_det_dist_unit='mm',
                delta_det_channel=0.1, delta_det_row=0.1, delta_det_channel_unit='mm', delta_det_row_unit='mm',
                iso_pixel_pitch=0.05, iso_pixel_pitch_unit='mm', opt_mag=1.0,
                num_views=NUM_VIEWS, num_det_rows=NUM_ROWS, num_det_channels=NUM_CHANNELS,
                object_x_positions=np.linspace(-0.2, 0.2, NUM_VIEWS), object_x_position_unit='mm',
                object_y_positions=np.zeros(NUM_VIEWS), object_y_position_unit='mm',
                object_z_positions=np.zeros(NUM_VIEWS), object_z_position_unit='mm',
                det_row_offset=0.0, det_channel_offset=0.0)


def _patch_loader(monkeypatch, module, scan_params):
    """Replace one reader's ``load_scans_and_params`` with the synthetic scans and geometry."""
    scans = _synthetic_scans()

    def fake_loader(*args, **kwargs):
        obj_scan, blank_scan, dark_scan = scans
        return obj_scan.copy(), blank_scan.copy(), dark_scan.copy(), dict(scan_params)

    monkeypatch.setattr(module, 'load_scans_and_params', fake_loader)


def _check_rotation_reaches_the_sinogram(reader):
    """Run one reader at three rotations and compare the sinograms.

    ``reader`` takes a rotation in radians, or None for the default argument, and returns the
    sinogram.
    """
    unrotated = reader(0.0)
    rotated = reader(DET_ROTATION)
    default = reader(None)
    assert rotated.shape == unrotated.shape == (NUM_VIEWS, NUM_ROWS, NUM_CHANNELS)
    # A rotation of 0.02 radians moves the edge channels of this detector by 0.3 pixels, which is
    # enough to change the sinogram well beyond the comparison tolerance.
    assert np.max(np.abs(rotated - unrotated)) > 1e-3
    expected = correct_det_rotation(unrotated, DET_ROTATION)
    assert np.allclose(rotated, expected, rtol=1e-4, atol=1e-5)
    assert np.array_equal(default, unrotated)


def test_zeiss_reader_applies_det_rotation(monkeypatch):
    """The Zeiss Versa reader rotates every view by the ``det_rotation`` it is given."""
    _patch_loader(monkeypatch, zeiss, _zeiss_scan_params())

    def reader(det_rotation):
        kwargs = {} if det_rotation is None else {'det_rotation': det_rotation}
        sino, _ = zeiss.get_sino_and_model('unused', zinger_correction=False, bg_option=None,
                                           verbose=0, **kwargs)
        return np.asarray(sino)

    _check_rotation_reaches_the_sinogram(reader)


def test_zeiss_tct_reader_applies_det_rotation(monkeypatch):
    """The Zeiss translation reader rotates every view by the ``det_rotation`` it is given.  Its
    background offset correction has no option to skip it, and the object scan here leaves the
    detector edges at the blank level, so that correction subtracts exactly zero in every case."""
    _patch_loader(monkeypatch, zeiss_tct, _tct_scan_params())

    def reader(det_rotation):
        kwargs = {} if det_rotation is None else {'det_rotation': det_rotation}
        sino, _, _ = zeiss_tct.get_sino_and_model('unused', verbose=0, **kwargs)
        return np.asarray(sino)

    _check_rotation_reaches_the_sinogram(reader)
