"""Data-free gates for the vendor loaders: the crop and geometry conversion
math, ported from mbirjax's TestConfigCropUnification.  Loader runs on real
scan data happen on the cluster (the increment-4 end-to-end gate)."""

import numpy as np
import pytest

import mbirtorch.preprocess as mtp


def _nsi_params():
    # A clean orthonormal cone geometry (source at -y, detector at +y, rows along x, cols along -z).
    sid, sdd = 100.0, 200.0
    nrows, nchan, dr, dc = 64, 80, 0.2, 0.2
    r_n = np.array([0., 1., 0.]); r_h = np.array([1., 0., 0.]); r_a = np.array([0., 0., -1.])
    r_s = np.array([0., -sid, 0.])
    r_v = np.cross(r_n, r_h)
    r_r = np.array([0., sdd - sid, 0.]) - (nchan / 2.0) * dc * r_h - (nrows / 2.0) * dr * r_v
    return dict(r_a=r_a, r_n=r_n, r_h=r_h, r_s=r_s, r_r=r_r,
                delta_det_channel=dc, delta_det_row=dr,
                num_det_channels=nchan, num_det_rows=nrows,
                angles=np.linspace(0, 2 * np.pi, 20, endpoint=False))


def _zeiss_params():
    return dict(source_iso_dist=50.0, iso_det_dist=150.0, source_iso_dist_unit='mm', iso_det_dist_unit='mm',
                delta_det_channel=0.15, delta_det_row=0.15, delta_det_channel_unit='mm', delta_det_row_unit='mm',
                iso_pixel_pitch=0.05, iso_pixel_pitch_unit='mm', opt_mag=None,
                num_det_rows=64, num_det_channels=80,
                angles=np.linspace(0, 360, 20, endpoint=False), angle_unit='deg',
                det_row_offset=3.0, det_channel_offset=2.0, scanner_type='versa')


def _tct_params():
    n = 20
    return dict(source_iso_dist=50.0, iso_det_dist=150.0, source_iso_dist_unit='mm', iso_det_dist_unit='mm',
                delta_det_channel=0.1, delta_det_row=0.1, delta_det_channel_unit='mm', delta_det_row_unit='mm',
                iso_pixel_pitch=0.05, iso_pixel_pitch_unit='mm', opt_mag=None,
                num_det_rows=64, num_det_channels=80,
                object_x_positions=np.linspace(-5, 5, n), object_x_position_unit='mm',
                object_y_positions=np.zeros(n), object_y_position_unit='mm',
                object_z_positions=np.linspace(-2, 2, n), object_z_position_unit='mm',
                det_row_offset=3.0, det_channel_offset=2.0)


def test_asymmetric_crop_shifts_row_offset_for_every_vendor():
    # crop_top=10, crop_bottom=0, sides=0: the row offset moves by half the
    # difference of the two row crops times the row pitch, and the channel
    # offset does not move, in all three vendor conversions.
    nsi_conv = mtp.nsi.convert_nsi_to_mbirtorch_params
    p = _nsi_params()
    _, base = nsi_conv(p, (1, 1), 0, 0, 0)
    cb, op = nsi_conv(p, (1, 1), 0, 10, 0)
    assert cb['sinogram_shape'] == (20, 54, 80)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])   # sides symmetric

    zeiss_conv = mtp.zeiss.convert_zeiss_to_mbirtorch_params
    p = _zeiss_params()
    _, base, _ = zeiss_conv(p, (1, 1), 0, 0, 0)
    gp, op, _ = zeiss_conv(p, (1, 1), 0, 10, 0)
    assert gp['sinogram_shape'] == (20, 54, 80)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])

    tct_conv = mtp.zeiss_tct.convert_zeiss_to_mbirtorch_params
    p = _tct_params()
    _, base = tct_conv(p, 0, 0, 0)
    tp, op = tct_conv(p, 0, 10, 0)
    assert tp['sinogram_shape'] == (20, 54, 80)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])
