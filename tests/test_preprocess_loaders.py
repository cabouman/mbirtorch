"""Data-free gates for the vendor loaders: the crop/geometry conversion math
(ported from mbirjax's TestConfigCropUnification), golden parity of the NSI
conversion and the pyMBIR beam-hardening linearization on shared inputs.
Loader runs on real scan data happen on the cluster (the increment-4
end-to-end gate)."""

import os

import numpy as np
import pytest

import mbirtorch.preprocess as mtp

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
_npz_path = os.path.join(GOLDEN_DIR, "preprocess_goldens.npz")


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


@pytest.mark.goldens
@pytest.mark.skipif(not os.path.exists(_npz_path), reason="no preprocess goldens")
def test_nsi_convert_golden_parity():
    golden = np.load(_npz_path)
    cb, op = mtp.nsi.convert_nsi_to_mbirtorch_params(_nsi_params(), (2, 2), 3, 5, 5)
    out = np.array([cb['sinogram_shape'][1], cb['sinogram_shape'][2],
                    cb['source_detector_dist'], cb['source_iso_dist'],
                    op['det_row_offset'], op['det_channel_offset'],
                    op['recon_slice_offset'], op['delta_det_row'],
                    op['delta_det_channel'], op['delta_voxel'],
                    op['det_rotation']], dtype=np.float64)
    assert np.allclose(out, golden['nsi_convert'], rtol=1e-10, atol=1e-12)


@pytest.mark.goldens
@pytest.mark.skipif(not os.path.exists(_npz_path), reason="no preprocess goldens")
def test_pymbir_bh_correction_golden_parity():
    golden = np.load(_npz_path)
    out = mtp.pymbir.apply_bh_correction(golden['bhcn_sino'].copy(), [0.6, 1.0, 4.0, 20.0])
    poly = mtp.pymbir.find_linearization_fit(0.6, 1.0, 4.0, max_thick=20.0)
    assert np.allclose(poly, golden['bhcn_poly'], rtol=1e-10)
    err = float(np.max(np.abs(out - golden['bhcn_out'])) / np.max(np.abs(golden['bhcn_out'])))
    print(f"pymbir BHCN rel_max = {err:.2e}")
    assert err < 1e-6
