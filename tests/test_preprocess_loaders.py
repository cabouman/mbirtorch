"""Data-free gates for the vendor loaders: the crop/geometry conversion math
(ported from mbirjax's TestConfigCropUnification), golden parity of the NSI
conversion and the pyMBIR beam-hardening linearization on shared inputs, and
the auto-crop -> build_model consistency chain.  Loader runs on real scan
data happen on the cluster (the increment-4 end-to-end gate)."""

import os

import numpy as np
import pytest

import mbirtorch
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


def test_nsi_symmetric_crop_is_byte_identical():
    conv = mtp.nsi.convert_nsi_to_mbirjax_params
    p = _nsi_params()
    _, base = conv(p, (1, 1), 0, 0, 0)
    cb, op = conv(p, (1, 1), 3, 5, 5)                    # symmetric sides + top == bottom
    assert cb['sinogram_shape'] == (20, 64 - 10, 80 - 6)
    for key in ('det_row_offset', 'det_channel_offset', 'recon_slice_offset',
                'delta_det_row', 'delta_det_channel', 'delta_voxel'):
        assert op[key] == pytest.approx(base[key]), key   # crop changes nothing but the shape


def test_nsi_asymmetric_crop_shifts_row_offset():
    conv = mtp.nsi.convert_nsi_to_mbirjax_params
    p = _nsi_params()
    _, base = conv(p, (1, 1), 0, 0, 0)
    cb, op = conv(p, (1, 1), 0, 10, 0)                   # crop_top=10, crop_bottom=0
    assert cb['sinogram_shape'] == (20, 54, 80)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])   # sides symmetric


def test_nsi_offset_shift_uses_raw_pitch_independent_of_downsample():
    conv = mtp.nsi.convert_nsi_to_mbirjax_params
    p = _nsi_params()
    _, base = conv(p, (1, 1), 0, 0, 0)
    raw_pitch = base['delta_det_row']
    expected = base['det_row_offset'] + (0 - 10) / 2 * raw_pitch   # physical shift; independent of downsample
    for ds in [(1, 1), (2, 2)]:
        cb, op = conv(p, ds, 0, 10, 0)
        assert op['det_row_offset'] == pytest.approx(expected), str(ds)
    cb2, op2 = conv(p, (2, 2), 0, 10, 0)
    assert op2['delta_det_row'] == pytest.approx(raw_pitch * 2)    # downsample still scales the pitch
    assert cb2['sinogram_shape'] == (20, 54 // 2, 80 // 2)


def test_zeiss_symmetric_crop_is_byte_identical():
    conv = mtp.zeiss.convert_zeiss_to_mbirjax_params
    p = _zeiss_params()
    _, base, _ = conv(p, (1, 1), 0, 0, 0)
    gp, op, _ = conv(p, (1, 1), 3, 5, 5)
    assert gp['sinogram_shape'] == (20, 54, 74)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])


def test_zeiss_asymmetric_crop_shifts_row_offset():
    conv = mtp.zeiss.convert_zeiss_to_mbirjax_params
    p = _zeiss_params()
    _, base, _ = conv(p, (1, 1), 0, 0, 0)
    gp, op, _ = conv(p, (1, 1), 0, 10, 0)
    assert gp['sinogram_shape'] == (20, 54, 80)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])


def test_zeiss_tct_symmetric_crop_is_byte_identical():
    conv = mtp.zeiss_tct.convert_zeiss_to_mbirjax_params
    p = _tct_params()
    _, base = conv(p, 0, 0, 0)
    tp, op = conv(p, 3, 5, 5)                                # sides=3, top=bottom=5 (symmetric)
    assert tp['sinogram_shape'] == (20, 54, 74)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])


def test_zeiss_tct_asymmetric_crop_shifts_row_offset():
    conv = mtp.zeiss_tct.convert_zeiss_to_mbirjax_params
    p = _tct_params()
    _, base = conv(p, 0, 0, 0)
    tp, op = conv(p, 0, 10, 0)                               # sides=0, top=10, bottom=0 (asymmetric)
    assert tp['sinogram_shape'] == (20, 54, 80)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])


def test_auto_crop_sino_consistent_and_survives_build_model():
    # A cone model's params round-trip through _auto_crop_sino + build_model (see the mbirjax test
    # of the same name for the full rationale).
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ConeBeamModel((12, 80, 100), angles, source_detector_dist=200,
                                    source_iso_dist=100)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, delta_det_row=0.5, delta_det_channel=0.5)
    required, optional, regularization = model.get_all_params()
    optional.pop('recon_shape', None)          # reader flow: let auto size the recon from the crop
    row_offset_before = optional['det_row_offset']
    recon_slice_offset_before = optional['recon_slice_offset']

    obj_row0, obj_col0 = 25, 30
    sino = np.zeros((12, 80, 100), dtype=np.float32)
    sino[:, obj_row0:60, obj_col0:70] = 5.0
    ct, cb, cl, cr = mtp.detect_blank_margins(sino, safety_buffer=5)

    sino, required, optional = mtp.utilities._auto_crop_sino(sino, required, optional, safety_buffer=5)
    assert tuple(required['sinogram_shape']) == sino.shape                  # array <-> geometry
    assert optional['det_row_offset'] == pytest.approx(row_offset_before + (cb - ct) / 2 * 0.5)
    assert optional['recon_slice_offset'] == recon_slice_offset_before      # crop leaves it alone
    nz_rows = np.asarray(np.any(sino != 0, axis=(0, 2)))
    nz_cols = np.asarray(np.any(sino != 0, axis=(0, 1)))
    assert int(np.argmax(nz_rows)) == obj_row0 - ct
    assert int(np.argmax(nz_cols)) == obj_col0 - cl

    # Sentinel: a value the crop never wrote must be overwritten by auto_set_recon_geometry,
    # proving build_model re-derives recon_slice_offset.
    optional['recon_slice_offset'] = 999.0
    rebuilt = mbirtorch.build_model(required, optional, regularization)
    assert tuple(rebuilt.get_params('sinogram_shape')) == sino.shape        # survives build
    reference = mbirtorch.ConeBeamModel(
        **{k: v for k, v in required.items() if k != 'geometry_type'})
    reference.configure_devices(devices=['cpu'])
    reference.set_params(**{k: v for k, v in optional.items() if k != 'recon_slice_offset'})
    reference.auto_set_recon_geometry()
    derived = float(rebuilt.get_params('recon_slice_offset'))
    assert derived != pytest.approx(999.0)                                  # sentinel overwritten
    assert derived == pytest.approx(float(reference.get_params('recon_slice_offset')))
    assert tuple(rebuilt.get_params('recon_shape')) == tuple(reference.get_params('recon_shape'))


@pytest.mark.skipif(not os.path.exists(_npz_path), reason="no preprocess goldens")
def test_nsi_convert_golden_parity():
    golden = np.load(_npz_path)
    cb, op = mtp.nsi.convert_nsi_to_mbirjax_params(_nsi_params(), (2, 2), 3, 5, 5)
    out = np.array([cb['sinogram_shape'][1], cb['sinogram_shape'][2],
                    cb['source_detector_dist'], cb['source_iso_dist'],
                    op['det_row_offset'], op['det_channel_offset'],
                    op['recon_slice_offset'], op['delta_det_row'],
                    op['delta_det_channel'], op['delta_voxel'],
                    op['det_rotation']], dtype=np.float64)
    assert np.allclose(out, golden['nsi_convert'], rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not os.path.exists(_npz_path), reason="no preprocess goldens")
def test_pymbir_bh_correction_golden_parity():
    golden = np.load(_npz_path)
    out = mtp.pymbir.apply_bh_correction(golden['bhcn_sino'].copy(), [0.6, 1.0, 4.0, 20.0])
    poly = mtp.pymbir.find_linearization_fit(0.6, 1.0, 4.0, max_thick=20.0)
    assert np.allclose(poly, golden['bhcn_poly'], rtol=1e-10)
    err = float(np.max(np.abs(out - golden['bhcn_out'])) / np.max(np.abs(golden['bhcn_out'])))
    print(f"pymbir BHCN rel_max = {err:.2e}")
    assert err < 1e-6
