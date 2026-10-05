"""Data-free gates for the vendor loaders: the crop and geometry conversion
math, ported from mbirjax's TestConfigCropUnification.  Loader runs on real
scan data happen on the cluster (the increment-4 end-to-end gate)."""

import json

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


def _volumax_params(nrows=64, nchan=80, tilt=0.0, lean=0.0, object_height=0.0):
    # A cone geometry in VoluMax form (source at -x, detector at +x, vertical rotation axis), with the detector
    # turned by `tilt` radians about its normal and leaning by `lean` radians toward the source.  The recorded
    # object position is `object_height` above the source.
    n, pitch = 20, 0.2
    c, s = np.cos(tilt), np.sin(tilt)
    cl, sl = np.cos(lean), np.sin(lean)
    span_u = pitch * np.array([0., -c, -s])
    span_v = pitch * np.array([-sl * c, s, -cl * c])
    meta = dict(num_views=n, num_det_rows=nrows, num_det_channels=nchan, delta_det_row=pitch,
                delta_det_channel=pitch, angles_deg=np.linspace(0, 360, n, endpoint=False),
                source=np.tile([-100., 0., 0.], (n, 1)), detector=np.tile([50., 1., 2.], (n, 1)),
                object=np.tile([0., 0., object_height], (n, 1)), span_u=np.tile(span_u, (n, 1)), span_v=np.tile(span_v, (n, 1)))
    params = mtp.volumax.volumax_vectors(meta)
    return params, mtp.volumax.compute_geometry(params, verbose=0)


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

    volumax_conv = mtp.volumax.convert_volumax_to_mbirtorch_params
    p, g = _volumax_params()
    _, base = volumax_conv(p, g, verbose=0)
    vp, op = volumax_conv(p, g, crop_pixels_top=10, verbose=0)
    assert vp['sinogram_shape'] == (20, 54, 80)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + (0 - 10) / 2 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'])


def test_volumax_geometry_and_downsampling_remainder():
    # The distances come from the positions along the detector normal.  Block averaging by 2 drops the last row
    # and channel of a 65 x 81 detector, which moves each offset by half a pixel.
    p, g = _volumax_params(nrows=65, nchan=81)
    assert g['source_iso_dist'] == pytest.approx(100.0)
    assert g['source_detector_dist'] == pytest.approx(150.0)
    assert g['det_rotation'] == pytest.approx(0.0, abs=1e-12)
    _, base = mtp.volumax.convert_volumax_to_mbirtorch_params(p, g, verbose=0)
    vp, op = mtp.volumax.convert_volumax_to_mbirtorch_params(p, g, downsample_factor=(2, 2), verbose=0)
    assert vp['sinogram_shape'] == (20, 32, 40)
    assert op['delta_det_row'] == pytest.approx(0.4) and op['delta_det_channel'] == pytest.approx(0.4)
    assert op['det_row_offset'] == pytest.approx(base['det_row_offset'] + 0.5 * base['delta_det_row'])
    assert op['det_channel_offset'] == pytest.approx(base['det_channel_offset'] + 0.5 * base['delta_det_channel'])


def test_volumax_offsets_follow_the_sinogram_rotation():
    # A spot placed at the offsets must stay at the offsets after the sinogram is corrected for the detector
    # rotation: the spot is rotated with correct_det_rotation and its centroid is compared with the rotated offsets.
    tilt = 0.05
    p, g = _volumax_params(nrows=65, nchan=65, tilt=tilt)
    assert abs(g['det_rotation']) == pytest.approx(tilt)
    _, op = mtp.volumax.convert_volumax_to_mbirtorch_params(p, g, verbose=0)
    pitch = g['delta_det_row']
    rows, chans = np.meshgrid(np.arange(65) - 32.0, np.arange(65) - 32.0, indexing='ij')
    row0, chan0 = g['det_row_offset'] / pitch, g['det_channel_offset_metadata'] / pitch
    spot = np.exp(-0.5 * ((rows - row0) ** 2 + (chans - chan0) ** 2) / 2.0 ** 2).astype(np.float32)
    rotated = np.asarray(mtp.correct_det_rotation(spot[None], det_rotation=g['det_rotation']))[0]
    assert (rows * rotated).sum() / rotated.sum() == pytest.approx(op['det_row_offset'] / pitch, abs=0.05)
    assert (chans * rotated).sum() / rotated.sum() == pytest.approx(op['det_channel_offset'] / pitch, abs=0.05)


def test_volumax_rotation_needs_square_pixels():
    p, g = _volumax_params(tilt=0.05)
    with pytest.raises(ValueError, match='square pixels'):
        mtp.volumax.convert_volumax_to_mbirtorch_params(p, g, downsample_factor=(2, 1), verbose=0)


def test_volumax_geometry_with_a_leaning_detector():
    # The distances are measured along the central ray, which is perpendicular to the rotation axis, and the row
    # offset locates the point where the central ray meets the detector.  Neither depends on the height of the
    # recorded object position.
    lean = 0.02
    _, g = _volumax_params(lean=lean)
    _, g_low = _volumax_params(lean=lean, object_height=-200.0)
    for key in ('source_iso_dist', 'source_detector_dist', 'det_row_offset', 'det_channel_offset_metadata'):
        assert g_low[key] == pytest.approx(g[key])
    assert g['source_iso_dist'] == pytest.approx(100.0)
    # The central ray runs along x from the source at (-100, 0, 0).  The detector plane passes through (50, 1, 2).
    normal = np.array([np.cos(lean), 0., -np.sin(lean)])
    sdd = np.dot(np.array([150., 1., 2.]), normal) / normal[0]
    assert g['source_detector_dist'] == pytest.approx(sdd)
    hit = np.array([-100. + sdd, 0., 0.]) - np.array([50., 1., 2.])
    assert g['det_row_offset'] == pytest.approx(np.dot(hit, [-np.sin(lean), 0., -np.cos(lean)]))
    assert g['det_channel_offset_metadata'] == pytest.approx(np.dot(hit, [0., -1., 0.]))


def test_volumax_get_sino_and_model_returns_the_metadata(tmp_path):
    # A scan folder with 6 views of an 8 x 10 detector, written in the VoluMax layout.
    n, nrows, nchan, pitch = 6, 8, 10, 0.2
    metrics = dict(sourcePosition=dict(x=-100., y=0., z=0.), detectorPosition=dict(x=50., y=0., z=0.),
                   objectPosition=dict(x=0., y=0., z=-30.), spanVectorU=dict(x=0., y=-pitch, z=0.),
                   spanVectorV=dict(x=0., y=0., z=-pitch))
    acquisition = dict(name='test', numberOfProjections=n, mode='ACQUISITION_MODE_STOP_GO',
                       tubeParameters=dict(accelerationVoltageInKV=350, sourceCurrentInMicroA=2000),
                       detectorParameters=dict(pixelPitch=dict(horizontal=pitch, vertical=pitch),
                                               imageSize=dict(width=nchan, height=nrows)),
                       startProjectionMetrics=dict(metrics, objectAngle=0.0))
    (tmp_path / 'Projections' / 'Metadata').mkdir(parents=True)
    (tmp_path / 'Projections' / 'Images').mkdir()
    (tmp_path / 'AcquisitionParameters.json').write_text(json.dumps(acquisition))
    for i in range(n):
        view = dict(projectionType='PROJECTION_TYPE_OBJECT', imageIndex=i, imageSize=dict(width=nchan, height=nrows),
                    projectionMetrics=dict(metrics, objectAngle=60.0 * i))
        (tmp_path / 'Projections' / 'Metadata' / f'Projection_{i:05d}.json').write_text(json.dumps(view))
        np.full((nrows, nchan), 0.5, dtype='<f4').tofile(tmp_path / 'Projections' / 'Images' / f'Projection_{i:05d}.float32')

    sino, model, metadata = mtp.volumax.get_sino_and_model(str(tmp_path), subsample_view_factor=2, verbose=0,
                                                           background_offset=None)
    assert sino.shape == (3, nrows, nchan)
    assert np.allclose(sino, np.log(2.0), atol=1e-5)
    assert model.get_params('source_iso_dist') == pytest.approx(100.0)
    assert metadata['acquisition'] == acquisition
    assert list(metadata['projections']['view_ids']) == [0, 2, 4]
    assert np.allclose(metadata['projections']['object_angle'], [0.0, 120.0, 240.0])
    assert metadata['projections']['source_position'].shape == (3, 3)
    assert metadata['geometry']['source_detector_dist'] == pytest.approx(150.0)
