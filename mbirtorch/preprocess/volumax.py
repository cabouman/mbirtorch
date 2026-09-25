import concurrent.futures as cf
import json
import time
from pathlib import Path
import numpy as np
import mbirtorch
import mbirtorch.preprocess as mtp


def get_sino_and_model(dataset_dir, downsample_factor=(1, 1), subsample_view_factor=1, crop_pixels_sides=0,
                       crop_pixels_top=0, crop_pixels_bottom=0, auto_crop=False, verbose=1, min_transmission=1e-4, background_offset='per_view', sinogram_path=None, num_workers=8,
                       batch_size=90, return_geometry=False):
    """
    Load a VoluMax scan, compute its sinogram and return a ready-to-reconstruct ``ConeBeamModel``.

    Geometry: distances, ``det_row_offset`` and the detector tilt come from the metadata (positions and span
    vectors); ``det_channel_offset`` and the rotation direction are measured from the projections, because the
    export does not locate the rotation axis.

    Args:
        dataset_dir (str): scan root, ``proj`` folder or acquisition folder.
        downsample_factor (tuple[int, int]): detector (row, channel) block averaging. (2, 2) -> 1512 x 1512.
        subsample_view_factor (int): keep every n-th view.
        crop_pixels_sides, crop_pixels_top, crop_pixels_bottom (int): raw pixels cropped before down-sampling;
            an asymmetric top/bottom crop shifts ``det_row_offset`` accordingly (``apply_config_crop``).
        auto_crop (bool): remove blank sinogram margins after the sinogram is computed.
        verbose (int): verbosity.
        min_transmission (float): clip transmission below this before -log.
        background_offset (str | None): ``correct_background_offset`` option ('per_view', 'global' or None).
        sinogram_path (str | None): precomputed -log sinogram (view, row, channel) to use instead of the raw
            transmission, e.g. a LEAP scatter/BH corrected stack made with the same binning / views / crop.
        num_workers (int): reader threads.  batch_size (int): views per device batch in ``scan_to_sino``.
        return_geometry (bool): also return the full-resolution geometry / calibration dict.

    Returns:
        tuple: ``(sino, model)`` (or ``(sino, model, geometry)``); ``sino`` is (num_views, rows, channels) float32.

    Example:
        from mbirtorch.preprocess import volumax
        sino, model = volumax.get_sino_and_model(dataset_dir, downsample_factor=(2, 2))
        weights = mbirtorch.gen_weights(sino, weight_type='transmission_root')
        recon, recon_dict = model.recon(sino, weights=weights)
    """
    sino, required, optional, geom = _compute_sino_and_params(
        dataset_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor,
        crop_pixels_sides=crop_pixels_sides, crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom,
        verbose=verbose, min_transmission=min_transmission, background_offset=background_offset, sinogram_path=sinogram_path, num_workers=num_workers, batch_size=batch_size)
    sino, model = mtp.finalize_model(sino, required, optional, auto_crop=auto_crop)
    if verbose > 0:
        print('\n########## mbirtorch model parameters')
        model.print_params()
    return (sino, model, geom) if return_geometry else (sino, model)


def _compute_sino_and_params(dataset_dir, downsample_factor=(1, 1), subsample_view_factor=1, crop_pixels_sides=0,
                             crop_pixels_top=0, crop_pixels_bottom=0, verbose=1, min_transmission=1e-4,
                             background_offset='per_view', sinogram_path=None, num_workers=8, batch_size=90):
    """Load the scan, calibrate the geometry, compute the sinogram; returns ``(sino, required, optional, geom)``."""
    if verbose > 0:
        print('\n########## Loading VoluMax transmission images and geometry')
    load_scans = sinogram_path is None
    obj_scan, params = load_scans_and_params(dataset_dir, subsample_view_factor=subsample_view_factor,
                                             downsample_factor=downsample_factor, crop_pixels_sides=crop_pixels_sides,
                                             crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom,
                                             min_transmission=min_transmission, num_workers=num_workers, verbose=verbose,
                                             load_scans=load_scans)
    geom = compute_geometry(params, num_workers=num_workers, verbose=verbose)
    cone_beam_params, optional_params = convert_volumax_to_mbirtorch_params(
        params, geom, downsample_factor=downsample_factor, crop_pixels_sides=crop_pixels_sides,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom)
    det_rotation = optional_params.pop('det_rotation')

    if sinogram_path is None:
        if verbose > 0:
            print(f'\n########## Computing sinogram: -log(I/I0) with unit blank / zero dark, detector rotation '
                  f'{det_rotation:+.3e} rad, fused on device')
        blank = np.ones((1,) + obj_scan.shape[1:], dtype=np.float32)
        dark = np.zeros((1,) + obj_scan.shape[1:], dtype=np.float32)
        sino = mtp.scan_to_sino(obj_scan, blank, dark, (), downsample_factor=(1, 1), det_rotation=det_rotation,
                                batch_size=batch_size)
        del obj_scan
    else:
        if verbose > 0:
            print(f'\n########## Loading precomputed sinogram {sinogram_path} (already -log; e.g. LEAP BH corrected)')
        sino = np.ascontiguousarray(np.load(sinogram_path, mmap_mode='r'), dtype=np.float32)
        if det_rotation != 0.0:
            sino = mtp.correct_det_rotation(sino, det_rotation=det_rotation)
    if tuple(sino.shape) != tuple(cone_beam_params['sinogram_shape']):
        raise ValueError(f'sinogram shape {sino.shape} does not match the geometry {cone_beam_params["sinogram_shape"]}; '
                         'match subsample_view_factor / downsample_factor / crops to the data')
    if background_offset not in (None, 'none'):
        if verbose > 0:
            print(f'\n########## Background offset correction ({background_offset}): air is ~1.03 in the VoluMax export')
        sino = mtp.correct_background_offset(sino, option=background_offset)
    if verbose > 0:
        print(f'sinogram shape = {sino.shape}, min {sino.min():.4f}, max {sino.max():.4f}')
    return sino, cone_beam_params, optional_params, geom


# ---------------------------------------------------------------------------------------------------------------------
# Geometry assembly
# ---------------------------------------------------------------------------------------------------------------------
def compute_geometry(params, *, calibration_row_band=None, calibration_view_step=2, num_workers=8, verbose=1):
    """
    Full-resolution geometry in mm.

    From the metadata (positions and span vectors): ``source_iso_dist``, ``source_detector_dist``,
    ``magnification``, ``det_row_offset`` (row component of the source's perpendicular foot relative to the
    panel centre) and ``det_rotation`` (tilt between the projected rotation axis and the detector columns).
    From the projections (opposite-ray consistency on the source-height rows): ``det_channel_offset`` (the
    column on which the rotation axis projects) and ``angle_sign`` (model angle = angle_sign * objectAngle).

    Args:
        params (dict): from :func:`load_scans_and_params`.
        calibration_row_band (tuple): full-res rows averaged for the calibration (default source-height row +/- 40).
        calibration_view_step (int): use every k-th view in the calibration.
    """
    r_a, r_n, r_h, r_v, r_s, r_d = (params[k] for k in ('r_a', 'r_n', 'r_h', 'r_v', 'r_s', 'r_d'))
    pitch_c, pitch_r = params['delta_det_channel'], params['delta_det_row']
    n_rows, n_cols = params['num_det_rows'], params['num_det_channels']
    # NSI's helpers take the first-pixel centre r_r; the panel centre r_d is passed with a 1 x 1 panel, so the
    # half-panel shifts back to the centre are exactly zero.
    sdd, sid, mag, det_rotation = (float(x) for x in mtp.nsi.calc_source_detector_params(r_a, r_n, r_h, r_s, r_d))
    # det_row_offset: row component of the source's perpendicular foot relative to the panel centre (positive
    # below it). det_channel_offset_meta takes objectPosition as a point on the rotation axis; on the VoluMax
    # export that is about 0.5 mm off the axis, so it is only an uncalibrated reference.
    # calc_row_channel_params only normalises r_h, so r_h is first made orthogonal to r_n.
    r_h_in_panel = r_h - mtp.project_vector_to_vector(r_h, mtp.unit_vector(r_n))
    det_channel_offset_meta, det_row_offset = (float(x) for x in mtp.nsi.calc_row_channel_params(
        r_a, r_n, r_h_in_panel, r_s, r_d, pitch_c, pitch_r, 1, 1, mag))
    center_col, center_row = (n_cols - 1) / 2.0, (n_rows - 1) / 2.0
    iso_row_full = center_row + det_row_offset / pitch_r
    geom = dict(source_detector_dist=sdd, source_iso_dist=sid, magnification=mag,
                delta_det_channel=pitch_c, delta_det_row=pitch_r, num_det_rows=n_rows, num_det_channels=n_cols,
                det_row_offset=det_row_offset, iso_row_full=iso_row_full, det_rotation=float(det_rotation),
                det_channel_offset_metadata=det_channel_offset_meta)
    if verbose > 0:
        print('\n########## VoluMax -> mbirtorch geometry (metadata)')
        print(f'   source_iso_dist {sid:.4f} mm, source_detector_dist {sdd:.4f} mm, magnification {mag:.5f}, '
              f'voxel at iso {pitch_c / mag:.6f} mm')
        print(f'   det_row_offset {det_row_offset:+.4f} mm (source-height row {iso_row_full:.2f} of {n_rows})')
        print(f'   det_channel_offset from the metadata alone {det_channel_offset_meta:+.4f} mm (objectPosition taken as the axis; reference only)')
        print(f'   det_rotation from span vectors {det_rotation:+.3e} rad = {np.rad2deg(det_rotation):+.4f} deg; '
              f'r_a={np.round(r_a, 4)}, r_n={np.round(r_n, 5)}, r_h={np.round(r_h, 5)}, r_v={np.round(r_v, 5)}')
        print(f'   position jitter over the views (max, mm): {params["position_jitter_mm"]}')

    acq_dir = params['acq_dir']
    angles_all = params.get('angles_deg_all')
    if angles_all is None:
        angles_all = read_volumax_metadata(acq_dir, verbose=0)['angles_deg']
    band = calibration_row_band or (max(0, int(round(iso_row_full)) - 40), min(n_rows, int(round(iso_row_full)) + 40))
    if verbose > 0:
        print('\n########## Data calibration: rotation direction and centre column (opposite-ray consistency)')
    calibration = estimate_rotation_sign_and_cor(acq_dir, angles_all, geom, band, view_step=calibration_view_step,
                                                 num_workers=num_workers, verbose=verbose)
    geom['calibration'] = calibration
    geom['angle_sign'] = float(calibration['angle_sign'])
    geom['det_channel_offset'] = float(calibration['offset_mm'])
    geom['iso_col_full'] = center_col + geom['det_channel_offset'] / pitch_c
    if verbose > 0:
        print_geometry_table(geom)
    return geom


def print_geometry_table(geom):
    pitch_c, pitch_r = geom['delta_det_channel'], geom['delta_det_row']
    print('\n########## Parameters handed to mbirtorch (full-resolution detector frame, mm)')
    print(f'   {"quantity":28s} {"value":>14s}   note')
    print(f'   {"source_iso_dist":28s} {geom["source_iso_dist"]:14.4f}   along the detector normal, metadata')
    print(f'   {"source_detector_dist":28s} {geom["source_detector_dist"]:14.4f}   along the detector normal, metadata')
    print(f'   {"magnification":28s} {geom["magnification"]:14.5f}')
    print(f'   {"det_channel_offset [meta]":28s} {geom["det_channel_offset_metadata"]:+14.4f}   {geom["det_channel_offset_metadata"] / pitch_c:+.2f} px, reference only (uncalibrated)')
    print(f'   {"det_channel_offset [data]":28s} {geom["det_channel_offset"]:+14.4f}   {geom["det_channel_offset"] / pitch_c:+.2f} px, from the projections <-- used')
    print(f'   {"det_row_offset":28s} {geom["det_row_offset"]:+14.4f}   {geom["det_row_offset"] / pitch_r:+.2f} px, metadata')
    print(f'   {"det_rotation [rad]":28s} {geom["det_rotation"]:+14.3e}   {np.rad2deg(geom["det_rotation"]):+.4f} deg, from spanVectorU/V')
    print(f'   {"angle_sign":28s} {geom["angle_sign"]:+14.0f}   angles = angle_sign * objectAngle, from the projections')
    print(f'   {"recon_slice_offset":28s} {-geom["det_row_offset"] / geom["magnification"]:+14.4f}   = -det_row_offset / magnification')


def rotate_offsets_for_det_rotation(det_channel_offset, det_row_offset, det_rotation):
    """
    Offsets (mm, relative to the panel centre) after the projections are rotated by ``det_rotation`` about the
    panel centre, as ``mbirtorch.preprocess.scan_to_sino`` / ``correct_det_rotation`` do. A feature at
    (row, col) relative to the centre moves to (cos t*row - sin t*col, sin t*row + cos t*col); the centre
    itself stays put. Exact for any angle (verified against the mbirtorch rotation kernel).
    """
    c, s = np.cos(det_rotation), np.sin(det_rotation)
    return (float(s * det_row_offset + c * det_channel_offset),      # channel offset in the rotated frame
            float(c * det_row_offset - s * det_channel_offset))      # row offset in the rotated frame


def convert_volumax_to_mbirtorch_params(params, geom, downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0,
                                        crop_pixels_bottom=0):
    """
    Geometry -> ``(cone_beam_params, optional_params)`` for ``mbirtorch.build_model``, adjusted for crop
    (first, in raw pixels), for the detector rotation applied to the sinogram (offsets rotated exactly into the
    corrected frame) and for down-sampling.
    """
    num_det_rows, num_det_channels = params['num_det_rows'], params['num_det_channels']
    delta_det_row, delta_det_channel = geom['delta_det_row'], geom['delta_det_channel']
    det_row_offset, det_channel_offset = geom['det_row_offset'], geom['det_channel_offset']
    num_det_rows, num_det_channels, det_row_offset, det_channel_offset = mtp.apply_config_crop(
        num_det_rows, num_det_channels, det_row_offset, det_channel_offset, delta_det_row, delta_det_channel,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom, crop_pixels_sides=crop_pixels_sides)
    det_rotation = float(geom['det_rotation'])
    if det_rotation != 0.0:
        # the sinogram is rotated by det_rotation about the (cropped) panel centre, so the offsets must describe
        # the rotated frame: exact rotation of the (channel, row) offset vector
        ch0, row0 = det_channel_offset, det_row_offset
        det_channel_offset, det_row_offset = rotate_offsets_for_det_rotation(ch0, row0, det_rotation)
        print(f'   offsets rotated by det_rotation {det_rotation:+.3e} rad: channel {ch0:+.4f} -> {det_channel_offset:+.4f} mm '
              f'({(det_channel_offset - ch0) / delta_det_channel:+.3f} px), row {row0:+.4f} -> {det_row_offset:+.4f} mm '
              f'({(det_row_offset - row0) / delta_det_row:+.3f} px)')
    num_det_rows //= downsample_factor[0]
    num_det_channels //= downsample_factor[1]
    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    angles = np.ascontiguousarray(geom['angle_sign'] * np.unwrap(np.deg2rad(params['angles_deg'])), dtype=np.float32)
    sid, sdd = geom['source_iso_dist'], geom['source_detector_dist']
    cone_beam_params = dict(sinogram_shape=(len(angles), int(num_det_rows), int(num_det_channels)), angles=angles,
                            source_detector_dist=float(sdd), source_iso_dist=float(sid),
                            geometry_type=str(mbirtorch.ConeBeamModel))
    optional_params = dict(delta_det_channel=float(delta_det_channel), delta_det_row=float(delta_det_row),
                           delta_voxel=float(delta_det_channel * sid / sdd),
                           det_channel_offset=float(det_channel_offset), det_row_offset=float(det_row_offset),
                           recon_slice_offset=float(-det_row_offset / geom['magnification']),
                           det_rotation=det_rotation,                          # popped before set_params
                           alu_unit='mm', alu_value=1.0)
    return cone_beam_params, optional_params


# ---------------------------------------------------------------------------------------------------------------------
# Locating and reading the export
# ---------------------------------------------------------------------------------------------------------------------
def find_acquisition_dir(dataset_dir):
    """Return the folder holding ``AcquisitionParameters.json`` (accepts scan root, ``proj`` or acquisition folder)."""
    dataset_dir = Path(dataset_dir).expanduser()
    if (dataset_dir / 'AcquisitionParameters.json').is_file():
        return dataset_dir
    candidates = sorted(dataset_dir.glob('*/AcquisitionParameters.json'))
    candidates += sorted(dataset_dir.glob('proj/*/AcquisitionParameters.json'))
    if not candidates:
        raise FileNotFoundError(f'No AcquisitionParameters.json under {dataset_dir}')
    if len(candidates) > 1:
        print(f'Warning: {len(candidates)} acquisitions found; using {candidates[0].parent}')
    return candidates[0].parent


def _vec3(d):
    return np.array([d['x'], d['y'], d['z']], dtype=np.float64)


def read_volumax_metadata(acq_dir, verbose=1):
    """Read AcquisitionParameters.json and every Projections/Metadata/Projection_%05d.json.

    Returns:
        dict with ``acq`` (the scan-level json), ``angles_deg`` (N,), ``source``/``detector``/``object`` (N,3),
        ``span_u``/``span_v`` (N,3) and the detector size / pitch.
    """
    acq_dir = Path(acq_dir)
    with open(acq_dir / 'AcquisitionParameters.json') as f:
        acq = json.load(f)
    num_views = int(acq['numberOfProjections'])
    det = acq['detectorParameters']
    meta_dir = acq_dir / 'Projections' / 'Metadata'
    keys = ['sourcePosition', 'detectorPosition', 'objectPosition', 'spanVectorU', 'spanVectorV']
    vec = {k: np.full((num_views, 3), np.nan) for k in keys}
    angles_deg = np.full(num_views, np.nan)
    missing = []
    for i in range(num_views):
        fn = meta_dir / f'Projection_{i:05d}.json'
        if not fn.is_file():
            missing.append(i)
            continue
        with open(fn) as f:
            pm = json.load(f)['projectionMetrics']
        angles_deg[i] = pm['objectAngle']
        for k in keys:
            vec[k][i] = _vec3(pm[k])
    if missing:
        good = np.setdiff1d(np.arange(num_views), missing)
        for k in keys:
            for c in range(3):
                vec[k][missing, c] = np.interp(missing, good, vec[k][good, c])
        ang = np.unwrap(np.deg2rad(angles_deg[good]))
        angles_deg[missing] = np.rad2deg(np.interp(missing, good, ang)) % 360.0
        print(f'Warning: {len(missing)} metadata files missing, interpolated (e.g. views {missing[:5]})')
    info = dict(
        acq=acq, acq_dir=acq_dir, num_views=num_views,
        num_det_rows=int(det['imageSize']['height']), num_det_channels=int(det['imageSize']['width']),
        delta_det_channel=float(det['pixelPitch']['horizontal']), delta_det_row=float(det['pixelPitch']['vertical']),
        angles_deg=angles_deg, source=vec['sourcePosition'], detector=vec['detectorPosition'],
        object=vec['objectPosition'], span_u=vec['spanVectorU'], span_v=vec['spanVectorV'],
    )
    if verbose > 0:
        step = np.diff(np.unwrap(np.deg2rad(angles_deg)))
        tube = acq.get('tubeParameters', {})
        print(f'VoluMax acquisition {acq_dir.name}')
        print(f'   {num_views} views, angle step {np.rad2deg(step.mean()):.5f} deg, span {np.rad2deg(step.sum()):.3f} deg, '
              f'mode {acq.get("mode", "?")}')
        print(f'   detector {info["num_det_rows"]} rows x {info["num_det_channels"]} channels, pitch '
              f'{info["delta_det_row"]} x {info["delta_det_channel"]} mm, bit depth {det.get("bitDepth")}, '
              f'binning {det.get("binning")}')
        print(f'   tube {tube.get("accelerationVoltageInKV")} kV, {tube.get("sourceCurrentInMicroA", 0):.0f} uA, '
              f'{det.get("integrationTimeInMs")} ms; filter {acq.get("filterChangerParameters", {}).get("material")}')
    return info


def projection_path(acq_dir, view_idx):
    return Path(acq_dir) / 'Projections' / 'Images' / f'Projection_{view_idx:05d}.float32'


def read_projection(acq_dir, view_idx, num_rows, num_cols, row_slice=None, col_slice=None, downsample_factor=(1, 1),
                    min_transmission=1e-4):
    """One transmission image, cropped and block-averaged, as float32 (rows, cols)."""
    fn = projection_path(acq_dir, view_idx)
    expected = num_rows * num_cols * 4
    if fn.stat().st_size != expected:
        raise ValueError(f'{fn} has {fn.stat().st_size} bytes, expected {expected} for {num_rows}x{num_cols} float32')
    mm = np.memmap(fn, dtype='<f4', mode='r', shape=(num_rows, num_cols))
    row_slice = row_slice if row_slice is not None else slice(0, num_rows)
    col_slice = col_slice if col_slice is not None else slice(0, num_cols)
    img = np.array(mm[row_slice, col_slice], dtype=np.float32)
    del mm
    dr, dc = int(downsample_factor[0]), int(downsample_factor[1])
    if dr > 1 or dc > 1:
        nr, nc = img.shape[0] // dr, img.shape[1] // dc
        img = img[:nr * dr, :nc * dc].reshape(nr, dr, nc, dc).mean(axis=(1, 3), dtype=np.float32)
    if min_transmission is not None:
        np.clip(img, min_transmission, None, out=img)
    return img


def load_scans_and_params(dataset_dir, view_id_start=0, view_id_end=None, subsample_view_factor=1,
                          downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0,
                          min_transmission=1e-4, num_workers=8, verbose=1, load_scans=True):
    """
    Load the VoluMax transmission images and the geometry vectors.

    The images are already normalised transmission (no blank / dark scans exist), and because a
    full-resolution stack is 73 GB the crop and the detector down-sampling are applied while reading.

    Args:
        dataset_dir (str): scan root, ``proj`` folder or acquisition folder.
        view_id_start, view_id_end (int): first view and one-past-last view (default all views).
        subsample_view_factor (int): keep every n-th view.
        downsample_factor (tuple[int, int]): block-average factor for (rows, channels).
        crop_pixels_sides, crop_pixels_top, crop_pixels_bottom (int): pixels removed from the raw images
            (full-resolution pixels) before down-sampling.
        min_transmission (float): transmission is clipped below this before -log (VoluMax clips at 1e-12).
        num_workers (int): reader threads.
        verbose (int): verbosity.
        load_scans (bool): False returns ``obj_scan=None`` (geometry only).

    Returns:
        tuple: ``(obj_scan, volumax_params)`` with ``obj_scan`` the transmission stack
        (num_views, rows, cols) after crop and down-sampling, and ``volumax_params`` the geometry vectors
        and detector numbers (see :func:`volumax_vectors`).
    """
    acq_dir = find_acquisition_dir(dataset_dir)
    meta = read_volumax_metadata(acq_dir, verbose=verbose)
    num_rows, num_cols = meta['num_det_rows'], meta['num_det_channels']
    if view_id_end is None:
        view_id_end = meta['num_views']
    view_ids = np.arange(view_id_start, view_id_end, subsample_view_factor, dtype=np.int64)

    row_slice = slice(int(crop_pixels_top), num_rows - int(crop_pixels_bottom))
    col_slice = slice(int(crop_pixels_sides), num_cols - int(crop_pixels_sides))
    params = volumax_vectors(meta, view_ids)
    params.update(dict(
        acq_dir=acq_dir, view_ids=view_ids,
        crop_pixels_sides=int(crop_pixels_sides), crop_pixels_top=int(crop_pixels_top),
        crop_pixels_bottom=int(crop_pixels_bottom), downsample_factor=(int(downsample_factor[0]), int(downsample_factor[1])),
        min_transmission=min_transmission,
    ))

    obj_scan = None
    if load_scans:
        missing = [int(i) for i in view_ids if not projection_path(acq_dir, i).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} projection files missing, e.g. views {missing[:5]}')
        first = read_projection(acq_dir, view_ids[0], num_rows, num_cols, row_slice, col_slice, downsample_factor,
                                min_transmission)
        obj_scan = np.empty((len(view_ids),) + first.shape, dtype=np.float32)
        obj_scan[0] = first
        t0 = time.time()

        def read_view_into_stack(k):
            obj_scan[k] = read_projection(acq_dir, view_ids[k], num_rows, num_cols, row_slice, col_slice,
                                          downsample_factor, min_transmission)

        with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(read_view_into_stack, range(1, len(view_ids))))
        if verbose > 0:
            print(f'Loaded {len(view_ids)} transmission images -> {obj_scan.shape} ({obj_scan.nbytes / 1e9:.2f} GB) '
                  f'in {time.time() - t0:.1f} s; min {obj_scan.min():.4g}, max {obj_scan.max():.4g}')
    return obj_scan, params


def volumax_vectors(meta, view_ids=None, axis_vector=(0.0, 0.0, -1.0)):
    """
    Form the geometry vectors from the per-view VoluMax positions (averaged over the views).

    r_n: detector normal (source -> detector), r_h: row direction (increasing channel), r_v = r_n x r_h:
    column direction (increasing row, downward), r_a: rotation axis pointing down, r_s / r_d: source and
    detector-centre positions relative to the origin (objectPosition). Also returns the raw detector size /
    pitch, the objectAngle array for the selected views (``angles_deg``) and for all views (``angles_deg_all``,
    used by the calibration) and the position jitter over the views.
    """
    sel = np.arange(meta['num_views']) if view_ids is None else np.asarray(view_ids)
    S = np.nanmean(meta['source'], axis=0)
    D = np.nanmean(meta['detector'], axis=0)
    O = np.nanmean(meta['object'], axis=0)
    U = np.nanmean(meta['span_u'], axis=0)
    V = np.nanmean(meta['span_v'], axis=0)
    r_h = mtp.unit_vector(U)                          # increasing channel index
    v_hat = mtp.unit_vector(V)                        # increasing row index (downward for this scanner)
    r_n = mtp.unit_vector(np.cross(r_h, v_hat))
    if np.dot(r_n, D - S) < 0:
        r_n = -r_n
    r_v = np.cross(r_n, r_h)                          # equals v_hat up to numerical noise
    r_a = mtp.unit_vector(np.asarray(axis_vector, dtype=np.float64))
    if np.dot(r_a, r_v) < 0:                          # the axis vector points down (along increasing rows)
        r_a = -r_a
    jitter = {k: float(np.nanmax(np.linalg.norm(meta[k] - meta[k].mean(axis=0), axis=1)))
              for k in ('source', 'detector', 'object')}
    return dict(
        r_a=r_a, r_n=r_n, r_h=r_h, r_v=r_v, v_hat=v_hat,
        r_s=S - O, r_d=D - O, r_o=np.zeros(3), source_world=S, detector_world=D, object_world=O,
        span_pitch_u=float(np.linalg.norm(U)), span_pitch_v=float(np.linalg.norm(V)),
        delta_det_channel=meta['delta_det_channel'], delta_det_row=meta['delta_det_row'],
        num_det_rows=meta['num_det_rows'], num_det_channels=meta['num_det_channels'],
        angles_deg=np.asarray(meta['angles_deg'])[sel], angles_deg_all=np.array(meta['angles_deg']),
        position_jitter_mm=jitter, num_views_total=meta['num_views'],
    )


# ---------------------------------------------------------------------------------------------------------------------
# Data-driven calibration: rotation direction and channel offset
# ---------------------------------------------------------------------------------------------------------------------
def _load_band_sinogram(acq_dir, view_ids, num_rows, num_cols, row_band, num_workers=8):
    r0, r1 = row_band

    def read_band_profile(i):
        img = read_projection(acq_dir, int(i), num_rows, num_cols, slice(r0, r1), None, (1, 1), 1e-4)
        return -np.log(img).mean(axis=0)

    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        sino = np.array(list(ex.map(read_band_profile, view_ids)), dtype=np.float32)
    sino[sino < 0] = 0.0
    return sino


def _rebin_fan_to_parallel(sino, angles_rad, cor_col, pitch, source_iso, source_det, phis, tt, sign):
    """Fan (central plane) -> parallel rebinning in the mbirtorch cone-beam convention.

    Parallel ray (phi, t) with normal (cos phi, sin phi) is the fan ray gamma = -asin(t/SID), u = SDD tan gamma
    at model angle a = pi + gamma - phi; the measured objectAngle for that model angle is sign * a.
    """
    from scipy.ndimage import map_coordinates
    n = len(angles_rad)
    th0 = angles_rad[0]
    th_ext = np.append(angles_rad, th0 + 2.0 * np.pi)
    sino_ext = np.vstack([sino, sino[:1]])
    gam = -np.arcsin(np.clip(tt / source_iso, -1.0, 1.0))
    col = cor_col + source_det * np.tan(gam) / pitch
    a = np.pi + gam[None, :] - phis[:, None]
    wrapped = th0 + np.mod(sign * a - th0, 2.0 * np.pi)
    idx = np.interp(wrapped.ravel(), th_ext, np.arange(n + 1)).reshape(a.shape)
    return map_coordinates(sino_ext, [idx.ravel(), np.broadcast_to(col[None, :], idx.shape).ravel()],
                           order=1, mode='nearest').reshape(idx.shape)


def estimate_rotation_sign_and_cor(acq_dir, angles_deg_all, geom, row_band, view_step=2, coarse_px=None,
                                   signs=(1.0, -1.0), num_workers=8, verbose=1):
    """
    Rotation direction and centre-of-rotation column from the data (full-resolution frame).

    Every parallel ray of a 360 deg scan is measured twice; after fan-to-parallel rebinning the two must
    agree. The mean squared mismatch is minimised over the centre column for both rotation signs.

    Args:
        acq_dir: acquisition folder.  angles_deg_all: objectAngle for ALL views.
        geom (dict): needs num_det_rows, num_det_channels, delta_det_channel, source_iso_dist, source_detector_dist.
        row_band (tuple): full-resolution rows [r0, r1) to average.
        view_step (int): use every k-th view.
        coarse_px (array): candidate centre offsets from the detector centre (default -40..40 px).
        signs (tuple): rotation signs to test.

    Returns:
        dict: angle_sign, cor_col (full-res index), offset_px, offset_mm, errors per sign, curve.
    """
    n_rows, n_cols = geom['num_det_rows'], geom['num_det_channels']
    pitch, sid, sdd = geom['delta_det_channel'], geom['source_iso_dist'], geom['source_detector_dist']
    if coarse_px is None:
        coarse_px = np.arange(-40.0, 40.01, 1.0)
    views = np.arange(0, len(angles_deg_all), view_step)
    th = np.unwrap(np.deg2rad(np.asarray(angles_deg_all)))[views]
    t0 = time.time()
    sino = _load_band_sinogram(acq_dir, views, n_rows, n_cols, row_band, num_workers)
    mag = sdd / sid
    phis = np.linspace(0.0, np.pi, min(1000, len(views)), endpoint=False)
    nt = (n_cols // 2) | 1
    tt = (np.arange(nt) - (nt - 1) / 2.0) * (2.0 * pitch / mag)
    tmask = np.abs(tt) < 0.9 * tt.max()
    center = (n_cols - 1) / 2.0

    def error(offset_px, sign):
        A = _rebin_fan_to_parallel(sino, th, center + offset_px, pitch, sid, sdd, phis, tt, sign)
        B = _rebin_fan_to_parallel(sino, th, center + offset_px, pitch, sid, sdd, phis + np.pi, tt, sign)[:, ::-1]
        return float(np.mean((A - B)[:, tmask] ** 2))

    coarse = {sgn: np.array([error(d, sgn) for d in coarse_px]) for sgn in signs}
    best_sign = min(coarse, key=lambda s: coarse[s].min())
    d0 = coarse_px[int(np.argmin(coarse[best_sign]))]
    fine_px = np.arange(d0 - 2.0, d0 + 2.01, 0.25)
    fine = np.array([error(d, best_sign) for d in fine_px])
    i = int(np.argmin(fine))
    frac = 0.0
    if 0 < i < len(fine) - 1:
        denom = fine[i - 1] - 2 * fine[i] + fine[i + 1]
        frac = 0.5 * (fine[i - 1] - fine[i + 1]) / denom if denom > 0 else 0.0
    offset_px = float(fine_px[i] + frac * 0.25)
    out = dict(angle_sign=float(best_sign), cor_col=center + offset_px, offset_px=offset_px, offset_mm=offset_px * pitch,
               error_best=float(fine[i]), errors_min={f'{s:+.0f}': float(coarse[s].min()) for s in signs},
               row_band=tuple(row_band), num_views=int(len(views)), seconds=time.time() - t0)
    if verbose > 0:
        others = [s for s in signs if s != best_sign]
        tail = (f', other sign {coarse[others[0]].min() / max(fine[i], 1e-30):.0f}x worse' if others else '')
        print(f'   rows {row_band[0]}:{row_band[1]}: sign {best_sign:+.0f}, centre column {out["cor_col"]:.2f} '
              f'({offset_px:+.2f} px = {out["offset_mm"]:+.4f} mm){tail} ({out["seconds"]:.0f} s)')
    return out
