import concurrent.futures as cf
import json
import time
import warnings
from pathlib import Path
import numpy as np
import mbirtorch
import mbirtorch.preprocess as mtp
from mbirtorch.preprocess import geometry_calibration as gc


def get_sino_and_model(scan_dir, *, downsample_factor=(1, 1), subsample_view_factor=1, crop_pixels_sides=0,
                       crop_pixels_top=0, crop_pixels_bottom=0, auto_crop=False, verbose=1, min_transmission=1e-4,
                       background_offset='per_view', sinogram_path=None, num_workers=8, batch_size=90):
    """
    Load a Zeiss VoluMax scan, compute its sinogram, and return a ready-to-reconstruct model.

    Args:
        scan_dir (str or pathlib.Path): Path to the scan folder, the folder that contains
            ``AcquisitionParameters.json`` (see :func:`load_scans_and_params` for its layout).
        downsample_factor (tuple[int, int], optional): Detector row/channel downsampling by block averaging.
            Defaults to ``(1, 1)``.
        subsample_view_factor (int, optional): Keep every n-th view. Defaults to ``1``.
        crop_pixels_sides (int, optional): Pixels to crop from each lateral side of the detector, before
            downsampling. Defaults to ``0``.
        crop_pixels_top (int, optional): Pixels to crop from the top of the detector, before downsampling.
            Defaults to ``0``.
        crop_pixels_bottom (int, optional): Pixels to crop from the bottom of the detector, before downsampling.
            Defaults to ``0``.
        auto_crop (bool, optional): If True, detect and remove blank sinogram margins after the sinogram is
            computed, shrinking the reconstruction. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to ``1``.
        min_transmission (float, optional): Transmission values below this are clipped before the logarithm is
            taken. Defaults to ``1e-4``.
        background_offset (str or None, optional): Background offset correction: ``'per_view'``, ``'global'``, or
            None for no correction (see :func:`mbirtorch.preprocess.correct_background_offset`).
            Defaults to ``'per_view'``.
        sinogram_path (str or None, optional): Path to a precomputed sinogram to use instead of the projections,
            for example one corrected for scatter or beam hardening: a ``.npy`` file of -log data with shape
            (num_views, num_det_rows, num_det_channels) for the views, crop, and downsampling requested here.  The
            detector rotation from the metadata and the background offset correction are applied to it as to a
            computed sinogram, so it must not already be rotation corrected (the ``sino`` returned here is).
            Defaults to None.
        num_workers (int, optional): Number of threads that read the projections. Defaults to ``8``.
        batch_size (int, optional): Number of views per device batch when the sinogram is computed.
            Defaults to ``90``.

    Returns:
        tuple: ``(sino, model)`` where

            - ``sino`` (numpy.ndarray): the computed sinogram, float32 with shape
              (num_views, num_det_rows, num_det_channels).
            - ``model`` (ConeBeamModel): a model with its calibrated geometry and its reconstruction geometry set.

    Example:
        .. code-block:: python

            import mbirtorch
            from mbirtorch.preprocess import volumax
            sino, model = volumax.get_sino_and_model(scan_dir, downsample_factor=(2, 2))
            weights = mbirtorch.gen_weights(sino, weight_type='transmission_root')
            recon, recon_dict = model.recon(sino, weights=weights)

    Note:
        Reconstruction weights are not returned; generate them with ``mbirtorch.gen_weights``.

    Raises:
        FileNotFoundError: If ``scan_dir`` does not contain ``AcquisitionParameters.json`` or any per-view metadata
            file, or, when ``sinogram_path`` is None, projection images of the selected views are missing.
        ValueError: If a precomputed sinogram does not match the requested views, crop, and downsampling, a file
            fails the checks of :func:`load_scans_and_params`, or the views do not cover a full rotation, which the
            channel-offset calibration needs.
    """
    sino, required_params, optional_params, geometry = _compute_sino_and_params(
        scan_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor,
        crop_pixels_sides=crop_pixels_sides, crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom,
        verbose=verbose, min_transmission=min_transmission, background_offset=background_offset,
        sinogram_path=sinogram_path, num_workers=num_workers, batch_size=batch_size)
    sino, model = mtp.finalize_model(sino, required_params, optional_params, auto_crop=auto_crop)
    calibrate_volumax_geometry(model, sino, det_rotation=geometry['det_rotation'], verbose=verbose)
    if verbose > 0:
        print('\n########## Model parameters')
        model.print_params()
    return sino, model


def _compute_sino_and_params(scan_dir, downsample_factor=(1, 1), subsample_view_factor=1, crop_pixels_sides=0,
                             crop_pixels_top=0, crop_pixels_bottom=0, verbose=1, min_transmission=1e-4,
                             background_offset='per_view', sinogram_path=None, num_workers=8, batch_size=90):
    """
    Load a VoluMax scan and compute the sinogram and the model parameters.

    This is the private helper for :func:`get_sino_and_model`, which documents the arguments.

    Returns:
        tuple: ``(sino, required_params, optional_params, geometry)``.  ``required_params`` holds the ConeBeamModel
        constructor arguments and a ``geometry_type`` entry that ``build_model`` uses to select the model class.
        ``optional_params`` holds the ``set_params`` arguments.  ``geometry`` is the full-resolution geometry from
        :func:`compute_geometry`.
    """
    if verbose > 0:
        print('\n########## Loading VoluMax projections and geometry')
    obj_scan, volumax_params = load_scans_and_params(
        scan_dir, subsample_view_factor=subsample_view_factor, downsample_factor=downsample_factor,
        crop_pixels_sides=crop_pixels_sides, crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom,
        min_transmission=min_transmission, num_workers=num_workers, verbose=verbose,
        load_scans=sinogram_path is None)
    geometry = compute_geometry(volumax_params, verbose=verbose)
    cone_beam_params, optional_params = convert_volumax_to_mbirtorch_params(
        volumax_params, geometry, downsample_factor=downsample_factor, crop_pixels_sides=crop_pixels_sides,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom, verbose=verbose)
    # det_rotation is not a TomographyModel parameter; it is applied to the sinogram instead.
    det_rotation = optional_params.pop('det_rotation')

    if sinogram_path is None:
        if verbose > 0:
            print(f'\n########## Computing sinogram (detector rotation {det_rotation:+.3e} rad)')
        # The projections are already normalized transmission, so the blank scan is one and the dark scan is zero.
        blank_scan = np.ones((1,) + obj_scan.shape[1:], dtype=np.float32)
        dark_scan = np.zeros((1,) + obj_scan.shape[1:], dtype=np.float32)
        sino = mtp.scan_to_sino(obj_scan, blank_scan, dark_scan, (), downsample_factor=(1, 1),
                                det_rotation=det_rotation, batch_size=batch_size)
        del obj_scan
    else:
        if verbose > 0:
            print(f'\n########## Loading precomputed sinogram {sinogram_path}')
        sino = np.ascontiguousarray(np.load(sinogram_path, mmap_mode='r'), dtype=np.float32)
        if det_rotation != 0.0:
            sino = mtp.correct_det_rotation(sino, det_rotation=det_rotation)
    if tuple(sino.shape) != tuple(cone_beam_params['sinogram_shape']):
        raise ValueError(f'The sinogram shape {sino.shape} does not match the geometry '
                         f'{cone_beam_params["sinogram_shape"]}.  The views, crop, and downsampling must match '
                         'the data.')
    if background_offset not in (None, 'none'):
        if verbose > 0:
            print(f'\n########## Correcting background offset ({background_offset})')
        sino = mtp.correct_background_offset(sino, option=background_offset)
    if verbose > 0:
        print(f'sinogram shape = {sino.shape}, min {sino.min():.4f}, max {sino.max():.4f}')
    return sino, cone_beam_params, optional_params, geometry


# ---------------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------------
def compute_geometry(params, *, verbose=1):
    """
    Compute the full-resolution cone-beam geometry, in mm, from the VoluMax metadata.

    Args:
        params (dict): Geometry parameters from :func:`load_scans_and_params`.
        verbose (int, optional): Verbosity level. Defaults to ``1``.

    Returns:
        dict: ``source_detector_dist`` and ``source_iso_dist`` (both along the detector normal), ``magnification``,
        ``det_row_offset``, ``det_channel_offset_metadata``, ``det_rotation`` (radians), ``iso_row_full`` (the
        full-resolution row index of the source's perpendicular projection onto the detector), and the detector size
        and pitch.
    """
    r_a, r_n, r_h, r_v, r_s, r_d = (params[k] for k in ('r_a', 'r_n', 'r_h', 'r_v', 'r_s', 'r_d'))
    pitch_c, pitch_r = params['delta_det_channel'], params['delta_det_row']
    n_rows, n_cols = params['num_det_rows'], params['num_det_channels']
    sdd, sid, mag, det_rotation = (float(x) for x in mtp.nsi.calc_source_detector_params(r_a, r_n, r_h, r_s, r_d))
    # det_row_offset is the row coordinate of the source's perpendicular foot on the detector, relative to the
    # detector center and positive toward higher row indices.  calc_row_channel_params takes the center of the first
    # detector pixel; the detector center is passed together with a 1 x 1 detector, which makes its shift from the
    # first pixel to the detector center exactly zero.  It only normalizes r_h, so r_h is made orthogonal to r_n for
    # vectors that are not already orthogonal (those from volumax_vectors are).
    r_h_in_panel = r_h - mtp.project_vector_to_vector(r_h, mtp.unit_vector(r_n))
    det_channel_offset_meta, det_row_offset = (float(x) for x in mtp.nsi.calc_row_channel_params(
        r_a, r_n, r_h_in_panel, r_s, r_d, pitch_c, pitch_r, 1, 1, mag))
    iso_row_full = (n_rows - 1) / 2.0 + det_row_offset / pitch_r
    geometry = dict(source_detector_dist=sdd, source_iso_dist=sid, magnification=mag,
                    delta_det_channel=pitch_c, delta_det_row=pitch_r, num_det_rows=n_rows, num_det_channels=n_cols,
                    det_row_offset=det_row_offset, iso_row_full=iso_row_full, det_rotation=float(det_rotation),
                    det_channel_offset_metadata=det_channel_offset_meta)
    if verbose > 0:
        print('\n########## Geometry from the metadata (full-resolution detector, mm)')
        print_geometry_table(geometry)
        print(f'   r_a = {np.round(r_a, 4)}, r_n = {np.round(r_n, 5)}, r_h = {np.round(r_h, 5)}, '
              f'r_v = {np.round(r_v, 5)}')
        print(f'   largest position deviation over the views (mm): {params["position_jitter_mm"]}')
    return geometry


def print_geometry_table(geometry):
    """
    Print the full-resolution geometry from :func:`compute_geometry` as a table.

    Args:
        geometry (dict): Geometry from :func:`compute_geometry`.
    """
    pitch_c, pitch_r = geometry['delta_det_channel'], geometry['delta_det_row']
    det_channel_offset, det_row_offset = geometry['det_channel_offset_metadata'], geometry['det_row_offset']
    print(f'   {"quantity":28s} {"value":>14s}   note')
    print(f'   {"source_iso_dist":28s} {geometry["source_iso_dist"]:14.4f}   along the detector normal')
    print(f'   {"source_detector_dist":28s} {geometry["source_detector_dist"]:14.4f}   along the detector normal')
    print(f'   {"magnification":28s} {geometry["magnification"]:14.5f}   voxel pitch at the axis '
          f'{pitch_c / geometry["magnification"]:.6f}')
    print(f'   {"det_channel_offset_metadata":28s} {det_channel_offset:+14.4f}   '
          f'{det_channel_offset / pitch_c:+.2f} px, starting value of the data calibration')
    print(f'   {"det_row_offset":28s} {det_row_offset:+14.4f}   {det_row_offset / pitch_r:+.2f} px, source projection '
          f'at row {geometry["iso_row_full"]:.2f} of {geometry["num_det_rows"]}')
    print(f'   {"det_rotation [rad]":28s} {geometry["det_rotation"]:+14.3e}   '
          f'{np.rad2deg(geometry["det_rotation"]):+.4f} deg, from the detector span vectors')
    print(f'   {"recon_slice_offset":28s} {-det_row_offset / geometry["magnification"]:+14.4f}   '
          f'-det_row_offset / magnification')


def rotate_offsets_for_det_rotation(det_channel_offset, det_row_offset, det_rotation):
    """
    Express the detector offsets in the frame of a sinogram rotated by ``det_rotation``.

    :func:`mbirtorch.preprocess.correct_det_rotation` and ``mbirtorch.preprocess.scan_to_sino`` rotate each view
    about the detector center, which moves a feature at (row, channel) relative to the center to
    (cos t * row - sin t * channel, sin t * row + cos t * channel).  The same rotation is applied here to the point
    that the offsets describe.  The mapping is exact for any angle when the detector pixels are square.

    Args:
        det_channel_offset (float): Channel offset relative to the detector center, in ALU.
        det_row_offset (float): Row offset relative to the detector center, in ALU.
        det_rotation (float): Detector rotation in radians.

    Returns:
        tuple: ``(det_channel_offset, det_row_offset)`` in the rotated frame.
    """
    c, s = np.cos(det_rotation), np.sin(det_rotation)
    return float(s * det_row_offset + c * det_channel_offset), float(c * det_row_offset - s * det_channel_offset)


def convert_volumax_to_mbirtorch_params(params, geometry, downsample_factor=(1, 1), crop_pixels_sides=0,
                                        crop_pixels_top=0, crop_pixels_bottom=0, verbose=1):
    """
    Convert the VoluMax geometry into mbirtorch parameters, accounting for the crop, the detector rotation, and the
    downsampling.

    The crop is applied first, in raw detector pixels.  The detector rotation is applied to the sinogram about the
    detector center, so the offsets are rotated into that frame (:func:`rotate_offsets_for_det_rotation`).  The channel
    offset is the metadata value, which :func:`calibrate_volumax_geometry` later refines from the sinogram.  The view
    angles are ``-objectAngle``: the VoluMax ``objectAngle`` turns opposite to the view angles of ``ConeBeamModel``.

    Args:
        params (dict): Geometry parameters from :func:`load_scans_and_params`.
        geometry (dict): Full-resolution geometry from :func:`compute_geometry`.
        downsample_factor (tuple[int, int], optional): Detector row/channel downsampling.  When the cropped detector
            size is not divisible by the factor, the remainder is dropped at the bottom and right, as
            :func:`read_projection` does. Defaults to ``(1, 1)``.
        crop_pixels_sides (int, optional): Pixels cropped from each lateral side of the detector. Defaults to ``0``.
        crop_pixels_top (int, optional): Pixels cropped from the top of the detector. Defaults to ``0``.
        crop_pixels_bottom (int, optional): Pixels cropped from the bottom of the detector. Defaults to ``0``.
        verbose (int, optional): Verbosity level. Defaults to ``1``.

    Returns:
        tuple: ``(cone_beam_params, optional_params)``.  ``cone_beam_params`` holds the ConeBeamModel constructor
        arguments and a ``geometry_type`` entry.  ``optional_params`` holds the ``set_params`` arguments and a
        ``det_rotation`` entry, which is applied to the sinogram rather than set on the model.
    """
    num_det_rows, num_det_channels = params['num_det_rows'], params['num_det_channels']
    delta_det_row, delta_det_channel = geometry['delta_det_row'], geometry['delta_det_channel']
    det_row_offset, det_channel_offset = geometry['det_row_offset'], geometry['det_channel_offset_metadata']
    num_det_rows, num_det_channels, det_row_offset, det_channel_offset = mtp.apply_config_crop(
        num_det_rows, num_det_channels, det_row_offset, det_channel_offset, delta_det_row, delta_det_channel,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom, crop_pixels_sides=crop_pixels_sides)
    # Block averaging drops the leftover rows and channels at the bottom and right, which moves the detector center
    # by half of them.
    det_row_offset += (num_det_rows % downsample_factor[0]) / 2.0 * delta_det_row
    det_channel_offset += (num_det_channels % downsample_factor[1]) / 2.0 * delta_det_channel
    det_rotation = float(geometry['det_rotation'])
    if det_rotation != 0.0:
        channel_before, row_before = det_channel_offset, det_row_offset
        det_channel_offset, det_row_offset = rotate_offsets_for_det_rotation(channel_before, row_before, det_rotation)
        if verbose > 0:
            channel_px = (det_channel_offset - channel_before) / delta_det_channel
            row_px = (det_row_offset - row_before) / delta_det_row
            print(f'   offsets rotated by the detector rotation: channel {channel_before:+.4f} -> '
                  f'{det_channel_offset:+.4f} mm ({channel_px:+.3f} px), row {row_before:+.4f} -> '
                  f'{det_row_offset:+.4f} mm ({row_px:+.3f} px)')
    num_det_rows //= downsample_factor[0]
    num_det_channels //= downsample_factor[1]
    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    # The VoluMax objectAngle turns opposite to the ConeBeamModel view angle, so the angles are negated.  This is a
    # property of the scanner; the opposite-view consistency of two scans (HIP_Can and Hexagonal) confirmed it.
    angles = np.ascontiguousarray(-np.unwrap(np.deg2rad(params['angles_deg'])), dtype=np.float32)
    sid, sdd = geometry['source_iso_dist'], geometry['source_detector_dist']
    cone_beam_params = dict(sinogram_shape=(len(angles), int(num_det_rows), int(num_det_channels)), angles=angles,
                            source_detector_dist=float(sdd), source_iso_dist=float(sid),
                            geometry_type=str(mbirtorch.ConeBeamModel))
    optional_params = dict(delta_det_channel=float(delta_det_channel), delta_det_row=float(delta_det_row),
                           delta_voxel=float(delta_det_channel * sid / sdd),
                           det_channel_offset=float(det_channel_offset), det_row_offset=float(det_row_offset),
                           recon_slice_offset=float(-det_row_offset / geometry['magnification']),
                           det_rotation=det_rotation, alu_unit='mm', alu_value=1.0)
    return cone_beam_params, optional_params


# ---------------------------------------------------------------------------------------------------------------------
# Reading the data
# ---------------------------------------------------------------------------------------------------------------------
def _vec3(entry):
    """Convert a metadata entry ``{'x': ..., 'y': ..., 'z': ...}`` to a float64 array."""
    return np.array([entry['x'], entry['y'], entry['z']], dtype=np.float64)


def read_volumax_metadata(scan_dir, verbose=1):
    """
    Read the scan-level and per-view metadata of a VoluMax scan.

    The scan-level parameters are read from ``AcquisitionParameters.json`` and the geometry of each view from
    ``Projections/Metadata/Projection_NNNNN.json``, where NNNNN is the view index counted from 0.  The values of missing
    per-view files are filled in from the files that are present, with a warning.  Positions and span vectors are
    interpolated linearly and held at the nearest present value beyond the first and last present views; angles are
    interpolated linearly in unwrapped angle, and the angle step is continued beyond them.

    Args:
        scan_dir (str or pathlib.Path): The scan folder, the folder that contains ``AcquisitionParameters.json``.
        verbose (int, optional): Verbosity level. Defaults to ``1``.

    Returns:
        dict: ``acq`` (the scan-level parameters), ``scan_dir``, ``num_views``, the detector size (``num_det_rows``,
        ``num_det_channels``) and pitch (``delta_det_row``, ``delta_det_channel``), ``angles_deg`` (``objectAngle`` of
        each view), and the per-view arrays ``source``, ``detector``, ``object``, ``span_u``, and ``span_v``, each of
        shape (num_views, 3).

    Raises:
        FileNotFoundError: If ``scan_dir`` does not contain ``AcquisitionParameters.json``, or no per-view metadata
            file is present.
        ValueError: If only one per-view metadata file is present, or one has no ``projectionMetrics`` entry.
    """
    scan_dir = Path(scan_dir).expanduser()
    if not scan_dir.is_dir():
        raise FileNotFoundError(f'The scan folder {scan_dir} does not exist or is not a folder.')
    parameter_file = scan_dir / 'AcquisitionParameters.json'
    if not parameter_file.is_file():
        raise FileNotFoundError(f'{scan_dir} does not contain AcquisitionParameters.json; scan_dir must be the scan '
                                'folder that holds it.')
    with open(parameter_file) as f:
        acq = json.load(f)
    num_views = int(acq['numberOfProjections'])
    det = acq['detectorParameters']
    num_rows, num_cols = int(det['imageSize']['height']), int(det['imageSize']['width'])

    meta_dir = scan_dir / 'Projections' / 'Metadata'
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
            data = json.load(f)
        if not isinstance(data, dict) or 'projectionMetrics' not in data:
            raise ValueError(f'{fn} has no "projectionMetrics" entry; it is not a VoluMax per-view metadata file.')
        pm = data['projectionMetrics']
        angles_deg[i] = pm['objectAngle']
        for k in keys:
            vec[k][i] = _vec3(pm[k])
    if len(missing) == num_views:
        raise FileNotFoundError(f'No per-view metadata files Projection_NNNNN.json found in {meta_dir}.')
    if missing:
        good = np.setdiff1d(np.arange(num_views), missing)
        if len(good) < 2:
            raise ValueError(f'1 of {num_views} per-view metadata files found in {meta_dir}; at least 2 are needed to '
                             'fill in the missing views.')
        for k in keys:
            for c in range(3):
                vec[k][missing, c] = np.interp(missing, good, vec[k][good, c])
        # np.interp holds the end values, which would repeat an angle when the first or last file is missing, so
        # the angle step is continued beyond the ends instead.
        ang = np.unwrap(np.deg2rad(angles_deg[good]))
        idx = np.asarray(missing)
        filled = np.interp(idx, good, ang)
        before, after = idx < good[0], idx > good[-1]
        filled[before] = ang[0] + (idx[before] - good[0]) * (ang[1] - ang[0]) / (good[1] - good[0])
        filled[after] = ang[-1] + (idx[after] - good[-1]) * (ang[-1] - ang[-2]) / (good[-1] - good[-2])
        angles_deg[missing] = np.rad2deg(filled) % 360.0
        warnings.warn(f'Per-view metadata missing for {len(missing)} of {num_views} views, interpolated (e.g. views '
                      f'{missing[:5]}).')
    info = dict(
        acq=acq, scan_dir=scan_dir, num_views=num_views, num_det_rows=num_rows, num_det_channels=num_cols,
        delta_det_channel=float(det['pixelPitch']['horizontal']), delta_det_row=float(det['pixelPitch']['vertical']),
        angles_deg=angles_deg, source=vec['sourcePosition'], detector=vec['detectorPosition'],
        object=vec['objectPosition'], span_u=vec['spanVectorU'], span_v=vec['spanVectorV'],
    )
    if verbose > 0:
        step = np.diff(np.unwrap(np.deg2rad(angles_deg)))
        tube = acq.get('tubeParameters') or {}
        filter_material = (acq.get('filterChangerParameters') or {}).get('material')
        print(f'VoluMax scan {scan_dir}')
        print(f'   {num_views} views, angle step {np.rad2deg(step.mean()):.5f} deg, '
              f'angular range {np.rad2deg(step.sum()):.3f} deg, mode {acq.get("mode")}')
        print(f'   detector {num_rows} rows x {num_cols} channels, pitch {info["delta_det_row"]} x '
              f'{info["delta_det_channel"]} mm, bit depth {det.get("bitDepth")}, binning {det.get("binning")}')
        print(f'   tube {tube.get("accelerationVoltageInKV")} kV, {tube.get("sourceCurrentInMicroA")} uA, integration '
              f'{det.get("integrationTimeInMs")} ms, filter {filter_material}')
    return info


def read_projection(path, num_rows, num_cols, row_slice=None, col_slice=None, downsample_factor=(1, 1),
                    min_transmission=1e-4):
    """
    Read one transmission image, cropped, block-averaged, and clipped.

    The image is stored as raw little-endian float32 of shape (num_rows, num_cols) and is read through a memory map,
    so only the cropped region is loaded.

    Args:
        path (str or pathlib.Path): The image file, ``Projections/Images/Projection_NNNNN.float32`` in the scan folder.
        num_rows (int): Number of rows of the stored image.
        num_cols (int): Number of columns of the stored image.
        row_slice (slice, optional): Rows to keep. Defaults to all rows.
        col_slice (slice, optional): Columns to keep. Defaults to all columns.
        downsample_factor (tuple[int, int], optional): Block-averaging factors for rows and columns; a remainder is
            dropped at the bottom and right. Defaults to ``(1, 1)``.
        min_transmission (float or None, optional): Values below this are clipped; None disables the clipping.
            Defaults to ``1e-4``.

    Returns:
        numpy.ndarray: The transmission image, float32.

    Raises:
        ValueError: If the file size does not match num_rows x num_cols float32 values.
    """
    path = Path(path)
    expected = num_rows * num_cols * 4
    if path.stat().st_size != expected:
        raise ValueError(f'{path} has {path.stat().st_size} bytes; {expected} are expected for {num_rows} x '
                         f'{num_cols} float32 values.')
    mm = np.memmap(path, dtype='<f4', mode='r', shape=(num_rows, num_cols))
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


def load_scans_and_params(scan_dir, view_id_start=0, view_id_end=None, subsample_view_factor=1,
                          downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0,
                          min_transmission=1e-4, num_workers=8, verbose=1, load_scans=True):
    """
    Load the transmission images and the geometry parameters of a VoluMax scan.

    The projections are stored as normalized transmission, so no blank or dark scans are needed.  The crop and the
    detector downsampling are applied as each image is read, so the full-resolution stack is never held in memory.

    Args:
        scan_dir (str or pathlib.Path): Path to the scan folder, the folder that contains
            ``AcquisitionParameters.json``.  The scan folder is assumed to have the following structure, where NNNNN is
            the view index counted from 0:

            - ``AcquisitionParameters.json`` (scan-level parameters)
            - ``Projections/Metadata/Projection_NNNNN.json`` (geometry of each view)
            - ``Projections/Images/Projection_NNNNN.float32`` (transmission image of each view)

        view_id_start (int, optional): Index of the first view. Defaults to ``0``.
        view_id_end (int, optional): Index one past the last view. Defaults to None, which reads to the last view.
        subsample_view_factor (int, optional): Keep every n-th view. Defaults to ``1``.
        downsample_factor (tuple[int, int], optional): Detector row/channel downsampling by block averaging.
            Defaults to ``(1, 1)``.
        crop_pixels_sides (int, optional): Raw pixels to crop from each lateral side of the detector.
            Defaults to ``0``.
        crop_pixels_top (int, optional): Raw pixels to crop from the top of the detector. Defaults to ``0``.
        crop_pixels_bottom (int, optional): Raw pixels to crop from the bottom of the detector. Defaults to ``0``.
        min_transmission (float or None, optional): Transmission values below this are clipped. Defaults to ``1e-4``.
        num_workers (int, optional): Number of threads that read the images. Defaults to ``8``.
        verbose (int, optional): Verbosity level. Defaults to ``1``.
        load_scans (bool, optional): If False, only the geometry is read, the images need not be present, and
            ``obj_scan`` is None. Defaults to True.

    Returns:
        tuple: ``(obj_scan, volumax_params)`` where

            - ``obj_scan`` (numpy.ndarray or None): the transmission images after the crop and downsampling, float32
              with shape (num_views, num_det_rows, num_det_channels).
            - ``volumax_params`` (dict): the geometry vectors and detector parameters from :func:`volumax_vectors`,
              plus the scan folder, the view indices, the crop and downsampling settings, and ``min_transmission``.

    Raises:
        FileNotFoundError: If ``scan_dir`` does not contain ``AcquisitionParameters.json`` or per-view metadata, or,
            when ``load_scans`` is True, projection images of the selected views are missing.
        ValueError: If only one per-view metadata file is present, a metadata file has no ``projectionMetrics``
            entry, or, when ``load_scans`` is True, an image file does not hold num_rows x num_cols float32 values.
    """
    scan_dir = Path(scan_dir).expanduser()
    meta = read_volumax_metadata(scan_dir, verbose=verbose)
    num_rows, num_cols = meta['num_det_rows'], meta['num_det_channels']
    if view_id_end is None:
        view_id_end = meta['num_views']
    view_ids = np.arange(view_id_start, view_id_end, subsample_view_factor, dtype=np.int64)

    row_slice = slice(int(crop_pixels_top), num_rows - int(crop_pixels_bottom))
    col_slice = slice(int(crop_pixels_sides), num_cols - int(crop_pixels_sides))
    volumax_params = volumax_vectors(meta, view_ids)
    volumax_params.update(dict(
        scan_dir=scan_dir, view_ids=view_ids,
        crop_pixels_sides=int(crop_pixels_sides), crop_pixels_top=int(crop_pixels_top),
        crop_pixels_bottom=int(crop_pixels_bottom),
        downsample_factor=(int(downsample_factor[0]), int(downsample_factor[1])), min_transmission=min_transmission,
    ))

    obj_scan = None
    if load_scans:
        image_dir = scan_dir / 'Projections' / 'Images'
        image_files = [image_dir / f'Projection_{i:05d}.float32' for i in view_ids]
        missing = [int(i) for i, fn in zip(view_ids, image_files) if not fn.is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} projection images missing in {image_dir} (e.g. views '
                                    f'{missing[:5]}).')
        # The first image sets the output shape; the others are read in parallel into the preallocated stack.
        first = read_projection(image_files[0], num_rows, num_cols, row_slice, col_slice, downsample_factor,
                                min_transmission)
        obj_scan = np.empty((len(view_ids),) + first.shape, dtype=np.float32)
        obj_scan[0] = first
        t0 = time.time()

        def read_view_into_stack(k):
            obj_scan[k] = read_projection(image_files[k], num_rows, num_cols, row_slice, col_slice,
                                          downsample_factor, min_transmission)

        with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(ex.map(read_view_into_stack, range(1, len(view_ids))))
        if verbose > 0:
            print(f'Loaded {len(view_ids)} transmission images, shape {obj_scan.shape} '
                  f'({obj_scan.nbytes / 1e9:.2f} GB) in {time.time() - t0:.1f} s; '
                  f'min {obj_scan.min():.4g}, max {obj_scan.max():.4g}')
    return obj_scan, volumax_params


def volumax_vectors(meta, view_ids=None, axis_vector=(0.0, 0.0, -1.0)):
    """
    Form the geometry vectors from the per-view positions, averaged over the views.

    The vectors follow the conventions of the NSI helpers in ``mbirtorch.preprocess.nsi``:

    - ``r_h``: unit vector along the detector rows, toward increasing channel index.
    - ``r_n``: unit detector normal, pointing from the source to the detector.
    - ``r_v = r_n x r_h``: unit vector along the detector columns, toward increasing row index.
    - ``r_a``: unit vector along the rotation axis, oriented like ``r_v``.
    - ``r_s``, ``r_d``: source and detector-center positions relative to ``objectPosition``.

    Args:
        meta (dict): Metadata from :func:`read_volumax_metadata`.
        view_ids (array-like, optional): Views whose angles are returned. Defaults to all views.
        axis_vector (tuple, optional): Direction of the rotation axis in the scanner coordinates, which the metadata
            do not record. Defaults to ``(0.0, 0.0, -1.0)``, the vertical axis.

    Returns:
        dict: The vectors above and ``v_hat`` (the unit span vector along the columns as recorded), the world
        positions ``source_world``, ``detector_world``, and ``object_world``, the span vector lengths ``span_pitch_u``
        and ``span_pitch_v``, the detector size and pitch, ``angles_deg`` of the selected views, ``num_views_total``,
        and ``position_jitter_mm``, a dict with the largest deviation of the ``'source'``, ``'detector'``, and
        ``'object'`` positions from their means over all views.
    """
    sel = np.arange(meta['num_views']) if view_ids is None else np.asarray(view_ids)
    S = np.nanmean(meta['source'], axis=0)
    D = np.nanmean(meta['detector'], axis=0)
    O = np.nanmean(meta['object'], axis=0)
    U = np.nanmean(meta['span_u'], axis=0)
    V = np.nanmean(meta['span_v'], axis=0)
    r_h = mtp.unit_vector(U)
    v_hat = mtp.unit_vector(V)
    r_n = mtp.unit_vector(np.cross(r_h, v_hat))
    if np.dot(r_n, D - S) < 0:
        r_n = -r_n
    r_v = np.cross(r_n, r_h)
    if np.dot(r_v, v_hat) < 0:
        warnings.warn('The detector span vectors describe a mirrored image: the rows increase opposite to '
                      'r_n x r_h, which the cone-beam model assumes.  The row offset, the detector rotation, and the '
                      'rotation direction will have the wrong sign unless the images are flipped.')
    r_a = mtp.unit_vector(np.asarray(axis_vector, dtype=np.float64))
    if np.dot(r_a, r_v) < 0:
        r_a = -r_a
    jitter = {k: float(np.nanmax(np.linalg.norm(meta[k] - meta[k].mean(axis=0), axis=1)))
              for k in ('source', 'detector', 'object')}
    return dict(
        r_a=r_a, r_n=r_n, r_h=r_h, r_v=r_v, v_hat=v_hat,
        r_s=S - O, r_d=D - O, r_o=np.zeros(3), source_world=S, detector_world=D, object_world=O,
        span_pitch_u=float(np.linalg.norm(U)), span_pitch_v=float(np.linalg.norm(V)),
        delta_det_channel=meta['delta_det_channel'], delta_det_row=meta['delta_det_row'],
        num_det_rows=meta['num_det_rows'], num_det_channels=meta['num_det_channels'],
        angles_deg=np.asarray(meta['angles_deg'])[sel], position_jitter_mm=jitter, num_views_total=meta['num_views'],
    )


# ---------------------------------------------------------------------------------------------------------------------
# Calibration from the sinogram
# ---------------------------------------------------------------------------------------------------------------------
def _estimate_channel_offset(model, sino):
    """
    Estimate the channel offset with ``estimate_det_channel_offset`` and apply it to ``model``.

    While the coarse minimum is at an edge of the search window, the search is restarted from the last estimate, for
    at most four searches in total.  The restarts end when the estimator reports that its window cannot move further,
    or when a restart raises ``ValueError`` because no channels remain to compare; the previous estimate is then kept.
    A ``ValueError`` from the first search propagates.

    Returns:
        tuple: ``(result, caught_warnings)`` of the last completed search; the caller decides whether to reissue the
        warnings.
    """
    # The search notes that geometry_calibration adds when the coarse minimum sits at an edge of the search window
    # and when that window cannot move further.
    edge_note = 'the coarse minimum sits at an edge of the bounds'
    stuck_note = 'the search window could not move further'
    for round_ in range(4):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = gc.estimate_det_channel_offset(model, sino)
        except ValueError:
            if round_ == 0:
                raise
            break
        offset, offset_warnings = result, caught
        gc.apply_calibration(model, sino, offset)
        notes = offset.reduction['search_notes']
        if edge_note not in notes or any(note.startswith(stuck_note) for note in notes):
            break
    return offset, offset_warnings


def calibrate_volumax_geometry(model, sino, det_rotation=0.0, verbose=1):
    """
    Measure the channel offset from the sinogram and set it on the model.

    The calibration uses the estimators in ``mbirtorch.preprocess.geometry_calibration`` in three steps:

    1. :func:`~mbirtorch.preprocess.geometry_calibration.estimate_det_channel_offset` refines the metadata channel
       offset by comparing each view with its opposite, and the result is applied with
       :func:`~mbirtorch.preprocess.geometry_calibration.apply_calibration`.
    2. The offsets were rotated into the frame of the rotation-corrected sinogram using the metadata channel offset,
       so ``det_row_offset`` is updated for the change in the channel offset.
    3. The detector rotation from the metadata is kept.
       :func:`~mbirtorch.preprocess.geometry_calibration.estimate_det_rotation` measures any rotation left in the
       sinogram, and a warning is issued when the data clearly show one that moves the edge channels by a pixel or
       more.

    The rotation direction is not estimated; it is fixed by :func:`convert_volumax_to_mbirtorch_params`.

    Args:
        model (ConeBeamModel): Model from ``mbirtorch.preprocess.finalize_model``.  Its channel offset, row offset, and
            ``recon_slice_offset`` are updated in place.
        sino (numpy.ndarray): The sinogram the model was built for. Not modified.
        det_rotation (float, optional): Detector rotation in radians that was already removed from ``sino``.
            Defaults to ``0.0``.
        verbose (int, optional): Verbosity level. Defaults to ``1``.

    Returns:
        dict: The ``CalibrationResult`` of the channel offset (``det_channel_offset``) and of the residual detector
        rotation (``det_rotation_residual``).
    """
    if verbose > 0:
        print('\n########## Calibrating the channel offset from the sinogram')
    delta = float(model.get_params('delta_det_channel'))
    start, row_start = (float(x) for x in model.get_params(['det_channel_offset', 'det_row_offset']))

    # Step 1: the channel offset, which _estimate_channel_offset applies to the model.
    offset, offset_warnings = _estimate_channel_offset(model, sino)
    for w in offset_warnings:
        warnings.warn(w.message, w.category)

    # Step 2: under the detector rotation, the channel correction moves the row offset by -tan(det_rotation) times
    # as much.
    row = float(row_start - np.tan(det_rotation) * (offset.value - start))
    if row != row_start:
        gc.apply_calibration(model, sino, offset._replace(parameter='det_row_offset', value=row))

    # Step 3: check for a detector rotation left in the sinogram.  After the metadata rotation is removed only a small
    # residual is expected, so +/- 1 degree is searched.
    bound = np.deg2rad(1.0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='estimate_det_rotation:')
        residual = gc.estimate_det_rotation(model, sino, bounds=(-bound, bound))
    edge_px = abs(residual.value) * sino.shape[2] / 2.0
    gain = float(residual.scores[int(np.argmin(np.abs(residual.candidates)))] / max(residual.score, 1e-30))
    # A residual is reported when it moves the edge channels by a pixel or more, scores more than 1.1 times better
    # than no rotation, and comes from a clean search.  Noise shows up as several local minima or as a minimum at the
    # bound, and a minimum at the bound needs a gain above 1.5.
    notes = residual.reduction['search_notes']
    several_minima = any('local minima' in note for note in notes)
    at_bound = 'the coarse minimum sits at an edge of the bounds' in notes
    rotation_left = edge_px >= 1.0 and not several_minima and gain > (1.5 if at_bound else 1.1)

    if verbose > 0:
        print(f'   det_channel_offset: metadata {start:+.4f} mm -> data {offset.value:+.4f} mm '
              f'({(offset.value - start) / delta:+.2f} channels), {offset.reduction["pairs_kept"]} of '
              f'{offset.reduction["num_pairs"]} view pairs kept; det_row_offset {row_start:+.4f} -> {row:+.4f} mm')
        print(f'   det_rotation {det_rotation:+.3e} rad from the metadata kept; residual rotation '
              f'{np.rad2deg(residual.value):+.4f} deg ({edge_px:.2f} px at the edge channels, score {gain:.3f}x better '
              f'than no rotation)' + (f'; search notes: {"; ".join(notes)}' if notes else ''))
    if rotation_left:
        warnings.warn(f'calibrate_volumax_geometry: a detector rotation of {np.rad2deg(residual.value):+.4f} deg '
                      f'remains after the metadata rotation was removed ({edge_px:.1f} px at the edge channels, score '
                      f'{gain:.2f}x better than no rotation).  The metadata rotation assumes that the rotation axis is '
                      'the vertical axis of the scanner; check the slices far from the central plane.')
    return dict(det_channel_offset=offset, det_rotation_residual=residual)
