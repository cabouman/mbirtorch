"""Helpers shared by the scanner loaders: image readers, geometry vectors, detector crops, unit
conversion, and the model built from a loader's parameter dictionaries."""

import glob
import os

import numpy as np
import mbirtorch as mt


def _normalize_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 and normalize if it is an integer dtype.

    - If `imgs.dtype` is an integer type, cast to float32 and divide by the max value for that dtype.
    - Otherwise, cast to float32 without scaling.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: float32 array, normalized to [0, 1] if input was integer.
    """
    if np.issubdtype(img.dtype, np.integer):
        maxval = np.iinfo(img.dtype).max
        return img.astype(np.float32) / maxval
    return img.astype(np.float32)


def read_tif_img(img_path):
    """
    Read a TIFF image and return it as a float32 array with the shape stored in the file.

    An integer-valued image is scaled to the range [0, 1] by dividing by the largest value its
    integer type can hold. A floating-point image is cast to float32 without scaling.

    Args:
        img_path (str): Path to the image file. The file must be readable by `tifffile`.

    Returns:
        np.ndarray: Image data as a float32 NumPy array. Can be 2D or higher dimensional depending on the input.
    """
    import tifffile
    img = tifffile.imread(img_path)
    img = _normalize_to_float32(img)
    return img


def read_tif_stack_dir(scan_dir, view_ids=None):
    """Reads a tif stack of scan images from a directory. This function is a subroutine to `load_scans_and_params`.

    Args:
        scan_dir (string): Path to a ConeBeam Scan directory.
            Example: "<absolute_path_to_dataset>/Radiographs"
        view_ids (ndarray of ints, optional, default=None): List of view indices to specify which scans to read.
    Returns:
        ndarray (float): 3D numpy array, (num_views, num_det_rows, num_det_channels). A stack of scan images.
    """

    import tifffile
    img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tif')))
    if len(img_path_list) == 0:
        img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tiff')))  # Assume files are '.tif' but check '.tiff' if not

    if len(img_path_list) == 0:
        raise FileNotFoundError('No scan images found in directory: {}'.format(scan_dir))

    # This assumes that all the views are labeled sequentially.
    if view_ids is None:
        view_ids = np.arange(len(img_path_list))
    else:
        max_view_id = np.amax(view_ids)
        if max_view_id >= len(img_path_list):
            raise FileNotFoundError('The max view index was given as {}, but there are only {} views in {}'.format(max_view_id, len(img_path_list), scan_dir))
    img_path_list = [img_path_list[idx] for idx in view_ids]

    output_views = tifffile.imread(img_path_list, ioworkers=48, maxworkers=8)
    output_views = _normalize_to_float32(output_views)

    return output_views


def unit_vector(v):
    """ Normalize v. Returns v/||v|| """
    return v / np.linalg.norm(v)


def project_vector_to_vector(u1, u2):
    """ Projects the vector u1 onto the vector u2. Returns the vector <u1|u2>.
    """
    u2 = unit_vector(u2)
    u1_proj = np.dot(u1, u2)*u2
    return u1_proj


def detect_blank_margins(sino, safety_buffer=20, max_views_to_use=20):
    """
    Detect how many blank detector rows/channels frame the object in a sinogram.

    The detected amounts feed :func:`apply_detector_crop` to assemble the automatic-crop step.
    Detection uses a support mask computed over a subsample of views, so it stays cheap on large
    sinograms.

    Args:
        sino (np.ndarray): Sinogram, shape ``(num_views, num_det_rows, num_det_channels)``.
        safety_buffer (int, optional): Blank margin, in pixels, to keep on each side of the detected
            object region. Defaults to 20.
        max_views_to_use (int, optional): Cap on the number of views sampled for detection.
            Defaults to 20.

    Returns:
        tuple: ``(crop_top, crop_bottom, crop_left, crop_right)`` -- detector rows to crop from the
        top and bottom, and detector channels to crop from the left and right.
    """
    # Detection is statistical across views, so a subsample of views keeps it fast on a large sinogram.
    sino = mt.TomographyModel.subsample_views(sino, max_views_to_use)

    sino_indicator_mask = mt.TomographyModel._get_sino_indicator(sino)

    union_mask = np.any(np.asarray(sino_indicator_mask), axis=0)

    rows = np.any(union_mask, axis=1)
    cols = np.any(union_mask, axis=0)

    # argmax on a binary array returns the index of the first 1.
    top_width = np.argmax(rows)
    bottom_width = np.argmax(rows[::-1])
    left_width = np.argmax(cols)
    right_width = np.argmax(cols[::-1])

    crop_pixels_top = max(top_width - safety_buffer, 0)
    crop_pixels_bottom = max(bottom_width - safety_buffer, 0)
    crop_pixels_left = max(left_width - safety_buffer, 0)
    crop_pixels_right = max(right_width - safety_buffer, 0)

    return crop_pixels_top, crop_pixels_bottom, crop_pixels_left, crop_pixels_right


def apply_detector_crop(required_params, optional_params, crop_top, crop_bottom, crop_left, crop_right):
    """
    Update the geometry for a detector crop: shrink the sinogram shape and shift the detector
    offsets by the amount the crop moves the detector center.

    This function changes only the parameters, not the data.  The caller must slice the array with
    the same crop amounts.

    Args:
        required_params (dict): Constructor parameters, read for ``sinogram_shape``.
        optional_params (dict): Parameters applied with ``set_params``.  ``det_row_offset`` and
            ``det_channel_offset`` are compensated when the geometry has them, using the detector
            pitches ``delta_det_row`` and ``delta_det_channel`` (each 1.0 if unset).
        crop_top (int): Detector rows removed from the top.
        crop_bottom (int): Detector rows removed from the bottom.
        crop_left (int): Detector channels removed from the left.
        crop_right (int): Detector channels removed from the right.

    Returns:
        tuple: new ``(required_params, optional_params)`` dicts, with the reduced ``sinogram_shape``
        and the compensated detector offsets.  The input dicts are not modified, so the caller must
        use the returned dicts.

    Raises:
        AssertionError: If any crop amount is negative, or if ``crop_top + crop_bottom >= num_det_rows``,
            or if ``crop_left + crop_right >= num_det_channels``.
    """
    num_views, num_det_rows, num_det_channels = required_params['sinogram_shape']
    # A crop as large as the detector dimension would give a negative sinogram_shape.
    assert (crop_top >= 0 and crop_bottom >= 0 and crop_left >= 0 and crop_right >= 0 and
            crop_top + crop_bottom < num_det_rows and crop_left + crop_right < num_det_channels), \
        ('apply_detector_crop: crop amounts must be nonnegative with crop_top + crop_bottom < num_det_rows'
         ' and crop_left + crop_right < num_det_channels (got top={}, bottom={}, left={}, right={} for a'
         ' {}x{} detector).'.format(crop_top, crop_bottom, crop_left, crop_right, num_det_rows, num_det_channels))
    # Work on copies so that the caller's dicts are not modified.
    required_params = dict(required_params)
    optional_params = dict(optional_params)
    required_params['sinogram_shape'] = (int(num_views),
                                         int(num_det_rows - crop_top - crop_bottom),
                                         int(num_det_channels - crop_left - crop_right))

    # Shift each detector offset by the move of the detector center.  Only a geometry that carries
    # the offset gets it compensated.
    if 'det_row_offset' in optional_params:
        delta_det_row = optional_params.get('delta_det_row', 1.0)
        optional_params['det_row_offset'] += (crop_bottom - crop_top) / 2 * delta_det_row
    if 'det_channel_offset' in optional_params:
        delta_det_channel = optional_params.get('delta_det_channel', 1.0)
        optional_params['det_channel_offset'] += (crop_right - crop_left) / 2 * delta_det_channel

    return required_params, optional_params


def _auto_crop_sino(sino, required_params, optional_params, safety_buffer=20):
    """
    Detect and remove blank sinogram margins, updating the detector-plane geometry to match.

    This combines :func:`detect_blank_margins`, slicing of the sinogram, and
    :func:`apply_detector_crop`.  A ``recon_slice_offset`` in the dicts is moved by the change in
    the automatic center.  Run this before ``build_model``.

    Args:
        sino (np.ndarray): Sinogram, shape ``(num_views, num_det_rows, num_det_channels)``.
        required_params (dict): Constructor parameters, including ``sinogram_shape``.
        optional_params (dict): Parameters applied with ``set_params`` (detector pitches/offsets).
        safety_buffer (int, optional): Blank margin, in pixels, to keep around the detected object
            region. Defaults to 20.

    Returns:
        tuple: ``(sino, required_params, optional_params)`` with the sinogram cropped and the
        geometry updated consistently (``required_params['sinogram_shape'] == sino.shape``).
    """
    crop_top, crop_bottom, crop_left, crop_right = detect_blank_margins(sino, safety_buffer)
    sino = sino[:, crop_top:sino.shape[1] - crop_bottom, crop_left:sino.shape[2] - crop_right]
    # A supplied recon_slice_offset keeps its place relative to the automatic center.
    supplied_offset = optional_params.get('recon_slice_offset')
    if supplied_offset is not None:
        _, automatic_before = mt.utilities._automatic_recon_geometry(required_params, optional_params, None)
    required_params, optional_params = apply_detector_crop(
        required_params, optional_params, crop_top, crop_bottom, crop_left, crop_right)
    if supplied_offset is not None and automatic_before is not None:
        _, automatic_after = mt.utilities._automatic_recon_geometry(required_params, optional_params, None)
        optional_params = dict(optional_params)
        optional_params['recon_slice_offset'] = float(supplied_offset) + (automatic_after - automatic_before)
    return sino, required_params, optional_params


def apply_config_crop(num_det_rows, num_det_channels, det_row_offset, det_channel_offset,
                      delta_det_row, delta_det_channel, *,
                      crop_pixels_top, crop_pixels_bottom, crop_pixels_sides):
    """
    Apply a configuration (manual) detector crop to a scanner loader's SCALAR geometry values.

    Scalar-in / scalar-out adapter around :func:`apply_detector_crop` -- the scanner loaders' configuration crop
    acts on loose scalars (not a param dict) at conversion time, so this packs them, applies the shared
    detector-plane crop (shape reduction + offset compensation for an asymmetric top/bottom crop), and
    unpacks the results.  A detector crop does not change the number of views, so it is neither taken nor
    returned.  ``crop_pixels_sides`` crops each lateral side symmetrically.

    Args:
        num_det_rows (int): Detector rows before the crop.
        num_det_channels (int): Detector channels before the crop.
        det_row_offset (float): Detector row offset (ALU) before the crop.
        det_channel_offset (float): Detector channel offset (ALU) before the crop.
        delta_det_row (float): Detector row pitch (scales the row-offset compensation).
        delta_det_channel (float): Detector channel pitch (scales the channel-offset compensation).
        crop_pixels_top (int): Detector rows cropped from the top.
        crop_pixels_bottom (int): Detector rows cropped from the bottom.
        crop_pixels_sides (int): Detector channels cropped from EACH lateral side.

    Returns:
        tuple: ``(num_det_rows, num_det_channels, det_row_offset, det_channel_offset)`` after the crop.
    """
    required, optional = apply_detector_crop(
        {'sinogram_shape': (0, num_det_rows, num_det_channels)},   # 0 = num_views placeholder (crop preserves it)
        {'delta_det_row': delta_det_row, 'delta_det_channel': delta_det_channel,
         'det_row_offset': det_row_offset, 'det_channel_offset': det_channel_offset},
        crop_pixels_top, crop_pixels_bottom, crop_pixels_sides, crop_pixels_sides)
    _, num_det_rows, num_det_channels = required['sinogram_shape']
    return num_det_rows, num_det_channels, optional['det_row_offset'], optional['det_channel_offset']


def finalize_model(sino, required_params, optional_params, *, auto_crop=False, safety_buffer=20):
    """
    Build a model from a scanner loader's sinogram and parameter dictionaries.

    This is the last step of each scanner loader's ``get_sino_and_model``.  With ``auto_crop`` the blank
    margins of the sinogram are removed first and the geometry is adjusted to match.

    Args:
        sino (numpy.ndarray): The sinogram.
        required_params (dict): The model constructor arguments plus a ``geometry_type`` entry that
            names the model class.
        optional_params (dict): Arguments for ``set_params``.
        auto_crop (bool, optional): If True, remove the blank margins of the sinogram.  Defaults to False.
        safety_buffer (int, optional): Blank margin in pixels kept when auto-cropping.  Defaults to 20.

    Returns:
        tuple: ``(sino, model)``: the sinogram, cropped if asked, and the model with its parameters set.
    """
    if auto_crop:
        sino, required_params, optional_params = _auto_crop_sino(sino, required_params, optional_params, safety_buffer)
    model = mt.build_model(required_params, optional_params)
    return sino, model


_ALU_UNIT_CONVERSION = {'um': 1.0, 'mm': 1000.0, 'cm': 1e4, 'm': 1e6}


def to_alu(value, from_unit, alu_unit):
    """
    Rescale a physical ``value`` from ``from_unit`` to ALU, where 1 ALU = 1 unit of ``alu_unit``.

    Args:
        value (float or array): The value(s) to rescale.
        from_unit (str): Unit of ``value`` (``'um'``, ``'mm'``, ``'cm'``, ``'m'``).
        alu_unit (str): The unit defining 1 ALU.

    Returns:
        ``value * conversion[from_unit] / conversion[alu_unit]``.
    """
    return value * _ALU_UNIT_CONVERSION[from_unit] / _ALU_UNIT_CONVERSION[alu_unit]
