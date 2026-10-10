"""Building a sinogram from the object, blank, and dark scans."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
import mbirtorch as mt

from . import _pipeline
from .corrections import _rotation_kernel

__all__ = ['scan_to_sino', 'crop_view_data', 'correct_zinger_pixels']


def _reference_scan_to_host(scan):
    """Bring a blank or dark scan to a host numpy array.  A numpy input is returned unchanged.

    The reference scans are reduced on the host with numpy functions, and those functions do not
    work correctly on a torch tensor."""
    return scan.detach().cpu().numpy() if torch.is_tensor(scan) else scan


def _put_in_slice(array, flat_indices, value):
    """
    Similar to numpy.put(array, flat_indices, value), which would produce array.flat[flat_indices] = value.
    However, this function requires that array have an extra leading dimension, and that the value for a given
    index is copied across that dimension.  Roughly, array[:, flat_indices] = value

    The input is not modified; a new tensor is returned.

    Args:
        array (tensor): Tensor of dimension n+1
        flat_indices (array of int): Indices obtained using ravel_multi_index using array.shape[1:]
        value (float or tensor): Values to be copied in.  Must be able to broadcast to array.shape[1:]

    Returns:
        tensor
    """
    array_shape = array.shape
    flat_indices = torch.as_tensor(np.asarray(flat_indices), dtype=torch.int64, device=array.device)
    # Clip out of range indices.  A clipped position receives the same value as an in-range write.
    flat_indices = torch.clamp(flat_indices, 0, int(np.prod(array_shape[1:])) - 1)
    array = array.reshape(array_shape[0], -1).clone()
    array[:, flat_indices] = value
    array = array.reshape(array_shape)
    return array


def _box3x3_sum(x):
    """Sum over each 3x3 (row, channel) window of every view, zero padded at the detector edges.

    ``x`` has shape (num_views, num_det_rows, num_det_channels).  The window spans the two detector
    axes."""
    return F.avg_pool2d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1,
                        count_include_pad=True).squeeze(1) * 9.0


def _interpolate_fill_pass(sino):
    """Replace each NaN pixel with the mean of its finite 3x3 neighbors within the same view.

    Finite pixels are unchanged.  A NaN with no finite neighbor stays NaN and is filled on a later
    pass."""
    is_nan = torch.isnan(sino)
    valid = (~is_nan).to(sino.dtype)
    filled = torch.where(is_nan, torch.zeros((), dtype=sino.dtype, device=sino.device), sino)
    neighbor_sum = _box3x3_sum(filled)
    neighbor_count = _box3x3_sum(valid)
    neighbor_mean = neighbor_sum / torch.clamp(neighbor_count, min=1.0)   # avoid 0/0 where no valid neighbor
    return torch.where(is_nan & (neighbor_count > 0), neighbor_mean, sino)


def _fill_nan_pixels(sino, num_passes=3):
    """Fill every NaN pixel with the mean of its finite 3x3 neighbors within the same view.

    Each of the ``num_passes`` passes fills the NaN region inward by one pixel.  A pixel that is
    still NaN after the last pass is set to 0, and the function warns."""
    for _ in range(num_passes):
        sino = _interpolate_fill_pass(sino)
    num_residual = int(torch.sum(torch.isnan(sino)).item())
    if num_residual > 0:
        warnings.warn(f"NaN-fill: {num_residual} pixel(s) left after the fill passes (defective/zinger "
                      f"clusters wider than the fill reach); setting them to 0. Increase num_passes if "
                      f"this is unexpected.")
    return torch.nan_to_num(sino, nan=0.0)


def _zinger_fill(sino, zinger_threshold, num_passes=3):
    """Mark zinger pixels (``value < zinger_threshold``) and non-finite values NaN, then fill them from
    their finite 3x3 in-view neighbors.  ``zinger_threshold`` is a precomputed scalar."""
    nan = torch.tensor(float('nan'), dtype=sino.dtype, device=sino.device)
    sino = torch.where(torch.isfinite(sino), sino, nan)        # +-inf / NaN -> NaN
    sino = torch.where(sino < zinger_threshold, nan, sino)     # flag zingers
    return _fill_nan_pixels(sino, num_passes)


def _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean, flat_indices):
    """Per-view-batch transmission kernel (pure device-tensor op).

    Computes ``-log`` of the dark-corrected, blank-normalized object scan, sets shared defective
    pixels to NaN, and interpolates the NaNs.  ``blank_minus_dark = |blank_mean - dark_mean|`` is
    precomputed by the caller (it does not vary across batches).
    """
    device = obj_batch.device
    blank_minus_dark = torch.as_tensor(np.asarray(blank_minus_dark, dtype=np.float32), device=device)
    dark_scan_mean = torch.as_tensor(np.asarray(dark_scan_mean, dtype=np.float32), device=device)
    obj_batch = torch.abs(obj_batch - dark_scan_mean)
    # A non-positive ratio becomes NaN, so the fill below removes it along with the defective pixels.
    ratio = obj_batch / blank_minus_dark
    nan = torch.tensor(float('nan'), dtype=ratio.dtype, device=device)
    sino_batch = torch.where(ratio > 0, -torch.log(torch.where(ratio > 0, ratio, torch.ones_like(ratio))), nan)
    if flat_indices is not None:
        sino_batch = _put_in_slice(sino_batch, flat_indices, float('nan'))
    # Any remaining non-finite value also becomes NaN, so the fill removes it too.
    sino_batch = torch.where(torch.isfinite(sino_batch), sino_batch, nan)
    return _fill_nan_pixels(sino_batch)


def compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array=(), batch_size=90,
                              devices=None):
    """
    Compute sinogram from object, blank, and dark scans.

    This function computes a sinogram by taking the negative logarithm of the normalized transmission image:
    `-log((obj - dark) / (blank - dark))`. It supports correction for defective pixels.

    The invalid sinogram entries are defined as:
    - Any values resulting in `inf` or `NaN`
    - Any indices listed in the `defective_pixel_array` (if provided)

    Accepted forms: the scans may be NumPy arrays or torch tensors (a tensor blank or dark scan is
    brought to the host and reduced with NumPy).  The result is always a NumPy array on the host.
    Any GPU use is internal -- the views are moved to a device one batch at a time and each batch's
    result is brought back.  An array in the divided device form (a ``Shards`` container, as
    produced by the multi-GPU projectors) is not accepted; gather it to the host first with
    ``shards.gather()``.

    Args:
        obj_scan (numpy array or tensor):
            A 3D object scan of shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray or tensor):
            A 3D blank scan of shape (num_blank_scans, num_det_rows, num_det_channels).
            If `num_blank_scans > 1`, a pixel-wise mean will be computed.
        dark_scan (ndarray or tensor):
            A 3D dark scan of shape (num_dark_scans, num_det_rows, num_det_channels).
            If `num_dark_scans > 1`, a pixel-wise mean will be computed.
        defective_pixel_array (ndarray or tuple, optional):
            An array of defective pixel indices, one (row_idx, channel_idx) pair per row;
            these pixels are treated as defective in every view.
            Defaults to the empty tuple `()`, meaning no known defective pixels; invalid pixels
            are then inferred from `NaN` or `inf` values alone. Do not pass `None`.
        batch_size (int):
            Number of views to process in each GPU batch.
        devices (sequence or None):
            devices to spread the views over.  None (default) uses all visible CUDA devices,
            capped by ``MBIRTORCH_NUM_DEVICES`` when that is set, or the default device when
            there are none.

    Returns:
        numpy.ndarray:
            The computed sinogram, with shape (num_views, num_det_rows, num_det_channels).

    Raises:
        TypeError: If any of the scans is in the divided device form.
    """
    _pipeline.reject_shards('compute_sino_transmission', obj_scan=obj_scan, blank_scan=blank_scan,
                           dark_scan=dark_scan, defective_pixel_array=defective_pixel_array)
    blank_scan = _reference_scan_to_host(blank_scan)
    dark_scan = _reference_scan_to_host(dark_scan)

    blank_scan_mean = np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan_mean = np.mean(dark_scan, axis=0, keepdims=True)
    blank_minus_dark = np.abs(blank_scan_mean - dark_scan_mean)

    if len(defective_pixel_array) > 0:
        defective_pixel_array = np.asarray(defective_pixel_array)
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, obj_scan.shape[1:]).astype(np.int64)
    else:
        defective_pixel_array = ()
        flat_indices = None

    sino = _pipeline.map_view_batches(
        obj_scan,
        lambda obj_batch: _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean,
                                               flat_indices),
        batch_size, devices=_pipeline.permitted_devices(devices))
    print("Sinogram computation complete.")
    return sino


def _downsample_obj_kernel(obj_batch, flat_indices, new_size1, new_size2, block_shape):
    """Per-view-batch object-scan downsample kernel (pure device-tensor op): set defective pixels to
    NaN, crop to a block-divisible size, and block-average with nanmean."""
    if flat_indices is not None:
        obj_batch = _put_in_slice(obj_batch, flat_indices, float('nan'))
    obj_batch = obj_batch[:, 0:new_size1, 0:new_size2]
    obj_batch = obj_batch.reshape((obj_batch.shape[0],) + block_shape)
    return torch.nanmean(obj_batch, dim=(2, 4))


def _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array=()):
    """Downsample the blank and dark scans on the host.

    Defective pixels are set to NaN, the scans are cropped to a size divisible by the downsample
    factor, and each block is averaged.  The blank and dark scans have the detector dimensions of
    the object scan, so the block parameters that :func:`_downsample_obj_kernel` needs are computed
    here as well.

    Returns:
        (blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape)
    """
    # Set defective pixels to NaN for use with nanmean.  The blank and dark scans may have
    # different numbers of views, so each is looped over its own leading dimension.
    if len(defective_pixel_array) > 0:
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, blank_scan.shape[1:]).astype(np.int64)
        for i in range(blank_scan.shape[0]):
            np.put(blank_scan[i], flat_indices, np.nan)
        for i in range(dark_scan.shape[0]):
            np.put(dark_scan[i], flat_indices, np.nan)
    else:
        flat_indices = None

    # Crop the scan if the size is not divisible by downsample_factor.
    new_size1 = downsample_factor[0] * (blank_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (blank_scan.shape[2] // downsample_factor[1])

    blank_scan = blank_scan[:, 0:new_size1, 0:new_size2]
    dark_scan = dark_scan[:, 0:new_size1, 0:new_size2]

    block_shape = (blank_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                   blank_scan.shape[2] // downsample_factor[1], downsample_factor[1])

    # A block of all NaNs gives NaN.
    blank_scan = np.stack([
        np.nanmean(scan.reshape(block_shape), axis=(1, 3))
        for scan in blank_scan
    ], axis=0)

    dark_scan = np.stack([
        np.nanmean(scan.reshape(block_shape), axis=(1, 3))
        for scan in dark_scan
    ], axis=0)

    # A downsampled pixel is defective when every pixel in its block was bad.
    nan_mask = np.isnan(blank_scan).any(axis=0)  # Combine across all views
    defective_pixel_array = np.argwhere(nan_mask)
    if len(defective_pixel_array) == 0:
        defective_pixel_array = ()

    # flat_indices stays on the host.  The kernel moves it to the device of each batch.
    return blank_scan, dark_scan, defective_pixel_array, flat_indices, new_size1, new_size2, block_shape


def downsample_view_data(obj_scan, blank_scan, dark_scan, downsample_factor, defective_pixel_array=(), batch_size=90,
                         devices=None):
    """
    Performs down-sampling of the scan images in the detector plane.
    This is done for the object, blank_scan, and dark_scan data,
    and the defective_pixel_array is updated to reflect the new pixel grid.

    Accepted forms: the scans may be NumPy arrays or torch tensors, and the returned scans are
    always NumPy arrays on the host.  Any GPU use is internal -- the object scan's views are moved
    to a device one batch at a time and each batch's result is brought back.  An array in the
    divided device form (a ``Shards`` container, as produced by the multi-GPU projectors) is not
    accepted; gather it to the host first with ``shards.gather()``.

    Args:
        obj_scan (ndarray): A stack of sinograms. 3D NumPy array of shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray): Blank scan(s). 3D NumPy array of shape (num_blank_views, num_det_rows, num_det_channels).
        dark_scan (ndarray): Dark scan(s). 3D NumPy array of shape (num_dark_views, num_det_rows, num_det_channels).
        downsample_factor (tuple of int): Two integers defining the down-sample factor. Must be ≥ 1 in each dimension.
        defective_pixel_array (ndarray): Array of shape (num_defective_pixels, 2) indicating defective pixel coordinates.
        batch_size (int): Number of views to include in one batch. Controls memory usage.
        devices (sequence or None): devices to spread the views over.  None (default) uses all
            visible CUDA devices, capped by ``MBIRTORCH_NUM_DEVICES`` when that is set, or the
            default device when there are none.

    Notes:
        This function supports both singleton blank/dark scans (shape (1, H, W)) and multi-view scans
        (shape (N, H, W), where N > 1). Downsampling is applied independently to each view.

    Returns:
        tuple:
        - **obj_scan** (numpy.ndarray): Downsampled object scan. Shape (num_views, new_rows, new_cols).
        - **blank_scan** (numpy.ndarray): Downsampled blank scan(s). Shape (num_blank_views, new_rows, new_cols).
        - **dark_scan** (numpy.ndarray): Downsampled dark scan(s). Shape (num_dark_views, new_rows, new_cols).
        - **defective_pixel_array** (numpy.ndarray): Updated defective pixel coordinates. Shape (N_def, 2).

    Raises:
        TypeError: If any of the scans is in the divided device form.
    """
    _pipeline.reject_shards('downsample_view_data', obj_scan=obj_scan, blank_scan=blank_scan,
                           dark_scan=dark_scan, defective_pixel_array=defective_pixel_array)
    blank_scan = _reference_scan_to_host(blank_scan)
    dark_scan = _reference_scan_to_host(dark_scan)

    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)
    assert (downsample_factor[0] >= 1 and downsample_factor[1] >= 1), 'factor({}) along each dimension should be greater or equal to 1'.format(downsample_factor)

    blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape = \
        _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array)

    obj_scan = _pipeline.map_view_batches(
        obj_scan,
        lambda b: _downsample_obj_kernel(b, obj_flat_indices, new_size1, new_size2, block_shape),
        batch_size, devices=_pipeline.permitted_devices(devices))

    return obj_scan, blank_scan, dark_scan, defective_pixel_array


def scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array=(),
                 downsample_factor=(1, 1), det_rotation=0.0,
                 batch_size=90, devices=None):
    """
    Compute the sinogram from the object, blank, and dark scans.

    The views are downsampled by block averaging, each is converted to ``-log(|obj - dark| /
    |blank - dark|)`` with the blank and dark scans averaged over their views, the defective and
    non-positive pixels are filled from their neighbors, and the detector rotation is removed.  A
    step is skipped at its default.  Takes numpy arrays or tensors and returns a numpy array.

    Args:
        obj_scan (numpy array or tensor): Object scan, shape (num_views, num_det_rows, num_det_channels).
        blank_scan (numpy array or tensor): Blank scan or scans, shape (num_blank, num_det_rows, num_det_channels).
        dark_scan (numpy array or tensor): Dark scan or scans, shape (num_dark, num_det_rows, num_det_channels).
        defective_pixel_array (numpy.ndarray or tuple, optional): The (row, col) coordinates of the
            defective pixels, shape (num_defective, 2), or () for none.  Defaults to ().
        downsample_factor (tuple[int, int], optional): Detector (row, channel) downsampling.  Defaults to (1, 1).
        det_rotation (float, optional): Detector rotation in radians.  Defaults to 0.0.
        batch_size (int, optional): Views processed at a time.  Defaults to 90.
        devices (sequence or None, optional): Devices to spread the views over.  None uses all visible
            CUDA devices, capped by ``MBIRTORCH_NUM_DEVICES`` when it is set, or the default device
            when there are none.  Defaults to None.

    Returns:
        numpy.ndarray: The sinogram, shape (num_views, num_det_rows, num_det_channels) after
        downsampling.
    """
    _pipeline.reject_shards('scan_to_sino', obj_scan=obj_scan, blank_scan=blank_scan,
                           dark_scan=dark_scan, defective_pixel_array=defective_pixel_array)
    blank_scan = _reference_scan_to_host(blank_scan)
    dark_scan = _reference_scan_to_host(dark_scan)

    devices = _pipeline.permitted_devices(devices)
    obj_flat_indices = new_size1 = new_size2 = block_shape = None
    do_downsample = downsample_factor[0] * downsample_factor[1] > 1
    if do_downsample:
        blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape = \
            _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array)

    blank_scan_mean = np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan_mean = np.mean(dark_scan, axis=0, keepdims=True)
    blank_minus_dark = np.abs(blank_scan_mean - dark_scan_mean)

    # The defective pixel indices are raveled against the detector grid the kernel sees.
    trans_det_shape = blank_scan.shape[1:]
    if len(defective_pixel_array) > 0:
        defective_pixel_array = np.asarray(defective_pixel_array)
        trans_flat_indices = np.ravel_multi_index(defective_pixel_array.T, trans_det_shape).astype(np.int64)
    else:
        defective_pixel_array = ()
        trans_flat_indices = None

    do_rotation = det_rotation != 0.0

    # Each view batch runs one fused kernel that downsamples, converts to transmission, and rotates.
    def fused_kernel(obj_batch):
        if do_downsample:
            obj_batch = _downsample_obj_kernel(obj_batch, obj_flat_indices, new_size1, new_size2, block_shape)
        sino_batch = _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean,
                                          trans_flat_indices)
        if do_rotation:
            sino_batch = _rotation_kernel(sino_batch, det_rotation)
        return sino_batch

    sino = _pipeline.map_view_batches(obj_scan, fused_kernel, batch_size, devices=devices)
    print("Sinogram computation complete.")
    return sino


def crop_view_data(obj_scan, blank_scan, dark_scan, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, defective_pixel_array=()):
    """
    Crop the object, blank, and dark scans and update the defective pixel list to match.

    The same number of pixels is cropped from the left and the right, so the detector center is
    kept.  Defective pixels outside the crop are dropped and the rest are re-indexed to the cropped
    scans.

    Args:
        obj_scan (numpy.ndarray): Object scan, shape (num_views, num_det_rows, num_det_channels).
        blank_scan (numpy.ndarray): Blank scan or scans, shape (num_blank, num_det_rows, num_det_channels).
        dark_scan (numpy.ndarray): Dark scan or scans, shape (num_dark, num_det_rows, num_det_channels).
        crop_pixels_sides (int, optional): Pixels to crop from each side.  Defaults to 0.
        crop_pixels_top (int, optional): Pixels to crop from the top (low row indices).  Defaults to 0.
        crop_pixels_bottom (int, optional): Pixels to crop from the bottom (high row indices).  Defaults to 0.
        defective_pixel_array (numpy.ndarray or tuple, optional): The (row, col) coordinates of the
            defective pixels, shape (num_defective, 2), or () for none.  Defaults to ().

    Returns:
        tuple: ``(obj_scan, blank_scan, dark_scan, defective_pixel_array)``, cropped.

    Raises:
        AssertionError: If a crop is negative, the top and bottom crops together reach the detector
            height, or twice the side crop reaches the detector width.
    """
    assert (0 <= crop_pixels_sides < obj_scan.shape[2] // 2 and
            0 <= crop_pixels_top and 0 <= crop_pixels_bottom and crop_pixels_top + crop_pixels_bottom < obj_scan.shape[1]), \
        ('crop_pixels should be nonnegative integers so that crop_pixels_top + crop_pixels_bottom < view height and'
         ' 2*crop_pixels_sides < view width')

    Nr_lo = crop_pixels_top
    Nr_hi = obj_scan.shape[1] - crop_pixels_bottom

    Nc_lo = crop_pixels_sides
    Nc_hi = obj_scan.shape[2] - crop_pixels_sides

    obj_scan = obj_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    blank_scan = blank_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    dark_scan = dark_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]

    if len(defective_pixel_array) > 0:
        in_bounds = (defective_pixel_array[:, 0] >= Nr_lo) & (defective_pixel_array[:, 0] < Nr_hi) & \
                    (defective_pixel_array[:, 1] >= Nc_lo) & (defective_pixel_array[:, 1] < Nc_hi)
        defective_pixel_array = defective_pixel_array[in_bounds]
        defective_pixel_array -= np.array([Nr_lo, Nc_lo]).reshape(1, 2)

    return obj_scan, blank_scan, dark_scan, defective_pixel_array


def _zinger_threshold(sino, zinger_pixel_ratio, max_views_to_use=20):
    """Zinger-detection threshold = ``-zinger_pixel_ratio * RMS(sino over its support)``, estimated
    from a subsample of at most ``max_views_to_use`` views.  Returns a negative Python float."""
    sino_sub = mt.TomographyModel.subsample_views(sino, max_views_to_use)
    sino_indicator = np.asarray(mt.TomographyModel._get_sino_indicator(sino_sub))
    typical_sino_value = float(np.average(np.asarray(sino_sub) ** 2, None, sino_indicator) ** 0.5)
    return -zinger_pixel_ratio * typical_sino_value


def correct_zinger_pixels(sino, zinger_pixel_ratio=0.1, num_passes=3, batch_size=90, devices=None,
                          max_views_to_use=20):
    """
    Replace the zinger pixels of a sinogram with the mean of their neighbors.

    A zinger is a pixel whose value is below ``-zinger_pixel_ratio`` times the RMS value of the
    sinogram.  The RMS value is estimated from ``max_views_to_use`` views, so the background offset
    must already be removed; see :func:`correct_background_offset`.  Zingers and non-finite pixels
    are replaced by the mean of their finite 3x3 neighbors in the same view, in ``num_passes``
    passes, so a cluster up to that radius is filled.  A pixel still unfilled is set to 0 with a
    warning.  Takes a numpy array or a tensor and returns a numpy array.

    Args:
        sino (numpy array or tensor): Sinogram with its background offset removed, shape (num_views,
            num_det_rows, num_det_channels).
        zinger_pixel_ratio (float, optional): The detection ratio.  Defaults to 0.1.
        num_passes (int, optional): Fill passes.  Defaults to 3.
        batch_size (int, optional): Views processed at a time.  Defaults to 90.
        devices (sequence or None, optional): Devices to spread the views over.  None uses all visible
            CUDA devices, capped by ``MBIRTORCH_NUM_DEVICES`` when it is set, or the default device
            when there are none.  Defaults to None.
        max_views_to_use (int, optional): Views sampled for the RMS estimate.  Defaults to 20.

    Returns:
        numpy.ndarray: The corrected sinogram, the shape of ``sino``.
    """
    _pipeline.reject_shards('correct_zinger_pixels', sino=sino)

    # The threshold is computed once on the whole sinogram, so the result does not depend on the
    # number of devices.
    zinger_threshold = _zinger_threshold(sino, zinger_pixel_ratio, max_views_to_use)
    kernel = lambda b: _zinger_fill(b, zinger_threshold, num_passes)
    return _pipeline.map_view_batches(sino, kernel, batch_size,
                                     devices=_pipeline.permitted_devices(devices))
