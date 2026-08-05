"""Partitions, masks, and weights, ported from mbirjax.vcd_utils.

The partition generators are numpy-for-numpy ports with the SAME global
np.random call sequence as mbirjax, so a seeded mbirtorch recon draws the
identical subsets (and subset order) as a seeded mbirjax recon -- the property
the cross-framework convergence-parity gate rests on.  Not ported: the grid
and blue-noise partition variants and the MAR weights.
"""

import warnings

import numpy as np
import torch


def get_2d_ror_mask(recon_shape, *, use_ror_mask=True, crop_radius_pixels=0,
                    crop_radius_fraction=0.0):
    """
    Get a binary mask for the region of reconstruction.  By default, the mask is
    an ellipse inscribed in the edges of the 2D recon_shape[0:2].  The size of
    this ellipse can be reduced by setting crop_radius_pixels or
    crop_radius_fraction, either of which is subtracted from the ellipse axes.
    Only one of these can be nonzero.  Negative values are clipped to 0.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices), or just
            (rows, columns).
        use_ror_mask (default is True):
            False:
                No mask.
            True:
                The mask is an ellipse inscribed in the reconstruction volume.
            2D array:
                Use a custom binary mask.  Must have shape recon_shape[:2].
        crop_radius_pixels (int): Number of column-pixel-equivalent pixels to
            subtract from the radius.
        crop_radius_fraction (float): Fraction to subtract from each axis radius.

    Returns:
        np.ndarray: Boolean 2D binary mask.
    """
    if use_ror_mask is False:
        if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
            raise ValueError('crop_radius_pixels and crop_radius_fraction must be zero '
                             'if use_ror_mask is set to False.')
        return np.ones(recon_shape[:2], dtype=bool)

    elif use_ror_mask is True:
        # Set up a mask to zero out points outside the RoR.
        if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
            raise ValueError('Only one of crop_radius_pixels and crop_radius_fraction '
                             'can be nonzero.')
        num_recon_rows, num_recon_cols = recon_shape[:2]
        row_center = (num_recon_rows - 1) / 2
        col_center = (num_recon_cols - 1) / 2

        row_radius = row_center - max(int(row_center * crop_radius_fraction), crop_radius_pixels, 0)
        col_radius = col_center - max(int(col_center * crop_radius_fraction), crop_radius_pixels, 0)

        col_coords = np.arange(num_recon_cols) - col_center
        row_coords = np.arange(num_recon_rows) - row_center
        coords = np.meshgrid(col_coords, row_coords)

        mask = (coords[0] / col_radius) ** 2 + (coords[1] / row_radius) ** 2 <= 1.0
        return mask

    else:  # user-provided mask
        if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
            raise ValueError('crop_radius_pixels and crop_radius_fraction must be zero '
                             'if use_ror_mask is a custom array.')
        mask = np.asarray(use_ror_mask)
        if mask.shape != tuple(recon_shape[:2]):
            raise ValueError(f"Custom use_ror_mask must have shape recon_shape[:2]. "
                             f"Got mask.shape={mask.shape}, expected {tuple(recon_shape[:2])}.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Custom use_ror_mask must contain only 0s and 1s.")
        return mask


def gen_pixel_partition(recon_shape, num_subsets, use_ror_mask=True):
    """
    Generates a partition of pixel indices into a specified number of subsets for
    use in tomographic reconstruction algorithms.  The function ensures that each
    subset contains an equal number of pixels, suitable for VCD reconstruction.

    Verbatim numpy port of mbirjax.vcd_utils.gen_pixel_partition, including its
    np.random call sequence (permutation, then choice) and its single-subset
    RNG skip -- the cross-framework trace-parity mechanism.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices).
        num_subsets (int): The number of subsets to divide the pixel indices into.
        use_ror_mask: as in :func:`get_2d_ror_mask` (True / False / custom array).

    Returns:
        np.ndarray: each row is a subset of pixel indices, sorted within each
        subset.
    """
    # Determine the 2D indices within the RoR.
    num_recon_rows, num_recon_cols = recon_shape[:2]
    max_index_val = num_recon_rows * num_recon_cols
    indices = np.arange(max_index_val, dtype=np.int32)

    # Mask off indices that are outside the region of reconstruction.
    if use_ror_mask is not False:
        mask = get_2d_ror_mask(recon_shape, use_ror_mask=use_ror_mask)
        indices = indices[mask.flatten() == 1]
    if num_subsets > len(indices):
        num_subsets = len(indices)
        warnings.warn('\nThe number of partition subsets is greater than the number of '
                      'pixels in the region of reconstruction.  \nReducing the number '
                      'of subsets to equal the number of indices.')

    # A single subset needs no permutation: the subsets are SORTED below, so a
    # shuffle would be exactly undone -- but it would consume global np.random
    # state.  Skipping it keeps full-index "partitions" (gen_full_indices: the
    # Hessian diagonal, the direct-recon init) from advancing the RNG, so a
    # restarted recon reproduces a continuous run's per-iteration subset
    # permutations -- and hence its trajectory -- exactly.
    if num_subsets == 1:
        return np.sort(indices).reshape(1, -1)

    # Determine the number of indices to repeat to make the total divisible by
    # num_subsets.
    num_indices_per_subset = int(np.ceil(len(indices) / num_subsets))
    array_size = num_subsets * num_indices_per_subset
    num_extra_indices = array_size - len(indices)
    indices = np.random.permutation(indices)

    # Enlarge the array to the desired length by adding random indices that are
    # not in the final subset.
    num_non_final_indices = (num_subsets - 1) * num_indices_per_subset
    extra_indices = np.random.choice(indices[:num_non_final_indices],
                                     size=num_extra_indices, replace=False)
    indices = np.concatenate((indices, extra_indices))

    # Reorganize into subsets, then sort each subset.
    indices = indices.reshape(num_subsets, indices.size // num_subsets)
    return np.sort(indices, axis=1)


def gen_set_of_pixel_partitions(recon_shape, granularity, device=None, use_ror_mask=True):
    """
    Generates a collection of voxel partitions for an array of specified
    partition sizes -- one randomly generated 2D partition per granularity entry.

    Args:
        recon_shape (tuple): Shape of recon in (rows, columns, slices).
        granularity (list or tuple): num_subsets to use for each partition.
        device (torch.device): device on which to place each partition tensor.
        use_ror_mask: as in :func:`get_2d_ror_mask`.

    Returns:
        list of int64 tensors, each a partition of voxels into the specified
        number of subsets.
    """
    partitions = []
    for num_subsets in granularity:
        partition = gen_pixel_partition(recon_shape, num_subsets, use_ror_mask=use_ror_mask)
        partitions.append(torch.as_tensor(np.ascontiguousarray(partition),
                                          dtype=torch.int64, device=device))
    return partitions


def gen_partition_sequence(partition_sequence, max_iterations):
    """
    Generates a sequence of voxel partitions of the specified length by extending
    the sequence with the last element if necessary (or trimming it).
    """
    partition_sequence = np.array(partition_sequence)
    current_length = partition_sequence.size
    if max_iterations > current_length:
        # Repeat the last element for the additional iterations needed.
        extension = np.full(max_iterations - current_length, partition_sequence[-1])
        return np.concatenate((partition_sequence, extension))
    return partition_sequence[:max_iterations]


def gen_full_indices(recon_shape, use_ror_mask=True):
    """
    Generates a full array of voxels in the region of reconstruction.
    This is useful for computing forward projections.
    """
    partition = gen_pixel_partition(recon_shape, num_subsets=1, use_ror_mask=use_ror_mask)
    return partition[0]


def gen_weights(sinogram, weight_type):
    """
    Compute optional weights used in MBIR reconstruction based on the noise model.

    The weights should be proportional to the inverse variance of the noise for
    each sinogram entry.  They can be used to improve reconstruction quality.

    The result is computed with the input's OWN array module, so it stays where
    the input is: a host (numpy) sinogram yields host weights; a torch tensor
    yields a tensor on the same device.

    Args:
        sinogram (ndarray or tensor): 3D array of shape
            (num_views, num_det_rows, num_det_channels).
        weight_type (str): The type of noise model to use for weighting.  One of:
            - 'unweighted': Use uniform weights (all ones).
            - 'transmission': Use exponential decay, `exp(-sinogram)`.
            - 'transmission_root': Use square-root decay, `exp(-sinogram / 2)`.
            - 'emission': Use reciprocal decay, `1 / (abs(sinogram) + 0.1)`.

    Returns:
        ndarray or tensor: weights with the same shape and residence as the input.

    Raises:
        Exception: If `weight_type` is not one of the supported options.

    Note:
        For transmission noise models, sinogram values should not be excessively
        large (e.g., > 5), as this corresponds to near-zero transmission, which
        is not physically meaningful in typical X-ray imaging.
    """
    xp = torch if isinstance(sinogram, torch.Tensor) else np
    if weight_type == 'unweighted':
        weights = xp.ones_like(sinogram)
    elif weight_type == 'transmission':
        weights = xp.exp(-sinogram)
    elif weight_type == 'transmission_root':
        weights = xp.exp(-sinogram / 2)
    elif weight_type == 'emission':
        weights = 1.0 / (xp.abs(sinogram) + 0.1)
    else:
        raise Exception("gen_weights: undefined weight_type {}".format(weight_type))
    return weights


def estimate_background_cluster_boundaries(sinogram):
    """
    Estimate background cluster left and right boundaries from the sinogram
    histogram.  This function assumes that the background takes on values near
    zero.  (Verbatim numpy port of the mbirjax.utilities function.)

    Args:
        sinogram (ndarray): 3D array with shape
            (num_views, num_det_rows, num_det_channels).

    Returns:
        left_boundary (float): value of the left boundary of the background cluster.
        right_boundary (float): value of the right boundary of the background cluster.
    """
    # Compute histogram of sinogram values.
    hist, edges = np.histogram(np.asarray(sinogram).ravel(), bins=400)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Find all local peaks in the histogram.
    peak_indices = []
    if len(hist) > 1 and hist[0] > hist[1]:
        peak_indices.append(0)
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
            peak_indices.append(i)

    # Choose the peak closest to intensity 0 (the background peak).
    if len(peak_indices) == 0:
        peak_idx = int(np.argmin(np.abs(centers - 0.0)))
    else:
        peak_idx = min(peak_indices, key=lambda i: abs(centers[i] - 0.0))

    # Define the background width cutoff level (10% of peak height).
    cutoff = 0.1 * hist[peak_idx]

    # Find the left and right boundaries of the background cluster.
    left_boundary_idx = peak_idx
    while left_boundary_idx > 0 and hist[left_boundary_idx] > cutoff:
        left_boundary_idx -= 1
    right_boundary_idx = peak_idx
    while right_boundary_idx < len(hist) - 1 and hist[right_boundary_idx] > cutoff:
        right_boundary_idx += 1

    return centers[left_boundary_idx], centers[right_boundary_idx]
