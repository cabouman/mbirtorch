"""Partitions, masks, and weights, ported from mbirjax.vcd_utils.

The partition generators are numpy-for-numpy ports with the SAME global
np.random call sequence as mbirjax, so a seeded mbirtorch recon draws the
identical subsets (and subset order) as a seeded mbirjax recon -- the property
the cross-framework convergence-parity gate rests on.
"""

import warnings

import numpy as np
import torch


def get_2d_ror_mask(recon_shape, *, use_ror_mask=True, crop_radius_pixels=0,
                    crop_radius_fraction=0.0):
    """Binary region-of-reconstruction mask (ellipse inscribed in the grid),
    as in mbirjax.vcd_utils.get_2d_ror_mask."""
    if use_ror_mask is False:
        if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
            raise ValueError('crop_radius_pixels and crop_radius_fraction must be zero '
                             'if use_ror_mask is set to False.')
        return np.ones(recon_shape[:2], dtype=bool)

    elif use_ror_mask is True:
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

    else:
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
    """Partition the in-mask pixel indices into equal-size subsets (sorted within
    each subset).  Verbatim numpy port of mbirjax.vcd_utils.gen_pixel_partition,
    including its np.random call sequence and its single-subset RNG-skip."""
    num_recon_rows, num_recon_cols = recon_shape[:2]
    max_index_val = num_recon_rows * num_recon_cols
    indices = np.arange(max_index_val, dtype=np.int32)

    if use_ror_mask is not False:
        mask = get_2d_ror_mask(recon_shape, use_ror_mask=use_ror_mask)
        indices = indices[mask.flatten() == 1]
    if num_subsets > len(indices):
        num_subsets = len(indices)
        warnings.warn('\nThe number of partition subsets is greater than the number of '
                      'pixels in the region of reconstruction.  \nReducing the number '
                      'of subsets to equal the number of indices.')

    # A single subset consumes no RNG state (subsets are sorted, so a shuffle
    # would be undone) -- the mbirjax restart-reproducibility property.
    if num_subsets == 1:
        return np.sort(indices).reshape(1, -1)

    num_indices_per_subset = int(np.ceil(len(indices) / num_subsets))
    array_size = num_subsets * num_indices_per_subset
    num_extra_indices = array_size - len(indices)
    indices = np.random.permutation(indices)

    num_non_final_indices = (num_subsets - 1) * num_indices_per_subset
    extra_indices = np.random.choice(indices[:num_non_final_indices],
                                     size=num_extra_indices, replace=False)
    indices = np.concatenate((indices, extra_indices))

    indices = indices.reshape(num_subsets, indices.size // num_subsets)
    return np.sort(indices, axis=1)


def gen_set_of_pixel_partitions(recon_shape, granularity, device=None, use_ror_mask=True):
    """One partition per granularity entry, as int64 tensors on ``device``."""
    partitions = []
    for num_subsets in granularity:
        partition = gen_pixel_partition(recon_shape, num_subsets, use_ror_mask=use_ror_mask)
        partitions.append(torch.as_tensor(np.ascontiguousarray(partition),
                                          dtype=torch.int64, device=device))
    return partitions


def gen_partition_sequence(partition_sequence, max_iterations):
    """Extend (repeat the last entry) or trim the sequence to max_iterations."""
    partition_sequence = np.array(partition_sequence)
    current_length = partition_sequence.size
    if max_iterations > current_length:
        extension = np.full(max_iterations - current_length, partition_sequence[-1])
        return np.concatenate((partition_sequence, extension))
    return partition_sequence[:max_iterations]


def gen_full_indices(recon_shape, use_ror_mask=True):
    """All in-mask pixel indices (the num_subsets=1 partition's single subset)."""
    partition = gen_pixel_partition(recon_shape, num_subsets=1, use_ror_mask=use_ror_mask)
    return partition[0]


def gen_weights(sinogram, weight_type):
    """Sinogram weights per noise model, computed with the input's own module
    (numpy in -> numpy out; tensor in -> tensor out on the same device)."""
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
    """Histogram-based background cluster boundaries (verbatim numpy port of
    mbirjax.utilities.estimate_background_cluster_boundaries)."""
    hist, edges = np.histogram(np.asarray(sinogram).ravel(), bins=400)
    centers = 0.5 * (edges[:-1] + edges[1:])

    peak_indices = []
    if len(hist) > 1 and hist[0] > hist[1]:
        peak_indices.append(0)
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
            peak_indices.append(i)

    if len(peak_indices) == 0:
        peak_idx = int(np.argmin(np.abs(centers - 0.0)))
    else:
        peak_idx = min(peak_indices, key=lambda i: abs(centers[i] - 0.0))

    cutoff = 0.1 * hist[peak_idx]

    left_boundary_idx = peak_idx
    while left_boundary_idx > 0 and hist[left_boundary_idx] > cutoff:
        left_boundary_idx -= 1
    right_boundary_idx = peak_idx
    while right_boundary_idx < len(hist) - 1 and hist[right_boundary_idx] > cutoff:
        right_boundary_idx += 1

    return centers[left_boundary_idx], centers[right_boundary_idx]
