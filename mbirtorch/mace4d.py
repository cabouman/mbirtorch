"""The frame-axis filter of the 4D reconstruction.

A scan divided into overlapping angular windows, one per time frame,
imprints a periodic modulation along the frame axis.
:func:`temporal_filter_matrix` builds the matrix of the filter that removes
it, and :func:`apply_temporal_filter` applies the matrix along one axis.
"""

import numpy as np
import torch


# ── the frame-axis filter as a matrix ────────────────────────────────────────
def temporal_filter_matrix(num_frames, period, harmonics=True, band_width=1):
    """
    The matrix of the frame-axis filter that removes a periodic modulation.

    A scan divided into overlapping angular windows imprints a modulation
    along the frame axis with a period equal to the number of frames per
    rotation.  The filter zeroes the type-I discrete cosine modes at that
    period and its harmonics, with ``band_width`` modes on each side.  It
    acts along the frame axis alone, so it is one matrix of shape
    ``(num_frames, num_frames)``.  Apply it with :func:`apply_temporal_filter`.

    At small frame counts the filter removes every mode.  At a period of 6
    and three frames the matrix is zero, and a run at that period needs well
    over eight frames for any temporal content to survive.

    Args:
        num_frames (int): number of frames, at least 2.
        period (float): the main period of the modulation, in frames.
        harmonics (bool or list of int, optional): True removes the main
            period and every harmonic with a period of at least two frames;
            False removes the main period only; a list names the harmonic
            indices to remove.  Defaults to True.
        band_width (int, optional): modes zeroed on each side of a removed
            mode.  Defaults to 1.

    Returns:
        torch.Tensor: the float32 matrix, on the CPU.
    """
    from scipy.fft import dct, idct

    num_frames = int(num_frames)
    if num_frames < 2:
        raise ValueError(f'the frame axis needs at least 2 frames to filter; got {num_frames}.')
    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        harmonic_list = list(range(1, int(np.floor(period / 2)) + 1))
    else:
        harmonic_list = list(harmonics)

    # Filtering the identity matrix column by column gives the matrix of the
    # filter, because the filter is linear along the frame axis.
    coefficients = dct(np.eye(num_frames, dtype=np.float32), type=1, norm='ortho', axis=0)
    for harmonic in harmonic_list:
        removed_period = period / harmonic
        center = int(round(2 * (num_frames - 1) / removed_period))
        low = max(0, center - band_width)
        high = min(num_frames, center + band_width + 1)
        if low < high:
            coefficients[low:high, :] = 0
    matrix = idct(coefficients, type=1, norm='ortho', axis=0).astype(np.float32, copy=False)
    return torch.as_tensor(np.ascontiguousarray(matrix))


def apply_temporal_filter(x, matrix, axis=0):
    """
    Apply a frame-axis filter matrix along one axis of a tensor.

    Args:
        x (torch.Tensor): the array to filter.
        matrix (torch.Tensor): a square matrix whose size is the length of
            ``axis``, from :func:`temporal_filter_matrix`.  It is moved to the
            device and dtype of ``x``.
        axis (int, optional): the axis the matrix acts along.  Defaults to 0.

    Returns:
        torch.Tensor: a new tensor of the shape of ``x``.
    """
    matrix = matrix.to(device=x.device, dtype=x.dtype)
    moved = x.movedim(axis, 0)
    filtered = torch.tensordot(matrix, moved, dims=1)
    return filtered.movedim(0, axis)
