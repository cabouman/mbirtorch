"""Tests for the frame-axis filter of the 4D reconstruction.

The filter matrix is checked against the transform it replaces.
"""

import numpy as np
import pytest
import torch
from scipy.fft import dct, idct

from mbirtorch.mace4d import apply_temporal_filter, temporal_filter_matrix


# ── the frame-axis filter matrix ─────────────────────────────────────────────
def _dct_filter(x, period, harmonics=True, band_width=1):
    """The frame-axis filter written as transforms: the reference the matrix
    must reproduce.  Zeroes the DCT-I modes at the period and its harmonics,
    with band_width modes on each side, along axis 0."""
    x = np.asarray(x, dtype=np.float32)
    num_frames = x.shape[0]
    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        harmonic_list = list(range(1, int(np.floor(period / 2)) + 1))
    else:
        harmonic_list = list(harmonics)
    coefficients = dct(x, type=1, norm='ortho', axis=0)
    for harmonic in harmonic_list:
        center = int(round(2 * (num_frames - 1) / (period / harmonic)))
        low, high = max(0, center - band_width), min(num_frames, center + band_width + 1)
        if low < high:
            coefficients[low:high, ...] = 0
    return idct(coefficients, type=1, norm='ortho', axis=0).astype(np.float32)


@pytest.mark.parametrize('num_frames', [12, 30])
def test_filter_matrix_equals_the_transform_filter(num_frames):
    """The matrix reproduces the transform filter on random inputs and on
    unit impulses at the first and last frames, where the end behavior of
    the transform shows.  The tolerance is float32 rounding relative to the
    largest input value."""
    matrix = temporal_filter_matrix(num_frames, period=6)
    assert matrix.shape == (num_frames, num_frames) and matrix.dtype == torch.float32
    rng = np.random.default_rng(num_frames)
    inputs = [rng.standard_normal((num_frames, 4, 5, 3)).astype(np.float32)]
    for frame in (0, num_frames - 1):
        impulse = np.zeros((num_frames, 1, 1, 1), dtype=np.float32)
        impulse[frame] = 1.0
        inputs.append(impulse)
    worst = 0.0
    for x in inputs:
        reference = _dct_filter(x, period=6)
        filtered = apply_temporal_filter(torch.as_tensor(x), matrix, axis=0).numpy()
        worst = max(worst, float(np.max(np.abs(filtered - reference)) / np.max(np.abs(x))))
    print(f"filter matrix vs transform filter, {num_frames} frames: {worst:.2e}")
    assert worst < 1e-5
    # A symmetric projection, to the same rounding.
    m = matrix.numpy().astype(np.float64)
    symmetry, idempotence = np.max(np.abs(m - m.T)), np.max(np.abs(m @ m - m))
    print(f"symmetry {symmetry:.2e}, idempotence {idempotence:.2e}")
    assert symmetry < 1e-5 and idempotence < 1e-5


def test_filter_matrix_for_three_frames_is_zero():
    """At period 6 the filter removes every mode of a three-frame axis.  The
    zero is exact: the transform of a zeroed coefficient array is zero.  One
    frame cannot be filtered and is refused."""
    assert torch.equal(temporal_filter_matrix(3, period=6), torch.zeros(3, 3))
    with pytest.raises(ValueError, match='at least 2'):
        temporal_filter_matrix(1, period=6)


def test_filter_applies_along_any_axis():
    """Applying the matrix along axis 1 of a stack equals moving that axis to
    the front, filtering, and moving it back.  This is data movement, so the
    check is exact."""
    torch.manual_seed(1)
    matrix = temporal_filter_matrix(12, period=6)
    x = torch.randn(5, 12, 4, 3)
    along_axis_1 = apply_temporal_filter(x, matrix, axis=1)
    moved = apply_temporal_filter(x.movedim(1, 0), matrix, axis=0).movedim(0, 1)
    assert torch.equal(along_axis_1, moved)
