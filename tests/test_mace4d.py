"""Tests for the 4D reconstruction model and the frame-axis filter.

The SOR filter matrix is checked on the harmonics of the frame angle that it
removes and on the terms that it keeps.  The model
is checked on a 24-view cone-beam scan with a smooth sinogram.  A four-frame
run with the filter on differs from the same run with it off, and a seeded
four-frame run gives the same image on two CPU workers as on one.  The smooth sinogram is
kept because a random one reconstructs to a volume with extreme values, on
which the qGGMRF line search computes zero over zero.
"""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch.mace4d import MACE4DModel, _frame_angles, apply_temporal_filter, sor_filter_matrix

NUM_VIEWS = 24          # 24 views over 360 degrees, 15 degrees per view
DET_ROWS = 8
DET_COLS = 10


def _rel_max(out, ref):
    out = np.asarray(out.detach().cpu() if torch.is_tensor(out) else out, dtype=np.float64)
    ref = np.asarray(ref.detach().cpu() if torch.is_tensor(ref) else ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def _small_model():
    """A small cone-beam model with evenly spaced angles over 360 degrees."""
    angles = np.radians(360.0 / NUM_VIEWS) * np.arange(NUM_VIEWS)
    model = mbirtorch.ConeBeamModel((NUM_VIEWS, DET_ROWS, DET_COLS), angles,
                                    source_detector_dist=100.0, source_iso_dist=50.0)
    model.set_params(no_warning=True, verbose=0)
    return model


def _smooth_sino(num_views=NUM_VIEWS, rows=DET_ROWS, cols=DET_COLS):
    """A smooth, positive sinogram: a Gaussian bump modulated slowly over the views."""
    v = np.linspace(-1.0, 1.0, rows)
    c = np.linspace(-1.0, 1.0, cols)
    base = np.exp(-(v[:, None] ** 2 + c[None, :] ** 2))
    return np.stack([base * (1.0 + 0.1 * np.sin(0.5 * a))
                     for a in range(num_views)]).astype(np.float32)


def _four_frame_model():
    """A four-frame model of the 24-view scan.  Its filter removes only the
    first harmonic, so it keeps two of the four modes and stays on without a
    warning."""
    model = MACE4DModel(_small_model(), frames_per_rotation=4, frame_overlap_factor=1.0)
    model.set_params(verbose=0, sor_order=1)
    return model


# ── the frame-axis filter matrix ─────────────────────────────────────────────
def test_sor_filter_removes_the_harmonics_of_the_frame_angle():
    """At 6 frames per rotation over 4 whole turns, the filter of order 3 removes 5
    harmonics, because sin(3 theta) is zero at these angles.  It removes cos(theta),
    sin(2 theta), and cos(3 theta) terms and keeps a constant.  At order 1, it removes the
    cos(theta) term and keeps the rest.  The matrix is symmetric and idempotent, and its rows
    sum to 1.  The tolerance is float32 rounding.  One frame is refused.  Applying the matrix to
    a tensor gives the same result, and applying it along another axis equals moving that
    axis to the front and filtering there."""
    frames_per_rotation, num_turns = 6, 4
    theta = 0.3 - 2 * np.pi / frames_per_rotation * np.arange(frames_per_rotation * num_turns)
    matrix = sor_filter_matrix(theta, 3)
    assert matrix.dtype == torch.float32
    m = matrix.numpy().astype(np.float64)
    symmetry, idempotence = np.max(np.abs(m - m.T)), np.max(np.abs(m @ m - m))
    row_sum = np.max(np.abs(m.sum(axis=1) - 1))
    removed = int(round(len(theta) - np.trace(m)))
    print(f"harmonics removed {removed}, symmetry {symmetry:.2e}, idempotence {idempotence:.2e}, "
          f"row sums {row_sum:.2e}")
    assert m.shape == (len(theta), len(theta)) and removed == 5
    assert symmetry < 1e-5 and idempotence < 1e-5 and row_sum < 1e-5

    once = 0.2 * np.cos(theta)
    higher = -0.1 * np.sin(2 * theta) + 0.05 * np.cos(3 * theta)
    constant = 0.7
    assert np.max(np.abs(m @ (constant + once + higher) - constant)) < 1e-5
    first = sor_filter_matrix(theta, 1).numpy().astype(np.float64)
    assert np.max(np.abs(first @ (constant + once + higher) - (constant + higher))) < 1e-5

    with pytest.raises(ValueError, match='at least 2'):
        sor_filter_matrix([0.0], 3)

    # On a tensor with the same signal at every voxel, the filter leaves the constant.
    signal = torch.as_tensor(constant + once + higher, dtype=torch.float32)
    filtered = apply_temporal_filter(signal[:, None, None].expand(-1, 2, 3), matrix, axis=0)
    assert float(torch.max(torch.abs(filtered - constant))) < 1e-5
    # Along another axis: data movement, so the check is exact.
    torch.manual_seed(1)
    x = torch.randn(5, len(theta), 4, 3)
    along_axis_1 = apply_temporal_filter(x, matrix, axis=1)
    moved = apply_temporal_filter(x.movedim(1, 0), matrix, axis=0).movedim(0, 1)
    assert torch.equal(along_axis_1, moved)


def test_frame_angles_are_the_middle_of_each_frame():
    """With a view every -2.5 degrees and frames of 48 views every 24 views, the center of
    frame k is the angle of view 24 k + 23.5.  This holds also for frames past the first
    turn."""
    angles = np.radians(-2.5 * np.arange(400))
    view_slices = [slice(24 * k, 24 * k + 48) for k in range(15)]
    expected = np.radians(-2.5 * (24 * np.arange(15) + 23.5))
    difference = np.angle(np.exp(1j * (_frame_angles(angles, view_slices) - expected)))
    assert np.max(np.abs(difference)) < 1e-12


def test_model_filter_from_float32_angles_equals_the_filter_of_the_exact_angles():
    """A model holds its angles in float32.  On a scan of 600 views, one every -2.5 degrees,
    the filter that the model builds at 6 frames per rotation and order 3 removes 5 harmonics.
    It equals the filter of the exact frame angles to float32 rounding.  This holds when
    the angles grow past 2 pi and when they are given mod 2 pi."""
    num_views, frames_per_rotation = 600, 6
    exact = np.radians(-2.5 * np.arange(num_views))
    for angles in (exact, np.mod(exact, 2 * np.pi)):
        ct_model = mbirtorch.ParallelBeamModel((num_views, 2, 8), angles)
        ct_model.set_params(no_warning=True, verbose=0)
        model = MACE4DModel(ct_model, frames_per_rotation=frames_per_rotation)
        matrix, note = model._temporal_filter()
        centers = np.array([exact[s].mean() for s in model.view_slices])
        reference = sor_filter_matrix(centers, 3)
        removed = int(round(model.num_frames - float(matrix.trace())))
        difference = float(torch.max(torch.abs(matrix - reference)))
        print(f"{model.num_frames} frames: harmonics removed {removed}, "
              f"difference {difference:.2e}")
        assert removed == 5 and '5 of 24 modes removed' in note
        assert difference < 1e-5


def test_model_turns_the_filter_off_when_it_would_remove_all_change():
    """Four frames at 90 degree steps hold three modes besides their mean.  At order 3 the
    filter would remove all three, so the model returns no matrix.  At order 1 it removes two
    modes and keeps two."""
    model = MACE4DModel(_small_model(), frames_per_rotation=4, frame_overlap_factor=1.0)
    assert model.num_frames == 4 and model.get_params('sor_order') == 3
    matrix, note = model._temporal_filter()
    assert matrix is None and 'turned off' in note
    model.set_params(sor_order=1)
    matrix, note = model._temporal_filter()
    assert matrix is not None and '2 of 4 modes removed' in note


def test_reconstructions_run_end_to_end_with_and_without_the_filter(device):
    """A four-frame run with a filter of order 1 runs the filter path: two
    iterations give finite values of the 4D shape that differ from the
    filter-off run, and the denoiser strength, estimated from the filtered
    initial image, differs from the filter-off value.  Four frames per
    rotation with no overlap give four frames from the 24 views, and the
    filter keeps two of their four modes, so the filtered stack is not zero."""
    np.random.seed(0)
    filtered = _four_frame_model()
    assert filtered.num_frames == 4
    filtered.set_device_pool([device])
    shape = (4,) + filtered.recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    on, on_dict = filtered.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                 stop_threshold_change_pct=0)
    assert on.shape == shape and np.all(np.isfinite(on))
    assert on_dict['recon_params']['iterations completed'] == 2
    assert '2 of 4 modes removed' in on_dict['recon_params']['temporal filter']
    filtered.set_params(dejitter=False)
    np.random.seed(0)
    off, off_dict = filtered.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                   stop_threshold_change_pct=0)
    rel = _rel_max(on, off)
    print(f"filter on vs off: rel_max = {rel:.2e}")
    assert rel > 1e-2                        # the filter is applied, not merely recorded
    assert on_dict['recon_params']['denoiser sigma_x'] != off_dict['recon_params']['denoiser sigma_x']


def test_two_workers_reproduce_one_worker_on_a_seeded_run():
    """A seeded run derives every draw from the seed and the name of what the
    draw is for, so no draw depends on which worker makes it or when.  Two CPU
    workers must therefore reach the same image as one, to a relative maximum
    difference of 1e-3.  The gate allows for the float rounding of a folding
    update that adds task outputs in whatever order they arrive, which two
    consensus iterations amplify to about 1e-4; a run that drew on its worker
    instead differed by most of the image."""
    seed = 5
    one_worker = _four_frame_model()
    shape = (4,) + one_worker.recon_shape
    init = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    init += 0.01 * np.random.default_rng(4).standard_normal(shape).astype(np.float32)

    one_worker.set_device_pool(['cpu'])
    one, one_dict = one_worker.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                                     stop_threshold_change_pct=0, seed=seed)
    two_workers = _four_frame_model()
    two_workers.set_device_pool(['cpu', 'cpu'])
    two, _ = two_workers.recon(_smooth_sino(), init_recon=init, max_iterations=2,
                               stop_threshold_change_pct=0, seed=seed)
    rel = _rel_max(two, one)
    print(f"two CPU workers vs one on seed {seed}: rel_max = {rel:.2e}")
    assert one_dict['recon_params']['seed'] == seed
    assert rel < 1e-3
