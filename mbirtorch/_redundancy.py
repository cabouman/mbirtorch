"""Redundancy weights of direct reconstructions over ranges that are not whole rotations.

FDK and FBP count every measurement of a line through the object equally, which is right only when every line is
measured the same number of times.  The functions here give the weights that make the measurements of each line add up
to one, for the cone beam (with the fan angle of each channel) and for the parallel beam (fan angle zero).
"""

import numpy as np
import torch

# A range within this many radians of a whole number of rotations needs no weights.
_FULL_ROTATION_TOL = 1e-6


def _whole_rotations(scan_range):
    """This function returns the number of whole rotations in ``scan_range``, or None when the range
    is not a whole number of rotations."""
    rotations = scan_range / (2.0 * np.pi)
    if abs(rotations - round(rotations)) * 2.0 * np.pi < _FULL_ROTATION_TOL:
        return int(round(rotations))
    return None


def _whole_half_rotations(scan_range):
    """This function returns the number of whole half rotations in ``scan_range``, or None when the
    range is not a whole number of half rotations.  A parallel-beam scan over such a range measures
    every line the same number of times."""
    halves = scan_range / np.pi
    if abs(halves - round(halves)) * np.pi < _FULL_ROTATION_TOL:
        return int(round(halves))
    return None


def _redundancy_weights(beta, gamma, scan_range):
    """This function returns the redundancy weights of a scan over part of a rotation or over
    a number of rotations that is not whole.

    The weights make the measurements of each line through the object add up to one.  For a
    range below a full rotation, they are the Parker weights in the form of Wesarg, Ebert and
    Bortfeld (Med. Phys. 29, 2002): a ray at position ``beta`` with fan angle ``gamma`` is
    measured again at position beta + pi + 2 gamma with fan angle -gamma, and the range must be
    at least pi plus the full fan angle.  For a range above a full rotation, the weights ramp
    up over the excess beyond the whole rotations at the start of the range and ramp down over
    the same excess at the end, so that a ray and its repeat one rotation later add up to one.
    A parallel beam is the case ``gamma = 0``.

    Args:
        beta (ndarray): The position of each view in the scan, from 0 to ``scan_range``.
        gamma (ndarray): The fan angle of each channel.  ``beta`` and ``gamma`` broadcast
            against each other.
        scan_range (float): The angular range of the scan in radians.

    Returns:
        weights (ndarray): The weights, with the broadcast shape of ``beta`` and ``gamma``.
    """
    beta, gamma = np.broadcast_arrays(np.asarray(beta, dtype=np.float64),
                                      np.asarray(gamma, dtype=np.float64))
    weights = np.ones(beta.shape, dtype=np.float64)
    if scan_range < 2.0 * np.pi - _FULL_ROTATION_TOL:
        # The overscan beyond 180 degrees, split evenly between the two ends of the range.
        delta = 0.5 * (scan_range - np.pi)
        with np.errstate(divide='ignore', invalid='ignore'):
            start = beta < 2.0 * (delta - gamma)
            start_weight = np.sin(np.pi / 4.0 * beta / np.where(start, delta - gamma, 1.0)) ** 2
            weights = np.where(start, start_weight, weights)
            end = beta >= np.pi - 2.0 * gamma
            end_arg = np.clip(np.pi + 2.0 * delta - beta, 0.0, None)
            end_weight = np.sin(np.pi / 4.0 * end_arg / np.where(end, delta + gamma, 1.0)) ** 2
            weights = np.where(end, end_weight, weights)
        return weights
    rotations = int(np.floor((scan_range + _FULL_ROTATION_TOL) / (2.0 * np.pi)))
    excess = scan_range - 2.0 * np.pi * rotations
    if excess > _FULL_ROTATION_TOL:
        start = beta < excess
        weights = np.where(start, np.sin(0.5 * np.pi * beta / excess) ** 2, weights)
        end = beta > 2.0 * np.pi * rotations
        end_arg = np.clip(scan_range - beta, 0.0, None)
        weights = np.where(end, np.sin(0.5 * np.pi * end_arg / excess) ** 2, weights)
    # Each line is measured twice per rotation, so the weights of one line add up to one.
    return weights / (2.0 * rotations)


def _apply_view_channel_weights(sinogram, weights):
    """This function multiplies a sinogram on the devices by weights of shape (num_views,
    num_channels), the same for every detector row.  The sinogram may be one tensor or a Shards
    container whose shards hold consecutive blocks of views.  The result is a new array; the input
    is not changed."""
    from . import _sharding
    weights = np.asarray(weights, dtype=np.float32)[:, None, :]
    if isinstance(sinogram, _sharding.Shards):
        tensors = []
        start = 0
        for tensor in sinogram.tensors:
            stop = start + int(tensor.shape[0])
            block = torch.as_tensor(np.ascontiguousarray(weights[start:stop]), device=tensor.device)
            tensors.append(tensor * block)
            start = stop
        return _sharding.Shards(tensors, sinogram.placement)
    return sinogram * torch.as_tensor(np.ascontiguousarray(weights), device=sinogram.device)
