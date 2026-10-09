"""Moving phantoms for 4D reconstruction demos and tests.

A moving phantom is a function of time: ``object(phantom_shape, t)`` returns the volume at
time ``t`` in [0, 1], where 0 is the start of the scan and 1 the end.  Each object is built
from its geometry at that time, so the frames are exact and need no interpolation.  The
objects are listed in ``MOVING_PHANTOMS`` by name; add a new one by writing its function and
adding it to the dict.

Volumes have shape (num_rows, num_cols, num_slices).  The slice axis is the rotation axis of
the scanner.
"""

import numpy as np


def _rack_and_pinion(phantom_shape, t, rotation_degrees=45.0, num_teeth=8):
    """
    A rack and pinion at time ``t``: a toothed bar along the slice axis and a toothed wheel
    beside it on a horizontal axle.

    The bar runs the full height of the volume along the slice axis, which is the rotation
    axis, so the volume should be tall: more slices than rows or columns.  The wheel turns by
    ``rotation_degrees * t`` and the bar slides along the slice axis by the wheel's radius
    times that angle, so the teeth stay meshed.  With the defaults, the wheel turns one tooth
    over the scan and the bar moves one tooth pitch.  The wheel's radius is a sixth of the
    number of columns, and the other sizes follow from it.  The bar has attenuation 1.0 and
    the wheel 0.7.

    Args:
        phantom_shape (tuple of int): (num_rows, num_cols, num_slices).
        t (float): Time in [0, 1].
        rotation_degrees (float, optional): The wheel's total turn over the scan.  Defaults to 45.
        num_teeth (int, optional): Teeth on the wheel.  Defaults to 8.

    Returns:
        numpy.ndarray: The volume at time ``t``, float32, shape ``phantom_shape``.
    """
    num_rows, num_cols, num_slices = phantom_shape
    radius = num_cols / 6.0                   # of the wheel, to the root of its teeth
    tooth_depth = radius / 3.0
    bar_width = radius / 2.0                  # along the columns
    thickness = num_rows / 5.0                # of both parts, along the rows
    pitch = 2.0 * np.pi * radius / num_teeth  # tooth spacing along the bar
    angle = np.deg2rad(rotation_degrees) * t

    # The pair is centered in the volume.  The bar is on the low-column side of the wheel.
    extent = bar_width + 2.0 * tooth_depth + 1.0 + 2.0 * radius
    bar_col = (num_cols - 1) / 2.0 - extent / 2.0 + bar_width / 2.0
    wheel_col = bar_col + bar_width / 2.0 + 2.0 * tooth_depth + 1.0 + radius
    center_row = (num_rows - 1) / 2.0
    wheel_slice = (num_slices - 1) / 2.0

    rows = np.arange(num_rows, dtype=np.float32)[:, None, None]
    cols = np.arange(num_cols, dtype=np.float32)[None, :, None]
    slices = np.arange(num_slices, dtype=np.float32)[None, None, :]
    in_thickness = np.abs(rows - center_row) <= thickness / 2.0

    # The bar: a box along the slice axis, with teeth on the face toward the wheel.  A
    # positive turn of the wheel drives the bar toward lower slice index.
    bar_shift = -radius * angle
    in_bar = np.abs(cols - bar_col) <= bar_width / 2.0
    on_bar_teeth = ((cols > bar_col + bar_width / 2.0)
                    & (cols <= bar_col + bar_width / 2.0 + tooth_depth)
                    & (np.mod(slices - bar_shift, pitch) < pitch / 2.0))
    bar = in_thickness & (in_bar | on_bar_teeth)

    # The wheel: a disk in the (column, slice) plane, with teeth around its rim.
    rho = np.sqrt((cols - wheel_col) ** 2 + (slices - wheel_slice) ** 2)
    phi = np.arctan2(slices - wheel_slice, cols - wheel_col)
    tooth_angle = 2.0 * np.pi / num_teeth
    on_wheel_teeth = ((rho > radius) & (rho <= radius + tooth_depth)
                      & (np.mod(phi - angle, tooth_angle) < tooth_angle / 2.0))
    wheel = in_thickness & ((rho <= radius) | on_wheel_teeth)

    volume = np.zeros(phantom_shape, dtype=np.float32)
    volume[np.broadcast_to(bar, phantom_shape)] = 1.0
    volume[np.broadcast_to(wheel, phantom_shape)] = 0.7
    return volume


# The moving phantoms by name.  Each takes (phantom_shape, t) and returns the volume at t.
MOVING_PHANTOMS = {
    'rack-and-pinion': _rack_and_pinion,
}


def _gen_moving_phantom(object_type, phantom_shape, num_steps, **options):
    """
    The volumes of a moving phantom at ``num_steps`` times: the scan is divided into
    ``num_steps`` equal intervals and the object is sampled at the middle of each.

    Args:
        object_type (str): A name in ``MOVING_PHANTOMS``, such as 'rack-and-pinion'.
        phantom_shape (tuple of int): (num_rows, num_cols, num_slices).
        num_steps (int): The number of times.
        **options: Passed to the phantom function.

    Returns:
        numpy.ndarray: The volumes, float32, shape (num_steps,) + phantom_shape.
    """
    try:
        phantom = MOVING_PHANTOMS[object_type]
    except KeyError:
        raise ValueError(f'object_type must be one of {sorted(MOVING_PHANTOMS)}; got {object_type!r}.')
    if num_steps < 1:
        raise ValueError(f'num_steps must be at least 1; got {num_steps}.')
    times = (np.arange(num_steps) + 0.5) / num_steps
    return np.stack([phantom(phantom_shape, t, **options) for t in times])
