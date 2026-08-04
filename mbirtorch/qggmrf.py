"""qGGMRF prior: gradient and Hessian-diagonal at pixel indices.

Ported from mbirjax.qggmrf (single-device path; the sharded halo/mask
machinery is out of Phase 1 scope).  The formulas are FCI Figure 8.5 /
Table 8.1, and ``b_tilde`` deliberately has ONE implementation
(``b_tilde_by_definition``), matching the mbirjax structure after its
qggmrf-flake fix: the clip floor lives only there.

Shapes: the jax version vmaps per cylinder and per slice; here the same math
is written directly batched -- per-cylinder over (N, S) arrays and per-slice
via flat gathers -- which is value-identical (the ops are elementwise on the
same operands; validated against the mbirjax goldens at ~1e-7 rel-max).
"""

import torch

_F32_EPS = torch.finfo(torch.float32).eps


def get_b_from_nbr_wts(qggmrf_nbr_wts):
    """Convert the 3-element list of neighbor weights to a 6-element tuple of
    weights in each direction.

    The order is [row+1, row-1, col+1, col-1, slice+1, slice-1] (see the
    defaults in _utils.py), normalized to sum to 1.
    """
    import numpy as np
    b = np.array([qggmrf_nbr_wts, qggmrf_nbr_wts]).T.flatten()
    b = b / np.sum(b)
    return tuple(b)


def b_tilde_by_definition(delta, sigma_x, p, q, T):
    """Compute ``b_tilde = rho'(delta) / (2 delta)`` from Table 8.1 in FCI.

    This is the ONE b_tilde implementation (the reference form): the
    production gradient/Hessian path delegates here via :func:`get_2_b_tilde`,
    so the two can never diverge (mbirjax previously carried two
    separately-coded forms that disagreed by up to ~3e-3 near ``delta -> 0``
    because they floored the removable 0/0 singularity differently).  The
    floor is a clip of ``|delta|`` at ``T * sigma_x * eps_f32``: exact above
    the floor, and for ``p = 2`` equal to the analytic ``rho''(0)/2`` to
    ~1e-6; below the floor the surrogate weight ``b_tilde * delta^2`` is
    ~1e-14, numerically inert in a reconstruction.
    """
    a_min = T * sigma_x * _F32_EPS
    abs_delta = torch.clamp(torch.abs(delta), min=a_min)
    delta_scale = abs_delta / (T * sigma_x)  # delta_scale has a min of eps

    ds_q_minus_p = delta_scale ** (q - p)
    ds_p_minus_2 = abs_delta ** (p - 2)

    numerator = ds_p_minus_2 / (2 * sigma_x ** p)
    numerator = numerator * ds_q_minus_p * ((q / p) + ds_q_minus_p)
    return numerator / (1 + ds_q_minus_p) ** 2


def get_2_b_tilde(delta, b_for_delta, qggmrf_params):
    """Compute ``b_for_delta * rho'(delta) / delta`` (i.e.
    ``b_for_delta * 2 * b_tilde``) -- the coefficient of the symmetric-bound
    surrogate (FCI Eq. 8.16-8.17).

    Args:
        delta (tensor): pixel differences between center and neighbor values.
        b_for_delta (float): the value of b associated with this delta.
        qggmrf_params (tuple): parameters in the form (b, sigma_x, p, q, T).

    Returns:
        tensor: ``b_for_delta * rho'(delta) / delta``.
    """
    b, sigma_x, p, q, T = qggmrf_params
    return b_for_delta * 2.0 * b_tilde_by_definition(delta, sigma_x, p, q, T)


def qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices,
                                           qggmrf_params):
    """
    Calculate the gradient and hessian at each index location in a reconstructed
    image using the surrogate function for the qGGMRF prior.
    Calculations taken from Figure 8.5 (page 119) of FCI for the qGGMRF prior model.

    Args:
        flat_recon (tensor): 2D reconstructed image array with shape
            (num_recon_rows x num_recon_cols, num_recon_slices).
        recon_shape (tuple of ints): shape of the original recon:
            (num_recon_rows, num_recon_cols, num_recon_slices).
        pixel_indices (int tensor): 1D array of shape (N_indices,) holding indices
            into the flattened (num_recon_rows x num_recon_cols) grid of voxel
            cylinders to be updated.
        qggmrf_params (tuple): The parameters b, sigma_x, p, q, T, with b the
            6-entry direction tuple from :func:`get_b_from_nbr_wts`.

    Returns:
        tuple of two tensors (first_derivative, second_derivative), each of shape
        (N_indices, num_recon_slices) representing the gradient and Hessian
        values at the specified indices.
    """
    # Neighborhood weight order is [row+1, row-1, col+1, col-1, slice+1, slice-1]
    # (see the definition in _utils.py).
    b, sigma_x, p, q, T = qggmrf_params
    num_rows, num_cols = recon_shape[0], recon_shape[1]

    # ── cylinder (slice-axis) term ────────────────────────────────────────────
    # Build delta[j] = v[j] - v[j-1] for interior positions, with explicit
    # boundary deltas at each end.  The jax version passes the boundary values
    # themselves (left_val = v[0], right_val = v[-1]) so both boundary deltas
    # are zero -- the reflected boundary condition; here the zero edges are
    # written directly.  Result per cylinder is
    #   0, v[1]-v[0], v[2]-v[1], ..., v[n-1]-v[n-2], 0.
    cylinders = flat_recon[pixel_indices]                     # (N, S)
    zero_edge = torch.zeros_like(cylinders[:, :1])
    delta = torch.cat((zero_edge, cylinders[:, 1:] - cylinders[:, :-1], zero_edge),
                      dim=1)                                  # (N, S+1)

    # Compute the primary quantity used for the gradient and Hessian.
    # Use b_for_delta = 1 here and scale by the slice-direction b below.
    b_tilde_2 = get_2_b_tilde(delta, 1.0, qggmrf_params)
    b_tilde_2_delta = b_tilde_2 * delta

    # The gradient gets a term from each neighbor, slice+1 and slice-1.
    # For slice+1, delta[1:] holds v[1]-v[0], v[2]-v[1], ..., so we need -delta
    # since delta is supposed to be xs - xr with xs the current point of
    # interest.  For slice-1, delta[:-1] holds v[0]-v[-1], v[1]-v[0], ... and
    # hence has the correct sign already.
    b_slice_plus, b_slice_minus = b[4], b[5]
    gradient = -b_slice_plus * b_tilde_2_delta[:, 1:] + b_slice_minus * b_tilde_2_delta[:, :-1]
    hessian = b_slice_plus * b_tilde_2[:, 1:] + b_slice_minus * b_tilde_2[:, :-1]

    # ── in-slice (row/col) term ──────────────────────────────────────────────
    # Add the contributions from the 4 in-plane neighbors.  Neighbor indices
    # clamp at the grid border (jax's ravel_multi_index mode='clip'): an edge
    # pixel's out-of-grid neighbor clips to itself, giving delta = 0 there --
    # the same reflected boundary condition as the cylinder term.
    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols

    # Access the central voxels' values at the given pixel_indices.
    xs0 = cylinders                                            # (N, S)

    # Relative positions and their b weights: row+1, row-1, col+1, col-1.
    offsets_and_b = [((1, 0), b[0]), ((-1, 0), b[1]),
                     ((0, 1), b[2]), ((0, -1), b[3])]
    for (dr, dc), b_value in offsets_and_b:
        r = (row_index + dr).clamp(0, num_rows - 1)
        c = (col_index + dc).clamp(0, num_cols - 1)
        neighbor = flat_recon[r * num_cols + c]                # (N, S)
        delta = xs0 - neighbor

        # Compute the primary quantity, then update the gradient and Hessian.
        b_tilde_2 = get_2_b_tilde(delta, b_value, qggmrf_params)
        gradient = gradient + b_tilde_2 * delta
        hessian = hessian + b_tilde_2

    return gradient, hessian


def prox_gradient_at_indices(recon, prox_input, pixel_indices, sigma_prox):
    """
    Calculate the gradient at each pixel index location in a reconstructed image
    using the proximal map prior.

    Args:
        recon (tensor): 2D reconstructed image array with shape
            (num_recon_rows x num_recon_cols, num_recon_slices).
        prox_input (tensor): 2D array with the same shape as ``recon``.
        pixel_indices (int tensor): 1D array of shape (N_indices,) holding indices
            into the flattened (num_recon_rows x num_recon_cols) grid of voxel
            cylinders to be updated.
        sigma_prox (float): Standard deviation parameter of the proximal map.

    Returns:
        tensor of shape (N_indices, num_recon_slices): the gradient of the prox
        term at the specified indices.
    """
    # Compute the prior model gradient at all voxels
    cur_diff = recon[pixel_indices] - prox_input[pixel_indices]
    return (1.0 / (sigma_prox ** 2.0)) * cur_diff
