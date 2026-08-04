"""qGGMRF prior: gradient and Hessian-diagonal at pixel indices.

Ported from mbirjax.qggmrf (single-device path; the sharded halo/mask
machinery is out of Phase 1 scope).  The formulas are FCI Figure 8.5 /
Table 8.1, and ``b_tilde`` deliberately has ONE implementation
(``b_tilde_by_definition``), matching the mbirjax structure after its
qggmrf-flake fix: the clip floor lives only there.

Shapes: the jax version vmaps per cylinder and per slice; here the same math
is written directly batched -- per-cylinder over (N, S) arrays and per-slice
via flat gathers -- which is value-identical (the ops are elementwise on the
same operands).
"""

import torch

_F32_EPS = torch.finfo(torch.float32).eps


def get_b_from_nbr_wts(qggmrf_nbr_wts):
    """[row_wt, col_wt, slice_wt] -> the 6-entry normalized b tuple
    (row+1, row-1, col+1, col-1, slice+1, slice-1), as in mbirjax."""
    import numpy as np
    b = np.array([qggmrf_nbr_wts, qggmrf_nbr_wts]).T.flatten()
    b = b / np.sum(b)
    return tuple(b)


def b_tilde_by_definition(delta, sigma_x, p, q, T):
    """rho'(delta) / (2 delta) from FCI Table 8.1 (the single b_tilde source).

    The removable 0/0 singularity is floored by clipping |delta| at
    T * sigma_x * eps_f32, exactly as in mbirjax.
    """
    a_min = T * sigma_x * _F32_EPS
    abs_delta = torch.clamp(torch.abs(delta), min=a_min)
    delta_scale = abs_delta / (T * sigma_x)

    ds_q_minus_p = delta_scale ** (q - p)
    ds_p_minus_2 = abs_delta ** (p - 2)

    numerator = ds_p_minus_2 / (2 * sigma_x ** p)
    numerator = numerator * ds_q_minus_p * ((q / p) + ds_q_minus_p)
    return numerator / (1 + ds_q_minus_p) ** 2


def get_2_b_tilde(delta, b_for_delta, qggmrf_params):
    """b_for_delta * rho'(delta) / delta (= b_for_delta * 2 * b_tilde)."""
    b, sigma_x, p, q, T = qggmrf_params
    return b_for_delta * 2.0 * b_tilde_by_definition(delta, sigma_x, p, q, T)


def qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices,
                                           qggmrf_params):
    """Gradient and Hessian diagonal of the qGGMRF surrogate at the given cylinders.

    Args:
        flat_recon: (num_rows*num_cols, num_slices) tensor.
        recon_shape: (num_rows, num_cols, num_slices).
        pixel_indices: (N,) int64 flat indices into the (rows, cols) grid.
        qggmrf_params: (b, sigma_x, p, q, T) with b the 6-entry tuple.

    Returns:
        (gradient, hessian): each (N, num_slices).
    """
    b, sigma_x, p, q, T = qggmrf_params
    num_rows, num_cols = recon_shape[0], recon_shape[1]

    # ── cylinder (slice-axis) term, reflected boundary condition ─────────────
    # delta[j] = v[j] - v[j-1] with explicit zero boundary deltas: the jax
    # version passes v[0] / v[-1] as the neighbor values, making both boundary
    # deltas exactly zero.
    cylinders = flat_recon[pixel_indices]                     # (N, S)
    zero_edge = torch.zeros_like(cylinders[:, :1])
    delta = torch.cat((zero_edge, cylinders[:, 1:] - cylinders[:, :-1], zero_edge),
                      dim=1)                                  # (N, S+1)
    b_tilde_2 = get_2_b_tilde(delta, 1.0, qggmrf_params)
    b_tilde_2_delta = b_tilde_2 * delta

    b_slice_plus, b_slice_minus = b[4], b[5]
    gradient = -b_slice_plus * b_tilde_2_delta[:, 1:] + b_slice_minus * b_tilde_2_delta[:, :-1]
    hessian = b_slice_plus * b_tilde_2[:, 1:] + b_slice_minus * b_tilde_2[:, :-1]

    # ── in-slice (row/col) term ──────────────────────────────────────────────
    # Neighbor indices via clip-at-the-border ravel (jax's mode='clip'): an
    # edge pixel's out-of-grid neighbor clips to itself, giving delta = 0
    # there -- the same reflected boundary condition.
    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols
    xs0 = cylinders                                            # (N, S)

    offsets_and_b = [((1, 0), b[0]), ((-1, 0), b[1]),
                     ((0, 1), b[2]), ((0, -1), b[3])]
    for (dr, dc), b_value in offsets_and_b:
        r = (row_index + dr).clamp(0, num_rows - 1)
        c = (col_index + dc).clamp(0, num_cols - 1)
        neighbor = flat_recon[r * num_cols + c]                # (N, S)
        delta = xs0 - neighbor
        b_tilde_2 = get_2_b_tilde(delta, b_value, qggmrf_params)
        gradient = gradient + b_tilde_2 * delta
        hessian = hessian + b_tilde_2

    return gradient, hessian


def prox_gradient_at_indices(recon, prox_input, pixel_indices, sigma_prox):
    """Gradient of the proximal-map prior at the given cylinders."""
    cur_diff = recon[pixel_indices] - prox_input[pixel_indices]
    return (1.0 / (sigma_prox ** 2.0)) * cur_diff
