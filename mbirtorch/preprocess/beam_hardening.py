import warnings

import numpy as np
import scipy
import torch
from mbirtorch import _sharding

from . import _pipeline
from .recon_utils import segment_plastic_metal, compute_scaling_factor

__all__ = ['BH_correction']


# These helpers apply an operation to a sinogram held either as one tensor or as a Shards
# container with one piece per device.  Reductions combine the per piece results on the host.

def _ps_map(fn, *xs):
    """Elementwise: fn over aligned inputs, returned in the same form."""
    if isinstance(xs[0], _sharding.Shards):
        parts = [fn(*[x.tensors[i] for x in xs])
                 for i in range(xs[0].placement.n_devices)]
        return _sharding.Shards(parts, xs[0].placement)
    return fn(*xs)


def _ps_sum(fn, *xs):
    """float: fn (a scalar reduction) summed across the pieces."""
    if isinstance(xs[0], _sharding.Shards):
        return sum(float(fn(*[x.tensors[i] for x in xs]))
                   for i in range(xs[0].placement.n_devices))
    return float(fn(*xs))


def _ps_max(fn, x):
    """float: fn (a scalar reduction) maximized across the pieces.  A piece with no elements is
    skipped, since a device may own no views."""
    if isinstance(x, _sharding.Shards):
        return max(float(fn(t)) for t in x.tensors if t.numel() > 0)
    return float(fn(x))


def _ps_numel(x):
    """int: total element count across the pieces."""
    if isinstance(x, _sharding.Shards):
        return sum(t.numel() for t in x.tensors)
    return x.numel()


def _ps_item(x, idx):
    """float: the value at a global (view, row, col) index."""
    if isinstance(x, _sharding.Shards):
        pl = x.placement
        for t, (_d, (v0, v1)) in zip(x.tensors, pl.shard_ranges()):
            if v0 <= idx[0] < v1:
                return float(t[idx[0] - v0, idx[1], idx[2]])
        raise IndexError(f'view {idx[0]} outside the sharded axis')
    return float(x[idx])


def _ps_argmin3d(x):
    """Global (view, row, col) of the minimum, plus the value.  A tie resolves to the first view,
    which matches the single tensor argmin."""
    if not isinstance(x, _sharding.Shards):
        return _argmin_3d(x)
    pl = x.placement
    best_idx, best_val = None, None
    for t, (_d, (v0, _v1)) in zip(x.tensors, pl.shard_ranges()):
        if t.numel() == 0:
            continue
        (v, r, c), val = _argmin_3d(t)
        if best_val is None or float(val) < best_val:
            best_idx, best_val = (v + v0, r, c), float(val)
    return best_idx, best_val


def BH_correction(sino, alpha, batch_size=64, devices=None):
    """
    Apply a polynomial beam hardening correction to a sinogram.

    Each value ``p`` of the sinogram is replaced by ``alpha[0] p + alpha[1] p**2 + alpha[2] p**3 + ...``.
    Takes a numpy array or a tensor and returns a numpy array.

    Args:
        sino (numpy array or tensor): Sinogram, shape (num_views, num_det_rows, num_det_channels).
        alpha (sequence of float): The polynomial coefficients; ``alpha[k]`` multiplies ``p**(k+1)``.
        batch_size (int, optional): Views processed at a time.  Defaults to 64.
        devices (sequence or None, optional): Devices to spread the views over.  None uses all visible
            CUDA devices, capped by ``MBIRTORCH_NUM_DEVICES`` when it is set, or the default device
            when there are none.  Defaults to None.

    Returns:
        numpy.ndarray: The corrected sinogram, the shape of ``sino``.

    Example:
        >>> sino = BH_correction(sino, alpha=[1.0, 0.2, 0.1])
    """
    _pipeline.reject_shards('BH_correction', sino=sino)

    alpha = np.asarray(alpha)

    def kernel(sino_batch):
        corrected = torch.zeros_like(sino_batch)
        for k in range(len(alpha)):
            corrected = corrected + float(alpha[k]) * torch.pow(sino_batch, k + 1)
        return corrected

    return _pipeline.map_view_batches(sino, kernel, batch_size,
                                     devices=_pipeline.permitted_devices(devices))


def _generate_metal_exponent_list(num_metal, max_order):
    """
    Generate all combinations of polynomial powers such that the total degree
    (sum of exponents) is <= max_order, excluding the all-zero combination.
    The combinations are sorted in increasing order of total degree.

    Args:
        num_metal (int): Number of metals.
        max_order (int): Maximum total degree of the polynomial.

    Returns:
        list[tuple[int]]: List of exponent tuples representing valid terms.
    """
    combinations = []

    def generate_recursive(current_combination, remaining_terms):
        if remaining_terms == 0:
            total_degree = sum(current_combination)
            if 0 < total_degree <= max_order:
                combinations.append(tuple(current_combination))
            return

        for power in range(max_order + 1):
            generate_recursive(current_combination + [power], remaining_terms - 1)

    generate_recursive([], num_metal)

    combinations.sort(key=lambda x: sum(x))
    return combinations


def _est_plastic_metal_sinos_from_recon(recon, num_metal, ct_model,
                                        radial_margin=None, top_margin=None, bottom_margin=None):
    """
    Segment plastic and metal regions from a reconstruction, project them,
    and return the unnormalized sinogram p, m0, m1, ... for beam hardening modeling.

    Args:
        recon (ndarray or tensor): Reconstructed image.
        num_metal (int): Number of metal types to segment.
        ct_model: Forward projection model with a `.forward_project()` method.
        radial_margin, top_margin, bottom_margin (int or None, optional): Segmentation mask
            margins; None (default) = size-relative (see segment_plastic_metal).

    Returns:
        plastic_sino_est (tensor): Unnormalized plastic sino estimation.
        metal_sino_est (list of tensor): List of unnormalized metal sino estimation.
    """
    recon = ct_model._shard_recon(recon)

    plastic_mask, metal_masks, plastic_scale, metal_scales = segment_plastic_metal(
        recon, num_metal=num_metal, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin)

    plastic_sino_est = _ps_map(lambda t: plastic_scale * t,
                               ct_model.forward_project(plastic_mask, output_sharded=True))

    metal_sino_est = []
    for mask in metal_masks:
        masked = (_sharding.Shards([mk * t for mk, t in zip(mask.tensors, recon.tensors)],
                                   recon.placement)
                  if isinstance(recon, _sharding.Shards) else mask * recon)
        m = ct_model.forward_project(masked, output_sharded=True)
        metal_sino_est.append(m)

    return plastic_sino_est, metal_sino_est


def _get_column_H(col_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    Compute the col_index-th column of the matrix H.

    The column is constructed as a monomial of the form:
        H[:, col_index] = p^e0 * m_0^e1 * m_1^e2 * ... * m_{n-1}^en

    where (e0, e1, ..., en) = H_exponent_list[col_index].

    Args:
        col_index (int): Index of the column to compute.
        plastic_sino_est (tensor): Normalized plastic sinogram estimation.
        metal_sino_est (list of tensor): Normalized metal sinogram estimation [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): List of exponent tuples defining each column of H.

    Returns:
        tensor: The computed column of H (same shape as p and m_i).
    """
    exponents = H_exponent_list[col_index]
    assert len(exponents) == 1 + len(metal_sino_est), "Mismatch between exponent tuple and number of sinograms."

    # An exponent of 0 contributes nothing, and an exponent of 1 needs no power operation.  Both are skipped.
    col = None
    for arr, exp in zip([plastic_sino_est] + list(metal_sino_est), exponents):
        if exp == 0:
            continue
        term = arr if exp == 1 else arr ** exp
        col = term if col is None else col * term
    if col is None:
        # An all zero exponent tuple gives the constant column.
        col = torch.ones_like(plastic_sino_est)
    return col

def _get_row_H(pixel_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    Compute the row of the matrix H for one sinogram pixel.

    H has one row per sinogram pixel, so ``pixel_index`` selects a row of H.  The middle entry of
    ``pixel_index`` is the detector row, which is a different axis from the row of H.

    Args:
        pixel_index (tuple of int): (view, row, col) of the pixel, identifying the row of H to compute.
        plastic_sino_est (tensor): Normalized plastic sinogram estimation.
        metal_sino_est (list of tensor): Normalized metal sinogram estimation [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): List of exponent tuples defining each column of H.

    Returns:
        ndarray: The computed row of H.
    """
    pi = _ps_item(plastic_sino_est, pixel_index)
    mi = [_ps_item(m, pixel_index) for m in metal_sino_est]
    row_vals = []
    for exps in H_exponent_list:
        val = (pi ** exps[0])
        for mk, ek in zip(mi, exps[1:]):
            val = val * (mk ** ek)
        row_vals.append(val)
    return np.asarray(row_vals, dtype=np.float32)


def _argmin_3d(x):
    """Index of the minimum of a 3D sinogram shaped array as Python ints (view, row, col), plus the
    minimum value.  A tie resolves to the first view and the first position within it.
    """
    num_views, num_rows, num_channels = x.shape
    per_view = x.reshape(num_views, -1)              # (V, R*C)
    per_view_min, plane_argmin = torch.min(per_view, dim=1)
    view = int(torch.argmin(per_view_min))
    row, col = divmod(int(plane_argmin[view]), num_channels)
    return (view, row, col), per_view_min[view]


# A pixel is eligible for a residual positivity constraint when some metal estimate exceeds this
# value.  The metal estimates are normalized to a maximum of 1 before the fit.
_METAL_SUPPORT_FLOOR = 1e-3


def _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms):
    """
    Compute the most violated constraints for the beam hardening model.

    The BH model enforces two types of inequality constraints:
        1. Plastic positivity:        H_p[i,:] θ_p ≥ 0
        2. Residual positivity:       y[i] − H_m[i,:] θ_m ≥ 0

    This function evaluates the indices and values of the entries that most violate
    the constraints.

    The residual search is restricted to pixels where some metal estimate exceeds
    ``_METAL_SUPPORT_FLOOR``.  Where every metal estimate is near zero the row H_m[i,:] is near
    zero, so no θ can move that residual.  Such a pixel makes OSQP declare the whole problem
    infeasible when y[i] is negative, which noisy sinograms contain on air rays.

    Returns:
        idx_min_Sp (tuple of int): (view, row, col) of the smallest Sp entry.
        v_min_Sp (scalar): Value of Sp at that entry.
        idx_min_residual (tuple of int): (view, row, col) of the smallest (y − Sm) entry.
        v_min_residual (scalar): Value of (y − Sm) at that entry.
    """
    num_cols = len(H_exponent_list)
    # Zeroing the p exponent gives the coefficient of p in each column.
    p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]

    def build_sp(p, *ms):
        sp = torch.zeros_like(p)
        for i in range(0, 1 + num_cross_terms):
            sp = sp + float(theta[i]) * _get_column_H(i, p, list(ms), p_coeff_exponents)
        return sp

    def build_y_minus_sm(y, p, *ms):
        out = y
        for j in range(1 + num_cross_terms, num_cols):
            out = out - float(theta[j]) * _get_column_H(j, p, list(ms), H_exponent_list)
        return out

    Sp = _ps_map(build_sp, plastic_sino_est, *metal_sino_est)
    y_minus_Sm = _ps_map(build_y_minus_sm, measured_sino, plastic_sino_est, *metal_sino_est)

    # A pixel with no metal support cannot be moved by theta, so it must never become a constraint.
    def mask_residual(ym, *ms):
        support = torch.zeros_like(ym, dtype=torch.bool)
        for metal in ms:
            support = support | (metal > _METAL_SUPPORT_FLOOR)
        inf = torch.tensor(float('inf'), dtype=ym.dtype, device=ym.device)
        return torch.where(support, ym, inf)

    ymSm_masked = _ps_map(mask_residual, y_minus_Sm, *metal_sino_est)
    idx_min_Sp, v_min_Sp = _ps_argmin3d(Sp)
    idx_min_residual, v_min_residual = _ps_argmin3d(ymSm_masked)

    return idx_min_Sp, v_min_Sp, idx_min_residual, v_min_residual



def _estimate_BH_model_params_using_OSQP(P, q, A, u):
    """
    Solve the constrained quadratic optimization problem:

        minimize_θ   0.5 * θᵀ P θ + qᵀ θ
        subject to   A θ ≤ u

    The problem is solved using the OSQP solver when constraints are provided.
    If `A` or `u` is `None`, an unconstrained least-squares solution is computed directly.

    Args:
        P (ndarray): Quadratic term matrix.
        q (ndarray): Linear term vector.
        A (ndarray): Inequality constraint matrix.
        u (ndarray): Right-hand side vector for the inequality constraints.

    Returns:
        ndarray or None: Solution vector θ, or ``None`` when the constrained solve fails
        (a non-solved OSQP status, or a non-finite solution vector).
    """
    P_numpy = np.asarray(P, dtype=np.float64)
    q_numpy = np.asarray(q, dtype=np.float64)

    if A is None or u is None:
        # There are no constraints, so solve the small unconstrained system directly.
        theta = np.linalg.solve(P_numpy, -q_numpy)
        return np.asarray(theta, dtype=np.float32)

    # osqp and scipy.sparse are imported here because most callers never fit a beam hardening model.
    from scipy.sparse import csc_matrix
    import osqp
    A_numpy = np.asarray(A, dtype=np.float64)
    u_numpy = np.asarray(u, dtype=np.float64)

    P_sparse = csc_matrix(P_numpy)
    A_sparse = csc_matrix(A_numpy)

    solver = osqp.OSQP()
    solver.setup(P=P_sparse, q=q_numpy, A=A_sparse, l=None, u=u_numpy, alpha=1.0, verbose=0)
    result = solver.solve()

    # OSQP reports failure in result.info.status rather than by raising.  On failure it fills
    # result.x with the finite sentinel 2143289344.0, so accept only a solved status with finite values.
    status = str(result.info.status).strip().lower()
    theta = np.asarray(result.x, dtype=np.float64)
    if not status.startswith('solved') or not np.all(np.isfinite(theta)):
        return None

    return np.asarray(theta, dtype=np.float32)

def _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta):
    """Compute entries for OSQP quadratic programming solver."""
    num_cols = len(H_exponent_list)

    HtH = np.zeros((num_cols, num_cols), dtype=np.float64)
    Hty = np.zeros(num_cols, dtype=np.float64)

    # Compute the upper triangle of HtH and mirror it.
    def column(i):
        return _ps_map(lambda p, *ms: _get_column_H(i, p, list(ms), H_exponent_list),
                       plastic_sino_est, *metal_sino_est)

    for i in range(num_cols):
        h_i = column(i)
        Hty[i] = _ps_sum(lambda a, b: torch.sum(a * b), h_i, measured_sino)
        for j in range(i, num_cols):
            h_j = column(j)
            dot_ij = _ps_sum(lambda a, b: torch.sum(a * b), h_i, h_j)
            HtH[i, j] = dot_ij
            if i != j:
                HtH[j, i] = dot_ij

    cross_degree = [sum(exponent) for exponent in H_exponent_list[0:1+num_cross_terms]]
    metal_degree = [sum(exponent) for exponent in H_exponent_list[1+num_cross_terms:]]

    # Diagonal regularization weights.  A higher degree term is penalized more when alpha > 0.
    weights = np.asarray(cross_degree + metal_degree, dtype=np.float64)
    weight_matrix = np.diag(1 + weights ** alpha)

    scaling_const = np.trace(HtH) / np.trace(weight_matrix)
    lambda_reg = beta * scaling_const

    P = HtH + lambda_reg * weight_matrix
    q = -Hty

    return P, q

def _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter=10, tolerance=-1e-5):
    """
    Estimate polynomial beam hardening model parameters with iterative constraints search.

    This function solves a regularized least squares problem with inequality constraints to
    enforce nonnegativity on the plastic and residual sinograms. The optimization problem is:

        minimize_θ   0.5‖Hθ − y‖² + 0.5λ‖θ‖²_Λ
        subject to   H_p[i,:] θ_p ≥ 0 and y[i] − H_m[i,:] θ_m ≥ 0

    where:
        - H_p contains the plastic and plastic–metal cross-term columns.
        - H_m contains the metal-only columns.

    The function uses an iterative active constraint selection method:
        1. Start from the unconstrained least squares estimate.
        2. Identify indices where the constraints are violated.
        3. Add the most violated constraints to the set.
        4. Re-solve the quadratic program (QP) using OSQP.
        5. Repeat until all constraints are satisfied or `num_constraint_update_iter` is reached.

    Args:
        plastic_sino_est (tensor): Normalized plastic sinogram estimation.
        metal_sino_est (list of tensor): List of normalized metal sino estimation.
        measured_sino (tensor): Measured sinogram.
        H_exponent_list (list of tuple[int]): List of exponent tuples defining each column of the matrix H.
        num_cross_terms (int): Number of cross terms (plastic × metal); remaining terms are metal-only.
        alpha (float): Regularization exponent; higher alpha penalizes higher-degree terms more.
        beta (float): Regularization strength scaling factor.
        num_constraint_update_iter (int): Number of iterations for updating constraints.
        tolerance (float): Tolerance for stopping criteria.

    Returns:
        theta (ndarray): Estimated model parameters corresponding to each column in H.

    """
    num_cols = len(H_exponent_list)
    dp = 1 + num_cross_terms

    C_p = []
    C_m = []

    P, q = _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta)
    A = np.zeros((0, num_cols))  # no active constraints yet
    u = np.zeros((0,))

    theta = _estimate_BH_model_params_using_OSQP(P, q, A=None, u=None)

    for iter in range(num_constraint_update_iter):
        idx_min_Sp, v_min_Sp, idx_min_residual, v_min_residual = _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms)

        # (1) Hp θp ≥ 0  ->  (-Hp) θ ≤ 0
        if v_min_Sp < tolerance and (idx_min_Sp not in C_p):
            # Zeroing the p exponent gives the coefficient of p in the row.
            p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]
            row_p = _get_row_H(idx_min_Sp, plastic_sino_est, metal_sino_est, p_coeff_exponents)
            # The sign of row_p[:dp] is negated so that Hp θp >= 0.
            A_p = np.concatenate([-row_p[:dp], np.zeros((num_cols - dp,))])
            u_p = np.array([0.0])
            A = np.vstack([A, A_p[None, :]])
            u = np.concatenate([u, u_p])
            C_p.append(idx_min_Sp)

        # (2) y − Hm θm ≥ 0  ->  (Hm) θ ≤ y
        if v_min_residual < tolerance and (idx_min_residual not in C_m):
            row_m = _get_row_H(idx_min_residual, plastic_sino_est, metal_sino_est, H_exponent_list)
            # row_m[dp:] is kept positive so that y - Hm θm >= 0.
            A_m = np.concatenate([np.zeros(dp), row_m[dp:]])
            # The right side is clamped at 0.  The metal contribution is a nonnegative attenuation,
            # and a negative measurement would force large negative metal coefficients.
            u_m = np.array([max(_ps_item(measured_sino, idx_min_residual), 0.0)])
            A = np.vstack([A, A_m[None, :]])
            u = np.concatenate([u, u_m])
            C_m.append(idx_min_residual)

        if (v_min_Sp >= tolerance) and (v_min_residual >= tolerance):
            break
        theta_new = _estimate_BH_model_params_using_OSQP(P, q, A, u)
        if theta_new is None:
            # Keep the last good theta rather than passing the OSQP failure sentinel downstream.
            warnings.warn("OSQP failed to solve the constrained beam-hardening fit; keeping the "
                          "parameters from the previous constraint iteration.", RuntimeWarning)
            break
        theta = theta_new
    return theta


def _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, num_metal_terms, p_normalization, gamma):
    """
    Perform beam hardening correction on the plastic sinogram.

    This function subtracts the metal-only contributions from the measured sinogram
    and normalizes the result using the linear plastic component, yielding a corrected
    sinogram that approximates the plastic-only contribution.

    The correction is based on a polynomial matrix H whose columns correspond to:
        - Plastic term: p
        - Cross terms: p*m, p*m^2, ...
        - Metal-only terms: m, m^2, m^3, ...

    The H matrix looks like: [p, p*m, p*m^2, m, m^2, m^3]
    The correction is applied as:
        corrected_plastic = p_normalization * max(y - H_metal·θ_m, 0) / (max(H_plastic·θ_p, γ * mean(H_plastic·θ_p))
    The stabilization term involving γ prevents division by near-zero or negative values, reducing streaks
    and numerical instability.

    Args:
        measured_sino (tensor): Measured sinogram.
        plastic_sino_est (tensor): Normalized plastic sino estimation.
        metal_sino_est (list of tensor): List of normalized metal sino estimation.
        theta (ndarray): Estimated coefficients for the polynomial terms in H.
        H_exponent_list (list of tuple): Exponent tuples defining each column of H.
        num_cross_terms (int): Number of cross terms involving both p and metal.
        num_metal_terms (int): Number of metal-only terms in H.
        p_normalization (float): Normalization factor applied to p.
        gamma (float, optional): Stabilization factor.

    Returns:
        corrected_plastic_sino (tensor): Beam-hardening-corrected plastic sinogram.
    """

    # The denominator uses the first 1 + num_cross_terms columns of H.  Zeroing the p exponent
    # gives the coefficient of p in each column.
    p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]

    def build_sp(p, *ms):
        sp = torch.zeros_like(p)
        for i in range(0, 1 + num_cross_terms):
            sp = sp + float(theta[i]) * _get_column_H(i, p, list(ms), p_coeff_exponents)
        return sp

    def build_y_minus_sm(y, p, *ms):
        out = y
        for j in range(1 + num_cross_terms, 1 + num_cross_terms + num_metal_terms):
            out = out - float(theta[j]) * _get_column_H(j, p, list(ms), H_exponent_list)
        return torch.clamp(out, min=0)

    Sp = _ps_map(build_sp, plastic_sino_est, *metal_sino_est)
    y_minus_Sm = _ps_map(build_y_minus_sm, measured_sino, plastic_sino_est, *metal_sino_est)

    # The central plastic coefficient sets a stabilization floor.  A sharded input is summed per
    # piece and divided on the host.
    if not isinstance(Sp, _sharding.Shards):
        mean_plastic_coef = torch.mean(Sp)
    else:
        mean_plastic_coef = _ps_sum(torch.sum, Sp) / _ps_numel(Sp)
    Sp_floor = gamma * mean_plastic_coef

    # A negative mean is not physical and may indicate instability.
    if float(mean_plastic_coef) <= 0:
        warnings.warn("Mean of Sp is negative", RuntimeWarning)

    # Clamp Sp at Sp_floor to prevent division by very small or negative values.
    def clamp_and_divide(sp, ym):
        floor = (Sp_floor if torch.is_tensor(Sp_floor)
                 else torch.as_tensor(Sp_floor, dtype=sp.dtype, device=sp.device))
        return p_normalization * ym / torch.maximum(sp, floor)

    corrected_plastic_sino = _ps_map(clamp_and_divide, Sp, y_minus_Sm)

    return corrected_plastic_sino

def _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected):
    # The scaling is a least squares fit between the corrected plastic sinogram and the measured
    # sinogram.  Only the locations where plastic is present and every metal is absent are used.
    def keep_plastic_only(x, p, *ms):
        condition = (p != 0)
        for metal in ms:
            condition = condition & (metal == 0)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.where(condition, x, zero)

    plastic_sino_scale = compute_scaling_factor(
        _ps_map(keep_plastic_only, measured_sino, plastic_sino_est, *metal_sino_est),
        _ps_map(keep_plastic_only, plastic_sino_corrected, plastic_sino_est, *metal_sino_est))
    return plastic_sino_scale

def correct_sino_plastic_metal(ct_model, measured_sino, recon, num_metal=1, order=3, alpha=1, beta=0.002, gamma=0.1, num_constraint_update_iter=10,
                               radial_margin=None, top_margin=None, bottom_margin=None):
    """
    This function corrects the measured sinogram of an object with plastic and multiple metal components by fitting a
    beam hardening model to the sinogram and removing the metal contributions.

    Args:
        ct_model: CT model object with a `forward_project` method and recon_placement / sino_placement.
        measured_sino (ndarray): Raw measured sinogram.
        recon (ndarray or tensor): Reconstructed 3D volume used for segmentation of plastic and metal regions.
        num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
        order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
        alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
            higher-order terms more strongly. Defaults to 1.
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.002.
        gamma (float, optional): Stabilization factor. Defaults to 0.1.
        num_constraint_update_iter (int, optional): Number of iterations for updating constraints. Defaults to 10.
        radial_margin, top_margin, bottom_margin (int or None, optional): Segmentation mask margins;
            None (default) = size-relative (see segment_plastic_metal).

    Returns:
        ndarray: Beam-hardening corrected sinogram of the same shape as `measured_sino`.
    """
    metal_exponent_list = _generate_metal_exponent_list(num_metal, order)
    cross_exponent_list = _generate_metal_exponent_list(num_metal, order - 1)
    num_metal_terms = len(metal_exponent_list)
    num_cross_terms = len(cross_exponent_list)

    # Each entry of H_exponent_list holds the exponents of (p, m_0, ..., m_{num_metal-1}).  The
    # first entry is the linear plastic term, then come the cross terms, then the metal only terms.
    H_exponent_list = (
            [(1,) + (0,) * num_metal] +
            [(1, *t) for t in cross_exponent_list] +
            [(0, *t) for t in metal_exponent_list])

    measured_sino = ct_model.prepare_sino_for_devices(measured_sino)

    plastic_sino_est, metal_sino_est = _est_plastic_metal_sinos_from_recon(
        recon, num_metal, ct_model, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin)
    plastic_sino_scale = _ps_max(lambda t: torch.max(torch.abs(t)), plastic_sino_est)
    metal_sino_scale = [_ps_max(lambda t: torch.max(torch.abs(t)), arr) for arr in metal_sino_est]
    # An empty plastic or metal estimate would fill the normalized sinogram with NaNs, so check the
    # scales here.  ``not > 0`` also catches a NaN scale.
    if not float(plastic_sino_scale) > 0:
        raise ValueError(
            "The estimated plastic sinogram is empty (the plastic segmentation class contains no "
            "voxels).  Check the input reconstruction, num_metal, and the cylindrical-mask margins.")
    for metal_index, scale in enumerate(metal_sino_scale):
        if not float(scale) > 0:
            raise ValueError(
                f"The estimated sinogram for metal {metal_index} is empty (its segmentation class "
                f"contains no voxels).  num_metal={num_metal} may be too large for this object.")
    plastic_sino_est = _ps_map(lambda t: t / plastic_sino_scale, plastic_sino_est)
    metal_sino_est = [_ps_map(lambda t, n=norm: t / n, arr)
                      for arr, norm in zip(metal_sino_est, metal_sino_scale)]

    theta = _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter)

    plastic_sino_corrected = _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list,
                                                       num_cross_terms, num_metal_terms, float(plastic_sino_scale), gamma)

    plastic_sino_corrected_scale = _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected)

    # Combine the corrected plastic sinogram with the metal sinograms and gather to the host.
    def combine(corrected, *ms):
        out = plastic_sino_corrected_scale * corrected
        for arr, norm in zip(ms, metal_sino_scale):
            out = out + arr * norm
        return out

    corrected_sino = _ps_map(combine, plastic_sino_corrected, *metal_sino_est)
    return ct_model._gather_sinogram(corrected_sino)


def fit_beam_hardening_curve(linear_projection, target_projection, num_parameters=5, zero_offset_normalized=True):
    """
    Fit a parametric beam-hardening function from paired samples.

    The fitted model is

        f(p) = -log( sum_{i=1..N} exp(theta_i - i * theta_0 * p) )

    with ``N = num_parameters - 1``.  The returned parameters are
    ``[theta_0, theta_1, ..., theta_N]``.  With
    ``zero_offset_normalized`` the model is shifted so that ``f(0) = 0``.
    Evaluate the fitted curve with :func:`apply_beam_hardening_curve`, and
    build the correction curve that inverts it with
    :func:`fit_inverse_beam_hardening_curve`.

    Args:
        linear_projection (np.ndarray): Ideal linear projection or path-length
            samples.
        target_projection (np.ndarray): Target beam-hardened projection
            samples paired with ``linear_projection``.
        num_parameters (int, optional): Total number of fitted parameters.
            Defaults to 5.
        zero_offset_normalized (bool, optional): If True, use the normalized
            forward model with ``h(0) = 0``. If False, use the
            unnormalized log-sum-exp form. Defaults to True.

    Returns:
        ndarray: Optimized parameter vector
            ``[theta_0, theta_1, ..., theta_{num_parameters-1}]``.

    Example:
        >>> linear_projection = sinogram.ravel()
        >>> target_projection = sinogram_nonlinear.ravel()
        >>> fitted_params = fit_beam_hardening_curve(
        ...     linear_projection, target_projection, num_parameters=5)
        >>> y_pred = apply_beam_hardening_curve(
        ...     sinogram_test, fitted_params)
    """
    num_parameters = int(num_parameters)
    if num_parameters < 2:
        raise ValueError(
            'fit_beam_hardening_curve: num_parameters must be at least 2.')

    linear_projection = np.asarray(linear_projection, dtype=np.float64)
    target_projection = np.asarray(target_projection, dtype=np.float64)

    if linear_projection.size != target_projection.size:
        raise ValueError(
            'fit_beam_hardening_curve: Input and target projection arrays must contain the same number of samples.')

    linear_projection = linear_projection.ravel()
    target_projection = target_projection.ravel()

    valid_mask = (
        np.isfinite(linear_projection) & np.isfinite(target_projection)
        & (linear_projection > 1e-6) & (target_projection > 1e-6)
    )
    linear_projection = linear_projection[valid_mask]
    target_projection = target_projection[valid_mask]

    if linear_projection.size == 0:
        raise ValueError(
            'fit_beam_hardening_curve: No valid training samples remain.')

    initial_params = np.zeros(num_parameters, dtype=np.float64)
    initial_params[0] = 1.0

    max_nfev = 20 * num_parameters
    solver_options = dict(
        loss="linear",
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        verbose=1,
    )

    optimization_result = scipy.optimize.least_squares(
        _beam_hardening_curve_residuals,
        initial_params,
        args=(linear_projection, target_projection, zero_offset_normalized),
        **solver_options,
    )

    if not optimization_result.success:
        warnings.warn(
            f'fit_beam_hardening_curve: Beam-hardening curve fit did not converge: {optimization_result.message}',
            RuntimeWarning)

    return optimization_result.x


def apply_beam_hardening_curve(linear_projection, params, zero_offset_normalized=True):
    """
    Apply a fitted parametric beam-hardening function.

    If ``zero_offset_normalized`` is True, this uses

        f(p) = log(sum_i exp(theta_i))
               - log(sum_i exp(theta_i - i * theta_0 * p)),

    which forces ``f(0) = 0``. If False, this uses the form

        f(p) = -log(sum_i exp(theta_i - i * theta_0 * p)).

    Args:
        linear_projection (np.ndarray): Linear projection values.
        params (np.ndarray): Parameter vector
            ``[theta_0, theta_1, ..., theta_N]``.
        zero_offset_normalized (bool, optional): Select the zero-normalized
            forward model. Defaults to True.

    Returns:
        ndarray: Beam-hardened projection values with the same shape as
            ``linear_projection``.
    """
    linear_projection = np.asarray(linear_projection, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64).reshape(-1)

    if params.size < 2:
        raise ValueError(
            'Expected at least 2 parameters: theta_0 and one log-weight.')

    theta_0 = params[0]
    theta_rest = params[1:]

    log_sum_exp_p = np.full_like(linear_projection, -np.inf, dtype=np.float64)
    for i, theta_i in enumerate(theta_rest, start=1):
        exponent = theta_i - i * theta_0 * linear_projection
        log_sum_exp_p = np.logaddexp(log_sum_exp_p, exponent)

    if not zero_offset_normalized:
        return -log_sum_exp_p

    log_sum_exp_0 = -np.inf
    for theta_i in theta_rest:
        log_sum_exp_0 = np.logaddexp(log_sum_exp_0, theta_i)

    return log_sum_exp_0 - log_sum_exp_p


def _beam_hardening_curve_residuals(params, linear_projection, target_projection, zero_offset_normalized):
    """
    Return fitted-minus-target residuals for nonlinear least-squares fitting.

    Args:
        params (np.ndarray): Current beam-hardening
            model parameters.
        linear_projection (np.ndarray): Filtered linear projection samples.
        target_projection (np.ndarray): Filtered target beam-hardened samples.
        zero_offset_normalized (bool, optional): Select the zero-normalized forward model.

    Returns:
        ndarray: One-dimensional residual vector used by
            :func:`scipy.optimize.least_squares`.
    """
    fitted_projection = apply_beam_hardening_curve(
        linear_projection, params,
        zero_offset_normalized=zero_offset_normalized)

    return fitted_projection.ravel() - target_projection.ravel()


def fit_inverse_beam_hardening_curve(forward_params, vmin=0.0, vmax=5.0, degree=10, num_samples=2000, zero_offset_normalized=True):
    """
    Fit a Chebyshev inverse that linearizes beam-hardened projections.

    Args:
        forward_params (np.ndarray): Forward beam-hardening parameters from
            :func:`fit_beam_hardening_curve`.
        vmin (float, optional): Minimum input projection value to correct.
            Defaults to 0.0.
        vmax (float, optional): Maximum input projection value to correct.
            Defaults to 5.0.
        degree (int, optional): Chebyshev polynomial degree. Defaults to 10.
        num_samples (int, optional): Number of fitting samples. Defaults to
            2000.
        zero_offset_normalized (bool, optional): Match the forward model
            normalization used to fit ``forward_params``. Defaults to True.

    Returns:
        tuple: ``(cheb_coeffs, y_domain)`` where ``cheb_coeffs`` is an
            ndarray of length ``degree + 1`` and ``y_domain`` is ``(vmin,
            vmax)`` for later inverse evaluation.

    Example:
        >>> forward_params = fit_beam_hardening_curve(
        ...     sinogram.ravel(), sinogram_nonlinear.ravel(),
        ...     num_parameters=5)
        >>> cheb_coeffs, y_domain = fit_inverse_beam_hardening_curve(
        ...     forward_params,
        ...     vmin=0.0,
        ...     vmax=float(sinogram_nonlinear.max()),
        ...     degree=10)
        >>> sinogram_linearized = apply_inverse_beam_hardening_curve(
        ...     sinogram_nonlinear, cheb_coeffs, y_domain)
    """
    forward_params = np.asarray(forward_params, dtype=np.float64).reshape(-1)
    vmin = float(vmin)
    vmax = float(vmax)
    degree = int(degree)
    num_samples = int(num_samples)

    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: require finite vmin < vmax.')
    if degree < 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: degree must be at least 1.')
    if num_samples < degree + 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: num_samples must be at least '
            'degree + 1.')

    # Estimate the effective attenuation h'(0).
    epsilon = 1e-6
    forward_at_zero = apply_beam_hardening_curve(
        0.0, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    forward_at_epsilon = apply_beam_hardening_curve(
        epsilon, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    effective_attenuation = float(
        (forward_at_epsilon - forward_at_zero) / epsilon)

    path_min = 0.0
    path_max = max(path_min + 1.0, abs(vmax) + 1.0)
    # Each pass doubles path_max.  This guard prevents an infinite expansion loop.
    max_expand_iterations = 64
    for _ in range(max_expand_iterations):
        y_at_path_max = apply_beam_hardening_curve(
            path_max, forward_params,
            zero_offset_normalized=zero_offset_normalized)
        if np.isfinite(y_at_path_max) and y_at_path_max >= vmax:
            break
        path_max = path_min + 2.0 * (path_max - path_min)
    else:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: could not expand the path '
            'length grid enough to cover vmax.')

    p_grid = np.linspace(
        path_min, path_max, 4 * num_samples, dtype=np.float64)
    y_grid = apply_beam_hardening_curve(
        p_grid, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    valid_mask = np.isfinite(p_grid) & np.isfinite(y_grid)
    p_grid = p_grid[valid_mask]
    y_grid = y_grid[valid_mask]
    if p_grid.size < degree + 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: too few finite forward '
            'samples.')

    sort_idx = np.argsort(y_grid)
    y_sorted = y_grid[sort_idx]
    p_sorted = p_grid[sort_idx]
    y_unique, unique_idx = np.unique(y_sorted, return_index=True)
    p_unique = p_sorted[unique_idx]

    if y_unique.size < degree + 1 or y_unique[-1] <= y_unique[0]:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: forward model is not '
            'invertible on the sampled grid.')
    if vmax > y_unique[-1]:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: sampled forward '
            f'model only reaches {y_unique[-1]:.6g}, below vmax={vmax:.6g}.')
    if vmin < y_unique[0]:
        warnings.warn(
            'fit_inverse_beam_hardening_curve: vmin is below the forward '
            'value at zero path length; low-end inverse samples will be '
            'clamped to zero.',
            RuntimeWarning)

    y_samples = np.linspace(vmin, vmax, num_samples, dtype=np.float64)
    p_samples = np.interp(
        y_samples, y_unique, p_unique,
        left=p_unique[0], right=p_unique[-1])
    linearized_projection_samples = p_samples * effective_attenuation

    y_scaled = 2.0 * (y_samples - vmin) / (vmax - vmin) - 1.0
    cheb_coeffs = np.polynomial.chebyshev.chebfit(
        y_scaled, linearized_projection_samples, deg=degree)

    return cheb_coeffs, (vmin, vmax)


def apply_inverse_beam_hardening_curve(beam_hardened_projection, cheb_coeffs, y_domain, clip=False):
    """
    Apply a fitted Chebyshev inverse to linearize projection values.

    Args:
        beam_hardened_projection (np.ndarray): Beam-hardened projection
            values. Arrays of any shape are accepted and the output preserves
            that shape.
        cheb_coeffs (np.ndarray): Coefficients returned by
            :func:`fit_inverse_beam_hardening_curve`.
        y_domain (tuple): ``(vmin, vmax)`` projection range used for fitting.
        clip (bool, optional): If True, clip input values into ``y_domain``
            before evaluation. If False, warn when extrapolating. Defaults to
            False.

    Returns:
        ndarray: Linearized projection values with the same shape as
            ``beam_hardened_projection``.
    """
    beam_hardened_projection = np.asarray(
        beam_hardened_projection, dtype=np.float64)
    cheb_coeffs = np.asarray(cheb_coeffs, dtype=np.float64).reshape(-1)
    y_min, y_max = float(y_domain[0]), float(y_domain[1])

    if y_max <= y_min:
        raise ValueError(
            'apply_inverse_beam_hardening_curve: y_domain must '
            'satisfy y_max > y_min.')

    if clip:
        y_eval = np.clip(beam_hardened_projection, y_min, y_max)
    else:
        if (np.any(beam_hardened_projection < y_min)
                or np.any(beam_hardened_projection > y_max)):
            warnings.warn(
                'apply_inverse_beam_hardening_curve: inputs lie '
                'outside the fitted y_domain; extrapolated values may be '
                'unreliable.',
                RuntimeWarning)
        y_eval = beam_hardened_projection

    y_scaled = 2.0 * (y_eval - y_min) / (y_max - y_min) - 1.0
    return np.polynomial.chebyshev.chebval(y_scaled, cheb_coeffs)
