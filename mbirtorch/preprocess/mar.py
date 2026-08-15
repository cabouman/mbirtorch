import os
import warnings

import numpy as np
import torch
import mbirtorch as mt
import mbirtorch.preprocess as mtp
from mbirtorch import _sharding
from . import pipeline


# ── device-form helpers ──────────────────────────────────────────────────────
# The beam-hardening arithmetic below runs on sinograms in either form: one
# tensor, or a Shards container (one piece per device).  Each helper applies
# an operation in whichever form it is given; aligned arguments share one
# placement.  Elementwise work stays on each piece's device; reductions
# combine tiny per-piece results on the host.

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
    """float: fn (a scalar reduction) maximized across the pieces.

    A piece with no elements is skipped: a device may own no views, and a
    reduction has no value to return there."""
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
    """Global (view, row, col) of the minimum, plus the value.  Pieces are
    visited in view order with a strict comparison, so ties resolve to the
    first view, matching the single-tensor argmin."""
    if not isinstance(x, _sharding.Shards):
        return _argmin_3d(x)
    pl = x.placement
    best_idx, best_val = None, None
    for t, (_d, (v0, _v1)) in zip(x.tensors, pl.shard_ranges()):
        if t.numel() == 0:
            # A device may own no views, and an argmin has no answer there.
            continue
        (v, r, c), val = _argmin_3d(t)
        if best_val is None or float(val) < best_val:
            best_idx, best_val = (v + v0, r, c), float(val)
    return best_idx, best_val


def gen_huber_weights(weights, sino_error, T=1.0, delta=1.0, epsilon=1e-6):
    """
    Generate generalized Huber weights that down-weight outliers in ``sino_error``.

    The per-element standard deviation is ``std = 1 / sqrt(weights)``.  A single global factor
    ``alpha = ||sino_error|| / ||std||`` rescales it, and an element counts as an outlier when
    ``|sino_error / (alpha * std)| > T``.  Each outlier is down-weighted by the generalized Huber
    function; all other elements get weight 1.

    Typically, to obtain the final robust weights, the returned weights should be multiplied by the original `weights`:

        final_weights = weights * ghuber_weights

    Args:
        weights: ndarray or tensor of shape (views, rows, cols):
            Initial weights, typically derived from inverse variance estimates.
        sino_error: ndarray or tensor of shape (views, rows, cols):
            Sinogram error array representing deviations from the model.
        T: float, optional (default=1.0):
            Outlier threshold on the normalized error ``|sino_error / (alpha * std)|``.
        delta: float, optional (default=1.0):
            Controls the strength of the generalized Huber function (delta=1 corresponds to the conventional Huber).
        epsilon: float, optional (default=1e-6):
            Small number to avoid division by zero.

    Returns:
        huber_weights: same array module as ``weights``, shape (views, rows, cols)
            The computed generalized Huber weights.

    Notes:
        The generalized Huber function used in this function is based on:
        Venkatakrishnan, S. V., Drummy, L. F., Jackson, M., De Graef, M., Simmons, J. P., and Bouman, C. A.,
        "Model-Based Iterative Reconstruction for Bright-Field Electron Tomography,"
        IEEE Transactions on Computational Imaging, vol. 1, no. 1, pp. 1–15, 2015. DOI: 10.1109/TCI.2014.2371751

    Example:
        >>> huber_weights = gen_huber_weights(weights, sino_error)
        >>> final_weights = weights * huber_weights
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must be between 0 and 1.")

    was_numpy = not isinstance(weights, torch.Tensor)
    weights_t = torch.as_tensor(np.asarray(weights)) if was_numpy else weights
    if not isinstance(sino_error, torch.Tensor):
        sino_error = torch.as_tensor(np.asarray(sino_error), device=weights_t.device)

    # Compute std and global alpha
    std = 1.0 / torch.clamp(torch.sqrt(weights_t), min=epsilon)
    alpha = torch.linalg.norm(sino_error) / (torch.linalg.norm(std) + epsilon)
    std_norm = alpha * std

    # Compute normalized error
    normalized_error = sino_error / std_norm
    abs_norm_error = torch.abs(normalized_error)

    # Apply generalized Huber function
    huber_weights = torch.where(abs_norm_error <= T,
                                torch.ones_like(abs_norm_error),
                                (delta * T) / (abs_norm_error + epsilon))

    return huber_weights.cpu().numpy() if was_numpy else huber_weights


def BH_correction(sino, alpha, batch_size=64, devices=None):
    """
    Apply a polynomial beam hardening correction to a sinogram.

    This function applies a polynomial correction to each view of the sinogram
    by evaluating powers of the sinogram values and weighting them by the coefficients in `alpha`.

    The corrected sinogram is computed as:

        corrected_sino = alpha[0] * sino + alpha[1] * sino**2 + alpha[2] * sino**3 + ...

    It processes the sinogram in batches of views for memory efficiency.

    Args:
        sino (ndarray of shape (views, rows, cols)):
            Input sinogram to correct.
        alpha (list or array of floats):
            Coefficients for the polynomial correction. The k-th term corresponds to sino^(k+1).
        batch_size (int, optional, default=64):
            Number of views to process in a single batch.
        devices (sequence, optional):
            Devices to spread the view batches over.  The views are split into contiguous blocks,
            one per device, and the blocks are processed at the same time.  None (the default) runs
            everything on a single device.

    Returns:
        corrected_sino: ndarray of shape (views, rows, cols)
            Beam hardening corrected sinogram.

    Example:
        >>> import mbirtorch.preprocess as mtp
        >>> alpha = [1.0, 0.2, 0.1]  # Correction: sino + 0.2 * sino^2 + 0.1 * sino^3
        >>> corrected_sino = mtp.BH_correction(sino, alpha)
    """
    alpha = np.asarray(alpha)

    # Per-view-batch polynomial evaluation, driven through the shared pipeline driver.  The
    # correction is per-pixel, so batching is exact.
    def kernel(sino_batch):
        corrected = torch.zeros_like(sino_batch)
        for k in range(len(alpha)):
            corrected = corrected + float(alpha[k]) * torch.pow(sino_batch, k + 1)
        return corrected

    return pipeline.map_view_batches(sino, kernel, batch_size, devices=devices)


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

    # Sort by total degree (sum of powers)
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
    # Put the recon in the model's device form once at entry; the segmentation and the
    # 1+num_metal forward projections below consume the SAME device recon.
    recon = ct_model._shard_recon(recon)

    # --- Segment plastic and metal regions in the reconstruction ---
    # plastic_mask: Mask for plastic regions.
    # metal_masks: List of masks for each metal.
    # plastic_scale: Scaling factor for the plastic region.
    # metal_scales: List of scaling factors for each metal region.
    plastic_mask, metal_masks, plastic_scale, metal_scales = mtp.segment_plastic_metal(
        recon, num_metal=num_metal, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin)

    # --- Forward project and scale plastic ---
    # Keep the OUTPUT on-device (output_sharded=True): the whole correction below runs on these
    # device sinograms.
    plastic_sino_est = _ps_map(lambda t: plastic_scale * t,
                               ct_model.forward_project(plastic_mask, output_sharded=True))

    # --- Forward project the masked out metal regions ---
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

    # Most exponents are 0 or 1 (the exponent tuples are sparse), so skip the no-op factors instead of
    # materializing full-sinogram-sized ones/copies: x**0 == 1 exactly (skip the factor) and x**1 == x
    # exactly (use the array directly, no power op).  Byte-identical to the dense product.
    col = None
    for arr, exp in zip([plastic_sino_est] + list(metal_sino_est), exponents):
        if exp == 0:
            continue
        term = arr if exp == 1 else arr ** exp
        col = term if col is None else col * term
    if col is None:
        # All-zero exponent tuple (the constant column) -- excluded by construction from H, but handle
        # it correctly if a caller ever asks.
        col = torch.ones_like(plastic_sino_est)
    return col

def _get_row_H(pixel_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    Compute the row of the matrix H for one sinogram pixel.

    H is conceptually (num_pixels x num_cols) -- one row per sinogram pixel -- so ``pixel_index``
    specifies a ROW of H.  It is named for the pixel rather than the matrix row because it is a
    (view, row, col) tuple whose middle entry is the DETECTOR row, a different axis.

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
    """Index of the minimum of a 3-D sinogram-shaped array as PER-AXIS Python ints (view, row, col),
    plus the minimum value.

    Equivalent to unraveling a flat argmin, staged per axis so every index stays within its own small
    axis length.  Tie-breaking matches the flat row-major argmin: the first view attaining the
    minimum, and the first plane position within it.
    """
    num_views, num_rows, num_channels = x.shape
    per_view = x.reshape(num_views, -1)              # (V, R*C)
    per_view_min, plane_argmin = torch.min(per_view, dim=1)
    view = int(torch.argmin(per_view_min))
    row, col = divmod(int(plane_argmin[view]), num_channels)
    return (view, row, col), per_view_min[view]


# Minimum NORMALIZED metal-sinogram value for a pixel to be eligible for a residual-positivity
# constraint (the metal estimates are normalized to max 1 before the fit, so this is relative).
# See _find_most_violated_constraints for why near-zero-support pixels must be excluded.
_METAL_SUPPORT_FLOOR = 1e-3


def _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms):
    """
    Compute the most violated constraints for the beam hardening model.

    The BH model enforces two types of inequality constraints:
        1. Plastic positivity:        H_p[i,:] θ_p ≥ 0
        2. Residual positivity:       y[i] − H_m[i,:] θ_m ≥ 0

    This function evaluates the indices and values of the entries that most violate
    the constraints.

    The residual argmin is further restricted to pixels where some metal estimate exceeds
    ``_METAL_SUPPORT_FLOOR``: where every metal estimate is (near) zero the row H_m[i,:] is (near)
    zero, so no θ can move that residual and the constraint is unactionable -- vacuous when
    y[i] ≥ 0, and STRUCTURALLY INFEASIBLE when y[i] < 0 (which noisy measured sinograms routinely
    contain on air rays from log-domain noise).  One such selected pixel makes OSQP declare the
    whole QP primal infeasible, and its sentinel "solution" used to silently poison theta and
    collapse the corrected plastic to ~0.  Near-zero rows are almost as bad: the constraint then
    demands metal-polynomial coefficients of order y/m.

    Returns:
        idx_min_Sp (tuple of int): (view, row, col) of the smallest Sp entry.
        v_min_Sp (scalar): Value of Sp at that entry.
        idx_min_residual (tuple of int): (view, row, col) of the smallest (y − Sm) entry.
        v_min_residual (scalar): Value of (y − Sm) at that entry.
    """
    num_cols = len(H_exponent_list)
    # The coefficient of p in column i is the column with its p factor removed, so zero the p exponent
    # (the sparse _get_column_H then SKIPS that factor) rather than passing a dummy full-sinogram ones
    # array.
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

    # Residual argmin restricted to the metal support (see the docstring): pixels where every metal
    # estimate is <= the floor cannot be moved by theta, so they must never become constraints.
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
        # No constraints - solve unconstrained QP directly on the host (the system is tiny,
        # num_cols x num_cols).
        theta = np.linalg.solve(P_numpy, -q_numpy)
        return np.asarray(theta, dtype=np.float32)

    # Convert arrays as required by OSQP. These matrices are small.
    # osqp (which pulls scipy.sparse) is imported here, at its one use site, so that
    # importing the preprocess package stays fast for the many callers that never fit
    # a beam-hardening model.
    from scipy.sparse import csc_matrix
    import osqp
    A_numpy = np.asarray(A, dtype=np.float64)
    u_numpy = np.asarray(u, dtype=np.float64)

    P_sparse = csc_matrix(P_numpy)
    A_sparse = csc_matrix(A_numpy)

    solver = osqp.OSQP()
    solver.setup(P=P_sparse, q=q_numpy, A=A_sparse, l=None, u=u_numpy, alpha=1.0, verbose=0)
    result = solver.solve()

    # OSQP reports failure through the status field, NOT by raising: on an infeasible or unsolved
    # QP it fills result.x with a no-solution sentinel (2143289344.0 -- the float32-NaN bit pattern
    # as a value), which is FINITE and would silently poison every downstream use of theta.
    # Accept only a solved status ('solved' / 'solved inaccurate') with finite values.
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

    # Compute the upper triangle of HtH and mirror it.  Each column is built
    # in the sinogram's own form and the inner products sum per piece.
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

    # Compute total degree for each cross term and metal term
    cross_degree = [sum(exponent) for exponent in H_exponent_list[0:1+num_cross_terms]]
    metal_degree = [sum(exponent) for exponent in H_exponent_list[1+num_cross_terms:]]

    # Construct diagonal regularization weights: higher-degree terms are penalized more.
    # This applies stronger regularization to higher-order terms when alpha > 0.
    # Add 1 to the beginning to represent the weight for the linear plastic term (p^1).
    weights = np.asarray(cross_degree + metal_degree, dtype=np.float64)
    weight_matrix = np.diag(1 + weights ** alpha)

    # --- Solve for theta ---
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

    # Lists that store the indices of the points that most violate the constraints
    C_p = []
    C_m = []

    # Construct the entries P, q, A and u of OSQP for solving the constraint optimization
    P, q = _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta)
    A = np.zeros((0, num_cols))  # no active constraints yet
    u = np.zeros((0,))

    # Initial θ solved without constraint
    theta = _estimate_BH_model_params_using_OSQP(P, q, A=None, u=None)

    for iter in range(num_constraint_update_iter):
        # Find the (view, row, col) indices and values of the points that most violate each constraint
        idx_min_Sp, v_min_Sp, idx_min_residual, v_min_residual = _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms)

        # (1) Hp θp ≥ 0  ->  (-Hp) θ ≤ 0
        if v_min_Sp < tolerance and (idx_min_Sp not in C_p):
            # Coefficient-of-p row: zero the p exponent (pi**0 == 1 exactly) instead of allocating a
            # full-sinogram dummy ones array just to read its one pixel.
            p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]
            row_p = _get_row_H(idx_min_Sp, plastic_sino_est, metal_sino_est, p_coeff_exponents)
            # Negative row_p[:dp] to ensure Hpθp >= 0
            A_p = np.concatenate([-row_p[:dp], np.zeros((num_cols - dp,))])
            u_p = np.array([0.0])
            A = np.vstack([A, A_p[None, :]])
            u = np.concatenate([u, u_p])
            C_p.append(idx_min_Sp)

        # (2) y − Hm θm ≥ 0  ->  (Hm) θ ≤ y
        if v_min_residual < tolerance and (idx_min_residual not in C_m):
            row_m = _get_row_H(idx_min_residual, plastic_sino_est, metal_sino_est, H_exponent_list)
            # Positive row_m[dp:] to ensure y-Hmθm >= 0
            A_m = np.concatenate([np.zeros(dp), row_m[dp:]])
            # RHS clamped at 0: the metal-only contribution H_m θ_m is a physical (nonnegative)
            # attenuation, so its tightest meaningful upper bound is max(y, 0).  A raw negative
            # measurement (log-domain noise) would force the metal polynomial NEGATIVE at this
            # pixel's metal values -- for small values that means huge negative coefficients.
            u_m = np.array([max(_ps_item(measured_sino, idx_min_residual), 0.0)])
            A = np.vstack([A, A_m[None, :]])
            u = np.concatenate([u, u_m])
            C_m.append(idx_min_residual)

        # Early exit if both constraints are satisfied (within tolerances)
        if (v_min_Sp >= tolerance) and (v_min_residual >= tolerance):
            break
        theta_new = _estimate_BH_model_params_using_OSQP(P, q, A, u)
        if theta_new is None:
            # Defensive: with the support-restricted constraint selection the QP is feasible by
            # construction (the two constraint families act on disjoint theta blocks, each
            # satisfiable), so a solver failure signals numerical trouble.  Keep the last good
            # theta rather than propagating OSQP's failure sentinel into the correction.
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

    # Compute the denominator (linear plastic + cross terms) from the first (1 + num_cross_terms) columns
    # of H.  The coefficient of p in column i is the column with its p factor removed, so zero the p
    # exponent (the sparse _get_column_H then SKIPS that factor).
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
        # Enforce non-negativity on the residual sinogram (plastic + cross terms)
        return torch.clamp(out, min=0)

    Sp = _ps_map(build_sp, plastic_sino_est, *metal_sino_est)
    y_minus_Sm = _ps_map(build_y_minus_sm, measured_sino, plastic_sino_est, *metal_sino_est)

    # Central plastic coefficient, used to define a stabilization floor.  The MEAN (rather than the
    # median) is a cheap reduction; over the sinogram support the two are close, and this only sets a
    # floor.
    #
    # The two forms below are NOT interchangeable, so the branch is on the form
    # of Sp rather than on convenience.  One tensor keeps the pre-sharding
    # reduction exactly: torch.mean, a 0-d float32 Sp_floor, and torch.maximum
    # against it.  The float32-sum / float64-divide form exists only to combine
    # per-piece partials that cannot be reduced in one kernel, so it is
    # reserved for the sharded branch; using it on one device would move the
    # divide to float64 and silently change a single-device result that this
    # port promises to leave alone.
    if not isinstance(Sp, _sharding.Shards):
        mean_plastic_coef = torch.mean(Sp)
    else:
        mean_plastic_coef = _ps_sum(torch.sum, Sp) / _ps_numel(Sp)
    Sp_floor = gamma * mean_plastic_coef

    # A negative mean would be non-physical and may indicate instability in the algorithm
    # In that case, issue a runtime warning to flag the potential problem
    if float(mean_plastic_coef) <= 0:
        warnings.warn("Mean of Sp is negative", RuntimeWarning)

    # Clamp Sp at Sp_floor to prevent division by very small or negative values.
    # torch.maximum against a 0-d tensor of the piece's own dtype and device is
    # the unsharded expression verbatim; the sharded branch's Python-float floor
    # is materialized per piece rather than clamped as a weak scalar, so both
    # branches run the same kernel.
    def clamp_and_divide(sp, ym):
        floor = (Sp_floor if torch.is_tensor(Sp_floor)
                 else torch.as_tensor(Sp_floor, dtype=sp.dtype, device=sp.device))
        return p_normalization * ym / torch.maximum(sp, floor)

    corrected_plastic_sino = _ps_map(clamp_and_divide, Sp, y_minus_Sm)

    return corrected_plastic_sino

def _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected):
    # Compute a scaling factor by performing least-squares fitting between the corrected plastic sinogram
    # and the measured sinogram at plastic-only locations (i.e., where plastic is present and all metals are absent)
    # Plastic-only locations.  Zero out the other locations and let compute_scaling_factor's inner
    # products (sum(a*b)/sum(b*b)) do the reduction -- no data-dependent-shape selection needed.
    def keep_plastic_only(x, p, *ms):
        condition = (p != 0)
        for metal in ms:
            condition = condition & (metal == 0)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.where(condition, x, zero)

    plastic_sino_scale = mtp.compute_scaling_factor(
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
    # Construct the exponent list of the metal sinograms.
    metal_exponent_list = _generate_metal_exponent_list(num_metal, order)
    cross_exponent_list = _generate_metal_exponent_list(num_metal, order - 1)
    num_metal_terms = len(metal_exponent_list)
    num_cross_terms = len(cross_exponent_list)

    # Construct the exponent list for each column of the matrix H.
    # Each entry in H_exponent_list is a tuple representing the exponents of (p, m_0, m_1, ..., m_{num_metal-1}).
    # - Linear plastic term: (1, 0, 0, ...)
    # - Cross terms: The leading 1 indicates the presence of a linear p term.
    # - Metal-only terms: The leading 0 indicates there is no p in the term.
    # - Total number of columns: 1 + num_cross_terms + num_metal_terms.
    H_exponent_list = (
            [(1,) + (0,) * num_metal] +
            [(1, *t) for t in cross_exponent_list] +
            [(0, *t) for t in metal_exponent_list])

    # Put the measured sinogram in the model's device form.
    measured_sino = ct_model.prepare_sino_for_devices(measured_sino)

    # Get normalized sinogram p and [m_0, m_1, ...].
    plastic_sino_est, metal_sino_est = _est_plastic_metal_sinos_from_recon(
        recon, num_metal, ct_model, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin)
    plastic_sino_scale = _ps_max(lambda t: torch.max(torch.abs(t)), plastic_sino_est)
    metal_sino_scale = [_ps_max(lambda t: torch.max(torch.abs(t)), arr) for arr in metal_sino_est]
    # An empty (all-zero) plastic or metal estimate would silently fill the normalized sinogram with
    # NaNs and fail far downstream.  Check the scales explicitly and fail fast with an actionable
    # message.  ``not > 0`` also catches a NaN scale (e.g. a NaN in the recon).
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

    # Estimate beam hardening model parameters theta
    theta = _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter)

    # Compute the corrected plastic sinogram
    plastic_sino_corrected = _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list,
                                                       num_cross_terms, num_metal_terms, float(plastic_sino_scale), gamma)

    # Compute and apply the scaling of the corrected plastic sino
    plastic_sino_corrected_scale = _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected)

    # Combine the scaled corrected plastic sino and the metal sinos, then gather to a host sinogram
    # for the downstream recon.
    def combine(corrected, *ms):
        out = plastic_sino_corrected_scale * corrected
        for arr, norm in zip(ms, metal_sino_scale):
            out = out + arr * norm
        return out

    corrected_sino = _ps_map(combine, plastic_sino_corrected, *metal_sino_est)
    return ct_model._gather_sinogram(corrected_sino)
