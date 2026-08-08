import numpy as np
import torch
import functools
import warnings
import mbirtorch as mt
import mbirtorch.preprocess as mtp
from . import pipeline

def gen_huber_weights(weights, sino_error, T=1.0, delta=1.0, epsilon=1e-6):
    """
    This function generates generalized Huber weights based on the method described in the referenced notes.
    It adds robustness by treating any element where ``|sino_error / weights| > T`` as an outlier,
    down-weighting it according to the generalized Huber function.

    The function returns new `ghuber_weights`.

    Typically, to obtain the final robust weights, the `ghuber_weights` should be multiplied by the original `weights`:

        final_weights = weights * ghuber_weights

    Args:
        weights: ndarray or tensor of shape (views, rows, cols):
            Initial weights, typically derived from inverse variance estimates.
        sino_error: ndarray or tensor of shape (views, rows, cols):
            Sinogram error array representing deviations from the model.
        T: float, optional (default=1.0):
            Threshold parameter; values greater than T are treated as outliers.
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
            Accepted for interface compatibility; the batches run on a single device.

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

    # Slice-padding info: valid_mask (True on real slices; None when unpadded) and num_real_slices
    # let the segmentation exclude any padded slices from its statistics and masks.
    pl = ct_model.recon_placement
    valid_mask = pl.real_mask(recon.ndim)

    # --- Segment plastic and metal regions in the reconstruction ---
    # plastic_mask: Mask for plastic regions.
    # metal_masks: List of masks for each metal.
    # plastic_scale: Scaling factor for the plastic region.
    # metal_scales: List of scaling factors for each metal region.
    plastic_mask, metal_masks, plastic_scale, metal_scales = mtp.segment_plastic_metal(
        recon, num_metal=num_metal, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin, valid_mask=valid_mask, num_real_slices=pl.real_size)

    # --- Forward project and scale plastic ---
    # Keep the OUTPUT on-device (output_sharded=True): the whole correction below runs on these
    # device sinograms.
    plastic_sino_est = plastic_scale * ct_model.forward_project(plastic_mask, output_sharded=True)

    # --- Forward project the masked out metal regions ---
    metal_sino_est = []
    for mask in metal_masks:
        m = ct_model.forward_project(mask * recon, output_sharded=True)
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
    pi = float(plastic_sino_est[pixel_index])
    mi = [float(m[pixel_index]) for m in metal_sino_est]
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


def _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, view_mask=None):
    """
    Compute the most violated constraints for the beam hardening model.

    The BH model enforces two types of inequality constraints:
        1. Plastic positivity:        H_p[i,:] θ_p ≥ 0
        2. Residual positivity:       y[i] − H_m[i,:] θ_m ≥ 0

    This function evaluates the indices and values of the entries that most violate
    the constraints.  When the sinogram is zero-padded on the view axis, ``view_mask`` (1 on real
    views, 0 on padded, broadcasting over the sinogram) excludes the padded views from the argmin so
    a padded entry is never selected as a constraint.

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
    Sp = torch.zeros_like(measured_sino)
    for i in range(0, 1 + num_cross_terms):
        Sp = Sp + float(theta[i]) * _get_column_H(i, plastic_sino_est, metal_sino_est, p_coeff_exponents)

    # y_minus_Sm = y - metal-only
    y_minus_Sm = measured_sino
    # Subtract metal-only terms (from H columns after the cross terms)
    for j in range(1 + num_cross_terms, num_cols):
        y_minus_Sm = y_minus_Sm - float(theta[j]) * _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)

    # Lower-bound violator: minimize Sp and y-Sm over the REAL views (padded views set to +inf so they
    # can't win the argmin).
    inf = torch.tensor(float('inf'), dtype=Sp.dtype, device=Sp.device)
    Sp_masked = Sp if view_mask is None else torch.where(view_mask, Sp, inf)

    # Residual argmin restricted to the metal support (see the docstring): pixels where every metal
    # estimate is <= the floor cannot be moved by theta, so they must never become constraints.
    # Padded views have all-zero metal estimates, so this mask also excludes them.
    metal_support = torch.zeros_like(y_minus_Sm, dtype=torch.bool)
    for metal in metal_sino_est:
        metal_support = metal_support | (metal > _METAL_SUPPORT_FLOOR)
    ymSm_masked = torch.where(metal_support, y_minus_Sm, inf)
    idx_min_Sp, v_min_Sp = _argmin_3d(Sp_masked)
    idx_min_residual, v_min_residual = _argmin_3d(ymSm_masked)

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

    # Compute the upper triangle of HtH and mirror it.
    for i in range(num_cols):
        h_i = _get_column_H(i, plastic_sino_est, metal_sino_est, H_exponent_list)
        Hty[i] = float(torch.sum(h_i * measured_sino))
        for j in range(i, num_cols):
            h_j = _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)
            dot_ij = float(torch.sum(h_i * h_j))
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

def _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter=10, tolerance=-1e-5, view_mask=None):
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
        idx_min_Sp, v_min_Sp, idx_min_residual, v_min_residual = _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, view_mask=view_mask)

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
            u_m = np.array([max(float(measured_sino[idx_min_residual]), 0.0)])
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


def _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, num_metal_terms, p_normalization, gamma, view_mask=None, num_real_pixels=None):
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
    and numerical instability.  (``view_mask`` / ``num_real_pixels`` restrict the mean to the real views
    when the sinogram is zero-padded on the view axis.)

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
    Sp = torch.zeros_like(measured_sino)
    for i in range(0, 1 + num_cross_terms):
        Sp = Sp + float(theta[i]) * _get_column_H(i, plastic_sino_est, metal_sino_est, p_coeff_exponents)

    y_minus_Sm = measured_sino
    # Subtract metal-only terms (from H columns after the cross terms)
    for j in range(1 + num_cross_terms, 1 + num_cross_terms + num_metal_terms):
        y_minus_Sm = y_minus_Sm - float(theta[j]) * _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)

    # Enforce non-negativity on the residual sinogram (plastic + cross terms)
    y_minus_Sm = torch.clamp(y_minus_Sm, min=0)

    # Central plastic coefficient, used to define a stabilization floor.  The MEAN (rather than the
    # median) is a cheap reduction; over the sinogram support the two are close, and this only sets a
    # floor.  When the sinogram is zero-padded on the view axis, exclude the padded views via
    # view_mask so they don't drag the mean toward 0.
    if view_mask is None:
        mean_plastic_coef = torch.mean(Sp)
    else:
        mean_plastic_coef = torch.sum(Sp * view_mask) / float(num_real_pixels)
    Sp_floor = gamma * mean_plastic_coef

    # A negative mean would be non-physical and may indicate instability in the algorithm
    # In that case, issue a runtime warning to flag the potential problem
    if float(mean_plastic_coef) <= 0:
        warnings.warn("Mean of Sp is negative", RuntimeWarning)

    # Clamp Sp at Sp_floor to prevent division by very small or negative values
    clamped_plastic_coef = torch.maximum(Sp, Sp_floor)
    corrected_plastic_sino = p_normalization * y_minus_Sm / clamped_plastic_coef

    return corrected_plastic_sino

def _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected):
    # Compute a scaling factor by performing least-squares fitting between the corrected plastic sinogram
    # and the measured sinogram at plastic-only locations (i.e., where plastic is present and all metals are absent)
    metal_absent = torch.ones_like(plastic_sino_est, dtype=torch.bool)
    for metal in metal_sino_est:
        metal_absent = metal_absent & (metal == 0)

    # Plastic-only locations.  Zero out the other locations and let compute_scaling_factor's inner
    # products (sum(a*b)/sum(b*b)) do the reduction -- no data-dependent-shape selection needed.
    condition = (plastic_sino_est != 0) & metal_absent

    zero = torch.zeros((), dtype=measured_sino.dtype, device=measured_sino.device)
    plastic_sino_scale = mtp.compute_scaling_factor(torch.where(condition, measured_sino, zero),
                                                    torch.where(condition, plastic_sino_corrected, zero))
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

    # Real-view mask (True on real views, False on padded; None when unpadded), plus the real pixel
    # count -- used to exclude padded views from the statistical reductions (the Sp mean floor and the
    # constraint argmins).
    pl = ct_model.sino_placement
    view_mask = pl.real_mask(measured_sino.ndim)
    if view_mask is None:
        num_real_pixels = None
    else:
        view_mask = torch.as_tensor(np.asarray(view_mask), device=measured_sino.device)
        num_real_pixels = pl.real_size * (measured_sino.numel() // measured_sino.shape[0])

    # Get normalized sinogram p and [m_0, m_1, ...].
    plastic_sino_est, metal_sino_est = _est_plastic_metal_sinos_from_recon(
        recon, num_metal, ct_model, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin)
    plastic_sino_scale = torch.max(torch.abs(plastic_sino_est))   # max over padded 0s is unaffected
    metal_sino_scale = [torch.max(torch.abs(arr)) for arr in metal_sino_est]
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
    plastic_sino_est = plastic_sino_est / plastic_sino_scale
    metal_sino_est = [arr / norm for arr, norm in zip(metal_sino_est, metal_sino_scale)]

    # Estimate beam hardening model parameters theta
    theta = _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter, view_mask=view_mask)

    # Compute the corrected plastic sinogram
    plastic_sino_corrected = _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list,
                                                       num_cross_terms, num_metal_terms, float(plastic_sino_scale), gamma,
                                                       view_mask=view_mask, num_real_pixels=num_real_pixels)

    # Compute and apply the scaling of the corrected plastic sino
    plastic_sino_corrected_scale = _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected)
    scaled_corrected_plastic_sino = plastic_sino_corrected_scale * plastic_sino_corrected

    # Combine the scaled corrected plastic sino and the metal sinos, then gather to a host sinogram
    # for the downstream recon.
    corrected_sino = scaled_corrected_plastic_sino + sum(arr * norm for arr, norm in zip(metal_sino_est, metal_sino_scale))
    return ct_model._gather_sinogram(corrected_sino)


def recon_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, num_constraint_update_iter=10, stop_threshold_change_pct=0.2,
                        num_metal=1, order=3, alpha=1, beta=0.002, gamma=0.1, verbose=0, output_sharded=False,
                        max_iterations=15, logfile_path='~/.mbirtorch/logs/recon.log',
                        radial_margin=None, top_margin=None, bottom_margin=None):
    """
    Perform iterative metal artifact reduction using plastic-metal beam hardening correction.  If num_metal is 0,
    then this performs a standard MBIR recon.

    This function alternates between adaptive beam hardening correction (via `correct_sino_plastic_metal`)
    and reconstruction, refining the image over several iterations to suppress metal-induced artifacts.

    Args:
        ct_model: MBIRTORCH cone beam model instance with `direct_recon` and `recon` methods.
        sino (ndarray):  Input sinogram data to be corrected.
        weights (ndarray): Transmission weights used in the reconstruction algorithm.
        num_BH_iterations (int, optional): Number of correction-reconstruction iterations. Defaults to 3.
        num_constraint_update_iter (int, optional): Number of iterations for updating constraints.
            At each iteration, the most violated constraints are activated and the quadratic program is re-solved via OSQP.
        stop_threshold_change_pct (float, optional): Relative change threshold (%) for early stopping in MBIR. Defaults to 0.2.
        num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
        order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
        alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
            higher-order terms more strongly. Defaults to 1.
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.002.
        gamma (float, optional): Stabilization factor used in plastic correction. Multiplies the mean of `s_p`
            to set a positive floor in the denominator, preventing division by near-zero or negative values. Defaults to 0.1.
        verbose (int, optional): Verbosity level for printing intermediate information. Defaults to 0.
        output_sharded (bool, optional): Choose the form of the returned reconstruction.  If False
            (default), return an ordinary host NumPy array.  If True, return the device tensor for a
            following on-device step.
        max_iterations (int, optional): Maximum MBIR iterations per reconstruction pass. Defaults to 15.
        logfile_path (str, optional): Accepted for interface compatibility; per-pass log files are not
            currently written.
        radial_margin, top_margin, bottom_margin (int or None, optional): Segmentation mask margins
            used when classifying plastic/metal; None (default) = size-relative
            (see segment_plastic_metal).

    Returns:
         numpy array or tensor: The final corrected reconstruction after iterative beam hardening
         correction -- a host NumPy array by default, or a device tensor if ``output_sharded=True``.

    Example:
        >>> recon = recon_plastic_metal(
        ...     ct_model, sino, weights,
        ...     num_BH_iterations=3,
        ...     stop_threshold_change_pct=0.2,
        ...     num_metal=1,
        ...     order=3,
        ...     alpha=1,
        ...     beta=0.005,
        ...     verbose=1
        ... )
        >>> mt.slice_viewer(recon)
    """
    # Check for nonnegative num_metals
    if num_metal < 0:
        raise ValueError("num_metal must be >= 0")

    # Use split sino recon for cone beam when the model provides it (it splits on the host so the
    # full sinogram is never device-resident); otherwise use the standard recon with a device-form
    # output so the next correction consumes it with no gather/re-upload.
    if 'cone' in ct_model.get_params('geometry_type') and hasattr(ct_model, 'split_sino_recon'):
        recon_function = ct_model.split_sino_recon
    else:
        recon_function = functools.partial(ct_model.recon, output_sharded=True)

    # Deliver the user-requested output form: _shard_recon / _gather_recon are each a no-op when the
    # loop's final recon is already in that form.
    def to_output_form(r):
        return ct_model._shard_recon(r) if output_sharded else ct_model._gather_recon(r)

    # Do a regular recon if num_metal == 0
    if num_metal == 0:
        recon, _ = recon_function(sino, weights=weights, max_iterations=max_iterations,
                                  stop_threshold_change_pct=stop_threshold_change_pct)
        return to_output_form(recon)

    # Continue with beam hardening and segmentation
    if verbose >= 1:
        print("\n************ Perform initial FDK reconstruction  **************")
    recon = ct_model.direct_recon(sino, output_sharded=True)

    for i in range(num_BH_iterations):
        # Estimate Corrected Sinogram
        if verbose >= 1:
            print(f"\n************ Correct sino plastic metal {i + 1}  **************")
        corrected_sinogram = correct_sino_plastic_metal(ct_model, sino, recon, num_metal=num_metal, order=order, alpha=alpha, beta=beta, gamma=gamma, num_constraint_update_iter=num_constraint_update_iter,
                                                        radial_margin=radial_margin, top_margin=top_margin, bottom_margin=bottom_margin)

        # Reconstruct Corrected Sinogram
        if verbose >= 1:
            print(f"\n************ Perform MBIR reconstruction {i + 1} **************")
        recon, _ = recon_function(corrected_sinogram, weights=weights, init_recon=recon,
                                  max_iterations=max_iterations,
                                  stop_threshold_change_pct=stop_threshold_change_pct)

        if verbose >= 2:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_masks, plastic_scale, metal_scales = mtp.segment_plastic_metal(
                recon, num_metal, radial_margin=radial_margin, top_margin=top_margin,
                bottom_margin=bottom_margin)
            labels = ['Plastic Mask'] + [f'Metal {j + 1} Mask' for j in range(len(metal_masks))]
            mt.slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0,
                            slice_label=labels,
                            title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

    return to_output_form(recon)
