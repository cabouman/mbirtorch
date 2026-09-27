import torch


def _randomized_svd(X, n_components, n_oversamples=10, n_iter=4, seed=0):
    """Truncated SVD via a randomized range finder.

    A full SVD of an (n_samples, n_features) matrix is wasted work when only a
    handful of components are wanted, and here n_components is the material count
    -- rarely above 20 against a thousand or more wavelength bins. Seeded so the
    result is reproducible.
    """
    n_rows, n_cols = X.shape
    rank = min(n_components + n_oversamples, n_rows, n_cols)
    generator = torch.Generator(device=X.device).manual_seed(seed)
    Q, _ = torch.linalg.qr(X @ torch.randn(n_cols, rank, generator=generator,
                                           dtype=X.dtype, device=X.device))
    for _ in range(n_iter):                      # power iterations sharpen the range
        Q, _ = torch.linalg.qr(X.T @ Q)
        Q, _ = torch.linalg.qr(X @ Q)
    U, singular_values, Vh = torch.linalg.svd(Q.T @ X, full_matrices=False)
    return Q @ U, singular_values, Vh


def nndsvda(X, n_components):
    """NNDSVD initialization for X ~= W @ H, with zeros filled.

    Every component after the first is one sign-half of a singular vector pair, so roughly half its entries are zero.
    They are filled with sqrt(mean X): each factor carries sqrt(s_k), so that is the scale of a factor entry, and the
    fill adds a fixed fraction of a typical entry to the product whatever the scale of X. A fill far below factor
    scale would be frozen at zero by the two-metric projection of block_newton_step.

    Args:
        X: Nonnegative array of shape (n_samples, n_features).
        n_components: Factorization rank.

    Returns:
        W: Shape (n_samples, n_components).
        H: Shape (n_components, n_features).
    """
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if n_components + 10 < min(X.shape):
        U, singular_values, Vh = _randomized_svd(X, n_components)
    else:
        U, singular_values, Vh = torch.linalg.svd(X, full_matrices=False)

    n_components = min(
        n_components,
        U.shape[1],
        Vh.shape[0],
    )

    W = torch.zeros((X.shape[0], n_components), dtype=X.dtype, device=X.device)
    H = torch.zeros((n_components, X.shape[1]), dtype=X.dtype, device=X.device)

    # First singular triplet.
    scale = torch.sqrt(singular_values[0])
    W[:, 0] = scale * torch.abs(U[:, 0])
    H[0, :] = scale * torch.abs(Vh[0, :])

    # Remaining components.
    for component in range(1, n_components):
        u = U[:, component]
        v = Vh[component, :]

        u_pos = torch.clip(u, min=0)
        u_neg = torch.clip(-u, min=0)
        v_pos = torch.clip(v, min=0)
        v_neg = torch.clip(-v, min=0)

        positive_strength = (
            torch.linalg.norm(u_pos) * torch.linalg.norm(v_pos)
        )
        negative_strength = (
            torch.linalg.norm(u_neg) * torch.linalg.norm(v_neg)
        )

        use_positive = positive_strength > negative_strength

        selected_u = torch.where(use_positive, u_pos, u_neg)
        selected_v = torch.where(use_positive, v_pos, v_neg)

        selected_u /= torch.linalg.norm(selected_u) + torch.finfo(X.dtype).eps
        selected_v /= torch.linalg.norm(selected_v) + torch.finfo(X.dtype).eps

        scale = torch.sqrt(singular_values[component])
        W[:, component] = scale * selected_u
        H[component, :] = scale * selected_v

    fill_value = torch.sqrt(torch.mean(X).clamp_min(0))
    W = torch.where(W == 0, fill_value, W)
    H = torch.where(H == 0, fill_value, H)

    return W, H


def _batched_spd_solve(M, g, jitter_rel=1e-9):
    """Solve M[b] d[b] = g[b] for a batch of small SPD matrices.

    The system is solved in Jacobi scaling, D^-1/2 M D^-1/2 with D = diag(M), so the Tikhonov damping is relative to
    each diagonal entry: a component whose curvature is orders of magnitude below another's (a skewed gauge, or a
    component next to a huge one) keeps its Newton step, and rescaling a component leaves the solve equivariant. The
    damping keeps the factorization well posed when Z is near zero (heavily attenuated pixels), and the fallback is
    branch-free so no host synchronization is introduced inside the iteration.
    """
    rank = M.shape[-1]
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    s = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(M.dtype).tiny).sqrt()
    A = M / (s[:, :, None] * s[:, None, :]) + jitter_rel * eye
    L, info = torch.linalg.cholesky_ex(A)
    failed = (info > 0)[:, None, None]
    L = torch.where(failed, eye.expand_as(L), L)
    gs = g / s
    d = torch.cholesky_solve(gs.unsqueeze(-1), L).squeeze(-1)
    d = torch.where(failed[:, :, 0], gs / torch.diagonal(A, dim1=-2, dim2=-1), d) / s
    # A row with no curvature (Z underflows in float32 above an attenuation of about 88) makes the scaling divide by
    # `tiny` and overflow; take no step there and let later gradient-driven steps bring the iterate back into range.
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _joint_dot(a, b, c, d):
    return (a * b).sum() + (c * d).sum()


def _joint_blocks(flat, rows, cols, rank, free, jitter):
    """(B,Q) upper triangle -> damped SPD (B,R,R) with frozen vars set to identity."""
    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    # Damping relative to each diagonal entry (Jacobi), so that one component's curvature does not set another's.
    M = M + torch.diag_embed(jitter * torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(M.dtype).tiny))
    L, info = torch.linalg.cholesky_ex(M)
    return torch.where((info > 0)[:, None, None], eye.expand_as(L), L)


def _reseed_dead(W, H, T=None, seed=0):
    """Re-seed any component whose map or spectrum is zero, both factors, with small random values. Returns (W, H,
    number re-seeded).

    With its spectrum at zero the map gets no gradient, and a spectrum whose every bin has an outward gradient stays at
    zero, so the component contributes nothing from then on: a degenerate stationary point, which the first projected
    step can reach from an ordinary start. The projected steps set a dying factor exactly to zero, and the test is for
    exact zeros rather than for a norm small next to the other components': how a component's scale splits between
    its map and its spectrum is arbitrary, and at low dose the fit can send one component to a huge scale on the zero
    counts, next to which a live component looks dead. For the same reasons the seed's scale comes from the data,
    sqrt(mean attenuation / R) in both factors (given T; otherwise the median over the live components of the
    geometric mean of their mean map and spectrum entries). The generator is seeded with `seed`, so repeated re-seeds
    of a component can differ, and a random rather than constant seed keeps the revived spectrum from being flat.
    """
    dead = (W.amax(0) <= 0) | (H.amax(1) <= 0)
    n_dead = int(dead.sum())
    if n_dead == 0:
        return W, H, 0
    live = ~dead
    W = W.clone()
    H = H.clone()
    if T is not None:
        scale = _attenuation_scale(T, W.shape[1])
    elif bool(live.any()):
        scale = float((W[:, live].mean(0).double() * H[live].mean(1).double()).sqrt().median())
    else:
        scale = 1.0
    g = torch.Generator(device=W.device).manual_seed(int(seed))
    W[:, dead] = 1e-2 * scale * torch.rand(W.shape[0], n_dead, generator=g, dtype=W.dtype, device=W.device)
    H[dead] = 1e-2 * scale * torch.rand(n_dead, H.shape[1], generator=g, dtype=H.dtype, device=H.device)
    return W, H, n_dead


def _attenuation_scale(T, rank, chunk=2 ** 23):
    """sqrt(mean attenuation / rank) over the entries with counts, in float64 and by blocks of rows: the scale of one
    factor entry of a component carrying 1/rank of a typical attenuation."""
    rows = max(1, chunk // max(T.shape[1], 1))
    total = torch.zeros((), dtype=torch.float64, device=T.device)
    count = 0
    for i in range(0, T.shape[0], rows):
        t = T[i:i + rows]
        real = t > 1e-12
        total = total + torch.where(real, -torch.log(t.clamp_min(1e-12)), 0).sum(dtype=torch.float64)
        count += int(real.sum())
    mean = float(total) / count if count else 1.0
    return max(mean, 0.0) ** 0.5 / max(rank, 1) ** 0.5


def _attenuation_for_start(T):
    """-log T with zero counts floored at half the smallest positive transmission (the model is X = W H, X = -log T);
    transmissions below 1e-12 count as zero counts."""
    real = T > 1e-12
    if bool(real.any()) and not bool(real.all()):
        T = torch.where(real, T, 0.5 * T[real].min())
    else:
        T = T.clamp_min(torch.finfo(T.dtype).tiny)
    return -torch.log(T)


def _nonneg_least_squares_start(A, H, ridge=1e-6):
    """W >= 0 approximately minimizing ||A - W H||, from the R x R normal equations with a ridge relative to each
    diagonal entry (so rescaling a row of H rescales its coefficient and nothing else), solved by Cholesky in float64
    so that a rank-deficient H gives finite values on every device."""
    Hd = H.double()
    G = Hd @ Hd.T
    diag = torch.diagonal(G)
    floor = 1e-12 * diag.mean().clamp_min(torch.finfo(torch.float64).tiny)     # a zero row of H: keep G definite
    G = G + ridge * torch.diag_embed(torch.maximum(diag, floor))
    W = torch.cholesky_solve((A.double() @ Hd.T).T, torch.linalg.cholesky(G)).T
    return W.clamp_(min=0).to(A.dtype)
