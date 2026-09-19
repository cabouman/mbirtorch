"""The masked hessian used by every reconstruction.

The hessian diagonal is computed only at the indices inside the region of
reconstruction.  The test here requires that those values equal the ones a
dense grid computes at the same indices.  It runs on CPU.
"""

import numpy as np
import mbirtorch
# ── the masked hessian ───────────────────────────────────────────────────────
def test_masked_hessian_agrees_with_the_full_grid_at_the_masked_indices():
    """The only places the engine ever reads the hessian.

    Back projection is independent per pixel, so a masked run must reproduce
    the dense run exactly at every masked index.  Outside the mask the masked
    run holds zeros instead of computed-but-never-read values.
    """
    angles = np.linspace(0, np.pi, 12, endpoint=False)
    model = mbirtorch.ParallelBeamModel((12, 8, 10), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    weights = np.abs(np.random.RandomState(0).randn(12, 8, 10)).astype(np.float32) + 0.5

    dense = model.compute_hessian_diagonal(weights=weights)
    indices = model.full_indices_device()
    masked = model.compute_hessian_diagonal(weights=weights, indices=indices)

    shape = tuple(model.get_params('recon_shape'))
    flat_dense = dense.reshape(-1, shape[2])
    flat_masked = masked.reshape(-1, shape[2])
    idx = indices.cpu().numpy()
    np.testing.assert_array_equal(flat_masked[idx], flat_dense[idx])
    # Outside the mask the masked form is exactly zero.
    outside = np.setdiff1d(np.arange(shape[0] * shape[1]), idx)
    if outside.size:
        assert np.all(flat_masked[outside] == 0)
        assert np.any(flat_dense[outside] != 0)


# The granularity list those runs used, which is the library default.
