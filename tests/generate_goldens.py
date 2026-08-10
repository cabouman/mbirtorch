"""Generate cross-framework golden data from mbirjax (a jax-side script writes
the goldens; mbirtorch tests read them, so the mbirtorch test env never
imports jax).

Run in the mbirjax conda env:
    /Users/gbuzzard/miniforge3/envs/mbirjax/bin/python tests/generate_goldens.py

Writes tests/goldens/golden_<cell>.npz (gitignored; regenerate at will).  The
recorded jax version is the frozen comparison baseline (0.10.1).

Contents per cell: the shepp-logan phantom, its sinogram, transmission-root
weights, sparse fwd/back outputs on a fixed subset, the qGGMRF gradient and
Hessian on that subset, the Hessian diagonal, the FBP recon, the auto-set
regularization values, and a seeded 5-iteration recon with its per-iteration
traces (fm_rmse, alpha, nmae) -- the convergence-parity reference.
"""

import os

import numpy as np

CELL = (64, 64, 64)          # small: the parity gate is per-iteration, not per-size
SEED = 42
RECON_SEED = 7
MAX_ITERATIONS = 5
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")


def main():
    import jax
    import mbirjax

    os.makedirs(OUT_DIR, exist_ok=True)
    n_views, n_rows, n_channels = CELL
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(CELL, angles)
    recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))

    phantom = np.asarray(mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape))
    sinogram = np.asarray(model.forward_project(phantom))
    weights = np.asarray(mbirjax.gen_weights(sinogram, weight_type='transmission_root'))

    # Sparse projector goldens on a fixed subset.
    rng = np.random.RandomState(SEED)
    full_indices = np.asarray(mbirjax.gen_full_indices(
        recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
    subset = np.sort(rng.choice(full_indices, size=min(500, len(full_indices)),
                                replace=False))
    voxel_values = rng.rand(len(subset), recon_shape[2]).astype(np.float32)
    sparse_fwd = np.asarray(model.sparse_forward_project(voxel_values, subset))
    sparse_back = np.asarray(model.sparse_back_project(sinogram, subset))

    # qGGMRF goldens at the same subset on the phantom.
    flat_phantom = phantom.reshape(-1, recon_shape[2])
    b = mbirjax.get_b_from_nbr_wts(model.get_params('qggmrf_nbr_wts'))
    sigma_x, p, q, T = model.get_params(['sigma_x', 'p', 'q', 'T'])
    qggmrf_params = tuple((b, float(sigma_x), float(p), float(q), float(T)))
    qg_grad, qg_hess = mbirjax.qggmrf_gradient_and_hessian_at_indices(
        flat_phantom, recon_shape, subset, qggmrf_params)

    hessian_diag = np.asarray(model.compute_hessian_diagonal(weights=weights))
    fbp = np.asarray(model.fbp_recon(sinogram))

    # Auto-set regularization values on this sinogram+weights.
    reg = model.auto_set_regularization_params(sinogram, weights=weights)

    # Seeded short recon with traces (verbose off to keep logs quiet).
    model.set_params(no_warning=True, verbose=0)
    np.random.seed(RECON_SEED)
    recon, recon_dict = model.recon(sinogram, weights=weights,
                                    max_iterations=MAX_ITERATIONS,
                                    stop_threshold_change_pct=0.0)
    rp = recon_dict['recon_params']
    if not isinstance(rp, dict):
        rp = dict(rp)

    # Denoiser golden: seeded AWGN on the phantom, explicit sigma, exact
    # iteration count (the harness's determinism recipe), seeded partition.
    noisy = phantom + 0.1 * np.random.RandomState(21).randn(*phantom.shape).astype(np.float32)
    denoiser = mbirjax.QGGMRFDenoiser(recon_shape)
    denoiser.set_params(no_warning=True, verbose=0)
    np.random.seed(RECON_SEED)
    denoised, den_dict = denoiser.denoise(noisy, sigma_noise=0.1,
                                          max_iterations=5,
                                          stop_threshold_change_pct=0.0,
                                          print_logs=False)
    den_rp = den_dict['recon_params']
    den_sigma_est = float(denoiser.estimate_image_noise_std(noisy))

    # Cone golden: the harness geometry convention (sdd = 4 * channels,
    # sid = sdd / 2, magnification 2), full-circle angles, seeded recon traces.
    cone_cell = (48, 48, 48)
    cone_angles = np.linspace(0, 2 * np.pi, cone_cell[0], endpoint=False)
    cone = mbirjax.ConeBeamModel(cone_cell, cone_angles,
                                 source_detector_dist=4 * cone_cell[2],
                                 source_iso_dist=2 * cone_cell[2])
    cone.set_params(no_warning=True, verbose=0)
    cone_recon_shape = tuple(int(x) for x in cone.get_params('recon_shape'))
    cone_phantom = np.asarray(
        mbirjax.generate_3d_shepp_logan_low_dynamic_range(cone_recon_shape))
    cone_sino = np.asarray(cone.forward_project(cone_phantom))
    cone_weights = np.asarray(mbirjax.gen_weights(cone_sino / max(1e-6, cone_sino.max()),
                                                  weight_type='transmission_root'))
    cone_full = np.asarray(mbirjax.gen_full_indices(
        cone_recon_shape, use_ror_mask=cone.get_params('use_ror_mask')))
    cone_subset = np.sort(np.random.RandomState(SEED).choice(
        cone_full, size=min(400, len(cone_full)), replace=False))
    cone_vals = np.random.RandomState(SEED + 1).rand(
        len(cone_subset), cone_recon_shape[2]).astype(np.float32)
    cone_sp_fwd = np.asarray(cone.sparse_forward_project(cone_vals, cone_subset))
    cone_sp_back = np.asarray(cone.sparse_back_project(cone_sino, cone_subset))
    cone_hess = np.asarray(cone.compute_hessian_diagonal(weights=cone_weights))
    cone_fdk = np.asarray(cone.fdk_recon(cone_sino))
    np.random.seed(RECON_SEED)
    cone_recon, cone_dict = cone.recon(cone_sino, weights=cone_weights,
                                       max_iterations=MAX_ITERATIONS,
                                       stop_threshold_change_pct=0.0)
    cone_rp = cone_dict['recon_params']

    # Helical cone golden: same cell and distances, flat detector, z-shifts
    # spanning a travel of +-8 ALU (detector half-height at iso is 12, so the
    # shifted views keep partial coverage -- the helical z-weight is nontrivial).
    # Includes a seeded 3-iteration recon: the z-shift path runs through the
    # FDK init (z-weight) and the per-view vertical fans in the VCD loop.
    chel_shifts = np.linspace(-8.0, 8.0, cone_cell[0]).astype(np.float32)
    chel = mbirjax.ConeBeamModel(cone_cell, cone_angles,
                                 source_detector_dist=4 * cone_cell[2],
                                 source_iso_dist=2 * cone_cell[2],
                                 helical_z_shifts=chel_shifts)
    chel.set_params(no_warning=True, verbose=0)
    chel_recon_shape = tuple(int(x) for x in chel.get_params('recon_shape'))
    chel_phantom = np.asarray(
        mbirjax.generate_3d_shepp_logan_low_dynamic_range(chel_recon_shape))
    chel_sino = np.asarray(chel.forward_project(chel_phantom))
    chel_weights = np.asarray(mbirjax.gen_weights(chel_sino / max(1e-6, chel_sino.max()),
                                                  weight_type='transmission_root'))
    chel_full = np.asarray(mbirjax.gen_full_indices(
        chel_recon_shape, use_ror_mask=chel.get_params('use_ror_mask')))
    chel_subset = np.sort(np.random.RandomState(SEED).choice(
        chel_full, size=min(400, len(chel_full)), replace=False))
    chel_vals = np.random.RandomState(SEED + 1).rand(
        len(chel_subset), chel_recon_shape[2]).astype(np.float32)
    chel_sp_fwd = np.asarray(chel.sparse_forward_project(chel_vals, chel_subset))
    chel_sp_back = np.asarray(chel.sparse_back_project(chel_sino, chel_subset))
    chel_fdk = np.asarray(chel.fdk_recon(chel_sino))
    np.random.seed(RECON_SEED)
    chel_recon, chel_dict = chel.recon(chel_sino, weights=chel_weights,
                                       max_iterations=3,
                                       stop_threshold_change_pct=0.0)
    chel_rp = chel_dict['recon_params']

    # Curved-detector cone golden: circular scan, curved detector (the
    # horizontal fan's theta = u/sdd branch, no cos correction).  Single-op
    # coverage: the VCD loop is identical given the operators.
    ccurv = mbirjax.ConeBeamModel(cone_cell, cone_angles,
                                  source_detector_dist=4 * cone_cell[2],
                                  source_iso_dist=2 * cone_cell[2],
                                  use_curved_detector=True)
    ccurv.set_params(no_warning=True, verbose=0)
    ccurv_recon_shape = tuple(int(x) for x in ccurv.get_params('recon_shape'))
    ccurv_phantom = np.asarray(
        mbirjax.generate_3d_shepp_logan_low_dynamic_range(ccurv_recon_shape))
    ccurv_sino = np.asarray(ccurv.forward_project(ccurv_phantom))
    ccurv_full = np.asarray(mbirjax.gen_full_indices(
        ccurv_recon_shape, use_ror_mask=ccurv.get_params('use_ror_mask')))
    ccurv_subset = np.sort(np.random.RandomState(SEED).choice(
        ccurv_full, size=min(400, len(ccurv_full)), replace=False))
    ccurv_vals = np.random.RandomState(SEED + 1).rand(
        len(ccurv_subset), ccurv_recon_shape[2]).astype(np.float32)
    ccurv_sp_fwd = np.asarray(ccurv.sparse_forward_project(ccurv_vals, ccurv_subset))
    ccurv_sp_back = np.asarray(ccurv.sparse_back_project(ccurv_sino, ccurv_subset))
    ccurv_fdk = np.asarray(ccurv.fdk_recon(ccurv_sino))

    # Translation (TCT) golden: a 4x4 grid of translations, dot phantom,
    # single ops, FDK, and a seeded 3-iteration recon with traces.
    tct_dets = (40, 32)
    tct_tvecs = np.asarray(mbirjax.gen_translation_vectors(
        4, 4, x_spacing=3.0, z_spacing=2.0), dtype=np.float32)
    tct_cell = (tct_tvecs.shape[0],) + tct_dets
    tct = mbirjax.TranslationModel(tct_cell, tct_tvecs,
                                   source_detector_dist=4 * tct_dets[1],
                                   source_iso_dist=tct_dets[1])
    tct.set_params(no_warning=True, verbose=0)
    tct_recon_shape = tuple(int(x) for x in tct.get_params('recon_shape'))
    tct_phantom = np.asarray(mbirjax.utilities.gen_translation_phantom(
        tct_recon_shape, 'dots', None, fill_rate=0.05))
    tct_sino = np.asarray(tct.forward_project(tct_phantom))
    tct_full = np.asarray(mbirjax.gen_full_indices(
        tct_recon_shape, use_ror_mask=tct.get_params('use_ror_mask')))
    tct_subset = np.sort(np.random.RandomState(SEED).choice(
        tct_full, size=min(400, len(tct_full)), replace=False))
    tct_vals = np.random.RandomState(SEED + 1).rand(
        len(tct_subset), tct_recon_shape[2]).astype(np.float32)
    tct_sp_fwd = np.asarray(tct.sparse_forward_project(tct_vals, tct_subset))
    tct_sp_back = np.asarray(tct.sparse_back_project(tct_sino, tct_subset))
    tct_fdk = np.asarray(tct.fdk_recon(tct_sino))
    np.random.seed(RECON_SEED)
    tct_recon, tct_dict = tct.recon(tct_sino, max_iterations=3,
                                    stop_threshold_change_pct=0.0)
    tct_rp = tct_dict['recon_params']

    out = os.path.join(OUT_DIR, f"golden_{'x'.join(map(str, CELL))}.npz")
    np.savez_compressed(
        out,
        jax_version=jax.__version__,
        cell=np.array(CELL), recon_shape=np.array(recon_shape), angles=angles,
        phantom=phantom.astype(np.float32), sinogram=sinogram.astype(np.float32),
        weights=weights.astype(np.float32),
        subset=subset, voxel_values=voxel_values,
        sparse_fwd=sparse_fwd.astype(np.float32),
        sparse_back=sparse_back.astype(np.float32),
        qg_grad=np.asarray(qg_grad, dtype=np.float32),
        qg_hess=np.asarray(qg_hess, dtype=np.float32),
        qggmrf_sigma_x=np.float32(sigma_x),
        hessian_diag=hessian_diag.astype(np.float32),
        fbp=fbp.astype(np.float32),
        sigma_y=np.float32(reg['sigma_y']), sigma_x_auto=np.float32(reg['sigma_x']),
        sigma_prox=np.float32(reg['sigma_prox']),
        recon=np.asarray(recon, dtype=np.float32),
        fm_rmse=np.array(rp['fm_rmse']), alpha_values=np.array(rp['alpha_values']),
        nmae_pct=np.array(rp['stop_threshold_change_pct']),
        recon_seed=np.array(RECON_SEED), max_iterations=np.array(MAX_ITERATIONS),
        den_noisy=noisy.astype(np.float32),
        den_out=np.asarray(denoised, dtype=np.float32),
        den_alpha=np.array(den_rp['alpha_values']),
        den_nmae_pct=np.array(den_rp['stop_threshold_change_pct']),
        den_sigma_est=np.float32(den_sigma_est),
        cone_cell=np.array(cone_cell), cone_angles=cone_angles,
        cone_recon_shape=np.array(cone_recon_shape),
        cone_sdd=np.float32(4 * cone_cell[2]), cone_sid=np.float32(2 * cone_cell[2]),
        cone_slice_offset=np.float32(cone.get_params('recon_slice_offset')),
        cone_phantom=cone_phantom.astype(np.float32),
        cone_sino=cone_sino.astype(np.float32),
        cone_weights=cone_weights.astype(np.float32),
        cone_subset=cone_subset, cone_vals=cone_vals,
        cone_sp_fwd=cone_sp_fwd.astype(np.float32),
        cone_sp_back=cone_sp_back.astype(np.float32),
        cone_hess=cone_hess.astype(np.float32),
        cone_fdk=cone_fdk.astype(np.float32),
        cone_recon=np.asarray(cone_recon, dtype=np.float32),
        cone_alpha=np.array(cone_rp['alpha_values']),
        cone_fm_rmse=np.array(cone_rp['fm_rmse']),
        cone_nmae_pct=np.array(cone_rp['stop_threshold_change_pct']),
        chel_shifts=chel_shifts,
        chel_recon_shape=np.array(chel_recon_shape),
        chel_phantom=chel_phantom.astype(np.float32),
        chel_sino=chel_sino.astype(np.float32),
        chel_weights=chel_weights.astype(np.float32),
        chel_subset=chel_subset, chel_vals=chel_vals,
        chel_sp_fwd=chel_sp_fwd.astype(np.float32),
        chel_sp_back=chel_sp_back.astype(np.float32),
        chel_fdk=chel_fdk.astype(np.float32),
        chel_recon=np.asarray(chel_recon, dtype=np.float32),
        chel_alpha=np.array(chel_rp['alpha_values']),
        chel_fm_rmse=np.array(chel_rp['fm_rmse']),
        ccurv_recon_shape=np.array(ccurv_recon_shape),
        ccurv_phantom=ccurv_phantom.astype(np.float32),
        ccurv_sino=ccurv_sino.astype(np.float32),
        ccurv_subset=ccurv_subset, ccurv_vals=ccurv_vals,
        ccurv_sp_fwd=ccurv_sp_fwd.astype(np.float32),
        ccurv_sp_back=ccurv_sp_back.astype(np.float32),
        ccurv_fdk=ccurv_fdk.astype(np.float32),
        tct_cell=np.array(tct_cell), tct_tvecs=tct_tvecs,
        tct_sdd=np.float32(4 * tct_dets[1]), tct_sid=np.float32(tct_dets[1]),
        tct_recon_shape=np.array(tct_recon_shape),
        tct_delta_voxel=np.float32(tct.get_params('delta_voxel')),
        tct_voxel_row_aspect=np.float32(tct.get_params('voxel_row_aspect')),
        tct_phantom=tct_phantom.astype(np.float32),
        tct_sino=tct_sino.astype(np.float32),
        tct_subset=tct_subset, tct_vals=tct_vals,
        tct_sp_fwd=tct_sp_fwd.astype(np.float32),
        tct_sp_back=tct_sp_back.astype(np.float32),
        tct_fdk=tct_fdk.astype(np.float32),
        tct_recon=np.asarray(tct_recon, dtype=np.float32),
        tct_alpha=np.array(tct_rp['alpha_values']),
        tct_fm_rmse=np.array(tct_rp['fm_rmse']),
        tct_nmae_pct=np.array(tct_rp['stop_threshold_change_pct']),
    )
    print('wrote', out)


if __name__ == '__main__':
    main()
