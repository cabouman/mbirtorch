"""Generate cross-framework golden data for the preprocess package from
mbirjax (a jax-side script writes the goldens; mbirtorch tests read them, so
the mbirtorch test env never imports jax).

Run in the mbirjax conda env:
    python tests/generate_preprocess_goldens.py

Writes tests/goldens/preprocess_goldens.npz and preprocess_goldens_cone_save.h5
(gitignored; regenerate at will).

Contents: seeded scan triples (object/blank/dark) with a defective-pixel list,
and the mbirjax outputs of the scan-to-sinogram chain -- transmission,
defective-pixel interpolation, detector rotation, background offset,
downsampling, the fused scan_to_sino, zinger correction, blank-margin
detection, the crop/geometry helpers, the cylindrical mask, the beam-hardening
curve family, and the small helpers.  It also carries the demo-data utilities:
the reference Shepp-Logan phantom on a non-cubic shape, the phantom, sinogram
and view geometry that generate_demo_data produces for the parallel, cone, and
helical-cone cases, and the recon shapes get_ct_model builds.  The HDF5 file is
an mbirjax-written save_cone_preprocessing output, pinning the on-disk layout as
shared between the two packages.  It carries the format tag
'mbirjax_preprocessing_v1'; mbirtorch now writes 'mbirtorch_preprocessing_v1'
and accepts both when loading, so this file does not need regenerating.
"""

import os

import numpy as np

SEED = 314
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")


def main():
    import jax
    import mbirjax
    import mbirjax.preprocess as mjp

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # Scan triple: object/blank/dark with realistic transmission ratios, plus
    # a defective-pixel list and a few non-positive object pixels (-> NaN path).
    num_views, num_rows, num_channels = 13, 24, 20
    blank = (0.8 + 0.2 * rng.rand(2, num_rows, num_channels)).astype(np.float32)
    dark = (0.01 * rng.rand(1, num_rows, num_channels)).astype(np.float32)
    obj = (0.1 + 0.7 * rng.rand(num_views, num_rows, num_channels)).astype(np.float32)
    obj[3, 5, 7] = 0.0                      # non-positive ratio -> NaN -> interpolated
    obj[8, 20, 2] = -0.5
    defective = np.array([[2, 3], [10, 15], [23, 0]])

    sino_trans = np.asarray(mjp.compute_sino_transmission(
        obj.copy(), blank.copy(), dark.copy(), defective_pixel_array=defective.copy(), batch_size=5))

    # interpolate_defective_pixels on a corrupted sinogram.
    corrupt = sino_trans.copy()
    corrupt[1, 4, 4] = np.nan
    corrupt[5, 0, 19] = np.inf
    interp = np.asarray(mjp.interpolate_defective_pixels(
        jax.numpy.array(corrupt), defective_pixel_array=defective.copy(), num_passes=3))

    det_rotation = 0.035
    rotated = np.asarray(mjp.correct_det_rotation(sino_trans.copy(), det_rotation=det_rotation, batch_size=4))

    bg_global = np.asarray(mjp.correct_background_offset(sino_trans.copy(), edge_width=3, option='global'))
    bg_per_view = np.asarray(mjp.correct_background_offset(sino_trans.copy(), edge_width=3, option='per_view'))

    ds_obj, ds_blank, ds_dark, ds_defective = mjp.downsample_view_data(
        obj.copy(), blank.copy(), dark.copy(), (2, 2), defective_pixel_array=defective.copy(), batch_size=5)
    ds_defective = np.asarray(ds_defective) if len(ds_defective) else np.zeros((0, 2), dtype=np.int64)

    cr_obj, cr_blank, cr_dark, cr_defective = mjp.crop_view_data(
        obj.copy(), blank.copy(), dark.copy(), crop_pixels_sides=2, crop_pixels_top=3,
        crop_pixels_bottom=1, defective_pixel_array=defective.copy())
    cr_defective = np.asarray(cr_defective) if len(cr_defective) else np.zeros((0, 2), dtype=np.int64)

    fused = np.asarray(mjp.scan_to_sino(
        obj.copy(), blank.copy(), dark.copy(), defective_pixel_array=defective.copy(),
        downsample_factor=(2, 2), det_rotation=det_rotation, batch_size=5))

    # Zinger correction on a background-corrected sinogram with injected zingers.
    zinger_in = bg_global.copy()
    zinger_in[4, 10, 10] = -30.0
    zinger_in[9, 2, 17] = -45.0
    zinger_out = np.asarray(mjp.correct_zinger_pixels(zinger_in.copy(), zinger_pixel_ratio=0.1,
                                                      num_passes=3, batch_size=5))

    # Blank-margin detection on a sinogram with an embedded object and blank frame.
    margin_sino = np.zeros((11, 40, 36), dtype=np.float32)
    margin_sino[:, 8:30, 6:28] = 2.0 + rng.rand(11, 22, 22).astype(np.float32)
    margins = np.array(mjp.detect_blank_margins(margin_sino.copy(), safety_buffer=4), dtype=np.int64)

    # Crop/geometry helpers (dict/scalar transforms).
    req = {'sinogram_shape': (11, 40, 36)}
    opt = {'det_row_offset': 1.5, 'det_channel_offset': -0.5,
           'delta_det_row': 0.8, 'delta_det_channel': 1.2}
    req2, opt2 = mjp.apply_detector_crop(dict(req), dict(opt), 3, 5, 2, 6)
    config_crop = np.array(mjp.apply_config_crop(
        40, 36, 1.5, -0.5, 0.8, 1.2,
        crop_pixels_top=3, crop_pixels_bottom=5, crop_pixels_sides=2), dtype=np.float64)

    # Cylindrical mask.
    mask_in = rng.rand(20, 18, 12).astype(np.float32)
    mask_out = np.asarray(mjp.apply_cylindrical_mask(mask_in.copy(), radial_margin=2,
                                                     top_margin=1, bottom_margin=2))

    # Beam-hardening curve family (host-side; scipy/numpy).
    p_samples = np.linspace(0.0, 4.0, 600)
    bh_samples = 2.5 * (1.0 - np.exp(-p_samples / 1.7)) + 0.02 * p_samples
    bh_params = mjp.fit_beam_hardening_curve(p_samples, bh_samples, num_parameters=4)
    bh_curve = mjp.apply_beam_hardening_curve(p_samples, bh_params)
    cheb_coeffs, y_domain = mjp.fit_inverse_beam_hardening_curve(
        bh_params, vmin=0.0, vmax=float(bh_curve.max() * 0.99), degree=8)
    bh_inverse = mjp.apply_inverse_beam_hardening_curve(
        np.clip(bh_curve, y_domain[0], y_domain[1]), cheb_coeffs, y_domain, clip=True)

    # Small helpers.
    scale_target = rng.rand(50).astype(np.float32)
    scale_vect = (0.5 * scale_target + 0.05 * rng.rand(50)).astype(np.float32)
    scaling_factor = mjp.compute_scaling_factor(jax.numpy.array(scale_target),
                                                jax.numpy.array(scale_vect))
    pis_in = rng.rand(4, 6, 5).astype(np.float32)
    pis_indices = np.ravel_multi_index(np.array([[1, 2], [5, 0]]).T, (6, 5))
    pis_out = np.asarray(mjp.put_in_slice(jax.numpy.array(pis_in), jax.numpy.array(pis_indices), -7.0))
    alu = mjp.to_alu(25.4, 'mm', 'cm')
    uv = mjp.unit_vector(np.array([3.0, 4.0, 12.0]))
    pv = mjp.project_vector_to_vector(np.array([1.0, 2.0, 2.0]), np.array([0.0, 3.0, 4.0]))

    # Stripe removal: a sinogram with injected full/partial/unresponsive stripes.
    stripe_sino = (1.0 + 0.15 * rng.rand(48, 5, 40)).astype(np.float32)
    ramp = np.abs(np.linspace(-1, 1, 40, dtype=np.float32))
    stripe_sino += 0.5 * (1 - ramp)[None, None, :]          # object-like channel profile
    stripe_sino[:, :, 13] *= 1.6                            # full stripe
    stripe_sino[10:35, :, 27] *= 0.6                        # partial stripe
    stripe_sino[:, :, 33] = 0.05                            # unresponsive column
    stripe_all = np.asarray(mjp.remove_all_stripe(stripe_sino.copy(), snr=3,
                                                  large_filter_size=11, small_filter_size=5))
    stripe_fw = np.asarray(mjp.remove_stripe_fw(jax.numpy.array(stripe_sino)))
    offset_in = stripe_sino + np.linspace(0, 0.8, 48).astype(np.float32)[:, None, None]
    offset_out = np.asarray(mjp.remove_sino_offset(offset_in.copy()))

    # Segmentation: Otsu thresholds (exact-match gate) and plastic/metal masks.
    otsu_img = np.concatenate([0.1 * rng.rand(3000), 0.5 + 0.1 * rng.rand(2000),
                               2.0 + 0.2 * rng.rand(800), 5.0 + 0.3 * rng.rand(300)]).astype(np.float32)
    otsu_3 = np.array(mjp.multi_threshold_otsu(otsu_img, classes=3), dtype=np.float64)
    otsu_4 = np.array(mjp.multi_threshold_otsu(otsu_img, classes=4), dtype=np.float64)
    otsu_mask = otsu_img < 4.0
    otsu_masked = np.array(mjp.multi_threshold_otsu(otsu_img, classes=3, valid_mask=otsu_mask),
                           dtype=np.float64)

    seg_vol = np.zeros((40, 40, 30), dtype=np.float32)
    seg_vol[8:32, 8:32, 5:25] = 0.4 + 0.05 * rng.rand(24, 24, 20).astype(np.float32)
    seg_vol[15:25, 15:25, 10:20] = 3.0 + 0.1 * rng.rand(10, 10, 10).astype(np.float32)
    seg_vol[18:22, 18:22, 13:17] = 7.0 + 0.2 * rng.rand(4, 4, 4).astype(np.float32)
    seg_pm, seg_mm, seg_ps, seg_ms = mjp.segment_plastic_metal(seg_vol.copy(), num_metal=2)

    # MAR: shared cone case.  The direct recon is saved as the SHARED input to the correction so the
    # theta / corrected-sinogram parity does not depend on recon parity.
    mar_cell = (24, 24, 24)
    mar_angles = np.linspace(0, 2 * np.pi, mar_cell[0], endpoint=False)
    mar_model = mbirjax.ConeBeamModel(mar_cell, mar_angles,
                                      source_detector_dist=4 * mar_cell[2],
                                      source_iso_dist=2 * mar_cell[2])
    mar_model.set_params(no_warning=True, verbose=0)
    mar_rshape = tuple(int(v) for v in mar_model.get_params('recon_shape'))
    mar_phantom = np.zeros(mar_rshape, dtype=np.float32)
    mar_phantom[6:18, 6:18, 6:18] = 0.02
    mar_phantom[10:14, 10:14, 10:14] = 0.2
    mar_sino = np.asarray(mar_model.forward_project(mar_phantom))
    mar_weights = np.asarray(mbirjax.gen_weights(mar_sino / mar_sino.max(),
                                                 weight_type='transmission_root'))
    mar_recon_input = np.asarray(mar_model.recon_direct(mar_sino))

    # Huber weights and BH_correction.
    hub_w = (0.5 + rng.rand(6, 8, 10)).astype(np.float32)
    hub_e = rng.randn(6, 8, 10).astype(np.float32)
    hub_out = np.asarray(mjp.gen_huber_weights(hub_w, hub_e, T=1.0, delta=0.7))
    bhc_alpha = [1.0, 0.2, 0.05]
    bhc_out = np.asarray(mjp.BH_correction(mar_sino, bhc_alpha, batch_size=7))

    # gen_weights_mar, both paths.
    gwm_sino_path = np.asarray(mbirjax.gen_weights_mar(mar_model, mar_sino))
    gwm_recon_path = np.asarray(mbirjax.gen_weights_mar(mar_model, mar_sino,
                                                        init_recon=mar_recon_input))

    # Fitted theta via the same internal chain the correction uses, on the shared recon input.
    import mbirjax.preprocess.mar as mjmar
    num_metal, order = 1, 3
    metal_exp = mjmar._generate_metal_exponent_list(num_metal, order)
    cross_exp = mjmar._generate_metal_exponent_list(num_metal, order - 1)
    H_exp = ([(1,) + (0,) * num_metal] + [(1, *t) for t in cross_exp] + [(0, *t) for t in metal_exp])
    p_est, m_est = mjmar._est_plastic_metal_sinos_from_recon(mar_recon_input, num_metal, mar_model)
    p_scale = float(jax.numpy.max(jax.numpy.abs(p_est)))
    m_scales = [float(jax.numpy.max(jax.numpy.abs(m))) for m in m_est]
    p_est_n = p_est / p_scale
    m_est_n = [m / s for m, s in zip(m_est, m_scales)]
    mar_theta = np.asarray(mjmar._estimate_BH_model_params(
        p_est_n, m_est_n, mar_model.prepare_sino_for_devices(mar_sino), H_exp, len(cross_exp),
        alpha=1, beta=0.002))

    # Corrected sinogram from the public entry point on the shared recon input.
    mar_corrected = np.asarray(mjp.correct_sino_plastic_metal(
        mar_model, mar_sino, mar_recon_input, num_metal=1, order=3))

    # One-BH-pass recon (seeded); parity is measured loosely (recon in the loop).
    np.random.seed(11)
    mar_recon_out = np.asarray(mjp.recon_plastic_metal(
        mar_model, mar_sino, mar_weights, num_BH_iterations=1, max_iterations=5,
        num_metal=1, verbose=0, logfile_path=None))

    # recon_split_sino on the shared cone case (seeded; recon in the loop, so parity is loose).
    np.random.seed(19)
    split_recon, split_dict = mar_model.recon_split_sino(mar_sino.copy(), weights=mar_weights.copy(),
                                                         half_overlap=4, max_iterations=5,
                                                         logfile_path=None)
    split_params = split_dict['split_params']

    # Alignment trio on the shared inputs.
    align_shifts = np.asarray(mjp.estimate_sino_view_offset(mar_model, mar_sino, mar_recon_input))
    align_out = np.asarray(mjp.align_sino_views(mar_model, mar_sino, mar_recon_input))

    # median_filter3d with min/max.
    med_in = rng.rand(20, 12, 14).astype(np.float32)
    med_out = np.asarray(mbirjax.median_filter3d(med_in, max_block_gb=0.0001))
    med_m, med_min, med_max = (np.asarray(a) for a in
                               mbirjax.median_filter3d(med_in, max_block_gb=0.0001,
                                                       return_min_max=True))

    # Vendor-loader conversion math on shared synthetic geometry dicts (no scan data needed).
    def _nsi_geom():
        sid, sdd = 100.0, 200.0
        nrows, nchan, dr, dc = 64, 80, 0.2, 0.2
        r_n = np.array([0., 1., 0.]); r_h = np.array([1., 0., 0.]); r_a = np.array([0., 0., -1.])
        r_s = np.array([0., -sid, 0.])
        r_v = np.cross(r_n, r_h)
        r_r = np.array([0., sdd - sid, 0.]) - (nchan / 2.0) * dc * r_h - (nrows / 2.0) * dr * r_v
        return dict(r_a=r_a, r_n=r_n, r_h=r_h, r_s=r_s, r_r=r_r,
                    delta_det_channel=dc, delta_det_row=dr,
                    num_det_channels=nchan, num_det_rows=nrows,
                    angles=np.linspace(0, 2 * np.pi, 20, endpoint=False))

    from mbirjax.preprocess.nsi import convert_nsi_to_mbirjax_params as nsi_conv
    nsi_cb, nsi_op = nsi_conv(_nsi_geom(), (2, 2), 3, 5, 5)
    nsi_convert = np.array([nsi_cb['sinogram_shape'][1], nsi_cb['sinogram_shape'][2],
                            nsi_cb['source_detector_dist'], nsi_cb['source_iso_dist'],
                            nsi_op['det_row_offset'], nsi_op['det_channel_offset'],
                            nsi_op['recon_slice_offset'], nsi_op['delta_det_row'],
                            nsi_op['delta_det_channel'], nsi_op['delta_voxel'],
                            nsi_op['det_rotation']], dtype=np.float64)

    # pymbir beam-hardening linearization on shared inputs.
    from mbirjax.preprocess.pymbir import apply_bh_correction, find_linearization_fit
    bhcn_params = [0.6, 1.0, 4.0, 20.0]
    bhcn_sino = (3.0 * rng.rand(6, 8, 10)).astype(np.float32)
    bhcn_out = np.asarray(apply_bh_correction(bhcn_sino.copy(), bhcn_params))
    bhcn_poly = np.asarray(find_linearization_fit(0.6, 1.0, 4.0, max_thick=20.0))

    # hsnt: seeded simulate -> dehydrate -> rehydrate on shared inputs, plus an mbirjax-written
    # hsnt HDF5 file pinning that format.
    np.random.seed(42)
    hsnt_basis = np.abs(np.stack([np.linspace(1, 2, 40) ** 2, np.sqrt(np.linspace(1, 3, 40)),
                                  np.ones(40) * 1.5])).astype(np.float64)
    hsnt_data, hsnt_angles, hsnt_gt = mbirjax.hsnt.generate_hyper_data(
        hsnt_basis, num_angles=2, detector_rows=24, detector_columns=24, verbose=0)
    # random_state must match the seed in tests/test_hsnt_vcls.py: NMF factorizations are not
    # unique, so an unseeded call here makes the factors an arbitrary local optimum and the parity
    # gate a coin flip (it failed ~35% of runs).  Seeded on both sides the factors agree exactly.
    hsnt_dehydrated = mbirjax.dehydrate(hsnt_data.copy(), num_materials=3, random_state=52, verbose=0)
    hsnt_rehydrated = mbirjax.rehydrate(hsnt_dehydrated)
    hsnt_md = mbirjax.hsnt.create_hsnt_metadata(dataset_name='golden', dataset_type='attenuation',
                                                angles=np.rad2deg(hsnt_angles))
    mbirjax.export_hsnt_data_hdf5(os.path.join(OUT_DIR, 'preprocess_goldens_hsnt.h5'),
                                  hsnt_dehydrated, hsnt_md)

    # vcls: seeded get_opt_views on a small parallel model.
    vcls_angles = np.linspace(0, np.pi, 24, endpoint=False)
    vcls_model = mbirjax.ParallelBeamModel((24, 8, 32), vcls_angles)
    vcls_model.set_params(no_warning=True, verbose=0)
    vcls_rshape = tuple(int(v) for v in vcls_model.get_params('recon_shape'))
    vcls_ref = np.zeros(vcls_rshape, dtype=np.float32)
    vcls_ref[10:22, 10:22, 2:6] = 1.0
    vcls_inds, vcls_value = mbirjax.get_opt_views(vcls_model, vcls_ref, num_selected_views=5,
                                                  r_1=0.05, seed=3)

    # Demo-data utilities: demo data for the parallel, cone, and helical-cone
    # paths, and the recon shapes get_ct_model produces.  (The reference-phantom
    # parity entry was dropped 2026-08-14: mbirjax's phantom transposes rows and
    # columns, and mbirtorch corrected that.)

    demo_cases = {}
    for tag, kwargs in [('par', dict(model_type='parallel')),
                        ('cone', dict(model_type='cone')),
                        ('hel', dict(model_type='cone', use_helical=True,
                                     helical_pitch=0.5, helical_z_range=16.0))]:
        ph, sino, dd_params = mbirjax.generate_demo_data(
            num_views=12, num_det_rows=24, num_det_channels=32, **kwargs)
        demo_cases['demo_' + tag + '_phantom'] = np.asarray(ph, dtype=np.float32)
        demo_cases['demo_' + tag + '_sino'] = np.asarray(sino, dtype=np.float32)
        demo_cases['demo_' + tag + '_angles'] = np.asarray(dd_params['angles'], dtype=np.float64)
        if tag == 'hel':
            demo_cases['demo_hel_z_shifts'] = np.asarray(dd_params['helical_z_shifts'],
                                                         dtype=np.float64)

    gcm_par = mbirjax.get_ct_model('parallel', (8, 10, 12),
                                   np.linspace(0, np.pi, 8, endpoint=False))
    gcm_cone = mbirjax.get_ct_model('cone', (8, 10, 12),
                                    np.linspace(0, 2 * np.pi, 8, endpoint=False),
                                    source_detector_dist=100.0, source_iso_dist=50.0)
    gcm_shapes = np.array([gcm_par.get_params('recon_shape'),
                           gcm_cone.get_params('recon_shape')], dtype=np.int64)

    # HDF5 family: mbirjax-written files pin the on-disk formats.
    h5_vol = rng.rand(10, 12, 14).astype(np.float32)
    h5_attrs = {'scan_id': 'sample1', 'notes': 'golden', 'nested': {'a': 1, 'b': 'two'}}
    mbirjax.save_data_hdf5(os.path.join(OUT_DIR, 'preprocess_goldens_data.h5'),
                           h5_vol, array_name='volume', attributes_dict=h5_attrs)
    mbirjax.export_recon_hdf5(os.path.join(OUT_DIR, 'preprocess_goldens_export.h5'),
                              h5_vol, recon_dict={'scan_id': 'sample1'})
    mbirjax.export_recon_hdf5(os.path.join(OUT_DIR, 'preprocess_goldens_export_flash.h5'),
                              h5_vol, recon_dict={'scan_id': 'sample1'}, remove_flash=True,
                              radial_margin=2, top_margin=2, bottom_margin=3)

    # mbirjax-written cone-preprocessing save: pins the on-disk layout, and carries the older
    # format tag that the mbirtorch loader still accepts.
    h5_path = os.path.join(OUT_DIR, 'preprocess_goldens_cone_save.h5')
    mjp.save_cone_preprocessing(
        h5_path, sino_trans,
        {'sinogram_shape': tuple(sino_trans.shape), 'angles': np.linspace(0, np.pi, num_views).astype(np.float32)},
        {'sharpness': 1.25, 'det_channel_offset': 0.75},
        weights=np.asarray(mbirjax.gen_weights(sino_trans, weight_type='transmission_root')))

    out = os.path.join(OUT_DIR, 'preprocess_goldens.npz')
    np.savez_compressed(
        out,
        jax_version=jax.__version__,
        obj=obj, blank=blank, dark=dark, defective=defective,
        sino_trans=sino_trans.astype(np.float32),
        corrupt=corrupt.astype(np.float32), interp=interp.astype(np.float32),
        det_rotation=np.float64(det_rotation), rotated=rotated.astype(np.float32),
        bg_global=bg_global.astype(np.float32), bg_per_view=bg_per_view.astype(np.float32),
        ds_obj=np.asarray(ds_obj, dtype=np.float32), ds_blank=np.asarray(ds_blank, dtype=np.float32),
        ds_dark=np.asarray(ds_dark, dtype=np.float32), ds_defective=ds_defective,
        cr_obj=np.asarray(cr_obj, dtype=np.float32), cr_blank=np.asarray(cr_blank, dtype=np.float32),
        cr_dark=np.asarray(cr_dark, dtype=np.float32), cr_defective=cr_defective,
        fused=fused.astype(np.float32),
        zinger_in=zinger_in.astype(np.float32), zinger_out=zinger_out.astype(np.float32),
        margin_sino=margin_sino, margins=margins,
        crop_req_shape=np.array(req2['sinogram_shape']),
        crop_opt=np.array([opt2['det_row_offset'], opt2['det_channel_offset']]),
        config_crop=config_crop,
        mask_in=mask_in, mask_out=mask_out.astype(np.float32),
        p_samples=p_samples, bh_samples=bh_samples, bh_params=bh_params,
        bh_curve=bh_curve, cheb_coeffs=cheb_coeffs, y_domain=np.array(y_domain),
        bh_inverse=bh_inverse,
        scale_target=scale_target, scale_vect=scale_vect,
        scaling_factor=np.float64(scaling_factor),
        pis_in=pis_in, pis_indices=pis_indices, pis_out=pis_out,
        alu=np.float64(alu), uv=uv, pv=pv,
        stripe_sino=stripe_sino, stripe_all=stripe_all.astype(np.float32),
        stripe_fw=stripe_fw.astype(np.float32),
        offset_in=offset_in.astype(np.float32), offset_out=offset_out.astype(np.float32),
        otsu_img=otsu_img, otsu_3=otsu_3, otsu_4=otsu_4,
        otsu_mask=otsu_mask, otsu_masked=otsu_masked,
        seg_vol=seg_vol, seg_pm=np.asarray(seg_pm, dtype=np.float32),
        seg_mm=np.stack([np.asarray(m, dtype=np.float32) for m in seg_mm]),
        seg_ps=np.float64(seg_ps), seg_ms=np.array(seg_ms, dtype=np.float64),
        mar_cell=np.array(mar_cell), mar_angles=mar_angles,
        mar_sdd=np.float64(4 * mar_cell[2]), mar_sid=np.float64(2 * mar_cell[2]),
        mar_phantom=mar_phantom, mar_sino=mar_sino.astype(np.float32),
        mar_weights=mar_weights.astype(np.float32),
        mar_recon_input=mar_recon_input.astype(np.float32),
        mar_theta=mar_theta.astype(np.float64),
        mar_corrected=mar_corrected.astype(np.float32),
        mar_recon_out=mar_recon_out.astype(np.float32),
        hub_w=hub_w, hub_e=hub_e, hub_out=hub_out.astype(np.float32),
        bhc_alpha=np.array(bhc_alpha), bhc_out=bhc_out.astype(np.float32),
        gwm_sino_path=gwm_sino_path.astype(np.float32),
        gwm_recon_path=gwm_recon_path.astype(np.float32),
        align_shifts=align_shifts.astype(np.float64),
        align_out=align_out.astype(np.float32),
        med_in=med_in, med_out=med_out.astype(np.float32),
        med_min=med_min.astype(np.float32), med_max=med_max.astype(np.float32),
        h5_vol=h5_vol,
        nsi_convert=nsi_convert,
        bhcn_sino=bhcn_sino, bhcn_out=bhcn_out.astype(np.float64), bhcn_poly=bhcn_poly,
        hsnt_basis=hsnt_basis, hsnt_data=hsnt_data.astype(np.float64),
        hsnt_gt=hsnt_gt.astype(np.float64),
        hsnt_sub_data=np.asarray(hsnt_dehydrated[0]), hsnt_sub_basis=np.asarray(hsnt_dehydrated[1]),
        hsnt_rehydrated=hsnt_rehydrated.astype(np.float64),
        vcls_ref=vcls_ref, vcls_angles=vcls_angles,
        vcls_inds=np.asarray(vcls_inds, dtype=np.int64), vcls_value=np.float64(vcls_value),
        split_recon=np.asarray(split_recon, dtype=np.float32),
        split_overlap_sino=np.int64(split_params['half_overlap_sino']),
        split_overlap_recon=np.int64(split_params['half_overlap_recon']),
        gcm_shapes=gcm_shapes,
        **demo_cases,
    )
    print('wrote', out)
    print('wrote', h5_path)


if __name__ == '__main__':
    main()
