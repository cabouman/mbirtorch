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
curve family, and the small helpers.  The HDF5 file is an mbirjax-written
save_cone_preprocessing output, pinning the on-disk format as shared between
the two packages.
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

    # mbirjax-written cone-preprocessing save: pins the on-disk format.
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
    )
    print('wrote', out)
    print('wrote', h5_path)


if __name__ == '__main__':
    main()
