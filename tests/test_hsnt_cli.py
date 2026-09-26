"""Tests for the mbirtorch-hsnt command line and the library entry points: loading and conversion of synthetic TIFF
and HDF5 inputs, and dehydrate / rehydrate / denoise runs whose outputs read back through the package's HDF5
importer."""
import json
import os

import h5py
import numpy as np
import pytest
import tifffile
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt.cli import main
from mbirtorch.hsnt.loading import _smoothing_kernel, _tif_names, infer_input_type, load_dataset
from mbirtorch.hsnt.outputs import component_check, fit_quality, mean_pixel_spectrum

ROWS, COLS, K, R, DOSE = 12, 10, 40, 2, 50.0


@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    """One intra-op thread, so parallel test workers do not each start one per core."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 0.8, size=(R, K))
    H[:, K // 2:] *= 0.5
    W = np.zeros((ROWS * COLS, R))
    m = rng.integers(0, R, ROWS * COLS)
    W[np.arange(ROWS * COLS), m] = rng.uniform(0.3, 1.5, ROWS * COLS)
    W[: ROWS * COLS // 5] = 0
    return W, H


def _report(out, stem):
    with open(os.path.join(out, f"{stem}_report.json")) as f:
        return json.load(f)


def _relative_error(estimate, X):
    return np.linalg.norm(estimate.reshape(-1, K) - X) / np.linalg.norm(X)


def _write_transmission(path, T):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=T)
        f.create_dataset("dataset_type", data=np.bytes_("transmission"))


@pytest.fixture(scope="module")
def stacks(tmp_path_factory):
    """A TIFF stack of counts, two open-beam observations, and the same data as an HDF5 attenuation file."""
    root = tmp_path_factory.mktemp("hsnt_cli")
    W, H = _truth()
    rng = np.random.default_rng(1)
    X = (W @ H).reshape(ROWS, COLS, K)
    ob_true = np.full((ROWS, COLS, K), DOSE)
    counts = rng.poisson(DOSE * np.exp(-X)).astype(np.float32)
    sample = root / "sample"
    sample.mkdir()
    for k in range(K):
        tifffile.imwrite(sample / f"wave_idx_{k:05d}.tif", counts[:, :, k])
    ob_root = root / "open_beam"
    for o in range(2):
        d = ob_root / f"observation_{o + 1:02d}"
        d.mkdir(parents=True)
        obs = rng.poisson(ob_true).astype(np.float32)
        for k in range(K):
            tifffile.imwrite(d / f"wave_idx_{k:05d}.tif", obs[:, :, k])
    A = -np.log(np.maximum(counts / DOSE, 1e-6)).astype(np.float32)
    h5 = root / "processed.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("sample_dataset")
        g.create_dataset("data", data=A[None])                      # (views, rows, cols, bins)
        g.create_dataset("dataset_type", data=np.bytes_("attenuation"))
    return dict(root=root, sample=str(sample), open_beam=str(ob_root), h5=str(h5), counts=counts)


def test_input_types_are_inferred_and_tiffs_read_in_natural_order(stacks, tmp_path):
    rng = np.random.default_rng(0)
    assert infer_input_type(rng.uniform(0, 1, 1000))[0] == "transmission"
    assert infer_input_type(rng.poisson(1.0, 1000).astype(np.float32))[0] == "counts"      # low-dose counts
    assert infer_input_type(rng.uniform(0, 1, 1000), source_dtype="uint16")[0] == "counts"
    assert infer_input_type(rng.normal(0.5, 0.4, 1000))[0] == "attenuation"
    with pytest.raises(ValueError, match="--input-type"):
        infer_input_type(rng.poisson(1.0, 1000) + rng.uniform(0, 1e-3, 1000))           # noisy transmission or not
    with pytest.raises(SystemExit, match="open-beam"):
        main(["inspect", stacks["sample"], "-q"])                        # counts without an open beam
    for name in ("img_10.TIF", "img_2.tif", "img_1.tiff"):
        (tmp_path / name).write_bytes(b"")
    assert [os.path.basename(f) for f in _tif_names(str(tmp_path))] == ["img_1.tiff", "img_2.tif", "img_10.TIF"]


def test_conversion_matches_the_direct_load(stacks, tmp_path, capsys):
    """TIFF counts with an open beam and HDF5 attenuation, converted in blocks below one --wave-bin group (one group
    per block), give what load_dataset gives; the dose, source bins and open-beam observations are recorded."""
    assert main(["inspect", stacks["sample"], "--open-beam", stacks["open_beam"], "-q"]) == 0
    out = capsys.readouterr().out
    assert "type counts" in out and f"x {K} bins" in out and "unknown" not in out.split("dose:")[1].split("\n")[0]
    conv = str(tmp_path / "blocks.h5")
    assert main(["convert", stacks["sample"], "--open-beam", stacks["open_beam"], "-o", conv, "--as-type",
                 "transmission", "--wave-bin", "4", "--downsample", "2", "--memory-budget", "0.01", "-q"]) == 0
    data, meta = hsnt.import_hsnt_data_hdf5(conv)
    ds = load_dataset(stacks["sample"], open_beam=[stacks["open_beam"]], wave_bin=4, downsample=2)
    assert ds.T.shape == ((ROWS + 1) // 2 * ((COLS + 1) // 2), K // 4) and ds.open_beam_observations == 2
    assert meta["dataset_type"] == "transmission" and np.allclose(data.reshape(-1, K // 4), ds.T, atol=1e-6)
    assert not any("--background-boxes" in c.message for c in ds.checks)          # matched exposures raise no alarm
    with h5py.File(conv) as f:
        assert f.attrs["block_bins"] == 4 and f["bin_indices"][()].tolist() == list(range(0, K, 4))
        assert f.attrs["open_beam_observations"] == 2 and abs(f.attrs["dose"] - ds.dose) / ds.dose < 0.05
        recorded = float(f.attrs["dose"])
    back = load_dataset(conv)                                            # read back from the converted file
    assert back.dose == pytest.approx(recorded) and back.open_beam_observations == 2
    assert back.bin_indices.tolist() == list(range(0, K, 4)) and back.dose_per_bin.shape == (K // 4,)
    conv = str(tmp_path / "h5h5.h5")                                     # attenuation in and out
    assert main(["convert", stacks["h5"], "-o", conv, "--wave-bin", "2", "--downsample", "2", "--memory-budget",
                 "0.01", "-q"]) == 0
    data, meta = hsnt.import_hsnt_data_hdf5(conv)
    with np.errstate(divide="ignore"):
        ref = -np.log(load_dataset(stacks["h5"], wave_bin=2, downsample=2).T)
    assert meta["dataset_type"] == "attenuation" and np.allclose(data.reshape(-1, K // 2), ref, atol=1e-5)


def test_a_converted_file_keeps_its_dose_source_bins_and_metadata(tmp_path):
    """The dose (per source bin, scaled by --wave-bin), the source bin of every column, the angles of the selected
    views and the wavelengths of the selected and grouped bins survive convert, dehydrate and rehydrate, and
    --wave-range counts source bins throughout."""
    W, H = _truth()
    src = str(tmp_path / "meta.h5")
    hsnt.export_hsnt_data_hdf5(src, np.stack([(W @ H).reshape(ROWS, COLS, K)] * 3).astype(np.float32),
                               hsnt.create_hsnt_metadata(dataset_type="attenuation", wavelengths=np.linspace(1, 4.9, K),
                                                         angles=np.array([0.0, 60.0, 120.0]), alu_unit="mm",
                                                         delta_det_row=0.1))
    conv = str(tmp_path / "conv.h5")
    assert main(["convert", src, "-o", conv, "--views", "1:3", "--wave-range", "4:40", "--wave-bin", "2",
                 "--downsample", "2", "--dose", str(DOSE), "-q"]) == 0
    lam = np.linspace(1, 4.9, K)[4:40].reshape(-1, 2).mean(1)
    _, meta = hsnt.import_hsnt_data_hdf5(conv)
    assert np.allclose(meta["angles"], [60.0, 120.0]) and np.allclose(meta["wavelengths"], lam)
    assert meta["alu_unit"] == "mm" and meta["delta_det_row"] == pytest.approx(0.2)
    ds = load_dataset(conv)
    assert ds.dose == pytest.approx(2 * DOSE) and ds.bin_indices.tolist() == list(range(4, 40, 2))
    assert load_dataset(conv, wave_bin=3).dose == pytest.approx(6 * DOSE)
    assert load_dataset(conv, dose=DOSE).dose == pytest.approx(2 * DOSE)            # a given dose is per source bin
    out = str(tmp_path / "res")                                          # support selection finds the dose itself
    assert main(["dehydrate", conv, "-o", out, "--rank", str(R), "--spectra", "support", "--max-steps", "50",
                 "--no-plots", "-q"]) == 0
    deh = os.path.join(out, "conv_dehydrated.h5")
    assert np.allclose(hsnt.import_hsnt_data_hdf5(deh)[1]["wavelengths"], lam)
    part = str(tmp_path / "part.h5")
    assert main(["rehydrate", deh, "-o", part, "--wave-range", "10:20", "-q"]) == 0
    data, meta = hsnt.import_hsnt_data_hdf5(part)
    assert data.shape[-1] == 5 and np.allclose(meta["wavelengths"], lam[3:8])
    with h5py.File(part) as f:
        assert f["bin_indices"][()].tolist() == [10, 12, 14, 16, 18]


def test_background_boxes_calibrate_each_tile_and_the_open_beam_can_be_smoothed(tmp_path):
    """A sample run whose exposure differs from the open beam's by tile reads T = 1 in every sample-free corner after
    the calibration, and the dose becomes the sample's; transmission and attenuation files of several views, binned
    and downsampled, are calibrated per view to their true transmission; the smoothed open beam is the Hamming-window
    filter of the earlier preprocessing and counts as the observations over its variance reduction."""
    from scipy.signal import convolve2d
    rng = np.random.default_rng(3)
    n, k, dose, exposure = 16, 12, 200.0, np.array([[1.15, 0.9], [1.1, 1.2]])
    X = np.zeros((n, n, k))
    X[4:12, 4:12] = rng.uniform(0.2, 1.0, (8, 8, k))                     # the sample, clear of the corners
    f = np.kron(exposure, np.ones((n // 2, n // 2)))[:, :, None]
    stacks = {"sample": rng.poisson(dose * f * np.exp(-X)), "ob/observation_01": rng.poisson(np.full(X.shape, dose)),
              "ob/observation_02": rng.poisson(np.full(X.shape, dose))}
    for name, counts in stacks.items():
        (tmp_path / name).mkdir(parents=True)
        for j, img in enumerate(np.moveaxis(counts.astype(np.float32), -1, 0)):
            tifffile.imwrite(tmp_path / name / f"wave_idx_{j:05d}.tif", img)
    sample, ob = str(tmp_path / "sample"), [str(tmp_path / "ob")]
    boxes = [(0, 3, 0, 3), (0, 3, 13, 16), (13, 16, 0, 3), (13, 16, 13, 16)]
    corners = [(slice(0, 3), slice(0, 3)), (slice(0, 3), slice(13, 16)), (slice(13, 16), slice(0, 3)),
               (slice(13, 16), slice(13, 16))]
    plain = load_dataset(sample, open_beam=ob)
    ds = load_dataset(sample, open_beam=ob, background_boxes=boxes, background_tiles=(2, 2))
    for c, e in zip(corners, exposure.ravel()):
        assert abs(ds.T.reshape(n, n, k)[c].mean() - 1) < 0.02 and abs(plain.T.reshape(n, n, k)[c].mean() - e) < 0.04
    assert any(c.level == "warn" and "--background-boxes" in c.message for c in plain.checks)
    assert not any("--background-boxes" in c.message for c in ds.checks)
    factor = ds.info["background"]["factor_median"]
    assert ds.dose == pytest.approx(plain.dose * factor) and factor == pytest.approx(exposure.mean(), abs=0.02)
    per_bin, (lo, hi) = ds.dose_per_bin / plain.dose_per_bin, ds.info["background"]["factor_range"]
    assert abs(per_bin.mean() - exposure.mean()) < 0.02 and np.all((per_bin >= lo) & (per_bin <= hi))
    assert ds.open_beam_observations == pytest.approx(2 / factor)       # dose x observations: the open beam's count
    views = np.stack([np.exp(-X), np.exp(-X[::-1])]) * np.stack([f, f[::-1, ::-1] * 0.9])[..., None][..., 0]
    truth = np.stack([np.exp(-X), np.exp(-X[::-1])])[:, ::2, ::2].reshape(2, n // 2, n // 2, k // 2, 2).mean(-1)
    for kind, data in (("transmission", views), ("attenuation", -np.log(views))):
        path = str(tmp_path / f"{kind}.h5")
        hsnt.export_hsnt_data_hdf5(path, data.astype(np.float32), hsnt.create_hsnt_metadata(dataset_type=kind))
        cal = load_dataset(path, background_boxes=boxes, background_tiles=(2, 2), wave_bin=2, downsample=2)
        assert np.allclose(cal.T, truth.reshape(-1, k // 2), atol=1e-5)
    smooth = load_dataset(sample, open_beam=ob, open_beam_smoothing=3)
    h = np.sqrt(np.outer(np.hamming(3), np.hamming(3)))
    mean_ob = (stacks["ob/observation_01"] + stacks["ob/observation_02"]) / 2
    ref = np.stack([convolve2d(np.pad(mean_ob[:, :, j], 1, mode="reflect"), h / h.sum(), mode="valid")
                    for j in range(k)], -1)
    assert np.allclose(smooth.T, (stacks["sample"] / ref).reshape(-1, k), rtol=1e-5)
    assert smooth.open_beam_observations == pytest.approx(2 / (_smoothing_kernel(3) ** 2).sum(), rel=0.2)
    conv = str(tmp_path / "cal.h5")
    assert main(["convert", sample, "--open-beam", *ob, "-o", conv, "--as-type", "transmission", "--background-boxes",
                 *[f"{y0}:{y1},{x0}:{x1}" for y0, y1, x0, x1 in boxes], "--background-tiles", "2x2",
                 "--open-beam-smoothing", "3", "--memory-budget", "0.01", "-q"]) == 0
    both = load_dataset(sample, open_beam=ob, background_boxes=boxes, background_tiles=(2, 2), open_beam_smoothing=3)
    back = load_dataset(conv)
    assert np.allclose(back.T, both.T, atol=1e-6) and back.dose == pytest.approx(both.dose, rel=0.01)
    assert back.open_beam_observations == pytest.approx(both.open_beam_observations, rel=0.2)
    assert back.info["background"]["recorded"] and not any("--background-boxes" in c.message for c in back.checks)
    for bad, match in ((["--background-boxes", "ornl-snap"], "512 x 512"),
                       (["--background-boxes", "0:3,0:3", "--background-tiles", "2x2"], "hold no box"),
                       (["--background-tiles", "2x2"], "need background"), (["--open-beam-smoothing", "4"], "odd")):
        with pytest.raises(SystemExit, match=match):
            main(["inspect", sample, "--open-beam", *ob, "-q"] + bad)


def test_dehydrate_rehydrate_and_denoise_write_readable_outputs(stacks, tmp_path, capsys):
    out = str(tmp_path / "res")
    assert main(["dehydrate", stacks["h5"], "-o", out, "--dry-run", "-q"]) == 0          # a dry run writes nothing
    assert "dry run" in capsys.readouterr().out and not os.listdir(out)
    run = ["dehydrate", stacks["h5"], "-o", out, "--dose", str(DOSE), "--max-steps", "200", "-q"]
    assert main(run) == 0
    assert all(os.path.exists(os.path.join(out, "processed" + s)) for s in ("_dehydrated.h5", "_report.json",
                                                                              "_maps.png", "_spectra.png"))
    deh = os.path.join(out, "processed_dehydrated.h5")
    (W4, H4, dtype), _ = hsnt.import_hsnt_data_hdf5(deh)
    assert W4.shape == (1, ROWS, COLS, R) and H4.shape == (R, K) and dtype == "attenuation" and W4.min() >= 0
    W, H = _truth()
    assert _relative_error(hsnt.rehydrate([W4, H4, dtype]), W @ H) < 0.35               # near the true attenuation
    rep = _report(out, "processed")["result"]
    assert rep["rank"] == R and "estimated" in rep["rank_note"] and rep["components"]["proportional_pairs"] == []
    assert 0.5 < rep["fit"]["reduced_chi2"] < 2.0                        # the fit sits at the Poisson noise level
    with pytest.raises(SystemExit, match="--overwrite"):
        main(run)
    assert main(["rehydrate", deh, "-o", out, "-q"]) == 0                # all bins, the file's type
    data, meta = hsnt.import_hsnt_data_hdf5(os.path.join(out, "processed_rehydrated.h5"))
    assert meta["dataset_type"] == "attenuation" and np.allclose(data, W4 @ H4, atol=1e-5)
    sub = os.path.join(out, "part.h5")                                   # a range, as transmission
    assert main(["rehydrate", deh, "-o", sub, "--wave-range", "5:15", "--as-type", "transmission", "-q"]) == 0
    part, meta = hsnt.import_hsnt_data_hdf5(sub)
    assert meta["dataset_type"] == "transmission" and np.allclose(part, np.exp(-(W4 @ H4[:, 5:15])), atol=1e-5)
    fixed = str(tmp_path / "fixed")                                      # maps for the spectra of a fitted file
    assert main(["dehydrate", stacks["h5"], "-o", fixed, "--basis", deh, "--dose", str(DOSE), "--no-plots", "-q"]) == 0
    (W5, H5, _), _ = hsnt.import_hsnt_data_hdf5(os.path.join(fixed, "processed_dehydrated.h5"))
    assert np.array_equal(H5, H4) and np.abs(W5 - W4).max() < 1e-3 * W4.max()
    assert "from the basis" in _report(fixed, "processed")["result"]["rank_note"]
    with pytest.raises(SystemExit, match="source bins"):
        main(["dehydrate", stacks["h5"], "-o", fixed, "--basis", deh, "--wave-range", "0:20", "--overwrite", "-q"])
    with pytest.raises(SystemExit, match="is an input"):                 # the basis is never replaced
        main(["dehydrate", stacks["h5"], "-o", out, "--basis", deh, "--overwrite", "-q"])
    with pytest.raises(SystemExit, match="is an input"):
        main(["rehydrate", deh, "-o", deh, "--overwrite", "-q"])
    with pytest.raises(SystemExit, match="not a dehydrated file"):
        main(["rehydrate", stacks["h5"], "-o", out, "-q"])
    with h5py.File(deh) as f:
        assert "subspace_basis" in f                                     # the dehydrated file is intact
    den = str(tmp_path / "den")
    assert main(["denoise", stacks["sample"], "--open-beam", stacks["open_beam"], "-o", den, "--rank", str(R),
                 "--no-dehydrated", "--as-type", "transmission", "--no-plots", "--max-steps", "100", "-q"]) == 0
    assert sorted(os.listdir(den)) == ["sample_denoised.h5", "sample_report.json"]
    data, meta = hsnt.import_hsnt_data_hdf5(os.path.join(den, "sample_denoised.h5"))
    assert meta["dataset_type"] == "transmission" and 0 < data.min() and data.max() < 2
    assert _relative_error(-np.log(data), W @ H) < 0.35


def test_stream_mode_and_the_spectra_estimators_on_the_command_line(stacks, tmp_path):
    out = str(tmp_path / "stream")
    assert main(["dehydrate", stacks["sample"], "--open-beam", stacks["open_beam"], "--rank", str(R), "-o", out,
                 "--mode", "stream", "--chunk-pixels", "40", "--max-passes", "2", "--spectra", "support",
                 "--support-penalty", "1", "--free-refit", "--no-plots", "-q"]) == 0
    rep = _report(out, "sample")["result"]
    assert rep["mode"] == "stream" and rep["passes"] >= 1 and 0 < rep["mean_support_size"] <= R
    assert rep["loss_final"] >= rep["loss_mle"] * (1 - 1e-6)            # the constrained refit cannot beat the MLE
    base = ["dehydrate", stacks["h5"], "--rank", str(R), "--max-steps", "150", "--no-plots", "-q"]
    with pytest.raises(SystemExit, match="dose"):
        main(base + ["-o", str(tmp_path / "none"), "--spectra", "support"])
    out = str(tmp_path / "unc")
    assert main(base + ["-o", out, "--spectra", "unconstrained"]) == 0
    rep = _report(out, "processed")["result"]
    assert rep["unconstrained_steps"] > 0 and rep["W_zero_frac"] < 1
    out = str(tmp_path / "sup")
    assert main(base + ["-o", out, "--dose", str(DOSE), "--spectra", "support", "--wald-screen", "0.5"]) == 0
    rep = _report(out, "processed")["result"]
    assert 0 < rep["mean_support_size"] <= R and rep["loss_final"] >= rep["loss_mle"] * (1 - 1e-9)


def test_library_dehydrate_and_hyper_denoise(stacks):
    with h5py.File(stacks["h5"]) as f:
        A = f["sample_dataset/data"][()]                                 # (1, ROWS, COLS, K) attenuation
    sub_data, basis, dtype = hsnt.dehydrate(A, "attenuation", num_materials=R, verbose=0)
    assert sub_data.shape == (1, ROWS, COLS, R) and basis.shape == (R, K) and dtype == "attenuation"
    assert sub_data.min() >= 0 and basis.min() >= 0 and sub_data.dtype == np.float32
    W, H = _truth()
    X = W @ H
    den = hsnt.hyper_denoise(A, "attenuation", num_materials=R, verbose=0)
    assert den.shape == A.shape and _relative_error(den, X) < 0.35
    est = hsnt.dehydrate(A[0], "attenuation", verbose=0)                 # rank estimated, pooling over (rows, cols)
    assert est[0].shape == (ROWS, COLS, R)
    den_t = hsnt.hyper_denoise(np.exp(-A), "transmission", num_materials=R, verbose=0)    # transmission in and out
    assert den_t.shape == A.shape and 0 < den_t.min() and _relative_error(-np.log(den_t), X) < 0.35
    sup = hsnt.dehydrate(A, num_materials=R, spectra="support", dose=DOSE, verbose=0)
    assert sup[0].shape == sub_data.shape and sup[0].min() >= 0
    with pytest.raises(ValueError, match="dose"):
        hsnt.dehydrate(A, num_materials=R, spectra="support", verbose=0)
    maps, same, _ = hsnt.dehydrate(A, subspace_basis=basis, verbose=0)  # the maps of a given basis
    assert np.array_equal(same, basis) and np.abs(maps - sub_data).max() < 1e-3 * sub_data.max()
    for bad in (dict(num_materials=R + 1), dict(spectra="support", dose=DOSE)):
        with pytest.raises(ValueError, match="subspace_basis"):
            hsnt.dehydrate(A, subspace_basis=basis, verbose=0, **bad)
    with pytest.raises(TypeError, match="MBIRJAX"):
        hsnt.hyper_denoise(A, num_materials=R, safety_factor=2, verbose=0)          # the NMF's keywords are refused


def test_bad_values_are_refused_and_a_failed_run_leaves_no_output(stacks, tmp_path):
    for bad in (["--wave-bin", "0"], ["--dose", "-1"], ["--max-rank", "0"], ["--rank", "0"], ["--rel-tol", "x"]):
        with pytest.raises(SystemExit):
            main(["dehydrate", stacks["h5"], "-o", str(tmp_path), "-q"] + bad)      # argparse refuses them
    W, H = _truth()
    T = np.exp(-W @ H).reshape(1, ROWS, COLS, K).astype(np.float32)
    T[:, :, :, 3] = -0.5                                                 # a transmission cannot be negative
    bad = str(tmp_path / "negative.h5")
    _write_transmission(bad, T)
    assert main(["inspect", bad, "-q"]) == 0                             # reported, not fatal
    with pytest.raises(SystemExit, match="check"):
        main(["inspect", bad, "--strict", "-q"])
    with pytest.raises(SystemExit, match="check"):                       # the check fails at the end of the conversion
        main(["convert", bad, "-o", str(tmp_path / "strict.h5"), "--strict", "-q"])
    assert not [f for f in os.listdir(tmp_path) if f.startswith("strict")]         # neither the file nor a .partial


def test_fit_diagnostics():
    """The chi-square sits at 1 with the open beam's count in each bin and the noise of its two observations; the
    component check flags proportional spectra, not proportional maps (which a mixed basis has); the mean-pixel
    spectrum is the sum of the components' shares."""
    rng = np.random.default_rng(1)
    W, H = _truth()
    flux = np.linspace(20.0, 80.0, K)                                    # an open beam that varies over the bins
    ob = rng.poisson(np.tile(flux, (W.shape[0], 1)), size=(2, W.shape[0], K)).mean(0)
    T = (rng.poisson(flux * np.exp(-W @ H)) / np.maximum(ob, 1)).astype(np.float32)
    exact = fit_quality(T, W, H, dose=50.0, dose_per_bin=flux, open_beam_observations=2)["reduced_chi2"]
    assert abs(exact - 1) < 0.05 and fit_quality(T, W, H, dose=50.0)["reduced_chi2"] > 1.2
    assert component_check(W, H)["proportional_pairs"] == []             # distinct maps
    W2 = np.stack([W[:, 0], W[:, 0] * (1 + 0.05 * rng.standard_normal(W.shape[0]))], 1)    # a split material
    H2 = np.stack([H[0] * 0.5, H[0] * 0.5])
    assert component_check(W2, H2)["proportional_pairs"][0][:2] == (0, 1)
    assert component_check(W2, H)["proportional_pairs"] == []            # proportional maps, distinct spectra: a mix
    total, contrib, n = mean_pixel_spectrum(W2, H2)
    assert total.shape == (K,) and n > 0 and np.allclose(total, contrib.sum(0))
