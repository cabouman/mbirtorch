"""Tests for the mbirtorch-hsnt command line: loaders and checks on synthetic TIFF and HDF5 inputs, the
convert round trip, and dehydrate / rehydrate / denoise runs whose outputs read back through the package's HDF5 importer."""
import json
import os

import numpy as np
import pytest

hsnt = pytest.importorskip("mbirtorch.hsnt")
tifffile = pytest.importorskip("tifffile")
import h5py
import torch

from mbirtorch.hsnt.cli import main, infer_input_type, _parse_slice, estimate_rank, load_hdf5

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ROWS, COLS, K, R, DOSE = 12, 10, 40, 2, 50.0


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 0.8, size=(R, K)); H[:, K // 2:] *= 0.5
    W = np.zeros((ROWS * COLS, R)); m = rng.integers(0, R, ROWS * COLS); W[np.arange(ROWS * COLS), m] = rng.uniform(0.3, 1.5, ROWS * COLS)
    W[: ROWS * COLS // 5] = 0
    return W, H


@pytest.fixture(scope="module")
def stacks(tmp_path_factory):
    """A TIFF stack of counts, two open-beam observations, and the same data as an HDF5 attenuation file."""
    root = tmp_path_factory.mktemp("hsnt_cli")
    W, H = _truth(); rng = np.random.default_rng(1)
    X = (W @ H).reshape(ROWS, COLS, K)
    ob_true = np.full((ROWS, COLS, K), DOSE)
    counts = rng.poisson(DOSE * np.exp(-X)).astype(np.float32)
    sample = root / "sample"; sample.mkdir()
    for k in range(K):
        tifffile.imwrite(sample / f"wave_idx_{k:05d}.tif", counts[:, :, k])
    ob_root = root / "open_beam"
    for o in range(2):
        d = ob_root / f"observation_{o + 1:02d}"; d.mkdir(parents=True)
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


def test_infer_input_type():
    assert infer_input_type(np.random.default_rng(0).uniform(0, 1, 1000))[0] == "transmission"
    assert infer_input_type(np.random.default_rng(0).poisson(30, 1000).astype(np.float32))[0] == "counts"
    assert infer_input_type(np.random.default_rng(0).normal(0.5, 0.4, 1000))[0] == "attenuation"
    assert _parse_slice("10:20", "x") == (10, 20) and _parse_slice(":5", "x") == (None, 5)


def test_inspect_tiff_with_open_beam(stacks, capsys):
    assert main(["inspect", stacks["sample"], "--open-beam", stacks["open_beam"], "-q"]) == 0
    out = capsys.readouterr().out
    assert "type counts" in out and f"x {K} bins" in out and "dose:" in out and "unknown" not in out.split("dose:")[1].split("\n")[0]


def test_inspect_hdf5_finds_the_group(stacks, capsys):
    assert main(["inspect", stacks["h5"], "-q", "--estimate-rank"]) == 0
    out = capsys.readouterr().out
    assert "type attenuation" in out and "1 view(s)" in out and f"rank {R} estimated" in out and "gains by component" in out


def test_counts_without_open_beam_is_an_error(stacks):
    with pytest.raises(SystemExit, match="open-beam"):
        main(["inspect", stacks["sample"], "-q"])


def test_convert_round_trip_matches_the_direct_load(stacks, tmp_path):
    out = str(tmp_path / "converted.h5")
    assert main(["convert", stacks["sample"], "--open-beam", stacks["open_beam"], "-o", out, "--as-type", "transmission", "-q"]) == 0
    data, meta = hsnt.import_hsnt_data_hdf5(out)
    assert data.shape == (1, ROWS, COLS, K) and meta["dataset_type"] == "transmission"
    with h5py.File(out) as f:
        assert f.attrs["dose"] > 0 and f["bin_indices"].shape == (K,)
    from mbirtorch.hsnt.cli import load_tiff
    ds = load_tiff(stacks["sample"], open_beam=[stacks["open_beam"]])
    assert np.allclose(data.reshape(-1, K), ds.T, atol=1e-6)
    assert ds.info["open_beam_observations"] == 2


def test_wave_bin_and_downsample(stacks):
    from mbirtorch.hsnt.cli import load_tiff
    ds = load_tiff(stacks["sample"], open_beam=[stacks["open_beam"]], wave_bin=4, downsample=2)
    assert ds.T.shape == ((ROWS + 1) // 2 * ((COLS + 1) // 2), K // 4) and ds.bin_indices.tolist() == list(range(0, K, 4))


@cuda
def test_dehydrate_writes_readable_output(stacks, tmp_path):
    out = str(tmp_path / "res")
    assert main(["dehydrate", stacks["h5"], "-o", out, "--dose", str(DOSE), "--max-steps", "200", "-q"]) == 0   # rank estimated
    files = sorted(os.listdir(out))
    assert any(f.endswith("_dehydrated.h5") for f in files) and any(f.endswith("_report.json") for f in files) and any(f.endswith("_maps.png") for f in files)
    data, meta = hsnt.import_hsnt_data_hdf5(os.path.join(out, "processed_dehydrated.h5"))
    Wd, Hd, dtype = data
    assert Wd.shape == (1, ROWS, COLS, R) and Hd.shape == (R, K) and dtype == "attenuation" and Wd.min() >= 0 and Hd.min() >= 0
    den = hsnt.rehydrate(data)
    assert den.shape == (1, ROWS, COLS, K)
    rep = json.load(open(os.path.join(out, "processed_report.json")))
    assert rep["result"]["mode"] == "full" and rep["result"]["loss_final"] > 0 and rep["result"]["rank"] == R
    assert rep["result"]["components"]["proportional_pairs"] == []                     # two distinct materials
    with h5py.File(os.path.join(out, "processed_dehydrated.h5")) as f:
        assert f["mean_pixel_spectrum"].shape == (K,) and f["mean_pixel_contributions"].shape == (R, K)
    W, H = _truth(); X = W @ H
    fit = -np.log(np.clip(den.reshape(-1, K), 1e-12, None)) if dtype == "transmission" else den.reshape(-1, K)
    assert np.linalg.norm(fit - X) / np.linalg.norm(X) < 0.35                                  # denoised attenuation near the truth


@cuda
def test_dehydrate_stream_mode_from_tiffs(stacks, tmp_path):
    out = str(tmp_path / "stream")
    assert main(["dehydrate", stacks["sample"], "--open-beam", stacks["open_beam"], "--rank", str(R), "-o", out, "--mode", "stream",
                 "--chunk-pixels", "40", "--warmup-pixels", "60", "--max-passes", "2", "--no-plots", "-q"]) == 0
    rep = json.load(open(os.path.join(out, "sample_report.json")))
    assert rep["result"]["mode"] == "stream" and rep["result"]["passes"] >= 1 and rep["dose"] > 0


@cuda
def test_dehydrate_stream_mode_with_support_selection(stacks, tmp_path):
    out = str(tmp_path / "stream_support")
    assert main(["dehydrate", stacks["sample"], "--open-beam", stacks["open_beam"], "--rank", str(R), "-o", out, "--mode", "stream",
                 "--chunk-pixels", "40", "--warmup-pixels", "60", "--max-passes", "2", "--spectra", "support", "--support-penalty", "1",
                 "--free-refit", "--no-plots", "-q"]) == 0
    rep = json.load(open(os.path.join(out, "sample_report.json")))
    assert rep["result"]["mode"] == "stream" and 0 < rep["result"]["mean_support_size"] <= R and rep["result"]["support_refit_passes"] >= 0
    assert rep["result"]["loss_final"] >= rep["result"]["loss_mle"] * (1 - 1e-6)     # the constrained refit cannot beat the MLE's fit


def test_rank_is_estimated_by_default(stacks):
    ds = load_hdf5(stacks["h5"])
    n, note, detail = estimate_rank(ds, "cpu", max_rank=4, pool=0)
    assert n == R and "estimated" in note and len(detail["gains"]) == 3 and 30 < detail["full"]["effective_dose"] < 90
    n2, note2, detail2 = estimate_rank(ds, "cpu", max_rank=4, pool=2)                   # pooled pass runs and is recorded
    assert n2 >= n and detail2["pool_block"] == 2 and detail2["pooled"]["pixels"] == (ROWS // 2) * (COLS // 2)


def test_pool_pixels_averages_blocks():
    from mbirtorch.hsnt.cli import pool_pixels
    T = np.arange(2 * 6 * 4 * 3, dtype=np.float32).reshape(2 * 6 * 4, 3)             # 2 views, 6 x 4 pixels, 3 bins
    Tp = pool_pixels(T, (2, 6, 4), 2)
    assert Tp.shape == (2 * 3 * 2, 3)
    block = T.reshape(2, 6, 4, 3)[0, :2, :2].mean(axis=(0, 1))
    assert np.allclose(Tp[0], block)


@cuda
def test_denoise_writes_readable_data_with_estimated_rank(stacks, tmp_path):
    out = str(tmp_path / "den")
    assert main(["denoise", stacks["h5"], "-o", out, "--dose", str(DOSE), "--max-steps", "200", "--no-plots", "-q"]) == 0
    data, meta = hsnt.import_hsnt_data_hdf5(os.path.join(out, "processed_denoised.h5"))
    assert data.shape == (1, ROWS, COLS, K) and meta["dataset_type"] == "attenuation" and np.isfinite(data).all()
    W, H = _truth()
    assert np.linalg.norm(data.reshape(-1, K) - W @ H) / np.linalg.norm(W @ H) < 0.35
    rep = json.load(open(os.path.join(out, "processed_report.json")))
    assert rep["result"]["rank"] == R and "estimated" in rep["result"]["rank_note"]
    assert 0.5 < rep["result"]["fit"]["reduced_chi2"] < 2.0                          # the fit sits at the Poisson noise level
    assert os.path.exists(os.path.join(out, "processed_dehydrated.h5"))
    with h5py.File(os.path.join(out, "processed_denoised.h5")) as f:
        assert f.attrs["rank"] == R and f["bin_indices"].shape == (K,)


@cuda
def test_denoise_without_dehydrated_file_transmission(stacks, tmp_path):
    out = str(tmp_path / "den2")
    assert main(["denoise", stacks["sample"], "--open-beam", stacks["open_beam"], "-o", out, "--rank", "2", "--no-dehydrated",
                 "--as-type", "transmission", "--no-plots", "--max-steps", "100", "-q"]) == 0
    files = os.listdir(out)
    assert not any(f.endswith("_dehydrated.h5") for f in files)
    data, meta = hsnt.import_hsnt_data_hdf5(os.path.join(out, "sample_denoised.h5"))
    assert meta["dataset_type"] == "transmission" and 0 < data.min() and data.max() < 2


def test_component_check_flags_proportional_maps():
    from mbirtorch.hsnt.cli import component_check, mean_pixel_spectrum
    rng = np.random.default_rng(0); W, H = _truth()
    ok = component_check(W, H)                                                        # distinct maps: no pair flagged
    assert ok["max_map_correlation"] < 0.5 and ok["proportional_pairs"] == []
    W2 = np.stack([W[:, 0], W[:, 0] * (1 + 0.05 * rng.standard_normal(W.shape[0]))], 1)   # a split material
    bad = component_check(W2, np.stack([H[0] * 0.5, H[0] * 0.5]))
    assert bad["proportional_pairs"] and bad["proportional_pairs"][0][:2] == (0, 1)
    total, contrib, n = mean_pixel_spectrum(W2, np.stack([H[0] * 0.5, H[0] * 0.5]))
    assert total.shape == (K,) and contrib.shape == (2, K) and n > 0 and np.allclose(total, contrib.sum(0))


def test_support_selection_without_dose_is_an_error(stacks, tmp_path):
    with pytest.raises(SystemExit, match="dose"):
        main(["dehydrate", stacks["h5"], "--rank", str(R), "--spectra", "support", "-o", str(tmp_path), "--device", "cpu", "-q"])


@cuda
def test_rehydrate_command_reconstructs_from_the_dehydrated_file(stacks, tmp_path):
    out = str(tmp_path / "rh")
    assert main(["dehydrate", stacks["h5"], "-o", out, "--rank", str(R), "--max-steps", "150", "--no-plots", "-q"]) == 0
    deh = os.path.join(out, "processed_dehydrated.h5")
    assert main(["rehydrate", deh, "-o", out, "-q"]) == 0                              # all bins, the file's type
    data, meta = hsnt.import_hsnt_data_hdf5(os.path.join(out, "processed_rehydrated.h5"))
    W4, H, dtype = hsnt.import_hsnt_data_hdf5(deh)[0]
    assert data.shape == (1, ROWS, COLS, K) and meta["dataset_type"] == "attenuation" and np.allclose(data, W4 @ H, atol=1e-5)
    sub = str(tmp_path / "rh" / "part.h5")
    assert main(["rehydrate", deh, "-o", sub, "--wave-range", "5:15", "--as-type", "transmission", "-q"]) == 0   # a range, as transmission
    part, meta2 = hsnt.import_hsnt_data_hdf5(sub)
    assert part.shape == (1, ROWS, COLS, 10) and meta2["dataset_type"] == "transmission" and np.allclose(part, np.exp(-(W4 @ H[:, 5:15])), atol=1e-5)
    with h5py.File(sub) as f:
        assert f["bin_indices"][()].tolist() == list(range(5, 15)) and f.attrs["rank"] == R and "dehydrated_source" in f.attrs
    with pytest.raises(SystemExit, match="not a dehydrated file"):
        main(["rehydrate", stacks["h5"], "-o", out, "-q"])                             # hyperspectral input is refused


@cuda
def test_library_dehydrate_and_hyper_denoise(stacks):
    with h5py.File(stacks["h5"]) as f:
        A = f["sample_dataset/data"][()]                                                  # (1, ROWS, COLS, K) attenuation
    sub_data, basis, dtype = hsnt.dehydrate(A, "attenuation", num_materials=R, verbose=0)
    assert sub_data.shape == (1, ROWS, COLS, R) and basis.shape == (R, K) and dtype == "attenuation"
    assert sub_data.min() >= 0 and basis.min() >= 0 and sub_data.dtype == np.float32
    W, H = _truth(); X = W @ H
    den = hsnt.hyper_denoise(A, "attenuation", num_materials=R, verbose=0)
    assert den.shape == A.shape and np.linalg.norm(den.reshape(-1, K) - X) / np.linalg.norm(X) < 0.35
    est = hsnt.dehydrate(A[0], "attenuation", verbose=0)                                  # rank estimated, pooling over (rows, cols)
    assert est[0].shape == (ROWS, COLS, R)
    Tt = np.exp(-A); den_t = hsnt.hyper_denoise(Tt, "transmission", num_materials=R, verbose=0)   # transmission in, transmission out
    assert den_t.shape == A.shape and 0 < den_t.min() and np.linalg.norm(-np.log(den_t).reshape(-1, K) - X) / np.linalg.norm(X) < 0.35
    assert callable(hsnt.l2_dehydrate) and callable(hsnt.l2_hyper_denoise)               # the L2 baseline stays importable


def test_convert_streams_in_small_blocks_and_matches_the_direct_load(stacks, tmp_path):
    out = str(tmp_path / "blocks.h5")
    assert main(["convert", stacks["sample"], "--open-beam", stacks["open_beam"], "-o", out, "--as-type", "transmission",
                 "--wave-bin", "4", "--downsample", "2", "--block-bins", "7", "-q"]) == 0        # 7 -> 4, a multiple of --wave-bin
    data, meta = hsnt.import_hsnt_data_hdf5(out)
    from mbirtorch.hsnt.cli import load_tiff
    ds = load_tiff(stacks["sample"], open_beam=[stacks["open_beam"]], wave_bin=4, downsample=2)
    assert data.shape == (1,) + tuple(ds.spatial_shape[1:]) + (K // 4,) and np.allclose(data.reshape(-1, K // 4), ds.T, atol=1e-6)
    with h5py.File(out) as f:
        assert f.attrs["block_bins"] == 4 and f["bin_indices"][()].tolist() == list(range(0, K, 4)) and f.attrs["open_beam_observations"] == 2
        assert f["data"].chunks[-1] <= 16 and abs(f.attrs["dose"] - ds.dose) / ds.dose < 0.05           # 16-bin slabs along the spectral axis
        checks = json.loads(f.attrs["checks"])
        assert any("dose" in c["message"] for c in checks) and all(c["level"] != "error" for c in checks)


def test_convert_hdf5_to_hdf5_streams(stacks, tmp_path):
    out = str(tmp_path / "h5h5.h5")
    assert main(["convert", stacks["h5"], "-o", out, "--wave-bin", "2", "--downsample", "2", "--block-bins", "6", "-q"]) == 0   # attenuation in and out
    data, meta = hsnt.import_hsnt_data_hdf5(out)
    ds = load_hdf5(stacks["h5"], wave_bin=2, downsample=2)
    assert meta["dataset_type"] == "attenuation" and data.shape == (1,) + tuple(ds.spatial_shape[1:]) + (K // 2,)
    with np.errstate(divide="ignore"):
        ref = -np.log(ds.T)
    assert np.allclose(data.reshape(-1, K // 2), ref, atol=1e-5)
