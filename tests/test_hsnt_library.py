"""Tests of the loader and of dehydrate and hyper_denoise on synthetic TIFF and HDF5 inputs."""
import os

import h5py
import numpy as np
import pytest
import tifffile
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt._fit import _fit, _fit_fixed_basis
from mbirtorch.hsnt.loading import _tif_names, infer_input_type, load_dataset
from hsnt_basis import material_basis

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


def _relative_error(estimate, X):
    return np.linalg.norm(estimate.reshape(-1, K) - X) / np.linalg.norm(X)


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
    for name in ("img_10.TIF", "img_2.tif", "img_1.tiff"):
        (tmp_path / name).write_bytes(b"")
    assert [os.path.basename(f) for f in _tif_names(str(tmp_path))] == ["img_1.tiff", "img_2.tif", "img_10.TIF"]


def test_background_boxes_calibrate_each_tile(tmp_path):
    """A sample run whose exposure differs from the open beam's by tile reads T = 1 in every sample-free corner after
    the calibration, and the dose becomes the sample's; transmission and attenuation files of several views, binned
    and downsampled, are calibrated per view to their true transmission."""
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


def test_library_dehydrate_and_hyper_denoise(stacks):
    with h5py.File(stacks["h5"]) as f:
        A = f["sample_dataset/data"][()]                                 # (1, ROWS, COLS, K) attenuation
    sub_data, basis = hsnt.dehydrate(A, "attenuation", num_materials=R, verbose=0)
    assert sub_data.shape == (1, ROWS, COLS, R) and basis.shape == (R, K)
    assert sub_data.min() >= 0 and basis.min() >= 0 and sub_data.dtype == np.float32
    W, H = _truth()
    X = W @ H
    den = hsnt.hyper_denoise(A, "attenuation", num_materials=R, verbose=0)
    assert den.shape == A.shape and _relative_error(den, X) < 0.35
    est = hsnt.dehydrate(A[0], "attenuation", verbose=0)                 # rank estimated, pooling over (rows, cols)
    assert est[0].shape == (ROWS, COLS, R)
    den_t = hsnt.hyper_denoise(np.exp(-A), "transmission", num_materials=R, verbose=0)    # transmission in and out
    assert den_t.shape == A.shape and 0 < den_t.min() and _relative_error(-np.log(den_t), X) < 0.35
    T2 = np.exp(-A).reshape(-1, K).astype(np.float32)                 # the streamed fit, in chunks of 40 pixels
    W_s, H_s, rep = _fit(T2, R, mode="stream", chunk_pixels=40, max_passes=2)
    assert rep["mode"] == "stream" and W_s.shape == (ROWS * COLS, R) and W_s.min() >= 0
    assert _relative_error(W_s @ H_s, X) < 0.35
    phantom = material_basis()
    counts, open_beam, _, truth = hsnt.generate_hyper_data(phantom, 1, 8, 8, 300, None, 0)  # the 7th argument is verbose
    assert counts.shape == open_beam.shape == truth.shape == (1, 8, 8, phantom.shape[1])
    assert np.all(open_beam == 300) and np.abs(counts / open_beam - np.exp(-truth)).max() > 1e-2
    maps, same = hsnt.dehydrate(A, "attenuation", subspace_basis=basis, verbose=0)  # the maps of a given basis
    assert np.array_equal(same, basis) and np.abs(maps - sub_data).max() < 1e-3 * sub_data.max()
    W_c, _, rep = _fit_fixed_basis(T2, basis, mode="stream", chunk_pixels=40)
    assert rep["chunks"] == 3 and np.abs(W_c.reshape(maps.shape) - maps).max() < 1e-4 * maps.max()
    with pytest.raises(ValueError, match="subspace_basis"):
        hsnt.dehydrate(A, "attenuation", subspace_basis=basis, num_materials=R + 1, verbose=0)
    with pytest.raises(TypeError, match="MBIRJAX"):
        hsnt.hyper_denoise(A, num_materials=R, safety_factor=2, verbose=0)          # the NMF's keywords are refused
