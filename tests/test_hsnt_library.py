"""Tests of dehydrate and hyper_denoise on a synthetic attenuation dataset."""

import h5py
import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt._fit import _fit, _fit_fixed_basis
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
    """The synthetic data as an HDF5 attenuation file."""
    root = tmp_path_factory.mktemp("hsnt")
    W, H = _truth()
    rng = np.random.default_rng(1)
    X = (W @ H).reshape(ROWS, COLS, K)
    counts = rng.poisson(DOSE * np.exp(-X)).astype(np.float32)
    A = -np.log(np.maximum(counts / DOSE, 1e-6)).astype(np.float32)
    h5 = root / "processed.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("sample_dataset")
        g.create_dataset("data", data=A[None])                      # (views, rows, cols, bins)
        g.create_dataset("dataset_type", data=np.bytes_("attenuation"))
    return dict(h5=str(h5))


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
