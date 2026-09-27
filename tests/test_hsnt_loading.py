"""Tests for the hsnt loader's data checks, open-beam handling and memory: exact value fractions, the bright-region
check against the loader's expected ratio, the calibration's view wording, and the working set of load_dataset."""
import h5py
import numpy as np
import pytest
from scipy.stats import poisson

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt.loading import _checks_from_summary, _stack_to_transmission, _summary_from_T, load_dataset


def _disk_file(path, n_obs, lam, exposure, K, seed, n=64):
    """A T = 0.5 disk in a sample-free field of n x n pixels, as convert records it: the loader's transmission of
    Poisson counts at the given exposure against an open beam of n_obs observations of Poisson(lam), with its dose,
    per-bin dose and observations."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    disk = ((yy - n // 2) ** 2 + (xx - n // 2) ** 2 < (n * 5 // 16) ** 2).reshape(-1, 1)
    ob = rng.poisson(lam, size=(n_obs, n * n, K)).mean(0).astype(np.float32)
    counts = rng.poisson(exposure * lam * np.where(disk, 0.5, 1.0), size=(n * n, K)).astype(np.float32)
    T, dose, info = _stack_to_transmission(counts, "counts", open_beam=ob)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=T.reshape(1, n, n, K))
        f.create_dataset("dataset_type", data=np.bytes_("transmission"))
        f.create_dataset("open_beam_dose", data=info["dose_per_bin"])
        f.attrs.update(dose=dose, open_beam_observations=n_obs, wave_bin=1)
    return path


def _bright_warnings(path):
    return [c.message for c in load_dataset(path).checks if "transparent" in c.message]


def test_value_fractions_are_exact_and_the_sample_holds_every_bin():
    """The negative, zero and above-one fractions count every entry, and the range comes from whole rows: a flat
    stride over a (P, 10) matrix of more than 4e6 entries sees only every other bin."""
    P, K = 500_000, 10
    T = np.full((P, K), 0.5, dtype=np.float32)
    T[:, 3], T[:, 5], T[:, 7] = -0.01, 0.0, 1.5
    shape = (1, 500, 1000)
    sm = _summary_from_T(T, 10.0, shape)
    st = sm["stats"]
    assert st["negative"] == st["zero"] == sm["above_one"] == pytest.approx(0.1) and st["nonfinite"] == 0
    assert st["min"] == pytest.approx(-0.01) and st["max"] == 1.5 and st["sampled"]
    assert any(c.level == "error" and "negative" in c.message for c in _checks_from_summary(sm, shape))


def test_the_ratio_expectation_matches_the_loaders_open_beam():
    """E[mu / S'] for the loader's open beam, a zero replaced by the median positive count, against a direct sum over
    the Poisson distribution, on both sides of the switch to the series."""
    from mbirtorch.hsnt.loading import _ratio_expectation
    for mu in (0.3, 1.0, 3.7, 20.0, 399.0, 401.0):
        s = np.arange(1, int(mu + 20 * np.sqrt(mu) + 40))
        p, p0 = poisson.pmf(s, mu), poisson.pmf(0, mu)
        median = s[np.searchsorted(np.cumsum(p) / (1 - p0), 0.5)]
        assert _ratio_expectation(mu) == pytest.approx((p * mu / s).sum() + p0 * mu / median, rel=1e-9)


def test_the_bright_region_check_expects_the_loaders_ratio_bias(tmp_path):
    """Matched exposures at 5 counts and 2000 bins raise no alarm (the loader's ratio reads about 1.29 there, not the
    first-order 1 + 1 / (n dose)); a sample run 7% below the open beam's exposure at 20 counts in each of 5
    observations is flagged, against the matched level 1 + bias; a matched run there is not."""
    assert not _bright_warnings(_disk_file(str(tmp_path / "low.h5"), 1, 5.0, 1.0, 2000, seed=200))
    assert "below" in " ".join(_bright_warnings(_disk_file(str(tmp_path / "under.h5"), 5, 20.0, 0.93, 200, seed=2)))
    assert not _bright_warnings(_disk_file(str(tmp_path / "matched.h5"), 5, 20.0, 1.0, 200, seed=2))


def test_the_calibration_reports_the_views_mean_exposure(tmp_path):
    """Views calibrated by sample-free boxes at exposures 1, 1 and 1.3 of the open beam's: the dose scales by the
    views' mean exposure, 1.1, as the checks say."""
    n, k = 16, 6
    X = np.zeros((n, n, k))
    X[4:12, 4:12] = np.linspace(0.2, 1.0, k)
    exposure = np.array([1.0, 1.0, 1.3])
    path = str(tmp_path / "views.h5")
    hsnt.export_hsnt_data_hdf5(path, (exposure[:, None, None, None] * np.exp(-X)).astype(np.float32),
                               hsnt.create_hsnt_metadata(dataset_type="transmission"))
    ds = load_dataset(path, dose=100.0, background_boxes=[(0, 3, 0, 3)])
    assert ds.dose == pytest.approx(100.0 * exposure.mean(), rel=1e-5)
    text = " ".join(c.message for c in ds.checks)
    assert "the dose is the views' mean" in text and "use the views' mean" in text and "median view" not in text
