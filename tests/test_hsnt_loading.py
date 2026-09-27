"""Tests for the hsnt loader's data checks, dose and open-beam handling, and memory: exact value fractions, the
bright-region check against the loader's expected ratio, the calibration's view wording, the dose of a range of a
converted file and the check of a given dose, the smoothing of a low-count open beam, and the working set of
load_dataset."""
import json
import tracemalloc
import warnings

import h5py
import numpy as np
import pytest
import tifffile
from scipy.stats import poisson

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt.loading import (_checks_from_summary, _smooth_open_beam, _stack_to_transmission, _summary_from_T,
                                    convert_to_hdf5, load_dataset)


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


def _write_stack(directory, counts):
    """A TIFF stack of (rows, cols, bins) counts, one image per bin."""
    directory.mkdir(parents=True)
    for k in range(counts.shape[-1]):
        tifffile.imwrite(directory / f"wave_idx_{k:05d}.tif", counts[:, :, k].astype(np.float32))
    return str(directory)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_wave_range_of_a_converted_file_takes_the_dose_of_its_columns(tmp_path):
    """An open beam falling from 40 to 5 counts across 60 bins: a converted file loaded with --wave-range 30:60 takes
    the median recorded open beam of those columns, as the TIFF stack loaded with that range estimates it, not the
    whole file's; --wave-bin scales it."""
    rng = np.random.default_rng(10)
    lam = np.linspace(40, 5, 60)
    sample = _write_stack(tmp_path / "sample", rng.poisson(0.7 * lam, (16, 16, 60)))
    ob = _write_stack(tmp_path / "ob", rng.poisson(lam, (16, 16, 60)))
    conv = str(tmp_path / "conv.h5")
    convert_to_hdf5(sample, output=conv, open_beam=[ob])
    whole, part = load_dataset(conv), load_dataset(conv, wave_range=(30, 60))
    with h5py.File(conv) as f:
        assert whole.dose == pytest.approx(f.attrs["dose"])
        assert part.dose == pytest.approx(np.median(f["open_beam_dose"][30:60])) and part.dose < 0.75 * whole.dose
    assert part.dose == pytest.approx(load_dataset(sample, open_beam=[ob], wave_range=(30, 60)).dose, rel=0.1)
    assert load_dataset(conv, wave_range=(30, 60), wave_bin=2).dose == pytest.approx(2 * part.dose)


def test_a_given_dose_is_checked_against_the_open_beam_dose_a_file_records(tmp_path):
    """A file recording dose 110 at a calibration factor of 1.1 has an open-beam dose of 100: a given 100 passes, the
    scaled 110 (which inspect prints) and 300 warn; on an uncalibrated file recording 100, 300 warns and 110 passes."""
    T = np.full((1, 8, 8, 4), 0.8, dtype=np.float32)
    background = dict(name="given", boxes=[[0, 2, 0, 2]], tiles=[1, 1], factor_median=1.1, tile_medians=[1.1],
                      factor_range=[1.1, 1.1])
    for name, dose, attrs in (("cal", 110.0, dict(background=json.dumps(background))), ("plain", 100.0, {})):
        with h5py.File(tmp_path / f"{name}.h5", "w") as f:
            f.create_dataset("data", data=T)
            f.create_dataset("dataset_type", data=np.bytes_("transmission"))
            f.attrs.update(dose=dose, **attrs)

    def warned(name, dose):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_dataset(str(tmp_path / f"{name}.h5"), dose=dose)
        return " ".join(str(x.message) for x in w if "given dose" in str(x.message))

    assert not warned("cal", 100.0) and "scaled twice" in warned("cal", 110.0) and "50%" in warned("cal", 300.0)
    assert "50%" in warned("plain", 300.0) and not warned("plain", 110.0)


def test_open_beam_smoothing_keeps_zero_counts_and_leaves_out_dead_pixels():
    """At one count per pixel and bin the smoothed open beam keeps the raw mean (leaving out each zero count raised
    it by 1 / (1 - e^-1) = 1.58); a pixel zero in every bin of the block is left out and filled from its
    neighbors, which it does not lower."""
    rng = np.random.default_rng(0)
    ob = rng.poisson(1.0, size=(1, 128, 128, 8)).astype(np.float32)
    assert _smooth_open_beam(ob.copy(), 3).mean() / ob.mean() == pytest.approx(1, abs=0.01)
    ob = rng.poisson(100.0, size=(1, 32, 32, 8)).astype(np.float32)
    ob[0, 10, 10] = 0
    smoothed = _smooth_open_beam(ob, 3)
    assert abs(smoothed[0, 10, 10].mean() / 100 - 1) < 0.05 and abs(smoothed[0, 9:12, 9:12].mean() / 100 - 1) < 0.03


def test_smoothing_one_observation_warns_that_its_variance_reduction_is_assumed(tmp_path):
    """With one open-beam observation the smoothing's variance reduction is that of independent pixels, which a
    detector with correlated noise does not reach: a warning; with two it is measured: ok."""
    rng = np.random.default_rng(4)
    sample = _write_stack(tmp_path / "sample", rng.poisson(150.0, (16, 16, 4)))
    for o in range(2):
        _write_stack(tmp_path / "ob" / f"observation_{o + 1:02d}", rng.poisson(200.0, (16, 16, 4)))
    one = load_dataset(sample, open_beam=[str(tmp_path / "ob" / "observation_01")], open_beam_smoothing=3)
    two = load_dataset(sample, open_beam=[str(tmp_path / "ob")], open_beam_smoothing=3)
    [c1], [c2] = ([c for c in ds.checks if "smoothed" in c.message] for ds in (one, two))
    assert c1.level == "warn" and "overstates" in c1.message and c2.level == "ok"


def test_load_dataset_stays_within_its_memory_budget(tmp_path):
    """Transmissions of (64, 64, 64, 100), and a (39999, 400) table whose bright-level check takes every row, load
    within 1.5 x 16 MiB + 8 MiB of traced memory beyond T at a 16 MiB budget, and within the default budget of
    512 MiB, with the same T and checks at both budgets."""
    MiB = 2**20
    for shape in ((64, 64, 64, 100), (39_999, 400)):
        path = str(tmp_path / f"T{len(shape)}.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.random.default_rng(6).random(shape, dtype=np.float32) * 1.2)
            f.create_dataset("dataset_type", data=np.bytes_("transmission"))
        loads = []
        for budget in (16, 512):
            tracemalloc.start()
            try:
                ds = load_dataset(path, memory_budget_mib=budget)
                excess = tracemalloc.get_traced_memory()[1] - ds.T.nbytes
            finally:
                tracemalloc.stop()
            assert excess <= (1.5 * budget + 8) * MiB if budget == 16 else excess <= budget * MiB, (shape, budget)
            loads.append(ds)
        assert np.array_equal(loads[0].T, loads[1].T) and [c.message for c in loads[0].checks] == \
            [c.message for c in loads[1].checks]
        del ds, loads
