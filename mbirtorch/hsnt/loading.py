"""Reading hyperspectral neutron data: TIFF stacks (with open beam) and hsnt HDF5 files, the data checks, and the
streamed conversion to the hsnt HDF5 layout.

A TIFF stack is a directory of one image per wavelength bin. A stack of counts needs an open-beam stack to become a
transmission ratio; a stack or file that already holds transmissions or attenuations is used as is, and the type is
inferred from the values unless given. Both formats are read through block readers that return bins k0:k1 as
(views, rows, cols, bins) float32: the in-memory load reads all bins in one block, the conversion a few at a time.
"""
import json
import logging
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .io import _create_hyperspectral, _data_selection, _decode, _find_data_group

log = logging.getLogger(__name__)

INPUT_TYPES = ("counts", "transmission", "attenuation")


@dataclass
class Check:
    level: str          # 'ok', 'warn' or 'error'
    message: str


@dataclass
class Dataset:
    """A loaded dataset as the solvers see it, with what is worth reporting about it."""
    T: np.ndarray                       # transmission ratio, (pixels, bins), float32
    dataset_type: str                   # type of the source data: counts, transmission or attenuation
    spatial_shape: tuple                # (views, rows, cols) after view selection and downsampling
    bin_indices: np.ndarray             # source spectral index of each column of T (first index of each bin group)
    dose: float | None                  # open-beam counts per pixel and bin, if known
    source: str
    checks: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def pixels(self):
        return self.T.shape[0]

    @property
    def bins(self):
        return self.T.shape[1]


def _frac(mask):
    return float(np.mean(mask))


def _stats(a):
    """Value statistics on a strided sample of a large array; exact on a small one."""
    s = a if a.size <= 4_000_000 else a.reshape(-1)[:: max(1, a.size // 2_000_000)]
    finite = s[np.isfinite(s)]
    have = finite.size > 0
    return dict(min=float(finite.min()) if have else float("nan"), max=float(finite.max()) if have else float("nan"),
                mean=float(finite.mean()) if have else float("nan"),
                median=float(np.median(finite)) if have else float("nan"),
                nonfinite=_frac(~np.isfinite(s)), negative=_frac(s < 0), zero=_frac(s == 0), sampled=s.size < a.size)


def infer_input_type(a):
    """Guess whether an array holds counts, transmissions or attenuations from its values. Returns (type, reason)."""
    st = _stats(a)
    s = a.reshape(-1)[:: max(1, a.size // 200_000)]
    s = s[np.isfinite(s)]
    if st["min"] >= 0 and st["max"] <= 1.05:
        return "transmission", f"values in [{st['min']:.3g}, {st['max']:.3g}] look like transmission ratios"
    integral = np.allclose(s, np.round(s), atol=1e-6)
    if st["min"] >= 0 and st["max"] > 5 and (integral or st["median"] > 3):
        return "counts", f"nonnegative with max {st['max']:.3g}" + (", integer-valued" if integral else "") + ": counts"
    return "attenuation", f"min {st['min']:.3g} max {st['max']:.3g} (negatives {st['negative']:.1%}): attenuation"


def _summary_from_T(T, dose):
    """The quantities the data checks need, from a transmission matrix held in memory."""
    st = _stats(T)
    P, K = T.shape
    above = _frac(T > 1) if T.size <= 4_000_000 else _frac(T.reshape(-1)[:: max(1, T.size // 2_000_000)] > 1)
    pos = T > 0
    dead_bins = int((pos.sum(0) == 0).sum())
    return dict(pixels=P, bins=K, nbytes=T.nbytes, stats=st, above_one=above, dead_px=_frac(pos.sum(1) == 0),
                dead_bins=dead_bins, const_bins=int((T.std(0) == 0).sum()) - dead_bins, dose=dose)


def _checks_from_summary(sm, spatial_shape, strict=False):
    """The standard data checks for a summary (from _summary_from_T or the conversion's accumulators), as a list of
    Check. Raises ValueError if strict and any check is an error."""
    st, c = sm["stats"], []
    P, K = sm["pixels"], sm["bins"]
    c.append(Check("ok", f"{P:,} pixels x {K:,} bins ({spatial_shape[0]} view(s) x {spatial_shape[1]} x "
                         f"{spatial_shape[2]}), {sm['nbytes'] / 2**30:.2f} GiB as float32"))
    if st["nonfinite"] > 0:
        c.append(Check("error", f"{st['nonfinite']:.2%} of T is NaN or inf; the loader should have replaced these"))
    if st["negative"] > 0:
        c.append(Check("error", f"{st['negative']:.2%} of T is negative: a transmission ratio cannot be"))
    above = sm["above_one"]
    if above > 0.5:
        c.append(Check("warn", f"{above:.1%} of T exceeds 1: the open beam may be too low or the sample missing"))
    elif above > 0:
        c.append(Check("ok", f"{above:.1%} of T exceeds 1 (noise around low attenuation; expected)"))
    if st["zero"] > 0.5:
        c.append(Check("warn", f"{st['zero']:.1%} of T is exactly zero: very low dose or a mostly opaque sample"))
    elif st["zero"] > 0:
        c.append(Check("ok", f"{st['zero']:.2%} of T is exactly zero (zero counts; the likelihood handles them)"))
    if sm["dead_px"] > 0:
        c.append(Check("warn", f"{sm['dead_px']:.2%} of pixels are zero in every bin (dead detector pixels or a mask)"))
    if sm["dead_bins"]:
        c.append(Check("warn", f"{sm['dead_bins']} bins are zero in every pixel; consider --wave-range to drop them"))
    if sm["const_bins"] > 0:
        c.append(Check("warn", f"{sm['const_bins']} bins are constant across pixels"))
    dose = sm["dose"]
    if dose is not None:
        if dose < 1:
            c.append(Check("warn", f"open-beam dose {dose:.3g} counts per pixel and bin is below 1: expect mostly "
                                   "zero counts"))
        else:
            c.append(Check("ok", f"dose {dose:.3g} open-beam counts per pixel and (binned) bin"))
    else:
        c.append(Check("warn", "dose unknown: --dose is needed for support selection"))
    if K > P:
        c.append(Check("warn", f"more bins ({K}) than pixels ({P}): the spectra are poorly determined; use "
                               "--downsample less or --wave-bin more"))
    errors = [x for x in c if x.level == "error"]
    if errors and strict:
        raise ValueError(f"{len(errors)} data check(s) failed: " + "; ".join(x.message for x in errors))
    return c


def _tif_names(directory):
    """Sorted .tif/.tiff paths in a directory, in any letter case (a glob for *.tif misses .TIF on Linux)."""
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".tif", ".tiff")))


def _tif_files(directory):
    """(files, None) for a directory of TIFFs, or (None, subdirectories) for a directory of TIFF directories."""
    files = _tif_names(directory)
    if not files:
        subdirs = sorted(os.path.join(directory, d) for d in os.listdir(directory)
                         if os.path.isdir(os.path.join(directory, d)) and _tif_names(os.path.join(directory, d)))
        if subdirs:
            return None, subdirs
        raise FileNotFoundError(f"no .tif/.tiff files in {directory}")
    idx = [int(m.group(1)) if (m := re.search(r"(\d+)\.tiff?$", os.path.basename(f))) else None for f in files]
    if all(i is not None for i in idx):
        gaps = [(a, b) for a, b in zip(idx, idx[1:]) if b != a + 1]
        if gaps:
            warnings.warn(f"file indices are not consecutive in {directory}: {len(gaps)} gap(s), first at {gaps[0]}")
    else:
        warnings.warn(f"some file names in {directory} carry no trailing index; relying on sorted order")
    return files, None


def _open_beam_dirs(paths):
    """The open-beam stacks named by paths; a directory of observation subdirectories expands to all of them."""
    dirs = []
    for p in paths:
        files, subdirs = _tif_files(p)
        dirs += subdirs if files is None else [p]
    return dirs


class _TiffBlocks:
    """Bins k0:k1 of a TIFF stack (one image per bin) as (1, rows, cols, k1 - k0) float32, decoded in parallel."""

    def __init__(self, directory, wave_range, downsample, workers, desc):
        import tifffile
        files, subdirs = _tif_files(directory)
        if files is None:
            raise ValueError(f"{directory} holds only subdirectories ({len(subdirs)}); pass one of them, or pass the "
                             "parent as the open beam to average them")
        self.files = files[slice(*wave_range)] if wave_range else files
        if not self.files:
            raise ValueError(f"the wave range selects no files of the {len(files)} in {directory}")
        self.first_index = files.index(self.files[0])
        with tifffile.TiffFile(self.files[0]) as t:
            page = t.pages[0]
            self.full_shape, self.source_dtype = tuple(page.shape), str(page.dtype)
        if len(self.full_shape) != 2:
            raise ValueError(f"{self.files[0]}: expected a 2-D image per wavelength bin, got shape {self.full_shape}")
        self.downsample, self.workers, self.desc = downsample, workers, desc
        self.rows, self.cols = np.empty(self.full_shape, dtype=bool)[::downsample, ::downsample].shape
        self.views = 1
        self.file_type = None
        self.metadata = dict(files=len(self.files), first_file=os.path.basename(self.files[0]),
                             last_file=os.path.basename(self.files[-1]))

    @property
    def bins(self):
        return len(self.files)

    def read(self, k0, k1):
        import tifffile
        sel = self.files[k0:k1]
        arr = tifffile.imread(sel[0])[None] if len(sel) == 1 else tifffile.imread(sel, ioworkers=self.workers,
                                                                                   maxworkers=1)
        if arr.ndim != 3 or tuple(arr.shape[1:]) != self.full_shape:
            raise ValueError(f"{self.desc}: images {k0}..{k1 - 1} have shape {arr.shape[1:]}, the first image "
                             f"{self.full_shape}")
        arr = arr[:, ::self.downsample, ::self.downsample]
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1), dtype=np.float32)[None]

    def close(self):
        pass


class _Hdf5Blocks:
    """Bins k0:k1 of an HDF5 dataset in the hsnt layout as (views, rows, cols, k1 - k0) float32."""

    def __init__(self, path, dataset, views, wave_range, downsample):
        import h5py
        self.f = h5py.File(path, "r")
        g, self.gname = _find_data_group(self.f, dataset)
        self.d = d = g["data"]
        self.file_type = _decode(g["dataset_type"][()]) if "dataset_type" in g else None
        self.sel, (self.views, self.rows, self.cols) = _data_selection(d.shape, views, downsample)
        self.full_shape = tuple(d.shape[1:3]) if d.ndim == 4 else tuple(d.shape[:2]) if d.ndim == 3 else (d.shape[0], 1)
        self.source_dtype = str(d.dtype)
        ks = range(d.shape[-1])[slice(*wave_range)] if wave_range else range(d.shape[-1])
        self.first_index, self.bins = (ks[0], len(ks)) if len(ks) else (0, 0)
        self.desc = f"{path}:{self.gname}"
        scalars = {k: g[k][()] for k in g if isinstance(g[k], h5py.Dataset) and k not in ("data", "dataset_type")
                   and g[k].ndim == 0}
        self.metadata = dict(hdf5_scalar_metadata={k: str(v) for k, v in scalars.items()})
        log.info("HDF5 %s: dataset %s/data shape %s dtype %s chunks %s", path, self.gname.rstrip("/"), d.shape, d.dtype,
                 d.chunks)

    def read(self, k0, k1):
        a = self.d[self.sel + (slice(self.first_index + k0, self.first_index + k1),)]
        if self.d.ndim == 3:
            a = a[None]
        elif self.d.ndim == 2:
            a = a[None, :, None, :]
        return np.ascontiguousarray(a, dtype=np.float32)

    def close(self):
        self.f.close()


def _open_source(path, dataset, views, wave_range, downsample, workers, open_beam):
    """The block reader for an input path (a TIFF directory or an HDF5 file)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"input not found: {path}")
    if os.path.isdir(path):
        src = _TiffBlocks(path, wave_range, downsample, workers, desc="sample")
    elif path.lower().endswith((".h5", ".hdf5", ".hdf")):
        if open_beam:
            warnings.warn("the open beam is ignored for HDF5 input")
        src = _Hdf5Blocks(path, dataset, views, wave_range, downsample)
        if src.bins == 0:
            src.close()
            raise ValueError("the wave range selects no bins")
    else:
        raise ValueError(f"{path}: not a directory of TIFFs and not an .h5/.hdf5 file")
    log.info("%s: %d view(s) x %d x %d pixels%s x %d bins, source dtype %s", src.desc, src.views, src.rows, src.cols,
             f" (every {downsample}th row and column of {src.full_shape[0]} x {src.full_shape[1]})" if downsample > 1
             else "", src.bins, src.source_dtype)
    return src


def _open_beam_blocks(paths, wave_range, downsample, workers, sample):
    """One block reader per open-beam observation, checked against the sample's geometry."""
    obs = [_TiffBlocks(d, wave_range, downsample, workers, desc=f"open beam {os.path.basename(d)}")
           for d in _open_beam_dirs(paths)]
    for o in obs:
        if (o.bins, o.rows, o.cols) != (sample.bins, sample.rows, sample.cols):
            raise ValueError(f"{o.desc} has {o.bins} bins of {o.full_shape}, the sample {sample.bins} of "
                             f"{sample.full_shape}")
    return obs


def _open_beam_mean(obs, k0, k1):
    """The mean over open-beam observations of bins k0:k1, as (pixels, bins), or None without observations."""
    ob = None
    for o in obs:
        blk = o.read(k0, k1)
        blk = blk.reshape(-1, blk.shape[-1])
        if ob is None:
            ob = blk
        else:
            ob += blk
    if ob is not None:
        ob /= len(obs)
    return ob


def _resolve_input_type(input_type, src, probe, open_beam):
    """The input type (given, from the file's dataset_type, or inferred from a probe of the values) and why."""
    if input_type != "auto":
        if src.file_type and src.file_type != input_type:
            warnings.warn(f"the given input type {input_type} overrides the file's dataset_type {src.file_type!r}")
        itype, why = input_type, "given"
    elif src.file_type in ("attenuation", "transmission"):
        itype, why = src.file_type, "from the file's dataset_type"
    else:
        itype, why = infer_input_type(probe)
    if itype == "counts" and not (isinstance(src, _TiffBlocks) and open_beam):
        raise ValueError(f"the input holds counts ({why}) but no open beam was given: pass --open-beam DIR, or "
                         "--input-type transmission/attenuation if the values are already normalized")
    if open_beam and itype != "counts" and isinstance(src, _TiffBlocks):
        warnings.warn(f"the open beam is ignored: the input type is {itype}, not counts")
    return itype, why


def _bin_spectral(a, n, how):
    """Group n adjacent bins along the last axis ('sum' for counts, 'mean' for transmissions)."""
    if n <= 1:
        return a
    K = a.shape[-1] // n * n
    g = a[..., :K].reshape(*a.shape[:-1], K // n, n)
    return g.sum(-1) if how == "sum" else g.mean(-1)


def _stack_to_transmission(a, input_type, open_beam=None, wave_bin=1):
    """Convert (pixels, bins) values of the given type to a transmission ratio, binning bins if asked.

    Returns (T, dose, info). Open-beam entries that are zero or negative take the bin's median open beam; every
    non-finite value becomes zero transmission.
    """
    info = {}
    if input_type == "counts":
        counts, ob = _bin_spectral(a, wave_bin, "sum"), _bin_spectral(open_beam, wave_bin, "sum")
        bad = ob <= 0
        if bad.any():
            info["open_beam_zero_frac"] = _frac(bad)
            stride = max(1, ob.shape[0] // 8192)                                   # medians on a pixel subsample
            sub = np.where(bad[::stride], np.nan, ob[::stride])
            med = np.nanmedian(sub, axis=0)
            med = np.nan_to_num(med, nan=float(np.nanmedian(sub)) if np.isfinite(sub).any() else 1.0)
            ob = np.where(bad, med[None, :], ob)
        T = counts / ob
        dose = float(np.median(ob.reshape(-1)[:: max(1, ob.size // 1_000_000)]))
    elif input_type == "transmission":
        T, dose = _bin_spectral(a, wave_bin, "mean"), None
    elif input_type == "attenuation":
        nonfinite = ~np.isfinite(a)
        if nonfinite.any():
            info["attenuation_nonfinite_frac"] = _frac(nonfinite)
        T, dose = _bin_spectral(np.exp(-np.where(nonfinite, np.inf, a)), wave_bin, "mean"), None
    else:
        raise ValueError(f"unknown input type {input_type!r}")
    T = np.nan_to_num(T.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return T, dose, info


def _warn_conversion(info):
    if info.get("open_beam_zero_frac"):
        warnings.warn(f"the open beam is zero or negative in {100 * info['open_beam_zero_frac']:.3g}% of pixel-bins; "
                      "those use the bin's median open beam")
    if info.get("attenuation_nonfinite_frac"):
        warnings.warn(f"{100 * info['attenuation_nonfinite_frac']:.3g}% of the attenuation is NaN or inf; treated as "
                      "zero transmission")


def _check_host_memory(n_bytes, what):
    """Refuse an allocation the host cannot hold, and warn about one that takes most of it."""
    import psutil
    avail = psutil.virtual_memory().available
    if n_bytes > avail:
        raise MemoryError(f"{what} needs {n_bytes / 2**30:.1f} GiB of host memory but {avail / 2**30:.1f} GiB is "
                          "available: use --downsample, --wave-range or --wave-bin, or convert the data first "
                          "(`mbirtorch-hsnt convert` streams)")
    if n_bytes > 0.5 * avail:
        warnings.warn(f"{what} needs {n_bytes / 2**30:.1f} GiB of the {avail / 2**30:.1f} GiB of host memory available")


def _merge_dose(dose, estimate):
    """The dose to use: the given one, checked against the open-beam estimate, else the estimate."""
    if dose is None:
        return estimate
    if estimate is not None and abs(dose - estimate) / estimate > 0.5:
        warnings.warn(f"the given dose {dose:.3g} differs from the open-beam estimate {estimate:.3g} by more than 50%")
    return dose


def load_dataset(path, open_beam=None, input_type="auto", dataset=None, dose=None, views=None, wave_range=None,
                 wave_bin=1, downsample=1, strict=False, workers=None):
    """Load a TIFF stack or an hsnt HDF5 file as a transmission ratio, with the data checks.

    Args:
        path (str): A directory of one TIFF per wavelength bin, or an HDF5 file in the hsnt layout.
        open_beam (list of str, optional): Open-beam TIFF stack(s) for a stack of counts; a directory of observation
            subdirectories is averaged over them. Defaults to None.
        input_type (str, optional): 'counts', 'transmission', 'attenuation' or 'auto' (the file's dataset_type, else
            inferred from the values). Defaults to 'auto'.
        dataset (str, optional): HDF5 group holding 'data'. Defaults to None: the root, or the only group with one.
        dose (float, optional): Open-beam counts per pixel and bin, overriding the open-beam estimate. Defaults to None.
        views (tuple, optional): (start, stop) of the views of 4-D HDF5 data. Defaults to None, all views.
        wave_range (tuple, optional): (start, stop) over the source bins. Defaults to None, all bins.
        wave_bin (int, optional): Group this many adjacent bins. Defaults to 1.
        downsample (int, optional): Keep every n-th row and column. Defaults to 1.
        strict (bool, optional): Raise ValueError if a data check reports an error. Defaults to False.
        workers (int, optional): Threads decoding TIFF images. Defaults to None, min(8, CPUs).

    Returns:
        Dataset: the transmission ratio (pixels, bins) with its geometry, dose, checks and load information.
    """
    t0 = time.perf_counter()
    workers = workers or min(8, os.cpu_count() or 1)
    src = _open_source(path, dataset, views, wave_range, downsample, workers, open_beam)
    try:
        V, rows, cols, nb = src.views, src.rows, src.cols, src.bins
        P = V * rows * cols
        full = V * int(np.prod(src.full_shape)) * nb * 4
        _check_host_memory(full + P * nb * 4 * (4 if open_beam else 2), f"loading {V} x {rows} x {cols} x {nb} values")
        a = src.read(0, nb).reshape(P, nb)
        itype, why = _resolve_input_type(input_type, src, a, open_beam)
        log.info("input type: %s (%s)", itype, why)
        obs = _open_beam_blocks(open_beam, wave_range, downsample, workers, src) if itype == "counts" else []
        ob, n_obs = _open_beam_mean(obs, 0, nb), len(obs)
        T, dose_estimate, info = _stack_to_transmission(a, itype, open_beam=ob, wave_bin=wave_bin)
        del a, ob
        _warn_conversion(info)
        if itype == "counts":
            info["open_beam_observations"] = n_obs
        info.update(src.metadata, load_seconds=round(time.perf_counter() - t0, 2))
        bin_indices = src.first_index + np.arange(0, T.shape[1] * wave_bin, wave_bin)
        ds = Dataset(T=T, dataset_type=itype, spatial_shape=(V, rows, cols), bin_indices=bin_indices,
                     dose=_merge_dose(dose, dose_estimate), source=path if isinstance(src, _TiffBlocks) else src.desc,
                     info=info)
    finally:
        src.close()
    sm = _summary_from_T(ds.T, ds.dose)
    ds.info["T_stats"] = sm["stats"]
    ds.checks = _checks_from_summary(sm, ds.spatial_shape, strict)
    return ds


def _block_bins(sample, n_obs, wave_bin, budget_mib, requested):
    """Bins per conversion block: the largest multiple of wave_bin whose working set fits the budget.

    Two blocks are alive (the one processed and the one prefetched), each the sample plus the open-beam mean, one
    observation is read at full resolution before downsampling, and the transmission and the output block exist.
    """
    if requested:
        block = max(wave_bin, requested // wave_bin * wave_bin)
    else:
        full = int(np.prod(sample.full_shape)) * sample.views * 4
        small = sample.rows * sample.cols * sample.views * 4
        per_bin = full * (2 + (3 if n_obs else 0)) + small * 4
        block = max(wave_bin, int(budget_mib * 2**20 // per_bin) // wave_bin * wave_bin)
    return min(block, sample.bins // wave_bin * wave_bin) or wave_bin


def convert_to_hdf5(path, output=None, open_beam=None, input_type="auto", dataset=None, dose=None, views=None,
                    wave_range=None, wave_bin=1, downsample=1, as_type="attenuation", block_bins=None,
                    memory_budget_mib=256, workers=None, strict=False, progress=False):
    """Convert a TIFF stack or HDF5 file to an hsnt HDF5 file, streamed in blocks of bins.

    Memory holds a few blocks, not the stack. The arguments of :func:`load_dataset` select and interpret the input.

    Args:
        output (str, optional): Output path. Defaults to None, the input's name with .h5.
        as_type (str, optional): Quantity stored, 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        block_bins (int, optional): Bins per block. Defaults to None, from memory_budget_mib.
        memory_budget_mib (float, optional): Working memory for the blocks in MiB. Defaults to 256.
        progress (bool, optional): Show a progress bar. Defaults to False.

    Returns:
        (path, checks, info): the output path, the data checks, and the conversion's statistics.
    """
    import h5py
    from tqdm import tqdm
    t0 = time.perf_counter()
    workers = workers or min(8, os.cpu_count() or 1)
    src = _open_source(path, dataset, views, wave_range, downsample, workers, open_beam)
    V, rows, cols, nb = src.views, src.rows, src.cols, src.bins
    P = V * rows * cols
    probe = src.read(0, min(nb, max(8, wave_bin)))
    itype, why = _resolve_input_type(input_type, src, probe, open_beam)
    log.info("input type: %s (%s)", itype, why if input_type != "auto" or src.file_type else
             f"{why}, from the first {probe.shape[-1]} bins")
    del probe
    obs = _open_beam_blocks(open_beam, wave_range, downsample, workers, src) if itype == "counts" else []
    wave_bin = max(1, wave_bin)
    K = nb // wave_bin
    if K == 0:
        raise ValueError(f"a wave bin of {wave_bin} exceeds the {nb} selected bins")
    block = _block_bins(src, len(obs), wave_bin, memory_budget_mib, block_bins)
    out = output or (os.path.splitext(os.path.normpath(path))[0] + ".h5")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    log.info("streaming %d bins per block (%d blocks), %d TIFF reader threads -> %s (%s, %.2f GiB)", block,
             -(-nb // block), workers, out, as_type, P * K * 4 / 2**30)
    tot = dict(n=0, nonfinite=0, negative=0, zero=0, above=0, inf_out=0)
    pos_px = np.zeros(P, dtype=np.int64)
    pos_bin = np.zeros(K, dtype=np.int64)
    bin_min = np.full(K, np.inf, dtype=np.float32)
    bin_max = np.full(K, -np.inf, dtype=np.float32)
    stride = max(1, -(-P * K // 4_000_000))
    sample = np.empty((-(-P // stride), K), dtype=np.float32)
    dose_blocks, ob_zero, nonfinite_in = [], 0.0, 0.0
    starts = [k0 for k0 in range(0, nb - nb % wave_bin, block) if min(k0 + block, nb) // wave_bin * wave_bin > k0]

    def read_block(k0):
        """The sample block and the open-beam mean over observations for bins k0:k1, as (pixels, bins)."""
        k1 = min(k0 + block, nb) // wave_bin * wave_bin
        return k1, src.read(k0, k1).reshape(P, k1 - k0), _open_beam_mean(obs, k0, k1)

    pool = ThreadPoolExecutor(1)                                    # the next block is read while this one is processed
    pending = pool.submit(read_block, starts[0]) if starts else None
    try:
        with h5py.File(out, "w") as f:
            d = _create_hyperspectral(f, (V, rows, cols, K), as_type, chunks=(1, min(rows, 128), cols, min(16, K)))
            for n_blk, k0 in enumerate(tqdm(starts, desc="convert", unit="block", disable=not progress, leave=False)):
                k1, a, ob = pending.result()
                pending = pool.submit(read_block, starts[n_blk + 1]) if n_blk + 1 < len(starts) else None
                T, dose_b, info_b = _stack_to_transmission(a, itype, open_beam=ob, wave_bin=wave_bin)
                del a, ob
                j0, j1 = k0 // wave_bin, k1 // wave_bin
                if dose_b is not None:
                    dose_blocks.append((dose_b, j1 - j0))
                ob_zero += info_b.get("open_beam_zero_frac", 0.0) * (j1 - j0)
                nonfinite_in += info_b.get("attenuation_nonfinite_frac", 0.0) * (j1 - j0)
                tot["n"] += T.size
                tot["nonfinite"] += int((~np.isfinite(T)).sum())
                tot["negative"] += int((T < 0).sum())
                tot["zero"] += int((T == 0).sum())
                tot["above"] += int((T > 1).sum())
                pos = T > 0
                pos_px += pos.sum(1)
                pos_bin[j0:j1] = pos.sum(0)
                bin_min[j0:j1] = T.min(0)
                bin_max[j0:j1] = T.max(0)
                sample[:, j0:j1] = T[::stride]
                if as_type == "attenuation":
                    with np.errstate(divide="ignore"):
                        values = -np.log(T)
                    tot["inf_out"] += int(np.isinf(values).sum())
                else:
                    values = T
                d[:, :, :, j0:j1] = values.reshape(V, rows, cols, j1 - j0)
                del T, pos, values
            dose_estimate = None
            if dose_blocks:
                vals, w = np.array([v for v, _ in dose_blocks]), np.array([n for _, n in dose_blocks])
                dose_estimate = float(np.median(np.repeat(vals, w)))
            dose = _merge_dose(dose, dose_estimate)
            _warn_conversion(dict(open_beam_zero_frac=ob_zero / K, attenuation_nonfinite_frac=nonfinite_in / K))
            if tot["inf_out"]:
                warnings.warn(f"{tot['inf_out']} zero-transmission entries are inf in the attenuation output; the "
                              "solvers map them back to zero transmission")
            st = dict(_stats(sample), nonfinite=tot["nonfinite"] / tot["n"], negative=tot["negative"] / tot["n"],
                      zero=tot["zero"] / tot["n"], sampled=stride > 1)
            dead_bins = int((pos_bin == 0).sum())
            sm = dict(pixels=P, bins=K, nbytes=P * K * 4, stats=st, above_one=tot["above"] / tot["n"],
                      dead_px=_frac(pos_px == 0), dead_bins=dead_bins,
                      const_bins=int((bin_max == bin_min).sum()) - dead_bins, dose=dose)
            checks = _checks_from_summary(sm, (V, rows, cols), strict)
            f.create_dataset("bin_indices", data=np.arange(src.first_index, src.first_index + K * wave_bin, wave_bin))
            attrs = dict(source=path if isinstance(src, _TiffBlocks) else src.desc, input_type=itype,
                         downsample=downsample, wave_bin=wave_bin, block_bins=block, mbirtorch_hsnt_cli="1",
                         checks=json.dumps([dict(level=c.level, message=c.message) for c in checks]))
            if dose is not None:
                attrs["dose"] = float(dose)
            if obs:
                attrs["open_beam_observations"] = len(obs)
            f.attrs.update(attrs)
    finally:
        pool.shutdown(wait=True)
        src.close()
    seconds = time.perf_counter() - t0
    info = dict(seconds=round(seconds, 2), block_bins=block, blocks=-(-nb // block), open_beam_observations=len(obs),
                stats=st, dose=dose)
    log.info("wrote %s: data %s %s, %.2f GiB, in %.1f s (%.0f bins/s)", out, (V, rows, cols, K), as_type,
             P * K * 4 / 2**30, seconds, nb / seconds)
    return out, checks, info
