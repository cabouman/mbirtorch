"""Command-line interface for the hsnt factorization: ``mbirtorch-hsnt`` (or ``python -m mbirtorch.hsnt``).

Three subcommands share one loader and one set of data checks:

    inspect    load a dataset, run the checks, print what a solve would see (no GPU needed)
    convert    write a TIFF stack (with its open beam) to the package's HDF5 layout, once
    factorize  fit the NNAL factorization and write maps, spectra, plots and a JSON report

Inputs are either an HDF5 file in the package layout (``data`` with the spectral axis last, ``dataset_type``,
optionally inside a group) or a directory of one TIFF image per wavelength bin. A TIFF stack of counts needs an
open-beam stack (``--open-beam``) to become a transmission ratio; a stack that already holds transmissions or
attenuations is used as is, and the type is inferred from the values unless ``--input-type`` says otherwise.

The factors are written in the package's dehydrated layout, so ``import_hsnt_data_hdf5`` reads them back and
``rehydrate`` reconstructs the denoised data from them. Run any subcommand with ``-h`` for the options.
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("mbirtorch.hsnt")

_BYTES_PER_ELEMENT_FULL = 48        # joint_newton's working set: ~12 float32 arrays of T's shape (57 KB/px at K = 1200)
_BYTES_PER_ELEMENT_STREAM = 24      # solve_W on one chunk plus the accumulators
_TYPES = ("counts", "transmission", "attenuation")


# ----------------------------------------------------------------------------------------------------------------------
# Data checks
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Check:
    level: str          # 'ok', 'warn' or 'error'
    message: str


@dataclass
class Dataset:
    """What the loader hands to the solver, plus everything worth reporting about it."""
    T: np.ndarray                       # transmission ratio, (pixels, bins), float32
    dataset_type: str                   # type of the SOURCE data: counts, transmission or attenuation
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


def _stats(a, name):
    """Value statistics on a strided sample of a large array; exact on a small one."""
    s = a if a.size <= 4_000_000 else a.reshape(-1)[:: max(1, a.size // 2_000_000)]
    finite = s[np.isfinite(s)]
    d = dict(min=float(finite.min()) if finite.size else float("nan"), max=float(finite.max()) if finite.size else float("nan"),
             mean=float(finite.mean()) if finite.size else float("nan"), median=float(np.median(finite)) if finite.size else float("nan"),
             nonfinite=_frac(~np.isfinite(s)), negative=_frac(s < 0), zero=_frac(s == 0), sampled=s.size < a.size)
    log.debug("%s: %s", name, ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))
    return d


def infer_input_type(a):
    """Guess whether an array holds counts, transmissions or attenuations from its values, with the reason."""
    st = _stats(a, "type inference")
    s = a.reshape(-1)[:: max(1, a.size // 200_000)]
    s = s[np.isfinite(s)]
    if st["min"] >= 0 and st["max"] <= 1.05:
        return "transmission", f"values in [{st['min']:.3g}, {st['max']:.3g}] look like transmission ratios"
    integral = np.allclose(s, np.round(s), atol=1e-6)
    if st["min"] >= 0 and st["max"] > 5 and (integral or st["median"] > 3):
        return "counts", f"nonnegative with max {st['max']:.3g}" + (", integer-valued" if integral else "") + ": counts"
    return "attenuation", f"min {st['min']:.3g} max {st['max']:.3g} (negatives {st['negative']:.1%}): attenuation"


def run_checks(ds: Dataset, strict=False):
    """Append the standard checks to ds.checks and raise if strict and any is an error."""
    T = ds.T
    c = ds.checks
    st = _stats(T, "T")
    ds.info["T_stats"] = st
    P, K = T.shape
    c.append(Check("ok", f"{P:,} pixels x {K:,} bins ({ds.spatial_shape[0]} view(s) x {ds.spatial_shape[1]} x {ds.spatial_shape[2]}), "
                         f"{T.nbytes / 2**30:.2f} GiB as float32"))
    if st["nonfinite"] > 0:
        c.append(Check("error", f"{st['nonfinite']:.2%} of T is NaN or inf; the loader should have replaced these"))
    if st["negative"] > 0:
        c.append(Check("error", f"{st['negative']:.2%} of T is negative: a transmission ratio cannot be"))
    above = _frac(T > 1) if T.size <= 4_000_000 else _frac(T.reshape(-1)[:: max(1, T.size // 2_000_000)] > 1)
    if above > 0.5:
        c.append(Check("warn", f"{above:.1%} of T exceeds 1: the open beam may be too low or the sample missing"))
    elif above > 0:
        c.append(Check("ok", f"{above:.1%} of T exceeds 1 (noise around low attenuation; expected)"))
    if st["zero"] > 0.5:
        c.append(Check("warn", f"{st['zero']:.1%} of T is exactly zero: very low dose or a mostly opaque sample"))
    elif st["zero"] > 0:
        c.append(Check("ok", f"{st['zero']:.2%} of T is exactly zero (zero counts; the likelihood handles them)"))
    dead_px = _frac((T > 0).sum(1) == 0)
    if dead_px > 0:
        c.append(Check("warn", f"{dead_px:.2%} of pixels are zero in every bin (dead detector pixels or a mask)"))
    dead_bins = int(((T > 0).sum(0) == 0).sum())
    if dead_bins:
        c.append(Check("warn", f"{dead_bins} bins are zero in every pixel; consider --wave-range to drop them"))
    const_bins = int((T.std(0) == 0).sum()) - dead_bins
    if const_bins > 0:
        c.append(Check("warn", f"{const_bins} bins are constant across pixels"))
    if ds.dose is not None:
        if ds.dose < 1:
            c.append(Check("warn", f"open-beam dose {ds.dose:.3g} counts per pixel and bin is below 1: expect mostly zero counts"))
        else:
            c.append(Check("ok", f"dose {ds.dose:.3g} open-beam counts per pixel and (binned) bin"))
    else:
        c.append(Check("warn", "dose unknown: --dose is needed for support selection and the gauge fix"))
    if K > P:
        c.append(Check("warn", f"more bins ({K}) than pixels ({P}): the spectra are poorly determined; use --downsample less or --wave-bin more"))
    errors = [x for x in c if x.level == "error"]
    for x in c:
        getattr(log, {"ok": "info", "warn": "warning", "error": "error"}[x.level])("check: %s", x.message)
    if errors and strict:
        raise SystemExit(f"{len(errors)} data check(s) failed (see above); drop --strict to proceed anyway")
    return c


# ----------------------------------------------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------------------------------------------
def _tif_files(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.tif")) + glob.glob(os.path.join(directory, "*.tiff")))
    if not files:
        subdirs = sorted(d for d in glob.glob(os.path.join(directory, "*")) if os.path.isdir(d)
                         and (glob.glob(os.path.join(d, "*.tif")) or glob.glob(os.path.join(d, "*.tiff"))))
        if subdirs:
            return None, subdirs
        raise FileNotFoundError(f"no .tif/.tiff files in {directory}")
    idx = [int(m.group(1)) if (m := re.search(r"(\d+)\.tiff?$", os.path.basename(f))) else None for f in files]
    if all(i is not None for i in idx):
        gaps = [(a, b) for a, b in zip(idx, idx[1:]) if b != a + 1]
        if gaps:
            log.warning("file indices are not consecutive in %s: %d gap(s), first at %s", directory, len(gaps), gaps[0])
    else:
        log.warning("some file names in %s carry no trailing index; relying on sorted order", directory)
    return files, None


def _check_host_memory(n_bytes, what):
    """Warn or stop before allocating a stack that the host cannot hold."""
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except ImportError:
        return
    if n_bytes > avail:
        raise SystemExit(f"{what} needs {n_bytes / 2**30:.1f} GiB of host memory but {avail / 2**30:.1f} GiB is available: use "
                         f"--downsample, --wave-range or --wave-bin, or convert once to HDF5 on a larger machine")
    if n_bytes > 0.5 * avail:
        log.warning("%s needs %.1f GiB of the %.1f GiB of host memory available", what, n_bytes / 2**30, avail / 2**30)


def read_tif_stack(directory, wave_range=None, downsample=1, desc="stack"):
    """Read one image per wavelength bin into an array (rows, cols, bins); checks shapes and dtypes agree."""
    import tifffile
    files, subdirs = _tif_files(directory)
    if files is None:
        raise FileNotFoundError(f"{directory} holds only subdirectories ({len(subdirs)}); pass one of them, or pass the "
                                f"parent as --open-beam to average them")
    sel = files[slice(*wave_range)] if wave_range else files
    log.info("%s: %d files in %s%s", desc, len(sel), directory,
             f" (of {len(files)}, --wave-range {wave_range[0]}:{wave_range[1]})" if wave_range else "")
    first = tifffile.imread(sel[0])
    if first.ndim != 2:
        raise ValueError(f"{sel[0]}: expected a 2-D image per wavelength bin, got shape {first.shape}")
    rows, cols = first[::downsample, ::downsample].shape
    out = np.empty((rows, cols, len(sel)), dtype=np.float32)
    dtypes = set()
    try:
        from tqdm import tqdm
        it = tqdm(sel, desc=desc, unit="img", disable=not log.isEnabledFor(logging.INFO), leave=False)
    except ImportError:
        it = sel
    for k, f in enumerate(it):
        img = tifffile.imread(f)
        dtypes.add(str(img.dtype))
        if img.shape != first.shape:
            raise ValueError(f"{f}: shape {img.shape} differs from the first image's {first.shape}")
        out[:, :, k] = img[::downsample, ::downsample]
    log.info("%s: %dx%d pixels%s, %d bins, source dtype %s, range [%.4g, %.4g]", desc, rows, cols,
             f" (downsampled {downsample}x from {first.shape[0]}x{first.shape[1]})" if downsample > 1 else "",
             len(sel), "/".join(sorted(dtypes)), np.nanmin(out), np.nanmax(out))
    if len(dtypes) > 1:
        log.warning("%s: mixed dtypes %s across files", desc, sorted(dtypes))
    return out, sel


def read_open_beam(paths, wave_range, downsample, expected_shape):
    """Average one or more open-beam stacks; a directory of observation subdirectories is averaged over them."""
    dirs = []
    for p in paths:
        files, subdirs = _tif_files(p)
        dirs += subdirs if files is None else [p]
    acc, n = None, 0
    for d in dirs:                                   # running mean: one observation in memory at a time
        ob, _ = read_tif_stack(d, wave_range, downsample, desc=f"open beam {os.path.basename(d)}")
        if ob.shape != expected_shape:
            raise ValueError(f"open beam {d} has shape {ob.shape}, the sample {expected_shape}")
        n += 1
        if acc is None:
            acc = ob
        else:
            acc += (ob - acc) / n
        del ob
    log.info("open beam: %d observation(s) averaged; per-pixel-bin counts median %.3g, mean %.3g", n, np.median(acc), acc.mean())
    return acc, n


def _bin_spectral(a, n, how):
    """Group n adjacent bins along the last axis: sum counts, average transmissions."""
    if n <= 1:
        return a
    K = a.shape[-1] // n * n
    if K < a.shape[-1]:
        log.debug("--wave-bin %d drops the last %d bin(s)", n, a.shape[-1] - K)
    g = a[..., :K].reshape(*a.shape[:-1], K // n, n)
    return g.sum(-1) if how == "sum" else g.mean(-1)


def _to_transmission(a, input_type, open_beam=None, wave_bin=1):
    """Convert a stack of the given type to a transmission ratio, binning bins if asked. Returns (T, dose, info)."""
    info = {}
    if wave_bin > 1:
        log.info("--wave-bin %d: %d source bins -> %d (dropping the last %d)", wave_bin, a.shape[-1], a.shape[-1] // wave_bin, a.shape[-1] % wave_bin)
    if input_type == "counts":
        if open_beam is None:
            raise SystemExit("a stack of counts needs an open beam: pass --open-beam DIR (or --input-type if the values "
                             "are already transmissions or attenuations)")
        counts, ob = _bin_spectral(a, wave_bin, "sum"), _bin_spectral(open_beam, wave_bin, "sum")
        bad = ob <= 0
        if bad.any():
            info["open_beam_zero_frac"] = _frac(bad)
            log.warning("open beam is zero or negative in %.3g%% of pixel-bins; those use the bin's median open beam", 100 * _frac(bad))
            med = np.median(np.where(bad, np.nan, ob), axis=(0, 1))
            med = np.nan_to_num(med, nan=float(np.nanmedian(ob)))
            ob = np.where(bad, np.broadcast_to(med, ob.shape), ob)
        T = counts / ob
        dose = float(np.median(ob))
        info["counts_stats"] = _stats(counts, "counts")
    elif input_type == "transmission":
        T, dose = _bin_spectral(a, wave_bin, "mean"), None
    elif input_type == "attenuation":
        A = a
        nonfinite = ~np.isfinite(A)
        if nonfinite.any():
            info["attenuation_nonfinite_frac"] = _frac(nonfinite)
            log.warning("%.3g%% of the attenuation is NaN/inf (zero counts logged?); treated as zero transmission",
                        100 * _frac(nonfinite))
        T = np.exp(-np.where(nonfinite, np.inf, A))
        T = _bin_spectral(T, wave_bin, "mean")
        dose = None
    else:
        raise ValueError(input_type)
    T = np.nan_to_num(T.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return T, dose, info


def _parse_slice(text, name):
    if text is None:
        return None
    m = re.fullmatch(r"(-?\d*):(-?\d*)", text.strip())
    if not m:
        raise SystemExit(f"--{name} expects START:STOP (Python slice), got {text!r}")
    return (int(m.group(1)) if m.group(1) else None, int(m.group(2)) if m.group(2) else None)


def _find_h5_dataset(f, dataset):
    import h5py
    if dataset:
        if dataset not in f:
            raise SystemExit(f"--dataset {dataset!r} not found; groups at the root: {[k for k, v in f.items() if isinstance(v, h5py.Group)]}")
        return f[dataset], dataset
    if "data" in f:
        return f, "/"
    if all(k in f for k in ("subspace_data", "subspace_basis")):
        raise SystemExit("this file already holds dehydrated factors (subspace_data/subspace_basis), not hyperspectral data")
    cands = [k for k, v in f.items() if isinstance(v, h5py.Group) and "data" in v]
    if len(cands) == 1:
        log.info("HDF5: using group %r (the only group with a 'data' dataset)", cands[0])
        return f[cands[0]], cands[0]
    if not cands:
        raise SystemExit(f"no 'data' dataset at the root or in a root group; root members: {list(f.keys())}")
    raise SystemExit(f"several groups hold a 'data' dataset: {cands}; choose one with --dataset")


def load_hdf5(path, dataset=None, views=None, wave_range=None, wave_bin=1, downsample=1, input_type="auto"):
    import h5py
    with h5py.File(path, "r") as f:
        g, gname = _find_h5_dataset(f, dataset)
        d = g["data"]
        log.info("HDF5 %s: dataset %s/data shape %s dtype %s chunks %s", path, gname.rstrip("/"), d.shape, d.dtype, d.chunks)
        dtype_str = g["dataset_type"][()] if "dataset_type" in g else None
        if isinstance(dtype_str, (bytes, np.bytes_)):
            dtype_str = dtype_str.decode()
        if d.ndim == 2:
            sel = (slice(None),)
            shape3 = (1, d.shape[0], 1)
        elif d.ndim == 3:
            sel = (slice(None, None, downsample), slice(None, None, downsample))
            shape3 = None
        elif d.ndim == 4:
            v = slice(*views) if views else slice(None)
            sel = (v, slice(None, None, downsample), slice(None, None, downsample))
            shape3 = None
        else:
            raise SystemExit(f"data has {d.ndim} dimensions; expected (views, rows, cols, bins), (rows, cols, bins) or (pixels, bins)")
        ws = slice(*wave_range) if wave_range else slice(None)
        a = d[sel + (ws,)]
        if d.ndim == 3:
            a = a[None]
        if d.ndim == 2:
            a = a[None, :, None, :]
        spatial = a.shape[:3]
        bin_idx = np.arange(d.shape[-1])[ws]
        meta = {k: g[k][()] for k in g if isinstance(g[k], h5py.Dataset) and k not in ("data", "dataset_type") and g[k].ndim == 0}
    a = a.astype(np.float32, copy=False)
    if input_type == "auto":
        if dtype_str in ("attenuation", "transmission"):
            itype, why = dtype_str, "from the file's dataset_type"
        else:
            itype, why = infer_input_type(a)
            log.warning("no dataset_type in the file; inferred %s (%s). Pass --input-type to override.", itype, why)
    else:
        itype = input_type
        if dtype_str and dtype_str != itype:
            log.warning("--input-type %s overrides the file's dataset_type %r", itype, dtype_str)
    log.info("input type: %s%s", itype, "" if input_type == "auto" and dtype_str else " (given)")
    T, dose, info = _to_transmission(a.reshape(-1, a.shape[-1]), itype, wave_bin=wave_bin)
    ds = Dataset(T=T, dataset_type=itype, spatial_shape=spatial, bin_indices=bin_idx[: T.shape[1] * wave_bin: wave_bin], dose=dose,
                 source=f"{path}:{gname}", info=dict(info, hdf5_scalar_metadata={k: str(v) for k, v in meta.items()}))
    return ds


def _stack_geometry(directory, wave_range, downsample):
    """(rows, cols, bins) a stack would load as, from its first image and file count, without reading it."""
    import tifffile
    files, subdirs = _tif_files(directory)
    if files is None:
        files, _ = _tif_files(subdirs[0])
    sel = files[slice(*wave_range)] if wave_range else files
    with tifffile.TiffFile(sel[0]) as t:
        shape = t.pages[0].shape
    return -(-shape[0] // downsample), -(-shape[1] // downsample), len(sel)


def load_tiff(path, open_beam=None, views=None, wave_range=None, wave_bin=1, downsample=1, input_type="auto"):
    rows, cols, nb = _stack_geometry(path, wave_range, downsample)
    n_stacks = 1 + (2 if open_beam else 0)             # the sample, plus the open-beam running mean and the observation being read
    _check_host_memory(rows * cols * nb * 4 * n_stacks, f"loading {rows}x{cols}x{nb} TIFF stack(s)")
    a, files = read_tif_stack(path, wave_range, downsample, desc="sample")
    ob = None
    if input_type == "auto":
        itype, why = infer_input_type(a)
        if itype == "counts" and not open_beam:
            raise SystemExit(f"the stack looks like counts ({why}) but no --open-beam was given; pass --open-beam DIR, or "
                             f"--input-type transmission/attenuation if the values are already normalised")
        log.info("input type: %s (%s)", itype, why)
    else:
        itype = input_type
        log.info("input type: %s (given)", itype)
    if open_beam and itype != "counts":
        log.warning("--open-beam ignored: the input type is %s, not counts", itype)
    if itype == "counts":
        ob, n_obs = read_open_beam(open_beam, wave_range, downsample, a.shape)
    all_idx = np.arange(len(_tif_files(path)[0]))
    bin_idx = all_idx[slice(*wave_range)] if wave_range else all_idx
    T, dose, info = _to_transmission(a.reshape(-1, a.shape[-1]), itype, open_beam=None if ob is None else ob.reshape(-1, ob.shape[-1]), wave_bin=wave_bin)
    if ob is not None:
        info["open_beam_observations"] = n_obs
    ds = Dataset(T=T, dataset_type=itype, spatial_shape=(1, a.shape[0], a.shape[1]), bin_indices=bin_idx[: T.shape[1] * wave_bin: wave_bin],
                 dose=dose, source=path, info=dict(info, files=len(files), first_file=os.path.basename(files[0]), last_file=os.path.basename(files[-1])))
    return ds


def load_dataset(args):
    """Dispatch on the input path: an HDF5 file or a TIFF directory."""
    p = args.input
    wave_range = _parse_slice(args.wave_range, "wave-range")
    views = _parse_slice(args.views, "views")
    if not os.path.exists(p):
        raise SystemExit(f"input not found: {p}")
    t0 = time.perf_counter()
    if os.path.isdir(p):
        ds = load_tiff(p, open_beam=args.open_beam, views=views, wave_range=wave_range, wave_bin=args.wave_bin,
                       downsample=args.downsample, input_type=args.input_type)
    elif p.lower().endswith((".h5", ".hdf5", ".hdf")):
        if args.open_beam:
            log.warning("--open-beam is ignored for HDF5 input")
        ds = load_hdf5(p, dataset=args.dataset, views=views, wave_range=wave_range, wave_bin=args.wave_bin,
                       downsample=args.downsample, input_type=args.input_type)
    else:
        raise SystemExit(f"{p}: not a directory of TIFFs and not an .h5/.hdf5 file")
    if args.dose is not None:
        if ds.dose is not None and abs(args.dose - ds.dose) / ds.dose > 0.5:
            log.warning("--dose %.3g differs from the open-beam estimate %.3g by more than 50%%", args.dose, ds.dose)
        ds.dose = args.dose
    ds.info["load_seconds"] = round(time.perf_counter() - t0, 2)
    log.info("loaded in %.1f s", ds.info["load_seconds"])
    run_checks(ds, strict=getattr(args, "strict", False))
    return ds


# ----------------------------------------------------------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------------------------------------------------------
def _device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no CUDA device is available")
    if name == "cpu":
        log.warning("running on the CPU: expect one to two orders of magnitude longer than a GPU")
    return name


def plan_memory(ds: Dataset, device, mode, chunk_pixels):
    """Decide full or streamed solve from the free device memory; returns (mode, chunk_pixels, note)."""
    import torch
    P, K = ds.T.shape
    need_full = P * K * _BYTES_PER_ELEMENT_FULL
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
    else:
        import psutil
        free = total = psutil.virtual_memory().available
        name = "cpu"
    note = f"{name}: {free / 2**30:.1f} GiB free of {total / 2**30:.1f}; a full solve needs about {need_full / 2**30:.1f} GiB"
    if mode == "auto":
        mode = "full" if need_full < 0.7 * free else "stream"
    if mode == "stream":
        if device != "cuda":
            raise SystemExit("stream mode needs a CUDA device (it pins host memory for the transfers); the data do not fit a full solve on the CPU")
        if chunk_pixels is None:
            chunk_pixels = int(0.4 * free / (K * _BYTES_PER_ELEMENT_STREAM)) // 1024 * 1024
            chunk_pixels = max(1024, min(chunk_pixels, P))
        note += f"; streaming in chunks of {chunk_pixels:,} pixels ({-(-P // chunk_pixels)} chunks)"
    log.info("plan: %s solve. %s", mode, note)
    return mode, chunk_pixels, note


def solve(ds: Dataset, args, device):
    """Run the factorization and the requested post-estimators. Returns (W, H, report) with W, H numpy."""
    import torch
    from mbirtorch.hsnt import (nnal_factorization, stream_factorization, stable_nnal, unconstrained_spectra,
                                support_selected_spectra, pure_pixel_gauge)
    rep = {}
    mode, chunk, rep["memory_plan"] = plan_memory(ds, device, args.mode, args.chunk_pixels)
    rep["mode"] = mode
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    if mode == "full":
        T = torch.from_numpy(ds.T).to(device)
        W, H, steps = nnal_factorization(T, method=args.method, num_materials=args.rank, max_steps=args.max_steps,
                                         rel_tol=args.rel_tol, random_state=args.seed)
        rep["steps"] = int(steps)
    else:
        if args.method != "joint_newton":
            log.warning("stream mode always uses joint_newton for the warm-up and block Newton for the polish; --method %s ignored", args.method)
        chunks = [torch.from_numpy(ds.T[i:i + chunk]) for i in range(0, ds.pixels, chunk)]
        stats = {}
        W_chunks, H, passes = stream_factorization(chunks, args.rank, max_passes=args.max_passes, rel_tol=args.rel_tol,
                                                   warmup_pixels=min(args.warmup_pixels, ds.pixels), device=device,
                                                   random_state=args.seed, verbose=log.isEnabledFor(logging.DEBUG), stats=stats,
                                                   nonneg_W=(args.spectra != "unconstrained"))
        W = torch.cat([w.to(device) for w in W_chunks])
        rep.update(passes=int(passes), loss_per_pass=stats.get("loss"), kkt_per_pass=stats.get("kkt"))
        T = None
    if device == "cuda":
        torch.cuda.synchronize()
    rep["solve_seconds"] = round(time.perf_counter() - t0, 2)

    def loss(Wx, Hx):
        if T is not None:
            return stable_nnal(Wx.double() @ Hx.double(), T.double()).item()
        return float(sum(stable_nnal(Wx[i:i + chunk].double() @ Hx.double(), torch.from_numpy(ds.T[i:i + chunk]).to(device).double()).item()
                         for i in range(0, ds.pixels, chunk)))
    rep["loss_mle"] = loss(W, H)
    log.info("factorization: %s, %s in %.1f s, loss %.6g, W zeros %.1f%%, H zeros %.1f%%", mode,
             f"{rep['steps']} steps" if "steps" in rep else f"{rep['passes']} polish passes", rep["solve_seconds"], rep["loss_mle"],
             100 * (W == 0).double().mean().item(), 100 * (H == 0).double().mean().item())

    needs_dose = args.spectra == "support" or args.gauge
    if needs_dose and ds.dose is None:
        raise SystemExit("support selection and the gauge fix need the dose (open-beam counts per pixel and bin): pass --dose, "
                         "or give --open-beam with a TIFF stack of counts")
    if mode == "full" and args.spectra == "unconstrained":
        t1 = time.perf_counter(); W, H, st = unconstrained_spectra(T, W, H)
        rep["unconstrained_steps"], rep["unconstrained_seconds"] = int(st), round(time.perf_counter() - t1, 2)
        log.info("unconstrained spectra: %d steps in %.1f s, loss %.6g", st, rep["unconstrained_seconds"], loss(W, H))
    elif mode == "full" and args.spectra == "support":
        if args.rank > 6:
            raise SystemExit("support selection enumerates all 2^R - 1 subsets and is limited to --rank 6")
        t1 = time.perf_counter(); W, H, support, st = support_selected_spectra(T, W, H, ds.dose)
        rep["support_steps"], rep["support_seconds"] = int(st), round(time.perf_counter() - t1, 2)
        rep["mean_support_size"] = support.sum(1).double().mean().item()
        log.info("support selection: mean %.2f materials per pixel, refit %d steps in %.1f s, loss %.6g",
                 rep["mean_support_size"], st, rep["support_seconds"], loss(W, H))
    elif args.spectra != "mle" and mode == "stream":
        log.info("stream mode: %s spectra handled inside the polish passes (nonneg_W=%s)", args.spectra, args.spectra != "unconstrained")
    if args.gauge:
        if mode != "full":
            raise SystemExit("--gauge needs a full solve (the clustering runs on all pixels at once); use --mode full, --downsample or --wave-bin")
        t1 = time.perf_counter(); W, H, A, labels = pure_pixel_gauge(T, W, H, ds.dose)
        sizes = [int((labels == k).sum()) for k in range(args.rank)]
        rep["gauge_seconds"], rep["gauge_cluster_sizes"] = round(time.perf_counter() - t1, 2), sizes
        rep["gauge_condition"] = torch.linalg.cond(A).item()
        log.info("gauge fix: clusters %s of %d pixels with material, cond(A) %.1f, %.1f s", sizes, int((labels >= 0).sum()),
                 rep["gauge_condition"], rep["gauge_seconds"])
        if min(sizes) < 0.01 * sum(sizes):
            log.warning("one gauge cluster holds under 1%% of the material pixels: a material without pure pixels; the fix may have failed")
    rep["loss_final"] = loss(W, H)
    rep["W_zero_frac"], rep["H_zero_frac"] = (W == 0).double().mean().item(), (H == 0).double().mean().item()
    if device == "cuda":
        rep["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return W.cpu().numpy(), H.cpu().numpy(), rep


# ----------------------------------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------------------------------
def _out_paths(args, ds):
    """Output base path: -o names a directory (created if needed) unless it ends in .h5/.hdf5, whose stem then names the files."""
    stem = os.path.splitext(os.path.basename(args.input.rstrip("/")))[0]
    out = args.output
    if out is not None and out.lower().endswith((".h5", ".hdf5")):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        return os.path.splitext(out)[0]
    d = out or "."
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, stem)


def write_factors(base, ds: Dataset, W, H, rep, args):
    """Write the factors in the dehydrated HDF5 layout plus the run's provenance; optionally plots and denoised data."""
    import h5py
    from mbirtorch.hsnt import export_hsnt_data_hdf5, rehydrate
    R = H.shape[0]
    W4 = W.reshape(*ds.spatial_shape, R)
    out_type = "attenuation" if ds.dataset_type in ("counts", "attenuation") else "transmission"
    h5 = base + "_factors.h5"
    export_hsnt_data_hdf5(h5, [W4, H, out_type], {"dataset_type": out_type, "dataset_modality": "hyperspectral neutron"})
    with h5py.File(h5, "a") as f:
        f.create_dataset("bin_indices", data=ds.bin_indices)
        f.attrs.update(dict(source=ds.source, input_type=ds.dataset_type, rank=R, method=args.method, mode=rep["mode"],
                            spectra=args.spectra, gauge=int(bool(args.gauge)), downsample=args.downsample, wave_bin=args.wave_bin,
                            dose=-1.0 if ds.dose is None else float(ds.dose), loss=rep["loss_final"], mbirtorch_hsnt_cli="1"))
    log.info("wrote %s: subspace_data %s (maps, per material), subspace_basis %s (spectra); rehydrate() reconstructs the %s",
             h5, W4.shape, H.shape, out_type)
    rep_path = base + "_report.json"
    report = dict(input=ds.source, input_type=ds.dataset_type, spatial_shape=list(ds.spatial_shape), pixels=ds.pixels, bins=ds.bins,
                  dose=ds.dose, args={k: v for k, v in vars(args).items() if k not in ("func",)},
                  checks=[dict(level=c.level, message=c.message) for c in ds.checks], info=ds.info, result=rep, outputs=[h5])
    if args.save_denoised:
        den = base + "_denoised.h5"
        export_hsnt_data_hdf5(den, rehydrate([W4, H, out_type]).astype(np.float32), {"dataset_type": out_type, "dataset_modality": "hyperspectral neutron"})
        report["outputs"].append(den)
        log.info("wrote %s (%s, %.2f GiB)", den, out_type, W4.shape[0] * W4.shape[1] * W4.shape[2] * H.shape[1] * 4 / 2**30)
    if not args.no_plots:
        report["outputs"] += write_plots(base, ds, W4, H)
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    log.info("wrote %s", rep_path)
    return report


def _short(source, n=60):
    s = os.path.basename(source.split(":")[0]) + (":" + source.split(":", 1)[1] if ":" in source else "")
    return s if len(s) <= n else "..." + s[-n:]


def write_plots(base, ds, W4, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = H.shape[0]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for r in range(R):
        ax.plot(ds.bin_indices, H[r], color=colors[r % len(colors)], lw=1.4, label=f"component {r}")
    ax.set_xlabel("source wavelength index"); ax.set_ylabel("attenuation per unit map value"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"fitted spectra, rank {R}: {_short(ds.source)}", fontsize=11)
    p1 = base + "_spectra.png"; fig.savefig(p1, dpi=130); plt.close(fig)
    V = W4.shape[0]
    fig, axes = plt.subplots(V, R, figsize=(3.2 * R, 3.2 * V), squeeze=False, constrained_layout=True)
    for v in range(V):
        for r in range(R):
            im = axes[v, r].imshow(W4[v, :, :, r], cmap="magma"); axes[v, r].set_title(f"view {v}, component {r}" if V > 1 else f"component {r}", fontsize=10)
            axes[v, r].axis("off"); fig.colorbar(im, ax=axes[v, r], fraction=0.046)
    fig.suptitle(f"material maps (W): {_short(ds.source)}", fontsize=11)
    p2 = base + "_maps.png"; fig.savefig(p2, dpi=110); plt.close(fig)
    log.info("wrote %s and %s", p1, p2)
    return [p1, p2]


# ----------------------------------------------------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------------------------------------------------
def cmd_inspect(args):
    ds = load_dataset(args)
    st = ds.info["T_stats"]
    print(f"\n{ds.source}\n  type {ds.dataset_type}; {ds.spatial_shape[0]} view(s) x {ds.spatial_shape[1]} x {ds.spatial_shape[2]} pixels "
          f"x {ds.bins} bins (source bins {ds.bin_indices[0]}..{ds.bin_indices[-1]}); {ds.T.nbytes / 2**30:.2f} GiB as float32")
    print(f"  T: min {st['min']:.4g}  median {st['median']:.4g}  mean {st['mean']:.4g}  max {st['max']:.4g}; zeros {st['zero']:.2%}, "
          f"above 1: {float(np.mean(ds.T > 1)):.2%}" + ("  (sampled)" if st["sampled"] else ""))
    print(f"  dose: {'unknown' if ds.dose is None else f'{ds.dose:.4g} counts per pixel and bin'}")
    for c in ds.checks:
        print(f"  [{c.level:5s}] {c.message}")
    if args.estimate_rank:
        from mbirtorch.hsnt.denoise import _estimate_subspace_dimension
        A = -np.log(np.clip(ds.T, np.finfo(np.float32).tiny, None))
        sub = A[:: max(1, ds.pixels // 65536)]
        n = _estimate_subspace_dimension(sub, safety_factor=1, random_state=args.seed, verbose=0)
        print(f"  estimated signal subspace dimension (log-singular-value fit on {sub.shape[0]} pixels): {n}; the material count is at most this")
    try:
        import torch
        for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
            plan_memory(ds, dev, "auto", None)
    except Exception as e:  # planning is advisory
        log.debug("memory plan skipped: %s", e)
    print()
    return 0


def cmd_convert(args):
    from mbirtorch.hsnt import export_hsnt_data_hdf5
    ds = load_dataset(args)
    out = args.output
    if out is None:
        out = os.path.splitext(args.input.rstrip("/"))[0] + ".h5"
    out_type = "attenuation" if args.as_type == "attenuation" else "transmission"
    if out_type == "attenuation":
        with np.errstate(divide="ignore"):
            data = -np.log(ds.T)
        n_inf = int(np.isinf(data).sum())
        if n_inf:
            log.warning("%d zero-count entries become inf in the attenuation (they are exact zeros in transmission); the factorizer "
                        "maps them back to zero transmission", n_inf)
    else:
        data = ds.T
    data = data.reshape(*ds.spatial_shape, ds.bins).astype(np.float32)
    meta = {"dataset_type": out_type, "dataset_modality": "hyperspectral neutron"}
    export_hsnt_data_hdf5(out, data, meta)
    import h5py
    with h5py.File(out, "a") as f:
        f.create_dataset("bin_indices", data=ds.bin_indices)
        if ds.dose is not None:
            f.attrs["dose"] = float(ds.dose)
        f.attrs.update(source=ds.source, input_type=ds.dataset_type, downsample=args.downsample, wave_bin=args.wave_bin)
    log.info("wrote %s: data %s %s, %.2f GiB", out, data.shape, out_type, data.nbytes / 2**30)
    print(out)
    return 0


def cmd_factorize(args):
    if args.rank < 1:
        raise SystemExit("--rank must be at least 1")
    ds = load_dataset(args)
    device = _device(args.device)
    if args.dry_run:
        plan_memory(ds, device, args.mode, args.chunk_pixels)
        print("dry run: data loaded and checked, no solve. Output base:", _out_paths(args, ds))
        return 0
    W, H, rep = solve(ds, args, device)
    base = _out_paths(args, ds)
    write_factors(base, ds, W, H, rep, args)
    n_err = sum(c.level == "error" for c in ds.checks)
    print(f"done: rank {args.rank}, {ds.pixels:,} pixels x {ds.bins} bins, {rep['mode']} solve in {rep['solve_seconds']} s, "
          f"loss {rep['loss_final']:.6g}; outputs at {base}_*" + (f"; {n_err} data check(s) had errors" if n_err else ""))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="mbirtorch-hsnt", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Examples:\n"
                                       "  mbirtorch-hsnt inspect data.h5\n"
                                       "  mbirtorch-hsnt inspect sample_tifs/ --open-beam open_beam/ --estimate-rank\n"
                                       "  mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5\n"
                                       "  mbirtorch-hsnt factorize sample.h5 --rank 3 -o results/\n"
                                       "  mbirtorch-hsnt factorize sample_tifs/ --open-beam open_beam/ --rank 2 --downsample 2 --wave-bin 4 --gauge -v\n")
    sub = p.add_subparsers(dest="command", required=True)

    def add_input(sp):
        sp.add_argument("input", help="HDF5 file (package layout) or a directory with one TIFF per wavelength bin")
        g = sp.add_argument_group("input interpretation")
        g.add_argument("--open-beam", nargs="+", metavar="DIR", help="open-beam TIFF stack(s) for a stack of counts; a directory of "
                       "observation subdirectories is averaged over them")
        g.add_argument("--input-type", choices=("auto",) + _TYPES, default="auto", help="what the values are (default: infer)")
        g.add_argument("--dataset", help="HDF5 group holding 'data' (default: the root, or the only group that has one)")
        g.add_argument("--dose", type=float, help="open-beam counts per pixel and bin, when the input is not counts with an open beam")
        g = sp.add_argument_group("selection")
        g.add_argument("--views", help="view slice START:STOP for 4-D HDF5 data (default: all)")
        g.add_argument("--wave-range", help="spectral slice START:STOP over the source bins (default: all)")
        g.add_argument("--wave-bin", type=int, default=1, metavar="N", help="group N adjacent bins (sum counts / average transmissions)")
        g.add_argument("--downsample", type=int, default=1, metavar="S", help="keep every S-th row and column")
        g = sp.add_argument_group("checks and logging")
        g.add_argument("--strict", action="store_true", help="stop if any data check reports an error")
        g.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO (default), -vv for DEBUG")
        g.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
        g.add_argument("--log-file", help="also write the log here")
        g.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("inspect", help="load, check and describe a dataset (no solve)")
    add_input(s)
    s.add_argument("--estimate-rank", action="store_true", help="fit the signal-subspace dimension to the singular values")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("convert", help="write a TIFF stack (and open beam) as an HDF5 dataset in the package layout")
    add_input(s)
    s.add_argument("-o", "--output", help="output .h5 (default: <input>.h5)")
    s.add_argument("--as-type", choices=("attenuation", "transmission"), default="attenuation", help="stored quantity (default attenuation)")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("factorize", help="fit the NNAL factorization and write maps, spectra, plots and a report")
    add_input(s)
    g = s.add_argument_group("model")
    g.add_argument("--rank", "-r", type=int, required=True, help="number of materials (see 'inspect --estimate-rank' for an upper bound)")
    g.add_argument("--method", choices=("joint_newton", "block_newton", "multiplicative", "lbfgsb"), default="joint_newton")
    g.add_argument("--max-steps", type=int, default=300)
    g.add_argument("--rel-tol", type=float, default=1e-6, help="relative loss change per step at which to stop")
    g.add_argument("--spectra", choices=("mle", "unconstrained", "support"), default="mle",
                   help="spectra estimator: maximum likelihood, the unconstrained-W re-estimate (pays above ~65k pixels), or per-pixel "
                        "support selection (needs the dose, rank <= 6)")
    g.add_argument("--gauge", action="store_true", help="pure-pixel gauge fix of the maps (assumes every material has pure pixels; needs the dose)")
    g = s.add_argument_group("compute")
    g.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    g.add_argument("--mode", choices=("auto", "full", "stream"), default="auto", help="full solve on the device or streamed by chunks (default: by free memory)")
    g.add_argument("--chunk-pixels", type=int, help="pixels per chunk in stream mode (default: from free memory)")
    g.add_argument("--max-passes", type=int, default=5, help="stream mode: polish passes over the data")
    g.add_argument("--warmup-pixels", type=int, default=16384, help="stream mode: pixels for the initial spectra fit")
    g.add_argument("--dry-run", action="store_true", help="load, check and plan, then stop")
    g = s.add_argument_group("output")
    g.add_argument("-o", "--output", help="output directory (created if needed; default: current directory), or a .h5 path whose stem names the files")
    g.add_argument("--save-denoised", action="store_true", help="also write the rehydrated data (as large as the input)")
    g.add_argument("--no-plots", action="store_true")
    s.set_defaults(func=cmd_factorize)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose >= 2 else logging.INFO)
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", handlers=handlers, force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
