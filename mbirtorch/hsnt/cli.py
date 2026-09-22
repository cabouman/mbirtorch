"""Command-line interface for the hsnt dehydration: ``mbirtorch-hsnt`` (or ``python -m mbirtorch.hsnt``).

Five subcommands share one loader and one set of data checks:

    inspect    load a dataset, run the checks, print what a solve would see (no GPU needed)
    convert    write a TIFF stack (with its open beam) or an HDF5 dataset to the package's HDF5 layout, streamed by blocks of bins
    dehydrate  fit the NNAL factorization X = W H and write it in the dehydrated layout, with plots and a JSON report
    rehydrate  multiply a dehydrated file back into hyperspectral data (all bins or a range of them)
    denoise    dehydrate and rehydrate in one run: write the denoised hyperspectral data (and the dehydrated file)

The rank (number of materials) is estimated by likelihood-ratio tests unless --rank gives it.

Inputs are either an HDF5 file in the package layout (``data`` with the spectral axis last, ``dataset_type``,
optionally inside a group) or a directory of one TIFF image per wavelength bin. A TIFF stack of counts needs an
open-beam stack (``--open-beam``) to become a transmission ratio; a stack that already holds transmissions or
attenuations is used as is, and the type is inferred from the values unless ``--input-type`` says otherwise.

The dehydrated layout (``subspace_data`` = maps, ``subspace_basis`` = spectra, ``dataset_type``) is the one
``import_hsnt_data_hdf5`` reads and ``rehydrate`` reconstructs from. Run any subcommand with ``-h`` for the options.
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

from .rank import _lrt_rank, pool_pixels, estimate_rank as _estimate_rank

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


def _summary_from_T(T, dose):
    """The quantities the data checks need, computed from a transmission matrix held in memory."""
    st = _stats(T, "T")
    P, K = T.shape
    above = _frac(T > 1) if T.size <= 4_000_000 else _frac(T.reshape(-1)[:: max(1, T.size // 2_000_000)] > 1)
    pos = T > 0
    dead_bins = int((pos.sum(0) == 0).sum())
    return dict(pixels=P, bins=K, nbytes=T.nbytes, stats=st, above_one=above, dead_px=_frac(pos.sum(1) == 0), dead_bins=dead_bins,
                const_bins=int((T.std(0) == 0).sum()) - dead_bins, dose=dose)


def _checks_from_summary(sm, spatial_shape, checks, strict=False):
    """Append the standard checks for a summary (from `_summary_from_T` or the streaming accumulators) and raise if
    strict and any is an error."""
    st, c = sm["stats"], checks
    P, K = sm["pixels"], sm["bins"]
    c.append(Check("ok", f"{P:,} pixels x {K:,} bins ({spatial_shape[0]} view(s) x {spatial_shape[1]} x {spatial_shape[2]}), "
                         f"{sm['nbytes'] / 2**30:.2f} GiB as float32"))
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
            c.append(Check("warn", f"open-beam dose {dose:.3g} counts per pixel and bin is below 1: expect mostly zero counts"))
        else:
            c.append(Check("ok", f"dose {dose:.3g} open-beam counts per pixel and (binned) bin"))
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


def run_checks(ds: Dataset, strict=False):
    """Append the standard checks to ds.checks and raise if strict and any is an error."""
    sm = _summary_from_T(ds.T, ds.dose)
    ds.info["T_stats"] = sm["stats"]
    return _checks_from_summary(sm, ds.spatial_shape, ds.checks, strict)


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
                         f"--downsample, --wave-range or --wave-bin (`mbirtorch-hsnt convert` streams and needs no such memory)")
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


def _to_transmission(a, input_type, open_beam=None, wave_bin=1, quiet=False):
    """Convert a stack of the given type to a transmission ratio, binning bins if asked. Returns (T, dose, info).
    quiet=True (a block of a streamed conversion) skips the per-call logging and the count statistics."""
    info = {}
    if wave_bin > 1 and not quiet:
        log.info("--wave-bin %d: %d source bins -> %d (dropping the last %d)", wave_bin, a.shape[-1], a.shape[-1] // wave_bin, a.shape[-1] % wave_bin)
    if input_type == "counts":
        if open_beam is None:
            raise SystemExit("a stack of counts needs an open beam: pass --open-beam DIR (or --input-type if the values "
                             "are already transmissions or attenuations)")
        counts, ob = _bin_spectral(a, wave_bin, "sum"), _bin_spectral(open_beam, wave_bin, "sum")
        bad = ob <= 0
        if bad.any():
            info["open_beam_zero_frac"] = _frac(bad)
            if not quiet:
                log.warning("open beam is zero or negative in %.3g%% of pixel-bins; those use the bin's median open beam", 100 * _frac(bad))
            stride = max(1, ob.shape[0] // 8192)                                        # per-bin medians on a pixel subsample
            sub = np.where(bad[::stride], np.nan, ob[::stride])
            med = np.nanmedian(sub, axis=0)
            med = np.nan_to_num(med, nan=float(np.nanmedian(sub)) if np.isfinite(sub).any() else 1.0)
            ob = np.where(bad, med[None, :], ob)
        T = counts / ob
        dose = float(np.median(ob.reshape(-1)[:: max(1, ob.size // 1_000_000)]))
        if not quiet:
            info["counts_stats"] = _stats(counts, "counts")
    elif input_type == "transmission":
        T, dose = _bin_spectral(a, wave_bin, "mean"), None
    elif input_type == "attenuation":
        A = a
        nonfinite = ~np.isfinite(A)
        if nonfinite.any():
            info["attenuation_nonfinite_frac"] = _frac(nonfinite)
            if not quiet:
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


def _attenuation_sample(ds: Dataset, max_pixels=65536):
    """Attenuation on a strided pixel subsample, with zero counts floored at half the smallest positive transmission."""
    T = ds.T[:: max(1, ds.pixels // max_pixels)]
    pos = T[T > 0]
    floor = 0.5 * float(pos.min()) if pos.size else np.finfo(np.float32).tiny
    return -np.log(np.maximum(T, floor)), floor


def estimate_rank(ds: Dataset, device, seed=0, max_rank=6, subsample=16384, pool="auto"):
    """:func:`mbirtorch.hsnt.estimate_rank` on a loaded Dataset (pixels pooled within its views x rows x cols)."""
    rank, note, detail = _estimate_rank(ds.T, ds.spatial_shape, device, seed=seed, max_rank=max_rank, subsample=subsample, pool=pool)
    return rank, note.replace("give the rank to override", "pass --rank N to override"), detail


def fit_quality(ds: Dataset, W, H, device, chunk=65536):
    """Reduced chi-square of the fit against Poisson noise: mean of dose (T - e^-X)^2 / e^-X over the data.
    Near 1 the residual is at the noise level; well above 1 the rank is too small or the model is wrong; well below 1
    the fit follows the noise. Needs the dose; without it returns the relative residual in transmission instead."""
    import torch
    W = torch.as_tensor(W, device=device); H = torch.as_tensor(H, device=device)
    num = den = res = tot = 0.0
    for i in range(0, ds.pixels, chunk):
        T = torch.from_numpy(ds.T[i:i + chunk]).to(device).double()
        Th = torch.exp(-(W[i:i + chunk].double() @ H.double()))
        d = T - Th
        res += (d * d).sum().item(); tot += (T * T).sum().item()
        num += ((d * d) / Th.clamp_min(1e-12)).sum().item(); den += T.numel()
    out = dict(relative_residual=float(np.sqrt(res / tot)))
    if ds.dose is not None:
        out["reduced_chi2"] = ds.dose * num / den
    return out


def component_check(W, H, corr_warn=0.8):
    """Are the components distinguishable? Two components whose maps are nearly proportional cannot have their spectra
    told apart by the data: only the weighted sum of their rows is determined, and each row on its own is an arbitrary,
    noisy slice of it (on a one-material sample every extra component behaves so, and the 'material' row gets noisier
    with every one added: 3x at rank 2, 13x at rank 10). Returns a dict with the maximum correlation, the pairs above
    corr_warn, and the per-row noise level (from second differences, relative to the row's median)."""
    W = np.asarray(W, dtype=np.float64); H = np.asarray(H, dtype=np.float64); R = H.shape[0]
    out = dict(max_map_correlation=0.0, proportional_pairs=[], row_noise_rel=[])
    for k in range(R):
        h = H[k]; lvl = float(np.median(h)) if np.median(h) > 0 else float(h.max()) or 1.0
        out["row_noise_rel"].append(float(np.std(np.diff(h, 2)) / np.sqrt(6) / lvl) if h.size > 3 else 0.0)
    if R < 2:
        return out
    C = np.corrcoef(W.T); np.fill_diagonal(C, 0.0); C = np.nan_to_num(C)
    out["max_map_correlation"] = float(C.max())
    out["proportional_pairs"] = [(int(i), int(j), round(float(C[i, j]), 3)) for i in range(R) for j in range(i + 1, R) if C[i, j] > corr_warn]
    return out


def mean_pixel_spectrum(W, H, frac=0.25):
    """Attenuation of the average material pixel, sum_k mean(W_pk) H_k over pixels with material, and each component's
    share of it. Unlike the rows of H it does not depend on how the solver split the spectrum among components, so it
    shows the Bragg edges at any rank. Pixels count as material when their total map value exceeds frac of the 99th
    percentile. Returns (total, contributions (R, K), number of pixels used)."""
    tot = W.sum(1); mat = tot > frac * np.percentile(tot, 99)
    if mat.sum() < 10:
        mat = np.ones_like(mat)
    wm = W[mat].mean(0)
    return wm @ H, wm[:, None] * H, int(mat.sum())


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
    rank = args.rank_value
    mode, chunk, rep["memory_plan"] = plan_memory(ds, device, args.mode, args.chunk_pixels)
    rep["mode"], rep["rank"], rep["rank_note"], rep["rank_search"] = mode, rank, args.rank_note, args.rank_detail
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    if mode == "full":
        T = torch.from_numpy(ds.T).to(device)
        W, H, steps = nnal_factorization(T, method=args.method, num_materials=rank, max_steps=args.max_steps,
                                         rel_tol=args.rel_tol, random_state=args.seed)
        rep["steps"] = int(steps)
    else:
        if args.method != "joint_newton":
            log.warning("stream mode always uses joint_newton for the warm-up and block Newton for the polish; --method %s ignored", args.method)
        chunks = [torch.from_numpy(ds.T[i:i + chunk]) for i in range(0, ds.pixels, chunk)]
        stats = {}
        W_chunks, H, passes = stream_factorization(chunks, rank, max_passes=args.max_passes, rel_tol=args.rel_tol,
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
        if rank > 6:
            raise SystemExit("support selection enumerates all 2^R - 1 subsets and is limited to rank 6")
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
        sizes = [int((labels == k).sum()) for k in range(rank)]
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


def _out_type(ds, args):
    given = getattr(args, "as_type", None)
    if given:
        return given
    return "attenuation" if ds.dataset_type in ("counts", "attenuation") else "transmission"


def _run_attrs(ds, rep, args, extra):
    """Provenance written as HDF5 attributes: where the data came from and how the solve was set up."""
    return dict(source=ds.source, input_type=ds.dataset_type, method=args.method, mode=rep["mode"], spectra=args.spectra,
                gauge=int(bool(args.gauge)), downsample=args.downsample, wave_bin=args.wave_bin,
                dose=-1.0 if ds.dose is None else float(ds.dose), mbirtorch_hsnt_cli="1", **extra)


def _provenance(h5, bin_indices, attrs):
    import h5py
    with h5py.File(h5, "a") as f:
        if "bin_indices" not in f:
            f.create_dataset("bin_indices", data=np.asarray(bin_indices))
        f.attrs.update(attrs)


def write_dehydrated(base, ds: Dataset, W, H, rep, args):
    """Write the factorization in the dehydrated HDF5 layout plus the run's provenance. Returns the path."""
    from mbirtorch.hsnt import export_hsnt_data_hdf5
    R = H.shape[0]
    W4 = W.reshape(*ds.spatial_shape, R)
    out_type = _out_type(ds, args)
    h5 = base + "_dehydrated.h5"
    export_hsnt_data_hdf5(h5, [W4, H, out_type], {"dataset_type": out_type, "dataset_modality": "hyperspectral neutron"})
    _provenance(h5, ds.bin_indices, _run_attrs(ds, rep, args, dict(rank=R, loss=rep["loss_final"])))
    total, contrib, n_mat = mean_pixel_spectrum(W, H)
    import h5py
    with h5py.File(h5, "a") as f:
        d = f.create_dataset("mean_pixel_spectrum", data=total.astype(np.float32))
        d.attrs["description"] = f"attenuation of the average material pixel ({n_mat} pixels), sum_k mean(W_pk) H_k; independent of the split among components"
        f.create_dataset("mean_pixel_contributions", data=contrib.astype(np.float32))
    log.info("wrote %s: subspace_data %s (maps, per material), subspace_basis %s (spectra), mean_pixel_spectrum over %s pixels; "
             "`mbirtorch-hsnt rehydrate` or rehydrate() reconstructs the %s", h5, W4.shape, H.shape, f"{n_mat:,}", out_type)
    return h5


def write_denoised(path, spatial_shape, W, H, out_type, bin_indices, attrs, chunk=16384):
    """Rehydrate W @ H into the package's hyperspectral HDF5 layout, written by pixel blocks so the full array is
    never held in memory; import_hsnt_data_hdf5 reads it back. W is (pixels, rank), spatial_shape (views, rows, cols)."""
    import h5py
    R, K = H.shape
    V, rows, cols = spatial_shape
    pixels = V * rows * cols
    with h5py.File(path, "w") as f:
        d = f.create_dataset("data", shape=(V, rows, cols, K), dtype=np.float32, chunks=(1, min(rows, 64), cols, K))
        for i in range(0, pixels, chunk):
            X = W[i:i + chunk] @ H
            block = (np.exp(-X) if out_type == "transmission" else X).astype(np.float32)
            p0, p1 = i, min(i + chunk, pixels)                              # pixel block -> (view, row, col) coordinates
            idx = np.arange(p0, p1)
            v, rc = np.divmod(idx, rows * cols); r, c = np.divmod(rc, cols)
            if v[0] == v[-1] and c[0] == 0 and c[-1] == cols - 1:
                d[v[0], r[0]:r[-1] + 1, :, :] = block.reshape(r[-1] - r[0] + 1, cols, K)
            else:
                for k in range(len(idx)):
                    d[v[k], r[k], c[k], :] = block[k]
        f.create_dataset("dataset_type", data=np.bytes_(out_type))
        f.create_dataset("dataset_modality", data=np.bytes_("hyperspectral neutron"))
    _provenance(path, bin_indices, dict(attrs, rank=R, rehydrated="1"))
    log.info("wrote %s: data %s %s, %.2f GiB", path, (V, rows, cols, K), out_type, V * rows * cols * K * 4 / 2**30)
    return path


def write_report(base, ds, rep, args, outputs):
    rep_path = base + "_report.json"
    report = dict(input=ds.source, input_type=ds.dataset_type, spatial_shape=list(ds.spatial_shape), pixels=ds.pixels, bins=ds.bins,
                  dose=ds.dose, args={k: v for k, v in vars(args).items() if k not in ("func",)},
                  checks=[dict(level=c.level, message=c.message) for c in ds.checks], info=ds.info, result=rep, outputs=outputs)
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
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#8B4513", "#555555", "#7F00FF"]
    W2 = W4.reshape(-1, R); total, contrib, n_mat = mean_pixel_spectrum(W2, H)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    for r in range(R):
        ax.plot(ds.bin_indices, H[r], color=colors[r % len(colors)], lw=1.2, label=f"component {r}")
    ax.set_ylabel("attenuation per unit map value"); ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=2 if R > 5 else 1)
    ax.set_title(f"rows of H, rank {R}: {_short(ds.source)}\n(when maps are proportional, the split among rows is arbitrary)", fontsize=11)
    ax2.plot(ds.bin_indices, total, color="black", lw=1.6, label=f"total, average of {n_mat:,} material pixels")
    for r in range(R):
        ax2.plot(ds.bin_indices, contrib[r], color=colors[r % len(colors)], lw=1.0, alpha=0.9, label=f"component {r} share")
    ax2.set_xlabel("source wavelength index"); ax2.set_ylabel("attenuation of the average material pixel"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, ncol=2 if R > 5 else 1); ax2.set_title("mean-pixel spectrum: independent of the split among components", fontsize=11)
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
# Streaming conversion: bins in blocks, never the whole stack
# ----------------------------------------------------------------------------------------------------------------------
class _TiffBlocks:
    """Bins k0:k1 of a TIFF stack (one image per bin) as (1, rows, cols, k1 - k0) float32, decoded in parallel."""

    def __init__(self, directory, wave_range, downsample, workers, desc):
        import tifffile
        files, subdirs = _tif_files(directory)
        if files is None:
            raise FileNotFoundError(f"{directory} holds only subdirectories ({len(subdirs)}); pass one of them, or pass the "
                                    f"parent as --open-beam to average them")
        self.all_files = files
        self.files = files[slice(*wave_range)] if wave_range else files
        self.first_index = files.index(self.files[0])
        with tifffile.TiffFile(self.files[0]) as t:
            page = t.pages[0]
            self.full_shape, self.source_dtype = tuple(page.shape), str(page.dtype)
        if len(self.full_shape) != 2:
            raise ValueError(f"{self.files[0]}: expected a 2-D image per wavelength bin, got shape {self.full_shape}")
        self.downsample, self.workers, self.desc = downsample, workers, desc
        self.rows, self.cols = np.empty(self.full_shape, dtype=bool)[::downsample, ::downsample].shape
        self.views = 1

    @property
    def bins(self):
        return len(self.files)

    def read(self, k0, k1):
        import tifffile
        sel = self.files[k0:k1]
        if len(sel) == 1:
            arr = tifffile.imread(sel[0])[None]
        else:
            try:
                arr = tifffile.imread(sel, ioworkers=self.workers, maxworkers=1)
            except TypeError:                                                          # older tifffile: no worker arguments
                arr = np.stack([tifffile.imread(f) for f in sel])
        if arr.ndim != 3 or tuple(arr.shape[1:]) != self.full_shape:
            raise ValueError(f"{self.desc}: images {k0}..{k1 - 1} have shape {arr.shape[1:]}, the first image {self.full_shape}")
        arr = arr[:, ::self.downsample, ::self.downsample]
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1), dtype=np.float32)[None]   # (1, rows, cols, b)

    def close(self):
        pass


class _Hdf5Blocks:
    """Bins k0:k1 of an HDF5 dataset in the package layout as (views, rows, cols, k1 - k0) float32."""

    def __init__(self, path, dataset, views, wave_range, downsample):
        import h5py
        self.f = h5py.File(path, "r")
        g, self.gname = _find_h5_dataset(self.f, dataset)
        self.d = g["data"]
        self.dtype_str = g["dataset_type"][()] if "dataset_type" in g else None
        if isinstance(self.dtype_str, (bytes, np.bytes_)):
            self.dtype_str = self.dtype_str.decode()
        d = self.d
        if d.ndim == 2:
            self.sel, shape = (slice(None),), (1, d.shape[0], 1)
        elif d.ndim == 3:
            self.sel = (slice(None, None, downsample), slice(None, None, downsample))
            shape = (1,) + np.empty(d.shape[:2], dtype=bool)[self.sel].shape
        elif d.ndim == 4:
            v = slice(*views) if views else slice(None)
            self.sel = (v, slice(None, None, downsample), slice(None, None, downsample))
            shape = np.empty(d.shape[:3], dtype=bool)[self.sel].shape
        else:
            raise SystemExit(f"data has {d.ndim} dimensions; expected (views, rows, cols, bins), (rows, cols, bins) or (pixels, bins)")
        self.views, self.rows, self.cols = shape
        self.full_shape = tuple(d.shape[1:3]) if d.ndim == 4 else tuple(d.shape[:2]) if d.ndim == 3 else (d.shape[0], 1)
        self.source_dtype = str(d.dtype)
        ks = range(d.shape[-1])[slice(*wave_range)] if wave_range else range(d.shape[-1])
        self.first_index, self.bins = (ks[0], len(ks)) if len(ks) else (0, 0)
        self.desc = f"{path}:{self.gname}"
        log.info("HDF5 %s: dataset %s/data shape %s dtype %s chunks %s", path, self.gname.rstrip("/"), d.shape, d.dtype, d.chunks)

    def read(self, k0, k1):
        a = self.d[self.sel + (slice(self.first_index + k0, self.first_index + k1),)]
        if self.d.ndim == 3:
            a = a[None]
        elif self.d.ndim == 2:
            a = a[None, :, None, :]
        return np.ascontiguousarray(a, dtype=np.float32)

    def close(self):
        self.f.close()


def _open_beam_blocks(paths, wave_range, downsample, workers, sample):
    """One block reader per open-beam observation (a directory of observation subdirectories expands to all of them),
    checked against the sample's geometry."""
    dirs = []
    for p in paths:
        files, subdirs = _tif_files(p)
        dirs += subdirs if files is None else [p]
    obs = [_TiffBlocks(d, wave_range, downsample, workers, desc=f"open beam {os.path.basename(d)}") for d in dirs]
    for o in obs:
        if (o.bins, o.rows, o.cols) != (sample.bins, sample.rows, sample.cols):
            raise ValueError(f"{o.desc} has {o.bins} bins of {o.full_shape}, the sample {sample.bins} of {sample.full_shape}")
    return obs


def _block_bins(sample, n_obs, wave_bin, budget_mib, requested):
    """Bins per block: the largest multiple of wave_bin whose working set (two blocks, the processed and the prefetched,
    of sample and open-beam mean, one observation read in flight at full resolution before downsampling, the
    transmission and the output) fits the budget."""
    if requested:
        block = max(wave_bin, requested // wave_bin * wave_bin)
    else:
        full = int(np.prod(sample.full_shape)) * sample.views * 4                      # bytes per bin as read
        small = sample.rows * sample.cols * sample.views * 4                           # bytes per bin after downsampling
        # two blocks alive (the one processed and the one prefetched), each the sample plus the open-beam mean, one
        # observation read in flight at full resolution, and the transmission, its checks and the output block
        per_bin = full * (2 + (3 if n_obs else 0)) + small * 4
        block = int(budget_mib * 2**20 // per_bin) // wave_bin * wave_bin
        block = max(wave_bin, block)
    return min(block, sample.bins // wave_bin * wave_bin) or wave_bin


def stream_convert(args):
    """Read the input in blocks of bins, normalise, check and write each block to the HDF5 output: memory is a few
    blocks, not the stack. Returns (path, checks, info)."""
    import h5py
    wave_range = _parse_slice(args.wave_range, "wave-range")
    views = _parse_slice(args.views, "views")
    p = args.input
    if not os.path.exists(p):
        raise SystemExit(f"input not found: {p}")
    t0 = time.perf_counter()
    workers = args.workers or min(8, os.cpu_count() or 1)
    if os.path.isdir(p):
        src = _TiffBlocks(p, wave_range, args.downsample, workers, desc="sample")
        dtype_str = None
    elif p.lower().endswith((".h5", ".hdf5", ".hdf")):
        if args.open_beam:
            log.warning("--open-beam is ignored for HDF5 input")
        src = _Hdf5Blocks(p, args.dataset, views, wave_range, args.downsample)
        dtype_str = src.dtype_str
    else:
        raise SystemExit(f"{p}: not a directory of TIFFs and not an .h5/.hdf5 file")
    if src.bins == 0:
        raise SystemExit("the selection holds no bins")
    V, rows, cols, nb = src.views, src.rows, src.cols, src.bins
    P = V * rows * cols
    log.info("%s: %d view(s) x %dx%d pixels%s x %d bins, source dtype %s", src.desc, V, rows, cols,
             f" (downsampled {args.downsample}x from {src.full_shape[0]}x{src.full_shape[1]})" if args.downsample > 1 else "", nb, src.source_dtype)
    # input type from the first bins (or the file's dataset_type)
    probe = src.read(0, min(nb, max(8, args.wave_bin)))
    if args.input_type != "auto":
        itype, why = args.input_type, "given"
        if dtype_str and dtype_str != itype:
            log.warning("--input-type %s overrides the file's dataset_type %r", itype, dtype_str)
    elif dtype_str in ("attenuation", "transmission"):
        itype, why = dtype_str, "from the file's dataset_type"
    else:
        itype, why = infer_input_type(probe)
        why += f" (from the first {probe.shape[-1]} bins)"
    log.info("input type: %s (%s)", itype, why)
    del probe
    obs = []
    if itype == "counts":
        if not (os.path.isdir(p) and args.open_beam):
            raise SystemExit("a stack of counts needs an open beam: pass --open-beam DIR (or --input-type if the values are "
                             "already transmissions or attenuations)")
        obs = _open_beam_blocks(args.open_beam, wave_range, args.downsample, workers, src)
        log.info("open beam: %d observation(s), averaged block by block", len(obs))
    elif args.open_beam and os.path.isdir(p):
        log.warning("--open-beam ignored: the input type is %s, not counts", itype)
    wave_bin = max(1, args.wave_bin)
    block = _block_bins(src, len(obs), wave_bin, args.memory_budget, args.block_bins)
    K = nb // wave_bin
    if K == 0:
        raise SystemExit(f"--wave-bin {wave_bin} exceeds the {nb} selected bins")
    if wave_bin > 1:
        log.info("--wave-bin %d: %d source bins -> %d (dropping the last %d)", wave_bin, nb, K, nb % wave_bin)
    out_type = "attenuation" if args.as_type == "attenuation" else "transmission"
    out = args.output or (os.path.splitext(p.rstrip("/"))[0] + ".h5")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    block_out = block // wave_bin
    log.info("streaming %d bins per block (%d blocks, %d output bins each), %d TIFF reader threads -> %s (%s, %.2f GiB)",
             block, -(-nb // block), block_out, workers, out, out_type, P * K * 4 / 2**30)
    # accumulators for the checks
    tot = dict(n=0, nonfinite=0, negative=0, zero=0, above=0, inf_out=0)
    pos_px = np.zeros(P, dtype=np.int64); pos_bin = np.zeros(K, dtype=np.int64)
    bin_min = np.full(K, np.inf, dtype=np.float32); bin_max = np.full(K, -np.inf, dtype=np.float32)
    stride = max(1, -(-P * K // 4_000_000))
    sample = np.empty((-(-P // stride), K), dtype=np.float32)
    dose_blocks, ob_zero = [], 0.0
    starts = [k0 for k0 in range(0, nb - nb % wave_bin, block) if min(k0 + block, nb) // wave_bin * wave_bin > k0]
    try:
        from tqdm import tqdm
        blocks = tqdm(starts, desc="convert", unit="block", disable=not log.isEnabledFor(logging.INFO), leave=False)
    except ImportError:
        blocks = starts

    def read_block(k0):
        """The sample block and the open-beam mean over observations for bins k0:k1, as (pixels, bins)."""
        k1 = min(k0 + block, nb) // wave_bin * wave_bin
        a = src.read(k0, k1).reshape(P, k1 - k0)
        ob = None
        if obs:
            for o in obs:                                                                # sum in place, divide once
                blk = o.read(k0, k1).reshape(P, k1 - k0)
                if ob is None:
                    ob = blk
                else:
                    ob += blk
                    del blk
            ob /= len(obs)
        return k1, a, ob
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)                                                         # the next block is read while this one is processed
    pending = pool.submit(read_block, starts[0]) if starts else None
    with h5py.File(out, "w") as f:
        d = f.create_dataset("data", shape=(V, rows, cols, K), dtype=np.float32,
                             chunks=(1, min(rows, 128), cols, min(16, K)))
        for n_blk, k0 in enumerate(blocks):
            k1, a, ob = pending.result()
            pending = pool.submit(read_block, starts[n_blk + 1]) if n_blk + 1 < len(starts) else None
            T, dose_b, info_b = _to_transmission(a, itype, open_beam=ob, wave_bin=wave_bin, quiet=True)
            del a, ob
            j0, j1 = k0 // wave_bin, k1 // wave_bin
            if dose_b is not None:
                dose_blocks.append((dose_b, j1 - j0))
            ob_zero += info_b.get("open_beam_zero_frac", 0.0) * (j1 - j0)
            finite = np.isfinite(T)
            tot["n"] += T.size; tot["nonfinite"] += int((~finite).sum()); tot["negative"] += int((T < 0).sum())
            tot["zero"] += int((T == 0).sum()); tot["above"] += int((T > 1).sum())
            pos = T > 0
            pos_px += pos.sum(1); pos_bin[j0:j1] = pos.sum(0)
            bin_min[j0:j1] = T.min(0); bin_max[j0:j1] = T.max(0)
            sample[:, j0:j1] = T[::stride]
            if out_type == "attenuation":
                with np.errstate(divide="ignore"):
                    block_out_data = -np.log(T)
                tot["inf_out"] += int(np.isinf(block_out_data).sum())
            else:
                block_out_data = T
            d[:, :, :, j0:j1] = block_out_data.reshape(V, rows, cols, j1 - j0)
            del T, pos, finite, block_out_data
        pool.shutdown(wait=True)
        src.close()
        for o in obs:
            o.close()
        dose = None
        if dose_blocks:
            vals, w = np.array([v for v, _ in dose_blocks]), np.array([n for _, n in dose_blocks])
            dose = float(np.median(np.repeat(vals, w)))
        if args.dose is not None:
            if dose is not None and abs(args.dose - dose) / dose > 0.5:
                log.warning("--dose %.3g differs from the open-beam estimate %.3g by more than 50%%", args.dose, dose)
            dose = args.dose
        finite_s = sample[np.isfinite(sample)]
        st = dict(min=float(finite_s.min()) if finite_s.size else float("nan"), max=float(finite_s.max()) if finite_s.size else float("nan"),
                  mean=float(finite_s.mean()) if finite_s.size else float("nan"), median=float(np.median(finite_s)) if finite_s.size else float("nan"),
                  nonfinite=tot["nonfinite"] / tot["n"], negative=tot["negative"] / tot["n"], zero=tot["zero"] / tot["n"], sampled=stride > 1)
        dead_bins = int((pos_bin == 0).sum())
        sm = dict(pixels=P, bins=K, nbytes=P * K * 4, stats=st, above_one=tot["above"] / tot["n"], dead_px=_frac(pos_px == 0), dead_bins=dead_bins,
                  const_bins=int((bin_max == bin_min).sum()) - dead_bins, dose=dose)
        checks = _checks_from_summary(sm, (V, rows, cols), [], strict=getattr(args, "strict", False))
        if ob_zero:
            log.warning("open beam is zero or negative in %.3g%% of pixel-bins; those used the block's median open beam", 100 * ob_zero / K)
        if tot["inf_out"]:
            log.warning("%d zero-transmission entries are inf in the attenuation output (exact zeros in transmission); the solver maps "
                        "them back to zero transmission", tot["inf_out"])
        f.create_dataset("dataset_type", data=np.bytes_(out_type))
        f.create_dataset("dataset_modality", data=np.bytes_("hyperspectral neutron"))
        f.create_dataset("bin_indices", data=np.arange(src.first_index, src.first_index + K * wave_bin, wave_bin))
        attrs = dict(source=src.desc if not os.path.isdir(p) else p, input_type=itype, downsample=args.downsample, wave_bin=wave_bin,
                     block_bins=block, mbirtorch_hsnt_cli="1", checks=json.dumps([dict(level=c.level, message=c.message) for c in checks]))
        if dose is not None:
            attrs["dose"] = float(dose)
        if obs:
            attrs["open_beam_observations"] = len(obs)
        f.attrs.update(attrs)
    seconds = time.perf_counter() - t0
    info = dict(seconds=round(seconds, 2), block_bins=block, blocks=-(-nb // block), open_beam_observations=len(obs), stats=st, dose=dose)
    log.info("wrote %s: data %s %s, %.2f GiB, in %.1f s (%.0f bins/s)", out, (V, rows, cols, K), out_type, P * K * 4 / 2**30, seconds, nb / seconds)
    return out, checks, info


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
        pool = args.rank_pool if str(args.rank_pool).lower() == "auto" else int(args.rank_pool)
        n, note, d = estimate_rank(ds, _device(args.device), args.seed, max_rank=args.max_rank, pool=pool)
        print(f"  {note.rsplit(';', 1)[0]}; effective dose {d['effective_dose']:.3g}; gains by component (deciding test): "
              + ", ".join(f"{r}: {g:,.0f}" for r, g in zip(range(2, d['max_rank'] + 1), d['gains'])) + f"; threshold {d['threshold']:,.0f}")
    try:
        import torch
        for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
            plan_memory(ds, dev, "auto", None)
    except Exception as e:  # planning is advisory
        log.debug("memory plan skipped: %s", e)
    print()
    return 0


def cmd_convert(args):
    out, checks, info = stream_convert(args)
    n_err = sum(c.level == "error" for c in checks)
    print(out + (f"  ({n_err} data check(s) had errors)" if n_err else ""))
    return 0


def _resolve_rank(ds, args, device):
    """--rank N or auto: sets args.rank_value and args.rank_note."""
    if str(args.rank).lower() == "auto":
        pool = args.rank_pool if str(args.rank_pool).lower() == "auto" else int(args.rank_pool)
        args.rank_value, args.rank_note, args.rank_detail = estimate_rank(ds, device, args.seed, max_rank=args.max_rank, pool=pool)
        log.info("rank: %s", args.rank_note)
    else:
        args.rank_detail = None
        try:
            args.rank_value = int(args.rank)
        except ValueError:
            raise SystemExit(f"--rank expects an integer or 'auto', got {args.rank!r}")
        if args.rank_value < 1:
            raise SystemExit("--rank must be at least 1")
        args.rank_note = f"rank {args.rank_value} given"


def _pipeline(args, denoise):
    """load -> checks -> rank -> solve -> fit quality -> outputs, for dehydrate and denoise."""
    ds = load_dataset(args)
    device = _device(args.device)
    _resolve_rank(ds, args, device)
    base = _out_paths(args, ds)
    if args.dry_run:
        plan_memory(ds, device, args.mode, args.chunk_pixels)
        print(f"dry run: data loaded and checked, {args.rank_note}; no solve. Output base: {base}")
        return 0
    W, H, rep = solve(ds, args, device)
    rep["fit"] = fit_quality(ds, W, H, device)
    rep["components"] = comp = component_check(W, H)
    if comp["proportional_pairs"]:
        worst = max(comp["proportional_pairs"], key=lambda p: p[2])
        log.warning("components %d and %d have nearly proportional maps (correlation %.2f; %d such pair(s)): the data cannot tell "
                    "their spectra apart, so each row of H is an arbitrary noisy slice of their sum. The rank is probably above the "
                    "number of distinct materials; %s. The mean-pixel spectrum in the outputs is unaffected.",
                    worst[0], worst[1], worst[2], len(comp["proportional_pairs"]),
                    "the estimate chose it" if args.rank_detail else "run without --rank to estimate it")
    log.info("components: max map correlation %.2f; per-row noise (rel. to level) %s", comp["max_map_correlation"],
             ", ".join(f"{x:.3f}" for x in comp["row_noise_rel"]))
    q = rep["fit"]
    if "reduced_chi2" in q:
        chi2 = q["reduced_chi2"]
        verdict = ("at the Poisson noise level" if 0.8 <= chi2 <= 1.3 else
                   "above the noise level: the rank may be too small or the model misspecified" if chi2 > 1.3 else
                   "below the noise level: the fit follows the noise (rank too large, or the dose is overestimated)")
        log.info("fit: reduced chi-square %.3f (%s); relative residual in transmission %.4g", chi2, verdict, q["relative_residual"])
        if not 0.5 <= chi2 <= 2.0:
            log.warning("reduced chi-square %.2f is far from 1; check the rank (%s) and the dose", chi2, args.rank_note)
    else:
        log.info("fit: relative residual in transmission %.4g (no dose, so no chi-square)", q["relative_residual"])
    outputs = []
    if not denoise or not args.no_dehydrated:
        outputs.append(write_dehydrated(base, ds, W, H, rep, args))
    if denoise or args.rehydrate:
        outputs.append(write_denoised(base + "_denoised.h5", ds.spatial_shape, W, H, _out_type(ds, args), ds.bin_indices,
                                      _run_attrs(ds, rep, args, dict(loss=rep["loss_final"]))))
    if not args.no_plots:
        outputs += write_plots(base, ds, W.reshape(*ds.spatial_shape, H.shape[0]), H)
    write_report(base, ds, rep, args, outputs)
    n_err = sum(c.level == "error" for c in ds.checks)
    print(f"done: {args.rank_note.split(';')[0]}, {ds.pixels:,} pixels x {ds.bins} bins, {rep['mode']} solve in {rep['solve_seconds']} s, "
          f"loss {rep['loss_final']:.6g}" + (f", reduced chi-square {q['reduced_chi2']:.2f}" if "reduced_chi2" in q else "")
          + f"; outputs at {base}_*" + (f"; {n_err} data check(s) had errors" if n_err else ""))
    return 0


def cmd_dehydrate(args):
    return _pipeline(args, denoise=False)


def cmd_denoise(args):
    return _pipeline(args, denoise=True)


def cmd_rehydrate(args):
    """Multiply a dehydrated file back into hyperspectral data, for all bins or a --wave-range of them."""
    import h5py
    from mbirtorch.hsnt import import_hsnt_data_hdf5
    if not os.path.isfile(args.input):
        raise SystemExit(f"input not found: {args.input}")
    data, meta = import_hsnt_data_hdf5(args.input)
    if not isinstance(data, list):
        raise SystemExit(f"{args.input}: not a dehydrated file (needs subspace_data, subspace_basis and dataset_type); "
                         "run `mbirtorch-hsnt dehydrate` first")
    W4, H, dtype = data
    if W4.ndim == 3:
        W4 = W4[None]
    if W4.ndim != 4:
        raise SystemExit(f"subspace_data has shape {W4.shape}; expected (views, rows, cols, rank) or (rows, cols, rank)")
    if W4.shape[-1] != H.shape[0]:
        raise SystemExit(f"rank mismatch: subspace_data has {W4.shape[-1]} components, subspace_basis {H.shape[0]} rows")
    with h5py.File(args.input, "r") as f:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        bin_indices = f["bin_indices"][()] if "bin_indices" in f else np.arange(H.shape[1])
    views = _parse_slice(args.views, "views")
    if views:
        W4 = W4[slice(*views)]
    wave = _parse_slice(args.wave_range, "wave-range")
    if wave:
        H = H[:, slice(*wave)]; bin_indices = bin_indices[slice(*wave)]
        if H.shape[1] == 0:
            raise SystemExit(f"--wave-range {args.wave_range} selects no bins of the {data[1].shape[1]} in the file")
    V, rows, cols, R = W4.shape
    out_type = args.as_type or dtype
    log.info("%s: %d component(s), %d view(s) x %d x %d pixels, %d of %d bins -> %s", args.input, R, V, rows, cols, H.shape[1],
             data[1].shape[1], out_type)
    stem = re.sub(r"_dehydrated$", "", os.path.splitext(os.path.basename(args.input))[0])
    out = args.output
    if out is not None and out.lower().endswith((".h5", ".hdf5")):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); path = out
    else:
        os.makedirs(out or ".", exist_ok=True); path = os.path.join(out or ".", stem + "_rehydrated.h5")
    attrs = {k: v for k, v in attrs.items() if k not in ("rank", "rehydrated")}
    attrs.update(dehydrated_source=os.path.abspath(args.input), rehydrated_bins=f"{bin_indices[0]}..{bin_indices[-1]}")
    write_denoised(path, (V, rows, cols), W4.reshape(-1, R).astype(np.float32), H.astype(np.float32), out_type, bin_indices, attrs)
    print(path)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="mbirtorch-hsnt", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Examples:\n"
                                       "  mbirtorch-hsnt inspect data.h5\n"
                                       "  mbirtorch-hsnt inspect sample_tifs/ --open-beam open_beam/ --estimate-rank\n"
                                       "  mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5\n"
                                       "  mbirtorch-hsnt dehydrate sample.h5 -o results/                    # rank estimated\n"
                                       "  mbirtorch-hsnt dehydrate sample_tifs/ --open-beam open_beam/ --rank 2 --downsample 2 --wave-bin 4 --gauge -v\n"
                                       "  mbirtorch-hsnt rehydrate results/sample_dehydrated.h5 --wave-range 100:200 -o results/\n"
                                       "  mbirtorch-hsnt denoise sample.h5 -o results/                      # denoised data + dehydrated file\n")
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
    s.add_argument("--estimate-rank", action="store_true", help="choose the rank by likelihood-ratio tests on a pixel subsample (runs solves)")
    s.add_argument("--max-rank", type=int, default=6, help="largest rank the estimate considers")
    s.add_argument("--rank-pool", default="auto", metavar="auto|B|0", help="also test on B x B pooled pixels (default auto; 0 disables)")
    s.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("convert", help="write a TIFF stack (and open beam) or an HDF5 dataset as an HDF5 dataset in the package "
                                       "layout, streamed in blocks of bins")
    add_input(s)
    s.add_argument("-o", "--output", help="output .h5 (default: <input>.h5)")
    s.add_argument("--as-type", choices=("attenuation", "transmission"), default="attenuation", help="stored quantity (default attenuation)")
    g = s.add_argument_group("streaming")
    g.add_argument("--block-bins", type=int, metavar="N", help="bins per block (default: from --memory-budget)")
    g.add_argument("--memory-budget", type=float, default=256, metavar="MiB", help="working memory for the blocks (default 256 MiB; smaller blocks overlap reading and writing better)")
    g.add_argument("--workers", type=int, metavar="N", help="threads decoding TIFF images of a block (default: min(8, CPUs))")
    s.set_defaults(func=cmd_convert)

    def add_solve(sp, denoise):
        g = sp.add_argument_group("model")
        g.add_argument("--rank", "-r", default="auto", metavar="N|auto",
                       help="number of materials; by default estimated by likelihood-ratio tests on a pixel subsample (see --max-rank)")
        g.add_argument("--max-rank", type=int, default=6, help="largest rank the estimate considers (default 6)")
        g.add_argument("--rank-pool", default="auto", metavar="auto|B|0", help="also test on B x B pooled pixels and take the larger rank (default: B chosen so pooled pixels ~ bins; 0 disables)")
        g.add_argument("--method", choices=("joint_newton", "block_newton", "multiplicative", "lbfgsb"), default="joint_newton")
        g.add_argument("--max-steps", type=int, default=300)
        g.add_argument("--rel-tol", type=float, default=1e-6, help="relative loss change per step at which to stop")
        g.add_argument("--spectra", choices=("mle", "unconstrained", "support"), default="mle",
                       help="spectra estimator: maximum likelihood, the unconstrained-W re-estimate (pays above ~65k pixels), or per-pixel "
                        "support selection (needs the dose, rank <= 6)")
        g.add_argument("--gauge", action="store_true", help="pure-pixel gauge fix of the maps (assumes every material has pure pixels; needs the dose)")
        g = sp.add_argument_group("compute")
        g.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
        g.add_argument("--mode", choices=("auto", "full", "stream"), default="auto", help="full solve on the device or streamed by chunks (default: by free memory)")
        g.add_argument("--chunk-pixels", type=int, help="pixels per chunk in stream mode (default: from free memory)")
        g.add_argument("--max-passes", type=int, default=5, help="stream mode: polish passes over the data")
        g.add_argument("--warmup-pixels", type=int, default=16384, help="stream mode: pixels for the initial spectra fit")
        g.add_argument("--dry-run", action="store_true", help="load, check and plan, then stop")
        g = sp.add_argument_group("output")
        g.add_argument("-o", "--output", help="output directory (created if needed; default: current directory), or a .h5 path whose stem names the files")
        g.add_argument("--as-type", choices=("attenuation", "transmission"), help="quantity stored in the outputs (default: the input's)")
        if denoise:
            g.add_argument("--no-dehydrated", action="store_true", help="write only the denoised data, not the dehydrated file")
        else:
            g.add_argument("--rehydrate", action="store_true", help="also write the rehydrated (denoised) data, as large as the input")
        g.add_argument("--no-plots", action="store_true")

    s = sub.add_parser("dehydrate", help="fit the NNAL factorization and write it in the dehydrated layout, with plots and a report")
    add_input(s); add_solve(s, denoise=False); s.set_defaults(func=cmd_dehydrate, no_dehydrated=False)

    s = sub.add_parser("rehydrate", help="multiply a dehydrated file back into hyperspectral data (all bins or a range)")
    s.add_argument("input", help="a dehydrated .h5 (subspace_data, subspace_basis, dataset_type), as written by dehydrate")
    s.add_argument("-o", "--output", help="output directory (default: current directory), or a .h5 path; default name <stem>_rehydrated.h5")
    s.add_argument("--wave-range", help="spectral slice START:STOP of the dehydrated file's bins to rehydrate (default: all)")
    s.add_argument("--views", help="view slice START:STOP (default: all)")
    s.add_argument("--as-type", choices=("attenuation", "transmission"), help="quantity to write (default: the file's dataset_type)")
    g = s.add_argument_group("logging")
    g.add_argument("-v", "--verbose", action="count", default=0); g.add_argument("-q", "--quiet", action="store_true"); g.add_argument("--log-file")
    s.set_defaults(func=cmd_rehydrate)

    s = sub.add_parser("denoise", help="dehydrate and rehydrate: write the denoised hyperspectral data in the package's HDF5 layout")
    add_input(s); add_solve(s, denoise=True); s.set_defaults(func=cmd_denoise, rehydrate=True)
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
