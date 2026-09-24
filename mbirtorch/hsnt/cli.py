"""Command-line interface for the hsnt dehydration: ``mbirtorch-hsnt`` (or ``python -m mbirtorch.hsnt``).

    inspect    load a dataset, run the checks, print what a solve would see (no GPU needed)
    convert    write a TIFF stack (with its open beam) or an HDF5 dataset in the package's HDF5 layout, streamed
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
import json
import logging
import math
import os
import re
import sys
import time
import warnings

import numpy as np

from ..utilities import makedirs
from .loading import INPUT_TYPES, convert_to_hdf5, load_dataset
from .outputs import component_check, fit_quality, write_dehydrated, write_denoised
from .rank import estimate_rank

log = logging.getLogger("mbirtorch.hsnt")

_BYTES_PER_ELEMENT_FULL = 48        # joint_newton's working set: about 12 float32 arrays of T's shape
_BYTES_PER_ELEMENT_STREAM = 24      # solve_W on one chunk plus the accumulators


def _parse_slice(text, name):
    """'START:STOP' as a (start, stop) tuple of optional ints, or None for no text."""
    if text is None:
        return None
    m = re.fullmatch(r"(-?\d*):(-?\d*)", text.strip())
    if not m:
        raise ValueError(f"--{name} expects START:STOP (Python slice), got {text!r}")
    return (int(m.group(1)) if m.group(1) else None, int(m.group(2)) if m.group(2) else None)


def _rank_arg(text):
    if text.lower() == "auto":
        return "auto"
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer or 'auto', got {text!r}")
    return int(text)


def _pool_arg(text):
    if text.lower() == "auto":
        return "auto"
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"expected 'auto' or a block size (0 disables), got {text!r}")
    return int(text)


def _load(args):
    """load_dataset with the command's input options, logging the checks."""
    ds = load_dataset(args.input, open_beam=args.open_beam, input_type=args.input_type, dataset=args.dataset,
                      dose=args.dose, views=_parse_slice(args.views, "views"),
                      wave_range=_parse_slice(args.wave_range, "wave-range"), wave_bin=args.wave_bin,
                      downsample=args.downsample, strict=args.strict)
    log.info("loaded in %.1f s", ds.info["load_seconds"])
    for c in ds.checks:
        getattr(log, {"ok": "info", "warn": "warning", "error": "error"}[c.level])("check: %s", c.message)
    return ds


def _device(name):
    """'auto' | 'cpu' | 'cuda' | 'cuda:N' as a validated torch device string."""
    import torch
    from ._device import _default_device
    m = re.fullmatch(r"auto|cpu|cuda(?::(\d+))?", name)
    if not m:
        raise ValueError(f"--device {name}: expected auto, cpu, cuda or cuda:N")
    if name.startswith("cuda") and int(m.group(1) or 0) >= torch.cuda.device_count():
        raise ValueError(f"--device {name}: {torch.cuda.device_count()} CUDA device(s) available")
    name = _default_device(name)
    if name == "cpu":
        log.warning("running on the CPU: expect one to two orders of magnitude longer than a GPU")
    return name


def plan_memory(ds, device, mode, chunk_pixels):
    """Decide a full or streamed solve from the memory available on the device. Returns (mode, chunk_pixels, note)."""
    import torch
    from .._memory_ledger import device_budget_bytes
    P, K = ds.T.shape
    need_full = P * K * _BYTES_PER_ELEMENT_FULL
    if device.startswith("cuda"):
        free, total = device_budget_bytes(device), torch.cuda.get_device_properties(device).total_memory
        name = torch.cuda.get_device_name(device)
    else:
        import psutil
        free = total = psutil.virtual_memory().available
        name = "cpu"
    note = (f"{name}: {free / 2**30:.1f} GiB available of {total / 2**30:.1f}; a full solve needs about "
            f"{need_full / 2**30:.1f} GiB")
    if mode == "auto":
        mode = "full" if need_full < 0.7 * free else "stream"
    if mode == "stream":
        if chunk_pixels is None:
            chunk_pixels = int(0.4 * free / (K * _BYTES_PER_ELEMENT_STREAM)) // 1024 * 1024
            chunk_pixels = max(1024, min(chunk_pixels, P))
        note += f"; streaming in chunks of {chunk_pixels:,} pixels ({-(-P // chunk_pixels)} chunks)"
    log.info("plan: %s solve. %s", mode, note)
    return mode, chunk_pixels, note


def _support_options(args, K):
    return dict(method=args.support_method, wald_screen=args.wald_screen, free_refit=args.free_refit,
                penalty="auto" if args.support_penalty == "auto" else float(args.support_penalty) * math.log(K))


def solve(ds, args, device):
    """Run the factorization and the requested post-estimator. Returns (W, H, report) with W and H numpy."""
    import torch
    from ._loss import stable_nnal
    from ._streaming import stream_factorization
    from .factorization import nnal_factorization
    from .spectra import support_selected_spectra, unconstrained_spectra
    if args.spectra == "support" and ds.dose is None:
        raise ValueError("support selection needs the dose (open-beam counts per pixel and bin): pass --dose, or "
                         "give --open-beam with a TIFF stack of counts")
    if args.spectra == "support" and args.support_method == "enumerate" and args.rank_value > 8:
        raise ValueError("--support-method enumerate solves all 2^R - 1 subsets and is limited to rank 8; use "
                         "branch_bound")
    rank = args.rank_value
    support_kw = _support_options(args, ds.bins)
    rep = dict(rank=rank, rank_note=args.rank_note, rank_search=args.rank_detail)
    mode, chunk, rep["memory_plan"] = plan_memory(ds, device, args.mode, args.chunk_pixels)
    rep["mode"] = mode
    t0 = time.perf_counter()
    if mode == "full":
        T = torch.from_numpy(ds.T).to(device)
        W, H, steps = nnal_factorization(T, method=args.method, num_materials=rank, max_steps=args.max_steps,
                                         rel_tol=args.rel_tol)
        rep["steps"] = int(steps)
    else:
        if args.method != "joint_newton":
            log.warning("stream mode always uses joint_newton for the warm-up and block Newton for the polish; "
                        "--method %s ignored", args.method)
        chunks = [torch.from_numpy(ds.T[i:i + chunk]) for i in range(0, ds.pixels, chunk)]
        stats = {}
        support = dict(support_kw, dose=ds.dose) if args.spectra == "support" else None
        W_chunks, H, passes = stream_factorization(chunks, rank, max_passes=args.max_passes, rel_tol=args.rel_tol,
                                                   warmup_pixels=min(args.warmup_pixels, ds.pixels), device=device,
                                                   verbose=int(log.isEnabledFor(logging.DEBUG)), stats=stats,
                                                   nonneg_W=(args.spectra != "unconstrained"),
                                                   support_selection=support)
        W = torch.cat([w.to(device) for w in W_chunks])
        rep.update(passes=int(passes), loss_per_pass=stats.get("loss"), kkt_per_pass=stats.get("kkt"))
        T = None
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    rep["solve_seconds"] = round(time.perf_counter() - t0, 2)

    def loss(Wx, Hx):
        if T is not None:
            return stable_nnal(Wx.double() @ Hx.double(), T.double()).item()
        return float(sum(stable_nnal(Wx[i:i + chunk].double() @ Hx.double(),
                                     torch.from_numpy(ds.T[i:i + chunk]).to(device).double()).item()
                         for i in range(0, ds.pixels, chunk)))

    if mode == "full":
        rep["loss_mle"] = loss(W, H)
    else:           # the last polish pass's loss is at the final W and H; with free-signed W it is not the MLE's
        rep["loss_mle"] = stats["loss"][-1] if args.spectra != "unconstrained" else None
    steps_text = f"{rep['steps']} steps" if "steps" in rep else f"{rep['passes']} polish passes"
    log.info("factorization: %s, %s in %.1f s, loss %s", mode, steps_text, rep["solve_seconds"],
             "n/a" if rep["loss_mle"] is None else f"{rep['loss_mle']:.6g}")

    if args.spectra == "unconstrained" and mode == "full":
        t1 = time.perf_counter()
        W, H, st = unconstrained_spectra(T, W, H)
        rep["unconstrained_steps"], rep["unconstrained_seconds"] = int(st), round(time.perf_counter() - t1, 2)
    elif args.spectra == "support" and mode == "full":
        t1 = time.perf_counter()
        W, H, S, st = support_selected_spectra(T, W, H, ds.dose, **support_kw)
        rep["support_steps"], rep["support_seconds"] = int(st), round(time.perf_counter() - t1, 2)
        rep["mean_support_size"] = S.sum(1).double().mean().item()
    elif args.spectra == "support":
        S = torch.cat(stats["support_chunks"])
        rep["mean_support_size"] = S.sum(1).double().mean().item()
        rep["support_refit_passes"], rep["loss_per_pass_refit"] = int(stats["refit_passes"]), stats.get("loss_refit")
    rep["loss_final"] = rep["loss_mle"] if args.spectra == "mle" else loss(W, H)
    if args.spectra != "mle":
        support_text = (f", mean {rep['mean_support_size']:.2f} materials per pixel" if "mean_support_size" in rep
                        else "")
        log.info("%s spectra%s: loss %.6g", args.spectra, support_text, rep["loss_final"])
    rep["W_zero_frac"], rep["H_zero_frac"] = (W == 0).double().mean().item(), (H == 0).double().mean().item()
    if device.startswith("cuda"):
        rep["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return W.cpu().numpy(), H.cpu().numpy(), rep


def _output_path(output, default_name):
    """-o names a directory (created if needed) unless it ends in .h5/.hdf5, which is then the path itself."""
    if output is not None and output.lower().endswith((".h5", ".hdf5")):
        path = output
    else:
        path = os.path.join(output or ".", default_name)
    makedirs(path)
    return path


def _out_type(ds, args):
    if args.as_type:
        return args.as_type
    return "attenuation" if ds.dataset_type in ("counts", "attenuation") else "transmission"


def _run_attrs(ds, rep, args, **extra):
    """Provenance written as HDF5 attributes: where the data came from and how the solve was set up."""
    return dict(source=ds.source, input_type=ds.dataset_type, method=args.method, mode=rep["mode"],
                spectra=args.spectra, support_penalty=str(args.support_penalty), free_refit=bool(args.free_refit),
                downsample=args.downsample, wave_bin=args.wave_bin, dose=-1.0 if ds.dose is None else float(ds.dose),
                mbirtorch_hsnt_cli="1", **extra)


def write_report(base, ds, rep, args, outputs):
    path = base + "_report.json"
    report = dict(input=ds.source, input_type=ds.dataset_type, spatial_shape=list(ds.spatial_shape), pixels=ds.pixels,
                  bins=ds.bins, dose=ds.dose, args={k: v for k, v in vars(args).items() if k != "func"},
                  checks=[dict(level=c.level, message=c.message) for c in ds.checks], info=ds.info, result=rep,
                  outputs=outputs)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    log.info("wrote %s", path)
    return report


def _log_fit(rep, args):
    comp = rep["components"]
    if comp["proportional_pairs"]:
        i, j, corr = max(comp["proportional_pairs"], key=lambda p: p[2])
        log.warning("components %d and %d have nearly proportional maps (correlation %.2f; %d such pair(s)): the data "
                    "cannot tell their spectra apart, so each row of H is an arbitrary noisy slice of their sum. The "
                    "rank is probably above the number of distinct materials; %s. The mean-pixel spectrum in the "
                    "outputs is unaffected.", i, j, corr, len(comp["proportional_pairs"]),
                    "the estimate chose it" if args.rank_detail else "run without --rank to estimate it")
    log.info("components: max map correlation %.2f; per-row noise (relative to level) %s", comp["max_map_correlation"],
             ", ".join(f"{x:.3f}" for x in comp["row_noise_rel"]))
    q = rep["fit"]
    if "reduced_chi2" not in q:
        log.info("fit: relative residual in transmission %.4g (no dose, so no chi-square)", q["relative_residual"])
        return
    chi2 = q["reduced_chi2"]
    if 0.8 <= chi2 <= 1.3:
        verdict = "at the Poisson noise level"
    elif chi2 > 1.3:
        verdict = "above the noise level: the rank may be too small or the model misspecified"
    else:
        verdict = "below the noise level: the fit follows the noise (rank too large, or the dose is overestimated)"
    log.info("fit: reduced chi-square %.3f (%s); relative residual in transmission %.4g", chi2, verdict,
             q["relative_residual"])
    if not 0.5 <= chi2 <= 2.0:
        log.warning("reduced chi-square %.2f is far from 1; check the rank (%s) and the dose", chi2, args.rank_note)


def cmd_inspect(args):
    import torch
    ds = _load(args)
    st = ds.info["T_stats"]
    V, rows, cols = ds.spatial_shape
    print(f"\n{ds.source}\n  type {ds.dataset_type}; {V} view(s) x {rows} x {cols} pixels x {ds.bins} bins (source "
          f"bins {ds.bin_indices[0]}..{ds.bin_indices[-1]}); {ds.T.nbytes / 2**30:.2f} GiB as float32")
    print(f"  T: min {st['min']:.4g}  median {st['median']:.4g}  mean {st['mean']:.4g}  max {st['max']:.4g}; zeros "
          f"{st['zero']:.2%}, above 1: {float(np.mean(ds.T > 1)):.2%}" + ("  (sampled)" if st["sampled"] else ""))
    print(f"  dose: {'unknown' if ds.dose is None else f'{ds.dose:.4g} counts per pixel and bin'}")
    for c in ds.checks:
        print(f"  [{c.level:5s}] {c.message}")
    if args.estimate_rank:
        _, note, d = estimate_rank(ds.T, ds.spatial_shape, _device(args.device), max_rank=args.max_rank,
                                   pool=args.rank_pool)
        gains = ", ".join(f"{r}: {g:,.0f}" for r, g in zip(range(2, d["max_rank"] + 1), d["gains"]))
        print(f"  {note}; effective dose {d['effective_dose']:.3g}; gains by component (deciding test): {gains}; "
              f"threshold {d['threshold']:,.0f}")
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        plan_memory(ds, dev, "auto", None)
    print()
    return 0


def cmd_convert(args):
    out, checks, _ = convert_to_hdf5(args.input, output=args.output, open_beam=args.open_beam,
                                     input_type=args.input_type, dataset=args.dataset, dose=args.dose,
                                     views=_parse_slice(args.views, "views"),
                                     wave_range=_parse_slice(args.wave_range, "wave-range"), wave_bin=args.wave_bin,
                                     downsample=args.downsample, as_type=args.as_type, block_bins=args.block_bins,
                                     memory_budget_mib=args.memory_budget, workers=args.workers, strict=args.strict,
                                     progress=not args.quiet)
    for c in checks:
        getattr(log, {"ok": "info", "warn": "warning", "error": "error"}[c.level])("check: %s", c.message)
    n_err = sum(c.level == "error" for c in checks)
    print(out + (f"  ({n_err} data check(s) had errors)" if n_err else ""))
    return 0


def _resolve_rank(ds, args, device):
    """Set args.rank_value, args.rank_note and args.rank_detail from --rank N or the likelihood-ratio estimate."""
    if args.rank == "auto":
        args.rank_value, args.rank_note, args.rank_detail = estimate_rank(ds.T, ds.spatial_shape, device,
                                                                          max_rank=args.max_rank, pool=args.rank_pool)
        log.info("%s; pass --rank N to override", args.rank_note)
    else:
        args.rank_value, args.rank_note, args.rank_detail = args.rank, f"rank {args.rank} given", None


def _pipeline(args, denoise):
    """load, checks, rank, solve, fit quality and outputs, for dehydrate and denoise."""
    ds = _load(args)
    device = _device(args.device)
    _resolve_rank(ds, args, device)
    stem = os.path.splitext(os.path.basename(os.path.normpath(args.input)))[0]
    base = os.path.splitext(_output_path(args.output, stem + ".h5"))[0]
    if args.dry_run:
        plan_memory(ds, device, args.mode, args.chunk_pixels)
        print(f"dry run: data loaded and checked, {args.rank_note}; no solve. Output base: {base}")
        return 0
    W, H, rep = solve(ds, args, device)
    rep["fit"] = fit_quality(ds.T, W, H, ds.dose, device)
    rep["components"] = component_check(W, H)
    _log_fit(rep, args)
    R, out_type, outputs = H.shape[0], _out_type(ds, args), []
    if not args.no_dehydrated:
        outputs.append(write_dehydrated(base + "_dehydrated.h5", W.reshape(*ds.spatial_shape, R), H, out_type,
                                        ds.bin_indices, _run_attrs(ds, rep, args, rank=R, loss=rep["loss_final"])))
    if args.rehydrate:
        outputs.append(write_denoised(base + "_denoised.h5", ds.spatial_shape, W, H, out_type, ds.bin_indices,
                                      _run_attrs(ds, rep, args, loss=rep["loss_final"])))
    if not args.no_plots:
        from .plots import plot_factorization
        outputs += plot_factorization(base, ds.source, W.reshape(*ds.spatial_shape, R), H, ds.bin_indices)
        log.info("wrote %s and %s", *outputs[-2:])
    write_report(base, ds, rep, args, outputs)
    n_err = sum(c.level == "error" for c in ds.checks)
    chi2 = rep["fit"].get("reduced_chi2")
    print(f"done: rank {R} ({'estimated' if args.rank_detail else 'given'}), {ds.pixels:,} pixels x {ds.bins} bins, "
          f"{rep['mode']} solve in {rep['solve_seconds']} s, loss {rep['loss_final']:.6g}"
          + (f", reduced chi-square {chi2:.2f}" if chi2 is not None else "") + f"; outputs at {base}_*"
          + (f"; {n_err} data check(s) had errors" if n_err else ""))
    return 0


def cmd_dehydrate(args):
    return _pipeline(args, denoise=False)


def cmd_denoise(args):
    return _pipeline(args, denoise=True)


def cmd_rehydrate(args):
    """Multiply a dehydrated file back into hyperspectral data, for all bins or a --wave-range of them."""
    import h5py
    from .io import import_hsnt_data_hdf5
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")
    data, _ = import_hsnt_data_hdf5(args.input)
    if not isinstance(data, list):
        raise ValueError(f"{args.input}: not a dehydrated file (needs subspace_data, subspace_basis and "
                         "dataset_type); run `mbirtorch-hsnt dehydrate` first")
    W4, H, dtype = data
    K_file = H.shape[1]
    if W4.ndim == 3:
        W4 = W4[None]
    if W4.ndim != 4:
        raise ValueError(f"subspace_data has shape {W4.shape}; expected (views, rows, cols, rank) or (rows, cols, "
                         "rank)")
    if W4.shape[-1] != H.shape[0]:
        raise ValueError(f"rank mismatch: subspace_data has {W4.shape[-1]} components, subspace_basis {H.shape[0]} "
                         "rows")
    with h5py.File(args.input, "r") as f:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        bin_indices = f["bin_indices"][()] if "bin_indices" in f else np.arange(K_file)
    views = _parse_slice(args.views, "views")
    if views:
        W4 = W4[slice(*views)]
    wave = _parse_slice(args.wave_range, "wave-range")
    if wave:
        H, bin_indices = H[:, slice(*wave)], bin_indices[slice(*wave)]
        if H.shape[1] == 0:
            raise ValueError(f"--wave-range {args.wave_range} selects no bins of the {K_file} in the file")
    V, rows, cols, R = W4.shape
    out_type = args.as_type or dtype
    log.info("%s: %d component(s), %d view(s) x %d x %d pixels, %d of %d bins -> %s", args.input, R, V, rows, cols,
             H.shape[1], K_file, out_type)
    stem = re.sub(r"_dehydrated$", "", os.path.splitext(os.path.basename(args.input))[0])
    path = _output_path(args.output, stem + "_rehydrated.h5")
    attrs = {k: v for k, v in attrs.items() if k not in ("rank", "rehydrated")}
    attrs.update(dehydrated_source=os.path.abspath(args.input), rehydrated_bins=f"{bin_indices[0]}..{bin_indices[-1]}")
    write_denoised(path, (V, rows, cols), W4.reshape(-1, R).astype(np.float32), H.astype(np.float32), out_type,
                   bin_indices, attrs)
    print(path)
    return 0


_EXAMPLES = """Examples:
  mbirtorch-hsnt inspect data.h5
  mbirtorch-hsnt inspect sample_tifs/ --open-beam open_beam/ --estimate-rank
  mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ --wave-bin 4 -o sample.h5
  mbirtorch-hsnt dehydrate sample.h5 -o results/                    # rank estimated
  mbirtorch-hsnt dehydrate sample_tifs/ --open-beam open_beam/ --rank 2 --downsample 2 --wave-bin 4 -v
  mbirtorch-hsnt rehydrate results/sample_dehydrated.h5 --wave-range 100:200 -o results/
  mbirtorch-hsnt denoise sample.h5 -o results/                      # denoised data + dehydrated file
"""


def _add_logging(g):
    g.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO (default), -vv for DEBUG")
    g.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    g.add_argument("--log-file", help="also write the log here")


def _add_input(sp):
    sp.add_argument("input", help="HDF5 file (package layout) or a directory with one TIFF per wavelength bin")
    g = sp.add_argument_group("input interpretation")
    g.add_argument("--open-beam", nargs="+", metavar="DIR", help="open-beam TIFF stack(s) for a stack of counts; a "
                   "directory of observation subdirectories is averaged over them")
    g.add_argument("--input-type", choices=("auto",) + INPUT_TYPES, default="auto",
                   help="what the values are (default: infer)")
    g.add_argument("--dataset", help="HDF5 group holding 'data' (default: the root, or the only group that has one)")
    g.add_argument("--dose", type=float,
                   help="open-beam counts per pixel and bin, when the input is not counts with an open beam")
    g = sp.add_argument_group("selection")
    g.add_argument("--views", help="view slice START:STOP for 4-D HDF5 data (default: all)")
    g.add_argument("--wave-range", help="spectral slice START:STOP over the source bins (default: all)")
    g.add_argument("--wave-bin", type=int, default=1, metavar="N",
                   help="group N adjacent bins (sum counts / average transmissions)")
    g.add_argument("--downsample", type=int, default=1, metavar="S", help="keep every S-th row and column")
    g = sp.add_argument_group("checks and logging")
    g.add_argument("--strict", action="store_true", help="stop if any data check reports an error")
    _add_logging(g)


def _add_rank(g):
    g.add_argument("--max-rank", type=int, default=6, help="largest rank the estimate considers (default 6)")
    g.add_argument("--rank-pool", type=_pool_arg, default="auto", metavar="auto|B|0",
                   help="also test on B x B pooled pixels and take the larger rank (default: B chosen so pooled pixels "
                        "hold about 64 counts per bin; 0 disables)")


def _add_device(g):
    g.add_argument("--device", default="auto", metavar="auto|cpu|cuda|cuda:N",
                   help="compute device (default: cuda if available)")


def _add_solve(sp, denoise):
    g = sp.add_argument_group("model")
    g.add_argument("--rank", "-r", type=_rank_arg, default="auto", metavar="N|auto",
                   help="number of materials; by default estimated by likelihood-ratio tests on a pixel subsample")
    _add_rank(g)
    g.add_argument("--method", choices=("joint_newton", "block_newton", "multiplicative", "lbfgsb"),
                   default="joint_newton")
    g.add_argument("--max-steps", type=int, default=300, help="full mode: largest number of solver steps")
    g.add_argument("--rel-tol", type=float, default=1e-6, help="relative loss change per step at which to stop")
    g.add_argument("--spectra", choices=("mle", "unconstrained", "support"), default="mle",
                   help="spectra estimator: maximum likelihood, the unconstrained-W re-estimate (pays above about "
                        "1e5 pixels), or per-pixel support selection (needs the dose)")
    g.add_argument("--support-method", choices=("branch_bound", "greedy", "enumerate"), default="branch_bound",
                   help="subset search of support selection: branch and bound (any rank, default), greedy (fastest, "
                        "heuristic), or the 2^R - 1 enumeration (rank <= 8)")
    g.add_argument("--support-penalty", default="2", metavar="F|auto",
                   help="penalty per selected material, F x log(bins) nats (default 2: essentially no false "
                        "admissions; 0.5-1 keeps a faint material in more of its pixels below about 10 counts per bin "
                        "at the cost of map noise above about 100) or 'auto', which moves from 0.5 to 2 with the "
                        "counts per pixel and bin")
    g.add_argument("--free-refit", action="store_true",
                   help="with --spectra support: drop the bound on the selected coefficients during the refit (as the "
                        "unconstrained estimator does for all of them), then re-solve W >= 0 on the supports")
    g.add_argument("--wald-screen", type=float, default=0.0, metavar="F",
                   help="skip single-material fits below F x penalty of Wald statistic in the full fit (0 = off; "
                        "trades rare-material recall for time)")
    g = sp.add_argument_group("compute")
    _add_device(g)
    g.add_argument("--mode", choices=("auto", "full", "stream"), default="auto",
                   help="full solve on the device or streamed by chunks (default: by available memory)")
    g.add_argument("--chunk-pixels", type=int, help="pixels per chunk in stream mode (default: from available memory)")
    g.add_argument("--max-passes", type=int, default=5, help="stream mode: polish passes over the data")
    g.add_argument("--warmup-pixels", type=int, default=16384, help="stream mode: pixels for the initial spectra fit")
    g.add_argument("--dry-run", action="store_true", help="load, check and plan, then stop")
    g = sp.add_argument_group("output")
    g.add_argument("-o", "--output", help="output directory (created if needed; default: current directory), or a .h5 "
                   "path whose stem names the files")
    g.add_argument("--as-type", choices=("attenuation", "transmission"),
                   help="quantity stored in the outputs (default: the input's)")
    if denoise:
        g.add_argument("--no-dehydrated", action="store_true",
                       help="write only the denoised data, not the dehydrated file")
    else:
        g.add_argument("--rehydrate", action="store_true",
                       help="also write the rehydrated (denoised) data, as large as the input")
    g.add_argument("--no-plots", action="store_true")


def build_parser():
    p = argparse.ArgumentParser(prog="mbirtorch-hsnt", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EXAMPLES)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="load, check and describe a dataset (no solve)")
    _add_input(s)
    g = s.add_argument_group("rank")
    g.add_argument("--estimate-rank", action="store_true",
                   help="choose the rank by likelihood-ratio tests on a pixel subsample (runs solves)")
    _add_rank(g)
    _add_device(g)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("convert", help="write a TIFF stack (and open beam) or an HDF5 dataset as an HDF5 dataset in "
                                       "the package layout, streamed in blocks of bins")
    _add_input(s)
    s.add_argument("-o", "--output", help="output .h5 (default: <input>.h5)")
    s.add_argument("--as-type", choices=("attenuation", "transmission"), default="attenuation",
                   help="stored quantity (default attenuation)")
    g = s.add_argument_group("streaming")
    g.add_argument("--block-bins", type=int, metavar="N", help="bins per block (default: from --memory-budget)")
    g.add_argument("--memory-budget", type=float, default=256, metavar="MiB",
                   help="working memory for the blocks (default 256 MiB; smaller blocks overlap reading and writing "
                        "better)")
    g.add_argument("--workers", type=int, metavar="N",
                   help="threads decoding TIFF images of a block (default: min(8, CPUs))")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("dehydrate", help="fit the NNAL factorization and write it in the dehydrated layout, with plots "
                                         "and a report")
    _add_input(s)
    _add_solve(s, denoise=False)
    s.set_defaults(func=cmd_dehydrate, no_dehydrated=False)

    s = sub.add_parser("rehydrate",
                       help="multiply a dehydrated file back into hyperspectral data (all bins or a range)")
    s.add_argument("input", help="a dehydrated .h5 (subspace_data, subspace_basis, dataset_type), as written by "
                                 "dehydrate")
    s.add_argument("-o", "--output", help="output directory (default: current directory), or a .h5 path; default name "
                                          "<stem>_rehydrated.h5")
    s.add_argument("--wave-range", help="spectral slice START:STOP of the dehydrated file's bins to rehydrate "
                                        "(default: all)")
    s.add_argument("--views", help="view slice START:STOP (default: all)")
    s.add_argument("--as-type", choices=("attenuation", "transmission"),
                   help="quantity to write (default: the file's dataset_type)")
    _add_logging(s.add_argument_group("logging"))
    s.set_defaults(func=cmd_rehydrate)

    s = sub.add_parser("denoise", help="dehydrate and rehydrate: write the denoised hyperspectral data in the "
                                       "package's HDF5 layout")
    _add_input(s)
    _add_solve(s, denoise=True)
    s.set_defaults(func=cmd_denoise, rehydrate=True)
    return p


def main(argv=None):
    """Run the command line. Errors in the data or the options exit with their message (the traceback with -vv)."""
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose >= 2 else logging.INFO)
    handlers = [logging.StreamHandler(sys.stderr)] + ([logging.FileHandler(args.log_file)] if args.log_file else [])
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    saved = log.level, log.propagate, warnings.showwarning
    log.setLevel(level)
    log.propagate = False
    warnings.showwarning = lambda message, *rest, **kw: log.warning("%s", message)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except (ValueError, KeyError, OSError, MemoryError) as e:
        if level <= logging.DEBUG:
            raise
        message = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        raise SystemExit(f"error: {message}") from None
    finally:
        log.setLevel(saved[0])
        log.propagate = saved[1]
        warnings.showwarning = saved[2]
        for h in handlers:
            log.removeHandler(h)
            h.close()


if __name__ == "__main__":
    sys.exit(main())
