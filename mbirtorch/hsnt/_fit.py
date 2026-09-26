"""The fit behind dehydrate and the command line: a memory plan, then the maximum-likelihood factorization held whole on
the device or streamed by chunks of pixels, then the requested spectra estimator."""
import logging
import time

import numpy as np
import torch

log = logging.getLogger("mbirtorch.hsnt")

# Device working set of a full solve, in bytes per data entry (measured on the H100: the joint-Newton MLE 49.5, the
# unconstrained re-estimate 50.5, support selection 67.7; a W solve against a fixed basis as a streamed chunk's), and
# of a streamed one (a chunk's W solve plus the sums).
_BYTES_PER_ELEMENT = dict(mle=50, unconstrained=52, support=70, basis=24)
_BYTES_PER_ELEMENT_STREAM = 24
SPECTRA = ("mle", "unconstrained", "support")


def _plan(P, K, device, spectra="mle", mode="auto", chunk_pixels=None):
    """A full or streamed solve from the memory the device has available. Returns (mode, chunk_pixels, note)."""
    need_full = P * K * _BYTES_PER_ELEMENT[spectra]
    if device.startswith("cuda"):
        from .._memory_ledger import device_budget_bytes
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
    if mode not in ("full", "stream"):
        raise ValueError(f"mode must be 'auto', 'full' or 'stream', got {mode!r}")
    if mode == "stream":
        if chunk_pixels is None:
            chunk_pixels = max(1024, int(0.4 * free / (K * _BYTES_PER_ELEMENT_STREAM)) // 1024 * 1024)
        chunk_pixels = max(1, min(int(chunk_pixels), P))
        note += f"; streaming in chunks of {chunk_pixels:,} pixels ({-(-P // chunk_pixels)} chunks)"
    log.info("plan: %s solve. %s", mode, note)
    return mode, chunk_pixels, note


def _loss(W, H, T_host, device):
    """The float64 NNAL loss of W @ H against the host data, in pixel chunks so no P x K float64 array is formed."""
    from ._loss import stable_nnal
    from .outputs import _CHUNK_ELEMENTS
    Hd = H.to(device).double()
    chunk = max(1, _CHUNK_ELEMENTS // T_host.shape[1])
    total = 0.0
    for i in range(0, T_host.shape[0], chunk):
        Tc = torch.from_numpy(T_host[i:i + chunk]).to(device).double()
        total += stable_nnal(W[i:i + chunk].to(device).double() @ Hd, Tc).item()
    return total


def _fit_fixed_basis(T, H, device="cpu", mode="auto", chunk_pixels=None, max_steps=1000, rel_tol=1e-8,
                     compile_mode="auto"):
    """The maps for a given basis: every pixel's maximum-likelihood W >= 0 with H fixed, independent problems solved
    by block Newton in chunks of pixels from the memory plan. Returns (W, H, report) as _fit does."""
    from ._newton import _resolve_compile, solve_W
    P, K = T.shape
    Hd = torch.as_tensor(np.ascontiguousarray(H, dtype=np.float32), device=device)
    mode, chunk, note = _plan(P, K, device, "basis", mode, chunk_pixels)
    chunk = P if mode == "full" else chunk
    rep = dict(mode=mode, memory_plan=note, spectra="given basis", chunks=-(-P // chunk))
    if mode != "full":                                  # as a streamed fit: chunks compile only when asked
        compile_mode = "on" if compile_mode == "on" else "off"
    t0 = time.perf_counter()
    parts = []
    for i in range(0, P, chunk):
        Tc = torch.from_numpy(T[i:i + chunk]).to(device)
        parts.append(solve_W(Tc, Hd, max_steps=max_steps, rel_tol=rel_tol,
                             compile_mode=_resolve_compile(compile_mode, Tc)).cpu())
        del Tc
    W = torch.cat(parts)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    rep["solve_seconds"] = round(time.perf_counter() - t0, 2)
    rep["loss_mle"] = rep["loss_final"] = _loss(W, Hd, T, device)
    log.info("maps for the given basis: %s, %d chunk(s) in %.1f s, loss %.6g", mode, rep["chunks"],
             rep["solve_seconds"], rep["loss_final"])
    rep["W_zero_frac"], rep["H_zero_frac"] = (W == 0).double().mean().item(), (Hd == 0).double().mean().item()
    if device.startswith("cuda"):
        rep["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return W.numpy().astype(np.float32), np.asarray(H, dtype=np.float32), rep


def _fit(T, rank, spectra="mle", dose=None, penalty="auto", free_refit=False, wald_screen=0.0, device="cpu",
         mode="auto", chunk_pixels=None, max_steps=1000, rel_tol=1e-8, max_passes=5, warmup_pixels=16384,
         compile_mode="auto"):
    """Fit T (host numpy, pixels x bins, float32) at the given rank. Returns (W, H, report): numpy factors and a dict
    of what was done (mode, steps or passes, seconds, losses, support size).

    spectra: 'mle', 'unconstrained' or 'support' (needs the dose). penalty: support selection's charge per component,
    'auto' or a multiple of log K. compile_mode applies to the full solve; a streamed solve compiles only with 'on'.
    """
    from ._streaming import _stream_factorization
    from .factorization import _nnal_factorization
    from .spectra import _support_selected_spectra, _unconstrained_spectra
    if spectra not in SPECTRA:
        raise ValueError(f"spectra must be one of {SPECTRA}, got {spectra!r}")
    if spectra == "support" and dose is None:
        raise ValueError("support selection needs the dose (open-beam counts per pixel and bin)")
    P, K = T.shape
    mode, chunk, note = _plan(P, K, device, spectra, mode, chunk_pixels)
    rep = dict(mode=mode, memory_plan=note, spectra=spectra)
    t0 = time.perf_counter()
    stats = {}
    if mode == "full":
        Td = torch.from_numpy(T).to(device)
        W, H, steps = _nnal_factorization(Td, rank, max_steps=max_steps, rel_tol=rel_tol, compile_mode=compile_mode)
        rep["steps"] = int(steps)
    else:
        chunks = [torch.from_numpy(T[i:i + chunk]) for i in range(0, P, chunk)]
        support = (dict(dose=dose, penalty=penalty, free_refit=free_refit, wald_screen=wald_screen)
                   if spectra == "support" else None)
        W_chunks, H, passes = _stream_factorization(chunks, rank, max_passes=max_passes, rel_tol=rel_tol,
                                                    warmup_pixels=min(warmup_pixels, P), device=device,
                                                    verbose=int(log.isEnabledFor(logging.DEBUG)), stats=stats,
                                                    nonneg_W=(spectra != "unconstrained"), support_selection=support,
                                                    compile_mode="on" if compile_mode == "on" else "off")
        W = torch.cat(W_chunks)
        rep.update(passes=int(passes), loss_per_pass=stats.get("loss"), kkt_per_pass=stats.get("kkt"))
        Td = None
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    rep["solve_seconds"] = round(time.perf_counter() - t0, 2)
    if mode == "full":
        rep["loss_mle"] = _loss(W, H, T, device)
    else:       # the last polish pass's loss is at the final W and H; with free-signed W it is not the MLE's
        rep["loss_mle"] = stats["loss"][-1] if spectra != "unconstrained" else None
    steps_text = f"{rep['steps']} steps" if "steps" in rep else f"{rep['passes']} polish passes"
    log.info("factorization: %s, %s in %.1f s, loss %s", mode, steps_text, rep["solve_seconds"],
             "n/a" if rep["loss_mle"] is None else f"{rep['loss_mle']:.6g}")
    if spectra == "unconstrained" and mode == "full":
        t1 = time.perf_counter()
        W, H, st = _unconstrained_spectra(Td, W, H, compile_mode=compile_mode)
        rep.update(unconstrained_steps=int(st), unconstrained_seconds=round(time.perf_counter() - t1, 2))
    elif spectra == "support" and mode == "full":
        t1 = time.perf_counter()
        W, H, S, st = _support_selected_spectra(Td, W, H, dose, penalty=penalty, wald_screen=wald_screen,
                                                free_refit=free_refit, compile_mode=compile_mode)
        rep.update(support_steps=int(st), support_seconds=round(time.perf_counter() - t1, 2),
                   mean_support_size=S.sum(1).double().mean().item())
    elif spectra == "support":
        S = torch.cat(stats["support_chunks"])
        rep.update(mean_support_size=S.sum(1).double().mean().item(), support_refit_passes=int(stats["refit_passes"]),
                   loss_per_pass_refit=stats.get("loss_refit"))
    rep["loss_final"] = rep["loss_mle"] if spectra == "mle" else _loss(W, H, T, device)
    if spectra != "mle":
        support_text = (f", mean {rep['mean_support_size']:.2f} materials per pixel" if "mean_support_size" in rep
                        else "")
        log.info("%s spectra%s: loss %.6g", spectra, support_text, rep["loss_final"])
    rep["W_zero_frac"], rep["H_zero_frac"] = (W == 0).double().mean().item(), (H == 0).double().mean().item()
    if device.startswith("cuda"):
        rep["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    return W.cpu().numpy().astype(np.float32), H.cpu().numpy().astype(np.float32), rep
