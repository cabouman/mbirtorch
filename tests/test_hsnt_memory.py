"""Tests for the memory of the hsnt solves: the plan that chooses a full or a streamed solve from the device's budget,
the budget itself, the streamed warm-up and stop, the rank test's float64 diagnostics, and the command line's advice
when the device runs out of memory.

The devices are faked: the plan reads its budget through torch.cuda's memory queries, which are patched here, so the
tests run on the CPU and describe a GPU of any size.
"""
import logging
from types import SimpleNamespace

import h5py
import numpy as np
import psutil
import pytest
import torch

import mbirtorch.kernel_availability as kernel_availability
from mbirtorch.hsnt import _loss, _streaming, factorization, outputs, rank
from mbirtorch.hsnt._fit import _fit, _plan
from mbirtorch.hsnt.cli import main

GiB = 2 ** 30


@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    """One intra-op thread, so parallel test workers do not each start one per core."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _fake_cuda(monkeypatch, total, free, allocated=0.0, unused=0.0, releases=True, triton=True):
    """A CUDA device of `total` GiB as the plan reads it: `free` GiB free, `allocated` GiB allocated by this process
    and `unused` GiB more reserved by its caching allocator. The driver counts the unused reserve as free as well, as
    the WSL2 driver was seen to; empty_cache returns the reserve to the driver unless releases=False (segments that
    stay partly in use). Returns the state, with the number of empty_cache calls."""
    dev = dict(free=free * GiB, allocated=allocated * GiB, unused=unused * GiB, emptied=0)

    def empty_cache():
        dev["emptied"] += 1
        if releases:
            dev["free"] += dev["unused"]
            dev["unused"] = 0

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(dev["free"] + dev["unused"]),
                                                                          int(total * GiB)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: int(dev["allocated"] + dev["unused"]))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(dev["allocated"]))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device=None: SimpleNamespace(total_memory=int(total * GiB)))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "test GPU")
    monkeypatch.setattr(kernel_availability, "triton_available", lambda: (triton, "faked"))
    return dev


def _problem(P, K, R=2, dose=20.0, seed=0):
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 1.0, (R, K))
    W = rng.dirichlet(np.full(R, 0.5), P) * rng.uniform(0.2, 2.0, (P, 1))
    W[: P // 8] = 0
    return (rng.poisson(dose * np.exp(-W @ H)) / dose).astype(np.float32)


def test_a_1m_pixel_fit_is_solved_whole_on_an_80_gb_gpu_when_it_compiles(monkeypatch):
    """1,048,576 pixels x 1200 bins with 78.6 GiB free (an H100) plans a full solve for all three estimators when
    the solve compiles (its peaks were 43.6 and 44.7 GiB); eager, whose peak was 62.4 GiB, it streams in two chunks
    with the whole warm-up."""
    P, K = 2 ** 20, 1200
    _fake_cuda(monkeypatch, total=79.2, free=78.6)
    for spectra in ("mle", "unconstrained", "support"):
        plan = _plan(P, K, "cuda", spectra)
        assert plan[0] == "full" and "compiled" in plan[-1] and "78.6 GiB available of 79.2" in plan[-1]
    _fake_cuda(monkeypatch, total=79.2, free=78.6, triton=False)
    for spectra in ("mle", "unconstrained", "support"):
        mode, chunk, warmup, note = _plan(P, K, "cuda", spectra)
        assert (mode, chunk, warmup) == ("stream", 658_432, 16384) and "eager" in note and "(2 chunks)" in note
    assert _plan(P, K, "cuda", "mle", compile_mode="off")[0] == "stream"                 # compile_mode is heeded
    assert _plan(P, K, "cuda", "mle", mode="full")[0] == "full"                            # and a mode given


def test_a_laptop_gpu_streams_and_caps_the_warm_up(monkeypatch):
    """262,144 pixels x 2500 bins with 3.2 GiB free streams in chunks sized to 0.6 of the budget at 64 bytes per
    entry, and fits its warm-up on fewer than 16,384 pixels, which would need more than that share."""
    _fake_cuda(monkeypatch, total=4.0, free=3.2)
    mode, chunk, warmup, note = _plan(262_144, 2500, "cuda", "mle")
    assert (mode, chunk, warmup) == ("stream", 12_288, 13_312) and "(22 chunks)" in note
    assert "warm-up fits 13,312 pixels, not 16,384" in note
    assert _plan(262_144, 2500, "cuda", "basis")[2] is None                   # a given basis has no warm-up
    _fake_cuda(monkeypatch, total=4.0, free=0.1)
    assert _plan(262_144, 2500, "cuda", "mle")[2] == 4096                    # at least a few thousand pixels


def test_the_device_budget_empties_the_cache_and_stays_within_the_device(monkeypatch):
    """The budget is read after the allocator's cache is emptied, and is at most the total less what is allocated;
    where free plus the reserve exceeds the total (the laptop's '5.1 GiB available of 4.0'), the driver's free memory
    is taken alone, and the note says so."""
    dev = _fake_cuda(monkeypatch, total=4.0, free=2.1, allocated=0.1, unused=1.5)
    note = _plan(1000, 100, "cuda")[-1]
    assert dev["emptied"] == 1 and "3.6 GiB available of 4.0;" in note
    _fake_cuda(monkeypatch, total=4.0, free=2.1, allocated=0.1, unused=1.5, releases=False)
    note = _plan(1000, 100, "cuda")[-1]
    assert "3.6 GiB available of 4.0 (the driver's free memory alone" in note
    _fake_cuda(monkeypatch, total=4.0, free=1.7, allocated=1.5, unused=0.5, releases=False)
    note = _plan(1000, 100, "cuda")[-1]                                       # 2.7 read, 2.5 not allocated
    assert "2.5 GiB available of 4.0;" in note


def test_the_streamed_warm_up_fits_the_capped_pixels(monkeypatch):
    """A streamed fit on a host with little memory fits its warm-up on the pixels the plan allows, not on every
    pixel asked for, and reports the cap in its memory plan."""
    T = _problem(6144, 40)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=21_000_000))
    fitted = []
    real = _streaming._nnal_factorization

    def spy(T_sub, *args, **kw):
        fitted.append(T_sub.shape[0])
        return real(T_sub, *args, **kw)

    monkeypatch.setattr(_streaming, "_nnal_factorization", spy)
    W, H, rep = _fit(T, 2, device="cpu", mode="stream", chunk_pixels=2048, max_passes=1)
    assert fitted == [5120] and "warm-up fits 5,120 pixels, not 6,144" in rep["memory_plan"]
    assert W.shape == (6144, 2) and rep["chunk_pixels"] == 2048


def test_a_stream_ended_by_max_passes_warns(caplog):
    """The polish loop warns, with the last pass's relative loss change, when max_passes ends it before the rel_tol
    stop fires; not when the stop fires, when rel_tol is 0, or with no passes."""
    T = torch.from_numpy(_problem(2048, 60))
    tiles = [T[:1024], T[1024:]]
    caplog.set_level(logging.WARNING, logger="mbirtorch.hsnt")
    _streaming._stream_factorization(tiles, 2, max_passes=1, rel_tol=1e-14, warmup_pixels=1024, device="cpu")
    [record] = [r for r in caplog.records if "max_passes" in r.getMessage()]
    change = float(record.getMessage().split("changed the loss by ")[1].split(",")[0])
    assert "max_passes (1) ended the polish passes" in record.getMessage() and 1e-14 < change < 1
    for kw in (dict(max_passes=3, rel_tol=0.5), dict(max_passes=1, rel_tol=0.0), dict(max_passes=0, rel_tol=1e-14)):
        caplog.clear()
        _streaming._stream_factorization(tiles, 2, warmup_pixels=1024, device="cpu", **kw)
        assert not [r for r in caplog.records if "max_passes" in r.getMessage()], kw


def test_the_rank_test_diagnostics_work_in_blocks(monkeypatch):
    """The rank test's float64 loss and residual are formed in blocks of outputs._CHUNK_ELEMENTS entries, never over
    the whole subsample, and give the rank and gains of the whole-array sums (float64 summation order aside)."""
    T = torch.from_numpy(_problem(1024, 100, R=3, dose=50.0))
    rank_one, detail_one = rank._lrt_rank(T, 4, "one block")
    sizes = []
    real = _loss.stable_nnal

    def spy(X, T, *args, **kw):
        if X.dtype == torch.float64:
            sizes.append(X.numel())
        return real(X, T, *args, **kw)

    monkeypatch.setattr(_loss, "stable_nnal", spy)
    monkeypatch.setattr(outputs, "_CHUNK_ELEMENTS", 2 ** 12)
    rank_blocks, detail_blocks = rank._lrt_rank(T, 4, "blocks")
    assert len(sizes) > 4 and max(sizes) <= 2 ** 12 and rank_blocks == rank_one == 3
    assert np.allclose(detail_blocks["gains"], detail_one["gains"], rtol=1e-9, atol=0)
    assert np.allclose(detail_blocks["losses"], detail_one["losses"], rtol=1e-13, atol=0)


@pytest.fixture
def small_h5(tmp_path):
    path = tmp_path / "small.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=_problem(120, 40).reshape(1, 12, 10, 40))
        f.create_dataset("dataset_type", data=np.bytes_("transmission"))
    return str(path)


def test_device_failures_on_the_command_line_get_the_plan_and_the_memory_advice(small_h5, tmp_path, monkeypatch):
    """A cuBLAS or cuSOLVER failure, which an oversubscribed device raises instead of an out-of-memory error, exits
    like one: with the memory plan and the advice to halve --chunk-pixels (streamed) or stream (full), or to use the
    CPU; -vv keeps the traceback, and other errors still raise."""
    run = ["dehydrate", small_h5, "-o", str(tmp_path / "out"), "--rank", "2", "--no-plots", "--device", "cpu"]

    def fail(message, error=RuntimeError):
        def raise_it(*args, **kw):
            raise error(message)
        return raise_it

    monkeypatch.setattr(_streaming, "_stream_factorization",
                        fail("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, ...)`\n"
                             "CUDA kernel errors might be asynchronously reported"))
    stream = run + ["--mode", "stream", "--chunk-pixels", "40"]
    with pytest.raises(SystemExit) as exit_info:
        main(stream + ["-q"])
    message = str(exit_info.value)
    assert "failed, most likely out of memory (CUDA error: CUBLAS_STATUS_EXECUTION_FAILED" in message
    assert "in a stream solve; plan: cpu:" in message and "chunks of 40 pixels (3 chunks)" in message
    assert message.endswith("Try half the chunk, --chunk-pixels 20, or --device cpu")
    with pytest.raises(RuntimeError, match="CUBLAS_STATUS_EXECUTION_FAILED"):
        main(stream + ["-vv"])
    monkeypatch.setattr(factorization, "_nnal_factorization", fail("cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR"))
    with pytest.raises(SystemExit, match=r"in a full solve; plan: .*\. Try --mode stream, .* or --device cpu$"):
        main(run + ["--mode", "full", "-q"])
    monkeypatch.setattr(factorization, "_nnal_factorization", fail("CUDA out of memory.", torch.cuda.OutOfMemoryError))
    with pytest.raises(SystemExit, match=r"^error: the device ran out of memory \(CUDA out of memory\.\) in a full"):
        main(run + ["--mode", "full", "-q"])
    monkeypatch.setattr(factorization, "_nnal_factorization", fail("shapes cannot be multiplied"))
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        main(run + ["--mode", "full", "-q"])
