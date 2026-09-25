"""The host-thread warning: raised when torch has one thread and more cores are
available, naming the variables that set it; silent otherwise."""

import warnings

import pytest
import torch

from mbirtorch import _host_threads


def _report(monkeypatch, threads, cores, variables):
    monkeypatch.setattr(torch, 'get_num_threads', lambda: threads)
    monkeypatch.setattr(_host_threads, 'available_cores', lambda: cores)
    for name in _host_threads.THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


def test_one_thread_with_many_cores_warns_and_names_the_variables(monkeypatch):
    _report(monkeypatch, threads=1, cores=8, variables={'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    with pytest.warns(RuntimeWarning) as record:
        text = _host_threads.warn_if_threads_unused()
    assert len(record) == 1
    assert '8 cores' in text and 'OMP_NUM_THREADS=1' in text and 'MKL_NUM_THREADS=1' in text
    assert 'torch.set_num_threads(8)' in text


def test_one_thread_without_the_variables_still_warns(monkeypatch):
    _report(monkeypatch, threads=1, cores=4, variables={})
    with pytest.warns(RuntimeWarning):
        text = _host_threads.warn_if_threads_unused()
    assert 'set to one' in text


@pytest.mark.parametrize('threads, cores', [(8, 8), (2, 8), (1, 1)])
def test_no_warning_when_threads_match_or_one_core(monkeypatch, threads, cores):
    _report(monkeypatch, threads=threads, cores=cores, variables={'OMP_NUM_THREADS': '1'})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _host_threads.warn_if_threads_unused() is None


def test_report_reads_torch_and_the_environment(monkeypatch):
    _report(monkeypatch, threads=3, cores=12, variables={'MKL_NUM_THREADS': '3'})
    assert _host_threads.host_thread_report() == (3, 12, {'MKL_NUM_THREADS': '3'})
