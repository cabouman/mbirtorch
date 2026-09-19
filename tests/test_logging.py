"""Tests for the run logging: the log file, the in-memory copy in
recon_dict, two live models keeping their logs apart, and merge_log_files.
The behavior matches mbirjax.
"""

import logging
import os

import numpy as np
import pytest

import mbirtorch


@pytest.fixture()
def small_parallel_case():
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 12, 16), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=1)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        tuple(model.get_params('recon_shape')))
    sino = model.forward_project(phantom)
    return model, sino


def _small_parallel_model():
    """A model matching the small_parallel_case fixture, for the tests that
    need two models of one class alive at the same time."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 12, 16), angles)
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=1)
    return model


def test_recon_writes_log_file_and_recon_log(tmp_path, small_parallel_case):
    model, sino = small_parallel_case
    logpath = os.path.join(str(tmp_path), 'run.log')
    recon, recon_dict = model.recon(sino, max_iterations=2, logfile_path=logpath)

    assert os.path.exists(logpath)
    with open(logpath) as f:
        content = f.read()
    assert 'MBIRTorch Version' in content
    assert 'After iteration' in content
    # The in-memory copy matches the file.
    assert recon_dict['recon_log'].strip() == content.strip()
    assert 'Reconstruction completed' in recon_dict['notes']

    # A second call continuing the same run appends to the log it already set
    # up rather than starting a new one, so both passes are in the one file
    # and in the one in-memory copy.
    _, second_dict = model.recon(sino, init_recon=recon, max_iterations=3,
                                 first_iteration=2, logfile_path=logpath)
    assert second_dict['recon_log'].count('MBIRTorch Version') == 2
    assert open(logpath).read().count('MBIRTorch Version') == 2


def test_two_live_models_keep_their_logs_apart(tmp_path, small_parallel_case):
    """Two models of the same class, both alive, keep separate logs.

    Setting up the log of the second model must not take over the file or the
    in-memory copy of the first: the first model goes on logging afterwards,
    and those lines belong to its own log.
    """
    first, sino = small_parallel_case
    second = _small_parallel_model()
    first_path = os.path.join(str(tmp_path), 'first.log')
    second_path = os.path.join(str(tmp_path), 'second.log')

    # The iteration lines name the iteration limit of the run that wrote them,
    # which is what tells the three passes below apart in the log text.
    recon, _ = first.recon(sino, max_iterations=1, logfile_path=first_path)
    _, second_dict = second.recon(sino, max_iterations=3,
                                  logfile_path=second_path)
    # The first model logs again, with the second model still alive.
    _, first_dict = first.recon(sino, init_recon=recon, max_iterations=2,
                                first_iteration=1, logfile_path=first_path)

    first_file = open(first_path).read()
    second_file = open(second_path).read()
    # Each log has its own model's lines ...
    assert 'of a max of 1' in first_file and 'of a max of 2' in first_file
    assert 'of a max of 3' in second_file
    assert 'of a max of 2' in first_dict['recon_log']
    assert 'of a max of 3' in second_dict['recon_log']
    # ... and none of the other model's.
    assert 'of a max of 3' not in first_file
    assert 'of a max of 3' not in first_dict['recon_log']
    assert 'of a max of 1' not in second_file and 'of a max of 2' not in second_file
    assert 'of a max of 1' not in second_dict['recon_log']
    # Both of the first model's passes are in its log and its in-memory copy,
    # and the second model's log holds its one pass only.
    assert first_dict['recon_log'].count('MBIRTorch Version') == 2
    assert first_file.count('MBIRTorch Version') == 2
    assert second_dict['recon_log'].count('MBIRTorch Version') == 1
    assert second_file.count('MBIRTorch Version') == 1


def test_the_log_file_is_closed_when_the_run_returns(tmp_path,
                                                     small_parallel_case):
    """A finished run does not hold its log file open.

    An open file handler keeps writing to the file it opened even after that
    file is deleted, and on Windows it blocks the delete outright, so the runs
    that merge and delete the logs of their parts depend on this.
    """
    model, sino = small_parallel_case
    logpath = os.path.join(str(tmp_path), 'closed.log')
    model.recon(sino, max_iterations=1, logfile_path=logpath)

    assert not any(isinstance(h, logging.FileHandler)
                   for h in model.logger.handlers)
    # Nothing holds the file, so it can be removed and written again.
    os.remove(logpath)
    model.recon(sino, max_iterations=1, logfile_path=logpath)
    assert 'After iteration' in open(logpath).read()


def test_merge_log_files_gathers_the_parts_and_removes_them(tmp_path):
    a = os.path.join(str(tmp_path), 'a.log')
    b = os.path.join(str(tmp_path), 'b.log')
    merged = os.path.join(str(tmp_path), 'merged.log')
    open(a, 'w').write('alpha\n')
    open(b, 'w').write('beta\n')
    mbirtorch.merge_log_files(merged, [('first', a), ('missing', '/nonexistent'),
                                       ('second', b)])
    content = open(merged).read()
    assert 'alpha' in content and 'beta' in content
    assert 'missing' not in content
    # The temp files are removed after the merge.
    assert not os.path.exists(a) and not os.path.exists(b)

    # With no part that exists there is nothing to merge, and no file is left
    # behind.
    empty = os.path.join(str(tmp_path), 'empty.log')
    mbirtorch.merge_log_files(empty, [('only', os.path.join(str(tmp_path), 'gone.log'))])
    assert not os.path.exists(empty)
