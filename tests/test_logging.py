"""Tests for the run logging: the log file, the in-memory copy in
recon_dict, console silencing, and the merged logs of the composite runs
(split_sino_recon and recon_plastic_metal).  The behavior matches mbirjax.
"""

import logging
import os

import numpy as np
import pytest

import mbirtorch
import mbirtorch.preprocess as mtp


def _device_line(recon_log):
    """The one line of a run log that reports the devices."""
    lines = [line for line in recon_log.splitlines()
             if line.startswith('Reconstruction devices:')]
    assert len(lines) == 1, lines
    return lines[0]


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


@pytest.fixture()
def small_cone_case():
    cell = (12, 16, 16)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                    source_iso_dist=2 * cell[2])
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=1)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        tuple(model.get_params('recon_shape')))
    sino = model.forward_project(phantom)
    return model, sino


def test_recon_writes_log_file_and_recon_log(tmp_path, small_parallel_case):
    model, sino = small_parallel_case
    logpath = os.path.join(str(tmp_path), 'run.log')
    _, recon_dict = model.recon(sino, max_iterations=2, logfile_path=logpath)

    assert os.path.exists(logpath)
    with open(logpath) as f:
        content = f.read()
    assert 'MBIRTorch Version' in content
    assert 'After iteration' in content
    # The in-memory copy matches the file.
    assert recon_dict['recon_log'].strip() == content.strip()
    assert 'Reconstruction completed' in recon_dict['notes']


def test_resumed_recon_sets_up_the_log(tmp_path, small_parallel_case):
    """A run started at a nonzero iteration, on a model that has not logged
    yet, still gets its log file and its recon_log."""
    model, sino = small_parallel_case
    logpath = os.path.join(str(tmp_path), 'resume.log')
    init = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        tuple(model.get_params('recon_shape')))
    _, recon_dict = model.recon(sino, init_recon=init, max_iterations=2,
                                first_iteration=1, logfile_path=logpath)

    assert os.path.exists(logpath)
    assert 'After iteration' in open(logpath).read()
    assert 'After iteration' in recon_dict['recon_log']


def test_continuing_run_keeps_one_log(tmp_path, small_parallel_case):
    """A second call in the same session continuing the same run appends to
    the log it already set up, rather than starting a new one."""
    model, sino = small_parallel_case
    logpath = os.path.join(str(tmp_path), 'run.log')
    recon, _ = model.recon(sino, max_iterations=1, logfile_path=logpath)
    _, recon_dict = model.recon(sino, init_recon=recon, max_iterations=1,
                                first_iteration=1, logfile_path=logpath)
    # Both passes are in the one log; the second call did not truncate it.
    assert recon_dict['recon_log'].count('MBIRTorch Version') == 2
    # The file has both passes too, not just the first.
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


def test_verbose_zero_writes_no_log_file(tmp_path, small_parallel_case):
    """At verbose=0 a run with nothing to report writes no file at all,
    rather than leaving an empty one behind."""
    model, sino = small_parallel_case
    model.set_params(no_warning=True, verbose=0)
    logpath = os.path.join(str(tmp_path), 'quiet.log')
    # stop_threshold_change_pct=0 runs every iteration, so the run cannot log
    # the warning it gives when it stops early: at verbose=0 a warning is the
    # one thing that would (rightly) create the file.
    _, recon_dict = model.recon(sino, max_iterations=1, logfile_path=logpath,
                                stop_threshold_change_pct=0)
    assert not os.path.exists(logpath)
    assert os.listdir(str(tmp_path)) == []
    assert 'After iteration' not in recon_dict['recon_log']


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


def test_prox_map_loop_keeps_one_growing_log(tmp_path, small_parallel_case):
    """A Plug-and-Play loop reuses its initialization after the first pass,
    so all its passes land in one log instead of each erasing the last."""
    model, sino = small_parallel_case
    logpath = os.path.join(str(tmp_path), 'prox.log')
    prox_input = np.zeros(tuple(model.get_params('recon_shape')),
                          dtype=np.float32)
    _, first = model.prox_map(prox_input, sino, max_iterations=1,
                              logfile_path=logpath, do_initialization=True)
    for _ in range(2):
        prox_input, last = model.prox_map(prox_input, sino, max_iterations=1,
                                          logfile_path=logpath,
                                          do_initialization=False)

    # One run header for the whole loop, and every pass is still in the log.
    assert last['recon_log'].count('MBIRTorch Version') == 1
    assert last['recon_log'].count('After iteration') > \
        first['recon_log'].count('After iteration')
    assert 'After iteration' in open(logpath).read()


def test_device_report_names_the_settled_layout():
    """The device line reports the layout the run actually uses, in the
    'N x PLATFORM (sharded)' form."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 11, 16), angles)
    model.configure_devices(devices=['cpu', 'cpu'])
    model.set_params(no_warning=True, verbose=1)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        tuple(model.get_params('recon_shape')))
    sino = model.forward_project(phantom)
    _, recon_dict = model.recon(sino, max_iterations=1, logfile_path=None)

    # The pieces are asserted rather than the whole sentence, so rewording the
    # line does not fail a test that is about what the line reports.
    line = _device_line(recon_dict['recon_log'])
    assert '2 x CPU' in line


def test_device_line_reflects_a_layout_chosen_during_the_run(monkeypatch):
    """A run that widens its layout reports the widened one.

    This is the multi-GPU case, where the automatic choice spreads the run
    across devices after it starts; standing in for it here on CPU.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 12, 16), angles)
    model.set_params(no_warning=True, verbose=1)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        tuple(model.get_params('recon_shape')))
    sino = model.forward_project(phantom)

    # The stand-in widens ONCE, as the real policy does: it settles at the
    # top of the run and every later call on the same shapes -- including the
    # one the nested direct reconstruction makes -- returns the settled
    # layout.  Re-installing on each call would re-place arrays the run is
    # already holding.
    settled = []

    def widen(**call_arrays):
        if not settled:
            model._install_device_layout(['cpu', 'cpu'])
            settled.append(True)
        return None

    monkeypatch.setattr(model, '_apply_device_policy', widen)
    _, recon_dict = model.recon(sino, max_iterations=1, logfile_path=None)

    assert 'Reconstruction devices: 2 x CPU' in recon_dict['recon_log']
    assert '1 x CPU' not in recon_dict['recon_log']


def test_explicit_devices_drop_an_earlier_automatic_search(monkeypatch):
    """A layout the caller placed is not explained as a search result.

    The automatic path records the device counts it turned down, for the
    device line.  Once configure_devices places the model by hand, that
    record describes a layout the run is no longer in.
    """
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, 12, 16), angles)
    model.set_params(no_warning=True, verbose=1)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
        tuple(model.get_params('recon_shape')))
    sino = model.forward_project(phantom)

    def settle_with_a_rejection(**call_arrays):
        # What the automatic path does when a wider count does not fit:
        # keep the current devices and record the count turned down.
        return model._settle([model.torch_device], None,
                             [(2, 'needs more memory than is free')])

    monkeypatch.setattr(model, '_apply_device_policy', settle_with_a_rejection)
    _, automatic = model.recon(sino, max_iterations=1, logfile_path=None)
    assert 'rejected' in _device_line(automatic['recon_log'])

    monkeypatch.undo()
    model.configure_devices(devices=['cpu'])
    # The record is dropped when the caller takes the layout over ...
    assert model.device_choice_rejections == []
    _, explicit = model.recon(sino, max_iterations=1, logfile_path=None)
    assert 'rejected' not in _device_line(explicit['recon_log'])

    # ... and a record that reached the line some other way still would not be
    # reported, because this layout did not come from a search.
    model.device_choice_rejections = [(2, 'needs more memory than is free')]
    _, still_explicit = model.recon(sino, max_iterations=1, logfile_path=None)
    assert 'rejected' not in _device_line(still_explicit['recon_log'])


def test_recon_logfile_none_writes_no_file(tmp_path, small_parallel_case, monkeypatch):
    model, sino = small_parallel_case
    monkeypatch.chdir(str(tmp_path))
    _, recon_dict = model.recon(sino, max_iterations=1, logfile_path=None)
    assert os.listdir(str(tmp_path)) == []
    # The in-memory copy is still recorded.
    assert 'After iteration' in recon_dict['recon_log']


def test_print_logs_false_silences_the_logger(small_parallel_case, capsys):
    model, sino = small_parallel_case
    model.recon(sino, max_iterations=1, logfile_path=None, print_logs=False)
    captured = capsys.readouterr()
    assert 'After iteration' not in captured.out
    assert 'After iteration' not in captured.err


def test_the_run_log_does_not_reach_the_root_logger(small_parallel_case):
    """An application's own logging setup does not undo print_logs=False.

    The buffer, console, and file handlers are the whole intended output, so
    a caller that ran logging.basicConfig does not get the run log too.
    """
    model, sino = small_parallel_case
    collected = []

    class _Collector(logging.Handler):
        def emit(self, record):
            collected.append(record)

    # A fresh logger propagates, which is the state the model's setup has to
    # change; assert against that rather than against whatever an earlier call
    # on this model left behind.
    model.logger.propagate = True
    logger_name = model.logger.name
    root = logging.getLogger()
    handler = _Collector()
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        model.recon(sino, max_iterations=1, logfile_path=None, print_logs=False)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)

    from_model = [r.getMessage() for r in collected if r.name == logger_name]
    assert from_model == []


def test_saved_recon_carries_the_log(tmp_path, small_parallel_case):
    model, sino = small_parallel_case
    recon, recon_dict = model.recon(sino, max_iterations=1, logfile_path=None)
    path = os.path.join(str(tmp_path), 'r.h5')
    model.save_recon_hdf5(path, recon, recon_dict=recon_dict)
    _, loaded = mbirtorch.TomographyModel.load_recon_hdf5(path)
    assert 'After iteration' in loaded['recon_log']


def test_merge_log_files(tmp_path):
    a = os.path.join(str(tmp_path), 'a.log')
    b = os.path.join(str(tmp_path), 'b.log')
    merged = os.path.join(str(tmp_path), 'merged.log')
    open(a, 'w').write('alpha\n')
    open(b, 'w').write('beta\n')
    mbirtorch.merge_log_files(merged, [('first', a), ('missing', '/nonexistent'),
                                       ('second', b)])
    content = open(merged).read()
    assert '======== first ========' in content and 'alpha' in content
    assert '======== second ========' in content and 'beta' in content
    assert 'missing' not in content
    # The closing line names the merged file, not the removed temps.
    assert content.rstrip().endswith('Merged logs written to ' + merged)
    # The temp files are removed after the merge.
    assert not os.path.exists(a) and not os.path.exists(b)


def test_merge_log_files_no_parts_writes_nothing(tmp_path):
    merged = os.path.join(str(tmp_path), 'merged.log')
    mbirtorch.merge_log_files(merged, [('only', os.path.join(str(tmp_path), 'gone.log'))])
    assert not os.path.exists(merged)


def test_split_sino_recon_merges_half_logs(tmp_path, small_cone_case):
    model, sino = small_cone_case
    logpath = os.path.join(str(tmp_path), 'split.log')
    model.split_sino_recon(sino, half_overlap=3, max_iterations=1, logfile_path=logpath)
    content = open(logpath).read()
    assert '======== split_sino_recon: top half ========' in content
    assert '======== split_sino_recon: bottom half ========' in content
    assert content.count('After iteration') >= 2
    # The merged file holds each half's whole log, not just its header.
    assert content.count('MBIRTorch Version') == 2
    assert content.count('Reconstruction devices:') == 2
    assert content.rstrip().endswith('Merged logs written to ' + logpath)
    # The per-half temp files are gone.
    assert not os.path.exists(logpath + '.top') and not os.path.exists(logpath + '.bot')


def test_recon_plastic_metal_merges_pass_logs(tmp_path, small_cone_case):
    model, sino = small_cone_case
    phantom = np.zeros(tuple(model.get_params('recon_shape')), dtype=np.float32)
    phantom[4:12, 4:12, 4:12] = 0.02
    phantom[6:9, 6:9, 6:9] = 0.2
    sino = model.forward_project(phantom)
    weights = mbirtorch.gen_weights(sino / max(1e-6, sino.max()),
                                    weight_type='transmission_root')
    logpath = os.path.join(str(tmp_path), 'mar.log')
    model.recon_plastic_metal(sino, weights, num_BH_iterations=2, max_iterations=1,
                            num_metal=1, verbose=0, logfile_path=logpath)
    content = open(logpath).read()
    assert '======== recon_plastic_metal: BH pass 1 ========' in content
    assert '======== recon_plastic_metal: BH pass 2 ========' in content
