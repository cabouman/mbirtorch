"""Tests of mbirtorch.estimate_resources: memory as the reconstruction check computes it, the split parts, and missing data."""
import numpy as np
import pytest

import mbirtorch
from mbirtorch import resources


def _hexagonal_scan_model():
    """Return the cone model of the full-resolution hexagonal VoluMax scan, built from its parameters."""
    model = mbirtorch.ConeBeamModel((2000, 3024, 3024), np.linspace(0, 2 * np.pi, 2000, endpoint=False),
                                    source_detector_dist=1328.7902617139346, source_iso_dist=755.7925412076279)
    model.set_params(no_warning=True, verbose=0, delta_det_channel=0.139, delta_det_row=0.139,
                     det_row_offset=17.64817885996169, det_channel_offset=0.4501348796433,
                     recon_shape=(3024, 3024, 3024), delta_voxel=0.07906075642995405,
                     recon_slice_offset=-10.037973886904282)
    return model


def _small_model(geometry):
    """Return a small model on the CPU."""
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    if geometry == 'cone':
        model = mbirtorch.ConeBeamModel((24, 40, 32), angles, source_detector_dist=128.0,
                                        source_iso_dist=64.0, compile_mode='off')
    else:
        model = mbirtorch.ParallelBeamModel((24, 40, 32), angles, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


@pytest.mark.filterwarnings('ignore:Cone angle')
def test_memory_matches_the_cluster_check():
    # The reconstruction check on 4 H100 GPUs on Gautschi reported these needs on 2026-10-10, without weights.
    model = _hexagonal_scan_model()
    assert resources._gpu_memory_gib(model, 4, 'recon', weights=False) == pytest.approx(151.54, abs=0.01)
    assert resources._gpu_memory_gib(model, 4, 'direct', weights=False) == pytest.approx(111.76, abs=0.01)
    parts, _ = resources._cone_split_parts(model)
    assert [p.get_params('sinogram_shape')[1] for p in parts] == [1643, 1391]
    assert [p.get_params('recon_shape')[2] for p in parts] == [1647, 1393]
    assert resources._gpu_memory_gib(parts[0], 4, 'recon', weights=False) == pytest.approx(83.93, abs=0.01)


def test_cone_split_parts_match_a_real_split():
    model = _small_model('cone')
    sino = np.asarray(model.forward_project(np.ones(model.get_params('recon_shape'), np.float32)))
    _, recon_dict = model.recon_split_sino(sino, max_iterations=1, print_logs=False, logfile_path=None)
    parts, _ = resources._cone_split_parts(model)
    real = [recon_dict['model_params_top'], recon_dict['model_params_bottom']]
    for part, params in zip(parts, real):
        assert tuple(part.get_params('sinogram_shape')) == tuple(params['sinogram_shape'])
        assert tuple(part.get_params('recon_shape')) == tuple(params['recon_shape'])


def test_parallel_split_parts_match_a_real_split():
    model = _small_model('parallel')
    sino = np.asarray(model.forward_project(np.ones(model.get_params('recon_shape'), np.float32)))
    _, recon_dict = model.recon_split_sino(sino, max_iterations=1, print_logs=False, logfile_path=None,
                                           slices_per_part=14)
    num_parts = recon_dict['split_params']['num_parts']
    largest = max(p['sinogram_shape'][1] for p in recon_dict['model_params_parts'])
    # The estimate uses the bound recon_split_sino uses to choose the number of parts, which can
    # exceed the real largest part by a row when the parts are unequal.
    assert largest <= resources._parallel_largest_part_rows(40, num_parts, 5) <= largest + 1


def test_estimate_reports_and_leaves_the_model_unchanged():
    model = _small_model('cone')
    before = model.get_params('recon_shape')
    estimate = mbirtorch.estimate_resources(model, gpu_model='H100', num_gpus=2)
    assert model.get_params('recon_shape') == before
    for step in (estimate.direct, estimate.recon, estimate.split):
        assert step.gpu_memory_gib > 0
        assert step.fits is True
    assert 'full reconstruction' in str(estimate)
    assert mbirtorch.estimate_resources(model, gpu_memory_gb=2.0001).recon.fits is False


def test_missing_information_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(resources, '_load_speed_file', lambda: (None, 'the speed file could not be read'))
    estimate = mbirtorch.estimate_resources(_small_model('parallel'), gpu_model='unknown GPU')
    assert estimate.recon.gpu_memory_gib > 0
    assert estimate.recon.fits is None
    assert estimate.recon.time_minutes is None
    assert estimate.split.gpu_memory_gib is None
    assert 'not available' in str(estimate)
    assert 'translation' not in str(estimate)
