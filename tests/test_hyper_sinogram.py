"""The simulated hyperspectral scan: a material phantom projected and combined with spectra."""
import numpy as np
import pytest

import mbirtorch
import mbirtorch.hsnt as hsnt


def test_material_phantom_has_three_separate_cylinders():
    maps = hsnt.gen_material_phantom((32, 32, 16))
    assert maps.shape == (3, 32, 32, 16) and maps.dtype == np.float32
    assert all(m.max() == np.float32(0.08) and m.sum() > 0 for m in maps)
    assert np.all((maps > 0).sum(axis=0) <= 1)        # no voxel holds two materials


def test_hyper_sinogram_is_the_projections_times_the_spectra():
    """Without noise the counts equal the open beam times the transmission of the projected
    maps, and with noise they are a Poisson draw of that."""
    angles = np.linspace(0, np.pi, 6, endpoint=False)
    ct_model = mbirtorch.ParallelBeamModel((6, 8, 32), angles)
    ct_model.set_params(verbose=0)
    maps = hsnt.gen_material_phantom(ct_model.get_params('recon_shape'))
    basis = hsnt.synthetic_material_basis(20)
    assert basis.shape == (3, 20) and np.all(basis > 0)

    counts, open_beam, attenuation = hsnt.generate_hyper_sinogram(ct_model, maps, basis, dosage_rate=100,
                                                                  noisy=False)
    assert counts.shape == open_beam.shape == attenuation.shape == (6, 8, 32, 20)
    expected = np.stack([ct_model.forward_project(m) for m in maps], axis=-1) @ basis
    assert np.allclose(attenuation, expected, atol=1e-5)
    assert np.allclose(counts, 100 * np.exp(-expected), atol=1e-3)
    assert np.all(open_beam == 100)

    noisy, _, _ = hsnt.generate_hyper_sinogram(ct_model, maps, basis, dosage_rate=100, seed=0)
    assert np.all(noisy >= 0) and np.all(noisy == np.round(noisy)) and not np.allclose(noisy, counts)

    with pytest.raises(ValueError, match='must match'):
        hsnt.generate_hyper_sinogram(ct_model, maps[:2], basis)
