"""A synthetic Ni, Cu, Al attenuation basis for the hsnt tests.

The spectra have the shape of neutron Bragg-edge spectra: an absorption term that grows with the wavelength,
plus coherent scattering that drops to zero at each Bragg edge of the FCC lattice (wavelength 2d_hkl). The two
coefficients per material are least-squares fits to the reference basis of experiments/hsnt (which needs a
conda-only package to generate); the fit is within 13 to 21 percent of it.
"""

import numpy as np

# element: FCC lattice constant in Angstroms, absorption slope, scattering scale
MATERIALS = {'Ni': (3.499, 0.133, 1.19), 'Cu': (3.597, 0.087, 0.65), 'Al': (4.046, 0.0073, 0.079)}
HKL = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0), (3, 3, 1), (4, 2, 0)]  # allowed FCC planes


def material_basis(num_wavelengths=1200, wavelengths=(1.5, 4.5)):
    """Return the (3, num_wavelengths) float32 spectra of Ni, Cu and Al on a uniform wavelength grid."""
    lam = np.linspace(*wavelengths, num_wavelengths)
    basis = []
    for a, absorption, scattering in MATERIALS.values():
        spectrum = absorption * lam
        for h, k, l in HKL:
            d = a / np.sqrt(h * h + k * k + l * l)
            spectrum = spectrum + scattering * (d / a) ** 2 * (lam / a) ** 2 * (lam < 2 * d)
        basis.append(spectrum)
    return np.asarray(basis, dtype=np.float32)
