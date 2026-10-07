"""
Generate the Ni, Cu, Al material basis
--------------------------------------

Writes binaries/material_basis.npy, the linear attenuation spectra of FCC Ni, Cu and Al at 300 K on 1200
wavelengths from 1.5 to 4.5 Angstroms, shape (3, 1200), float32. The example scripts in this folder load it.

The spectra are Bragg-edge neutron cross sections from ORNL's braggedgemodeling package (imported as ``bem``),
which is on conda-forge only and is not a dependency of mbirtorch. Run this script through
exp_0_generate_material_basis.sh, which runs it in a conda env of its own; the binaries folder is ignored by git,
so run it once after cloning.
"""

import os

import numpy as np
from bem.matter import Atom, Lattice, Structure
from bem import xscalc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'binaries', 'material_basis.npy')

# Element, FCC lattice constant in Angstroms, and the factor that turns the cross section per unit cell
# (8 atoms, barns) into a linear attenuation coefficient in the units the examples use.
MATERIALS = [('Ni', 3.499, 0.0693), ('Cu', 3.597, 0.075), ('Al', 4.046, 0.0603)]
TEMPERATURE = 300.0  # K
WAVELENGTHS = np.linspace(1.5, 4.5, 1200)  # Angstroms


def generate_material_basis(wavelengths=WAVELENGTHS, temperature=TEMPERATURE):
    """Return the (3, len(wavelengths)) float32 attenuation spectra of FCC Ni, Cu and Al."""
    basis = np.zeros((len(MATERIALS), len(wavelengths)), dtype=np.float32)
    for i, (element, a, factor) in enumerate(MATERIALS):
        atoms = [Atom(element, (0, 0, 0)), Atom(element, (0.5, 0.5, 0)),
                 Atom(element, (0.5, 0, 0.5)), Atom(element, (0, 0.5, 0.5))]
        lattice = Lattice(a=a, b=a, c=a, alpha=90.0, beta=90.0, gamma=90.0)
        structure = Structure(atoms, lattice, sgid=225)
        basis[i] = xscalc.XSCalculator(structure, temperature).xs(wavelengths) * factor / 8
    return basis


if __name__ == '__main__':
    basis = generate_material_basis()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, basis)
    print(f'Wrote {OUTPUT_PATH}: shape {basis.shape}, {basis.dtype}')
