# hsnt experiments

`compare_solvers.py`: the NNAL solvers (joint Newton, multiplicative, block Newton) against the L2 baseline on one
low-dose projection of the three-material phantom, with each fit's NNAL and L2 losses and the recovered spectra and
maps. The public-API example is `demo/demo_13_hsnt.py`.

The phantom's linear attenuation spectra (Ni, Cu, Al; shape (3, 1200), float32) ship with the package:
`mbirtorch.hsnt.load_material_basis()` returns them with their calibrated wavelength axis (1.5099 + 0.0025196 * bin
Angstrom, from the nickel edges).
