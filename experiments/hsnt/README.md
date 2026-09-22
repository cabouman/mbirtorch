# hsnt experiments

`binaries/material_basis.npy`: the three-material phantom's linear attenuation spectra (Ni, Cu, Al; shape (3, 1200),
float32) used by `demo/demo_10_hsnt.py` and `mbirtorch.hsnt.generate_hyper_data` / `generate_sphere_data`. The file
carries no wavelength axis; `mbirtorch.hsnt.simulate.material_basis_wavelengths(1200)` gives the calibrated one
(1.5099 + 0.0025196 * bin Angstrom, from the nickel edges).
