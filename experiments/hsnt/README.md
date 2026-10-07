# Hyperspectral neutron (hsnt) examples

Scripts that exercise `mbirtorch.hsnt` on simulated Ni, Cu, Al data. Run them from this folder with the
mbirtorch environment active.

## Setup: generate the material basis

The experiments load `binaries/material_basis.npy`, the Bragg-edge attenuation spectra of Ni, Cu and Al. The file
is not in the repository (the `binaries/` folder is ignored by git). Generate it once:

```
./exp_0_generate_material_basis.sh
```

This creates a small conda env named `bem` holding ORNL's `braggedgemodeling` package and runs
`exp_0_generate_material_basis.py` in it. Do not install `braggedgemodeling` in the mbirtorch env: it breaks
torch's import.

## The examples

| script | what it does |
|---|---|
| `exp_1_denoise_simulated_data.py` | simulate a noisy hyperspectral projection, dehydrate it, rehydrate it, and plot the images and spectra against the ground truth |
| `exp_2_fast_hyper_recon_dehydrate_and_export.py` | dehydrate a simulated multi-view dataset, reconstruct each component, and export the result to `processed_data/` as HDF5 |
| `exp_3_fast_hyper_recon_import_and_rehydrate.py` | import the HDF5 file written by experiment 2 and rehydrate the reconstruction (run experiment 2 first) |
| `exp_4_hdf5_utils.py` | export and import hyperspectral and dehydrated datasets in HDF5, and compare file sizes |
| `exp_5_measured_transmission.py` | denoise a measured low-count transmission image (ORNL SNAP, Ni cylinder, 0.8 C) read from the Purdue data depot; runs on Gilbreth or Gautschi |

`plot_utils.py` holds the plotting helpers the scripts share. Outputs go to `processed_data/`, which git ignores.
