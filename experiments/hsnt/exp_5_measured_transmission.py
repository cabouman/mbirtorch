"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script denoises a measured low-count transmission image: one view of a Ni cylinder at ORNL's SNAP
instrument with a 0.8 C proton charge. The data are on the Purdue data depot (readable from Gilbreth and Gautschi).
"""

import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from mbirtorch.hsnt import hyper_denoise

DATA_DIR = '/depot/bouman/data/ORNL/hsnt/old/Ni_single_view_temp'


def load_transmission(data_dir=DATA_DIR, charge='0_8c', sample='Ni_cylinder_projections'):
    """Return (transmission, wavelengths): the sample counts divided by the mean open beam, (rows, cols, bins)."""
    read_stack = lambda folder: np.stack([tifffile.imread(f) for f in sorted(glob.glob(os.path.join(folder, 'wave_idx_*.tif')))], axis=-1)
    counts = read_stack(os.path.join(data_dir, charge, sample))
    open_beam = np.mean([read_stack(d) for d in sorted(glob.glob(os.path.join(data_dir, charge, 'open_beam', 'observation_*')))], axis=0)
    transmission = np.divide(counts, open_beam, out=np.zeros_like(counts), where=open_beam > 0).astype(np.float32)
    wavelengths = np.load(os.path.join(data_dir, 'wave_angstrom.npy'))
    return transmission, wavelengths


def plot_pixel_spectra(wavelengths, noisy_data, denoised_data, dataset_type, pixel=(200, 200),
                       region=(slice(160, 350), slice(160, 350))):
    """Plot one pixel's spectrum and the mean spectrum over a region, raw and denoised."""
    plt.figure()
    plt.plot(wavelengths, noisy_data[pixel], '.', color='gray', markersize=1, label='raw pixel')
    plt.plot(wavelengths, denoised_data[pixel], '.', color='navy', markersize=1, label='denoised pixel')
    plt.plot(wavelengths, np.mean(noisy_data[region], axis=(0, 1)), '.', color='red', markersize=1,
             label='raw avg over pixels')
    plt.plot(wavelengths, np.mean(denoised_data[region], axis=(0, 1)), '.', color='green', markersize=1,
             label='denoised avg over pixels')
    plt.ylim(-0.5, 1.05)
    plt.grid(linestyle='--')
    plt.legend(markerscale=10, loc='best', ncol=2)
    plt.title(f'Pixel Spectra at ({pixel[0]},{pixel[1]}) using "{dataset_type}"', fontweight='bold', fontsize=12,
              y=1.02)


def main():
    start_time = time.time()

    # Parameters
    num_materials = 1
    max_steps = 300

    # Load the measured transmission and its wavelengths (about 3 GB of TIFFs; a few minutes)
    noisy_data, wavelengths = load_transmission()
    print(f'Loaded transmission {noisy_data.shape}, wavelengths {wavelengths.min():.2f} to {wavelengths.max():.2f} A')

    # Denoise using wrong (attenuation) and right (transmission) mode
    denoised_data = {dataset_type: hyper_denoise(noisy_data, dataset_type=dataset_type, num_materials=num_materials,
                                                 max_steps=max_steps)
                     for dataset_type in ('attenuation', 'transmission')}

    # Plot results for each mode
    for dataset_type, denoised in denoised_data.items():
        plot_pixel_spectra(wavelengths, noisy_data, denoised, dataset_type)

    print(f'Total time elapsed: {time.time() - start_time:.2f} seconds')

    plt.show()


if __name__ == "__main__":
    main()
