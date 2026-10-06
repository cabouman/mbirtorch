"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This mini script tests the performance of the algorithm for low quality transmission data.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from mbirtorch.hsnt import hyper_denoise

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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

    # Load noisy transmission data and corresponding wavelength values
    input_path = os.path.join(SCRIPT_DIR, "input_data")
    noisy_data = np.load(os.path.join(input_path, "test_transmission_data_0.8C.npy"))
    wavelengths = np.load(os.path.join(input_path, "test_wavelengths_0.8C.npy"))

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
