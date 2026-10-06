"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script demonstrates the use of dehydration and rehydration for hyperspectral data denoising.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import mbirtorch.hsnt as hsnt
from plot_utils import plot_images, plot_spectra

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    start_time = time.time()

    # Simulation parameters
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 250  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Denoiser parameters
    num_materials = None  # Number of materials
    verbose = 2  # Verbosity level

    # Display parameters
    display_wave_idx = 200  # Wavelength index of displayed images
    display_pix_idx = [10, 32]  # Pixel index [row, column] of displayed spectra
    vmax = 2.5  # Maximum pixel value for displayed images
    vmin = 0  # Minimum pixel value for displayed images
    y_lim_attenuation = (1, 2.7)  # (y_min, y_max) to set y-axis range for attenuation spectra
    y_lim_transmission = (0, 0.35)  # (y_min, y_max) to set y-axis range for transmission spectra

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al (run exp_0_generate_material_basis.py once)
    basis_path = os.path.join(SCRIPT_DIR, 'binaries', 'material_basis.npy')
    if not os.path.exists(basis_path):
        raise SystemExit(f'{basis_path} not found: run exp_0_generate_material_basis.py first')
    material_basis = np.load(basis_path)

    # Generate simulated noisy hyperspectral data and ground truth
    noisy_hyper_projection, _, gt_hyper_projection = hsnt.generate_hyper_data(
        material_basis,
        detector_rows=detector_rows,
        detector_columns=detector_columns,
        dosage_rate=dosage_rate,
        material_density=material_density,
        verbose=verbose)

    # Perform hyperspectral denoising (dehydrate + rehydrate)
    denoised_hyper_projection = hsnt.hyper_denoise(noisy_hyper_projection,
                                                   dataset_type=dataset_type,
                                                   num_materials=num_materials,
                                                   verbose=verbose)

    # Plot hyperspectral projections and spectra
    if verbose > 1:
        projections = {'Fig (a): Ground truth': gt_hyper_projection,
                       'Fig (b): Noisy': noisy_hyper_projection,
                       'Fig (c): Denoised': denoised_hyper_projection}
        plot_images(images=[projection[0, :, :, display_wave_idx] for projection in projections.values()],
                    titles=[f'{name} hyperspectral projection \n\nWavelength index: {display_wave_idx}'
                            for name in projections],
                    vmax=vmax, vmin=vmin)

        row, column = display_pix_idx
        noisy_spectrum = noisy_hyper_projection[0, row, column, :]
        denoised_spectrum = denoised_hyper_projection[0, row, column, :]

        plot_spectra(spectra=[noisy_spectrum, denoised_spectrum],
                     labels=['Noisy', 'Denoised'],
                     title='Single pixel spectra (attenuation) for noisy and denoised data',
                     x_label='wavelength index',
                     y_label='attenuation',
                     y_lim=y_lim_attenuation)

        plot_spectra(spectra=[np.exp(-noisy_spectrum), np.exp(-denoised_spectrum)],
                     labels=['Noisy', 'Denoised'],
                     title='Single pixel spectra (transmission) for noisy and denoised data',
                     x_label='wavelength index',
                     y_label='transmission',
                     y_lim=y_lim_transmission)

    print(f'Total time elapsed: {time.time() - start_time:.2f} seconds')

    plt.show()


if __name__ == "__main__":
    main()
