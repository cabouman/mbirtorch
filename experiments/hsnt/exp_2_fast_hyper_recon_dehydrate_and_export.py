"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

Experiments 2 and 3 demonstrate the use of dehydration and rehydration for fast hyperspectral reconstruction.
A simulated hyperspectral neutron dataset containing three materials (Ni, Cu, and Al) is used for the purpose.
This script - experiment 2 - performs dehydration followed by MBIR and then exports the dehydrated reconstructions.
"""

import os
import time

import numpy as np

import mbirtorch as mt
import mbirtorch.hsnt as hsnt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    start_time = time.time()

    # Set output path and dataset name
    output_path = os.path.join(SCRIPT_DIR, 'processed_data')  # Path to export data
    dataset_name = "hsnt_dehydrated_recons"  # Name of the output dataset
    os.makedirs(output_path, exist_ok=True)

    # Simulation parameters
    num_angles = 16  # Number of view angles
    detector_rows = 64  # Number of rows in the detector
    detector_columns = 64  # Number of columns in the detector
    dosage_rate = 300  # Neutron dosage rate
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}  # Define material density (vol. fraction)
    dataset_type = 'attenuation'  # Choose between 'attenuation' or 'transmission'

    # Fast hyperspectral reconstruction parameters
    num_materials = 3  # Number of materials
    recon_snr_db = 20  # Assumed SNR for the reconstruction
    verbose = 2  # Verbosity level

    # Fix seed for random number generation
    np.random.seed(129)

    # Load theoretical linear attenuation coefficients for Ni, Cu, and Al (run exp_0_generate_material_basis.py once)
    basis_path = os.path.join(SCRIPT_DIR, 'binaries', 'material_basis.npy')
    if not os.path.exists(basis_path):
        raise SystemExit(f'{basis_path} not found: run exp_0_generate_material_basis.py first')
    material_basis = np.load(basis_path)

    # Generate simulated noisy hyperspectral projection data
    hsnt_data, angles, _ = hsnt.generate_hyper_data(material_basis,
                                                    num_angles=num_angles,
                                                    detector_rows=detector_rows,
                                                    detector_columns=detector_columns,
                                                    dosage_rate=dosage_rate,
                                                    material_density=material_density,
                                                    verbose=verbose)

    # MBIR model setup
    ct_model = mt.ParallelBeamModel((num_angles, detector_rows, detector_columns), angles)
    ct_model.set_params(snr_db=recon_snr_db, verbose=0)

    # Perform dehydration
    subspace_data, subspace_basis, dataset_type = hsnt.dehydrate(hsnt_data, dataset_type=dataset_type,
                                                                 num_materials=num_materials, mode='stream',
                                                                 chunk_pixels=16384, verbose=verbose)

    # Perform MBIR on each subspace component
    subspace_recons = []
    for idx in range(subspace_data.shape[-1]):
        print(f"Reconstructing data for subspace index: {idx}")
        subspace_recon, _ = ct_model.recon(subspace_data[:, :, :, idx])
        subspace_recons.append(subspace_recon)
    subspace_recons = np.stack(subspace_recons, axis=-1)

    # Export dehydrated reconstructions into HDF5 file
    hsnt_dehydrated_recons = [subspace_recons, subspace_basis, dataset_type]
    metadata = hsnt.create_hsnt_metadata(dataset_name=dataset_name)
    filename = os.path.join(output_path, dataset_name + ".h5")
    hsnt.export_hsnt_data_hdf5(filename, hsnt_dehydrated_recons, metadata)

    print(f'Total time elapsed: {time.time() - start_time:.2f} seconds')


if __name__ == "__main__":
    main()
