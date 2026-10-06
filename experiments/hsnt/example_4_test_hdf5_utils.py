"""
Hyperspectral Dehydration & Rehydration
---------------------------------------

This script shows the workflow for exporting, importing, and handling hyperspectral neutron datasets in HDF5 format.
It showcases both dehydrated and rehydrated formats and the storage efficiency achieved with the dehydrated format.
"""

import os

import numpy as np

import mbirtorch.hsnt as hsnt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Set output path
    output_path = os.path.join(SCRIPT_DIR, 'processed_data')  # Path to export data
    os.makedirs(output_path, exist_ok=True)

    # 1. Generate random sample dehydrated data
    hsnt_dehydrated = [
        np.random.rand(3, 64, 64, 5).astype(np.float32),  # subspace_data
        np.random.rand(5, 1000).astype(np.float32),  # subspace_basis
        "attenuation",  # dataset_type
    ]
    print("Dehydrated data shape:", hsnt_dehydrated[0].shape)

    # 2. Rehydrate to get full hyperspectral data
    hsnt_rehydrated = hsnt.rehydrate(hsnt_dehydrated)
    print("Rehydrated data shape:", hsnt_rehydrated.shape)

    # 3. Create metadata dictionaries
    metadata_dehydrated = hsnt.create_hsnt_metadata(
        dataset_name="hsnt_dehydrated",
        dataset_type="attenuation",
        dataset_modality="hyperspectral neutron",
        wavelengths=None,
        alu_unit="mm",
        alu_value=1.0,
        dataset_geometry="parallel",
        angles=np.array([0, 45, 90]),
        delta_det_channel=0.5,
        delta_det_row=0.5,
        det_channel_offset=0.0,
        source_detector_dist=1000.0,
        source_iso_dist=500.0,
    )

    metadata_rehydrated = metadata_dehydrated.copy()
    metadata_rehydrated["dataset_name"] = "hsnt_rehydrated"

    # 4. Export datasets into separate files
    filename_dehydrated = os.path.join(output_path, "hsnt_dehydrated.h5")
    hsnt.export_hsnt_data_hdf5(filename_dehydrated, hsnt_dehydrated, metadata_dehydrated)

    filename_rehydrated = os.path.join(output_path, "hsnt_rehydrated.h5")
    hsnt.export_hsnt_data_hdf5(filename_rehydrated, hsnt_rehydrated, metadata_rehydrated)

    # 5. Import datasets back
    dehydrated_imported, metadata_dehydrated_imported = hsnt.import_hsnt_data_hdf5(filename_dehydrated)
    rehydrated_imported, metadata_rehydrated_imported = hsnt.import_hsnt_data_hdf5(filename_rehydrated)

    print("\nImported dehydrated data shape:", dehydrated_imported[0].shape)
    print("--Imported metadata for dehydrated:", metadata_dehydrated_imported)
    print("\nImported rehydrated data shape:", rehydrated_imported.shape)
    print("--Imported metadata for rehydrated:", metadata_rehydrated_imported)

    # 6. Compare file sizes
    size_dehydrated = os.path.getsize(filename_dehydrated)
    size_rehydrated = os.path.getsize(filename_rehydrated)

    print("\nDehydrated file size:", size_dehydrated, "bytes")
    print("Rehydrated file size:", size_rehydrated, "bytes")
    print("Compression ratio (rehydrated / dehydrated):", np.round(size_rehydrated / size_dehydrated, 3))


if __name__ == "__main__":
    main()
