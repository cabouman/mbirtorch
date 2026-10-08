# This script reconstructs Translation CT BGA data from the Zeiss scanner with two direct methods:
# the TCT direct reconstruction (TranslationModel.recon_tct_direct) and the FDK direct recon.
# The setup follows mbirjax/experiments/translation/TCT_BGA_physical_exp.py.

import mbirtorch as mt
import mbirtorch.preprocess as mtp
import numpy as np
import os
import sys
import time


def main(dataset_url="/depot/bouman/data/Translation/purdue_BGA_xrm.tgz"):
    # Download (or copy) and extract data
    download_dir = "./purdue_BGA_data"
    dataset_dir = mt.download_and_extract(dataset_url, download_dir)

    # Load and preprocess data (returns a ready-to-reconstruct model and a data-specific weight mask)
    sino, tct_model, weights = mtp.zeiss_tct.get_sino_and_model(dataset_dir)

    recon_shape = tct_model.get_params('recon_shape')
    voxel_row_aspect = tct_model.get_params('voxel_row_aspect')

    # Set parameters for recon
    tct_model.set_params(positivity_flag=True)
    tct_model.set_params(voxel_row_aspect=voxel_row_aspect / 2)
    tct_model.set_params(recon_shape=(recon_shape[0] * 2, 2900, 2500))
    tct_model.set_params(partition_sequence=[0, 2, 4, 6])
    tct_model.set_params(qggmrf_nbr_wts=[0.1, 1, 1])
    tct_model.print_params()

    # TCT direct reconstruction
    t0 = time.time()
    tct_direct_recon = tct_model.recon_tct_direct(sino, weights=weights)
    print(f'TCT direct recon time: {time.time() - t0:.1f} s')

    # FDK direct reconstruction
    t0 = time.time()
    fdk_recon, fdk_dict = tct_model.recon(sino, weights=weights, max_iterations=0)
    print(f'FDK recon time: {time.time() - t0:.1f} s')

    # Save reconstruction results
    output_path = './output/'  # path to store output recon
    os.makedirs(output_path, exist_ok=True)
    mt.export_recon_hdf5(os.path.join(output_path, "TCT_BGA_tct_direct_recon.h5"), tct_direct_recon,
                         recon_dict=None, top_margin=0, bottom_margin=0)
    mt.export_recon_hdf5(os.path.join(output_path, "TCT_BGA_fdk_recon.h5"), fdk_recon, recon_dict=fdk_dict,
                         top_margin=0, bottom_margin=0)

    # Display results
    mt.slice_viewer(fdk_recon.transpose(0, 2, 1), np.asarray(tct_direct_recon).transpose(0, 2, 1),
                    data_dicts=[fdk_dict, None], vmin=0, vmax=0.5,
                    title='FDK recon (left) and TCT direct recon (right)', slice_axis=0)


if __name__ == '__main__':
    # Optional argument: path or URL of purdue_BGA_xrm.tgz (defaults to the lab depot path)
    main(*sys.argv[1:2])
