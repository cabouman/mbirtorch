# This script reconstructs Translation CT BGA data from the Zeiss scanner with the TCT direct reconstruction
# (TranslationModel.recon_tct_direct) and with MBIR.  The setup follows mbirjax/experiments/translation/TCT_BGA_physical_exp.py.
#
# Usage: python TCT_BGA_direct_recon_exp.py [path or URL of purdue_BGA_xrm.tgz] [--direct-only]
# The path defaults to the data depot.  The option --direct-only skips MBIR, for example for a quick test on a
# computer with a small GPU.

import mbirtorch as mt
import mbirtorch.preprocess as mtp
import numpy as np
import os
import sys
import time


def main(dataset_url="/depot/bouman/data/Translation/purdue_BGA_xrm.tgz", run_mbir=True):
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

    output_path = './output/'  # path to store output recon
    os.makedirs(output_path, exist_ok=True)

    # TCT direct reconstruction
    t0 = time.time()
    direct_recon = tct_model.recon_tct_direct(sino)
    print(f'TCT direct recon time: {time.time() - t0:.1f} s')
    mt.export_recon_hdf5(os.path.join(output_path, "TCT_BGA_direct_recon.h5"), direct_recon,
                         recon_dict=None, top_margin=0, bottom_margin=0)
    recons, data_dicts, titles = [direct_recon], [None], ['TCT direct recon']

    # MBIR reconstruction
    if run_mbir:
        t0 = time.time()
        mbir_recon, mbir_dict = tct_model.recon(sino, weights=weights, max_iterations=200,
                                                stop_threshold_change_pct=0.2)
        print(f'MBIR recon time: {time.time() - t0:.1f} s')
        mt.export_recon_hdf5(os.path.join(output_path, "TCT_BGA_mbir_recon.h5"), mbir_recon, recon_dict=mbir_dict,
                             top_margin=0, bottom_margin=0)
        recons.append(mbir_recon)
        data_dicts.append(mbir_dict)
        titles.append('MBIR recon')

    # Display results
    mt.slice_viewer(*[np.asarray(recon).transpose(0, 2, 1) for recon in recons], data_dicts=data_dicts,
                    vmin=0, vmax=0.5, title=' | '.join(titles), slice_axis=0)


if __name__ == '__main__':
    arguments = [a for a in sys.argv[1:] if a != '--direct-only']
    main(*arguments[:1], run_mbir='--direct-only' not in sys.argv[1:])
