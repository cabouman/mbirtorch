"""Make a synthetic cone-beam sinogram for testing stream_fdk_cone.py, and check a
reconstruction against its phantom.

The phantom is the set of ellipsoids of ``make_synthetic_sinogram.py``.  It is defined relative
to the shape of the volume, so the same phantom serves any volume.  The script builds the
phantom one slab of slices at a time.  It forward projects each slab through an mbirtorch
cone-beam model of the slab and of the band of detector rows that the slab touches, and it adds
the result to these rows of the sinogram file.  The sum over the slabs is the forward
projection of the whole phantom.  The file is a ``.npy`` memory map, so the host holds only one
slab of the phantom and one band of the sinogram.

    python make_synthetic_cone_sinogram.py 1601,3048,3048 sino.npy --source-detector-dist 5080 --source-iso-dist 2540
    python make_synthetic_cone_sinogram.py --check recon.h5 --slices 5

The geometry options are those of ``stream_fdk_cone.py``, and the reconstruction must be given
the same options.

``--check`` reads a few slices of a reconstruction written by ``stream_fdk_cone.py`` and reports
the NRMSE of each slice against the phantom.  It reads only those slices, so it works on a
volume of any size.  FDK is an approximate reconstruction, and its error grows toward the two
ends of the rotation axis.
"""

import argparse
import time

import numpy as np

import stream_fdk_cone
from make_synthetic_sinogram import check, phantom_band
from stream_recon_parallel import GB

DEFAULT_SLAB_SLICES = 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('shape', nargs='?', type=stream_fdk_cone.shape_from_text, metavar='VIEWS,ROWS,CHANNELS',
                   help='sinogram shape to make')
    p.add_argument('path', nargs='?', help='output .npy file for the sinogram')
    stream_fdk_cone.add_geometry_arguments(p)
    p.add_argument('--slab-slices', type=int, default=DEFAULT_SLAB_SLICES,
                   help='slices of the phantom projected at a time (default %(default)s)')
    p.add_argument('--devices', help="device list for the projection, e.g. 'cpu' (default: the package chooses)")
    p.add_argument('--check', metavar='RECON',
                   help='compare this reconstruction file with the phantom instead of making a sinogram')
    p.add_argument('--slices', type=int, default=5, help='with --check: slices to compare (default %(default)s)')
    args = p.parse_args()
    if args.check is None:
        if args.shape is None or args.path is None:
            p.error('give VIEWS,ROWS,CHANNELS and an output path, or --check RECON')
        stream_fdk_cone.require_distances(p, args)
    return args


def make_sinogram(model, path, slab_slices):
    """Write the sinogram of the phantom for the scan of model to path, one slab of slices at a
    time."""
    sinogram_shape = tuple(int(s) for s in model.get_params('sinogram_shape'))
    num_recon_rows, num_recon_cols, num_slices = (int(s) for s in model.get_params('recon_shape'))
    # A new memory map holds zeros, and the bands of the slabs are added to it.
    sino = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=sinogram_shape)
    t0 = time.time()
    for slice_lo in range(0, num_slices, slab_slices):
        slice_hi = min(slice_lo + slab_slices, num_slices)
        row_lo, row_hi = stream_fdk_cone.band_rows(model, slice_lo, slice_hi)
        if row_hi > row_lo:
            part = stream_fdk_cone.part_model(model, row_lo, row_hi, slice_lo, slice_hi)
            slab = phantom_band(num_recon_rows, num_recon_cols, num_slices, slice_lo, slice_hi, part.torch_device)
            sino[:, row_lo:row_hi, :] += np.asarray(part.forward_project(slab))
            del slab, part
        print(f'  slices {slice_lo}-{slice_hi - 1} of {num_slices} projected to detector rows {row_lo}-{row_hi - 1}, '
              f'{time.time() - t0:.0f} s', flush=True)
    sino.flush()
    print(f'wrote {path}: shape {sino.shape}, {sino.nbytes / GB:.1f} GB')


def main():
    args = parse_args()
    if args.check:
        check(args.check, args.slices)
        return
    model = stream_fdk_cone.scan_model(args, args.shape)
    if args.devices:
        model.configure_devices(devices=args.devices.split(','))
    print(f'volume {tuple(int(s) for s in model.get_params("recon_shape"))}, '
          f'voxel pitch {model.get_params("delta_voxel"):.4g} ALU')
    make_sinogram(model, args.path, args.slab_slices)


if __name__ == '__main__':
    main()
