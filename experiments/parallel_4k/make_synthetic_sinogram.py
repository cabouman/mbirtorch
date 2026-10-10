"""Make a synthetic parallel-beam sinogram for testing the large reconstruction scripts, and
check a reconstruction against its phantom.

The phantom is a set of ellipsoids, a low dynamic range variant of the Shepp-Logan head.  It
is defined in coordinates where the reconstruction's inscribed circle has radius 1 and the
slice axis spans [-1, 1], so the same phantom serves any volume shape.  Its sinogram is not
computed from a formula.  Each band of slices is built as a volume and forward projected
through an mbirtorch parallel beam model of that band, so the sinogram follows the library's
own geometry conventions by construction.  In parallel beam a detector row is a slice, so the
bands are exact pieces of the whole sinogram.  The file is written band by band as a ``.npy``
memory map, so it never has to fit in host memory.

    python make_synthetic_sinogram.py 900,4096,4096 sino_4k.npy
    python make_synthetic_sinogram.py --check recon_4k.h5 --slices 5

The reconstruction scripts must be given the same angles: equally spaced over
``--angle-span`` degrees (default 180), or the ``--angles`` file.

``--check`` reads a few slices of a reconstruction written by ``recon_parallel_split.py`` or
``stream_recon_parallel.py`` (an HDF5 file in the ``export_recon_hdf5`` layout, or the
streaming harness's ``.npy`` file) and reports each slice's NRMSE against the phantom.  It
reads only those slices, so it works on a volume of any size.
"""

import argparse
import math
import time

import numpy as np
import torch

import mbirtorch

GB = 1e9

# Ellipsoids as (x0, y0, z0, a, b, c, angle_deg, value).  The head is 1.0 in its shell and
# 0.2 inside, with features of 0.1 to 0.3.
ELLIPSOIDS = [
    (0.0, 0.0, 0.0, 0.69, 0.92, 0.90, 0.0, 1.0),
    (0.0, -0.0184, 0.0, 0.6624, 0.874, 0.88, 0.0, -0.8),
    (0.22, 0.0, 0.0, 0.11, 0.31, 0.21, -18.0, -0.1),
    (-0.22, 0.0, 0.0, 0.16, 0.41, 0.22, 18.0, -0.1),
    (0.0, 0.35, -0.15, 0.21, 0.25, 0.5, 0.0, 0.2),
    (0.0, 0.1, 0.25, 0.046, 0.046, 0.046, 0.0, 0.3),
    (0.0, -0.1, 0.25, 0.046, 0.046, 0.046, 0.0, 0.3),
    (-0.08, -0.605, 0.0, 0.046, 0.023, 0.02, 0.0, 0.3),
    (0.0, -0.606, 0.0, 0.023, 0.023, 0.02, 0.0, 0.3),
    (0.06, -0.605, 0.0, 0.023, 0.046, 0.02, 0.0, 0.3),
]
# A path through the whole head at value 1 integrates to this many attenuation units, which
# keeps the sinogram in the range where transmission weights are meaningful.
HEAD_INTEGRAL = 2.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('shape', nargs='?', metavar='VIEWS,ROWS,CHANNELS', help='sinogram shape to make')
    p.add_argument('path', nargs='?', help='output .npy file for the sinogram')
    p.add_argument('--angles', help='.npy file of view angles in radians, one per view')
    p.add_argument('--angle-span', type=float, default=180.0,
                   help='without --angles: equally spaced views over this many degrees (default 180)')
    p.add_argument('--band-rows', type=int, default=64,
                   help='detector rows projected per band (default 64)')
    p.add_argument('--devices', help="device list for the projection, e.g. 'cpu' (default: one GPU "
                                     'if present, else the default device)')
    p.add_argument('--check', metavar='RECON',
                   help='compare this reconstruction file with the phantom instead of making a sinogram')
    p.add_argument('--slices', type=int, default=5, help='with --check: slices to compare (default 5)')
    args = p.parse_args()
    if args.check is None and (args.shape is None or args.path is None):
        p.error('give VIEWS,ROWS,CHANNELS and an output path, or --check RECON')
    return args


def phantom_band(num_rows, num_cols, num_slices, s0, s1, device):
    """The phantom on slices [s0, s1) of a (num_rows, num_cols, num_slices) volume, as a
    float32 tensor of shape (num_rows, num_cols, s1 - s0) on ``device``."""
    radius = 0.5 * (min(num_rows, num_cols) - 1)
    y = (torch.arange(num_rows, device=device, dtype=torch.float32) - 0.5 * (num_rows - 1)) / radius
    x = (torch.arange(num_cols, device=device, dtype=torch.float32) - 0.5 * (num_cols - 1)) / radius
    z = (torch.arange(s0, s1, device=device, dtype=torch.float32) - 0.5 * (num_slices - 1)) \
        / max(0.5 * (num_slices - 1), 1.0)
    yy, xx, zz = y[:, None, None], x[None, :, None], z[None, None, :]
    band = torch.zeros((num_rows, num_cols, s1 - s0), device=device, dtype=torch.float32)
    # A path through the whole head at value 1 spans 2 * 0.92 * radius voxels.
    scale = HEAD_INTEGRAL / (2 * 0.92 * radius)
    for x0, y0, z0, a, b, c, angle_deg, value in ELLIPSOIDS:
        phi = math.radians(angle_deg)
        xr = (xx - x0) * math.cos(phi) + (yy - y0) * math.sin(phi)
        yr = -(xx - x0) * math.sin(phi) + (yy - y0) * math.cos(phi)
        inside = (xr / a) ** 2 + (yr / b) ** 2 + ((zz - z0) / c) ** 2 <= 1.0
        band += inside.to(torch.float32) * (value * scale)
    return band


def make_sinogram(num_views, num_rows, num_channels, angles, path, band_rows, devices):
    """Write the phantom's sinogram to ``path``, one band of detector rows at a time."""
    sino = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                     shape=(num_views, num_rows, num_channels))
    band_rows = min(band_rows, num_rows)
    model, largest, t0 = None, 0.0, time.time()
    for s0 in range(0, num_rows, band_rows):
        s1 = min(s0 + band_rows, num_rows)
        if model is None or model.get_params('sinogram_shape')[1] != s1 - s0:
            model = mbirtorch.ParallelBeamModel((num_views, s1 - s0, num_channels), angles)
            model.set_params(verbose=0, no_warning=True)
            if devices:
                model.configure_devices(devices=devices)
            else:
                # A band is small, so one device projects it.
                model.configure_devices(num_devices=1)
        recon_shape = model.get_params('recon_shape')
        band = phantom_band(recon_shape[0], recon_shape[1], num_rows, s0, s1, model.torch_device)
        projection = model.forward_project(band)
        sino[:, s0:s1, :] = projection
        largest = max(largest, float(projection.max()))
        del band, projection
        print(f'  rows {s0}-{s1 - 1} of {num_rows} written, {time.time() - t0:.0f} s', flush=True)
    sino.flush()
    print(f'wrote {path}: shape {sino.shape}, {sino.nbytes / GB:.1f} GB, largest value {largest:.3f}')


class ReconFile:
    """Reads single slices of a reconstruction file in (row, col) order.  A ``.npy`` file is the
    streaming harness's (slice, row, col) array; anything else is the ``export_recon_hdf5``
    layout, dataset ``recon`` with axes (slice, col, row)."""

    def __init__(self, path):
        self._file = None
        if path.endswith('.npy'):
            self.array = np.load(path, mmap_mode='r')
            self.num_slices, self.num_rows, self.num_cols = self.array.shape
            self._transpose = False
        else:
            import h5py
            self._file = h5py.File(path, 'r')
            names = list(self._file.keys())
            self.array = self._file['recon' if 'recon' in names else names[0]]
            self.num_slices, self.num_cols, self.num_rows = self.array.shape
            self._transpose = True

    def read_slice(self, z):
        block = np.asarray(self.array[z])
        return block.T if self._transpose else block

    def close(self):
        if self._file is not None:
            self._file.close()


def check(path, count):
    recon = ReconFile(path)
    print(f'{path}: {recon.num_slices} slices of {recon.num_rows} x {recon.num_cols}')
    for z in np.linspace(0, recon.num_slices - 1, count + 2)[1:-1].round().astype(int):
        z = int(z)
        truth = phantom_band(recon.num_rows, recon.num_cols, recon.num_slices, z, z + 1,
                             torch.device('cpu')).numpy()[:, :, 0]
        norm = np.linalg.norm(truth)
        nrmse = np.linalg.norm(recon.read_slice(z) - truth) / norm if norm > 0 else float('nan')
        print(f'  slice {z}: NRMSE {nrmse:.3f}')
    recon.close()


def main():
    args = parse_args()
    if args.check:
        check(args.check, args.slices)
        return
    num_views, num_rows, num_channels = (int(v) for v in args.shape.split(','))
    if args.angles:
        angles = np.load(args.angles).astype(np.float32).reshape(-1)
        if angles.shape[0] != num_views:
            raise SystemExit(f'{args.angles} holds {angles.shape[0]} angles for {num_views} views')
    else:
        angles = np.linspace(0, np.deg2rad(args.angle_span), num_views, endpoint=False,
                             dtype=np.float32)
    devices = args.devices.split(',') if args.devices else None
    make_sinogram(num_views, num_rows, num_channels, angles, args.path, args.band_rows, devices)


if __name__ == '__main__':
    main()
