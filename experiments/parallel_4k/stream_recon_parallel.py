"""Stream a large parallel-beam reconstruction through the GPUs, one band of slices at a time.

This is a proof of concept for a volume that fits neither on the GPUs nor in host memory
as a whole.  One band of detector rows is in play at a time.  The harness reads the band's
rows of the sinogram from disk, reconstructs them on the GPUs with
``ParallelBeamModel.recon``, blends the band's first slices with the previous band's last
slices at the seam, writes the band to the output file, and moves on.  The host holds one
band of the sinogram, its weights, and one reconstructed band.  The volume exists only on
disk.

The bands and the result are those of ``recon_split_sino``.  The band size comes from the
library's memory model for the visible GPUs.  Every band extends ``half_overlap`` rows
past its seams, so consecutive bands share ``2 * half_overlap`` slices, and the seam is
blended over those slices with ``stitch_arrays`` and its weights.  The regularization
parameters are set once from the whole sinogram and copied into every band's model.  With
the same seed and the same bands, the output equals ``recon_split_sino``'s to floating
point round-off.

    python stream_recon_parallel.py --sino sino.npy --angles angles.npy --weights transmission --output recon.h5
    python stream_recon_parallel.py --sino sino.npy --angles angles.npy --output recon.h5 --resume

For a test without data, ``make_synthetic_sinogram.py`` writes a phantom sinogram and checks a
reconstruction against the phantom.

Inputs.  The sinogram is a ``.npy`` file, which is memory mapped so that only a band's rows
are read, or an HDF5 file, from which a band's rows are read as a hyperslab.  Its axes are
(views, detector rows, detector channels) in float32.  The angles are radians in a ``.npy``
file, or ``--angle-span`` gives equally spaced views over that many degrees.  Weights are
``none``, ``transmission`` or ``transmission_root`` computed per band, or a ``.npy`` file
read per band.

Output.  A ``.h5`` path gives an HDF5 file in the layout of ``export_recon_hdf5``: dataset
``recon`` with axes (slice, col, row), readable with ``import_recon_hdf5``.  A ``.npy``
path gives a memory mapped array with axes (slice, row, col).  Both are written band by
band.  A progress file beside the output records each finished band, and ``--resume``
continues an interrupted run from the first unfinished band.

Private library calls.  The band size uses ``TomographyModel._fits_available_devices``,
the memory check ``recon_split_sino`` itself uses.  Everything else is public API.
"""

import argparse
import json
import math
import os
import resource
import sys
import time

# torch's caching allocator holds memory it is not using.  This setting lets it return
# that memory and was measured to halve the reserve at no time cost.  It must be in the
# environment before torch is imported.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mbirtorch  # noqa: E402

GB = 1e9
GiB = 1024 ** 3
FLOAT_BYTES = 4
# Bands are written to disk in slabs of this many slices.  A slab of a 4096 x 4096 volume
# is 4.3 GB.
WRITE_SLAB = 64




def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    src = p.add_argument_group('input')
    src.add_argument('--sino', help='sinogram file: .npy (memory mapped) or .h5/.hdf5')
    src.add_argument('--dataset', help='dataset name inside an HDF5 sinogram (default: the only one)')
    src.add_argument('--angles', help='.npy file of view angles in radians, one per view')
    src.add_argument('--angle-span', type=float, default=180.0,
                     help='without --angles: equally spaced views over this many degrees (default 180)')
    src.add_argument('--weights', default='none',
                     help="'none' (default), 'transmission', 'transmission_root', or a .npy file")
    out = p.add_argument_group('output')
    out.add_argument('--output', help='reconstruction file: .h5 (export_recon_hdf5 layout) or .npy')
    out.add_argument('--resume', action='store_true',
                     help='continue an interrupted run from its progress file')
    out.add_argument('--log', default='~/.mbirtorch/logs/stream_recon_parallel.log',
                     help='merged reconstruction log (default ~/.mbirtorch/logs/stream_recon_parallel.log)')
    out.add_argument('--quiet', action='store_true', help='do not print the per-iteration log lines')
    dev = p.add_argument_group('devices and memory')
    dev.add_argument('--num-gpus', type=int, help='pin the GPU count (default: all visible GPUs)')
    dev.add_argument('--devices', help="explicit device list for every band, e.g. 'cpu' or 'cuda:0,cuda:1'")
    dev.add_argument('--slices-per-part', type=int,
                     help='slices per band (default: the most the memory model says fit the GPUs)')
    dev.add_argument('--margin', type=float, default=0.3,
                     help='safety margin of the memory check when choosing the band: a band fits when '
                          '(1 + margin) x its modeled peak is below the free GPU memory (default 0.3; '
                          'the library uses 0.15)')
    dev.add_argument('--half-overlap', type=int, default=5,
                     help='rows kept past each seam on each side (default 5)')
    rec = p.add_argument_group('reconstruction')
    rec.add_argument('--max-iterations', type=int, default=15)
    rec.add_argument('--stop-pct', type=float, default=0.2,
                     help='stop when the relative change per iteration is below this percent '
                          '(default 0.2; 0 runs exactly max-iterations)')
    rec.add_argument('--sharpness', type=float, default=1.0)
    rec.add_argument('--snr-db', type=float, default=30.0)
    rec.add_argument('--positivity', action='store_true')
    rec.add_argument('--seed', type=int, help='numpy seed for the pixel partitions')
    rec.add_argument('--max-parts', type=int, help='stop after this many bands (for testing resume)')
    args = p.parse_args()
    if args.sino is None or args.output is None:
        p.error('--sino and --output are required')
    return args


# ── the sinogram on disk ─────────────────────────────────────────────────────
class SinogramSource:
    """Reads bands of detector rows from a sinogram file without loading the file."""

    def __init__(self, path, dataset=None):
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        self._file = None
        if ext == '.npy':
            self.array = np.load(path, mmap_mode='r')
        elif ext in ('.h5', '.hdf5'):
            import h5py
            self._file = h5py.File(path, 'r')
            names = list(self._file.keys())
            if dataset is None:
                if len(names) != 1:
                    raise ValueError(f'{path} holds datasets {names}; name one with --dataset')
                dataset = names[0]
            self.array = self._file[dataset]
        else:
            raise ValueError(f'unrecognized sinogram file type {ext!r}; use .npy or .h5')
        if self.array.ndim != 3:
            raise ValueError(f'the sinogram must be 3D (views, rows, channels); got {self.array.shape}')
        self.shape = tuple(int(s) for s in self.array.shape)

    def read_rows(self, lo, hi):
        """The rows [lo, hi) as a contiguous float32 array (views, hi - lo, channels)."""
        return np.ascontiguousarray(self.array[:, lo:hi, :], dtype=np.float32)

    def view_subsample(self, max_views=20):
        """About ``max_views`` equally spaced views, as a host array, for the
        regularization statistics.  The stride is the one subsample_views uses."""
        step = max(1, self.shape[0] // max_views)
        return np.ascontiguousarray(self.array[::step], dtype=np.float32)

    def close(self):
        if self._file is not None:
            self._file.close()


# ── the volume on disk ───────────────────────────────────────────────────────
class VolumeSink:
    """Writes and reads bands of slices of the output volume.

    An ``.h5`` path is the layout of export_recon_hdf5, dataset 'recon' with axes
    (slice, col, row), one chunk per slice so that an unfinished run has written only the
    chunks it finished.  A ``.npy`` path is a memory mapped array with axes
    (slice, row, col).  Bands arrive and leave in the library's (row, col, slice) order."""

    def __init__(self, path, recon_shape, resume):
        self.path = path
        self.num_rows, self.num_cols, self.num_slices = (int(s) for s in recon_shape)
        self.is_hdf5 = os.path.splitext(path)[1].lower() != '.npy'
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        if self.is_hdf5:
            import h5py
            if resume:
                self._file = h5py.File(path, 'r+')
                self.dset = self._file['recon']
                if self.dset.shape != (self.num_slices, self.num_cols, self.num_rows):
                    raise ValueError(f'{path} holds a volume of shape {self.dset.shape}, not '
                                     f'{(self.num_slices, self.num_cols, self.num_rows)}')
            else:
                self._file = h5py.File(path, 'w')
                self.dset = self._file.create_dataset(
                    'recon', shape=(self.num_slices, self.num_cols, self.num_rows),
                    dtype=np.float32, chunks=(1, self.num_cols, self.num_rows))
            # (row, col, slice) -> (slice, col, row) is its own inverse.
            self._to_file, self._from_file = (2, 1, 0), (2, 1, 0)
        else:
            shape = (self.num_slices, self.num_rows, self.num_cols)
            if resume:
                self.dset = np.load(path, mmap_mode='r+')
                if self.dset.shape != shape:
                    raise ValueError(f'{path} holds a volume of shape {self.dset.shape}, not {shape}')
            else:
                self.dset = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
            # (row, col, slice) -> (slice, row, col), and back.
            self._to_file, self._from_file = (2, 0, 1), (1, 2, 0)

    def write(self, band, z0):
        """Write ``band`` (rows, cols, n) to slices [z0, z0 + n), one slab at a time."""
        n = band.shape[2]
        for a in range(0, n, WRITE_SLAB):
            b = min(a + WRITE_SLAB, n)
            self.dset[z0 + a:z0 + b] = np.ascontiguousarray(
                np.transpose(band[:, :, a:b], self._to_file))

    def read(self, z0, z1):
        """The slices [z0, z1) as an array (rows, cols, z1 - z0)."""
        block = np.asarray(self.dset[z0:z1])
        return np.ascontiguousarray(np.transpose(block, self._from_file))

    def set_attributes(self, attributes):
        if self.is_hdf5:
            for key, value in attributes.items():
                self.dset.attrs[key] = value

    def close(self):
        if self.is_hdf5:
            self._file.close()
        else:
            self.dset.flush()


# ── the bands ────────────────────────────────────────────────────────────────
def band_ranges(num_rows, num_parts, half_overlap):
    """The kept slice range and the model row range of each band, as recon_split_sino
    tiles them: nearly equal kept ranges, each model extended half_overlap rows past its
    interior seams."""
    base, extra = divmod(num_rows, num_parts)
    ranges, start = [], 0
    for index in range(num_parts):
        stop = start + base + (1 if index < extra else 0)
        model_lo, model_hi = max(start - half_overlap, 0), min(stop + half_overlap, num_rows)
        ranges.append(((start, stop), (model_lo, model_hi)))
        start = stop
    return ranges


def worst_part_rows(num_rows, num_parts, half_overlap):
    biggest_kept = -(-num_rows // num_parts)
    if num_parts == 1:
        return biggest_kept
    if num_parts == 2:
        return biggest_kept + half_overlap
    return biggest_kept + 2 * half_overlap


def band_model(full_model, num_band_rows):
    """A copy of the full model covering ``num_band_rows`` detector rows, and so that many
    slices, with the full model's regularization parameters fixed."""
    recon_rows, recon_cols = full_model.get_params('recon_shape')[:2]
    model = mbirtorch.copy_ct_model(full_model, new_num_det_rows=num_band_rows, no_warning=True)
    model.set_params(no_warning=True, auto_regularize_flag=False,
                     recon_shape=(recon_rows, recon_cols, num_band_rows))
    return model


def choose_num_parts(full_model, num_rows, half_overlap, margin, weights_supplied):
    """The fewest bands whose largest band's modeled peak fits the GPUs with ``margin``.
    This is the choice recon_split_sino makes, with the margin as a parameter."""
    if not torch.cuda.is_available():
        raise SystemExit('without CUDA the memory model cannot read a GPU budget; '
                         'give --slices-per-part')
    max_parts = max(1, num_rows // (2 * half_overlap))
    for num_parts in range(1, max_parts + 1):
        model = band_model(full_model, worst_part_rows(num_rows, num_parts, half_overlap))
        model.memory_preflight_margin = margin
        call_arrays = {'weights': 'supplied'} if weights_supplied else {}
        if model._fits_available_devices(**call_arrays):
            return num_parts
    raise SystemExit(f'no band of at least {2 * half_overlap} slices fits the GPUs at margin {margin}')


def blend_ramp(half_overlap):
    """The ramp recon_split_sino hands to stitch_arrays: at most 4 blended slices, even."""
    ramp = min(4, half_overlap)
    return ramp - ramp % 2


def peak_host_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == 'darwin' else rss * 1024


# ── the run ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.num_gpus is not None:
        os.environ['MBIRTORCH_NUM_DEVICES'] = str(args.num_gpus)
    devices = args.devices.split(',') if args.devices else None

    def angles_of(num_views):
        if args.angles:
            angles = np.load(args.angles).astype(np.float32).reshape(-1)
            if angles.shape[0] != num_views:
                raise ValueError(f'{args.angles} holds {angles.shape[0]} angles for {num_views} views')
            return angles
        return np.linspace(0, np.deg2rad(args.angle_span), num_views, endpoint=False,
                           dtype=np.float32)

    t_start = time.time()
    source = SinogramSource(args.sino, args.dataset)
    num_views, num_rows, num_channels = source.shape
    angles = angles_of(num_views)
    weights_source = None
    if args.weights.endswith('.npy'):
        weights_source = SinogramSource(args.weights)
        if weights_source.shape != source.shape:
            raise ValueError(f'weights shape {weights_source.shape} differs from {source.shape}')
    weights_supplied = args.weights != 'none'

    # The full model carries the geometry and the regularization.  No data of its size is
    # ever placed on a device: the regularization statistics run on a view subsample, as
    # they do in recon and in recon_split_sino.
    full = mbirtorch.ParallelBeamModel(source.shape, angles)
    full.set_params(sharpness=args.sharpness, snr_db=args.snr_db,
                    positivity_flag=args.positivity, verbose=1, no_warning=True)
    recon_shape = tuple(int(s) for s in full.get_params('recon_shape'))
    regularization = full.auto_set_regularization_params(source.view_subsample())
    print(f'sinogram {source.shape} ({num_views * num_rows * num_channels * FLOAT_BYTES / GB:.1f} GB), '
          f'volume {recon_shape} ({math.prod(recon_shape) * FLOAT_BYTES / GB:.1f} GB)')
    print('regularization from the whole sinogram: '
          + ', '.join(f'{k}={v:.4g}' for k, v in regularization.items()), flush=True)

    ho = args.half_overlap
    if args.slices_per_part is not None:
        num_parts = max(1, -(-num_rows // args.slices_per_part))
    else:
        num_parts = choose_num_parts(full, num_rows, ho, args.margin, weights_supplied)
    if num_parts > 1 and num_rows // num_parts < 2 * ho:
        raise SystemExit(f'{num_parts} bands leave fewer than {2 * ho} slices each; use fewer bands '
                         'or a smaller --half-overlap')
    ranges = band_ranges(num_rows, num_parts, ho)
    print(f'{num_parts} band(s) of about {-(-num_rows // num_parts)} slices, overlap {ho} rows per seam',
          flush=True)

    # The progress file makes the run resumable and records each band's cost.
    progress_path = os.path.splitext(args.output)[0] + '_progress.json'
    config = {'sinogram_shape': list(source.shape), 'recon_shape': list(recon_shape),
              'num_parts': num_parts, 'half_overlap': ho, 'weights': args.weights,
              'max_iterations': args.max_iterations, 'stop_pct': args.stop_pct,
              'sharpness': args.sharpness, 'snr_db': args.snr_db,
              'regularization': regularization}
    done = []
    if args.resume:
        if not os.path.exists(progress_path):
            raise SystemExit(f'--resume needs {progress_path}, which does not exist')
        with open(progress_path) as f:
            record = json.load(f)
        if record['config'] != config:
            raise SystemExit(f'{progress_path} was written by a run with different settings')
        done = record['parts']
        if [p['index'] for p in done] != list(range(len(done))):
            raise SystemExit(f'{progress_path} has bands out of order; start over')
        print(f'resuming after {len(done)} finished band(s)', flush=True)
    sink = VolumeSink(args.output, recon_shape, resume=args.resume)

    def save_progress():
        with open(progress_path, 'w') as f:
            json.dump({'config': config, 'parts': done}, f, indent=1)

    if args.seed is not None:
        np.random.seed(args.seed)
    log_path = os.path.expanduser(args.log)
    part_logs = [f'{log_path}.part{k}' for k in range(num_parts)]
    ramp = blend_ramp(ho)
    for k, ((lo, hi), (mlo, mhi)) in enumerate(ranges):
        if k < len(done):
            continue
        if args.max_parts is not None and k >= args.max_parts:
            print(f'stopping after {k} band(s) as asked; resume with --resume', flush=True)
            break
        t_band = time.time()
        sino_band = source.read_rows(mlo, mhi)
        if weights_source is not None:
            weights = weights_source.read_rows(mlo, mhi)
        elif weights_supplied:
            weights = mbirtorch.gen_weights(sino_band, weight_type=args.weights)
        else:
            weights = None
        t_read = time.time() - t_band

        model = band_model(full, mhi - mlo)
        if devices:
            model.configure_devices(devices=devices)
        print(f'\n=== band {k + 1} of {num_parts}: slices {lo}-{hi - 1}, model rows {mlo}-{mhi - 1} ===',
              flush=True)
        band, band_dict = model.recon(sino_band, weights=weights, max_iterations=args.max_iterations,
                                      stop_threshold_change_pct=args.stop_pct,
                                      logfile_path=part_logs[k], print_logs=not args.quiet)
        del sino_band, weights, model
        t_recon = time.time() - t_band - t_read

        # The band's first 2 * ho slices are the seam it shares with the previous band, whose
        # last 2 * ho slices are already on disk.  The blend is stitch_arrays over those two
        # slabs, so it is the blend recon_split_sino computes.  The rest of the band is
        # written as it is; its own last 2 * ho slices wait for the next band's blend.
        if k > 0:
            seam = 2 * ho
            previous_tail = sink.read(mlo, mlo + seam)
            blended = mbirtorch.stitch_arrays([previous_tail, band[:, :, :seam]],
                                              overlap=seam, axis=2, ramp_overlap=ramp)
            sink.write(blended, mlo)
            sink.write(band[:, :, seam:], mlo + seam)
        else:
            sink.write(band, mlo)
        del band
        t_write = time.time() - t_band - t_read - t_recon

        entry = {'index': k, 'kept': [lo, hi], 'model_rows': [mlo, mhi],
                 'iterations': int(band_dict['recon_params']['num_iterations']),
                 'read_s': round(t_read, 1), 'recon_s': round(t_recon, 1), 'write_s': round(t_write, 1),
                 'peak_host_rss_GB': round(peak_host_bytes() / GB, 1)}
        done.append(entry)
        save_progress()
        print(f'band {k + 1} done: read {t_read:.0f} s, recon {t_recon:.0f} s '
              f'({entry["iterations"]} iterations), write {t_write:.0f} s; '
              f'peak host RSS {entry["peak_host_rss_GB"]:.0f} GB', flush=True)

    finished = len(done) == num_parts
    if finished:
        sink.set_attributes({'sinogram_shape': str(source.shape), 'num_parts': num_parts,
                             'half_overlap': ho, 'weights': args.weights, 'sharpness': args.sharpness,
                             'snr_db': args.snr_db, 'mbirtorch_version': mbirtorch.__version__,
                             'iterations_per_part': str([p['iterations'] for p in done])})
    sink.close()
    source.close()
    if weights_source is not None:
        weights_source.close()
    if finished:
        labels = [f'stream_recon_parallel: band {k + 1} of {num_parts} (slices {lo}-{hi - 1})'
                  for k, ((lo, hi), _m) in enumerate(ranges)]
        mbirtorch.merge_log_files(log_path, zip(labels, part_logs))

    total = time.time() - t_start
    gpu_lines = []
    for entry in mbirtorch.get_memory_stats(print_results=False):
        if str(entry['id']).startswith('GPU'):
            gpu_lines.append(f'  {entry["id"]}: peak allocated {entry["peak_bytes_in_use"] / GiB:.1f} GiB, '
                             f'peak reserved {entry["peak_reserved_bytes"] / GiB:.1f} GiB '
                             f'of {entry["bytes_limit"] / GiB:.1f} GiB')
    print(f'\n{"finished" if finished else "stopped"}: {len(done)} of {num_parts} band(s), '
          f'{total:.0f} s this run, peak host RSS {peak_host_bytes() / GB:.0f} GB; output {args.output}')
    for line in gpu_lines:
        print(line)


if __name__ == '__main__':
    main()
