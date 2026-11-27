"""Code that the two reconstruction scripts of this folder share.

``stream_recon_parallel.py`` and ``stream_fdk_cone.py`` reconstruct a scan in parts, and they
write each part to the output file before they continue.  A part is a band of the parallel-beam
script or a slab of the cone-beam script.  This module holds what does not depend on the
geometry or on the reconstruction method: the options that both scripts have, the reader of the
sinogram file, the writer of the output file, the progress file, the search for the number of
parts, and the loop over the parts.
"""

import json
import math
import os
import resource
import sys
import time

# torch's caching allocator holds memory it is not using.  This setting lets it return
# that memory and was measured to halve the reserve at no time cost.  That measurement is of the
# parallel-beam reconstruction, and the effect on FDK has not been measured.  The setting must
# be in the environment before torch is imported, so a script imports this module before torch.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mbirtorch.utilities import _recon_attributes  # noqa: E402

GB = 1e9
GiB = 1024 ** 3
FLOAT_BYTES = 4
VOLUME_NDIM = 3
# Bands are written to disk in slabs of this many slices.  A slab of a 4096 x 4096 volume
# is 4.3 GB.
WRITE_SLAB = 64
# The package uses a smaller margin, 0.15.  In a test of FDK by slabs on H100 GPUs, the margin of
# the package accepted a slab that ran out of memory.  This margin rejects the two slabs that
# ran out of memory in that test, and it accepts the three slabs that finished.  The original
# parallel-beam script has the same default.
DEFAULT_MARGIN = 0.3


# ── the options that both scripts have ───────────────────────────────────────
def add_input_arguments(parser):
    """Add the option group 'input' with the options of the sinogram file to parser.  Return the
    group, so that a script can add its own options to it."""
    group = parser.add_argument_group('input')
    group.add_argument('--sino', required=True, help='sinogram file: .npy (memory mapped) or .h5/.hdf5')
    group.add_argument('--dataset', help='dataset name inside an HDF5 sinogram (default: the only one)')
    return group


def add_angle_arguments(group, default_span):
    """Add the two options that give the view angles to group.  default_span is the default of
    --angle-span in degrees."""
    group.add_argument('--angles', help='.npy file of view angles in radians, one per view')
    group.add_argument('--angle-span', type=float, default=default_span,
                       help='without --angles: equally spaced views over this many degrees (default %(default)s)')


def add_output_arguments(parser, part_name):
    """Add the option group 'output' with the options of the output file to parser.  part_name
    is the name of a part in the help texts, 'band' or 'slab'.  Return the group."""
    group = parser.add_argument_group('output')
    group.add_argument('--output', required=True, help='reconstruction file: .h5 (export_recon_hdf5 layout) or .npy')
    group.add_argument('--resume', action='store_true',
                       help='continue an interrupted run from its progress file')
    group.add_argument('--max-parts', type=int, help=f'stop after this many {part_name}s (for testing resume)')
    return group


def add_device_arguments(parser, part_name):
    """Add the option group 'devices and memory' to parser.  part_name is the name of a part in
    the help texts, 'band' or 'slab'.  Return the group."""
    group = parser.add_argument_group('devices and memory')
    group.add_argument('--num-gpus', type=int, help='pin the GPU count (default: the package chooses)')
    group.add_argument('--devices', help=f"explicit device list for every {part_name}, e.g. 'cpu' or 'cuda:0,cuda:1'")
    group.add_argument('--margin', type=float, default=DEFAULT_MARGIN,
                       help=f'safety margin of the memory check that chooses the number of {part_name}s: a '
                            f'{part_name} fits when (1 + margin) x its estimated peak is at most the free GPU '
                            'memory (default %(default)s; the package uses 0.15)')
    return group


def device_list(args):
    """Apply --num-gpus.  Return the devices of --devices as a list, or None if the option is
    not given."""
    if args.num_gpus is not None:
        os.environ['MBIRTORCH_NUM_DEVICES'] = str(args.num_gpus)
    return args.devices.split(',') if args.devices else None


def view_angles(args, num_views):
    """Return the angles of the views in radians, from the file of --angles or from --angle-span."""
    if args.angles:
        angles = np.load(args.angles).astype(np.float32).reshape(-1)
        if angles.shape[0] != num_views:
            raise SystemExit(f'{args.angles} holds {angles.shape[0]} angles for {num_views} views')
        return angles
    return np.linspace(0, np.deg2rad(args.angle_span), num_views, endpoint=False, dtype=np.float32)


def print_sizes(sinogram_shape, recon_shape):
    """Print the shapes of the sinogram and of the volume, and their sizes in GB."""
    print(f'sinogram {tuple(sinogram_shape)} ({math.prod(sinogram_shape) * FLOAT_BYTES / GB:.1f} GB), '
          f'volume {tuple(recon_shape)} ({math.prod(recon_shape) * FLOAT_BYTES / GB:.1f} GB)', flush=True)


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
                # The new file is flushed at once, so that it can be opened again if the process
                # is killed during the first part.
                self._file.flush()
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

    def flush(self):
        """Write the buffered data to disk.  If the process is killed before the next write, the
        file can be opened again, and it holds all data written so far."""
        (self._file if self.is_hdf5 else self.dset).flush()

    def close(self):
        if self.is_hdf5:
            self._file.close()
        else:
            self.dset.flush()


def write_recon_attributes(sink, recon_dict):
    """Write the attributes of an export_recon_hdf5 file to an HDF5 output.  An attribute that
    the HDF5 library refuses, for example because it is too large, is reported and left out."""
    for key, value in _recon_attributes(recon_dict, VOLUME_NDIM).items():
        try:
            sink.set_attributes({key: value})
        except (OSError, RuntimeError) as error:
            print(f'the attribute {key!r} was not written: {error}')


# ── the progress file ────────────────────────────────────────────────────────
def save_json(path, record):
    """Write record to path.  The folder of path is created if it does not exist.  The file at
    path is replaced in one step, so a process that is killed during the write leaves either the
    old file or the new file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(record, f, indent=1)
    os.replace(path + '.tmp', path)


class ProgressFile:
    """The record of a run, in a JSON file beside the output file.

    The file holds the settings of the run and one entry for each finished part.  With resume,
    the object reads the file of the interrupted run, and the run continues after the parts
    that this file records.  part_name is the name of a part in the messages, 'band' or
    'slab'."""

    def __init__(self, output_path, part_name, resume):
        self.path = os.path.splitext(output_path)[0] + '_progress.json'
        self.part_name, self.resume = part_name, resume
        self.config, self.done = None, []
        self._interrupted = None
        if resume:
            if not os.path.exists(self.path):
                raise SystemExit(f'--resume needs {self.path}, which does not exist')
            with open(self.path) as f:
                self._interrupted = json.load(f)

    def num_parts(self, given, choose):
        """Return the number of parts of the run.  A resumed run keeps the number of the
        interrupted run.  A new run takes given, or the result of choose() if given is None."""
        if self.resume:
            num_parts = self._interrupted['config']['num_parts']
            if given is not None and given != num_parts:
                raise SystemExit(f'{self.path} was written by a run with {num_parts} {self.part_name}(s)')
            return num_parts
        return given if given is not None else choose()

    def start(self, config):
        """Begin the run with the settings config, which must hold 'num_parts'.  A new run
        replaces the file of an earlier run.  A resumed run must have the settings of the
        interrupted run, and it takes over the finished parts of that run."""
        # The settings are compared in the form that the file holds.
        self.config = json.loads(json.dumps(config))
        if not self.resume:
            self._save()
            return
        if self._interrupted['config'] != self.config:
            raise SystemExit(f'{self.path} was written by a run with different settings')
        self.done = self._interrupted['parts']
        if [entry['index'] for entry in self.done] != list(range(len(self.done))):
            raise SystemExit(f'{self.path} has {self.part_name}s out of order; start over')
        print(f'resuming after {len(self.done)} finished {self.part_name}(s)', flush=True)

    def record(self, entry):
        """Add the entry of a finished part to the file.  Call this function only after the
        output file is flushed."""
        self.done.append(entry)
        self._save()

    @property
    def finished(self):
        """True if the file records every part of the run."""
        return len(self.done) == self.config['num_parts']

    def _save(self):
        save_json(self.path, {'config': self.config, 'parts': self.done})


def open_output(path, recon_shape, progress, config):
    """Begin the run of progress with the settings config, then open the output file at path.
    Return the VolumeSink of the output file.  A new run replaces the progress file of an earlier
    run before it creates the output file.  --resume then cannot skip parts that the new output
    file does not hold."""
    progress.start(config)
    return VolumeSink(path, recon_shape, resume=progress.resume)


# ── the number of parts ──────────────────────────────────────────────────────
def smallest_fitting_count(fits, max_count):
    """Return the smallest number of parts, from 1 to max_count, for which fits(count) is true.
    Return None if no number fits.  fits(count) must be true for every number above the first
    number that fits, as it is when more parts are smaller parts.  The function tests max_count
    first, so that it ends at once if no number fits."""
    if not fits(max_count):
        return None
    for count in range(1, max_count):
        if fits(count):
            return count
    return max_count


# ── the run ──────────────────────────────────────────────────────────────────
def peak_host_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == 'darwin' else rss * 1024


def gpu_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def start_gpu_peaks():
    """Release the unused GPU memory and restart the peak counters.  The peaks of a part then do
    not include memory that an earlier part reserved."""
    for index in range(gpu_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)


def gpu_peaks():
    """Return two lists with one value in GiB for each GPU: the peak memory that torch allocated
    since start_gpu_peaks, and the peak memory that torch reserved."""
    return ([round(torch.cuda.max_memory_allocated(index) / GiB, 2) for index in range(gpu_count())],
            [round(torch.cuda.max_memory_reserved(index) / GiB, 2) for index in range(gpu_count())])


def run_parts(plan, read, reconstruct, sink, progress, max_parts=None, write=None):
    """Reconstruct the parts of plan that progress does not record, in order.

    plan holds one (slice_lo, slice_hi, row_lo, row_hi) tuple for each part.  The slices are
    those that the part gives to the volume, and the rows are the detector rows that its
    reconstruction reads.  For part k, the function calls data = read(k), then
    part, values = reconstruct(k, data), then write(k, part).  part is an array (rows, cols,
    slices), and values is a dict that is recorded with the part.  Without write, the function
    writes part to the slices of the plan.

    After the write, the function flushes the output file, and only then it records the part in
    the progress file.  A process that is killed at any time then leaves a progress file that
    records only parts that the output file holds.

    The function stops before the part of index max_parts if max_parts is not None."""
    name, num_parts = progress.part_name, len(plan)
    for k, (slice_lo, slice_hi, row_lo, row_hi) in enumerate(plan):
        if k < len(progress.done):
            continue
        if max_parts is not None and k >= max_parts:
            print(f'stopping after {k} {name}(s) as asked; resume with --resume', flush=True)
            break
        rows = f'detector rows {row_lo}-{row_hi - 1}' if row_hi > row_lo else 'no detector rows'
        print(f'\n=== {name} {k + 1} of {num_parts}: slices {slice_lo}-{slice_hi - 1}, {rows} ===', flush=True)
        t_part = time.time()
        data = read(k)
        t_read = time.time() - t_part
        start_gpu_peaks()
        part, values = reconstruct(k, data)
        gpu_peak, gpu_reserved_peak = gpu_peaks()
        del data
        t_recon = time.time() - t_part - t_read
        if write is None:
            sink.write(part, slice_lo)
        else:
            write(k, part)
        del part
        sink.flush()
        t_write = time.time() - t_part - t_read - t_recon

        entry = {'index': k, 'slices': [slice_lo, slice_hi], 'rows': [row_lo, row_hi], **values,
                 'gpu_peak_GiB': gpu_peak, 'gpu_reserved_peak_GiB': gpu_reserved_peak,
                 'read_s': round(t_read, 1), 'recon_s': round(t_recon, 1), 'write_s': round(t_write, 1),
                 'peak_host_rss_GB': round(peak_host_bytes() / GB, 1)}
        progress.record(entry)
        recorded = ''.join(f', {key} {value}' for key, value in values.items())
        print(f'{name} {k + 1} done: read {t_read:.0f} s, recon {t_recon:.0f} s, write {t_write:.0f} s{recorded}; '
              f'peak host RSS {entry["peak_host_rss_GB"]:.0f} GB', flush=True)


def finish_output(sink, progress, model, recon_params):
    """Close the output file.  If every part is finished, first write the attributes of an
    export_recon_hdf5 file to an HDF5 output.  The attributes hold the parameters of model, which
    is the model of the whole scan, and the dict recon_params."""
    if progress.finished and sink.is_hdf5:
        write_recon_attributes(sink, model.get_recon_dict(recon_params=recon_params, save_log=False))
    sink.close()


def print_summary(progress, seconds, output_path):
    """Print whether the run is finished, the seconds of this run, and for each GPU the largest
    peaks of a part."""
    done, name = progress.done, progress.part_name
    print(f'\n{"finished" if progress.finished else "stopped"}: {len(done)} of {progress.config["num_parts"]} '
          f'{name}(s), {seconds:.0f} s this run, peak host RSS {peak_host_bytes() / GB:.0f} GB; output {output_path}')
    for index in range(gpu_count()):
        # A part that an earlier process recorded can have values for fewer GPUs, or no values.
        allocated = max((entry['gpu_peak_GiB'][index] for entry in done
                         if index < len(entry.get('gpu_peak_GiB', []))), default=0.0)
        reserved = max((entry['gpu_reserved_peak_GiB'][index] for entry in done
                        if index < len(entry.get('gpu_reserved_peak_GiB', []))), default=0.0)
        total_memory = torch.cuda.get_device_properties(index).total_memory
        print(f'  GPU {index}: largest peak of a {name} {allocated:.1f} GiB allocated, {reserved:.1f} GiB reserved '
              f'of {total_memory / GiB:.1f} GiB')
