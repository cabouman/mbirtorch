"""Check on small scans that stream_fdk_cone.py gives the FDK reconstruction of the whole volume.

    python check_stream_fdk_cone.py
    python check_stream_fdk_cone.py --devices cuda:0 --compile

For each case, the script writes a sinogram of white noise to a temporary folder, so that every
detector row and channel has its own content.  It runs ``stream_fdk_cone.py`` for several
numbers of slabs, and it compares each result with ``ConeBeamModel.recon_fdk`` of the whole
scan.  The script also runs eight other checks:

  band      The band of each slab must hold every detector row that the slab touches.  A
            forward projection of the slab finds these rows.
  control   Each slab is reconstructed from the rows that it touches, without one row at each
            end.  The difference from the whole reconstruction must then exceed the tolerance.
            This shows that the comparison detects a band that is too small.
  formats   The ``.npy`` output and the ``.h5`` output must hold the same values, and
            ``import_recon_hdf5`` must read the ``.h5`` output without a warning.
  resume    A run that stops after two slabs and is continued with ``--resume`` must give the
            same file as a run without a stop.
  kill      A run whose process ends without cleanup while it writes the third slab, and that is
            continued with ``--resume``, must also give the same file.
  restart   A new run has the output name of a finished run, and its process ends without cleanup
            while it writes the first slab.  Continued with ``--resume``, it must give the same
            file.
  HDF5 in   A sinogram in an HDF5 file must give the same result as the ``.npy`` sinogram.
  phantom   The sinogram that ``make_synthetic_cone_sinogram.py`` makes one slab at a time must
            equal the forward projection of the whole phantom.

The script prints one line for each check, and it ends with an error if a check fails.  By
default it runs on the CPU without torch.compile, which takes less than one minute.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import warnings

import h5py
import numpy as np

import mbirtorch

import make_synthetic_cone_sinogram
import stream_fdk_cone
from make_synthetic_sinogram import phantom_band

SINOGRAM_SHAPE = (72, 48, 48)     # views, detector rows, detector channels
MAGNIFICATION = 2.0
SLAB_COUNTS = (1, 2, 3, 7, 48)
CONTROL_SLABS = 4                 # slabs of the control
RESUME_SLABS = 5                  # slabs of the format, resume, kill, and HDF5 checks
RESUME_STOP = 2                   # slabs that the first run of the resume and kill checks finishes
PHANTOM_SLAB_SLICES = 11          # slices that the sinogram generator projects at a time
TOLERANCE = 1e-4                  # largest accepted difference, relative to the largest value
SEED = 5
# This program runs stream_fdk_cone.py.  After {stop} finished slabs, its process ends without
# cleanup after the write of the next slab, before the script records that slab as finished.
KILL_PROGRAM = """
import os, sys
import stream_common
import stream_fdk_cone
writes = []
write = stream_common.VolumeSink.write
def write_then_end(self, band, z0):
    write(self, band, z0)
    writes.append(z0)
    if len(writes) > {stop}:
        os._exit(1)
stream_common.VolumeSink.write = write_then_end
stream_fdk_cone.main(sys.argv[1:])
"""


def distances(fan_ratio):
    """Return the options for the two distances of a scan with the given fan ratio.  The fan ratio
    is the radius of the default volume divided by the distance from the source to the axis."""
    source_iso_dist = 0.5 * SINOGRAM_SHAPE[2] / MAGNIFICATION / fan_ratio      # ALU, with a detector pitch of 1
    return ['--source-detector-dist', repr(MAGNIFICATION * source_iso_dist),
            '--source-iso-dist', repr(source_iso_dist)]


# Each case has a name, the geometry options of stream_fdk_cone.py, and the model parameters that
# these options must produce.
CASES = [
    ('default volume, fan ratio 0.10', distances(0.10), {}),
    ('default volume, fan ratio 0.28', distances(0.28), {}),
    ('default volume, fan ratio 0.45', distances(0.45), {}),
    ('tall volume (axial_pad_fraction 1)', distances(0.28) + ['--param', 'axial_pad_fraction=1.0'],
     {'axial_pad_fraction': 1.0}),
    ('detector offsets, pitches, and square support',
     distances(0.20) + ['--delta-det-channel', '0.8', '--delta-det-row', '1.1', '--det-row-offset', '2.3',
                        '--det-channel-offset', '-1.7', '--param', 'use_ror_mask=false'],
     {'delta_det_channel': 0.8, 'delta_det_row': 1.1, 'det_row_offset': 2.3, 'det_channel_offset': -1.7,
      'use_ror_mask': False}),
    ('thick slices (aspect 1.5)', distances(0.28) + ['--param', 'voxel_slice_aspect=1.5'],
     {'voxel_slice_aspect': 1.5}),
    ('thin slices (aspect 0.5)', distances(0.28) + ['--param', 'voxel_slice_aspect=0.5'],
     {'voxel_slice_aspect': 0.5}),
    ('curved detector', distances(0.28) + ['--curved-detector'], {'use_curved_detector': True}),
    ('volume of a given shape, pitch, and offset',
     distances(0.28) + ['--recon-shape', '40,44,50', '--delta-voxel', '0.55', '--recon-slice-offset', '1.9'],
     {'recon_shape': (40, 44, 50), 'delta_voxel': 0.55, 'recon_slice_offset': 1.9}),
    ('views over 200 degrees', distances(0.28) + ['--angle-span', '200'], {}),
]


def run_stream(options):
    """Run stream_fdk_cone.py with the given options in this process, without its printout."""
    with contextlib.redirect_stdout(io.StringIO()):
        stream_fdk_cone.main(options)


def read_volume(path):
    """Return the volume in the file at path with axes (row, col, slice).  The function raises
    an error if import_recon_hdf5 gives a warning for an HDF5 file."""
    if path.endswith('.npy'):
        return np.transpose(np.load(path), (1, 2, 0))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        recon, _recon_dict = mbirtorch.import_recon_hdf5(path)
    return recon


def largest_difference(x, x_whole):
    """Return the largest difference between x and x_whole, relative to the largest value of
    x_whole."""
    return float(np.abs(x.astype(np.float64) - x_whole).max() / np.abs(x_whole).max())


def whole_reconstruction(options, sino):
    """Return the model that stream_fdk_cone.py builds from options, and the FDK reconstruction
    of the whole volume with this model."""
    args = stream_fdk_cone.parse_args(options + ['--sino', 'unused', '--output', 'unused'])
    model = stream_fdk_cone.scan_model(args, sino.shape)
    if args.devices:
        model.configure_devices(devices=args.devices.split(','))
    return model, np.asarray(model.recon_fdk(sino)).astype(np.float64)


def rows_touched(model, slice_lo, slice_hi):
    """Return the range of detector rows that the voxels of slices [slice_lo, slice_hi) touch, found
    by a forward projection of these voxels.  The range is (0, 0) if they touch no row."""
    volume = np.zeros(model.get_params('recon_shape'), dtype=np.float32)
    volume[:, :, slice_lo:slice_hi] = 1.0
    rows = np.flatnonzero((np.asarray(model.forward_project(volume)) > 0).any(axis=(0, 2)))
    return (int(rows.min()), int(rows.max()) + 1) if rows.size else (0, 0)


def band_and_control(model, sino, x_whole):
    """Return two values for the slabs of the control.  The first value is True if every band holds
    the rows that its slab touches.  The second value is the smallest difference from the whole
    reconstruction when a slab is reconstructed without the first and the last row that it
    touches."""
    bands_hold, smallest = True, np.inf
    for slice_lo, slice_hi, row_lo, row_hi in stream_fdk_cone.slab_plan(model, CONTROL_SLABS):
        touched_lo, touched_hi = rows_touched(model, slice_lo, slice_hi)
        if touched_hi == touched_lo:
            continue
        bands_hold = bands_hold and row_lo <= touched_lo and touched_hi <= row_hi
        if touched_hi - touched_lo <= 2:
            continue
        part = stream_fdk_cone.part_model(model, touched_lo + 1, touched_hi - 1, slice_lo, slice_hi)
        slab = np.asarray(part.recon_fdk(np.ascontiguousarray(sino[:, touched_lo + 1:touched_hi - 1, :])))
        difference = np.abs(slab - x_whole[:, :, slice_lo:slice_hi]).max() / np.abs(x_whole).max()
        smallest = min(smallest, float(difference))
    return bands_hold, smallest


def run_killed(options, finished_slabs):
    """Run stream_fdk_cone.py with the given options in a process that ends without cleanup while
    it writes the slab after finished_slabs slabs.  Return the number of slabs that the progress
    file then records."""
    output = options[options.index('--output') + 1]
    subprocess.run([sys.executable, '-c', KILL_PROGRAM.format(stop=finished_slabs)] + options,
                   cwd=os.path.dirname(os.path.abspath(__file__)), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    with open(os.path.splitext(output)[0] + '_progress.json') as f:
        return len(json.load(f)['parts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--devices', default='cpu', help="device list for every run (default %(default)s)")
    parser.add_argument('--compile', action='store_true', help='build the models with torch.compile')
    args = parser.parse_args()
    common = ['--devices', args.devices] + ([] if args.compile else ['--compile-off'])
    warnings.simplefilter('ignore')
    rng = np.random.default_rng(SEED)
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)
            print('    FAILED:', message)

    with tempfile.TemporaryDirectory() as folder:
        sino_path = os.path.join(folder, 'sino.npy')
        for name, geometry, expected in CASES:
            sino = rng.standard_normal(SINOGRAM_SHAPE).astype(np.float32)
            np.save(sino_path, sino)
            options = geometry + common
            model, x_whole = whole_reconstruction(options, sino)
            for key, value in expected.items():
                check(model.get_params(key) == value, f'{name}: {key} is {model.get_params(key)!r}, not {value!r}')
            num_slices = int(model.get_params('recon_shape')[2])
            worst = 0.0
            for num_slabs in SLAB_COUNTS:
                out_path = os.path.join(folder, f'recon_{num_slabs}.h5')
                run_stream(options + ['--sino', sino_path, '--output', out_path,
                                      '--num-slabs', str(min(num_slabs, num_slices))])
                worst = max(worst, largest_difference(read_volume(out_path), x_whole))
            bands_hold, control = band_and_control(model, sino, x_whole)
            check(worst <= TOLERANCE, f'{name}: the largest difference is {worst:.2e}')
            check(bands_hold, f'{name}: a band does not hold the rows that its slab touches')
            check(control > TOLERANCE, f'{name}: the control differs by only {control:.2e}')
            print(f'{name}: volume {tuple(model.get_params("recon_shape"))}, largest difference {worst:.1e} '
                  f'for {len(SLAB_COUNTS)} numbers of slabs, control {control:.1e}')

        # The remaining checks use the sinogram and the options of the last case.
        stream_options = options + ['--sino', sino_path]
        h5_path, npy_path = os.path.join(folder, 'formats.h5'), os.path.join(folder, 'formats.npy')
        run_stream(stream_options + ['--output', h5_path, '--num-slabs', str(RESUME_SLABS)])
        run_stream(stream_options + ['--output', npy_path, '--num-slabs', str(RESUME_SLABS)])
        same = np.array_equal(read_volume(h5_path), read_volume(npy_path))
        check(same, 'the .npy output differs from the .h5 output')
        print(f'formats: the .npy output {"equals" if same else "DIFFERS FROM"} the .h5 output, and '
              'import_recon_hdf5 reads the .h5 output without a warning')

        for extension in ('.h5', '.npy'):
            resumed_path = os.path.join(folder, 'resumed' + extension)
            first = stream_options + ['--output', resumed_path, '--num-slabs', str(RESUME_SLABS)]
            run_stream(first + ['--max-parts', str(RESUME_STOP)])
            run_stream(stream_options + ['--output', resumed_path, '--resume'])
            reference = h5_path if extension == '.h5' else npy_path
            same = np.array_equal(read_volume(resumed_path), read_volume(reference))
            check(same, f'the resumed {extension} run differs from the run without a stop')
            print(f'resume, {extension}: the resumed run {"equals" if same else "DIFFERS FROM"} the run without a stop')

            killed_path = os.path.join(folder, 'killed' + extension)
            recorded = run_killed(stream_options + ['--output', killed_path, '--num-slabs', str(RESUME_SLABS)],
                                  RESUME_STOP)
            check(recorded == RESUME_STOP, f'the killed {extension} run recorded {recorded} slabs, not {RESUME_STOP}')
            run_stream(stream_options + ['--output', killed_path, '--resume'])
            same = np.array_equal(read_volume(killed_path), read_volume(reference))
            check(same, f'the killed and resumed {extension} run differs from the run without a stop')
            print(f'kill, {extension}: the killed and resumed run {"equals" if same else "DIFFERS FROM"} the run '
                  'without a stop')

            # The file at killed_path now holds a finished run.  The new run takes its name.
            recorded = run_killed(stream_options + ['--output', killed_path, '--num-slabs', str(RESUME_SLABS)], 0)
            check(recorded == 0, f'the new {extension} run left a progress file that records {recorded} slabs')
            run_stream(stream_options + ['--output', killed_path, '--resume'])
            same = np.array_equal(read_volume(killed_path), read_volume(reference))
            check(same, f'the new {extension} run, killed and resumed, differs from the run without a stop')
            print(f'restart, {extension}: the new run, killed in its first slab and resumed, '
                  f'{"equals" if same else "DIFFERS FROM"} the run without a stop')

        sino_h5_path = os.path.join(folder, 'sino.h5')
        with h5py.File(sino_h5_path, 'w') as f:
            f.create_dataset('sino', data=sino)
        from_h5_path = os.path.join(folder, 'from_h5.h5')
        run_stream(options + ['--sino', sino_h5_path, '--output', from_h5_path, '--num-slabs', str(RESUME_SLABS)])
        same = np.array_equal(read_volume(from_h5_path), read_volume(h5_path))
        check(same, 'the HDF5 sinogram gives a different result')
        print(f'HDF5 in: the HDF5 sinogram {"gives the same result as" if same else "DIFFERS FROM"} the .npy sinogram')

        phantom_path = os.path.join(folder, 'phantom.npy')
        with contextlib.redirect_stdout(io.StringIO()):
            make_synthetic_cone_sinogram.make_sinogram(model, phantom_path, PHANTOM_SLAB_SLICES)
        num_rows, num_cols, num_slices = (int(n) for n in model.get_params('recon_shape'))
        phantom = phantom_band(num_rows, num_cols, num_slices, 0, num_slices, model.torch_device)
        whole_sino = np.asarray(model.forward_project(phantom)).astype(np.float64)
        difference = largest_difference(np.load(phantom_path), whole_sino)
        check(difference <= TOLERANCE, f'the sinogram of the phantom differs by {difference:.2e}')
        print(f'phantom: the sinogram made one slab at a time differs from the projection of the whole phantom '
              f'by {difference:.1e}')

    if failures:
        raise SystemExit(f'{len(failures)} check(s) failed')
    print('all checks passed')


if __name__ == '__main__':
    main()
